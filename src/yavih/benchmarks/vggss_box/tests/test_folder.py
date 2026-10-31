"""Folder tests for vggss_box.

Run only this file: ``python -m pytest "$(yavih bench path vggss_box)/tests" -q -p no:cacheprovider``.

Three groups:

- pure checks of the frozen artefacts and the docs (yaml, manifest, full-frame boxes, the shipped geometry
  sidecars, curation, contracts, the freeze rules on a synthetic fixture); numpy + pyyaml only, always run;
- checks through the core (``Benchmark`` schema 2, ``RawStream``, ``grid_pos(region=…)``, ``yavih.media.rules``):
  always run;
- checks that need local data (VGGSound media, ``vggss.json`` and ``vggsound.csv``); they skip otherwise.
"""

from __future__ import annotations

import gzip
import inspect
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_rows, mm, bench = F.spec, F.rows, F.mm, F.bench

REMOVED_PROTOCOL_KEYS = {"windows", "window_shift", "frame_rule", "frame_rule_params", "frame_grid", "frame_select", "pad_policy",
                    "short_clip_policy", "crop", "audio_norm_context"}
INFO_VOCAB = {
    "video": {"duration_s", "fps", "n_frames", "width", "height", "container", "bytes"},
    "audio": {"duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"},
}
# (clip_id, full-frame box, frame_w, frame_h)
PINNED_BOXES = [
    ("-0BIyqJj9ZU_000030", "0.343750|0.000000|0.678125|0.734375", 320, 240),
    ("-23CeprtibU_000030", "0.437500|0.312500|0.909375|0.928125", 1280, 720),
    ("-2Dm0VjW8oM_000001", "0.165625|0.196875|0.728125|0.993750", 1280, 720),
    ("-Q3MX7Tgx_E_000111", "0.153125|0.146875|0.990625|0.868750", 202, 360),
]
SHORT_VIDEO_EVAL = ("vhUM-UOKpSk_000053", "fPFpRsKbzfU_000000", "1F3_bvIT0ig_000000")  # 5.09 s, 5.53 s, 8.0 s videos


def _load_module(name: str, file: str):
    return F.load_module(HERE / file, name)


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


# ----------------------------------------------------------------------------- benchmark.yaml (schema 2)


def test_yaml_schema2_and_changelog(spec):
    assert spec["schema_version"] == 3 and spec["name"] == HERE.name == "vggss_box"
    assert spec["scorer"] == "box" and spec["aliases"] == []
    assert spec["protocol_version"] == 1
    assert [c["version"] for c in spec["changelog"]] == [1]
    entry = spec["changelog"][0]
    assert entry["invalidates"] is False and entry["supersedes"] == []
    assert "requires" not in spec  # lives in output_contract
    assert not REMOVED_PROTOCOL_KEYS & set(spec["protocol"])  # windows, crop, padding are bridge decisions
    assert all("bridge_decode_allowed" not in s for s in F.file_fields(spec).values())


def test_protocol_label_facts(spec):
    assert spec["protocol"] == {
        "clip_seconds": 10.0, "train_draw": None, "views": "none",
        "box_population": "single", "gt_geometry": "full_frame", "heatmap_grid": 14, "annotated_second": 5.0,
    }
    assert "target_region" not in spec["protocol"]  # the scored region is the whole frame
    assert spec["readout_params"] == {"grid": 14, "visual_stream": "video", "conditioning_stream": "audio", "shuffled_control": True, "shuffle_seed": 20260913}


def test_input_contract(spec):
    ic = spec["input_contract"]
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(ic["info"]) == set(F.file_fields(spec)) == {"video", "audio"}
    for name, keys in ic["info"].items():
        assert set(keys) <= INFO_VOCAB[F.file_fields(spec)[name]["kind"]], name
    assert ic["info"]["video"] == ["duration_s", "fps", "width", "height", "container", "bytes"]
    assert ic["info"]["audio"] == ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]
    assert ic["facts"] == ["annotated_second"] and set(ic["facts"]) <= set(spec["protocol"])
    assert ic["materialize"] is None and spec["data"]["derived"] == {}
    for s in F.file_fields(spec).values():
        assert s["layout"] == "dir" and s["root"] == "media_root" and s["required"] is True


def test_output_contract(spec):
    assert spec["output_contract"] == {
        "shape": "fixed",
        "coordinates": "media",
        "target_frame": "full_frame",
        "requires": {"pos": ["y", "x"], "streams": ["video", "audio"], "grid": {"h": 14, "w": 14, "min_cell_coverage": 0.25}, "t_bins": None,
                     "annotated_frame": {"fact": "annotated_second", "stream": "video", "tol_s": 0.5}},
    }
    # the annotated-frame requirement names the fact the bridge is told (input_contract.facts)
    assert spec["output_contract"]["requires"]["annotated_frame"]["fact"] in spec["input_contract"]["facts"]
    assert set(spec["output_contract"]["requires"]["streams"]) == {spec["readout_params"]["visual_stream"], spec["readout_params"]["conditioning_stream"]}


def test_external_empty_no_probe(spec):
    # the official VGG-SS test.py scores heatmaps (cIoU/AUC), not boxes: not applicable (setup.md "Dependencies")
    assert spec["external"] == [] and not (HERE / "eval.py").exists()


def test_metrics_block(spec):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert m["headline"] == "iou_mean"
    assert m["list"] == ["iou_mean", "acc@0.5", "acc@0.3", "center_in_gt", "n_eval", "n_excluded_degenerate"]
    assert m["floors"] == ["iou_floor"] and m["controls"] == ["iou_shuffled", "audio_blind"] and m["chance"] == {}
    assert m["ci"] == {"iou_mean": "bootstrap_clip", "acc@0.5": "binomial", "acc@0.3": "binomial", "center_in_gt": "binomial"}


# ----------------------------------------------------------------------------- manifest


def test_sha256_and_columns(spec, manifest_rows):
    F.assert_manifest_hashes(HERE, spec)
    assert len(manifest_rows) == 4614
    assert list(manifest_rows[0].keys()) == ["clip_id", "split", "box", "class_name", "rel_video", "rel_audio", "frame_w", "frame_h"]
    ids = [r["clip_id"] for r in manifest_rows]
    assert len(set(ids)) == len(ids) and ids == sorted(ids)  # sorted clip_id
    for r in manifest_rows:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,32}_\d{6}", r["clip_id"])
        assert r["rel_video"] == f"video/{r['clip_id']}.mp4" and r["rel_audio"] == f"audio/{r['clip_id']}.wav"
        assert int(r["frame_w"]) > 0 and int(r["frame_h"]) > 0


def test_splits_assigned_once(spec, manifest_rows):
    assert F.assert_splits_assigned_once(spec, manifest_rows) == {"train": 3700, "eval": 914}


def test_short_video_entries_are_eval(manifest_rows):
    """The single-box entries outside the split draw (videos too short for the duration gate, sources without
    another entry) are eval rows."""
    by = {r["clip_id"]: r for r in manifest_rows}
    for cid in SHORT_VIDEO_EVAL:
        assert by[cid]["split"] == "eval"
    sources = Counter(cid.rsplit("_", 1)[0] for cid in by)
    assert all(sources[c.rsplit("_", 1)[0]] == 1 for c in SHORT_VIDEO_EVAL)


def test_source_level_split_no_leak(manifest_rows):
    sides: dict[str, set[str]] = {}
    for r in manifest_rows:
        sides.setdefault(r["clip_id"].rsplit("_", 1)[0], set()).add(r["split"])
    assert all(len(v) == 1 for v in sides.values())  # zero straddling sources
    leaked = {"9wW-losmGws", "JixXL5s6fnE", "K_EsxukdNXM", "rajXrf_Dzfw"}
    assert not (leaked & set(sides))


def test_boxes_full_frame_single_regressable(manifest_rows):
    edge = 0
    for r in manifest_rows:
        assert ";" not in r["box"]
        parts = r["box"].split("|")
        assert len(parts) == 4 and all(re.fullmatch(r"\d\.\d{6}", p) for p in parts)
        x1, y1, x2, y2 = (float(p) for p in parts)
        assert 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0
        edge += min(x1, y1) == 0.0 or max(x2, y2) == 1.0
    assert edge == 566  # boxes touching a frame edge


def test_pinned_boxes_and_frame_sizes(manifest_rows):
    by = {r["clip_id"]: r for r in manifest_rows}
    for cid, full, w, h in PINNED_BOXES:
        assert by[cid]["box"] == full and (int(by[cid]["frame_w"]), int(by[cid]["frame_h"])) == (w, h)


def test_shipped_geometry_sidecars(manifest_rows, mm):
    """prepare/frame_sizes.json and prepare/video_geometry.json (sha256-pinned) cover every non-leaked entry; the
    manifest's frame size is frame_sizes.json's; the duration gate leaves 19 entries outside the draw, of which the
    18 single-box ones are eval rows; 221 rows have a video shorter than clip_seconds."""
    sizes, geometry = mm.frame_sizes(), mm.video_geometry()
    assert set(sizes) == set(geometry) and len(sizes) == 5154
    assert all((int(r["frame_w"]), int(r["frame_h"])) == sizes[r["clip_id"]] for r in manifest_rows)
    assert sum(not mm.offset_grid_ok(*g) for g in geometry.values()) == 19
    outside = [r for r in manifest_rows if not mm.offset_grid_ok(*geometry[r["clip_id"]])]
    assert len(outside) == 18 and {r["split"] for r in outside} == {"eval"} and set(SHORT_VIDEO_EVAL) <= {r["clip_id"] for r in outside}
    assert sum(n / fps < mm.CLIP_SECONDS for n, fps in (geometry[r["clip_id"]] for r in manifest_rows)) == 221


def test_classes_absent_class_name_column(spec, manifest_rows):
    assert spec["manifest"]["classes"] is None and not (HERE / "classes.csv").exists()
    assert len({r["class_name"] for r in manifest_rows}) == 221
    assert len({r["class_name"] for r in manifest_rows if r["split"] == "eval"}) == 220


def test_curation_sums(spec, manifest_rows, mm):
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["total_upstream"] == 5158
    for v in cur["per_split"].values():
        assert v["staged"] + v["missing"] == v["listed"]
    # no crop or duration filter: leakage, multi-box and degenerate boxes only
    assert cur["excluded"] == {"leakage": 4, "multi_box": 538, "degenerate_box": 2}
    assert cur["leakage_audit"]["excluded"] == cur["excluded"]["leakage"] == 4
    up = spec["manifest"]["upstream"]
    assert up["rows"] == 5158 and up["sha256"] == mm.VGGSS_JSON_SHA == up["upstream_files"]["vggss/vggss.json"]
    assert up["upstream_files"]["prepare/frame_sizes.json"] == mm.FRAME_SIZES_SHA
    assert up["upstream_files"]["prepare/video_geometry.json"] == mm.VIDEO_GEOMETRY_SHA
    # short_clips = rows whose audio (pinned list) or video is shorter than clip_seconds (10 s)
    short_audio = mm.short_audio_clips()
    assert len(short_audio) == 487 and short_audio <= {r["clip_id"] for r in manifest_rows}
    assert Counter(r["split"] for r in manifest_rows if r["clip_id"] in short_audio) == {"train": 384, "eval": 103}
    assert cur["short_clips"] == 565


def test_floor_value(manifest_rows, mm):
    f = mm.floor_stats(manifest_rows)
    assert f["iou_floor"] == 0.3432 and f["mean_train_box_cxcywh"] == [0.500316, 0.507904, 0.506354, 0.564836]
    assert (f["acc@0.5_floor"], f["acc@0.3_floor"], f["center_in_gt_floor"]) == (0.186, 0.6247, 0.7024)
    try:
        from yavih.metrics import box as B
    except ImportError:
        return
    if not hasattr(B, "mean_train_box"):
        return

    def cxcywh(split):
        x = np.asarray([[float(v) for v in r["box"].split("|")] for r in manifest_rows if r["split"] == split])
        return np.stack([(x[:, 0] + x[:, 2]) / 2, (x[:, 1] + x[:, 3]) / 2, x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]], axis=1)

    tr, ev = cxcywh("train"), cxcywh("eval")
    assert abs(B.iou_mean(np.broadcast_to(B.mean_train_box(tr), ev.shape), ev, None, {}) - 0.3432) < 1e-4


def test_crop_coverage_supports_threshold(spec, manifest_rows, mm):
    """The facts behind min_cell_coverage 0.25 (setup.md): Resize(256)+CenterCrop(224) ViT/16 → 96/196 on 16:9,
    120/196 on 4:3, median 0.49 ≥ 0.25 (eligible); a 7 × 7 grid in the crop → 0.21 (not eligible)."""
    thr = spec["output_contract"]["requires"]["grid"]["min_cell_coverage"]
    assert mm.cell_coverage(mm.crop_region_resize_centre_crop(1280, 720)) == 96 / 196
    assert mm.cell_coverage(mm.crop_region_resize_centre_crop(480, 360)) == 120 / 196
    assert mm.cell_coverage((0.0, 0.0, 1.0, 1.0)) == 1.0
    cov = mm.crop_coverage(manifest_rows)
    assert cov["median"] == 0.4898 >= thr and cov["aspect_rows"] == {"16:9": 2825, "4:3": 1242, "other": 547}
    seven = np.median([mm.cell_coverage(mm.crop_region_resize_centre_crop(int(r["frame_w"]), int(r["frame_h"])), tokens=7) for r in manifest_rows])
    assert seven < thr


# ----------------------------------------------------------------------------- docs


def test_readme_facts():
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    assert "```bibtex" in F.section((HERE / "README.md").read_text(encoding="utf-8"), 2, "Citation")


def test_setup_facts(spec):
    """Folder facts in setup.md (the setup.md structure and contract copies are ``yavih bench lint``'s)."""
    text = (HERE / "setup.md").read_text(encoding="utf-8")
    assert spec["manifest"]["sha256"] in F.section(text, 2, "Prepare & verify")


def test_folder_ships_no_reader_py():
    """The folder ships no reader.py: ``Benchmark.reader()`` falls back to the generic
    ``yavih.sample.iter_samples``, which hands over what benchmark.yaml declares; the reader smokes in this
    file go through ``bench.reader``."""
    assert not (HERE / "reader.py").exists()


# ----------------------------------------------------------------------------- make_manifest.py


def test_make_manifest_ports(mm):
    # torchvision Resize(256) + CenterCrop(224) arithmetic, behind the crop-coverage facts
    assert mm.crop_window(1280, 720) == (455, 256, 116, 16)
    assert mm.crop_window(480, 360) == (341, 256, 58, 16)
    assert mm.crop_window(406, 720) == (256, 453, 16, 114)
    assert mm.format_boxes([[-0.0, 0.5, 1.0, 0.25]]) == "0.000000|0.500000|1.000000|0.250000"
    assert mm.sanitise_box([-2.8823e16, 0.1, 0.5, 0.9]) == (0.0, 0.1, 0.5, 0.9)  # the official max(x, 0)
    # build_split is a pure function of the entry set and the seed, source-level
    entries = [{"clip_id": f"s{i // 2}_{i:06d}", "source": f"s{i // 2}", "label": "ab"[i % 2]} for i in range(40)]
    a, b = mm.build_split(entries), mm.build_split(list(reversed(entries)))
    assert a == b
    per_source: dict[str, set[str]] = {}
    for e in entries:
        per_source.setdefault(e["source"], set()).add(a[e["clip_id"]])
    assert all(len(v) == 1 for v in per_source.values())
    assert mm.offset_grid_ok(300, 30.0) and not mm.offset_grid_ok(240, 30.0)


def _entry(cid: str, label: str, *boxes: list[float]) -> dict:
    return {"clip_id": cid, "source": cid.rsplit("_", 1)[0], "label": label, "bbox": [list(b) for b in boxes]}


def test_make_manifest_derive_synthetic(mm):
    """The freeze rules on a synthetic fixture: leakage at source level, multi-box and degenerate entries
    excluded, boxes after max(x, 0) in full-frame fractions, an entry outside the draw (too short for the duration
    gate) goes to eval, a non-test entry is refused."""
    box = [0.1, 0.2, 0.5, 0.6]
    entries = [_entry(f"src{i:02d}_{i:06d}", "ab"[i % 2], box) for i in range(20)]
    entries += [
        _entry("leak_000000", "a", box),                                    # source in VGGSound train
        _entry("multi_000000", "a", box, box),                              # two boxes
        _entry("degen_000000", "b", [0.3, -0.1, 0.3, 0.4]),                 # zero width
        _entry("short_000000", "b", [-2.8823e16, 0.0, 1.0, 0.5]),           # outside the draw; corrupt coordinate
    ]
    splits = {(e["source"], int(e["clip_id"][-6:])): "test" for e in entries}
    splits[("leak", 30)] = "train"
    sizes = {e["clip_id"]: (640, 360) for e in entries}
    geometry = {e["clip_id"]: (300, 30.0) for e in entries}
    geometry["short_000000"] = (150, 30.0)  # 5 s: no real frame at the 8 × 1 s grid's last centre
    rows, stats = mm.derive(entries, splits, sizes, geometry)
    by = {r["clip_id"]: r for r in rows}
    assert stats["excluded"] == {"leakage": 1, "multi_box": 1, "degenerate_box": 1} and stats["leaked_sources"] == ["leak"]
    assert stats["draw"] == {"entries": 22, "outside_draw": 1, "rows_outside_draw": 1}
    assert len(rows) == 21 and [r["clip_id"] for r in rows] == sorted(by)
    assert by["short_000000"]["split"] == "eval" and by["short_000000"]["box"] == "0.000000|0.000000|1.000000|0.500000"
    assert by["src00_000000"]["box"] == "0.100000|0.200000|0.500000|0.600000" and by["src00_000000"]["frame_w"] == "640"
    assert {r["split"] for r in rows} == {"train", "eval"}
    assert rows == mm.derive(list(reversed(entries)), splits, sizes, geometry)[0]  # order-independent
    splits[("src00", 0)] = "val"
    with pytest.raises(SystemExit, match="not in the official VGGSound test split"):
        mm.derive(entries, splits, sizes, geometry)


def test_make_manifest_check_if_inputs_present(mm):
    """The public re-derivation (vggss.json + vggsound.csv + the shipped sidecars) reproduces the committed
    manifest byte for byte and every value benchmark.yaml records (skips without the staged inputs)."""
    from yavih import config

    try:
        staging, root = mm.default_staging_dir(), mm.default_root()
    except config.ConfigError as exc:
        pytest.skip(f"vggss_box roots not configured: {exc}")
    if not (staging / "vggss.json").is_file():
        pytest.skip(f"vggss.json not present under {staging}")
    if not (root / "metadata" / "vggsound.csv").is_file() or not (root / "video").is_dir():
        pytest.skip(f"VGGSound metadata / media not present under {root}")
    out = mm.freeze(staging, root)
    rep = out["report"]
    assert mm.yaml_drift(rep) == []
    assert rep["draw"] == {"entries": 5135, "outside_draw": 19, "rows_outside_draw": 18}
    assert rep["leaked_sources"] == ["9wW-losmGws", "JixXL5s6fnE", "K_EsxukdNXM", "rajXrf_Dzfw"]
    assert rep["short_clips_per_split"] == {"train": 453, "eval": 112}
    assert out["manifest_bytes"] == gzip.decompress((HERE / "manifest.csv.gz").read_bytes())
    assert mm.gzip_bytes(out["manifest_bytes"]) == (HERE / "manifest.csv.gz").read_bytes()


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_module():
    mod = _load_module("vggss_box_bridge_template", "bridge_template.py")
    cls = mod.VGGSSBoxBridge
    assert cls.streams == ("video", "audio") and cls.pos_mode == "per_sample"
    assert not hasattr(cls, "decode_plan")  # no prefetch plan in the Bridge API
    assert "PREPROCESSING SOURCE" in (mod.__doc__ or "")
    assert mod.SPATIAL["crop"] == {"kind": "center", "size": [224, 224]}


def test_bridge_template_positions_media_coordinates(mm):
    """Output contract on the template's positions: video tokens of a Resize(256)+CenterCrop(224) bridge on a
    1280×720 clip lie inside the crop region of the ORIGINAL frame and cover 96 / 196 cells; t is a fraction of the
    media duration; audio tokens carry y = x = -1 and t = -1 in padding beyond the file."""
    from yavih.bridge import grid_pos

    assert "region" in inspect.signature(grid_pos).parameters
    mod = _load_module("vggss_box_bridge_template_pos", "bridge_template.py")
    region = mm.crop_region_resize_centre_crop(1280, 720)
    try:
        from yavih.media import rules

        assert np.allclose(rules.crop_region(mod.SPATIAL, 1280, 720), region, atol=1e-9)
    except ImportError:
        pass
    pts = np.arange(1.0, 9.0)
    pos = mod.VGGSSBoxBridge.positions(pts, {"duration_s": 10.0, "width": 1280, "height": 720}, {"duration_s": 7.5}, region)
    n_vid = mod.N_FRAMES * mod.GRID_H * mod.GRID_W
    assert pos.shape == (n_vid + mod.AUDIO_T, 4) and pos.dtype == np.float32
    vid, aud = pos[:n_vid], pos[n_vid:]
    assert np.all(vid[:, 0] == 0) and np.allclose(sorted(set(vid[:, 1].tolist())), pts / 10.0)
    y0, x0, y1, x1 = region
    assert np.all((vid[:, 2] > y0) & (vid[:, 2] < y1) & (vid[:, 3] > x0) & (vid[:, 3] < x1))
    cells = {(min(int(14 * y), 13), min(int(14 * x), 13)) for y, x in vid[:, 2:4]}
    assert len(cells) == 96
    assert np.all(aud[:, 0] == 1) and np.all(aud[:, 2:] == -1)
    assert np.all((aud[:, 1] == -1) == ((np.arange(mod.AUDIO_T) + 0.5) * 10.0 / mod.AUDIO_T > 7.5))


def test_bridge_template_describe(bench):
    pytest.importorskip("av")
    from yavih.bridge import validate_describe

    mod = _load_module("vggss_box_bridge_template_describe", "bridge_template.py")
    b = mod.VGGSSBoxBridge(bench.view(), {})
    d = b.describe()
    assert d["views"] == 1 and "decode" not in d
    assert d["input"]["video"]["temporal"] == {"rule": "explicit", "times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "edge": "clamp"}
    validate_describe(d, b, bench)


# ----------------------------------------------------------------------------- reader (raw hand-off)


def test_benchmark_loads(bench, spec):
    assert bench.name == "vggss_box" and bench.spec.readout == "box"
    assert len(bench.protocol_hash) == 64
    assert bench.manifest_hash == "4739a39a14b9d222f58080e9aabe7c378d5ca12921e992faf97b44a310d64554"
    for split, n in (("train", 3700), ("eval", 914)):
        assert bench.n_rows(split) == n
    view = bench.view()
    assert view.facts == {"annotated_second": 5.0} and view.streams == {"video": "video", "audio": "audio"}
    assert view.output_contract == spec["output_contract"]


def _write_synthetic_media(root: Path, media: dict, *, n_frames: int = 30, fps: int = 10, size=(320, 180), wav_seconds: float = 2.5) -> None:
    av = pytest.importorskip("av")
    sf = pytest.importorskip("soundfile")
    (root / "video").mkdir(exist_ok=True)
    (root / "audio").mkdir(exist_ok=True)
    w, h = size
    with av.open(str(root / media["video"]), mode="w") as out:
        vs = out.add_stream("mpeg4", rate=fps)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        for k in range(n_frames):
            img = np.full((h, w, 3), (k * 37) % 200, dtype=np.uint8)
            for pkt in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in vs.encode():
            out.mux(pkt)
    rng = np.random.default_rng(0)
    sf.write(root / media["audio"], rng.uniform(-0.5, 0.5, int(wav_seconds * 16000)).astype(np.float32), 16000, subtype="PCM_16")


def test_reader_smoke_synthetic_raw(bench, tmp_path, spec):
    """Reader smoke on synthetic raw files: a 3.0 s 320×180 mp4 and a 2.5 s 16 kHz wav under a temporary
    media root. The reader hands over the stored bytes (nothing decoded, cut, cropped or padded), the guaranteed
    media facts, and meta == {"facts": {"annotated_second": 5.0}}."""
    from yavih import Benchmark

    media = bench.manifest("train").media[0]
    _write_synthetic_media(tmp_path, media)
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    samples = list(b.reader("train", rows=range(0, 1)))
    assert len(samples) == 1
    s = samples[0]
    assert s.row == 0 and set(s.streams) == {"video", "audio"}
    assert s.meta == {"facts": {"annotated_second": 5.0}}
    v, a = s.streams["video"], s.streams["audio"]
    assert v.bytes() == (tmp_path / media["video"]).read_bytes() and a.bytes() == (tmp_path / media["audio"]).read_bytes()
    with v.open() as fh:
        assert fh.read() == v.bytes()
    for name, st in (("video", v), ("audio", a)):
        assert st.kind == name and st.name == name
        for key in spec["input_contract"]["info"][name]:
            assert st.info[key] is not None, (name, key)
    assert (v.info["width"], v.info["height"]) == (320, 180) and abs(v.info["duration_s"] - 3.0) < 0.15
    assert (a.info["sample_rate"], a.info["n_samples"], a.info["channels"]) == (16000, 40000, 1)
    assert abs(a.info["duration_s"] - 2.5) < 1e-9  # shorter than clip_seconds and the annotated second: not padded


def test_reader_missing_media_is_missing_row(bench, tmp_path):
    """An absent file is ``MediaMissing``: raised by the reader or on the first byte read, never a
    silently empty stream."""
    import yavih
    import yavih.media
    from yavih import Benchmark, sample

    missing = next(getattr(m, "MediaMissing") for m in (yavih.media, sample, yavih) if hasattr(m, "MediaMissing"))  # noqa: B009
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    with pytest.raises(missing):
        for s in b.reader("train", rows=range(0, 1)):
            s.streams["video"].bytes()


def test_reader_smoke_if_media_present(bench):
    """First rows of each split on the site's VGGSound media (skips without media): stored bytes, facts."""
    root = F.site_root(HERE, "media_root")
    if root is None:
        pytest.skip("vggss_box media_root not configured or not present on this host")
    man = bench.manifest("train")
    if not (root / man.media[0]["video"]).is_file() or not (root / man.media[0]["audio"]).is_file():
        pytest.skip(f"VGGSound media not present under {root}")
    for split in ("train", "eval"):
        samples = list(bench.reader(split, rows=range(0, 2)))
        assert [s.row for s in samples] == [0, 1]
        s = samples[0]
        m = bench.manifest(split)
        assert s.meta == {"facts": {"annotated_second": 5.0}}
        assert s.streams["video"].bytes() == (root / m.media[0]["video"]).read_bytes()
        assert (s.streams["video"].info["width"], s.streams["video"].info["height"]) == (m.extra[0]["frame_w"], m.extra[0]["frame_h"])
        assert s.streams["audio"].info["sample_rate"] == 16000 and s.streams["audio"].info["channels"] == 1
