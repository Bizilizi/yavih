"""Folder tests for k400.

Run only this file: ``python -m pytest "$(yavih bench path k400)/tests" -q -p no:cacheprovider``.

Groups:
- yaml / manifest / docs checks: pure yaml + csv, run everywhere;
- checks through ``yavih.Benchmark``, the reader, ``yavih.media.rules`` and ``grid_pos(region=)``: always run;
- ``make_manifest.py`` on a synthetic release (a small staged tree, the same listing as zips, annotation CSVs);
  needs torch for the draw (freeze-time only);
- the reader smoke builds a tiny synthetic copy of this folder (four generated mp4s, incl. a 0.5-s clip and a
  portrait clip) and needs PyAV to write them;
- the public re-freeze from the staged release runs only where ``k400.media_root`` is configured and staged.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pytest
import yaml

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
spec, manifest_rows = F.spec, F.rows

VIDEO_INFO_VOCAB = {"duration_s", "fps", "n_frames", "width", "height", "container", "bytes", "rotation"}  # RawStream.info keys

# The staged release from the site config, resolved at collection (before ``_isolated_site`` hides it); None when
# unconfigured or not staged here.
MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("train", "val"))


def _load_module(name: str, path: Path):
    return F.load_module(path, name)


@pytest.fixture(autouse=True)
def _isolated_site(monkeypatch, tmp_path_factory):
    """No site config and a fresh YAVIH_HOME (stricter than ``F.isolated_yavih_home``)."""
    monkeypatch.setenv("YAVIH_CONFIG", "/nonexistent/yavih-config.toml")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path_factory.mktemp("yavih_home")))


@pytest.fixture(scope="module")
def readme():
    return (HERE / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def setup_md():
    return (HERE / "setup.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- benchmark.yaml (schema 2)


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_yaml_schema2_and_versions(spec):
    assert spec["schema_version"] == 3
    assert spec["name"] == HERE.name == "k400" and spec["aliases"] == []
    assert spec["scorer"] == "clf" and spec["manifest"]["label"]["kind"] == "single"
    assert spec["source"]["kind"] == "torrent"
    log = spec["changelog"]
    assert [e["version"] for e in log] == [1] and spec["protocol_version"] == log[-1]["version"] == 1
    assert log[0]["invalidates"] is False and log[0]["supersedes"] == []
    h = spec["header"]
    assert h["name"] == "Kinetics-400" and h["title"] == "The Kinetics Human Action Video Dataset"
    assert len(h["abstract"].split()) <= 60


def test_stream_layouts(spec):
    for s in F.file_fields(spec).values():
        assert s["layout"] in {"dir", "tar_index"}


def test_streams_and_input_contract(spec):
    assert F.file_fields(spec) == {
        "video": {"kind": "video", "media_column": "rel_media", "root": "media_root", "layout": "dir", "layout_params": {}, "required": True}
    }
    ic = spec["input_contract"]
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(ic["info"]) == set(F.file_fields(spec)) == {"video"}
    assert "exactly as stored" in ic["handed_over"]["video"]
    assert ic["info"]["video"] == ["duration_s", "fps", "width", "height", "container", "bytes"]
    assert set(ic["info"]["video"]) <= VIDEO_INFO_VOCAB
    assert ic["facts"] == [] and ic["materialize"] is None
    assert spec["data"]["derived"] == {} and spec["targets"] is None


def test_output_contract(spec):
    assert spec["output_contract"] == {
        "shape": "fixed",
        "coordinates": "media",
        "target_frame": None,  # clf: no spatial target
        "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None},
    }


def test_external_empty_and_no_probe_py(spec):
    # the Kinetics authors ship no scorer package; top-1 is the registry metric `acc`
    assert spec["external"] == []
    assert not (HERE / "eval.py").exists()


def test_protocol_holds_label_facts_only(spec):
    p = spec["protocol"]
    assert set(p) == {"clip_seconds", "train_draw", "views"}
    assert p["clip_seconds"] == 10.0 and p["views"] == "none"
    assert p["train_draw"] == {**p["train_draw"], "cap": 20000, "seed": 0, "balance": "class"}


# ----------------------------------------------------------------------------- frozen manifest and classes


def test_sha256_match(spec, manifest_rows):
    F.assert_manifest_hashes(HERE, spec)
    assert len(manifest_rows) == 33492
    assert list(manifest_rows[0].keys()) == ["clip_id", "split", "label", "rel_media"]
    assert spec["manifest"]["columns"] == {"rel_media": {"type": "str", "role": "media", "stream": "video"}}


def test_splits_assigned_once(spec, manifest_rows):
    assert F.assert_splits_assigned_once(spec, manifest_rows) == {"train": 20000, "val": 13492} == spec["budget"]["clips"]
    assert [r["split"] for r in manifest_rows] == ["train"] * 20000 + ["val"] * 13492  # contiguous populations


def test_classes_contiguous_frozen_locale_order():
    rows = list(csv.DictReader((HERE / "classes.csv").open(newline="", encoding="utf-8")))
    assert list(rows[0].keys()) == ["index", "name"]
    assert [int(r["index"]) for r in rows] == list(range(400))
    names = [r["name"] for r in rows]
    assert len(set(names)) == 400
    # en_US locale order of the staged val dirs (`ls val | sort`), NOT Python sorted(): 15 indices differ
    assert names != sorted(names)
    assert sum(a != b for a, b in zip(names, sorted(names), strict=True)) == 15
    assert names[35:38] == ["brushing_hair", "brushing_teeth", "brush_painting"]
    assert names[0] == "abseiling" and names[-1] == "zumba"


def test_labels_balanced_and_known(manifest_rows):
    names = {r["name"] for r in csv.DictReader((HERE / "classes.csv").open(newline="", encoding="utf-8"))}
    assert {r["label"] for r in manifest_rows} == names
    train = Counter(r["label"] for r in manifest_rows if r["split"] == "train")
    val = Counter(r["label"] for r in manifest_rows if r["split"] == "val")
    assert len(train) == 400 and set(train.values()) == {50}
    assert len(val) == 400 and min(val.values()) == 11 and max(val.values()) == 48


def test_rows_layout_and_leakage(manifest_rows):
    for r in manifest_rows:
        assert r["rel_media"] == f"{r['split']}/{r['label']}/{r['clip_id']}.mp4"
    sfx = re.compile(r"^(.*)_(\d{6})_(\d{6})$")
    train_y = {(sfx.match(r["clip_id"]).group(1) if sfx.match(r["clip_id"]) else r["clip_id"]) for r in manifest_rows if r["split"] == "train"}
    val_y = {r["clip_id"] for r in manifest_rows if r["split"] == "val"}
    assert not (train_y & val_y)


def test_curation_sums(spec, manifest_rows):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"]
    for split, v in cur["per_split"].items():
        assert v["listed"] == sum(r["split"] == split for r in manifest_rows)
        assert v["staged"] + v["missing"] == v["listed"]
    # on-disk census: 240 618 train mp4 + 19 404 val files (13 492 mp4 + 5 402 mkv + 510 webm)
    assert cur["total_upstream"] == 240618 + 19404
    assert cur["excluded"]["val_non_mp4_container"] == 5402 + 510
    assert cur["excluded"]["draw_remainder"] == 240618 - 20000
    assert cur["short_clips"] == 4555


def test_make_manifest_pins_agree_with_yaml(spec):
    mod = _load_module("k400_make_manifest", HERE / "make_manifest.py")
    up = spec["manifest"]["upstream"]
    assert mod.TREE_SHA256 == up["sha256"] and mod.TORRENT in up["source"]
    assert up["upstream_files"] == {f"annotations/{k}": v for k, v in mod.ANNOTATION_SHA.items()}
    assert sum(sum(v.values()) for v in mod.DISK_CENSUS.values()) == up["rows"] == up["curation"]["total_upstream"]
    assert sum(mod.SHORT_CLIPS.values()) == up["curation"]["short_clips"]
    assert (mod.CAP, mod.SEED) == (spec["protocol"]["train_draw"]["cap"], spec["protocol"]["train_draw"]["seed"])
    assert set(spec["source"]["sha256"]) == {z for z, _ in mod.ZIP_LAYOUT.values()}


def test_draw_semantics():
    mod = _load_module("k400_make_manifest_draw", HERE / "make_manifest.py")
    pytest.importorskip("torch")
    labels = ["b", "a", "b", "a", "c", "c", "b"]
    keep = mod.class_balanced_subsample(labels, 4, 0)
    assert keep == sorted(keep) and len(keep) == 4
    assert mod.class_balanced_subsample(labels, 0, 0) == list(range(7))
    assert mod.class_balanced_subsample(labels, 99, 0) == list(range(7))
    assert max(Counter(labels[i] for i in keep).values()) <= 2


def test_id_and_label_normalization():
    mod = _load_module("k400_make_manifest_norm", HERE / "make_manifest.py")
    assert mod.ytid("035LtPeUFTE_000085_000095") == ("035LtPeUFTE", 85, 95)
    assert mod.ytid("youtube-NLf-rU1wlTY-161-171") == ("NLf-rU1wlTY", 161, 171)
    assert mod.ytid("0wR5jVB-WPk") == ("0wR5jVB-WPk", None, None)
    assert mod.norm_ytid("-3B32lodo2M") == "3B32lodo2M" and mod.norm_ytid("a-_b") == "a_b"
    assert mod.class_dir("hurling (sport)") == "hurling_-sport-"
    assert mod.class_dir("massaging person's head") == "massaging_person-s_head"


def _synthetic_release(tmp_path: Path) -> tuple[Path, Path, list[str], Path]:
    """A 3-class release: staged tree, the same listing as the two torrent zips, and annotation CSVs."""
    names = ["b_class", "a_class", "c_-x-"]  # classes.csv order is not a sort
    files = {
        "train": {"b_class": ["k1_000001_000011.mp4", "-k2_000002_000012.mp4"], "a_class": ["k3_000003_000013.mp4"],
                  "c_-x-": ["youtube-k4-4-14.mp4", "k5_000005_000015.mp4", "k6_000006_000016.mp4"]},
        "val": {"b_class": ["v1.mp4", "v2.mkv"], "a_class": ["v3.mp4", "v4.webm"], "c_-x-": ["v5.mp4"]},
    }
    tree = tmp_path / "media"
    zips = tmp_path / "zips"
    zips.mkdir()
    with zipfile.ZipFile(zips / "kinetics400.zip", "w") as zt, zipfile.ZipFile(zips / "kinetics400_validation.zip", "w") as zv:
        for split, d in files.items():
            z, prefix = (zt, "kinetics400/train_256/") if split == "train" else (zv, "val_256/")
            for c, fs in d.items():
                (tree / split / c).mkdir(parents=True)
                z.writestr(prefix + c + "/", b"")
                for f in fs:
                    (tree / split / c / f).write_bytes(b"x")
                    z.writestr(prefix + c + "/" + f, b"x")
        zt.writestr("kinetics400/test_256/b_class/t1.mp4", b"x")  # never read
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / "train.csv").write_text("label,youtube_id,time_start,time_end,split,is_cc\n"
                                   "b class,k1,1,11,train,0\nb class,-k2,2,12,train,0\na class,k3,3,13,train,0\n"
                                   "c (x),k4,4,14,train,0\nc (x),k5,5,15,train,0\nc (x),k6,6,16,train,0\n", encoding="utf-8")
    (ann / "val.csv").write_text("label,youtube_id,time_start,time_end,split,is_cc\n"
                                 "b class,v1,0,10,val,0\na class,v3,0,10,val,0\nc (x),v5,0,10,val,0\n", encoding="utf-8")
    return tree, zips, names, ann


def test_make_manifest_on_synthetic_release(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    mod = _load_module("k400_make_manifest_synth", HERE / "make_manifest.py")
    tree, zips, names, ann = _synthetic_release(tmp_path)
    listing = mod.list_tree(tree)
    assert listing == mod.list_zips(zips)
    assert mod.census(listing) == {"train": {".mp4": 6}, "val": {".mkv": 1, ".mp4": 3, ".webm": 1}}
    with pytest.raises(SystemExit, match="different media set"):
        mod.freeze(listing, names, 4, 0)  # the real pins refuse a different listing
    monkeypatch.setattr(mod, "ANNOTATION_SHA", {k: hashlib.sha256((ann / k).read_bytes()).hexdigest() for k in ("train.csv", "val.csv")})
    res = mod.freeze(listing, names, 4, 0, pins=False, annotations=ann)
    rows = list(csv.DictReader(io.StringIO(res["manifest_bytes"].decode("utf-8"))))
    assert list(rows[0]) == mod.MANIFEST_COLUMNS
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    assert len(train) == 4 and [r["clip_id"] for r in val] == ["v1", "v3", "v5"]  # classes.csv order, mp4 only
    order = {c: i for i, c in enumerate(names)}
    assert [order[r["label"]] for r in train] == sorted(order[r["label"]] for r in train)
    assert all(r["rel_media"] == f"{r['split']}/{r['label']}/{r['clip_id']}.mp4" for r in rows)
    rep = res["report"]
    assert rep["annotation_check"]["unmatched"] == rep["annotation_check"]["label_mismatch"] == 0
    cur = rep["upstream"]["curation"]
    assert cur["total_upstream"] == 11 and cur["excluded"] == {"draw_remainder": 2, "val_non_mp4_container": 2, "train_non_mp4_container": 0}
    assert rep["upstream"]["sha256"] == mod.listing_sha256(listing)
    # a relabelled annotation is refused
    (ann / "val.csv").write_text((ann / "val.csv").read_text(encoding="utf-8").replace("a class,v3", "b class,v3"), encoding="utf-8")
    monkeypatch.setattr(mod, "ANNOTATION_SHA", {k: hashlib.sha256((ann / k).read_bytes()).hexdigest() for k in ("train.csv", "val.csv")})
    with pytest.raises(SystemExit, match="annotation check failed"):
        mod.freeze(listing, names, 4, 0, pins=False, annotations=ann)


def test_refreeze_from_staged_release_is_byte_identical():
    if MEDIA_ROOT is None:
        pytest.skip("k400.media_root not configured or not staged here")
    pytest.importorskip("torch")
    mod = _load_module("k400_make_manifest_refreeze", HERE / "make_manifest.py")
    res = mod.freeze(mod.list_tree(MEDIA_ROOT), mod.read_classes(HERE / "classes.csv"))
    assert mod.gzip_bytes(res["manifest_bytes"]) == (HERE / "manifest.csv.gz").read_bytes()


def test_metric_names_resolve(spec):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["metrics"]["headline"] == "acc" and spec["metrics"]["chance"] == {"acc": 0.0025}
    assert spec["metrics"]["ci"] == {"acc": "binomial"} and spec["metrics"]["controls"] == []


# ----------------------------------------------------------------------------- documentation


def test_readme_facts(readme):
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    assert "setup.md" in F.section(readme, 2, "In yavih") and "make_manifest.py" in F.section(readme, 3, "Re-freezing the manifest")
    assert "```bibtex" in F.section(readme, 2, "Citation") and "1705.06950" in readme.split("\n\n")[1]


def test_setup_obtain_lists_source_hashes(spec, setup_md):
    obtain = F.section(setup_md, 2, "Obtain the data")
    for sha in spec["source"]["sha256"].values():  # expected sha256 per downloaded file
        assert sha in obtain


def test_docs_quote_no_stale_hashes(readme, setup_md):
    """protocol_hash depends on core, the yaml, the manifest and eval.py; the docs must not quote one."""
    for text in (readme, setup_md):
        assert not re.findall(r"protocol_hash`?\s*`?\s*[0-9a-f]{12,64}", text)


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_declares_rules():
    mod = _load_module("k400_bridge_template", HERE / "bridge_template.py")
    cls = mod.K400Bridge
    assert cls.streams == ("video",) and cls.pos_mode == "per_sample"
    assert not hasattr(cls, "decode_plan")  # no prefetch plan: the bridge decodes
    assert mod.TEMPORAL == {"rule": "uniform", "n": mod.N_FRAMES, "span": "full"}
    assert mod.SPATIAL["crop"] == {"kind": "center", "size": [224, 224]}
    assert mod.NORM["source"]  # the norm constants carry their source
    src = (HERE / "bridge_template.py").read_text(encoding="utf-8")
    assert "PREPROCESSING SOURCE" in src and '"views": 1' in src and '"decode":' not in src


@pytest.mark.parametrize(("width", "height", "duration"), [(454, 256, 10.0), (340, 256, 12.0), (144, 256, 0.533)])
def test_template_positions_meet_output_contract(width, height, duration):
    """The template's pos: [T*H*W, 4], stream 0, t in [0, 1] (media fraction), (y, x) inside the crop region of the
    ORIGINAL frame — what validate's E_POS_RANGE / E_RULE_MISMATCH (spatial) check."""
    import numpy as np

    from yavih.media.rules import crop_region, rule_times

    mod = _load_module("k400_bridge_template_pos", HERE / "bridge_template.py")
    info = {"duration_s": duration, "fps": 30.0, "width": width, "height": height, "container": "mp4", "bytes": 1}
    times = rule_times(mod.TEMPORAL, info)
    assert times.shape == (mod.N_FRAMES,) and times.min() >= 0.0 and times.max() <= duration
    pos = mod.K400Bridge._pos(times, info)
    assert pos.shape == (mod.T_TOK * mod.H_TOK * mod.W_TOK, 4) and pos.dtype == np.float32
    assert (pos[:, 0] == 0).all() and (pos[:, 1] >= 0).all() and (pos[:, 1] <= 1).all()
    y0, x0, y1, x1 = crop_region(mod.SPATIAL, width, height)
    assert 0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1
    assert (pos[:, 2] > y0 - 1e-6).all() and (pos[:, 2] < y1 + 1e-6).all()
    assert (pos[:, 3] > x0 - 1e-6).all() and (pos[:, 3] < x1 + 1e-6).all()
    if width > height:  # K400 frames are 256 high: resize-short-256 is the identity, the crop keeps rows 16..240, 224 columns
        assert (y0, y1) == pytest.approx((16 / 256, 240 / 256)) and x1 - x0 == pytest.approx(224 / width, abs=1e-6)
    else:  # portrait (144 wide): upscaled to 256 wide, so the crop keeps 224/256 of the width and 224/455 of the height
        assert x1 - x0 == pytest.approx(224 / 256, abs=1e-6) and y1 - y0 == pytest.approx(224 / int(256 * height / width), abs=1e-6)


# ----------------------------------------------------------------------------- through yavih core


def test_benchmark_loads_and_view():
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE)
    assert b.name == "k400" and b.spec.readout == "clf"
    assert b.n_rows("train") == 20000 and b.n_rows("val") == 13492 and b.unknown_labels == 0
    assert len(b.protocol_hash) == 64 and len(b.manifest_hash) == 64
    assert b.manifest_hash == "015401933025818bcbf4da39f9f3bdb973c0c9891b8d8cbb3365bdaf7a39d5ab"
    v = b.view()
    assert v.streams == {"video": "video"} and v.facts == {} and not hasattr(v, "manifest")
    assert Benchmark.load("k400").protocol_hash == b.protocol_hash


def _write_mp4(path: Path, *, width: int, height: int, seconds: float, fps: int = 25) -> None:
    av = pytest.importorskip("av")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(path), "w", format="mp4") as c:
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = width, height, "yuv420p"
        for k in range(max(1, round(seconds * fps))):
            img = np.full((height, width, 3), (k * 7) % 256, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def test_reader_smoke_on_synthetic_raw_files(tmp_path, spec):
    """A 4-row copy of this folder (same yaml blocks, no reader.py) over generated mp4s: the generic reader hands over
    the stored bytes, the guaranteed info keys, facts only — no decoding, no window, no media_key."""
    folder = tmp_path / "k400"
    folder.mkdir()
    media = tmp_path / "media"
    geoms = {  # (split, label, clip_id) -> (width, height, seconds)
        ("train", "abseiling", "aaaaaaaaaaa_000001_000011"): (96, 64, 2.0),
        ("train", "zumba", "bbbbbbbbbbb_000002_000012"): (64, 96, 0.5),  # portrait, short
        ("val", "abseiling", "ccccccccccc"): (96, 64, 2.4),
        ("val", "zumba", "ddddddddddd"): (96, 64, 1.0),
    }
    lines = ["clip_id,split,label,rel_media"]
    for (split, label, cid), (w, h, sec) in geoms.items():
        rel = f"{split}/{label}/{cid}.mp4"
        _write_mp4(media / rel, width=w, height=h, seconds=sec)
        lines.append(f"{cid},{split},{label},{rel}")
    man = ("\n".join(lines) + "\n").encode("utf-8")
    (folder / "manifest.csv.gz").write_bytes(gzip.compress(man, mtime=0))
    shutil.copy(HERE / "classes.csv", folder / "classes.csv")
    y = dict(spec)
    y["manifest"] = {**spec["manifest"], "sha256": hashlib.sha256(man).hexdigest()}
    y["manifest"]["upstream"] = {**spec["manifest"]["upstream"], "curation": {
        "total_upstream": 4, "per_split": {"train": {"listed": 2, "staged": 2, "missing": 0}, "val": {"listed": 2, "staged": 2, "missing": 0}},
        "excluded": {}, "short_clips": 4, "leakage_audit": None}}
    y["budget"] = {**spec["budget"], "clips": {"train": 2, "val": 2}}
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")

    from yavih import Benchmark
    from yavih.sample import RawStream

    b = Benchmark.from_dir(folder, data={"media_root": str(media)})
    for split in ("train", "val"):
        samples = list(b.reader(split, rows=range(0, 2)))
        assert [s.row for s in samples] == [0, 1]
        for s in samples:
            assert set(s.streams) == {"video"} and s.meta == {"facts": {}}
            assert not hasattr(s, "media_key")
            v = s.streams["video"]
            assert isinstance(v, RawStream) and v.kind == "video" and v.name == "video"
            key = next(k for k in geoms if k[0] == split and k[2] == s.clip_id)
            f = media / f"{key[0]}/{key[1]}/{key[2]}.mp4"
            assert Path(v._real_path) == f and Path(v.path).resolve() == f.resolve() and v.bytes() == f.read_bytes()
            with v.open() as fh:
                assert fh.read() == f.read_bytes()
            for k in spec["input_contract"]["info"]["video"]:
                assert v.info[k] is not None, k
            w, h, sec = geoms[key]
            assert (v.info["width"], v.info["height"]) == (w, h)
            assert v.info["duration_s"] == pytest.approx(sec, abs=0.1) and v.info["bytes"] == f.stat().st_size
            assert v.info["container"] in {"mp4", "mov", "mov,mp4,m4a,3gp,3g2,mj2"}
