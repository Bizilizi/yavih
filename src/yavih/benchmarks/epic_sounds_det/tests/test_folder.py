"""Folder tests of epic_sounds_det (run: ``python -m pytest "$(yavih bench path epic_sounds_det)/tests" -q -p no:cacheprovider``)."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows = F.spec, F.rows

READOUT = "epic_sounds_det"  # the folder's own scorer (eval.py + detection.py)
HEADLINE = "map_avg"
STREAMS = {"audio": "audio"}
CLIPS = {"train": 495, "validation": 137}
SEGMENTS = {"train": 60055, "validation": 8035}
READINGS = ["a"]
COLUMNS = ["clip_id", "split", "labels", "rel_audio", "events"]
INFO_KEYS = ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]
OUTPUT_CONTRACT = {"shape": "fixed", "coordinates": "media", "target_frame": None,
                   "requires": {"pos": ["t"], "streams": ["audio"], "grid": None, "t_bins": None}}
EXTERNAL: list[str] = []
HAS_PROBE = True
#: the C10 baseline file actionformer_baseline_validation.json scored by probe.official_map (README "## In yavih")
BASELINE_C10 = [16.736838654256243, 15.480657336554968, 14.010561695760487, 12.263180170795609, 10.354604142156187]


def _events(cell: str) -> list[tuple[int, int, int]]:
    return [tuple(int(x) for x in e.split(":")) for e in cell.split(";")]  # type: ignore[misc]


# ----------------------------------------------------------------------------- yaml + files


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]


def test_manifest_columns_and_events(rows: list[dict[str, str]]):
    assert list(rows[0]) == COLUMNS
    seg = {s: 0 for s in CLIPS}
    classes = [r["name"] for r in F.read_classes(HERE)]
    assert len(classes) == 44 and classes[13] == "background"
    for r in rows:
        assert r["rel_audio"] == f"audio/{r['clip_id']}.m4a" and re.match(r"^P\d\d_\d{2,3}$", r["clip_id"])
        ev = _events(r["events"])
        seg[r["split"]] += len(ev)
        assert all(0 <= c < 44 and 0 <= s < e for c, s, e in ev)
        assert r["labels"].split(";") == [classes[c] for c in sorted({c for c, _, _ in ev})]
    assert seg == SEGMENTS


def test_curation_sums(spec: dict):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"] == 699
    assert {k: v["listed"] for k, v in cur["per_split"].items()} == CLIPS


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE


def test_input_contract(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items()} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)
    assert ic["info"]["audio"] == INFO_KEYS and ic["facts"] == [] and ic["materialize"] is None


def test_output_contract_and_external(spec: dict):
    assert spec["output_contract"] == OUTPUT_CONTRACT
    assert [e["name"] for e in spec["external"] or []] == EXTERNAL
    assert (HERE / "eval.py").is_file() is HAS_PROBE and (HERE / "detection.py").is_file()


# ----------------------------------------------------------------------------- README.md + prepare/


def test_readme_facts(spec: dict):
    text = (HERE / "README.md").read_text(encoding="utf-8")
    sec = F.section(text, 2, "In yavih")
    assert spec["metrics"]["headline"] in sec and "make_manifest.py" in text and "detection.py" in sec
    assert "16.48 / 15.15 / 13.70 / 11.99 / 10.12" in sec and "16.74 / 15.48 / 14.01 / 12.26 / 10.35" in sec
    assert "@article{EPICSOUNDS2025" in text


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_annotations.sh", "02_audio.py", "03_check_manifest.sh"]
    assert all(re.match(r"^\d+_[A-Za-z0-9_.-]+\.(?:sh|py)$", n) for n in names), names


def test_prepare_01_pins_match_make_manifest():
    mm = F.load_module(HERE / "make_manifest.py")
    sh = (HERE / "prepare" / "01_annotations.sh").read_text()
    assert mm.UPSTREAM_COMMIT in sh
    for name, (sha, _, _) in mm.UPSTREAM_FILES.items():
        assert re.search(rf"{re.escape(name)}\"\s+\d+\s+{sha}", sh), name
    p02 = F.load_module(HERE / "prepare" / "02_audio.py")
    assert p02.MD5_SHA256 in sh


# ----------------------------------------------------------------------------- eval.py: the C10 scorer transcription


def _probe():
    return F.load_module(HERE / "eval.py")


def test_official_map_hand_cases():
    p = _probe()
    thr = [0.1, 0.2, 0.3, 0.4, 0.5]
    gt = [("v1", 0, 0.0, 1.0), ("v1", 0, 2.0, 3.0), ("v1", 1, 0.0, 4.0), ("v2", 0, 5.0, 6.0), ("v1", 13, 0.0, 9.0)]
    perfect = [(v, c, s, e, 1.0) for v, c, s, e in gt]
    m, avg = p.official_map(gt, perfect, thr)
    assert np.allclose(m, 1.0) and avg == pytest.approx(1.0)
    # background (13) is not scored: a missing background detection changes nothing
    m2, _ = p.official_map(gt, perfect[:-1], thr)
    assert np.allclose(m2, 1.0)
    # one false positive ranked first for class 0: AP0 = VOC-interpolated (prec 1/2, 2/3, 3/4 at recall 1/3, 2/3, 1)
    fp = perfect + [("v2", 0, 10.0, 11.0, 2.0)]
    m3, _ = p.official_map(gt, fp, thr)
    assert m3 == pytest.approx(np.full(5, (0.75 + 1.0) / 2))
    # a box half-overlapping (tIoU 1/3) counts at 0.1-0.3 only
    shifted = [("v1", 1, 2.0, 6.0, 1.0)]
    m4, _ = p.official_map([g for g in gt if g[1] == 1], shifted, thr)
    assert m4 == pytest.approx([1, 1, 1, 0, 0])
    # predictions of classes absent from the ground truth are dropped, not counted as false positives
    m5, _ = p.official_map(gt, perfect + [("v1", 7, 0.0, 1.0, 9.0)], thr)
    assert np.allclose(m5, 1.0)


def test_metric_interface(spec: dict):
    p = _probe()
    proto = spec["protocol"]
    targets = [np.array([[0, 0.0, 1.0], [13, 2.0, 3.0]]), np.array([[1, 5.0, 7.0]])]
    dets = [np.array([[0, 0.0, 1.0, 0.9]]), np.array([[1, 5.0, 6.0, 0.8]])]  # decoded (class, start, end, score); tIoU 0.5 on the second
    outputs = np.zeros((2, 8, 46), dtype=np.float32)  # the raw outputs those detections decode from (seeded in the decode cache)
    meta = {"eval_clip_ids": ["P01_11", "P01_12"], "_det_cache": {id(outputs): (outputs, dets)}}
    assert p.METRICS["map_avg"].fn(outputs, targets, proto, meta) == pytest.approx(100.0)
    assert p.METRICS["map@0.5"].fn(outputs, targets, proto, meta) == pytest.approx(100.0)
    assert p.METRICS["map_avg_paper"].fn(outputs, targets, proto, meta) == pytest.approx(100 * 2 / 3)  # background AP 0
    assert set(spec["metrics"]["list"]) - {"n_eval"} | set(spec["metrics"]["floors"]) == set(p.METRICS)


BASELINE = F.site_root(HERE, "media_root", needs=("annotations/actionformer_baseline_validation.json",
                                                   "annotations/EPIC_Sounds_validation.csv"))


@pytest.mark.skipif(BASELINE is None, reason="C10 baseline json not staged next to the annotations")
def test_official_map_reproduces_c10_baseline():
    """The transcription on the C10 baseline file gives the published Table V row with background kept."""
    p = _probe()

    def sec(ts: str) -> float:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    with open(BASELINE / "annotations" / "EPIC_Sounds_validation.csv", newline="") as fh:
        gt = [(r["video_id"], int(r["class_id"]), sec(r["start_timestamp"]), sec(r["stop_timestamp"])) for r in csv.DictReader(fh)]
    sub = json.loads((BASELINE / "annotations" / "actionformer_baseline_validation.json").read_text())
    pr = [(k, int(v["interaction"]), float(v["segment"][0]), float(v["segment"][1]), float(v["score"]))
          for k, vs in sub["results"].items() for v in vs]
    m, _ = p.official_map(gt, pr, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(m * 100, BASELINE_C10, atol=1e-9)
    mb, avg_b = p.official_map(gt, pr, [0.1, 0.2, 0.3, 0.4, 0.5], excluded=())
    assert [round(x * 100, 1) for x in mb] == [16.5, 15.2, 13.7, 12.0, 10.1] and round(avg_b * 100, 1) == 13.5


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values()
              if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    d = cls(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] in READINGS and set(d["input"]) <= set(STREAMS)
    assert mod.N_TOKENS * mod.HOP_S >= 3708.04  # the longest recording fits


# ----------------------------------------------------------------------------- make_manifest.py


def _write_mini_upstream(d: Path) -> None:
    hdr = "annotation_id,participant_id,video_id,start_timestamp,stop_timestamp,start_sample,stop_sample,description,class,class_id\n"
    (d / "EPIC_Sounds_train.csv").write_text(hdr + "P01_01_0,P01,P01_01,00:00:02.466,00:00:05.315,59184,127560,clang,plastic-only collision,6\n"
                                             "P01_01_1,P01,P01_01,00:01:08.981,00:01:09.871,0,0,x,open / close,3\n")
    (d / "EPIC_Sounds_validation.csv").write_text(hdr + "P01_11_0,P01,P01_11,00:00:00.100,00:00:00.300,0,0,y,open / close,3\n")


def test_make_manifest_derive_format(tmp_path: Path):
    """The row format on a hand-made upstream: one row per recording, ms-exact events in file order."""
    mm = F.load_module(HERE / "make_manifest.py")
    _write_mini_upstream(tmp_path)
    mm.UPSTREAM_FILES = {"EPIC_Sounds_train.csv": ("", 2, 1), "EPIC_Sounds_validation.csv": ("", 1, 1)}
    with pytest.raises(SystemExit, match="class ids"):  # 44 classes required
        mm.derive(tmp_path)
    assert mm.ts_ms("01:02:03.456") == 3723456
    with pytest.raises(SystemExit):
        mm.ts_ms("00:00:01.5")


def test_make_manifest_check_upstream_pins(tmp_path: Path):
    """A file with another sha256 is refused, naming both digests."""
    mm = F.load_module(HERE / "make_manifest.py")
    for name in mm.UPSTREAM_FILES:
        (tmp_path / name).write_text("x\n")
    with pytest.raises(SystemExit, match="expected"):
        mm.main(["--no-media", "--metadata", str(tmp_path), "--check"])


ANN = F.site_root(HERE, "media_root", needs=("annotations/EPIC_Sounds_train.csv", "annotations/EPIC_Sounds_validation.csv",
                                             "annotations/EPIC_Sounds_recognition_test_timestamps.csv",
                                             "annotations/EPIC_Sounds_detection_test_videos.csv"))


@pytest.mark.skipif(ANN is None, reason="EPIC-SOUNDS annotations not staged (yavih bench prepare epic_sounds_det 01)")
def test_make_manifest_check(capsys: pytest.CaptureFixture):
    """``--check`` re-derives manifest.csv.gz and classes.csv byte-identically from the public files and prints the
    committed ``manifest.upstream`` block."""
    mm = F.load_module(HERE / "make_manifest.py")
    assert mm.main(["--no-media", "--metadata", str(ANN / "annotations"), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    y = F.load_yaml(HERE)
    assert report["manifest"]["sha256"] == y["manifest"]["sha256"]
    assert report["upstream"] == {k: v for k, v in y["manifest"]["upstream"].items() if k != "note"}
    assert report["derived"]["segments"] == SEGMENTS


# ----------------------------------------------------------------------------- reader smoke on synthetic m4a


def _write_m4a(path: Path, seconds: float, seed: int, sr: int = 48000) -> None:
    av = pytest.importorskip("av")
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = (0.1 * rng.standard_normal((2, n))).astype(np.float32)
    with av.open(str(path), "w", format="ipod") as c:
        st = c.add_stream("aac", rate=sr, layout="stereo")
        for i in range(0, n, 1024):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(x[:, i:i + 1024]), format="fltp", layout="stereo")
            fr.sample_rate = sr
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode(None):
            c.mux(pkt)


def test_reader_smoke_synthetic_raw(tmp_path: Path):
    """The reader hands over each stored m4a byte for byte with its header facts and empty facts; the template
    consumes it and reports ``used`` inside the media with a fixed N."""
    from yavih import Benchmark

    m = Benchmark.from_dir(HERE).manifest("validation")
    media = tmp_path / "media"
    (media / "audio").mkdir(parents=True)
    secs = [3.0, 12.5, 7.0, 2.0, 4.0, 5.0, 6.0, 2.5]
    for i, s in enumerate(secs):
        _write_m4a(media / "audio" / f"{m.clip_id[i]}.m4a", s, i)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    samples = list(b.reader("validation", rows=range(0, 8), batch_size=1, canaries=1))
    assert len(samples) == 9
    for s, sec in zip(samples[:8], secs, strict=True):
        assert set(s.streams) == {"audio"} and s.meta == {"facts": {}} and s.split == "validation"
        st = s.streams["audio"]
        f = media / "audio" / f"{s.clip_id}.m4a"
        assert st.bytes() == f.read_bytes()
        assert st.info["sample_rate"] == 48000 and st.info["channels"] == 2 and st.info["container"] == "m4a"
        assert abs(st.info["duration_s"] - sec) < 0.1 and st.info["bytes"] == f.stat().st_size
    c = samples[8]
    assert c._canary_of == c.row and c.streams["audio"].bytes() == samples[c.row].streams["audio"].bytes()
    mod = F.load_module(HERE / "bridge_template.py")
    br = mod.EpicSoundsDetBridge(b.view(), {"layer": -1})
    feats, pos, used = F.encode_prepared(br, samples[:3])
    assert feats.shape == (3, mod.N_TOKENS, mod.D_PLACEHOLDER) and pos.shape == (3, mod.N_TOKENS, 4)
    for i, s in enumerate(samples[:3]):
        n = s.streams["audio"].info["n_samples"]
        assert used[i]["audio"]["samples"] == [[0, n]]
        timed = pos[i, :, 1] >= 0
        assert timed.sum() == pytest.approx(s.streams["audio"].info["duration_s"] / mod.HOP_S, abs=1)
        assert (pos[i, timed, 1] <= 1).all() and np.all(feats[i, ~timed] == 0)


def test_manifest_digest_is_decompressed_sha(spec: dict):
    import gzip

    data = gzip.decompress((HERE / "manifest.csv.gz").read_bytes())
    assert hashlib.sha256(data).hexdigest() == spec["manifest"]["sha256"]
