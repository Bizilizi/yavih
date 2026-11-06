"""Folder tests of perception_test_action_loc (run: ``python -m pytest "$(yavih bench path perception_test_action_loc)/tests" -q -p no:cacheprovider``)."""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows, classes, bench, mm = F.spec, F.rows, F.classes, F.bench, F.mm

TASK = "action_localisation"
READOUT = "perception_test_action_loc"  # the folder's own scorer (eval.py + detection.py)
HEADLINE = "tmap"
STREAMS = {"video": "video", "audio": "audio"}
CLIPS = {"train": 2009, "valid": 5359}
EVENTS = {"train": 13097, "valid": 35440}
NUM_CLASSES = 63
FIRST_CLASSES = ["Breaking something", "Browsing through something", "Buttoning up a button"]
READINGS = ["a", "av", "v"]
DEFAULT_READING = "v"
OUTPUT_CONTRACT = {"shape": "fixed", "coordinates": "media", "target_frame": None,
                   "requires": {"pos": ["t"], "streams": ["video"], "grid": None, "t_bins": None}}
FLOOR = 0.012  # README "## In yavih" (rounded)
COLUMNS = ["clip_id", "split", "label", "events", "duration_s", "rel_video", "rel_audio"]


# ----------------------------------------------------------------------------- yaml + files


def _detection():
    """The folder's own detection scorer module (``detection.py``, loaded by ``eval.py``) and its events format."""
    ev = F.load_module(HERE / "eval.py")
    return ev.detection, ev.EVENTS_FORMAT


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE
    assert spec["protocol"]["tiou_thresholds"] == [0.1, 0.2, 0.3, 0.4, 0.5] and spec["readout_params"] == {"T": 128}


def test_input_contract(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items()} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS) and ic["facts"] == [] and ic["materialize"] is None
    assert spec["data"]["roots"]["media_root"]["default"] == "${datasets.perception_test}"


def test_output_contract_and_external(spec: dict):
    assert spec["output_contract"] == OUTPUT_CONTRACT
    assert spec["external"] == [] and (HERE / "eval.py").is_file() and (HERE / "detection.py").is_file() and not (HERE / "reader.py").is_file()


# ----------------------------------------------------------------------------- manifest facts


def test_columns_and_charsets(rows: list[dict[str, str]]):
    assert list(rows[0]) == COLUMNS
    for r in rows:
        assert re.fullmatch(r"video_\d+", r["clip_id"]), r["clip_id"]
        assert r["rel_video"] == f"videos/{r['clip_id']}.mp4" and r["rel_audio"] == f"audios/{r['clip_id']}.wav"


def test_classes(classes: list[dict[str, str]]):
    names = [c["name"] for c in classes]
    assert len(names) == NUM_CLASSES and names[: len(FIRST_CLASSES)] == FIRST_CLASSES
    assert all(n == n.strip() for n in names)


def test_events_cells_agree_with_labels_and_durations(rows: list[dict[str, str]], classes: list[dict[str, str]]):
    det, fmt = _detection()

    names = [c["name"] for c in classes]
    n_events: Counter = Counter()
    for r in rows:
        ev = det.parse_events(r["events"], fmt)
        assert len(ev) >= 1, r["clip_id"]
        present = sorted({int(c) for c in ev[:, 0]})
        assert r["label"] == ";".join(names[c] for c in present), r["clip_id"]
        d = float(r["duration_s"])
        assert 6.0 < d < 37.0 and (ev[:, 2] <= d + 0.05).all() and (ev[:, 1] <= ev[:, 2]).all(), r["clip_id"]
        n_events[r["split"]] += len(ev)
    assert dict(n_events) == EVENTS


def test_curation_sums(spec: dict):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = {s: v["listed"] for s, v in cur["per_split"].items()}
    assert listed == CLIPS and sum(listed.values()) + sum(cur["excluded"].values()) == cur["total_upstream"] == sum(CLIPS.values())
    assert set(spec["manifest"]["upstream"]["upstream_files"]) == {mm_file for mm_file, *_ in F.load_module(HERE / "make_manifest.py").ANNOTATIONS.values()}


def test_floor_value(bench):
    """The floor metric on the full manifest (labels only, no features), the value README.md quotes."""
    det, fmt = _detection()
    metrics = F.load_module(HERE / "eval.py").METRICS

    class Ctx:
        def __init__(self, b):
            self.bench, self.protocol = b, dict(b.spec.protocol)

        def manifest(self, split):
            return self.bench.manifest(split)

        def clip_ids(self, split, rows):
            m = self.bench.manifest(split)
            return [m.clip_id[i] for i in rows]

    ctx, T = Ctx(bench), int(bench.spec.readout_params["T"])
    every = lambda s: np.arange(bench.n_rows(s))  # noqa: E731
    tr = det._inputs(ctx, ["train"], every, fmt, T=T, num_classes=NUM_CLASSES, with_media=False)
    ev = det._inputs(ctx, ["valid"], every, fmt, T=T, num_classes=NUM_CLASSES, with_media=False)
    floor = det.floor_outputs(tr["targets"], len(ev["rows"]))
    value = metrics["tmap"].fn(floor, ev["events"], dict(bench.spec.protocol), {"duration_s": ev["duration_s"]})
    assert round(value, 3) == FLOOR


# ----------------------------------------------------------------------------- prepare/


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_fetch.py", "02_verify.sh"]
    assert all(re.match(r"^\d+_[A-Za-z0-9_.-]+\.(?:sh|py)$", n) for n in names), names


def test_fetch_pins_cover_the_annotations_make_manifest_reads():
    fetch = F.load_module(HERE / "prepare" / "01_fetch.py")
    ann = F.load_module(HERE / "make_manifest.py").ANNOTATIONS
    assert fetch.ANNOTATIONS[TASK.split("_")[0]] == {z: sha for z, _, sha, _ in ann.values()}
    assert {k for k in fetch.MEDIA_ZIPS} == {(s, k) for s in CLIPS for k in ("videos", "audios")}


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    view = Benchmark.from_dir(HERE).view()
    d = cls(view, {}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] == DEFAULT_READING and set(d["input"]) <= set(STREAMS)
    for reading in READINGS:
        streams = cls(view, {"reading": reading}).describe()["streams"]
        assert set(streams) == {"a": {"audio"}, "v": {"video"}, "av": {"video", "audio"}}[reading]


# ----------------------------------------------------------------------------- make_manifest.py


def _standin(rows: list[dict[str, str]], classes: list[dict[str, str]]) -> dict[str, dict]:
    """The two upstream JSONs rebuilt from the committed manifest: label_id = class index, frame_rate = 1e6 and
    num_frames = duration in microseconds (so num_frames / frame_rate prints the committed duration)."""
    det, fmt = _detection()

    names = [c["name"] for c in classes]
    dbs: dict[str, dict] = {s: {} for s in CLIPS}
    for r in rows:
        ev = [(int(c), int(round(a * 1e6)), int(round(b * 1e6))) for c, a, b in det.parse_events(r["events"], fmt)]
        dbs[r["split"]][r["clip_id"]] = {
            "metadata": {"split": r["split"], "video_id": r["clip_id"], "frame_rate": 1e6, "num_frames": int(round(float(r["duration_s"]) * 1e6))},
            TASK: [{"label": names[c], "label_id": c, "timestamps": [a, b]} for c, a, b in ev],
        }
    return dbs


def test_make_manifest_check(mm, rows: list[dict[str, str]], classes: list[dict[str, str]], spec: dict):
    """The derivation re-creates manifest.csv.gz and classes.csv byte for byte from (a stand-in of) the upstream files,
    and a zip with another sha256 is refused."""
    from yavih.freeze import csv_bytes

    out_rows, names, facts = mm.derive(_standin(rows, classes))
    assert csv_bytes(mm.MANIFEST_COLUMNS, out_rows) == F.read_manifest_bytes(HERE, spec)
    assert csv_bytes(mm.CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(names)]) == (HERE / "classes.csv").read_bytes()
    assert facts["events"] == EVENTS


def test_make_manifest_refuses_unpinned_zip(mm, tmp_path):
    for zname, *_ in mm.ANNOTATIONS.values():
        (tmp_path / zname).write_bytes(b"not the upstream zip")
    with pytest.raises(SystemExit, match="sha256"):
        mm.main(["--no-media", "--metadata", str(tmp_path), "--check"])


ANN_ROOT = F.site_root(HERE, "media_root", needs=tuple(f"annotations/{z}" for z, *_ in
                                                     F.load_module(HERE / "make_manifest.py").ANNOTATIONS.values()))


@pytest.mark.skipif(ANN_ROOT is None, reason="Perception Test annotations not staged on this machine (yavih bench prepare ... 01)")
def test_make_manifest_check_on_the_public_files(mm, spec: dict, capsys: pytest.CaptureFixture):
    assert mm.main(["--no-media", "--metadata", str(ANN_ROOT / "annotations"), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    up = {k: v for k, v in spec["manifest"]["upstream"].items() if k != "note"}
    assert report["upstream"] == up and report["manifest"]["sha256"] == spec["manifest"]["sha256"]


# ----------------------------------------------------------------------------- reader


def _write_synthetic_media(root, media: dict, *, n_frames: int = 30, fps: int = 10, size=(320, 180), wav_seconds: float = 2.5) -> None:
    av = pytest.importorskip("av")
    sf = pytest.importorskip("soundfile")
    for sub in ("videos", "audios"):
        (root / sub).mkdir(exist_ok=True)
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
    sf.write(root / media["audio"], rng.uniform(-0.5, 0.5, (int(wav_seconds * 48000), 2)).astype(np.float32), 48000, subtype="PCM_32")


def test_reader_smoke_synthetic_raw(bench, tmp_path, spec):
    """Synthetic raw files under a temporary media root: the reader hands over the stored bytes of the mp4 and of the
    wav (48 kHz stereo 32-bit PCM, one of the upstream formats), the guaranteed media facts and no label facts."""
    from yavih import Benchmark

    media = bench.manifest("train").media[0]
    _write_synthetic_media(tmp_path, media)
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    (s,) = list(b.reader("train", rows=range(0, 1)))
    assert s.row == 0 and set(s.streams) == {"video", "audio"} and s.meta == {"facts": {}}
    v, a = s.streams["video"], s.streams["audio"]
    assert v.bytes() == (tmp_path / media["video"]).read_bytes() and a.bytes() == (tmp_path / media["audio"]).read_bytes()
    for name, st in (("video", v), ("audio", a)):
        assert st.kind == name and st.name == name
        for key in spec["input_contract"]["info"][name]:
            assert st.info[key] is not None, (name, key)
    assert (v.info["width"], v.info["height"]) == (320, 180) and abs(v.info["duration_s"] - 3.0) < 0.15
    assert (a.info["sample_rate"], a.info["n_samples"], a.info["channels"]) == (48000, 120000, 2)


def test_reader_smoke_if_media_present(bench):
    """The first train row on the site's media (skips without it): stored bytes and upstream-like facts."""
    root = F.site_root(HERE, "media_root")
    if root is None:
        pytest.skip("perception_test media_root not configured or not present on this host")
    m = bench.manifest("train")
    if not (root / m.media[0]["video"]).is_file() or not (root / m.media[0]["audio"]).is_file():
        pytest.skip(f"Perception Test media not present under {root}")
    (s,) = list(bench.reader("train", rows=range(0, 1)))
    assert s.streams["video"].bytes() == (root / m.media[0]["video"]).read_bytes()
    assert s.streams["audio"].info["sample_rate"] in (44100, 48000) and s.streams["audio"].info["channels"] in (1, 2)
    assert abs(s.streams["video"].info["duration_s"] - m.extra[0]["duration_s"]) < 0.5
