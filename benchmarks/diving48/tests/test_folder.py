"""Folder tests of diving48 (run: ``python -m pytest "$(yavih bench path diving48)/tests" -q -p no:cacheprovider``).

What nothing else checks (``yavih bench lint``, run once per folder by the repository, covers the layout, the yaml,
the hashes, the conformance fixture and the README.md / setup.md structure):

* the frozen facts: rows per split, columns and charsets, the 48-class vocabulary with class 30 absent from V2, the
  curation sums, the pins shared by ``make_manifest.py``, the prepare steps and ``benchmark.yaml``;
* ``make_manifest.py --no-media --metadata --check`` on the three upstream JSON files, rebuilt byte for byte from
  the committed manifest and classes (their sha256 must equal the pinned upstream digests), so it runs anywhere;
* the reader on synthetic mp4s written for a few manifest rows, and the bridge template on what it hands over;
* the checks against the staged files (the upstream JSONs, real-media reader smoke) skip when they are absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
spec, rows, classes = F.spec, F.rows, F.classes

READOUT = "clf"
HEADLINE = "acc"
STREAMS = {"video": "video"}
CLIPS = {"train": 15027, "test": 1970}
READINGS = ["v"]
EXTERNAL: list[str] = []
HAS_EVAL = False
NUM_CLASSES = 48
ABSENT_CLASS = (30, "Forward Dive NoTwis STR")
COLUMNS = ["clip_id", "rel_media", "label", "split", "end_frame"]
INFO_KEYS = ["duration_s", "fps", "n_frames", "width", "height", "container", "bytes"]
MANIFEST_SHA256 = "cab70696095ce4f7c12feebbb13678828610786cbf6f50690b8cc720979e3d1a"


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No site configuration: data roots come only from what a test passes."""
    monkeypatch.setenv("YAVIH_CONFIG", str(tmp_path / "_no_such_config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "yavih_home"))
    monkeypatch.delenv("YAVIH_DATA__DIVING48__MEDIA_ROOT", raising=False)
    return tmp_path


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)
    assert spec["manifest"]["sha256"] == MANIFEST_SHA256


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]
    assert spec["splits"]["train"] == ["train"] and spec["splits"]["eval"] == ["test"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE
    assert spec["metrics"]["chance"] == {"acc": round(1 / NUM_CLASSES, 6)}
    assert spec["protocol"]["train_draw"] is None  # VideoGLUE trains on all 15 027 V2 train clips


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)
    assert ic["info"]["video"] == INFO_KEYS and ic["facts"] == [] and ic["materialize"] is None
    assert [e["name"] for e in spec["external"] or []] == EXTERNAL
    assert (HERE / "eval.py").is_file() is HAS_EVAL
    assert spec["obtain"]["tier"] == "auto" and spec["source"]["kind"] == "public_url"


def test_manifest_columns_and_charsets(rows: list[dict[str, str]]):
    assert list(rows[0]) == COLUMNS
    for r in rows:
        assert re.fullmatch(r"[A-Za-z0-9_-]{11}_\d{5}", r["clip_id"]), r["clip_id"]
        assert r["rel_media"] == f"rgb/{r['clip_id']}.mp4"
        assert int(r["end_frame"]) >= 24
    assert len({r["clip_id"] for r in rows}) == len(rows)
    # file order: the V2 train file first, then the V2 test file
    assert [r["split"] for r in rows] == ["train"] * CLIPS["train"] + ["test"] * CLIPS["test"]


def test_classes_are_the_vocabulary(classes: list[dict[str, str]], rows: list[dict[str, str]]):
    assert [int(c["index"]) for c in classes] == list(range(NUM_CLASSES))
    names = [c["name"] for c in classes]
    assert all(len(n.split(" ")) == 4 for n in names) and len(set(names)) == NUM_CLASSES
    assert names[0] == "Back 15som 05Twis FREE" and names[ABSENT_CLASS[0]] == ABSENT_CLASS[1]
    for split in CLIPS:
        present = {r["label"] for r in rows if r["split"] == split}
        assert set(names) - present == {ABSENT_CLASS[1]}, split


def test_floor_and_majority(rows: list[dict[str, str]], classes: list[dict[str, str]]):
    """The numbers metrics.notes quotes: the train-majority class and its share of the test split."""
    from collections import Counter

    tr = Counter(r["label"] for r in rows if r["split"] == "train")
    te = Counter(r["label"] for r in rows if r["split"] == "test")
    order = [c["name"] for c in classes]
    top = max(tr.values())
    majority = min((n for n, k in tr.items() if k == top), key=order.index)
    assert (majority, top, te[majority]) == ("Forward 35som NoTwis PIKE", 1053, 69)
    assert te.most_common(1) == [("Reverse Dive NoTwis PIKE", 163)]


def test_curation_sums(spec: dict, rows: list[dict[str, str]]):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"] == len(rows) == spec["manifest"]["upstream"]["rows"]
    assert {s: v["listed"] for s, v in cur["per_split"].items()} == CLIPS


def test_pins_agree(spec: dict):
    mm = F.load_module(HERE / "make_manifest.py", "diving48_make_manifest_pins")
    up = spec["manifest"]["upstream"]
    assert up["upstream_files"] == mm.UPSTREAM_FILES and up["sha256"] == mm.UPSTREAM_FILES["Diving48_V2_train.json"]
    for name, sha in mm.UPSTREAM_FILES.items():
        assert spec["source"]["sha256"][name] == sha
    step1 = (HERE / "prepare" / "01_download.sh").read_text(encoding="utf-8")
    for name, sha in spec["source"]["sha256"].items():
        assert re.search(rf"^{re.escape(name)} \S+ \d+ {sha}$", step1, re.M), name
    step3 = (HERE / "prepare" / "03_make_manifest.sh").read_text(encoding="utf-8")
    assert spec["manifest"]["sha256"] in step3 and spec["manifest"]["classes_sha256"] in step3


# ----------------------------------------------------------------------------- prepare/


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_download.sh", "02_extract.sh", "03_make_manifest.sh"]
    assert all(re.match(r"^\d+_[A-Za-z0-9_.-]+\.(?:sh|py)$", n) for n in names), names


# ----------------------------------------------------------------------------- make_manifest.py


def _upstream_files(rows: list[dict[str, str]], classes: list[dict[str, str]]) -> dict[str, bytes]:
    """The three upstream JSON files, rebuilt byte for byte from the committed manifest and classes (the upstream
    formatting: ``json.dumps(indent=2)`` for the splits, ``indent=2`` with ``", "`` separators for the vocabulary)."""
    index = {c["name"]: int(c["index"]) for c in classes}
    out = {}
    for split, name in (("train", "Diving48_V2_train.json"), ("test", "Diving48_V2_test.json")):
        entries = [{"vid_name": r["clip_id"], "label": index[r["label"]], "start_frame": 0, "end_frame": int(r["end_frame"])}
                   for r in rows if r["split"] == split]
        out[name] = json.dumps(entries, indent=2).encode()
    out["Diving48_vocab.json"] = json.dumps([c["name"].split(" ") for c in classes], indent=2, separators=(", ", ": ")).encode()
    return out


def test_make_manifest_check(tmp_path: Path, spec: dict, rows: list[dict[str, str]], classes: list[dict[str, str]],
                             capsys: pytest.CaptureFixture):
    mm = F.load_module(HERE / "make_manifest.py", "diving48_make_manifest")
    meta = tmp_path / "meta"
    meta.mkdir()
    for name, data in _upstream_files(rows, classes).items():
        (meta / name).write_bytes(data)
        assert hashlib.sha256(data).hexdigest() == mm.UPSTREAM_FILES[name], name
    assert mm.main(["--no-media", "--metadata", str(meta), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    up = {k: v for k, v in spec["manifest"]["upstream"].items() if k in report["upstream"]}
    assert report["upstream"] == up
    assert report["manifest"] == {"sha256": spec["manifest"]["sha256"], "classes_sha256": spec["manifest"]["classes_sha256"], "rows": len(rows)}
    # a changed upstream file is refused (pinned sha256)
    (meta / "Diving48_V2_test.json").write_bytes(b"[]")
    with pytest.raises(SystemExit, match="sha256"):
        mm.main(["--no-media", "--metadata", str(meta), "--check"])


META_ROOT = F.site_root(HERE, "media_root", needs=("Diving48_V2_test.json",))


@pytest.mark.skipif(META_ROOT is None, reason="Diving48 annotation files not staged on this machine")
def test_staged_metadata_is_the_pinned_files(rows: list[dict[str, str]], classes: list[dict[str, str]]):
    for name, data in _upstream_files(rows, classes).items():
        assert (META_ROOT / name).read_bytes() == data, name


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values()
              if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    d = cls(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] in READINGS and set(d["input"]) <= set(STREAMS) and d["views"] == 1


def test_bridge_template_defaults_are_videoglues():
    """16 frames at stride 4 of 25 fps (6.25 Hz) from a centred 2.56-s window; short side 256, centre crop 224."""
    from yavih.media.rules import crop_region, rule_times

    mod = F.load_module(HERE / "bridge_template.py", "diving48_template_defaults")
    t = mod.Diving48Bridge.TEMPORAL
    assert t == {"rule": "fps", "fps": 6.25, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
    times = rule_times(t, {"duration_s": 6.0, "fps": 25.0, "n_frames": 150})
    assert times == pytest.approx((6.0 - 2.56) / 2 + np.arange(16) * 4 / 25)
    short = rule_times(t, {"duration_s": 1.0, "fps": 25.0, "n_frames": 25})
    assert short.min() >= 0 and short.max() <= 1.0  # clamped to the clip
    y0, x0, y1, x1 = crop_region(mod.Diving48Bridge.SPATIAL, 640, 480)
    assert y1 - y0 == pytest.approx(224 / 256) and x1 - x0 == pytest.approx(224 / round(640 * 256 / 480), abs=1e-3)


# ----------------------------------------------------------------------------- reader smoke on synthetic raw files


def _write_mp4(path: Path, *, width: int, height: int, seconds: float, fps: int) -> None:
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(path), "w", format="mp4") as c:
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = width, height, "yuv420p"
        for k in range(max(1, round(seconds * fps))):
            img = np.full((height, width, 3), (k * 7) % 256, dtype=np.uint8)
            img[: height // 2, : width // 2] = (255 - (k * 5) % 256)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


GEOMS = [(160, 120, 3.0, 25), (160, 120, 1.0, 60), (160, 120, 4.0, 30), (160, 120, 2.0, 50)]  # width, height, seconds, fps


def _synthetic_media(root: Path, clip_ids: list[str]) -> None:
    for cid, (w, h, sec, fps) in zip(clip_ids, GEOMS[: len(clip_ids)], strict=True):
        _write_mp4(root / "rgb" / f"{cid}.mp4", width=w, height=h, seconds=sec, fps=fps)


def test_reader_smoke_synthetic_raw(isolated: Path):
    """The reader hands over each stored mp4 byte for byte with the guaranteed info keys and ``meta == {"facts": {}}``;
    a clip shorter than the template's window is handed over as stored; canaries equal their originals."""
    from yavih import Benchmark
    from yavih.sample import RawStream

    m = Benchmark.from_dir(HERE).manifest("test")
    ids = list(m.clip_id[:4])
    media = isolated / "media"
    _synthetic_media(media, ids)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    samples = list(b.reader("test", rows=range(0, 4), canaries=1))
    assert [s.row for s in samples[:4]] == [0, 1, 2, 3] and len(samples) == 5
    for s, (w, h, sec, fps) in zip(samples[:4], GEOMS, strict=True):
        assert set(s.streams) == {"video"} and s.meta == {"facts": {}} and s.split == "test"
        v = s.streams["video"]
        f = media / "rgb" / f"{s.clip_id}.mp4"
        assert isinstance(v, RawStream) and v.kind == "video" and v.name == "video"
        assert Path(v.path).resolve() == f.resolve() and v.bytes() == f.read_bytes()
        with v.open() as fh:
            assert fh.read() == f.read_bytes()
        for k in INFO_KEYS:
            assert v.info[k] is not None, k
        assert (v.info["width"], v.info["height"]) == (w, h) and v.info["fps"] == pytest.approx(fps, rel=0.01)
        assert v.info["duration_s"] == pytest.approx(sec, abs=0.1) and v.info["bytes"] == f.stat().st_size
    c = samples[4]
    assert c._canary_of == c.row and c.streams["video"].bytes() == samples[c.row].streams["video"].bytes()


def test_reader_to_template_end_to_end(isolated: Path):
    """The template consumes what the reader hands over: N fixed, pos in [0, 1], ``used`` inside the media."""
    from yavih import Benchmark
    from yavih.bridge import validate_describe

    m = Benchmark.from_dir(HERE).manifest("train")
    ids = list(m.clip_id[:3])
    media = isolated / "media"
    _synthetic_media(media, ids)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    mod = F.load_module(HERE / "bridge_template.py", "diving48_template_e2e")
    bridge = mod.Diving48Bridge(b.view(), {})
    validate_describe(bridge.describe(), bridge, b)
    samples = list(b.reader("train", rows=range(0, 3)))
    feats, pos, used = F.encode_prepared(bridge, samples)
    n = (mod.N_FRAMES // mod.TUBELET) * mod.GRID * mod.GRID
    assert np.asarray(feats).shape == (3, n, 3) and np.asarray(pos).shape == (3, n, 4)
    assert ((pos[..., 1] >= 0) & (pos[..., 1] <= 1)).all()
    for s, u in zip(samples, used, strict=True):
        t = np.asarray(u["video"]["times"])
        assert len(t) == mod.N_FRAMES and (t >= 0).all() and (t <= s.streams["video"].info["duration_s"] + 1e-6).all()


# ----------------------------------------------------------------------------- reader smoke on local media


MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("rgb",))  # resolved at collection, before `isolated` hides the site config


@pytest.mark.skipif(MEDIA_ROOT is None, reason="Diving48 media not staged on this machine")
def test_reader_smoke_real_media(isolated: Path):
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": str(MEDIA_ROOT)})
    for split in CLIPS:
        samples = list(b.reader(split, rows=range(0, 3)))
        assert [s.row for s in samples] == [0, 1, 2]
        for s in samples:
            v = s.streams["video"]
            assert (v.info["width"], v.info["height"]) == (640, 480) and v.info["container"] == "mp4"
            assert v.bytes()[4:8] == b"ftyp" and s.meta == {"facts": {}}
