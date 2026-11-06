"""Folder tests of unav100 (run: ``python -m pytest "$(yavih bench path unav100)/tests" -q -p no:cacheprovider``).

Pins the folder's own facts: the frozen rows and events per split, the class vocabulary, the curation sums, the
protocol (the authors' tIoU thresholds bit for bit), the contracts, the prepare steps and
the bridge template; re-derives the manifest from rows rebuilt from the committed one (the parquet path needs
pyarrow, which CI does not have); and runs the reader over a synthetic tar laid out like the upstream one.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows = F.spec, F.rows

READOUT = "unav100"  # the folder's own scorer (eval.py + detection.py)
HEADLINE = "map_avg"
STREAMS = {"video": "video", "audio": "audio"}
CLIPS = {"train": 6489, "test": 2167}
ROWS = {"train": 6489, "validation": 2134, "test": 2167}
EVENTS = {"train": 18123, "validation": 6036, "test": 5900}
READINGS = ["a", "av", "v"]
THRESHOLDS = [float(x) for x in np.linspace(0.1, 0.9, 9)]
OUTPUT_CONTRACT = {"shape": "fixed", "coordinates": "media", "target_frame": None,
                   "requires": {"pos": ["t"], "streams": [], "grid": None, "t_bins": None}}
EXTERNAL: list[str] = []
HAS_EVAL = True
COLUMNS = ["clip_id", "split", "labels", "rel_media", "duration_s", "events"]


def _manifest_rows() -> list[dict[str, str]]:
    text = gzip.decompress((HERE / "manifest.csv.gz").read_bytes()).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _classes() -> list[str]:
    return [r["name"] for r in csv.DictReader(io.StringIO((HERE / "classes.csv").read_text(encoding="utf-8")))]


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
    assert dict(counts) == ROWS
    assert spec["splits"]["unused"] == ["validation"] and spec["splits"]["eval"] == ["test"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE
    assert spec["metrics"]["floors"] == ["map_avg_floor"] and spec["metrics"]["ci"] == {"map_avg": "bootstrap_clip"}


def test_protocol_is_the_authors(spec: dict):
    assert spec["protocol"]["tiou_thresholds"] == THRESHOLDS  # np.linspace(0.1, 0.9, 9), bit for bit
    assert spec["readout_params"] == {"T": 128} and spec["protocol"]["clip_seconds"] is None
    assert spec["protocol"]["train_draw"] is None


def test_input_contract(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items()} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)
    assert ic["facts"] == [] and ic["materialize"] is None


def test_output_contract_and_external(spec: dict):
    assert spec["output_contract"] == OUTPUT_CONTRACT
    assert [e["name"] for e in spec["external"] or []] == EXTERNAL
    assert (HERE / "eval.py").is_file() is HAS_EVAL and (HERE / "detection.py").is_file()


# ----------------------------------------------------------------------------- manifest facts


def test_rows_classes_and_events():
    mrows = _manifest_rows()
    assert list(mrows[0]) == COLUMNS
    names = _classes()
    assert len(names) == 100 and names == sorted(names)
    events = {s: 0 for s in ROWS}
    det, fmt = _detection()

    order = {s: k for k, s in enumerate(ROWS)}
    assert [order[r["split"]] for r in mrows] == sorted(order[r["split"]] for r in mrows)
    for r in mrows:
        assert re.fullmatch(r"[A-Za-z0-9_-]{11}", r["clip_id"]) and r["rel_media"] == f"video/{r['clip_id']}.mp4"
        ev = det.parse_events(r["events"], fmt)  # (class, start_s, end_s)
        dur = float(r["duration_s"])
        assert 10.0 <= dur <= 60.0 and np.all(ev[:, 2] <= dur + 1e-6)
        assert sorted({names[int(c)] for c in ev[:, 0]}) == r["labels"].split(";")
        events[r["split"]] += len(ev)
    assert events == EVENTS and sum(events.values()) == 30059


def test_curation_sums(spec: dict):
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["total_upstream"] == 10790 and cur["excluded"] == {}
    assert {s: v["listed"] for s, v in cur["per_split"].items()} == ROWS
    assert sum(v["listed"] for v in cur["per_split"].values()) == cur["total_upstream"]


def _upstream_events_from_manifest() -> list[tuple[str, str, float, float, float, str]]:
    """The upstream event rows (video_id, split, duration, start, end, label) rebuilt from the committed manifest, in
    a scrambled video order (the derivation must not depend on it)."""
    names = _classes()
    out = []
    for r in reversed(_manifest_rows()):
        for s, e, c in (tok.split(":") for tok in r["events"].split(";")):
            out.append((r["clip_id"], r["split"], float(r["duration_s"]), float(s), float(e), names[int(c)]))
    return out


def test_make_manifest_check():
    """``derive`` re-creates manifest.csv.gz and classes.csv byte for byte from the upstream event rows (rebuilt from
    the committed manifest: the HF parquets need pyarrow and the network); the parquet path runs when they are staged."""
    from yavih.freeze import csv_bytes

    mm = F.load_module(HERE / "make_manifest.py")
    rows_out, names, per_split = mm.derive(_upstream_events_from_manifest())
    assert csv_bytes(mm.MANIFEST_COLUMNS, rows_out) == gzip.decompress((HERE / "manifest.csv.gz").read_bytes())
    assert csv_bytes(mm.CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(names)]) == (HERE / "classes.csv").read_bytes()
    assert dict(per_split) == ROWS and len(rows_out) == mm.TOTAL_VIDEOS


def test_make_manifest_check_on_staged_parquets(spec: dict, capsys):
    root = F.site_root(HERE, "media_root")
    meta = None if root is None else root / "upstream" / "data"
    if meta is None or not (meta / "test.parquet").is_file():
        pytest.skip("unav100 parquets not staged on this host")
    pytest.importorskip("pyarrow")
    mm = F.load_module(HERE / "make_manifest.py")
    assert mm.main(["--no-media", "--metadata", str(meta), "--check"]) == 0
    printed = json.loads(capsys.readouterr().out)
    up = spec["manifest"]["upstream"]
    for key in ("source", "sha256", "rows", "upstream_files"):
        assert printed["upstream"][key] == up[key], key
    assert printed["manifest"]["sha256"] == spec["manifest"]["sha256"]


# ----------------------------------------------------------------------------- prepare/


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_download.sh", "02_unpack.sh", "03_index_tar.py", "04_verify.sh"]
    assert all(re.match(r"^\d+_[A-Za-z0-9_.-]+\.(?:sh|py)$", n) for n in names), names
    step01 = (HERE / "prepare" / "01_download.sh").read_text(encoding="utf-8")
    for name, sha in F.load_yaml(HERE)["source"]["sha256"].items():
        assert sha in step01, name  # every pinned upstream file is checked by the download step


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
    assert d["input"]["video"]["temporal"]["span"] == "full" and d["input"]["audio"]["temporal"]["max_s"] is None


# ----------------------------------------------------------------------------- reader


def _has_encoder(av, name: str) -> bool:
    try:
        av.codec.Codec(name, "w")
        return True
    except Exception:  # noqa: BLE001
        return False


def _write_mp4(path: Path, seconds: float, fps: int = 25, rate: int = 44100, size: tuple[int, int] = (32, 48)) -> None:
    """Tiny UnAV-shaped mp4: ``fps`` video + mono AAC at ``rate`` Hz, ``seconds`` long."""
    import av

    vcodec = next((c for c in ("libx264", "mpeg4") if _has_encoder(av, c)), None)
    if vcodec is None or not _has_encoder(av, "aac"):
        pytest.skip("PyAV build without an mp4 video encoder or aac")
    h, w = size
    with av.open(str(path), "w", format="mp4") as out:
        vs = out.add_stream(vcodec, rate=fps)
        vs.width, vs.height = w, h
        vs.pix_fmt = "yuv420p"
        aus = out.add_stream("aac", rate=rate)
        aus.layout = "mono"
        for i in range(int(round(seconds * fps))):
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            arr[..., 1] = i % 256
            for pkt in vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                out.mux(pkt)
        for pkt in vs.encode(None):
            out.mux(pkt)
        n = int(round(seconds * rate))
        wave = (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(n) / rate)).astype(np.float32)
        for k in range(0, n, 1024):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(wave[k : k + 1024][None, :]), format="fltp", layout="mono")
            fr.sample_rate = rate
            fr.pts = k
            for pkt in aus.encode(fr):
                out.mux(pkt)
        for pkt in aus.encode(None):
            out.mux(pkt)


def test_reader_smoke_synthetic_raw(tmp_path):
    """Two synthetic mp4s packed under the upstream member path ``.../raw_videos/<ytid>.mp4``, indexed with
    ``prepare/03_index_tar.index_tar``; the reader hands over the stored member bytes as both streams, the guaranteed
    facts, and ``meta == {"facts": {}}``; nothing decoded, no window."""
    pytest.importorskip("av")
    from yavih import Benchmark

    idx = F.load_module(HERE / "prepare" / "03_index_tar.py")
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    man = b.manifest("test")
    ytids = [man.media[k]["video"].removeprefix("video/").removesuffix(".mp4") for k in (0, 1)]
    staging = tmp_path / "staging"
    staging.mkdir()
    stored = {}
    with tarfile.open(tmp_path / idx.TAR_FILENAME, "w") as tar:
        for k, y in enumerate(ytids):
            mp4 = staging / f"{y}.mp4"
            _write_mp4(mp4, seconds=12.0 if k == 0 else 20.0, size=(32, 48) if k == 0 else (36, 64))
            stored[y] = mp4.read_bytes()
            tar.add(mp4, arcname=f"ssd3/gengtiantian/audio_visual_data/unav100/raw_videos/{y}.mp4")
        tar.add(staging / f"{ytids[0]}.mp4", arcname="ssd3/README.mp4")  # not under raw_videos/: skipped
    members, stats = idx.index_tar(tmp_path / idx.TAR_FILENAME)
    assert sorted(members) == sorted(f"video/{y}.mp4" for y in ytids) and stats["other"] == 1
    (tmp_path / "index.json").write_text(json.dumps({"version": 1, "shards": [idx.TAR_FILENAME], "members": members}), encoding="utf-8")
    ic = F.load_yaml(HERE)["input_contract"]
    samples = list(b.reader("test", rows=range(0, 2)))
    assert [s.row for s in samples] == [0, 1]
    for s, y in zip(samples[:2], ytids, strict=True):
        assert s.meta == {"facts": {}} and set(s.streams) == {"video", "audio"}
        v, a = s.streams["video"], s.streams["audio"]
        assert v.bytes() == a.bytes() == stored[y] and v.open().read() == stored[y]
        for st, keys in ((v, ic["info"]["video"]), (a, ic["info"]["audio"])):
            assert not [k for k in keys if st.info.get(k) is None], st.name
        assert v.info["fps"] == pytest.approx(25.0) and a.info["sample_rate"] == 44100
    assert samples[0].streams["video"].info["duration_s"] == pytest.approx(12.0, abs=0.1)
    assert samples[1].streams["video"].info["width"] == 64


def test_reader_smoke_if_media_present(spec: dict):
    """First present rows of each read split through the reader against the real tar (skips elsewhere)."""
    root = F.site_root(HERE, "media_root")
    if root is None or not (root / "index.json").is_file():
        pytest.skip("unav100 media_root / index.json not present on this host")
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": str(root)})
    members = set(json.loads((root / "index.json").read_text(encoding="utf-8"))["members"])
    for split in ("train", "test"):
        man = b.manifest(split)
        present = [k for k in range(len(man.clip_id)) if man.media[k]["video"] in members][:2]
        if not present:
            continue
        for s in (x for k in present for x in b.reader(split, rows=range(k, k + 1))):  # rows= takes a range
            data = s.streams["video"].bytes()
            assert data == s.streams["audio"].bytes() and data[4:8] == b"ftyp"
            for name in ("video", "audio"):
                info = s.streams[name].info
                assert all(info.get(k) is not None for k in spec["input_contract"]["info"][name]), (name, info)
            assert 9.9 <= s.streams["video"].info["duration_s"] <= 60.5
