"""``RawStream`` / ``Sample`` (stored bytes + facts, nothing decoded), the generic
``iter_samples`` over ``dir`` / ``tar_index`` layouts, the media-info table (census vs live
probe), canaries, batching and the byte read-ahead. Benchmark folders are written by ``test_benchmark``'s
builders; media by ``test_media_helpers``."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from test_benchmark import dir_stream, make_folder, set_streams, touch_media, with_roots
from test_media_helpers import (
    needs_av,
    needs_sf,
    write_av_mp4,
    write_tar_with_index,
    write_video,
    write_wav,
)
from yavih.benchmark import Benchmark
from yavih.media import DecoderMissing, MediaMissing
from yavih.sample import (
    INFO_VOCAB,
    MEDIA_INFO_KEYS,
    TRACK_FACTS,
    MediaInfoTable,
    RawStream,
    ReadAhead,
    Sample,
    _Source,
    batches,
    canary_rows,
    default_decode_workers,
    fetch,
    iter_samples,
    media_info_path,
)

# ----------------------------------------------------------------------------- folder builders


def _rows(n_train: int, n_val: int, ext: str = "mp4") -> list[list[str]]:
    return [[f"{sp}{k:02d}", sp, "a" if k % 2 else "b", f"{sp}/{sp}{k:02d}.{ext}"] for sp, n in (("train", n_train), ("val", n_val)) for k in range(n)]


def _video_bench(tmp_path: Path, monkeypatch, *, n_train: int = 8, n_val: int = 4, info: list[str] | None = None, write: bool = True, mutate=None,
                 probe_py: str | None = None) -> tuple[Benchmark, Path]:
    rows = _rows(n_train, n_val)
    media = tmp_path / "media"
    if write:
        for r in rows:
            write_video(media / r[3], fps=10, n_frames=10 + int(r[0][-2:]) % 3, width=32, height=24)

    def mut(s):
        s["budget"]["clips"] = {"train": n_train, "val": n_val}
        s["manifest"]["upstream"] = None
        s["input_contract"]["info"]["video"] = info if info is not None else ["duration_s", "fps", "n_frames", "width", "height", "container", "bytes"]
        if mutate is not None:
            mutate(s)

    folder = make_folder(tmp_path, "synth", rows=rows, mutate=mut, probe_py=probe_py)
    with_roots(monkeypatch, "synth", media_root=media)
    return Benchmark.from_dir(folder), media


# ----------------------------------------------------------------------------- RawStream


def test_raw_stream_dir_hands_over_the_stored_bytes(tmp_path: Path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"\x00stored-bytes\xff")
    st = RawStream("video", "video", src=_Source(("file", str(p))), clip_id="c0", info={"duration_s": 1.0})
    assert (st.name, st.kind, st._real_path, st.member) == ("video", "video", str(p), None)
    # path is an opaque handle: a hash-named symlink to the stored file, never the stored name
    assert st.path != str(p) and "x.mp4" not in st.path and st.path.endswith(".mp4") and os.path.realpath(st.path) == os.path.realpath(p)
    assert st.path == st.path and st.with_handle("other").path != st.path and st.with_handle("other").bytes() == p.read_bytes()
    b1 = st.bytes()
    assert b1 == p.read_bytes() and st.bytes() is b1  # cached after the first read
    fh = st.open()
    assert fh.read() == b1 and fh.seekable() and fh.seek(1) == 1 and fh.read(6) == b"stored"
    assert st.nbytes == len(b1)
    assert st.info == {"duration_s": 1.0}
    st.info["duration_s"] = 99.0  # a fresh dict per access: a bridge cannot alter the facts
    assert st.info == {"duration_s": 1.0}
    with pytest.raises(AttributeError, match="read-only"):
        st._path = "/elsewhere"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        st.path = "/elsewhere"  # type: ignore[misc]
    with pytest.raises(ValueError):
        RawStream("video", "text", src=_Source(("file", str(p))))


def test_raw_stream_missing_media_is_media_missing_with_a_reason(tmp_path: Path):
    absent = RawStream("audio", "audio", src=_Source(("file", str(tmp_path / "nope.wav"))), clip_id="c7")
    with pytest.raises(MediaMissing, match="absent file") as ei:
        absent.bytes()
    assert (ei.value.clip_id, ei.value.stream) == ("c7", "audio")
    no_entry = RawStream("audio", "audio", src=_Source(("absent", "no index entry 'k'")), clip_id="c8")
    with pytest.raises(MediaMissing, match="no index entry"):
        no_entry.open()
    tar = tmp_path / "s.tar"
    write_tar_with_index(tar, {"k": b"x" * 100})
    short = RawStream("audio", "audio", src=_Source(("member", str(tar), 512, 10**6)), clip_id="c9")
    with pytest.raises(MediaMissing, match="short read"):
        short.bytes()
    gone = RawStream("audio", "audio", src=_Source(("member", str(tmp_path / "gone.tar"), 0, 1)), clip_id="c9")
    with pytest.raises(MediaMissing, match="absent tar shard"):
        gone.bytes()


@needs_av
def test_info_live_probe_and_guaranteed_keys(tmp_path: Path, monkeypatch):
    p = tmp_path / "v.mp4"
    write_video(p, fps=10, n_frames=12, width=32, height=24)
    st = RawStream("video", "video", src=_Source(("file", str(p))), clip_id="c", required_info=("duration_s", "fps", "n_frames", "width", "height"), ext="mp4")
    info = st.info
    assert info["fps"] == 10.0 and info["n_frames"] == 12 and (info["width"], info["height"]) == (32, 24) and info["container"] == "mp4"
    assert info["bytes"] == p.stat().st_size and info["sample_rate"] is None
    # junk bytes: unreadable header → a missing row when a fact is guaranteed, nulls otherwise
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"junk")
    with pytest.raises(MediaMissing, match="unreadable header"):
        RawStream("video", "video", src=_Source(("file", str(junk))), clip_id="j", required_info=("duration_s",)).info  # noqa: B018
    loose = RawStream("video", "video", src=_Source(("file", str(junk))), clip_id="j").info
    assert loose["duration_s"] is None and loose["bytes"] == 4
    # a census entry with a null guaranteed key is a missing row too (never a silent null)
    entry = RawStream("video", "video", src=_Source(("file", str(p))), clip_id="e", info_entry={"duration_s": 1.0, "fps": None}, required_info=("fps",))
    with pytest.raises(MediaMissing, match="fps"):
        entry.info  # noqa: B018
    # no decoder installed (neither PyAV nor the ffprobe fallback): guaranteed facts cannot be probed →
    # DecoderMissing (exit 3), never a fake value
    monkeypatch.setitem(sys.modules, "av", None)
    monkeypatch.setattr("yavih.media.info._ffprobe_available", lambda: False)
    with pytest.raises(DecoderMissing):
        RawStream("video", "video", src=_Source(("file", str(p))), required_info=("fps",)).info  # noqa: B018


@needs_av
def test_a_file_without_the_track_is_handed_over_with_has_track_false(tmp_path: Path):
    """audioset20k: 3 published mp4s hold no audio track ("No clips are excluded"); the audio
    stream of a video-only mp4 is NOT a missing row, even when the contract guarantees the audio facts; its info
    says ``has_track: false`` with the track facts null and ``container`` / ``bytes`` set, from a live probe and
    from a census entry alike. A file with the track says ``has_track: true``; an unreadable file stays missing."""
    from yavih.media import DecodeError
    from yavih.media import decode as _decode
    from yavih.media.info import probe_info

    p = tmp_path / "video_only.mp4"
    write_video(p, fps=10, n_frames=12, width=32, height=24)
    audio_facts = ("duration_s", "sample_rate", "n_samples", "channels", "container", "bytes")
    au = RawStream("audio", "audio", src=_Source(("file", str(p))), clip_id="c", required_info=audio_facts, ext="mp4")
    info = au.info
    assert info["has_track"] is False and info["container"] == "mp4" and info["bytes"] == p.stat().st_size
    assert all(info[k] is None for k in TRACK_FACTS["audio"]) and set(MEDIA_INFO_KEYS) <= set(info)
    vi = RawStream("video", "video", src=_Source(("file", str(p))), clip_id="c", required_info=("duration_s", "fps", "n_frames"), ext="mp4")
    assert vi.info["has_track"] is True and vi.info["n_frames"] == 12
    assert au.bytes() == vi.bytes() == p.read_bytes()  # the same mp4 is handed over whole
    census = probe_info(p, "audio", demux=True)  # what `bench verify --census` writes for the cell
    assert census["has_track"] is False and census["sample_rate"] is None
    entry = RawStream("audio", "audio", src=_Source(("file", str(p))), clip_id="e", info_entry=census, required_info=audio_facts)
    assert entry.info["has_track"] is False
    # a census entry that claims the track but lacks a guaranteed fact is still a missing row
    with pytest.raises(MediaMissing, match="sample_rate"):
        RawStream("audio", "audio", src=_Source(("file", str(p))), clip_id="x", info_entry=dict(census, has_track=True), required_info=audio_facts).info  # noqa: B018
    # decoding the absent track is an explicit error naming the fact (the bridge decides what to feed instead)
    with pytest.raises(DecodeError, match="has_track: false"):
        _decode.audio(au)
    # a file that has both tracks
    q = tmp_path / "av.mp4"
    write_av_mp4(q)
    assert RawStream("audio", "audio", src=_Source(("file", str(q))), clip_id="q", required_info=audio_facts, ext="mp4").info["has_track"] is True
    # junk is still unreadable (no has_track fact can be read from it)
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"junk")
    with pytest.raises(MediaMissing, match="unreadable header"):
        RawStream("audio", "audio", src=_Source(("file", str(junk))), clip_id="j", required_info=audio_facts).info  # noqa: B018


@needs_av
def test_two_streams_of_one_mp4_share_the_bytes_and_carry_their_own_track_facts(tmp_path: Path, monkeypatch):
    """AVE / avsync: ``video`` and ``audio`` read the same media cell; the file is read once and each stream's
    ``info`` describes its own track."""
    rows = [[f"y{k}#0", sp, "a" if k % 2 else "b", f"y{k}.mp4"] for k, sp in enumerate(["train"] * 4 + ["val"] * 2)]
    media = tmp_path / "media"
    for r in rows:
        write_av_mp4(media / r[3], fps=10, n_frames=10, rate=8000)

    def mut(s):
        s["manifest"]["columns"]["rel_media"]["stream"] = ["video", "audio"]
        set_streams(s, {"video": dir_stream("video", "rel_media"), "audio": dir_stream("audio", "rel_media")},
                    {"video": ["duration_s", "fps", "width", "height", "container", "bytes"], "audio": ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]})

    with_roots(monkeypatch, "synth", media_root=media)
    b = Benchmark.from_dir(make_folder(tmp_path, "synth", rows=rows, mutate=mut))
    s = b.sample("val", 0)
    v, a = s.streams["video"], s.streams["audio"]
    assert v._real_path == a._real_path == str(media / "y4.mp4") and v._src is a._src
    assert os.path.realpath(v.path) == os.path.realpath(a.path) == os.path.realpath(media / "y4.mp4") and "y4" not in v.path + a.path
    assert v.bytes() is a.bytes()
    assert v.info["fps"] == 10.0 and v.info["sample_rate"] is None
    assert a.info["sample_rate"] == 8000 and a.info["n_samples"] == 8000 and a.info["fps"] is None and a.info["duration_s"] == 1.0


# ----------------------------------------------------------------------------- Sample + generic reader


def test_iter_samples_dir_layout_meta_and_order(tmp_path: Path, monkeypatch):
    b, media = _video_bench(tmp_path, monkeypatch, write=False)
    touch_media(media, [f"train/train{k:02d}.mp4" for k in range(8)], b"stored")
    samples = list(iter_samples(b, "train"))
    assert [s.row for s in samples] == list(range(8)) and all(isinstance(s, Sample) for s in samples)
    s = samples[3]
    assert (s.clip_id, s.split, s._canary_of) == ("train03", "train", None)
    assert set(s.streams) == {"video"} and s.streams["video"]._real_path == str(media / "train/train03.mp4")
    assert "train" not in s.streams["video"].path and os.path.realpath(s.streams["video"].path) == os.path.realpath(media / "train/train03.mp4")
    assert s.streams["video"].bytes() == b"stored"
    assert s.meta == {"facts": {}} and json.dumps(s.meta)
    assert not hasattr(s, "media_key")
    assert [x.row for x in iter_samples(b, "train", rows=range(4, 7))] == [4, 5, 6]
    with pytest.raises(ValueError):
        list(iter_samples(b, "train", rows=range(6, 12)))
    # shard without rows: the plan_shards range
    from yavih.io.shards import plan_shards

    lo, hi = plan_shards(8, 2, 3)[1]
    assert [x.row for x in iter_samples(b, "train", batch_size=2, shard=(1, 3))] == list(range(lo, hi))
    # construction never touches the bytes: a missing file surfaces at bytes()
    v0 = next(iter_samples(b, "val"))
    with pytest.raises(MediaMissing, match="absent file"):
        v0.streams["video"].bytes()


def test_meta_holds_exactly_the_input_contract_facts(tmp_path: Path, monkeypatch):
    def mut(s):
        s["protocol"].update({"annotated_second": 5.0, "offsets_seconds": [-1.0, 0.0, 1.0]})
        s["input_contract"]["facts"] = ["annotated_second", "clip_seconds"]

    # offsets_seconds: a label fact the folder's probe.py consumes, never handed over
    b, _ = _video_bench(tmp_path, monkeypatch, write=False, mutate=mut, probe_py='PROTOCOL_KEYS = frozenset({"offsets_seconds"})\n')
    s = b.sample("train", 0)
    assert s.meta == {"facts": {"annotated_second": 5.0, "clip_seconds": 10.0}}  # never offsets, labels or columns
    s.meta["facts"]["annotated_second"] = 0.0
    assert b.sample("train", 0).meta["facts"]["annotated_second"] == 5.0 and b.spec.protocol["annotated_second"] == 5.0


def test_iter_samples_is_duck_typed_on_the_benchmark(tmp_path: Path):
    """``iter_samples`` needs only ``manifest(split)``, ``manifest_hash``, ``spec`` and roots: a folder
    ``reader.py`` or a test may hand it any such object."""
    from types import SimpleNamespace

    p = tmp_path / "m" / "a.wav"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"RIFF")
    spec = {
        "streams": {"audio": {"kind": "audio", "media_column": "c", "root": "r", "layout": "dir", "layout_params": {}, "required": True}},
        "splits": {"train": ["train"], "eval": []},
        "input_contract": {"info": {"audio": []}, "facts": []},
        "protocol": {},
        "data": {"roots": {"r": {"default": "/x"}}, "tar_index": None},
    }

    class _M(list):
        pass

    m = _M([0])
    m.clip_id, m.row, m.media = ["a"], np.arange(1), [{"audio": "a.wav"}]
    bench = SimpleNamespace(spec=spec, manifest_hash="h" * 64, manifest=lambda split: m)
    (s,) = list(iter_samples(bench, "train", roots={"r": str(tmp_path / "m")}))
    assert s.streams["audio"].bytes() == b"RIFF" and s.meta == {"facts": {}}


def test_required_false_stream_absent_is_omitted(tmp_path: Path, monkeypatch):
    header = ["clip_id", "split", "label", "rel_media", "rel_audio"]
    rows = [r + [f"{r[1]}/{r[0]}.wav" if i % 2 == 0 else ""] for i, r in enumerate(_rows(4, 2))]
    media = tmp_path / "media"
    touch_media(media, [r[3] for r in rows] + [r[4] for r in rows if r[4] and r[0] != "train02"])

    def mut(s):
        s["manifest"]["columns"]["rel_audio"] = {"type": "str", "role": "media", "stream": "audio"}
        set_streams(s, {"video": dir_stream("video", "rel_media"), "audio": dir_stream("audio", "rel_audio", required=False)})

    with_roots(monkeypatch, "synth", media_root=media)
    b = Benchmark.from_dir(make_folder(tmp_path, "synth", header=header, rows=rows, mutate=mut))
    got = {s.clip_id: set(s.streams) for s in b.reader("train")}
    assert got == {"train00": {"video", "audio"}, "train01": {"video"}, "train02": {"video"}, "train03": {"video"}}


def test_tar_index_audio(tmp_path: Path, monkeypatch):
    from test_benchmark import TAR_CLASSES, TAR_HEADER, TAR_ROWS, _tar_mutate

    media = tmp_path / "media"
    members = {r[3]: f"audio-{r[0]}".encode() for r in TAR_ROWS}
    write_tar_with_index(media / "shard-000000.tar", members, media / "index.json")
    with_roots(monkeypatch, "tarbench", media_root=media)
    b = Benchmark.from_dir(make_folder(tmp_path, "tarbench", header=TAR_HEADER, rows=TAR_ROWS, classes=TAR_CLASSES, mutate=_tar_mutate))
    tr, ev = list(b.reader("train")), list(b.reader("eval"))
    assert len(tr) == 4 and len(ev) == 2
    au = ev[1].streams["audio"]
    assert au.bytes() == b"audio-b1" and au.member[0] == str(media / "shard-000000.tar") and au.member[2] == len(b"audio-b1")
    for s in tr + ev:
        assert fetch(s) >= 0.0
    # the index itself missing: every row is missing at the read-ahead, the reader still yields rows
    (media / "index.json").unlink()
    from yavih.media import tarindex

    tarindex._INDEX_CACHE.clear()
    s = next(Benchmark.from_dir(b.folder).reader("train"))
    with pytest.raises(MediaMissing, match="tar index"):
        fetch(s)


# ----------------------------------------------------------------------------- media-info table


@needs_av
def test_census_info_equals_live_probe_and_needs_no_decoder(tmp_path: Path, monkeypatch, yavih_home):
    from yavih.benchmark import bench_verify
    from yavih.media.info import write_census

    b, media = _video_bench(tmp_path, monkeypatch, n_train=4, n_val=2)
    assert b.media_info() is None
    live = {(s.split, s.row): s.streams["video"].info for split in ("train", "val") for s in b.reader(split)}
    res = write_census(b, workers=2)
    assert res["rows"] == 6 and res["missing"] == 0 and res["failed"] == 0
    assert res["path"] == str(media_info_path("synth", b.manifest_hash))
    table = MediaInfoTable.load(res["path"])
    assert table.sha256 == res["sha256"] and len(table) == 6
    assert MediaInfoTable.write(tmp_path / "again.jsonl.gz", {k: table.get(*k) for k in table._entries}).sha256 == table.sha256  # byte-identical
    b2 = Benchmark.from_dir(b.folder)
    # with the census present the reader imports no decoder
    for mod in ("av", "soundfile", "PIL"):
        monkeypatch.setitem(sys.modules, mod, None)
    census = {(s.split, s.row): s.streams["video"].info for split in ("train", "val") for s in b2.reader(split)}
    assert census == live
    # bench verify reports the table (its own census call needs the decoders back)
    monkeypatch.undo()
    with_roots(monkeypatch, "synth", media_root=media)
    rep = bench_verify(str(b.folder), census=True, decode_sample=4)
    assert rep["media_info"]["rows"] == 6 and rep["media_info"]["sha256"] == table.sha256 and rep["media_info"]["failed"] == 0
    assert rep["decode_sample"]["ok"] == 4 and rep["decode_sample"]["lib"]["video"]["lib"] == "pyav" and rep["read"]["ok"] == 4


@needs_sf
def test_census_skips_missing_media_and_counts_unreadable_headers(tmp_path: Path, monkeypatch, yavih_home):
    from yavih.media.info import write_census

    rows = _rows(4, 2, "wav")
    media = tmp_path / "media"
    for r in rows[:4]:
        write_wav(media / r[3], seconds=0.5, rate=8000, channels=1)
    touch_media(media, [rows[4][3]], b"not a wav")  # val00 unreadable, val01 missing

    def mut(s):
        s["manifest"]["columns"]["rel_media"]["stream"] = "audio"
        set_streams(s, {"audio": dir_stream("audio", "rel_media")}, {"audio": ["duration_s", "sample_rate", "n_samples"]})

    with_roots(monkeypatch, "synth", media_root=media)
    b = Benchmark.from_dir(make_folder(tmp_path, "synth", rows=rows, mutate=mut))
    res = write_census(b)
    assert (res["rows"], res["missing"], res["failed"]) == (4, 1, 1) and res["failures"][0]["key"] == rows[4][3]
    s = Benchmark.from_dir(b.folder).sample("train", 0)
    assert s.streams["audio"].info["n_samples"] == 4000 and s.streams["audio"].info["container"] == "wav"


# The census probes stored files by path, drops a tar member's bytes after its probe, bounds the cells in flight and
# prints progress, so its memory stays flat in the number of media cells.


def _track_source_reads(monkeypatch) -> list[_Source]:
    seen: list[_Source] = []
    orig = _Source.read

    def read(self, clip_id, stream):
        seen.append(self)
        return orig(self, clip_id, stream)

    monkeypatch.setattr(_Source, "read", read)
    return seen


def test_source_release_drops_cached_bytes(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"stored")
    src = _Source(("file", str(p)))
    assert src.read("c", "video") == b"stored" and src.cached
    src.release()
    assert not src.cached and src.size == 6
    p.write_bytes(b"stored again")
    assert src.read("c", "video") == b"stored again" and src.cached  # read again from the store
    _Source(("absent", "x")).release()  # nothing cached: a no-op


@needs_av
@needs_sf
def test_probe_info_of_a_file_stream_reads_by_path_and_caches_nothing(tmp_path: Path, monkeypatch):
    from yavih.media.info import probe_info

    write_av_mp4(tmp_path / "x.mp4", n_frames=10)
    write_wav(tmp_path / "y.wav", seconds=0.5, rate=8000, channels=1)
    seen = _track_source_reads(monkeypatch)
    for name, kind in (("x.mp4", "video"), ("x.mp4", "audio"), ("y.wav", "audio")):
        p = tmp_path / name
        seen.clear()
        st = RawStream(kind, kind, src=_Source(("file", str(p))), clip_id="c0")
        by_path = probe_info(st, kind, demux=True)
        assert seen == [] and not st._src.cached  # the header was read in place: no bytes loaded or cached
        assert by_path == probe_info(p.read_bytes(), kind, demux=True, ext=name.rsplit(".", 1)[-1])  # the same facts
        assert by_path["bytes"] == p.stat().st_size
        st.bytes()  # already cached: probed from the cached bytes, same facts
        assert probe_info(st, kind, demux=True) == by_path
    gone = RawStream("video", "video", src=_Source(("file", str(tmp_path / "gone.mp4"))), clip_id="c9")
    with pytest.raises(MediaMissing, match="absent file"):  # an absent file is still MediaMissing, not DecodeError
        probe_info(gone, "video")


@needs_sf
def test_census_reads_no_file_bytes_bounds_in_flight_and_reports_progress(tmp_path: Path, monkeypatch, yavih_home):
    import io as _io
    from concurrent.futures import ThreadPoolExecutor

    import yavih.media.info as info_mod

    rows = _rows(16, 8, "wav")
    media = tmp_path / "media"
    for r in rows:
        write_wav(media / r[3], seconds=0.25, rate=8000, channels=1)

    def mut(s):
        s["manifest"]["columns"]["rel_media"]["stream"] = "audio"
        set_streams(s, {"audio": dir_stream("audio", "rel_media")}, {"audio": ["duration_s", "sample_rate", "n_samples"]})

    with_roots(monkeypatch, "synth", media_root=media)
    b = Benchmark.from_dir(make_folder(tmp_path, "synth", rows=rows, mutate=mut))
    live = {(s.split, s.row): s.streams["audio"].info for split in ("train", "val") for s in b.reader(split)}

    outstanding = {"now": 0, "peak": 0}

    class CountingPool(ThreadPoolExecutor):
        def submit(self, fn, *a, **k):
            outstanding["now"] += 1
            outstanding["peak"] = max(outstanding["peak"], outstanding["now"])
            fut = super().submit(fn, *a, **k)
            fut.add_done_callback(lambda _f: outstanding.__setitem__("now", outstanding["now"] - 1))
            return fut

    monkeypatch.setattr(info_mod, "ThreadPoolExecutor", CountingPool)
    seen = _track_source_reads(monkeypatch)
    log = _io.StringIO()
    res = info_mod.write_census(b, workers=1, progress_every=5, progress_file=log)
    assert (res["rows"], res["missing"], res["failed"]) == (24, 0, 0)
    assert seen == []  # every stored file was probed by path: no media bytes held anywhere
    assert outstanding["peak"] <= 4  # 4 x workers cells in flight, never the whole table (24)
    lines = log.getvalue().splitlines()
    assert len(lines) == 5 and all(x.startswith("[yavih] census synth: ") for x in lines)
    assert [x.split(": ")[1].split(" ")[0] for x in lines] == ["5/24", "10/24", "15/24", "20/24", "24/24"] and "cells done" in lines[-1]
    quiet = _io.StringIO()
    info_mod.write_census(b, workers=2, progress_file=quiet)  # fewer cells than progress_every (1000): silent
    assert quiet.getvalue() == ""
    census = {(s.split, s.row): s.streams["audio"].info for split in ("train", "val") for s in Benchmark.from_dir(b.folder).reader(split)}
    assert census == live  # path probe == the reader's live (bytes) probe


@needs_sf
def test_census_drops_tar_member_bytes_after_each_probe(tmp_path: Path, monkeypatch, yavih_home):
    from test_benchmark import TAR_CLASSES, TAR_HEADER, TAR_ROWS, _tar_mutate
    from yavih.media.info import write_census

    media = tmp_path / "media"
    wav = tmp_path / "w.wav"
    write_wav(wav, seconds=0.5, rate=8000, channels=1)
    members = {r[3]: wav.read_bytes() for r in TAR_ROWS}
    write_tar_with_index(media / "shard-000000.tar", members, media / "index.json")
    with_roots(monkeypatch, "tarbench", media_root=media)
    b = Benchmark.from_dir(make_folder(tmp_path, "tarbench", header=TAR_HEADER, rows=TAR_ROWS, classes=TAR_CLASSES, mutate=_tar_mutate))
    seen = _track_source_reads(monkeypatch)
    res = write_census(b, workers=2)
    assert (res["rows"], res["missing"], res["failed"]) == (6, 0, 0)
    assert len({id(s) for s in seen}) == 6  # every member was read ...
    assert not any(s.cached for s in seen)  # ... and none stays resident once its probe is done
    table = MediaInfoTable.load(res["path"])
    assert table.get("audio", TAR_ROWS[0][3])["n_samples"] == 4000


# ----------------------------------------------------------------------------- canaries, batches


def test_canary_rows_rule_and_determinism():
    rows = range(0, 40)
    a = canary_rows(40, rows, 4, 8, manifest_hash="h" * 64, split="train", shard_index=0, canary_seed=0)
    assert len(a) == 8 and len(set(a)) == 8 and all(r < 40 - 8 for r in a)
    assert a == canary_rows(40, rows, 4, 8, manifest_hash="h" * 64, split="train", shard_index=0, canary_seed=0)
    assert a != canary_rows(40, rows, 4, 8, manifest_hash="h" * 64, split="train", shard_index=1, canary_seed=0)
    assert a != canary_rows(40, rows, 4, 8, manifest_hash="g" * 64, split="train", shard_index=0, canary_seed=0)
    one = canary_rows(40, range(0, 7), 1, 8, manifest_hash="h" * 64, split="train", shard_index=0, canary_seed=0)
    assert len(one) == 1 and one[0] < 7 - 2  # min(8, 7 // 4) = 1, from rows >= 2B before the end
    assert canary_rows(40, range(0, 3), 1, 8, manifest_hash="h" * 64, split="train", shard_index=0, canary_seed=0) == []
    assert canary_rows(40, range(0, 8), 4, 8, manifest_hash="h" * 64, split="train", shard_index=0, canary_seed=0) == []  # none 2B before the end


def test_canaries_equal_their_originals_except_canary_of(tmp_path: Path, monkeypatch):
    b, media = _video_bench(tmp_path, monkeypatch, n_train=16, write=False)
    touch_media(media, [f"train/train{k:02d}.mp4" for k in range(16)], b"x")
    out = list(iter_samples(b, "train", batch_size=2, canaries=8, canary_seed=3))
    regular, canaries = out[:16], out[16:]
    assert len(canaries) == 4 and all(c._canary_of == c.row for c in canaries)
    for c in canaries:
        r = regular[c.row]
        # the same row, facts and stored bytes; a different opaque path, so a path-keyed cache misses
        assert (c.row, c.clip_id, c.split) == (r.row, r.clip_id, r.split) and c.meta == r.meta and set(c.streams) == set(r.streams)
        assert c.streams["video"]._real_path == r.streams["video"]._real_path and c.streams["video"].path != r.streams["video"].path
        assert c.streams["video"] is not r.streams["video"] and c.streams["video"]._src is not r.streams["video"]._src
        assert c.streams["video"].bytes() == r.streams["video"].bytes()
        assert "canary" not in json.dumps(c.meta)
    # batches: canaries never share a batch with regular rows
    bs = list(batches(out, 3))
    assert [len(x) for x in bs] == [3, 3, 3, 3, 3, 1, 3, 1]
    assert all(len({s._canary_of is None for s in batch}) == 1 for batch in bs)
    with pytest.raises(ValueError):
        list(batches(out, 0))


def test_sample_pickles_for_process_pools(tmp_path: Path, monkeypatch):
    b, media = _video_bench(tmp_path, monkeypatch, write=False)
    touch_media(media, ["train/train00.mp4"], b"payload")
    s = b.sample("train", 0)
    s.streams["video"].bytes()
    clone = pickle.loads(pickle.dumps(s))
    assert clone == s and clone.streams["video"].bytes() == b"payload"


# ----------------------------------------------------------------------------- byte read-ahead


def test_default_decode_workers():
    assert default_decode_workers({"SLURM_CPUS_PER_TASK": "8"}) == 7
    assert default_decode_workers({"SLURM_CPUS_PER_TASK": "1"}) == 1
    assert default_decode_workers({"SLURM_CPUS_PER_TASK": "64"}) == 16
    assert default_decode_workers({"SLURM_CPUS_PER_TASK": "junk"}) == 1
    n = default_decode_workers({})
    from yavih.resources import available_cpus  # affinity mask / cgroup quota, not os.cpu_count()

    assert 1 <= n <= 16 and n == max(1, min(16, available_cpus({}) // 2))
    assert default_decode_workers({"YAVIH_CPUS": "4", "SLURM_CPUS_PER_TASK": "64"}) == 3  # exec --cpus / --slots wins


@needs_av
def test_read_ahead_reads_bytes_and_turns_missing_media_into_missing_rows(tmp_path: Path, monkeypatch):
    b, media = _video_bench(tmp_path, monkeypatch, n_train=10)
    (media / "train/train03.mp4").unlink()
    (media / "train/train07.mp4").write_bytes(b"truncated garbage")  # header unreadable → guaranteed facts missing
    with ReadAhead(workers=3, depth=2) as ra:
        out = list(ra.run(batches(b.reader("train"), 4)))
    assert [len(x.samples) + len(x.missing) for x in out] == [4, 4, 2]
    present = [s.row for x in out for s in x.samples]
    missing = {s.row: str(e) for x in out for s, e in x.missing}
    assert present == [0, 1, 2, 4, 5, 6, 8, 9] and set(missing) == {3, 7}
    assert "absent file" in missing[3] and "unreadable header" in missing[7]
    for x in out:
        assert x.read_s >= 0.0
        for s in x.samples:  # in memory before the bridge sees them
            assert s.streams["video"]._src.cached and s.streams["video"]._info is not None
    with pytest.raises(ValueError):
        ReadAhead(workers=0)


def test_read_ahead_propagates_decoder_missing(tmp_path: Path, monkeypatch):
    """A missing decoder is a site problem (exit 3), never a missing row."""
    b, media = _video_bench(tmp_path, monkeypatch, n_train=4, write=False)
    touch_media(media, [f"train/train{k:02d}.mp4" for k in range(4)], b"x")
    monkeypatch.setitem(sys.modules, "av", None)
    monkeypatch.setattr("yavih.media.info._ffprobe_available", lambda: False)  # no ffprobe fallback either
    with ReadAhead(workers=2, depth=0) as ra, pytest.raises(DecoderMissing):
        list(ra.run(batches(b.reader("train"), 2)))


def test_media_missing_is_reexported():
    import yavih.media
    import yavih.sample

    assert yavih.sample.MediaMissing is yavih.media.MediaMissing is MediaMissing


def test_info_vocabulary_matches_the_documented_table():
    assert INFO_VOCAB["video"] == {"duration_s", "fps", "n_frames", "width", "height", "container", "bytes"}
    assert INFO_VOCAB["audio"] == {"duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"}
    assert INFO_VOCAB["image"] == {"width", "height", "container", "bytes"}
    assert set(INFO_VOCAB) == {"video", "audio", "image"}
    assert set(MEDIA_INFO_KEYS) == set().union(*INFO_VOCAB.values()) | {"has_track"}
    assert "has_track" not in set().union(*INFO_VOCAB.values())  # a fact of the file, never a guaranteed key
    assert TRACK_FACTS["audio"] == {"duration_s", "sample_rate", "n_samples", "channels"}
    assert TRACK_FACTS["video"] == {"duration_s", "fps", "n_frames", "width", "height"}


def test_sample_module_is_core_only():
    """``yavih.sample`` imports numpy + stdlib (+ ``yavih.media.tarindex``); decoders only lazily."""
    import subprocess

    code = (
        "import sys\n"
        "for m in ('av', 'soundfile', 'PIL', 'scipy', 'torch'): sys.modules[m] = None\n"
        "import yavih.sample\n"
        "bad = [m for m in ('yavih.media.info', 'yavih.media.decode') if m in sys.modules]\n"
        "print(','.join(bad) or 'ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr




@pytest.mark.parametrize("name", ["k400", "in1k", "esc50"])
def test_stream_path_is_opaque_for_label_bearing_layouts(name: str, tmp_path: Path, monkeypatch):
    """The stored paths carry the label (k400 ``val/<class>/…``, in1k ``val/<wnid>/…``, ESC-50's
    ``…-<target>.wav``) and the split; ``RawStream.path`` hands over a hash-named handle instead."""
    import re

    monkeypatch.setenv("YAVIH_DATA__ROOT", str(tmp_path / "data"))  # paths only: no media needs to exist
    b = Benchmark.load(name)
    split = list(b.spec.splits.read_values())[-1]
    for i in (0, 1, 2):
        s = b.sample(split, i)
        for st in s.streams.values():
            real = st._real_path
            assert real is not None
            assert re.fullmatch(r"[0-9a-f]{24}\.\w+", os.path.basename(st.path)), st.path
            parent = os.path.basename(os.path.dirname(real))
            stem = os.path.splitext(os.path.basename(real))[0]
            assert parent not in st.path and stem not in st.path and f"/{split}/" not in st.path
