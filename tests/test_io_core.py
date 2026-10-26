"""io-core contract tests (docs/features.md, docs/execution.md).

Synthetic data only: a duck-typed Benchmark/Bridge pair stands in for Benchmark and Bridge (the writer reads only
``name, spec.protocol_version, manifest_hash, protocol_hash, data_status()`` and
``name, batch_size, dtype, pos_mode, args``).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from yavih import hashing
from yavih.io import format as fmt
from yavih.io.format import FeatureMetaError, FeatureShapeError, FeatureValueError, FormatError
from yavih.io.reader import DEFAULT_RAM_BUDGET_BYTES, FeatureReader, RowView, ram_budget_bytes
from yavih.io.shards import n_batches, plan_shards, shard_of_row
from yavih.io.sidecar import (
    RESUME_MATCH_KEYS,
    SIDECAR_KEYS,
    build_sidecar,
    clean_tmp,
    move_stale,
    peek_shard,
    read_not_applicable,
    read_sidecar,
    resume_shard,
    sidecar_matches_plan,
    write_json_atomic,
    write_not_applicable,
)
from yavih.io.writer import FeatureWriter, split_dir_for

# ----------------------------------------------------------------------------- fakes


class _Spec:
    protocol_version = 1


class FakeBench:
    name = "synth_clf"
    spec = _Spec()
    manifest_hash = "m" * 64
    protocol_hash = "p" * 64
    input_hash = "i" * 64

    def __init__(self, data_status_ok: bool = False) -> None:
        self._ok = data_status_ok

    def data_status(self):
        if not self._ok:
            raise NotImplementedError("no data status")

        class DS:
            media_fingerprint = "f" * 64
            roots = {"media_root": "/data/synth"}

        return DS()


class FakeBridge:
    name = "fake-bridge"
    batch_size = 4
    dtype = "float16"
    pos_mode = "none"
    streams: tuple[str, ...] = ()

    def __init__(self, **over):
        for k, v in over.items():
            setattr(self, k, v)
        self.args = {"layer": "12"}


def make_describe(bridge: FakeBridge, weights: str | None = "a" * 64) -> dict:
    return {
        "weights_sha256": weights,
        "weights_ref": "/some/path.pt",
        "dtype": bridge.dtype,
        "batch_size": bridge.batch_size,
        "streams": list(bridge.streams),
        "pos_mode": bridge.pos_mode,
        "model": "fake-model",
        "reading": "v",
        "layer": 12,
        "pooling": "none",
        "views": 1,
        "input": {"video": {"temporal": {"rule": "uniform", "n": 2, "span": "full"}, "spatial": {"resize": None, "crop": None,
                                                                                                   "interp": "bilinear", "antialias": True},
                            "norm": {"mean": [0.5], "std": [0.5], "source": "fake"}, "decoder": {"lib": "pyav", "version": "0"}}},
        "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
        "args": {"layer": 12},
    }


#: The RawStream facts run_extract copies into every member.
VIDEO_INFO = {"duration_s": 10.0, "fps": 2.0, "n_frames": 20, "sample_rate": None, "n_samples": None, "channels": None,
              "width": 16, "height": 16, "container": "mp4", "bytes": 1000}


def vmeta(rows) -> list[dict]:
    """Row meta (``write(meta=…)``): ``media_key``, ``media`` and ``used`` per row."""
    return [{"media_key": f"mk{r}", "media": {"video": dict(VIDEO_INFO)}, "used": {"video": {"times": [2.5, 7.5]}}} for r in rows]


def feats_for(rows: list[int], N: int, D: int, seed: int = 0) -> np.ndarray:
    """A pure function of the row so shard layouts can be compared bit for bit."""
    out = np.empty((len(rows), N, D), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i] = np.random.default_rng(seed * 1_000_003 + r).standard_normal((N, D)).astype(np.float32)
    return out


def write_split(
    out_dir: Path,
    *,
    n_rows: int,
    n_shards: int,
    bridge: FakeBridge | None = None,
    N: int = 6,
    D: int = 16,
    missing: set[int] = frozenset(),
    canaries: int = 0,
    const_pos: np.ndarray | None = None,
    split: str = "train",
    bench: FakeBench | None = None,
    env: dict | None = None,
) -> Path:
    bridge = bridge or FakeBridge()
    bench = bench or FakeBench()
    describe = make_describe(bridge)
    ranges = plan_shards(n_rows, bridge.batch_size, n_shards)
    if bridge.pos_mode == "constant" and const_pos is None:
        const_pos = np.random.default_rng(99).random((N, 4)).astype(np.float32)
    for i, (lo, hi) in enumerate(ranges):
        with FeatureWriter(
            out_dir, split, shard=(i, n_shards), rows=(lo, hi), bench=bench, bridge=bridge, describe=describe, source_sha256="s" * 64
        ) as w:
            written: list[int] = []
            for b in range(lo, hi, bridge.batch_size):
                batch = [r for r in range(b, min(b + bridge.batch_size, hi))]
                keep = []
                for r in batch:
                    if r in missing:
                        w.record_missing(r, f"clip{r:04d}", "MediaMissing: video")
                    else:
                        keep.append(r)
                if not keep:
                    continue
                f = feats_for(keep, N, D)
                pos = None
                if bridge.pos_mode == "constant":
                    pos = const_pos
                elif bridge.pos_mode == "per_sample":
                    pos = np.stack([np.full((N, 4), r / max(n_rows, 1), dtype=np.float32) for r in keep])
                # the stream facts + what the bridge used (uniform n = 2 over a 10 s clip: bin centres)
                meta = vmeta(keep)
                w.write(keep, [f"clip{r:04d}" for r in keep], f, pos, meta=meta)
                written.extend(keep)
            for r in written[: min(canaries, len(written))]:
                w.write([r], [f"clip{r:04d}"], feats_for([r], N, D), None, canary=True, meta=vmeta([r]))
            w.close(
                timing={"wall_s": 1.0, "clips_per_s": 10.0},
                env=env or {"host": "test", "decode_workers": 1},
                canary_stats={"max_abs_diff": 0.0, "max_rel_l2": 0.0},
                side_effects={"writes_outside_run_dir": []},
            )
    return split_dir_for(out_dir, split)


# ----------------------------------------------------------------------------- plan_shards


@pytest.mark.parametrize("n_rows", [0, 1, 7, 8, 9, 31, 32, 33, 100, 20000])
@pytest.mark.parametrize("batch_size", [1, 3, 8, 32])
@pytest.mark.parametrize("n_shards", [1, 2, 3, 7, 32, 50])
def test_plan_shards_grid_property(n_rows, batch_size, n_shards):
    ranges = plan_shards(n_rows, batch_size, n_shards)
    assert len(ranges) == n_shards
    cursor = 0
    nb = n_batches(n_rows, batch_size)
    for lo, hi in ranges:
        assert lo == cursor  # ascending, disjoint, no gaps
        assert hi >= lo
        assert lo % batch_size == 0
        if hi != n_rows:
            assert hi % batch_size == 0  # only the final boundary may be short
        cursor = hi
    assert cursor == n_rows  # tiles [0, n_rows)
    assert ranges == plan_shards(n_rows, batch_size, n_shards)  # deterministic
    n_empty = sum(1 for lo, hi in ranges if lo == hi)
    assert n_empty == max(0, n_shards - nb)
    # batches dealt as evenly as possible: every non-empty shard has floor/ceil(nb / n_shards) batches
    counts = [n_batches(hi - lo, batch_size) for lo, hi in ranges]
    assert sum(counts) == nb
    assert max(counts) - min(counts) <= 1


def test_plan_shards_documented_examples():
    ranges = plan_shards(20000, 32, 32)
    assert ranges[3] == (1856, 2496)  # docs/features.md example
    assert ranges[0] == (0, 608)  # docs/features.md example
    assert ranges[-1][1] == 20000


def test_plan_shards_short_final_batch_in_last_non_empty_shard():
    ranges = plan_shards(9, 4, 5)  # 3 batches, 5 shards -> 2 empty
    assert ranges == [(0, 0), (0, 4), (4, 4), (4, 8), (8, 9)]
    ranges = plan_shards(9, 4, 2)
    assert ranges == [(0, 4), (4, 9)]


def test_plan_shards_rejects_bad_args():
    with pytest.raises(ValueError):
        plan_shards(10, 0, 1)
    with pytest.raises(ValueError):
        plan_shards(10, 4, 0)
    with pytest.raises(ValueError):
        plan_shards(-1, 4, 1)


def test_shard_of_row():
    ranges = plan_shards(9, 4, 5)
    assert [shard_of_row(r, ranges) for r in range(9)] == [1, 1, 1, 1, 3, 3, 3, 3, 4]
    with pytest.raises(IndexError):
        shard_of_row(9, ranges)
    with pytest.raises(IndexError):
        shard_of_row(-1, ranges)


# ----------------------------------------------------------------------------- format helpers


def test_member_names_and_parse():
    assert fmt.member_basename(7) == "000000007"
    assert fmt.canary_basename(7) == "c000000007"
    assert fmt.parse_member_name("000000007.feat.npy") == (7, "feat.npy", False)
    assert fmt.parse_member_name("c000000007.json") == (7, "json", True)
    with pytest.raises(FormatError):
        fmt.parse_member_name("_pos.npy")
    assert fmt.shard_tar_name(3) == "shard-000003.tar"
    assert fmt.parse_shard_name("shard-000003.json") == (3, "json")
    with pytest.raises(FormatError):
        fmt.parse_shard_name("shard-000003.json.tmp")


def test_npy_bytes_is_v1_header_and_memmappable():
    arr = np.arange(12, dtype=np.float16).reshape(3, 4)
    payload, header_len = fmt.npy_bytes(arr)
    assert payload[:8] == b"\x93NUMPY\x01\x00"  # v1.0
    assert header_len + arr.nbytes == len(payload)
    back = np.frombuffer(payload, dtype=np.float16, offset=header_len).reshape(3, 4)
    assert np.array_equal(back, arr)
    assert np.array_equal(np.load(io.BytesIO(payload)), arr)


# ----------------------------------------------------------------------------- sidecar


def test_write_json_atomic_leaves_no_tmp(tmp_path):
    p = write_json_atomic(tmp_path / "x" / "a.json", {"b": (1, 2), "f": float("nan"), "n": np.int64(3)})
    assert p.exists() and not (tmp_path / "x" / "a.json.tmp").exists()
    assert json.loads(p.read_text()) == {"b": [1, 2], "f": None, "n": 3}


def test_build_sidecar_requires_every_key():
    with pytest.raises(ValueError, match="missing blocks"):
        build_sidecar(format_version=1)
    sc = build_sidecar(**{k: None for k in SIDECAR_KEYS}, extra=1)
    assert list(sc)[: len(SIDECAR_KEYS)] == list(SIDECAR_KEYS)


def test_read_sidecar_refuses_newer_format(tmp_path):
    p = write_json_atomic(tmp_path / "shard-000000.json", {"format_version": fmt.FEATURE_FORMAT_VERSION + 1})
    with pytest.raises(FormatError, match="E_FORMAT"):
        read_sidecar(p)
    p2 = write_json_atomic(tmp_path / "shard-000001.json", {"nope": 1})
    with pytest.raises(FormatError):
        read_sidecar(p2)


def test_sidecar_matches_plan_keys(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=8, n_shards=1)
    sc = read_sidecar(split_dir / "shard-000000.json")
    plan = dict(
        manifest_hash=FakeBench.manifest_hash,
        input_hash=FakeBench.input_hash,
        model=sc["model"],
        reading=sc["reading"],
        describe_sha256=sc["bridge"]["describe_sha256"],
        n=1,
        dtype="float16",
        pos_mode="none",
    )
    assert sidecar_matches_plan(sc, **plan)
    for key, bad in [("manifest_hash", "x"), ("input_hash", "x"), ("model", "x"), ("reading", "a"), ("describe_sha256", "d" * 64),
                     ("n", 2), ("dtype", "float32"), ("pos_mode", "constant"), ("rows", (0, 4))]:
        assert not sidecar_matches_plan(sc, **{**plan, key: bad}), key
    # the layout is frozen per split: n / rows are only a layout check
    assert sidecar_matches_plan(sc, **{k: v for k, v in plan.items() if k != "n"})
    assert sidecar_matches_plan(sc, **plan, rows=(0, 8))
    # model, reading and the configuration digest; the benchmark side is the INPUT hash; a scoring edit leaves finished shards alone
    assert RESUME_MATCH_KEYS == ("benchmark.manifest_hash", "benchmark.input_hash", "model", "reading", "bridge.describe_sha256",
                                 "features.dtype", "features.pos_mode")
    sc_scoring = {**sc, "benchmark": {**sc["benchmark"], "protocol_hash": "q" * 64}}
    assert sidecar_matches_plan(sc_scoring, **plan)


def test_not_applicable_record(tmp_path):
    d = tmp_path / "features" / "train"
    p = write_not_applicable(d, reason="audio-only bridge", benchmark="k400", split="train", model="beats-iter3", reading="a", describe={"a": 1})
    rec = read_not_applicable(d)
    assert p.name == "NOT_APPLICABLE.json"
    assert rec["status"] == "not_applicable" and rec["reason"] == "audio-only bridge"
    assert set(rec) == {"status", "reason", "benchmark", "split", "model", "reading", "describe_sha256", "bridge", "created"}
    assert (rec["model"], rec["reading"]) == ("beats-iter3", "a") and rec["describe_sha256"] == hashing.describe_sha256({"a": 1})
    r = FeatureReader(d, require_meta=False)
    assert r.not_applicable == rec and r.n_rows == 0 and r.n_present == 0


# ----------------------------------------------------------------------------- writer


def test_tar_is_deterministic_and_stock_readable(tmp_path):
    a = write_split(tmp_path / "a", n_rows=8, n_shards=1)
    b = write_split(tmp_path / "b", n_rows=8, n_shards=1)
    ta, tb = (a / "shard-000000.tar").read_bytes(), (b / "shard-000000.tar").read_bytes()
    assert ta == tb  # mtime 0, uid/gid 0, same content -> identical bytes
    sc = read_sidecar(a / "shard-000000.json")
    assert sc["tar"]["sha256"] == hashlib.sha256(ta).hexdigest()
    assert sc["tar"]["bytes"] == len(ta)
    with tarfile.open(a / "shard-000000.tar") as tar:
        names = tar.getnames()
        for ti in tar.getmembers():
            assert (ti.mtime, ti.uid, ti.gid, ti.mode) == (0, 0, 0, 0o644)
            assert ti.uname == "" and ti.gname == ""
    assert names == [f"{r:09d}.{ext}" for r in range(8) for ext in ("json", "feat.npy")]
    # WebDataset grouping key = everything before the first dot
    assert {n.split(".")[0] for n in names} == {f"{r:09d}" for r in range(8)}


def test_writer_tar_bytes_identical_with_large_copybuf(tmp_path, monkeypatch):
    """The 8 MiB file buffer and tar copy buffer change how the bytes reach the file, never which bytes: a member
    larger than the buffer (copied in several chunks) gives the same tar as stdlib-default buffers."""
    from yavih.io import writer as writer_mod

    D = (9 << 20) // 2 + 3  # fp16: one member is 9 MiB + 6 B, more than one copy buffer
    big = write_split(tmp_path / "big", n_rows=2, n_shards=1, N=1, D=D)
    monkeypatch.setattr(writer_mod, "WRITE_BUFFER_BYTES", io.DEFAULT_BUFFER_SIZE)
    small = write_split(tmp_path / "small", n_rows=2, n_shards=1, N=1, D=D)
    ta, tb = (big / "shard-000000.tar").read_bytes(), (small / "shard-000000.tar").read_bytes()
    assert len(ta) > writer_mod.WRITE_BUFFER_BYTES
    assert hashlib.sha256(ta).hexdigest() == hashlib.sha256(tb).hexdigest()
    sa, sb = read_sidecar(big / "shard-000000.json"), read_sidecar(small / "shard-000000.json")
    assert sa["tar"] == sb["tar"] and sa["members"] == sb["members"]


def test_sidecar_fields_present_and_offsets_memmappable(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=10, n_shards=2, missing={3}, canaries=2, bench=FakeBench(data_status_ok=True))
    sc = read_sidecar(split_dir / "shard-000001.json")
    assert list(sc)[: len(SIDECAR_KEYS)] == list(SIDECAR_KEYS)
    assert sc["format_version"] == fmt.FEATURE_FORMAT_VERSION
    assert sc["benchmark"] == {
        "name": "synth_clf",
        "protocol_version": 1,
        "manifest_hash": "m" * 64,
        "input_hash": "i" * 64,
        "input_components": None,  # this fake has no ProtocolHashes
        "protocol_hash": "p" * 64,
        "scoring_hash": None,
        "folder_source": None,  # this fake has no probe.py / reader.py digests
        "media_fingerprint": "f" * 64,
        "media_info_sha256": None,  # copied from data_status.json (this fake has no census)
        "data_roots": {"media_root": "/data/synth"},
    }
    assert sc["shard"] == {"index": 1, "n": 2, "rows": [4, 10], "batch_size": 4}
    assert sc["rows"] == {"n_range": 6, "n_written": 6, "n_missing": 0, "n_canary": 2}
    assert sc["features"] == {"dtype": "float16", "N": 6, "D": 16, "pos_mode": "none", "pos_sha256": None}
    describe = make_describe(FakeBridge())
    assert sc["bridge"]["describe"] == describe
    assert sc["bridge"]["describe_sha256"] == hashing.describe_sha256(describe)
    assert sc["bridge"]["name"] == "fake-bridge"
    assert sc["bridge"]["source_sha256"] == "s" * 64
    assert sc["bridge"]["args"] == {"layer": "12"}
    # the declared names, copied from describe()
    assert (sc["model"], sc["reading"]) == ("fake-model", "v")
    assert "decoders" not in sc and sc["side_effects"] == {"writes_outside_run_dir": []}
    assert sc["env"]["decode_workers"] == 1
    assert sc["timing"]["wall_s"] == 1.0 and sc["timing"]["started"].endswith("Z") and "finished" in sc["timing"]
    assert sc["canary"] == {"max_abs_diff": 0.0, "max_rel_l2": 0.0}
    assert [m["row"] for m in sc["members"]] == [4, 5, 6, 7, 8, 9]
    assert set(sc["members"][0]) == {"row", "clip_id", "media_key", "offset_data", "header_len", "nbytes", "duration_s"}
    assert sc["members"][0]["duration_s"] == {"video": 10.0}  # media.duration_s per stream
    sc0 = read_sidecar(split_dir / "shard-000000.json")
    assert sc0["missing"] == [{"row": 3, "clip_id": "clip0003", "reason": "MediaMissing: video"}]
    assert sc0["rows"] == {"n_range": 4, "n_written": 3, "n_missing": 1, "n_canary": 2}
    # offset_data + header_len addresses the array in place
    raw = (split_dir / "shard-000000.tar").read_bytes()
    for m in sc0["members"]:
        off = m["offset_data"] + m["header_len"]
        arr = np.frombuffer(raw, dtype=np.float16, count=6 * 16, offset=off).reshape(6, 16)
        assert np.array_equal(arr, feats_for([m["row"]], 6, 16)[0].astype(np.float16))
        assert raw[m["offset_data"] : m["offset_data"] + 8] == b"\x93NUMPY\x01\x00"
        assert m["nbytes"] == 6 * 16 * 2  # array bytes, header excluded


def test_row_json_member_content(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=4, n_shards=1, canaries=1)
    with tarfile.open(split_dir / "shard-000000.tar") as tar:
        rec = json.loads(tar.extractfile("000000002.json").read())
        assert rec == {
            "row": 2,
            "clip_id": "clip0002",
            "media_key": "mk2",
            "media": {"video": VIDEO_INFO},
            "used": {"video": {"times": [2.5, 7.5]}},
        }
        names = tar.getnames()
        assert names[-2:] == ["c000000000.json", "c000000000.feat.npy"]  # canaries after all regular members
        crec = json.loads(tar.extractfile("c000000000.json").read())
        assert crec["canary_of"] == 0 and crec["row"] == 0


def test_empty_shard_has_sidecar_and_empty_tar(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=4, n_shards=3)  # 1 batch, 3 shards
    for i, (lo, hi) in enumerate(plan_shards(4, 4, 3)):
        sc = read_sidecar(split_dir / f"shard-{i:06d}.json")
        assert sc["shard"]["rows"] == [lo, hi]
        with tarfile.open(split_dir / f"shard-{i:06d}.tar") as tar:
            assert (len(tar.getnames()) == 0) == (lo == hi)
        if lo == hi:
            assert sc["members"] == [] and sc["features"]["N"] is None and sc["rows"]["n_written"] == 0
    r = FeatureReader(split_dir, require_meta=False)
    assert r.n_rows == 4 and r.n_present == 4 and r.N == 6


def _writer(tmp_path, bridge=None, rows=(0, 8), shard=(0, 1), bench=None):
    bridge = bridge or FakeBridge()
    return FeatureWriter(
        tmp_path, "train", shard=shard, rows=rows, bench=bench or FakeBench(), bridge=bridge, describe=make_describe(bridge), source_sha256="s" * 64
    )


def test_writer_shape_error_after_first_call(tmp_path):
    w = _writer(tmp_path)
    w.write([0, 1], ["a", "b"], np.zeros((2, 6, 16), np.float32), meta=vmeta([0, 1]))
    with pytest.raises(FeatureShapeError) as ei:
        w.write([2], ["c"], np.zeros((1, 7, 16), np.float32), meta=vmeta([2]))
    assert ei.value.clip_id == "c" and ei.value.got == (7, 16) and ei.value.expected == (6, 16)
    with pytest.raises(FeatureShapeError):
        w.write([2], ["c"], np.zeros((1, 16), np.float32), meta=vmeta([2]))  # not [B, N, D]
    w.abort()
    assert not list(tmp_path.rglob("*.tmp"))


def test_writer_value_errors_and_fp16_saturation(tmp_path):
    w = _writer(tmp_path)
    x = np.zeros((1, 6, 16), np.float32)
    x[0, 0, 0] = np.nan
    with pytest.raises(FeatureValueError, match="non-finite"):
        w.write([0], ["a"], x, meta=vmeta([0]))
    x[0, 0, 0] = 2e4  # representable in fp16 (< 65504) but above the 1e4 guard
    with pytest.raises(FeatureValueError, match="float32") as ei:
        w.write([0], ["a"], x, meta=vmeta([0]))
    assert ei.value.clip_id == "a"
    x[0, 0, 0] = 1e5  # would be inf in fp16
    with pytest.raises(FeatureValueError):
        w.write([0], ["a"], x, meta=vmeta([0]))
    x[0, 0, 0] = 9e3
    w.write([0], ["a"], x, meta=vmeta([0]))  # fine
    w.abort()
    # float32 declared: 2e4 is fine, inf is not
    w32 = _writer(tmp_path, bridge=FakeBridge(dtype="float32"))
    x[0, 0, 0] = 2e4
    w32.write([0], ["a"], x, meta=vmeta([0]))
    x[0, 0, 0] = np.inf
    with pytest.raises(FeatureValueError):
        w32.write([1], ["b"], x, meta=vmeta([1]))
    w32.abort()


def test_writer_casts_to_declared_dtype_and_accepts_tensor_like(tmp_path):
    class TensorLike:
        def __init__(self, a):
            self.a = a

        def detach(self):
            return self

        def float(self):
            return TensorLike(self.a.astype(np.float32))

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    with _writer(tmp_path, rows=(0, 2)) as w:
        w.write([0], ["a"], TensorLike(np.full((1, 6, 16), 0.1, np.float64)), meta=vmeta([0]))
        w.write([1], ["b"], np.full((1, 6, 16), 0.2, np.float64), meta=vmeta([1]))
        w.close(timing={}, env={}, canary_stats={}, side_effects={})
    r = FeatureReader(split_dir_for(tmp_path, "train"), require_meta=False)
    assert r.dtype == np.float16
    assert np.array_equal(r.rows([0])[0], np.full((6, 16), np.float16(0.1)))


def test_writer_length_assertion_and_row_discipline(tmp_path):
    w = _writer(tmp_path)
    with pytest.raises(AssertionError):
        w.write([0, 1], ["a"], np.zeros((2, 6, 16), np.float32), meta=vmeta([0, 1]))
    with pytest.raises(AssertionError):
        w.write([0], ["a"], np.zeros((2, 6, 16), np.float32), meta=vmeta([0]))
    with pytest.raises(ValueError, match="outside the shard range"):
        w.write([8], ["x"], np.zeros((1, 6, 16), np.float32), meta=vmeta([8]))
    w.write([0, 1], ["a", "b"], np.zeros((2, 6, 16), np.float32), meta=vmeta([0, 1]))
    with pytest.raises(ValueError, match="already written"):
        w.write([1], ["b"], np.zeros((1, 6, 16), np.float32), meta=vmeta([1]))
    with pytest.raises(ValueError, match="ascending"):
        w.write([3, 2], ["d", "c"], np.zeros((2, 6, 16), np.float32), meta=vmeta([3, 2]))
    with pytest.raises(ValueError, match="already written"):
        w.record_missing(0, "a", "x")
    with pytest.raises(ValueError, match="not written"):
        w.write([5], ["f"], np.zeros((1, 6, 16), np.float32), canary=True, meta=vmeta([5]))
    w.write([1], ["b"], np.zeros((1, 6, 16), np.float32), canary=True, meta=vmeta([1]))
    with pytest.raises(ValueError, match="after canary"):
        w.write([2], ["c"], np.zeros((1, 6, 16), np.float32), meta=vmeta([2]))
    w.abort()


def test_close_refuses_unaccounted_rows_and_removes_tmp(tmp_path):
    w = _writer(tmp_path, rows=(0, 8))
    w.write([0, 1], ["a", "b"], np.zeros((2, 6, 16), np.float32), meta=vmeta([0, 1]))
    assert w.tmp_path.exists() and w.tmp_path.name.startswith("shard-000000.tar.") and w.tmp_path.name.endswith(".tmp")
    with pytest.raises(ValueError, match="E_ROWS"):
        w.close(timing={}, env={}, canary_stats={}, side_effects={})
    assert sorted(p.name for p in split_dir_for(tmp_path, "train").iterdir()) == []


def test_abort_on_exception_in_context_manager(tmp_path):
    split_dir = split_dir_for(tmp_path, "train")
    with pytest.raises(RuntimeError):
        with _writer(tmp_path) as w:
            w.write([0], ["a"], np.zeros((1, 6, 16), np.float32), meta=vmeta([0]))
            assert w.tmp_path.exists() and w.tmp_path.parent == split_dir  # the per-writer tmp
            raise RuntimeError("bridge crashed")
    assert list(split_dir.iterdir()) == []
    # leaving the block without close() is also an abort: no half-written shard survives
    with _writer(tmp_path) as w:
        w.write([0], ["a"], np.zeros((1, 6, 16), np.float32), meta=vmeta([0]))
    assert list(split_dir.iterdir()) == []


def test_pos_modes(tmp_path):
    # none: pos must be None
    w = _writer(tmp_path / "n")
    with pytest.raises(ValueError, match="pos_mode is 'none'"):
        w.write([0], ["a"], np.zeros((1, 6, 16), np.float32), np.zeros((6, 4), np.float32), meta=vmeta([0]))
    w.abort()
    # constant: required on the first call, _pos.npy first member, identical afterwards
    br = FakeBridge(pos_mode="constant")
    w = _writer(tmp_path / "c", bridge=br, rows=(0, 4))
    with pytest.raises(ValueError, match="constant"):
        w.write([0], ["a"], np.zeros((1, 6, 16), np.float32), None, meta=vmeta([0]))
    pos = np.random.default_rng(1).random((6, 4)).astype(np.float32)
    w.write([0], ["a"], np.zeros((1, 6, 16), np.float32), pos, meta=vmeta([0]))
    w.write([1], ["b"], np.zeros((1, 6, 16), np.float32), None, meta=vmeta([1]))  # omitted after the first call: fine
    w.write([2], ["c"], np.zeros((1, 6, 16), np.float32), np.stack([pos]), meta=vmeta([2]))  # broadcast [B, N, 4] form
    with pytest.raises(ValueError, match="differs"):
        w.write([3], ["d"], np.zeros((1, 6, 16), np.float32), pos + 1, meta=vmeta([3]))
    w.write([3], ["d"], np.zeros((1, 6, 16), np.float32), pos, meta=vmeta([3]))
    side = w.close(timing={}, env={}, canary_stats={}, side_effects={})
    sc = read_sidecar(side)
    payload, _ = fmt.npy_bytes(pos)
    assert sc["features"]["pos_sha256"] == hashlib.sha256(payload).hexdigest() == fmt.pos_sha256(payload)
    with tarfile.open(side.with_suffix(".tar")) as tar:
        assert tar.getnames()[0] == "_pos.npy"
    r = FeatureReader(side.parent, require_meta=False)
    assert np.array_equal(r.pos(), pos) and np.array_equal(r.pos(2), pos)
    for _, _, p in r.iter(with_pos=True):
        assert np.array_equal(p, pos)
    # per_sample: [B, N, 4] required, .pos.npy member per row, readable via pos(row) and iter(with_pos)
    br = FakeBridge(pos_mode="per_sample")
    w = _writer(tmp_path / "p", bridge=br, rows=(0, 4))
    with pytest.raises(ValueError, match="per_sample"):
        w.write([0], ["a"], np.zeros((1, 6, 16), np.float32), None, meta=vmeta([0]))
    with pytest.raises(ValueError, match="per_sample"):
        w.write([0], ["a"], np.zeros((1, 6, 16), np.float32), np.zeros((6, 4), np.float32), meta=vmeta([0]))
    w.abort()
    split_dir = write_split(tmp_path / "p2", n_rows=8, n_shards=2, bridge=br)
    r = FeatureReader(split_dir, require_meta=False)
    allpos = r.pos()
    assert allpos.shape == (8, 6, 4) and allpos.dtype == np.float32
    for row in range(8):
        assert np.array_equal(r.pos(row), np.full((6, 4), row / 8, np.float32))
    streamed = {row: p for row, _, p in r.iter(with_pos=True)}
    assert all(np.array_equal(streamed[row], allpos[i]) for i, row in enumerate(r.present_rows))
    with tarfile.open(split_dir / "shard-000000.tar") as tar:
        assert "000000000.pos.npy" in tar.getnames()
    assert read_sidecar(split_dir / "shard-000000.json")["features"]["pos_sha256"] is None


def test_null_weights_are_recorded_and_change_nothing_but_describe_sha256(tmp_path):
    """A null weights_sha256 is recorded as is."""
    br = FakeBridge()
    d = make_describe(br, weights=None)
    w = FeatureWriter(tmp_path, "train", shard=(0, 1), rows=(0, 1), bench=FakeBench(), bridge=br, describe=d, source_sha256="s" * 64)
    w.write([0], ["a"], np.zeros((1, 2, 2), np.float32), meta=vmeta([0]))
    sc = read_sidecar(w.close(timing={}, env={}, canary_stats={}, side_effects={}))
    assert (sc["model"], sc["reading"]) == ("fake-model", "v") and sc["bridge"]["describe"]["weights_sha256"] is None
    assert "~" not in sc["model"]


@pytest.mark.parametrize("drop", ["model", "reading"])
def test_writer_refuses_a_describe_without_model_or_reading(tmp_path, drop):
    d = {k: v for k, v in make_describe(FakeBridge()).items() if k != drop}
    with pytest.raises(ValueError, match="model"):
        FeatureWriter(tmp_path, "train", shard=(0, 1), rows=(0, 1), bench=FakeBench(), bridge=FakeBridge(), describe=d, source_sha256="s" * 64)
    assert not (tmp_path / "features" / "train").exists() or not list((tmp_path / "features" / "train").iterdir())


def test_bridge_label_defaults_to_the_class_name(tmp_path):
    br = FakeBridge(name="")
    w = FeatureWriter(tmp_path, "train", shard=(0, 1), rows=(0, 1), bench=FakeBench(), bridge=br, describe=make_describe(br), source_sha256="s" * 64)
    w.write([0], ["a"], np.zeros((1, 2, 2), np.float32), meta=vmeta([0]))
    sc = read_sidecar(w.close(timing={}, env={}, canary_stats={}, side_effects={}))
    assert sc["bridge"]["name"] == "FakeBridge"


def test_split_dir_for():
    assert split_dir_for("/runs/k400/m@1", "train") == Path("/runs/k400/m@1/features/train")
    assert split_dir_for("/runs/k400/m@1/features/train", "train") == Path("/runs/k400/m@1/features/train")


def test_writer_rejects_bad_dtype_and_shard(tmp_path):
    with pytest.raises(ValueError):
        _writer(tmp_path, bridge=FakeBridge(dtype="bfloat16"))
    with pytest.raises(ValueError):
        _writer(tmp_path, shard=(2, 2))
    with pytest.raises(ValueError):
        _writer(tmp_path, rows=(5, 4))


# ----------------------------------------------------------------------------- reader


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_round_trip_bit_identical(tmp_path, dtype):
    split_dir = write_split(tmp_path / "run", n_rows=23, n_shards=4, bridge=FakeBridge(dtype=dtype), missing={5, 17})
    r = FeatureReader(split_dir, require_meta=False)
    assert r.dtype == np.dtype(dtype) and (r.N, r.D) == (6, 16) and r.n_rows == 23
    assert r.present_rows.tolist() == [i for i in range(23) if i not in (5, 17)]
    assert r.present_rows.dtype == np.int64
    assert r.clip_ids() == [f"clip{i:04d}" for i in r.present_rows]
    expected = feats_for(r.present_rows.tolist(), 6, 16).astype(dtype)
    got = r.rows(r.present_rows)
    assert got.dtype == np.dtype(dtype) and got.shape == (21, 6, 16)
    assert np.array_equal(got, expected)  # bit-identical after the declared cast
    assert got.tobytes() == expected.tobytes()
    assert [m["row"] for m in r.missing] == [5, 17]


def test_memmap_rows_equals_streamed_iter(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=37, n_shards=5, missing={0, 36}, canaries=3)
    r = FeatureReader(split_dir, require_meta=False)
    streamed = list(r.iter())
    assert [row for row, _, _ in streamed] == r.present_rows.tolist()
    assert all(p is None for _, _, p in streamed)
    stacked = np.stack([f for _, f, _ in streamed])
    assert np.array_equal(r.rows(r.present_rows), stacked)
    assert stacked.dtype == r.dtype
    # arbitrary order / repeated gather by manifest row
    idx = [20, 3, 3, 35, 1]
    g = r.rows(idx)
    for k, row in enumerate(idx):
        assert np.array_equal(g[k], stacked[r.present_rows.tolist().index(row)])
    with pytest.raises(KeyError):
        r.rows([0])  # missing row
    with pytest.raises(KeyError):
        r.rows([999])
    assert r.rows([]).shape == (0, 6, 16)


def test_shard_order_independence(tmp_path):
    rows = 40
    a = FeatureReader(write_split(tmp_path / "a", n_rows=rows, n_shards=1), require_meta=False)
    b = FeatureReader(write_split(tmp_path / "b", n_rows=rows, n_shards=4), require_meta=False)
    c = FeatureReader(write_split(tmp_path / "c", n_rows=rows, n_shards=13), require_meta=False)
    ra, rb, rc = (x.rows(x.present_rows) for x in (a, b, c))
    assert np.array_equal(ra, rb) and np.array_equal(ra, rc)
    assert a.clip_ids() == b.clip_ids() == c.clip_ids()
    assert np.array_equal(np.stack([f for _, f, _ in c.iter()]), ra)


def test_canaries_pairs(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=16, n_shards=2, canaries=2)
    r = FeatureReader(split_dir, require_meta=False)
    pairs = list(r.canaries())
    assert [row for row, _, _ in pairs] == [0, 1, 8, 9]
    for row, c, o in pairs:
        assert np.array_equal(c, o) and np.array_equal(o, r.rows([row])[0])
    sc = read_sidecar(split_dir / "shard-000000.json")
    assert sc["rows"]["n_canary"] == 2 and len(sc["canary_members"]) == 2
    assert len(sc["members"]) == 8  # canaries are not regular members


def test_matrix_backends_and_rowview(tmp_path, monkeypatch):
    split_dir = write_split(tmp_path / "run", n_rows=12, n_shards=3, missing={4})
    r = FeatureReader(split_dir, require_meta=False)
    full = r.rows(r.present_rows)
    ram = r.matrix(ram_budget_bytes=r.nbytes)
    mm = r.matrix(ram_budget_bytes=r.nbytes - 1)
    assert isinstance(ram, RowView) and ram.backend == "ram" and mm.backend == "memmap"
    for v in (ram, mm):
        assert len(v) == 11 and v.shape == (11, 6, 16) and v.dtype == np.float16
        assert np.array_equal(v[0], full[0]) and np.array_equal(v[-1], full[-1])
        assert np.array_equal(v[np.array([2, 0, 7])], full[[2, 0, 7]])
        assert np.array_equal(v[[1, 1]], full[[1, 1]])
        assert v[3].dtype == np.float16  # stored fp16 stays fp16 (upcast is the probe's job)
        assert np.array_equal(np.asarray(v), full)
        with pytest.raises(IndexError):
            v[11]
        assert v.present_rows.tolist() == r.present_rows.tolist()
    monkeypatch.setenv("YAVIH_RAM_BUDGET_GB", "0")
    assert r.matrix().backend == "memmap"
    monkeypatch.setenv("YAVIH_RAM_BUDGET_GB", "1")
    assert r.matrix().backend == "ram"


def test_matrix_is_cached_per_reader(tmp_path, monkeypatch):
    """One RAM array per reader whatever the number of matrix() calls (arms, folds and seeds share the reader);
    the memmap backend drops it; values never change; gathers are copies of the read-only cache."""
    split_dir = write_split(tmp_path / "run", n_rows=12, n_shards=3, missing={4})
    r = FeatureReader(split_dir, require_meta=False)
    calls = []
    real_rows = r.rows
    monkeypatch.setattr(r, "rows", lambda idx: calls.append(len(idx)) or real_rows(idx))
    a, b = r.matrix(ram_budget_bytes=r.nbytes), r.matrix(ram_budget_bytes=r.nbytes)
    assert calls == [11] and a._array is b._array and a.backend == b.backend == "ram"
    assert not a._array.flags.writeable
    g = a[np.array([0, 3])]
    assert g.flags.writeable and not np.shares_memory(g, a._array)
    full = real_rows(r.present_rows)
    assert np.array_equal(np.asarray(b), full)
    mm = r.matrix(ram_budget_bytes=r.nbytes - 1)
    assert mm.backend == "memmap" and r._matrix is None
    assert np.array_equal(np.asarray(mm), full)
    c = r.matrix(ram_budget_bytes=r.nbytes)
    assert c.backend == "ram" and np.array_equal(np.asarray(c), full) and c._array is not a._array
    r.close()
    assert r._matrix is None


def test_ram_budget_rule():
    gib = 1024**3
    assert ram_budget_bytes({}) == DEFAULT_RAM_BUDGET_BYTES == 16 * gib
    assert ram_budget_bytes({"SLURM_MEM_PER_NODE": "16000"}) == int(0.45 * 16000 * 1024**2)  # 7.03 GiB
    assert ram_budget_bytes({"SLURM_MEM_PER_NODE": "64000"}) == 16 * gib  # 0.45 * 62.5 GiB = 28 GiB -> capped
    assert ram_budget_bytes({"SLURM_MEM_PER_NODE": "64000", "YAVIH_RAM_BUDGET_GB": "2.5"}) == int(2.5 * gib)
    assert ram_budget_bytes({"SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_ON_NODE": "8"}) == int(0.45 * 16000 * 1024**2)
    with pytest.raises(ValueError):
        ram_budget_bytes({"YAVIH_RAM_BUDGET_GB": "lots"})


def test_reader_requires_meta_by_default_and_reads_it(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=8, n_shards=2)
    with pytest.raises(FormatError, match="_meta.json"):
        FeatureReader(split_dir)
    write_json_atomic(split_dir / "_meta.json", {"format_version": fmt.FEATURE_FORMAT_VERSION, "split": "train", "features": {"dtype": "float16", "N": 6, "D": 16, "pos_mode": "none"}, "coverage": {"nominal": 8}})
    r = FeatureReader(split_dir)
    assert r.meta["coverage"]["nominal"] == 8 and r.n_rows == 8 and r.split == "train"
    write_json_atomic(split_dir / "_meta.json", {"format_version": 99})
    with pytest.raises(FormatError):
        FeatureReader(split_dir)
    with pytest.raises(FileNotFoundError):
        FeatureReader(tmp_path / "nope")


def test_reader_refuses_inconsistent_sidecars(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=8, n_shards=2)
    sc = read_sidecar(split_dir / "shard-000001.json")
    sc["shard"]["n"] = 3
    write_json_atomic(split_dir / "shard-000001.json", sc)
    with pytest.raises(FormatError, match="disagree on n"):
        FeatureReader(split_dir, require_meta=False)
    sc["shard"]["n"] = 2
    sc["features"]["dtype"] = "float32"
    write_json_atomic(split_dir / "shard-000001.json", sc)
    with pytest.raises(FormatError, match="dtype"):
        FeatureReader(split_dir, require_meta=False)
    sc["features"]["dtype"] = "float16"
    sc["format_version"] = 7
    write_json_atomic(split_dir / "shard-000001.json", sc)
    with pytest.raises(FormatError, match="E_FORMAT"):
        FeatureReader(split_dir, require_meta=False)


def test_reader_refuses_a_meta_of_another_configuration(tmp_path):
    """A re-extraction under another describe_sha256 (or model / reading) leaves the old _meta.json
    behind; the reader compares the configuration and asks for a re-validation."""
    split_dir = write_split(tmp_path / "run", n_rows=8, n_shards=1)
    sc = read_sidecar(split_dir / "shard-000000.json")
    good = {"format_version": fmt.FEATURE_FORMAT_VERSION, "split": "train", "model": sc["model"], "reading": sc["reading"],
            "bridge": {"describe_sha256": sc["bridge"]["describe_sha256"]},
            "features": {"dtype": "float16", "N": 6, "D": 16, "pos_mode": "none"}, "coverage": {"nominal": 8}}
    write_json_atomic(split_dir / "_meta.json", good)
    assert FeatureReader(split_dir).n_rows == 8
    for patch in ({"model": "other"}, {"reading": "a"}, {"bridge": {"describe_sha256": "0" * 64}}):
        write_json_atomic(split_dir / "_meta.json", {**good, **patch})
        with pytest.raises(FormatError, match="stale meta"):
            FeatureReader(split_dir)


def test_fingerprint_matches_hashing(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=10, n_shards=3, missing={2})
    r = FeatureReader(split_dir, require_meta=False)
    scs = [read_sidecar(p) for p in sorted(split_dir.glob("shard-*.json"))]
    expect = hashing.features_fingerprint(
        format_version=fmt.FEATURE_FORMAT_VERSION,
        dtype="float16",
        N=6,
        D=16,
        pos_mode="none",
        pos_sha256=None,
        shards=[[sc["shard"]["rows"][0], sc["shard"]["rows"][1], sc["tar"]["sha256"]] for sc in scs],
        n_present=9,
    )
    assert r.fingerprint() == expect
    # a different shard layout of the same features is a different fingerprint (tar shas differ)
    r2 = FeatureReader(write_split(tmp_path / "run2", n_rows=10, n_shards=2, missing={2}), require_meta=False)
    assert r2.fingerprint() != expect


def test_reader_close_and_context(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=4, n_shards=1)
    with FeatureReader(split_dir, require_meta=False) as r:
        r.rows([0])
        assert r._maps
    assert not r._maps


# ----------------------------------------------------------------------------- resumption


def test_resume_shard_protocol(tmp_path):
    split_dir = write_split(tmp_path / "run", n_rows=8, n_shards=2)
    sc = read_sidecar(split_dir / "shard-000001.json")
    plan = dict(manifest_hash="m" * 64, input_hash="i" * 64, model=sc["model"], reading=sc["reading"],
                describe_sha256=sc["bridge"]["describe_sha256"], n=2, dtype="float16", pos_mode="none")
    (split_dir / "shard-000001.tar.tmp").write_bytes(b"junk")
    (split_dir / "shard-000000.tar.tmp").write_bytes(b"sibling")
    assert resume_shard(split_dir, 1, **plan) == "skip"
    assert not (split_dir / "shard-000001.tar.tmp").exists()
    assert (split_dir / "shard-000000.tar.tmp").exists()  # a sibling's tmp is not ours to delete
    # mismatch (another configuration under the same model name) -> moved to _stale/<utc>/ and re-run
    assert resume_shard(split_dir, 1, **{**plan, "describe_sha256": "0" * 64}) == "run"
    assert not (split_dir / "shard-000001.json").exists() and not (split_dir / "shard-000001.tar").exists()
    stale = list((split_dir / "_stale").iterdir())
    assert len(stale) == 1 and sorted(p.name for p in stale[0].iterdir()) == ["shard-000001.json", "shard-000001.tar"]
    assert (split_dir / "shard-000000.json").exists()  # other shards untouched
    # nothing there -> run
    assert resume_shard(split_dir, 1, **plan) == "run"
    # tar without sidecar (killed between rename and sidecar write) -> moved aside, run
    (split_dir / "shard-000001.tar").write_bytes(b"orphan")
    assert resume_shard(split_dir, 1, **plan) == "run"
    assert not (split_dir / "shard-000001.tar").exists()
    assert clean_tmp(split_dir, 0) == [split_dir / "shard-000000.tar.tmp"]
    st = move_stale(split_dir, 0, when="T")
    assert st == split_dir / "_stale" / "T" and (st / "shard-000000.tar").exists()
    # the reader ignores _stale and tmps
    r = FeatureReader(split_dir, require_meta=False)
    assert r.sidecars == []


# ----------------------------------------------------------------------------- import boundary


def test_io_core_imports_without_torch():
    code = "import sys; sys.modules['torch'] = None\nimport yavih.io.format, yavih.io.shards, yavih.io.writer, yavih.io.reader, yavih.io.sidecar\nprint('ok')"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_not_applicable_concurrent_writers_all_succeed(tmp_path):
    """A bridge that raises ``NotApplicable`` at run time is discovered by EVERY array task of the
    split, so ``NOT_APPLICABLE.json`` has many concurrent writers; each must publish without tripping over
    another's tmp (per-writer tmp names, idempotent replace) and no tmp may be left behind."""
    from concurrent.futures import ThreadPoolExecutor

    from yavih.io.sidecar import read_not_applicable, write_not_applicable, writer_tmp_name

    split = tmp_path / "features" / "train"

    def one(i: int) -> str:
        return str(write_not_applicable(split, reason=f"writer {i}", benchmark="b", split="train", model="m", reading="v", describe={}))

    with ThreadPoolExecutor(max_workers=16) as ex:
        paths = list(ex.map(one, range(96)))
    assert set(paths) == {str(split / "NOT_APPLICABLE.json")}
    rec = read_not_applicable(split)
    assert rec["status"] == "not_applicable" and rec["model"] == "m" and rec["describe_sha256"] is None and rec["reason"].startswith("writer ")
    assert not list(split.glob("*.tmp"))
    assert writer_tmp_name("x.json").startswith("x.json.") and writer_tmp_name("x.json").endswith(".tmp")


def test_writer_refuses_meta_without_media_or_used(tmp_path):
    """``FeatureMetaError`` when a row's meta lacks ``media`` or a ``used`` entry for a stream of
    ``describe()["input"]``; nothing of the refused call is written."""
    w = _writer(tmp_path)
    x = np.zeros((1, 6, 16), np.float32)
    with pytest.raises(FeatureMetaError, match="media"):
        w.write([0], ["a"], x)
    with pytest.raises(FeatureMetaError, match="media"):
        w.write([0], ["a"], x, meta=[{"media_key": None, "used": {"video": {"times": [1.0]}}}])
    with pytest.raises(FeatureMetaError, match="used"):
        w.write([0], ["a"], x, meta=[{"media": {"video": dict(VIDEO_INFO)}}])
    with pytest.raises(FeatureMetaError, match="video"):
        w.write([0], ["a"], x, meta=[{"media": {"video": dict(VIDEO_INFO)}, "used": {}}])
    w.write([0], ["a"], x, meta=vmeta([0]))  # the refused calls left nothing behind (row 0 still free)
    w.abort()


def test_reader_durations_equal_members_media(tmp_path):
    """``FeatureReader.durations(idx, stream)`` equals the members' ``media.duration_s``; NaN for a
    stream the members do not carry."""
    split_dir = write_split(tmp_path / "run", n_rows=10, n_shards=3, missing={4})
    r = FeatureReader(split_dir, require_meta=False)
    d = r.durations(r.present_rows, "video")
    assert d.shape == (9,) and np.all(d == 10.0)
    assert np.isnan(r.durations([0, 1], "audio")).all()
    with pytest.raises(KeyError):
        r.durations([4], "video")


def test_resume_keeps_a_shard_whose_input_hash_is_superseded(tmp_path):
    """A shard whose recorded ``input_hash`` the folder changelog lists under ``supersedes_input`` (a bridge handed
    exactly the same thing) is "already done"; without that entry it is moved aside."""
    from yavih import hash_history

    split_dir = write_split(tmp_path / "run", n_rows=8, n_shards=2)
    side = split_dir / "shard-000001.json"
    sc = read_sidecar(side)
    old = "e" * 64
    sc["benchmark"]["input_hash"] = old
    write_json_atomic(side, sc)
    plan = dict(manifest_hash="m" * 64, input_hash="b" * 64, model=sc["model"], reading=sc["reading"],
                describe_sha256=sc["bridge"]["describe_sha256"], n=2, dtype="float16", pos_mode="none")
    try:
        hash_history.register("synth_clf", (), [{"version": 1, "protocol_hash": "c" * 64, "input_hash": "b" * 64, "supersedes_input": [old]}])
        assert peek_shard(split_dir, 1, **plan) == "skip"
        assert resume_shard(split_dir, 1, **plan) == "skip" and side.exists()
        hash_history.register("synth_clf", (), [{"version": 1, "protocol_hash": "c" * 64, "input_hash": "b" * 64, "supersedes_input": []}])
        assert resume_shard(split_dir, 1, **plan) == "run" and not side.exists()
    finally:
        hash_history.reset()


def test_a_sampled_layout_is_another_identity(tmp_path):
    """``sample: [lo, hi]`` is part of a layout's identity: a full split's layout (no key, also every layout written
    before samples existed) never serves a sample and the other way round; ``freeze_layout`` refuses the mix with
    ``E_LAYOUT`` naming both, ``frozen_layout`` too; a sampled layout is never inferred from shards."""
    from yavih.io.layout import (
        LayoutError,
        build_layout,
        freeze_layout,
        frozen_layout,
        infer_layout,
        layout_identity,
        layout_matches,
        layout_sample,
        read_layout,
        sample_conflict,
    )

    base = dict(manifest_hash="m" * 64, input_hash="b" * 64, model="m1", reading="v", describe_sha256="d" * 64)
    full, s4, s8 = layout_identity(**base), layout_identity(**base, sample=(0, 4)), layout_identity(**base, sample=[0, 8])
    full_rec = build_layout(full, n=2, batch_size=4, n_rows=16)
    assert "sample" not in full_rec and layout_sample(full_rec) is None and layout_matches(full_rec, full)
    assert not layout_matches(full_rec, s4) and "a --rows 0:4 sample" in sample_conflict(full_rec, s4)
    sd = tmp_path / "features" / "train"
    rec = freeze_layout(sd, s4, n=7, batch_size=4, n_rows=99)  # n / n_rows are the sample's, not the caller's
    assert rec["sample"] == [0, 4] and rec["n"] == 1 and rec["ranges"] == [[0, 4]] and rec["n_rows"] == 4
    assert read_layout(sd) == rec and layout_sample(rec) == (0, 4) and frozen_layout(sd, s4) == rec
    assert sample_conflict(rec, {**s4, "model": "m2"}) is None  # another configuration: moved aside, not a conflict
    for other in (full, s8):
        with pytest.raises(LayoutError) as ei:
            freeze_layout(sd, other, n=2, batch_size=4, n_rows=16)
        assert ei.value.code == "E_LAYOUT" and ei.value.exit_code == 1 and "sample" in str(ei.value)
        with pytest.raises(LayoutError):
            frozen_layout(sd, other, n_rows=16)
    assert infer_layout(sd, s4, n_rows=4) is None


def test_a_layout_belongs_to_its_configuration_through_supersedes_input():
    """``_layout.json`` keys on ``benchmark.input_hash`` (read through ``supersedes_input``) and the describe digest."""
    from yavih import hash_history
    from yavih.io.layout import build_layout, layout_identity, layout_matches

    old_ih, ih = "e" * 64, "b" * 64
    ident = layout_identity(manifest_hash="m" * 64, input_hash=ih, model="m1", reading="v", describe_sha256="d" * 64)
    new = build_layout(ident, n=2, batch_size=4, n_rows=8)
    assert new["benchmark"]["input_hash"] == ih and new["bridge"]["describe_schema"] == "yavih-describe-2"
    assert layout_matches(new, ident) and not layout_matches(new, {**ident, "bridge.describe_sha256": "0" * 64})
    old = {**new, "benchmark": {"manifest_hash": "m" * 64, "input_hash": old_ih}}
    try:
        assert not layout_matches(old, ident)  # no history: another input side
        hash_history.register("synth_clf", (), [{"version": 1, "protocol_hash": "c" * 64, "input_hash": ih, "supersedes_input": [old_ih]}])
        assert layout_matches(old, ident)
        assert not layout_matches(old, {**ident, "model": "m2"})
    finally:
        hash_history.reset()
