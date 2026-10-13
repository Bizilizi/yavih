"""``validate_features``, ``finalize``, ``inspect``/``diff``/``stage`` (docs/validation.md, docs/features.md, docs/cli.md).

The feature directories are built by a minimal writer in this file that follows docs/features.md byte for
byte (member names, ``.npy`` v1.0, deterministic tar headers, the ``media`` + ``used`` member provenance,
the sidecar), so nothing here depends on ``FeatureWriter`` or ``Benchmark.load``. The benchmark is a duck-typed ``FakeBench`` exposing the surface
validate reads (``spec.output_contract.requires``, ``readout_params``, ``protocol``, ``streams``).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from yavih import hashing
from yavih.io.diff import diff_features
from yavih.io.finalize import build_meta, finalize, meta_fingerprint, read_meta
from yavih.io.format import FEATURE_FORMAT_VERSION, FormatError
from yavih.io.inspect import inspect_features
from yavih.io.stage import StageError, stage_features
from yavih.io.validate import (
    CODES,
    Finding,
    ValidationError,
    ValidationReport,
    rule_times,
    rules_line,
    validate_features,
    validate_tree,
)
from yavih.waivers import WaiveError

MH = hashlib.sha256(b"manifest").hexdigest()
PH = hashlib.sha256(b"protocol").hexdigest()
IH = hashlib.sha256(b"input").hexdigest()
WEIGHTS = hashlib.sha256(b"weights").hexdigest()
DUR = 10.0
VIDEO_MEDIA = {"duration_s": DUR, "fps": 25.0, "n_frames": 250, "width": 320, "height": 240, "container": "mp4", "bytes": 1000}
AUDIO_MEDIA = {"duration_s": DUR, "sample_rate": 16000, "n_samples": 160000, "channels": 1, "container": "wav", "bytes": 1000}
NORM = {"mean": [0.5], "std": [0.5], "source": "example/model@abc transforms.py"}
DECODER = {"lib": "pyav", "version": "14.3.0", "libav": "61.19.100"}
FULL_FRAME = {"resize": None, "crop": None, "interp": "bilinear", "antialias": True}
CENTRE_CROP = {"resize": {"short": 224}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}
UNIFORM4 = {"rule": "uniform", "n": 4, "span": "full"}


# ----------------------------------------------------------------------------- helpers: pos, describe


def grid_pos(T: int, H: int, W: int, stream: int = 0, cls: bool = False, region=(0.0, 0.0, 1.0, 1.0), t_centres=None) -> np.ndarray:
    """``grid_pos`` re-implemented for the fixture: row-major (t, y, x) centres in media coordinates."""
    y0, x0, y1, x1 = region
    tc = [(t + 0.5) / T for t in range(T)] if t_centres is None else [float(v) for v in t_centres]
    rows = []
    if cls:
        rows.append([float(stream), -1.0, -1.0, -1.0])
    for t in range(T):
        for y in range(H):
            for x in range(W):
                rows.append([float(stream), tc[t], y0 + (y + 0.5) * (y1 - y0) / H, x0 + (x + 0.5) * (x1 - x0) / W])
    return np.asarray(rows, dtype=np.float32)


def video_input(temporal=None, spatial=None) -> dict:
    return {"temporal": dict(temporal or UNIFORM4), "spatial": dict(spatial or FULL_FRAME), "norm": dict(NORM), "decoder": dict(DECODER)}


def audio_input(temporal=None) -> dict:
    return {"temporal": dict(temporal or {"rule": "span", "start_s": 0.0, "length_s": DUR, "pad": "zeros"}),
            "front_end": {"sample_rate": 16000, "resample": "none", "features": "fbank"}, "norm": dict(NORM), "decoder": {"lib": "soundfile", "version": "0.12.1"}}


def make_describe(*, dtype="float16", batch_size=4, pos_mode="constant", streams=("video",), weights=WEIGHTS, args=None, inp=None, **extra) -> dict:
    d = {
        "weights_sha256": weights,
        "weights_ref": "/nowhere/model.pt",
        "dtype": dtype,
        "batch_size": batch_size,
        "streams": list(streams),
        "pos_mode": pos_mode,
        "model": "synth-model",
        "reading": "v",
        "layer": 12,
        "pooling": "none",
        "views": 1,
        "input": inp if inp is not None else {s: (audio_input() if s == "audio" else video_input()) for s in streams},
        "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
        "args": {"layer": 12} if args is None else args,
    }
    d.update(extra)
    return d


def expected_used(describe: dict, media: dict) -> dict:
    """What a rule-abiding bridge reports: the declared rule's times (video) / ranges (audio) on the media."""
    from yavih.io.validate import audio_ranges

    out: dict[str, Any] = {}
    for sname, entry in describe["input"].items():
        info = media.get(sname) or {}
        t = entry.get("temporal")
        if t is None:
            out[sname] = {}
        elif "front_end" in entry:
            try:
                r = audio_ranges(t, info)
            except ValueError:  # facts missing on this (deliberately broken) member
                r = None
            out[sname] = {"samples": r[0].tolist(), "pad": list(r[1])} if r is not None else {"samples": [], "pad": [0, 0]}
        else:
            try:
                times = rule_times(t, info)
            except ValueError:
                times = None
            out[sname] = {"times": [] if times is None else [float(v) for v in times]}
    return out


def default_feat(row: int, clip_id: str, N: int, D: int) -> np.ndarray:
    return np.random.default_rng(hashing.rng_seed(f"synth:{clip_id}")).standard_normal((N, D))


def tile(n_rows: int, batch_size: int, n_shards: int) -> list[tuple[int, int]]:
    n_batches = math.ceil(n_rows / batch_size)
    out = []
    for i in range(n_shards):
        b_lo = (i * n_batches) // n_shards
        b_hi = ((i + 1) * n_batches) // n_shards
        out.append((b_lo * batch_size, min(b_hi * batch_size, n_rows)))
    return out


def npy_bytes(arr: np.ndarray) -> tuple[bytes, int]:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(arr), version=(1, 0), allow_pickle=False)
    data = buf.getvalue()
    return data, len(data) - arr.nbytes


# ----------------------------------------------------------------------------- the feature writer


@dataclass
class Spec:
    """Everything the fixture writer needs; every field has the clean default."""

    split: str = "train"
    n_rows: int = 24
    n_shards: int = 2
    batch_size: int = 4
    N: int = 4
    D: int = 8
    dtype: str = "float16"
    pos_mode: str = "constant"
    pos: np.ndarray | Callable[[int, int], np.ndarray] | None = None  # constant array, or fn(shard, row) for per_sample / per-shard
    describe: dict | None = None
    bridge_args: dict | None = None
    bridge_name: str = "synth"
    manifest_hash: str = MH
    protocol_hash: str = PH
    input_hash: str = IH
    media_fingerprint: str | None = "mediafp"
    feat_fn: Callable[[int, str, int, int], np.ndarray] = default_feat
    missing: tuple[int, ...] = ()
    canaries: int = 0
    canary_noise: float = 0.0
    canary_used_fn: Callable[[int, dict], dict] | None = None  # row, original used -> canary used
    media_key_fn: Callable[[int, str], str | None] = lambda row, cid: None
    media_fn: Callable[[int], dict | None] | None = None  # row -> media block (None = omit the key)
    used_fn: Callable[[int, dict], dict | None] | None = None  # row, media -> used block (None = omit the key)
    json_row_fn: Callable[[int], int] = lambda row: row
    side_effects: list[str] = field(default_factory=list)
    n_override: int | None = None
    duplicate_rows: tuple[int, ...] = ()
    omit_pos: bool = False
    format_version: int = FEATURE_FORMAT_VERSION
    yavih_version: str = "0.2.0"
    tar_format: int = tarfile.PAX_FORMAT

    def clip_id(self, row: int) -> str:
        return f"{self.split}_{row:04d}"


def write_features(split_dir: Path, spec: Spec | None = None, **kw) -> Spec:
    """Write ``shard-*.tar`` + ``shard-*.json`` under ``split_dir`` per docs/features.md."""
    spec = spec or Spec(**kw)
    split_dir.mkdir(parents=True, exist_ok=True)
    for stale in split_dir.iterdir():  # a re-write replaces the directory's feature set
        if stale.name.startswith("shard-") or stale.name in ("_meta.json", "NOT_APPLICABLE.json"):
            stale.unlink()
    describe = spec.describe or make_describe(dtype=spec.dtype, batch_size=spec.batch_size, pos_mode=spec.pos_mode)
    describe_sha = hashing.describe_sha256(describe)
    np_dtype = np.dtype(spec.dtype)
    ranges = tile(spec.n_rows, spec.batch_size, spec.n_shards)

    def media_of(row: int) -> dict | None:
        if spec.media_fn is not None:
            return spec.media_fn(row)
        return {s: dict(AUDIO_MEDIA if "front_end" in e else VIDEO_MEDIA) for s, e in describe["input"].items()}

    for i, (lo, hi) in enumerate(ranges):
        tar_path = split_dir / f"shard-{i:06d}.tar"
        tmp = split_dir / f"shard-{i:06d}.tar.tmp"
        members: list[dict] = []
        missing: list[dict] = []
        pos_sha: str | None = None
        canary_max = 0.0
        n_canary = 0
        feats_written: dict[int, np.ndarray] = {}
        used_written: dict[int, Any] = {}
        with tarfile.open(tmp, "w", format=spec.tar_format) as tf:

            def add(name: str, data: bytes) -> int:
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                ti.mtime = 0
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mode = 0o644
                offset_data = tf.offset + len(ti.tobuf(tf.format, tf.encoding, tf.errors))
                tf.addfile(ti, io.BytesIO(data))
                return offset_data

            if spec.pos_mode == "constant" and not spec.omit_pos:
                pos = spec.pos(i, lo) if callable(spec.pos) else (spec.pos if spec.pos is not None else grid_pos(1, 2, 2))
                data, _ = npy_bytes(np.asarray(pos, dtype=np.float32))
                add("_pos.npy", data)
                pos_sha = hashlib.sha256(data).hexdigest()

            rows_to_write: list[int] = []
            for row in range(lo, hi):
                rows_to_write.append(row)
                if row in spec.duplicate_rows:
                    rows_to_write.append(row)
            for row in rows_to_write:
                cid = spec.clip_id(row)
                if row in spec.missing:
                    missing.append({"row": row, "clip_id": cid, "reason": "MediaMissing: video"})
                    continue
                mk = spec.media_key_fn(row, cid)
                j: dict[str, Any] = {"row": spec.json_row_fn(row), "clip_id": cid, "media_key": mk}
                media = media_of(row)
                if media is not None:
                    j["media"] = media
                used = spec.used_fn(row, media or {}) if spec.used_fn is not None else expected_used(describe, media or {})
                if used is not None:
                    j["used"] = used
                used_written[row] = used
                add(f"{row:09d}.json", json.dumps(j).encode("utf-8"))
                feat = np.asarray(spec.feat_fn(row, cid, spec.N, spec.D))
                if feat.dtype.kind == "f" and feat.dtype != np_dtype and feat.dtype == np.float64:
                    feat = feat.astype(np_dtype)
                data, header_len = npy_bytes(feat)
                off = add(f"{row:09d}.feat.npy", data)
                durations = {s: (m or {}).get("duration_s") for s, m in (media or {}).items()}
                members.append({"row": row, "clip_id": cid, "media_key": mk, "offset_data": off, "header_len": header_len, "nbytes": int(feat.nbytes), "duration_s": durations})
                feats_written[row] = feat
                if spec.pos_mode == "per_sample":
                    pos = spec.pos(i, row) if callable(spec.pos) else (spec.pos if spec.pos is not None else grid_pos(1, 2, 2))
                    data, _ = npy_bytes(np.asarray(pos, dtype=np.float32))
                    add(f"{row:09d}.pos.npy", data)
            for row in sorted(feats_written)[: spec.canaries]:
                cid = spec.clip_id(row)
                base = feats_written[row]
                cf = base.astype(np.float64)
                if spec.canary_noise:
                    cf = cf + spec.canary_noise * np.abs(cf).mean()
                cf = cf.astype(base.dtype)
                cused = spec.canary_used_fn(row, used_written[row]) if spec.canary_used_fn else used_written[row]
                cj = {"row": row, "clip_id": cid, "media_key": spec.media_key_fn(row, cid), "media": media_of(row), "used": cused, "canary_of": row}
                add(f"c{row:09d}.json", json.dumps(cj).encode("utf-8"))
                add(f"c{row:09d}.feat.npy", npy_bytes(cf)[0])
                rel = float(np.linalg.norm(cf.astype(np.float64) - base.astype(np.float64)) / max(np.linalg.norm(base.astype(np.float64)), 1e-12))
                canary_max = max(canary_max, rel)
                n_canary += 1
        os.replace(tmp, tar_path)
        sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        sidecar = {
            "format_version": spec.format_version,
            "yavih_version": spec.yavih_version,
            "benchmark": {
                "name": "synth",
                "protocol_version": 2,
                "manifest_hash": spec.manifest_hash,
                "protocol_hash": spec.protocol_hash,
                "input_hash": spec.input_hash,
                "media_fingerprint": spec.media_fingerprint,
                "media_info_sha256": None,
                "data_roots": {"media_root": "/data/synth"},
            },
            "split": spec.split,
            "model": describe.get("model"),
            "reading": describe.get("reading"),
            "shard": {"index": i, "n": spec.n_override or spec.n_shards, "rows": [lo, hi], "batch_size": spec.batch_size},
            "rows": {"n_range": hi - lo, "n_written": len(members), "n_missing": len(missing), "n_canary": n_canary},
            "missing": missing,
            # an empty shard never saw a feature: null N/D/pos_sha256, like FeatureWriter
            "features": {"dtype": spec.dtype, "N": spec.N if members else None, "D": spec.D if members else None, "pos_mode": spec.pos_mode, "pos_sha256": pos_sha if members else None},
            "bridge": {
                "name": spec.bridge_name,
                "describe": describe,
                "describe_sha256": describe_sha,
                "source_sha256": hashlib.sha256(b"bridge.py").hexdigest(),
                "source_path": "bridges/synth.py",
                "args": spec.bridge_args if spec.bridge_args is not None else dict(describe.get("args") or {}),
            },
            "tar": {"file": tar_path.name, "bytes": tar_path.stat().st_size, "sha256": sha},
            "members": members,
            "canary": {"max_abs_diff": 0.0, "max_rel_l2": canary_max, "tolerance": 1e-5},
            "side_effects": {"writes_outside_run_dir": list(spec.side_effects)},
            "env": {"host": "test", "python": "3.11", "torch": None, "cuda": None, "gpu": None, "slurm_job_id": None, "decode_workers": 1, "read_ahead": 2},
            "timing": {"started": "2026-10-08T00:00:00Z", "finished": "2026-10-08T00:00:01Z", "wall_s": 1.0, "clips_per_s": 10.0, "read_s": 0.5, "encode_s": 0.5},
        }
        (split_dir / f"shard-{i:06d}.json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    return spec


def edit_sidecar(split_dir: Path, i: int, fn: Callable[[dict], None]) -> None:
    p = split_dir / f"shard-{i:06d}.json"
    sc = json.loads(p.read_text(encoding="utf-8"))
    fn(sc)
    p.write_text(json.dumps(sc), encoding="utf-8")


# ----------------------------------------------------------------------------- fake benchmark


def requires(pos=(), streams=(), grid=None, t_bins=None) -> dict:
    return {"pos": list(pos), "streams": list(streams), "grid": grid, "t_bins": t_bins}


class FakeBench:
    """Duck-typed benchmark surface: ``name, spec, manifest_hash, protocol_hash, n_rows(), data_status()``."""

    def __init__(
        self,
        *,
        readout: str = "clf",
        n_rows: dict[str, int] | None = None,
        requires: dict | None = None,
        readout_params: dict | None = None,
        streams: dict[str, str] | None = None,
        protocol: dict | None = None,
        manifest_hash: str = MH,
        protocol_hash: str = PH,
        input_hash: str = IH,
        media_fingerprint: str | None = None,
    ) -> None:
        self.name = "synth"
        self.manifest_hash = manifest_hash
        self.protocol_hash = protocol_hash
        self.input_hash = input_hash
        self._n = n_rows or {"train": 24, "val": 24}
        self._fp = media_fingerprint
        streams = streams or {"video": "video"}
        self.spec = SimpleNamespace(
            readout=readout,
            output_contract={"shape": "fixed", "coordinates": "media", "target_frame": None, "requires": requires or {"pos": [], "streams": []}},
            readout_params=readout_params or {},
            streams={n: SimpleNamespace(kind=k) for n, k in streams.items()},
            protocol=protocol if protocol is not None else {"clip_seconds": DUR},
            splits=SimpleNamespace(train=("train",), eval=("val",), floor=(), folds=None, unused=()),
        )

    def n_rows(self, split: str) -> int:
        return self._n[split]

    def data_status(self):
        if self._fp is None:
            raise RuntimeError("no data_status on this fake")
        return SimpleNamespace(media_fingerprint=self._fp, media_info_sha256=None)


def codes(report: ValidationReport) -> dict[str, list[str]]:
    return {
        "errors": [f.code for f in report.errors],
        "warnings": [f.code for f in report.warnings],
        "notes": [f.code for f in report.notes],
    }


def has(report: ValidationReport, code: str) -> bool:
    return any(f.code == code for f in report.errors + report.warnings + report.notes)


@pytest.fixture
def split_dir(tmp_path: Path) -> Path:
    return tmp_path / "run" / "features" / "train"


# ============================================================================= clean case


def test_clean_passes_and_writes_meta(split_dir: Path) -> None:
    spec = write_features(split_dir, canaries=2)
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok, codes(rep)
    assert rep.errors == [] and rep.warnings == [], codes(rep)
    assert [n.code for n in rep.notes] == ["N_ELIGIBILITY"]
    assert (rep.N, rep.D, rep.dtype, rep.pos_mode) == (4, 8, "float16", "constant")
    assert rep.coverage == {"nominal": 24, "present": 24, "missing": 0, "ids_sha256": hashlib.sha256("\n".join(sorted(spec.clip_id(r) for r in range(24))).encode()).hexdigest()}
    assert rep.bytes == 24 * 4 * 8 * 2 + 2 * 4 * 4 * 4  # feats + one _pos.npy per shard
    assert rep.backend_hint == "ram"
    assert rep.missing_shards == []
    assert rep.shards == {"n": 2, "done": 2, "in_progress": 0, "missing": 0}
    assert rep.rules_checked == {"video": {"temporal": "uniform", "spatial": True}}
    assert rep.eligibility["eligible"] is True and rep.eligibility["readout"] == "clf"
    assert rep.eligibility["stream_tags"] == ["video"]
    lines = rep.summary_lines()
    assert "shards 2/2 done, 0 in progress, 0 missing" in lines[0]
    assert lines[1] == "rules checked: video uniform + spatial" and lines[2] == "eligible for clf: yes"

    meta = read_meta(split_dir)
    assert rep.meta_path == str(split_dir / "_meta.json")
    assert meta["format_version"] == 4
    assert (meta["model"], meta["reading"]) == ("synth-model", "v")
    assert meta["split"] == "train" and meta["benchmark"]["manifest_hash"] == MH
    assert meta["benchmark"]["media_info_sha256"] is None  # the sidecar's benchmark block passes through
    assert meta["yavih"] == {"writer_versions": ["0.2.0"], "writer_commits": [], "finalizer_version": meta["yavih"]["finalizer_version"],
                             "finalizer_commit": meta["yavih"]["finalizer_commit"]}
    assert meta["features"] == {"dtype": "float16", "N": 4, "D": 8, "pos_mode": "constant", "pos_sha256": meta["features"]["pos_sha256"], "bytes": rep.bytes}
    assert meta["features"]["pos_sha256"] is not None
    assert [s["rows"] for s in meta["shards"]] == [[0, 12], [12, 24]]
    assert meta["coverage"] == rep.coverage
    assert meta["canary"] == {"max_rel_l2": 0.0, "n": 4, "tolerance": 1e-5}
    assert meta["rules_checked"] == rep.rules_checked
    assert meta["dead_rows"] == []
    assert meta["eligibility"] == rep.eligibility
    assert meta["validation"]["ok"] is True and meta["validation"]["errors"] == []
    assert meta["validation"]["notes"][0]["code"] == "N_ELIGIBILITY"
    sidecars = [json.loads((split_dir / f"shard-{i:06d}.json").read_text()) for i in range(2)]
    fp = hashing.features_fingerprint(
        format_version=4, dtype="float16", N=4, D=8, pos_mode="constant", pos_sha256=meta["features"]["pos_sha256"],
        shards=[[s["shard"]["rows"][0], s["shard"]["rows"][1], s["tar"]["sha256"]] for s in sidecars], n_present=24,
    )
    assert meta["fingerprint"] == fp == meta_fingerprint(sidecars)
    assert list(meta) == [
        "format_version", "yavih", "finalized", "benchmark", "split", "model", "reading", "bridge", "features", "shards",
        "coverage", "canary", "rules_checked", "dead_rows", "waived", "fingerprint", "eligibility", "validation",
    ]  # fmt: skip
    assert set(meta["eligibility"]) == {"readout", "eligible", "missing", "reason", "target_region", "grid_cell_coverage", "t_bins", "stream_tags"}


def test_rerun_is_cheap_and_does_not_rewrite(split_dir: Path) -> None:
    write_features(split_dir)
    rep1 = validate_features(split_dir, bench=FakeBench())
    assert rep1.ok
    meta_path = split_dir / "_meta.json"
    mtime = meta_path.stat().st_mtime_ns
    time.sleep(0.01)
    rep2 = validate_features(split_dir, bench=FakeBench())
    assert rep2.ok and rep2.shards.get("reused_meta") is True
    assert meta_path.stat().st_mtime_ns == mtime
    assert rep2.eligibility == rep1.eligibility and rep2.coverage == rep1.coverage and rep2.rules_checked == rep1.rules_checked
    rep3 = validate_features(split_dir, bench=FakeBench(), verify_sha=True)
    assert rep3.ok and "reused_meta" not in rep3.shards
    assert meta_path.stat().st_mtime_ns == mtime
    # a scoring-side edit (another protocol_hash, same input_hash) leaves the features valid
    rep_scoring = validate_features(split_dir, bench=FakeBench(protocol_hash="0" * 64))
    assert rep_scoring.ok
    rep4 = validate_features(split_dir, bench=FakeBench(input_hash="0" * 64))
    assert not rep4.ok and "E_HASH_INPUT" in codes(rep4)["errors"]
    assert read_meta(split_dir)["validation"]["ok"] is False  # stale finalized meta demoted


def test_empty_shard_with_null_features_block(tmp_path: Path) -> None:
    d = tmp_path / "run" / "features" / "val"
    write_features(d, split="val", n_rows=4, n_shards=2, canaries=1)
    sc0 = json.loads((d / "shard-000000.json").read_text())
    assert sc0["shard"]["rows"] == [0, 0] and sc0["features"]["N"] is None and sc0["members"] == []
    rep = validate_features(d, bench=FakeBench(n_rows={"val": 4}))
    assert rep.ok, codes(rep)
    assert (rep.N, rep.D, rep.coverage["present"]) == (4, 8, 4)
    meta = read_meta(d)
    assert meta["features"]["N"] == 4 and meta["features"]["pos_sha256"] is not None
    assert [s["rows"] for s in meta["shards"]] == [[0, 0], [0, 4]]
    assert inspect_features(d)["N"] == 4
    assert diff_features(d, d)["verdict"] == "identical"
    assert validate_features(d, bench=FakeBench(n_rows={"val": 4})).shards.get("reused_meta") is True


def test_no_write_meta(split_dir: Path) -> None:
    write_features(split_dir)
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok and not (split_dir / "_meta.json").exists() and rep.meta_path is None


def test_validate_tree_over_features_parent(tmp_path: Path) -> None:
    root = tmp_path / "run" / "features"
    write_features(root / "train", split="train")
    write_features(root / "val", split="val")
    reports = validate_tree(root, bench=FakeBench())
    assert [r.split for r in reports] == ["train", "val"] and all(r.ok for r in reports)
    assert len(validate_tree(root / "val", bench=FakeBench())) == 1
    with pytest.raises(FileNotFoundError):
        validate_tree(tmp_path / "nothing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        validate_tree(tmp_path / "empty")


def test_bench_not_loadable_leaves_eligibility_undecided_but_checks_rules(split_dir: Path) -> None:
    write_features(split_dir)
    rep = validate_features(split_dir)  # Benchmark.load("synth") resolves nothing
    assert rep.ok, codes(rep)
    assert rep.eligibility["eligible"] is None and "undecided" in rep.eligibility["reason"]
    assert rep.coverage["nominal"] == 24
    assert read_meta(split_dir)["eligibility"]["eligible"] is None
    assert "undecided" in rep.summary_lines()[-1]
    # stream kinds are inferred from the members, so the rule checks still run without the benchmark
    write_features(split_dir, used_fn=lambda row, media: {"video": {"times": [0.0, 1.0, 2.0, 3.0]}})
    assert has(validate_features(split_dir, write_meta=False), "E_RULE_MISMATCH")


def test_report_json_and_raise(split_dir: Path) -> None:
    write_features(split_dir)
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    j = rep.to_json()
    assert set(j) >= {"ok", "errors", "warnings", "notes", "split", "coverage", "N", "D", "dtype", "pos_mode", "bytes", "backend_hint", "eligibility", "missing_shards", "rules_checked"}
    json.dumps(j)
    rep.raise_for_errors()
    rep.errors.append(Finding("E_ROWS", "boom", "shard-000000.json"))
    with pytest.raises(ValidationError) as ei:
        rep.raise_for_errors()
    assert "E_ROWS: boom" in str(ei.value)
    assert Finding.from_json(Finding("W_SIZE", "m", None).to_json()) == Finding("W_SIZE", "m", None)
    assert all(CODES[c] == ("error" if c[0] == "E" else "warning" if c[0] == "W" else "note") for c in CODES)


def test_code_table_has_rule_checks() -> None:
    for new in ("E_TIMES_OUT_OF_MEDIA", "E_RULE_MISMATCH", "E_BATCH_INVARIANCE", "W_RULE_UNCHECKED", "W_LOOSE_TOLERANCE", "W_SIDE_EFFECT"):
        assert new in CODES


def test_code_table_rule_and_weights_levels() -> None:
    """The opt-out is a note; W_SOURCE_SHA is in the table."""
    assert CODES["N_RULE_CUSTOM"] == "note" and CODES["W_RULE_UNCHECKED"] == "warning"
    assert CODES["W_WEIGHTS_UNFINGERPRINTED"] == "warning" and CODES["W_SOURCE_SHA"] == "warning"


# ============================================================================= E_* corruptions


def test_E_FORMAT(split_dir: Path) -> None:
    write_features(split_dir)
    edit_sidecar(split_dir, 0, lambda sc: sc.__setitem__("format_version", 99))
    rep = validate_features(split_dir, bench=FakeBench())
    assert not rep.ok and "E_FORMAT" in codes(rep)["errors"]
    assert not (split_dir / "_meta.json").exists()
    (split_dir / "shard-000001.json").write_text("{not json", encoding="utf-8")
    rep = validate_features(split_dir, bench=FakeBench())
    assert codes(rep)["errors"].count("E_FORMAT") == 2


def test_E_FORMAT_older_feature_set_must_be_re_extracted(split_dir: Path) -> None:
    """Only this yavih's feature format is read; an older one is re-extracted."""
    write_features(split_dir, format_version=FEATURE_FORMAT_VERSION - 1)
    rep = validate_features(split_dir, bench=FakeBench())
    assert not rep.ok and codes(rep)["errors"].count("E_FORMAT") == 2
    assert "re-extract" in rep.errors[0].message and "unsupported format_version" in rep.errors[0].message
    assert not (split_dir / "_meta.json").exists()


def test_E_NA_AND_SHARDS_and_na_alone(tmp_path: Path, split_dir: Path) -> None:
    write_features(split_dir)
    na = {"status": "not_applicable", "reason": "audio-only bridge: benchmark has no audio stream", "benchmark": "synth", "split": "train", "model": "x", "reading": "v", "bridge": {}, "created": "…"}
    (split_dir / "NOT_APPLICABLE.json").write_text(json.dumps(na))
    rep = validate_features(split_dir, bench=FakeBench())
    assert not rep.ok and "E_NA_AND_SHARDS" in codes(rep)["errors"]
    na_dir = tmp_path / "na" / "features" / "train"
    na_dir.mkdir(parents=True)
    (na_dir / "NOT_APPLICABLE.json").write_text(json.dumps(na))
    rep = validate_features(na_dir, bench=FakeBench())
    assert rep.ok and rep.errors == []
    assert rep.eligibility["eligible"] is False and "audio-only" in rep.eligibility["reason"]
    assert rep.coverage["not_applicable"] is True and rep.coverage["nominal"] == 24
    assert not (na_dir / "_meta.json").exists()
    assert rep.shards["not_applicable"] is True


def test_E_SHARD_SET_missing_index_lists_resume_args(split_dir: Path) -> None:
    write_features(split_dir, n_shards=4)
    (split_dir / "shard-000002.tar").unlink()
    (split_dir / "shard-000002.json").unlink()
    rep = validate_features(split_dir, bench=FakeBench())
    assert not rep.ok and "E_SHARD_SET" in codes(rep)["errors"]
    assert rep.missing_shards == ["--shard 2/4"]
    assert rep.shards == {"n": 4, "done": 3, "in_progress": 0, "missing": 1}
    assert "1 missing: --shard 2/4" in rep.summary_lines()[0]
    assert "E_RANGE_TILING" not in codes(rep)["errors"]


def test_E_SHARD_SET_variants(split_dir: Path) -> None:
    write_features(split_dir, n_shards=3)
    (split_dir / "shard-000001.tar").unlink()
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_SHARD_SET" and "sidecar without tar" in f.message for f in rep.errors)
    assert rep.missing_shards == ["--shard 1/3"]
    write_features(split_dir, n_shards=3)
    (split_dir / "shard-000002.json").unlink()
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_SHARD_SET" and "tar without sidecar" in f.message for f in rep.errors)
    write_features(split_dir, n_shards=3)
    edit_sidecar(split_dir, 1, lambda sc: sc["shard"].__setitem__("n", 5))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_SHARD_SET" and "disagree on n" in f.message for f in rep.errors)
    assert rep.shards["n"] is None


def test_W_STALE_TMP_and_in_progress(split_dir: Path) -> None:
    write_features(split_dir)
    stale = split_dir / "shard-000005.tar.tmp"
    stale.write_bytes(b"x")
    old = time.time() - 40 * 3600
    os.utime(stale, (old, old))
    (split_dir / "shard-000006.tar.tmp").write_bytes(b"x")
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and codes(rep)["warnings"] == ["W_STALE_TMP"]
    assert rep.shards["in_progress"] == 1


def test_E_RANGE_TILING(split_dir: Path) -> None:
    write_features(split_dir)
    edit_sidecar(split_dir, 1, lambda sc: sc["shard"].__setitem__("rows", [8, 24]))
    assert "E_RANGE_TILING" in codes(validate_features(split_dir, bench=FakeBench()))["errors"]
    write_features(split_dir)
    edit_sidecar(split_dir, 1, lambda sc: sc["shard"].__setitem__("rows", [14, 24]))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_RANGE_TILING" and "multiple of batch_size" in f.message for f in rep.errors)
    write_features(split_dir)
    rep = validate_features(split_dir, bench=FakeBench(n_rows={"train": 30}))
    assert any(f.code == "E_RANGE_TILING" and "30 rows" in f.message for f in rep.errors)


def test_E_HASH_MANIFEST_and_INPUT(split_dir: Path) -> None:
    write_features(split_dir)
    rep = validate_features(split_dir, bench=FakeBench(manifest_hash="1" * 64, input_hash="2" * 64))
    errs = codes(rep)["errors"]
    assert errs.count("E_HASH_MANIFEST") == 2 and errs.count("E_HASH_INPUT") == 2
    msg = next(f.message for f in rep.errors if f.code == "E_HASH_INPUT")
    assert "input_hash" in msg and "written by yavih" in msg and "read by yavih" in msg
    write_features(split_dir)
    edit_sidecar(split_dir, 1, lambda sc: sc["benchmark"].__setitem__("manifest_hash", "3" * 64))
    rep = validate_features(split_dir, bench=FakeBench(manifest_hash=MH))
    assert codes(rep)["errors"].count("E_HASH_MANIFEST") == 1


def test_E_DESCRIBE_MISMATCH_incl_declared_rules_and_decoders(split_dir: Path) -> None:
    write_features(split_dir)

    def bump(change):
        def edit(sc: dict) -> None:
            change(sc["bridge"]["describe"])
            sc["bridge"]["describe_sha256"] = hashing.describe_sha256(sc["bridge"]["describe"])

        return edit

    # yavih-describe-2: another decoder build or batch size is provenance, not another configuration
    for change in (lambda d: d["input"]["video"]["decoder"].__setitem__("libav", "60.3.100"), lambda d: d.__setitem__("batch_size", 2)):
        write_features(split_dir)
        edit_sidecar(split_dir, 1, bump(change))
        rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
        assert "E_DESCRIBE_MISMATCH" not in codes(rep)["errors"], codes(rep)
    for change in (lambda d: d.__setitem__("layer", 11), lambda d: d["input"]["video"]["temporal"].__setitem__("span", [0.0, 8.0])):
        write_features(split_dir)
        edit_sidecar(split_dir, 1, bump(change))
        rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
        msgs = [f.message for f in rep.errors if f.code == "E_DESCRIBE_MISMATCH"]
        assert any("bridge.describe " in m and "differs" in m for m in msgs) and any("bridge.describe_sha256 differs" in m for m in msgs)
        # the message names which shards belong to which configuration and how the foreign ones leave
        (dmsg,) = [m for m in msgs if m.startswith("bridge.describe_sha256 differs")]
        assert "shards [1]" in dmsg and "Re-run `yavih extract`" in dmsg and "_stale/" in dmsg, dmsg

    # one feature set has one model and one reading (consistency, E_DESCRIBE_MISMATCH)
    for key, other in (("model", "another-model"), ("reading", "a")):
        def rename(sc: dict, key: str = key, other: str = other) -> None:
            sc[key] = other
            sc["bridge"]["describe"][key] = other
            sc["bridge"]["describe_sha256"] = hashing.describe_sha256(sc["bridge"]["describe"])

        write_features(split_dir)
        edit_sidecar(split_dir, 1, rename)
        rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
        msgs = [f.message for f in rep.errors if f.code == "E_DESCRIBE_MISMATCH"]
        assert any(m.startswith(f"{key} differs") for m in msgs), msgs
    # a sidecar whose model / reading disagrees with its own describe()
    write_features(split_dir)
    for i in range(2):
        edit_sidecar(split_dir, i, lambda sc: sc.__setitem__("reading", "av"))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert any(f.code == "E_DESCRIBE_MISMATCH" and "reading 'av'" in f.message for f in rep.errors)

    write_features(split_dir)
    edit_sidecar(split_dir, 0, lambda sc: sc["features"].__setitem__("N", 5))
    edit_sidecar(split_dir, 0, lambda sc: sc["features"].__setitem__("dtype", "float32"))
    rep = validate_features(split_dir, bench=FakeBench())
    msgs = [f.message for f in rep.errors if f.code == "E_DESCRIBE_MISMATCH"]
    assert any("features.N" in m for m in msgs) and any("features.dtype" in m for m in msgs)

    write_features(split_dir)
    for i in range(2):
        edit_sidecar(split_dir, i, lambda sc: sc["bridge"].__setitem__("describe_sha256", "f" * 64))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any("sha256(canon(describe_subset(describe)))" in f.message for f in rep.errors if f.code == "E_DESCRIBE_MISMATCH")


def test_E_ARGS_UNCONSUMED(split_dir: Path) -> None:
    write_features(split_dir, bridge_args={"layer": 12, "typo_arg": 1})
    rep = validate_features(split_dir, bench=FakeBench())
    assert codes(rep)["errors"].count("E_ARGS_UNCONSUMED") == 2
    assert "typo_arg" in rep.errors[0].message


def test_E_TAR_SHA(split_dir: Path) -> None:
    write_features(split_dir)
    edit_sidecar(split_dir, 0, lambda sc: sc["tar"].__setitem__("sha256", "0" * 64))
    assert validate_features(split_dir, bench=FakeBench()).ok  # sha is only checked with verify_sha
    rep = validate_features(split_dir, bench=FakeBench(), verify_sha=True)
    assert codes(rep)["errors"] == ["E_TAR_SHA"]
    write_features(split_dir)
    edit_sidecar(split_dir, 1, lambda sc: sc["tar"].__setitem__("bytes", 7))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_MEMBERS" and "sidecar says 7 B" in f.message for f in rep.errors)
    assert "E_TAR_SHA" not in codes(rep)["errors"]


def test_E_MEMBERS_variants(split_dir: Path) -> None:
    write_features(split_dir)
    edit_sidecar(split_dir, 0, lambda sc: sc["members"].pop())
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_MEMBERS" and "member set ≠ sidecar members" in f.message for f in rep.errors)

    def feat_bad_shape(row, cid, N, D):
        f = default_feat(row, cid, N, D)
        return f[:-1] if row == 3 else f

    write_features(split_dir, feat_fn=feat_bad_shape)
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_MEMBERS" and "shape (3, 8)" in f.message and f.where == "shard-000000.json:000000003.feat.npy" for f in rep.errors)
    write_features(split_dir, feat_fn=lambda row, cid, N, D: default_feat(row, cid, N, D).astype(np.float32))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_MEMBERS" and "dtype float32 ≠ declared float16" in f.message for f in rep.errors)
    write_features(split_dir, json_row_fn=lambda row: row + 1 if row == 5 else row)
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_MEMBERS" and ".json.row 6 ≠ member name row 5" in f.message for f in rep.errors)
    write_features(split_dir)
    edit_sidecar(split_dir, 0, lambda sc: sc["members"][0].__setitem__("offset_data", 1))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_MEMBERS" and "offset_data/header_len" in f.message for f in rep.errors)


@pytest.mark.parametrize(
    "kw, needle",
    [
        (dict(media_fn=lambda row: None), "lacks 'media'"),
        (dict(used_fn=lambda row, media: None), "lacks 'used'"),
        (dict(used_fn=lambda row, media: {}), "used lacks stream 'video'"),
        (dict(used_fn=lambda row, media: {"video": {"times": "soon"}}), "video used entry"),
        (dict(media_fn=lambda row: {"audio": dict(AUDIO_MEDIA)}), "media lacks stream 'video'"),
    ],
)
def test_E_MEMBERS_media_and_used_provenance(split_dir: Path, kw, needle) -> None:
    write_features(split_dir, **kw)
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert any(f.code == "E_MEMBERS" and needle in f.message for f in rep.errors), codes(rep)


def test_E_ROWS(split_dir: Path) -> None:
    write_features(split_dir, missing=(5,))
    edit_sidecar(split_dir, 0, lambda sc: sc.__setitem__("missing", []))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_ROWS" and "neither written nor in missing: [5]" in f.message for f in rep.errors)
    write_features(split_dir, duplicate_rows=(3,))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_ROWS" and "written twice: [3]" in f.message for f in rep.errors)
    write_features(split_dir)
    edit_sidecar(split_dir, 0, lambda sc: sc["missing"].append({"row": 2, "clip_id": "train_0002", "reason": "x"}))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_ROWS" and "both written and listed missing" in f.message for f in rep.errors)


# ============================================================================= rules re-evaluated on the media facts


def test_rule_mismatch_temporal_video(split_dir: Path) -> None:
    bench = FakeBench()
    # linspace endpoints instead of bin centres
    write_features(split_dir, used_fn=lambda row, m: {"video": {"times": [0.0, 10 / 3, 20 / 3, 9.96]}})
    rep = validate_features(split_dir, bench=bench, write_meta=False)
    assert any(f.code == "E_RULE_MISMATCH" and "uniform" in f.message for f in rep.errors)
    # the nearest decoded frame (within half a frame + 1 ms at 25 fps) is the rule
    write_features(split_dir, used_fn=lambda row, m: {"video": {"times": [round(t * 25) / 25 for t in rule_times(UNIFORM4, m["video"])]}})
    assert validate_features(split_dir, bench=bench, write_meta=False).ok
    write_features(split_dir, used_fn=lambda row, m: {"video": {"times": [t + 0.03 for t in rule_times(UNIFORM4, m["video"])]}})
    assert has(validate_features(split_dir, bench=bench, write_meta=False), "E_RULE_MISMATCH")
    # a different frame count
    write_features(split_dir, used_fn=lambda row, m: {"video": {"times": [1.0, 5.0, 9.0]}})
    rep = validate_features(split_dir, bench=bench, write_meta=False)
    assert any(f.code == "E_RULE_MISMATCH" and "reports 3 times" in f.message for f in rep.errors)


def test_rule_is_re_evaluated_per_clip_on_its_own_duration(split_dir: Path) -> None:
    """Media coordinates: every clip has its own duration; a uniform rule over span full gives other times."""
    media = lambda row: {"video": dict(VIDEO_MEDIA, duration_s=8.0 + row % 3, n_frames=int(25 * (8.0 + row % 3)))}  # noqa: E731
    write_features(split_dir, media_fn=media)
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok, codes(rep)
    # the 10 s times reported for an 8 s clip are out of media
    write_features(split_dir, media_fn=media, used_fn=lambda row, m: {"video": {"times": [1.25, 3.75, 6.25, 8.75]}})
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert has(rep, "E_TIMES_OUT_OF_MEDIA") and has(rep, "E_RULE_MISMATCH")


def test_times_out_of_media_and_edge_policy(split_dir: Path) -> None:
    write_features(split_dir, used_fn=lambda row, m: {"video": {"times": [1.0, 3.0, 5.0, 11.0]}})
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert any(f.code == "E_TIMES_OUT_OF_MEDIA" and "outside [0, duration_s=10.0]" in f.message for f in rep.errors)
    # a declared span longer than a short clip with edge: error
    span8 = {"rule": "uniform", "n": 4, "span": [0.0, 8.0]}
    short = lambda row: {"video": dict(VIDEO_MEDIA, duration_s=6.0, n_frames=150)}  # noqa: E731
    d = make_describe(inp={"video": video_input(temporal=span8)})
    write_features(split_dir, describe=d, media_fn=short, used_fn=lambda row, m: {"video": {"times": [1.0, 3.0, 5.0, 5.96]}})
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert any(f.code == "E_TIMES_OUT_OF_MEDIA" and "edge: error" in f.message for f in rep.errors)
    # the same media under edge: clamp is the bridge's declared policy: the last frame serves the late time
    d = make_describe(inp={"video": video_input(temporal=dict(span8, edge="clamp"))})
    write_features(split_dir, describe=d, media_fn=short, used_fn=lambda row, m: {"video": {"times": [1.0, 3.0, 5.0, 5.96]}})
    assert validate_features(split_dir, bench=FakeBench(), write_meta=False).ok


AUDIO_SPAN = {"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"}


def _audio_describe(temporal=None) -> dict:
    return make_describe(streams=("audio",), inp={"audio": audio_input(temporal=temporal or AUDIO_SPAN)})


def test_rule_mismatch_and_out_of_media_audio(split_dir: Path) -> None:
    bench = FakeBench(streams={"audio": "audio"})
    write_features(split_dir, describe=_audio_describe())
    rep = validate_features(split_dir, bench=bench, write_meta=False)
    assert rep.ok and rep.rules_checked == {"audio": {"temporal": "span", "spatial": None}}
    # ±1 sample is within tolerance, 100 samples are not
    write_features(split_dir, describe=_audio_describe(), used_fn=lambda row, m: {"audio": {"samples": [[1, 160000]], "pad": [0, 1]}})
    assert validate_features(split_dir, bench=bench, write_meta=False).ok
    write_features(split_dir, describe=_audio_describe(), used_fn=lambda row, m: {"audio": {"samples": [[100, 160000]], "pad": [0, 0]}})
    assert has(validate_features(split_dir, bench=bench, write_meta=False), "E_RULE_MISMATCH")
    write_features(split_dir, describe=_audio_describe(), used_fn=lambda row, m: {"audio": {"samples": [[0, 170000]], "pad": [0, 0]}})
    assert has(validate_features(split_dir, bench=bench, write_meta=False), "E_TIMES_OUT_OF_MEDIA")
    # a 9.375 s file: span 10 s with pad zeros reports the shortfall as right padding
    short = lambda row: {"audio": dict(AUDIO_MEDIA, n_samples=150000, duration_s=9.375)}  # noqa: E731
    write_features(split_dir, describe=_audio_describe(), media_fn=short, used_fn=lambda row, m: {"audio": {"samples": [[0, 150000]], "pad": [0, 10000]}})
    assert validate_features(split_dir, bench=bench, write_meta=False).ok
    write_features(split_dir, describe=_audio_describe(), media_fn=short, used_fn=lambda row, m: {"audio": {"samples": [[0, 150000]], "pad": [0, 0]}})
    assert has(validate_features(split_dir, bench=bench, write_meta=False), "E_RULE_MISMATCH")  # unreported padding
    nopad = dict(AUDIO_SPAN, pad="none")
    write_features(split_dir, describe=_audio_describe(nopad), media_fn=short, used_fn=lambda row, m: {"audio": {"samples": [[0, 150000]], "pad": [0, 0]}})
    rep = validate_features(split_dir, bench=bench, write_meta=False)
    assert any(f.code == "E_TIMES_OUT_OF_MEDIA" and "pad: none" in f.message for f in rep.errors)


def test_trackless_stream_reports_an_empty_used_range(split_dir: Path) -> None:
    """A clip whose file holds no audio track (``media.audio.has_track`` false, the track facts
    null) is handed over; the bridge feeds its own declared choice and reports an empty consumption. That row
    is valid (no unchecked-rule warning: no rule can select from a track that is not there); a reported range on
    it is out of media; the audio ``pad`` stays the bridge's own."""
    bench = FakeBench(streams={"audio": "audio"})
    d = _audio_describe()
    trackless = dict(AUDIO_MEDIA, has_track=False, duration_s=None, sample_rate=None, n_samples=None, channels=None, container="mp4")
    media = lambda row: {"audio": trackless if row % 7 == 3 else dict(AUDIO_MEDIA, has_track=True)}  # noqa: E731
    empty = lambda row, m: {"audio": {"samples": [], "pad": [0, 0]}} if row % 7 == 3 else expected_used(d, m)  # noqa: E731
    write_features(split_dir, describe=d, media_fn=media, used_fn=empty)
    rep = validate_features(split_dir, bench=bench, write_meta=False)
    assert rep.ok and codes(rep)["warnings"] == [], codes(rep)
    assert "temporal_unchecked_rows" not in rep.rules_checked["audio"]
    # the pad is the bridge's own choice for the absent track (e.g. 10 s of its own silence at 16 kHz)
    own = lambda row, m: {"audio": {"samples": [], "pad": [0, 160000]}} if row % 7 == 3 else expected_used(d, m)  # noqa: E731
    write_features(split_dir, describe=d, media_fn=media, used_fn=own)
    assert validate_features(split_dir, bench=bench, write_meta=False).ok
    # a bridge that claims to have consumed samples of a track that is not there
    claims = lambda row, m: {"audio": {"samples": [[0, 160000]], "pad": [0, 0]}} if row % 7 == 3 else expected_used(d, m)  # noqa: E731
    write_features(split_dir, describe=d, media_fn=media, used_fn=claims)
    rep = validate_features(split_dir, bench=bench, write_meta=False)
    assert any(f.code == "E_TIMES_OUT_OF_MEDIA" and "no audio track (has_track: false)" in f.message for f in rep.errors), codes(rep)
    # the same for a video stream without a video track: times must be []
    vd = make_describe()
    novideo = dict(VIDEO_MEDIA, has_track=False, duration_s=None, fps=None, n_frames=None, width=None, height=None)
    vmedia = lambda row: {"video": novideo if row == 5 else dict(VIDEO_MEDIA, has_track=True)}  # noqa: E731
    write_features(split_dir, describe=vd, media_fn=vmedia, used_fn=lambda row, m: {"video": {"times": []}} if row == 5 else expected_used(vd, m))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok and codes(rep)["warnings"] == [], codes(rep)
    write_features(split_dir, describe=vd, media_fn=vmedia, used_fn=lambda row, m: {"video": {"times": [1.0]}} if row == 5 else expected_used(vd, m))
    assert has(validate_features(split_dir, bench=FakeBench(), write_meta=False), "E_TIMES_OUT_OF_MEDIA")


def test_rule_mismatch_spatial_token_centres_outside_the_declared_crop(split_dir: Path) -> None:
    crop = make_describe(pos_mode="per_sample", inp={"video": video_input(spatial=CENTRE_CROP)})
    region43 = (0.0, 0.125, 1.0, 0.875)  # resize short 224 + centre crop 224 of 320x240
    write_features(split_dir, describe=crop, pos_mode="per_sample", N=16, pos=lambda shard, row: grid_pos(1, 4, 4, region=region43))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok, codes(rep)
    # full-frame positions (a bridge that forgot the crop): tokens at x = 1/16 lie outside
    write_features(split_dir, describe=crop, pos_mode="per_sample", N=64, pos=lambda shard, row: grid_pos(1, 8, 8))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert any(f.code == "E_RULE_MISMATCH" and "outside the declared crop region" in f.message for f in rep.errors)
    # per clip: a 16:9 clip keeps x in [0.21875, 0.78125]; positions computed for 4:3 are wrong there
    wide = lambda row: {"video": dict(VIDEO_MEDIA, width=1280, height=720) if row % 2 else dict(VIDEO_MEDIA)}  # noqa: E731
    region169 = (0.0, 0.21875, 1.0, 0.78125)
    write_features(split_dir, describe=crop, pos_mode="per_sample", N=64, media_fn=wide, pos=lambda shard, row: grid_pos(1, 8, 8, region=region169 if row % 2 else region43))
    assert validate_features(split_dir, bench=FakeBench(), write_meta=False).ok
    write_features(split_dir, describe=crop, pos_mode="per_sample", N=64, media_fn=wide, pos=lambda shard, row: grid_pos(1, 8, 8, region=region43))
    assert has(validate_features(split_dir, bench=FakeBench(), write_meta=False), "E_RULE_MISMATCH")


def test_custom_rules_are_a_note_and_still_bounded(split_dir: Path) -> None:
    """A custom rule is a legitimate opt-out → the note N_RULE_CUSTOM and NO warning; the
    out-of-media and position checks still apply."""
    custom = make_describe(inp={"video": video_input(temporal={"rule": "custom", "description": "TSN"}, spatial={"rule": "custom", "description": "letterbox"})})
    write_features(split_dir, describe=custom, used_fn=lambda row, m: {"video": {"times": [4.9, 5.1]}})  # pos t 0.5 → 5.0 s
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and codes(rep)["warnings"] == [], codes(rep)
    notes = [f for f in rep.notes if f.code == "N_RULE_CUSTOM"]
    assert len(notes) == 1 and "temporal rule custom — TSN" in notes[0].message and "spatial rule custom — letterbox" in notes[0].message
    assert "checked: out-of-media" in notes[0].message
    assert read_meta(split_dir)["rules_checked"] == {"video": {"temporal": "custom", "spatial": False}}
    assert "rules checked: video custom (opted out: temporal: TSN; spatial: letterbox)" in rep.summary_lines()
    # the cached re-run keeps the note and still raises no warning
    rep2 = validate_features(split_dir, bench=FakeBench())
    assert rep2.shards.get("reused_meta") is True and codes(rep2)["warnings"] == []
    assert [f.code for f in rep2.notes].count("N_RULE_CUSTOM") == 1
    write_features(split_dir, describe=custom, used_fn=lambda row, m: {"video": {"times": [12.0]}})
    assert has(validate_features(split_dir, bench=FakeBench(), write_meta=False), "E_TIMES_OUT_OF_MEDIA")
    # a custom rule that names its times still has its token positions checked against them
    write_features(split_dir, describe=custom, used_fn=lambda row, m: {"video": {"times": [0.1, 0.2]}})
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert any(f.code == "E_RULE_MISMATCH" and "pos t of" in f.message for f in rep.errors)


def test_findings_per_code_are_capped(split_dir: Path) -> None:
    write_features(split_dir, n_rows=48, used_fn=lambda row, m: {"video": {"times": [0.0, 1.0, 2.0, 3.0]}})
    rep = validate_features(split_dir, bench=FakeBench(n_rows={"train": 48}), write_meta=False)
    mism = [f for f in rep.errors if f.code == "E_RULE_MISMATCH"]
    assert len(mism) == 21 and "suppressed" in mism[-1].message


# ============================================================================= numbers


def test_E_FINITE(split_dir: Path) -> None:
    def feat_nan(row, cid, N, D):
        f = default_feat(row, cid, N, D)
        if row == 7:
            f[1, 2] = np.nan
        if row == 8:
            f[0, 0] = np.inf
        return f

    write_features(split_dir, feat_fn=feat_nan)
    rep = validate_features(split_dir, bench=FakeBench())
    assert codes(rep)["errors"].count("E_FINITE") == 2
    assert rep.errors[0].where == "shard-000000.json:000000007.feat.npy"


def test_E_POS_SHAPE_and_RANGE(split_dir: Path) -> None:
    write_features(split_dir, omit_pos=True)
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_POS_SHAPE" and "_pos.npy missing" in f.message for f in rep.errors)
    write_features(split_dir, pos=grid_pos(1, 2, 2)[:, :3])
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_POS_SHAPE" and "(4, 3) ≠ (N, 4)" in f.message for f in rep.errors)
    bad = grid_pos(1, 2, 2)
    bad[0, 2] = 1.5
    write_features(split_dir, pos=bad)
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_POS_RANGE" and "outside [0, 1]" in f.message for f in rep.errors)
    write_features(split_dir, pos=grid_pos(1, 2, 2, stream=3))
    rep = validate_features(split_dir, bench=FakeBench())
    # the stream column and the cross-shard pos identity are structural (E_POS_SHAPE, never waivable);
    # E_POS_RANGE is the coordinates t, y, x only (waivable)
    assert any(f.code == "E_POS_SHAPE" and "stream index outside [0, len(describe.streams) = 1)" in f.message for f in rep.errors)
    assert "E_POS_RANGE" not in codes(rep)["errors"]
    write_features(split_dir, pos=lambda shard, row: grid_pos(1, 2, 2) * (1.0 if shard == 0 else 0.5))
    rep = validate_features(split_dir, bench=FakeBench())
    assert any(f.code == "E_POS_SHAPE" and "differs across shards" in f.message for f in rep.errors)
    write_features(split_dir, pos_mode="per_sample", describe=make_describe(pos_mode="per_sample"))
    assert validate_features(split_dir, bench=FakeBench(), write_meta=False).ok
    write_features(split_dir, pos_mode="none", describe=make_describe(pos_mode="none"))
    assert validate_features(split_dir, bench=FakeBench(), write_meta=False).ok
    # a token centred in bridge padding carries t = -1
    padded = grid_pos(2, 1, 2)
    padded[2:, 1] = -1.0
    write_features(split_dir, pos=padded)
    assert validate_features(split_dir, bench=FakeBench(), write_meta=False).ok


def test_cheap_recheck_drops_codes_not_in_the_table(split_dir: Path) -> None:
    """Codes carried from a reused ``_meta.json`` take the current table's level; a code not in the table is dropped."""
    import json as _json

    write_features(split_dir, canaries=2, canary_noise=1e-7, dtype="float32", describe=make_describe(dtype="float32"))
    assert validate_features(split_dir, bench=FakeBench()).ok
    mp = split_dir / "_meta.json"
    meta = _json.loads(mp.read_text())
    meta["validation"]["warnings"] = [{"code": "W_NOT_A_CODE", "message": "unknown", "where": None},
                                      {"code": "W_SIZE", "message": "sentinel: only the cheap path carries this", "where": None}]
    meta["validation"]["notes"] = [{"code": "N_CANARY_DRIFT", "message": "carried", "where": None}]
    mp.write_text(_json.dumps(meta))
    rep = validate_features(split_dir, bench=FakeBench())
    c = codes(rep)
    assert "W_SIZE" in c["warnings"], "the cheap re-run was not taken"
    assert "W_NOT_A_CODE" not in c["warnings"] + c["notes"] and "N_CANARY_DRIFT" in c["notes"]


def test_E_BATCH_INVARIANCE_and_N_CANARY_DRIFT(split_dir: Path) -> None:
    f32 = dict(dtype="float32", describe=make_describe(dtype="float32"))
    write_features(split_dir, canaries=2, canary_noise=0.1, **f32)
    rep = validate_features(split_dir, bench=FakeBench())
    assert codes(rep)["errors"] == ["E_BATCH_INVARIANCE"] and "batch" in rep.errors[0].message
    write_features(split_dir, canaries=2, canary_noise=1e-7, **f32)
    rep = validate_features(split_dir, bench=FakeBench())
    # in-tolerance drift is a recorded NOTE, never a warning
    assert rep.ok and codes(rep)["warnings"] == [] and codes(rep)["notes"].count("N_CANARY_DRIFT") == 1
    drift = next(f for f in rep.notes if f.code == "N_CANARY_DRIFT")
    assert "within tolerance" in drift.message and "no action needed" in drift.message
    assert "canary.max_rel_l2" in drift.message
    assert [f["code"] for f in read_meta(split_dir)["validation"]["notes"]].count("N_CANARY_DRIFT") == 1
    meta = read_meta(split_dir)
    assert 0 < meta["canary"]["max_rel_l2"] <= 1e-5 and meta["canary"]["n"] == 4 and meta["canary"]["tolerance"] == 1e-5
    rep = validate_features(split_dir, bench=FakeBench(), canary_rel_tol=1e-9, verify_sha=True)
    assert "E_BATCH_INVARIANCE" in codes(rep)["errors"]
    # 1e-4 drift: an error at the default tolerance ...
    write_features(split_dir, canaries=2, canary_noise=1e-4, **f32)
    assert "E_BATCH_INVARIANCE" in codes(validate_features(split_dir, bench=FakeBench(), write_meta=False))["errors"]
    # ... recorded drift under a declared one (never looser than 1e-3)
    loose = make_describe(dtype="float32", batch_tolerance={"rel_l2": 1e-3, "reason": "bf16 kernels differ between batch sizes"})
    write_features(split_dir, canaries=2, canary_noise=1e-4, dtype="float32", describe=loose)
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and set(codes(rep)["warnings"]) == {"W_LOOSE_TOLERANCE"} and "N_CANARY_DRIFT" in codes(rep)["notes"]
    assert any("bf16 kernels" in f.message for f in rep.warnings)
    assert read_meta(split_dir)["canary"]["tolerance"] == 1e-3
    huge = make_describe(dtype="float32", batch_tolerance={"rel_l2": 0.5, "reason": "too loose"})
    write_features(split_dir, canaries=2, canary_noise=1e-2, dtype="float32", describe=huge)
    assert "E_BATCH_INVARIANCE" in codes(validate_features(split_dir, bench=FakeBench(), write_meta=False))["errors"]  # capped at 1e-3


def test_E_BATCH_INVARIANCE_when_a_canary_reports_other_used(split_dir: Path) -> None:
    write_features(split_dir, canaries=2, canary_used_fn=lambda row, used: {"video": {"times": [t + 0.04 for t in used["video"]["times"]]}})
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert any(f.code == "E_BATCH_INVARIANCE" and "used" in f.message for f in rep.errors)


def _feat_dead(row, cid, N, D):
    f = default_feat(row, cid, N, D)
    f[0] = 1.0
    return f


def test_E_DEAD_ROWS_and_its_waiver(split_dir: Path) -> None:
    write_features(split_dir, feat_fn=_feat_dead)
    rep = validate_features(split_dir, bench=FakeBench())
    assert codes(rep)["errors"] == ["E_DEAD_ROWS"] and "[0]" in rep.errors[0].message and "describe()['waive']" in rep.errors[0].message
    write_features(split_dir, feat_fn=_feat_dead, describe=make_describe(waive={"E_DEAD_ROWS": "a padding token"}))
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and codes(rep)["warnings"] == ["W_WAIVED"]
    assert rep.warnings[0].message.startswith("E_DEAD_ROWS: ") and "waived: a padding token" in rep.warnings[0].message
    meta = read_meta(split_dir)
    assert meta["dead_rows"] == [0]
    assert meta["waived"] == [{"code": "E_DEAD_ROWS", "reason": "a padding token", "source": "describe", "count": 1}]


# ----------------------------------------------------------------------------- waivers (yavih.waivers)


def test_describe_waive_turns_a_waivable_error_into_W_WAIVED(split_dir: Path) -> None:
    same = lambda row, cid, N, D: default_feat(0, "fixed", N, D)  # noqa: E731
    write_features(split_dir, feat_fn=same, describe=make_describe(waive={"E_DUPLICATE_CLIPS": "silent audio windows"}))
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and codes(rep)["errors"] == []
    waived = [f for f in rep.warnings if f.code == "W_WAIVED"]
    assert waived and waived[0].message.startswith("E_DUPLICATE_CLIPS: ") and waived[0].message.endswith("waived: silent audio windows")
    meta = read_meta(split_dir)
    assert meta["waived"] == [{"code": "E_DUPLICATE_CLIPS", "reason": "silent audio windows", "source": "describe", "count": 1}]
    assert meta["validation"]["ok"] is True and any(w["code"] == "W_WAIVED" for w in meta["validation"]["warnings"])
    # the cheap re-check of a finalized directory keeps the record
    again = validate_features(split_dir, bench=FakeBench())
    assert again.ok and again.shards.get("reused_meta") and again.waived == meta["waived"]


def test_star_waives_every_waivable_check(split_dir: Path) -> None:
    write_features(split_dir, feat_fn=_feat_dead, describe=make_describe(waive={"*": "research probe of a broken checkpoint"}))
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and codes(rep)["warnings"] == ["W_WAIVED"]
    from yavih.waivers import WAIVABLE

    assert {w["code"] for w in rep.waived} == set(WAIVABLE)  # every waiver in force is recorded …
    assert {w["code"]: w["count"] for w in rep.waived}["E_DEAD_ROWS"] == 1  # … with what it waived
    assert all(w["count"] == 0 for w in rep.waived if w["code"] != "E_DEAD_ROWS")


@pytest.mark.parametrize("waive, needle", [
    ({"E_HASH_INPUT": "different manifest"}, "identity"),
    ({"E_FINITE": "nan is fine"}, "structural"),
    ({"E_TEMPLATE_UNFILLED": "later"}, "unfilled-template"),
    ({"W_COVERAGE": "x"}, "nothing to waive"),
    ({"E_DUPLICATE_CLIPS": ""}, "non-empty reason"),
    ({"E_DUPLICATE_CLIPS": None}, "non-empty reason"),
    ({"E_NOPE": "x"}, "not a waivable check"),
    (["E_DUPLICATE_CLIPS"], "mapping"),
])
def test_invalid_describe_waiver_is_E_WAIVE_INVALID(split_dir: Path, waive, needle) -> None:
    write_features(split_dir, describe=make_describe(waive=waive))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert "E_WAIVE_INVALID" in codes(rep)["errors"], codes(rep)
    assert any(needle in f.message for f in rep.errors if f.code == "E_WAIVE_INVALID")


def test_non_waivable_errors_stay_errors_under_star(split_dir: Path) -> None:
    write_features(split_dir, describe=make_describe(waive={"*": "anything"}), feat_fn=lambda row, cid, N, D: np.full((N, D), np.nan, np.float16))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert not rep.ok and "E_FINITE" in codes(rep)["errors"]


def test_cli_waiver_is_recorded_and_carried_over(split_dir: Path) -> None:
    write_features(split_dir, feat_fn=_feat_dead)
    assert not validate_features(split_dir, bench=FakeBench()).ok
    rep = validate_features(split_dir, bench=FakeBench(), waive={"E_DEAD_ROWS": "register tokens are constant by design"})
    assert rep.ok and rep.waived == [{"code": "E_DEAD_ROWS", "reason": "register tokens are constant by design", "source": "cli", "count": 1}]
    assert "(--waive)" in next(f.message for f in rep.warnings if f.code == "W_WAIVED")
    meta = read_meta(split_dir)
    assert meta["waived"] == rep.waived
    # the describe and its digest are untouched by an operator's waiver
    sc = json.loads((split_dir / "shard-000000.json").read_text())
    assert meta["bridge"]["describe_sha256"] == hashing.describe_sha256(sc["bridge"]["describe"]) and "waive" not in sc["bridge"]["describe"]
    # a full re-validation (--verify-sha) of the same configuration keeps the operator's waiver
    rep2 = validate_features(split_dir, bench=FakeBench(), verify_sha=True)
    assert rep2.ok and rep2.waived[0]["source"] == "cli"
    with pytest.raises(WaiveError):
        validate_features(split_dir, bench=FakeBench(), waive={"E_ROWS": "x"})


def test_all_alias_on_the_command_line(split_dir: Path, capsys) -> None:
    from yavih import cli

    write_features(split_dir, feat_fn=_feat_dead)
    assert cli.main(["--json", "features", "validate", str(split_dir)]) == 2
    capsys.readouterr()
    assert cli.main(["--json", "features", "validate", str(split_dir), "--waive", "all=checkpoint under study"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["status"] == "ok" and {w["code"] for w in env["data"]["waived"]} >= {"E_DEAD_ROWS", "E_DUPLICATE_CLIPS"}
    assert all(w["source"] == "cli" for w in env["data"]["waived"])
    assert cli.main(["--json", "features", "validate", str(split_dir), "--waive", "E_FORMAT=x"]) == 1
    env = json.loads(capsys.readouterr().out)
    assert env["errors"][0]["code"] == "E_WAIVE_INVALID" and "structural" in env["errors"][0]["message"]
    assert cli.main(["--json", "features", "validate", str(split_dir), "--waive", "E_DEAD_ROWS"]) == 1
    assert "CODE=reason" in json.loads(capsys.readouterr().out)["errors"][0]["message"]


def test_waived_E_POS_RANGE_and_structural_pos_checks(split_dir: Path) -> None:
    bad = grid_pos(1, 2, 2)
    bad[0, 2] = 1.5  # a token centre below the frame (letterbox padding)
    write_features(split_dir, pos=bad, describe=make_describe(waive={"E_POS_RANGE": "letterbox padding tokens"}))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok and any(f.code == "W_WAIVED" and f.message.startswith("E_POS_RANGE: ") for f in rep.warnings)
    # the stream column is structural (the readouts index streams by it): never waivable
    write_features(split_dir, pos=grid_pos(1, 2, 2, stream=3), describe=make_describe(waive={"E_POS_RANGE": "letterbox padding tokens"}))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert not rep.ok and "E_POS_SHAPE" in codes(rep)["errors"]


def test_E_DUPLICATE_CLIPS_and_same_media_key_exempt(split_dir: Path) -> None:
    same = lambda row, cid, N, D: default_feat(0, "fixed", N, D)  # noqa: E731
    write_features(split_dir, feat_fn=same)
    rep = validate_features(split_dir, bench=FakeBench())
    assert "E_DUPLICATE_CLIPS" in codes(rep)["errors"] and "E_DEAD_ROWS" not in codes(rep)["errors"]
    # rows that share a media_key read identical media (AVE's events of one video) → exempt
    write_features(split_dir, feat_fn=lambda row, cid, N, D: default_feat(row // 2, f"m{row // 2}", N, D), media_key_fn=lambda row, cid: f"m{row // 2}")
    assert validate_features(split_dir, bench=FakeBench()).ok
    write_features(split_dir, feat_fn=lambda row, cid, N, D: default_feat(0, "a", N, D) if row in (1, 20) else default_feat(row, cid, N, D))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert "E_DUPLICATE_CLIPS" in codes(rep)["errors"]
    rep_a = validate_features(split_dir, bench=FakeBench(), write_meta=False, sample_rows=5)
    rep_b = validate_features(split_dir, bench=FakeBench(), write_meta=False, sample_rows=5)
    assert codes(rep_a) == codes(rep_b)


# ============================================================================= N_ELIGIBILITY (media coordinates)

AVE_PROTOCOL = {"num_segments": 10, "segment_seconds": 1.0, "segments_scored": 10, "segment_offset": 0, "background_index": 28}
SEQ_BENCH = dict(readout="seq", requires=requires(pos=("t",), streams=("video",), t_bins=8), readout_params={"T": 10}, protocol=AVE_PROTOCOL)


def test_eligibility_N1_dense_readout_is_ineligible_but_ok(split_dir: Path) -> None:
    write_features(split_dir, N=1, pos_mode="none", describe=make_describe(pos_mode="none"))
    rep = validate_features(split_dir, bench=FakeBench(**SEQ_BENCH))
    assert rep.ok and rep.errors == []
    assert rep.eligibility["eligible"] is False
    assert rep.eligibility["missing"] == ["tokens>1", "pos.t", "stream:video", "t_bins"]
    assert "N == 1" in rep.eligibility["reason"]
    line = rep.summary_lines()[-1]
    assert line.startswith("eligible for seq: no (") and line.endswith("→ probe will record not_applicable")
    meta = read_meta(split_dir)
    assert meta["eligibility"]["eligible"] is False and meta["validation"]["ok"] is True
    rep = validate_features(split_dir, bench=FakeBench(readout="clf"))
    assert rep.ok and rep.eligibility["eligible"] is True and rep.eligibility["t_bins"] is None


def _uniform_used(n: int):
    rule = {"rule": "uniform", "n": n, "span": "full"}
    return lambda row, m: {"video": {"times": [float(t) for t in rule_times(rule, m["video"])]}}


def test_eligibility_seq_counts_scored_seconds_in_media_coordinates(split_dir: Path) -> None:
    # 8 temporal tokens over a 10 s clip land in seconds 0,1,3,4,5,6,8,9: 8 scored seconds ≥ t_bins 8
    u8 = make_describe(inp={"video": video_input(temporal={"rule": "uniform", "n": 8, "span": "full"})})
    write_features(split_dir, N=8, pos=grid_pos(8, 1, 1), describe=u8, used_fn=_uniform_used(8))
    rep = validate_features(split_dir, bench=FakeBench(**SEQ_BENCH), write_meta=False)
    assert rep.ok and rep.eligibility["eligible"] is True and rep.eligibility["t_bins"] == {"median": 8, "min": 8}
    # 4 tokens cover 4 seconds
    write_features(split_dir, N=4, pos=grid_pos(4, 1, 1))
    rep = validate_features(split_dir, bench=FakeBench(**SEQ_BENCH), write_meta=False)
    assert rep.eligibility["eligible"] is False and rep.eligibility["missing"] == ["t_bins"] and rep.eligibility["t_bins"]["median"] == 4
    assert "median 4 < 8" in rep.eligibility["reason"]
    # the same 8 tokens on 5 s clips (half the rows) cover only 5 seconds: the median over the sample decides
    half_short = lambda row: {"video": dict(VIDEO_MEDIA, duration_s=5.0 if row % 2 else 10.0)}  # noqa: E731
    write_features(split_dir, N=8, pos=grid_pos(8, 1, 1), describe=u8, media_fn=half_short, used_fn=_uniform_used(8))
    rep = validate_features(split_dir, bench=FakeBench(**SEQ_BENCH), write_meta=False)
    assert rep.eligibility["t_bins"] == {"median": 6.5, "min": 5} and rep.eligibility["eligible"] is False


IMG_MEDIA = {"width": 640, "height": 480, "container": "png", "bytes": 1000}
IMG_DESC = make_describe(streams=("image",), inp={"image": {"spatial": dict(FULL_FRAME), "norm": dict(NORM), "decoder": dict(DECODER)}})
GRID_BENCH = dict(
    readout="grid",
    requires=requires(pos=("y", "x"), streams=("image",), grid={"h": 2, "w": 2, "min_cell_coverage": 0.5}),
    readout_params={"h": 2, "w": 2, "out": 1, "visual_stream": "image"},
    streams={"image": "image"},
    protocol={"target_region": [0.0, 0.125, 1.0, 0.875], "max_depth": 10.0, "target_size": 224},
)


def _grid_write(split_dir: Path, pos: np.ndarray) -> None:
    write_features(split_dir, N=len(pos), pos=pos, describe=IMG_DESC, media_fn=lambda row: {"image": dict(IMG_MEDIA)}, used_fn=lambda row, m: {"image": {}})


def test_eligibility_grid_coverage_of_the_target_region(split_dir: Path) -> None:
    gb = FakeBench(**GRID_BENCH)
    _grid_write(split_dir, grid_pos(1, 4, 4, t_centres=[-1.0]))  # full-frame 4×4 centres: every cell of the 2×2 target region holds one
    rep = validate_features(split_dir, bench=gb, write_meta=False)
    assert rep.ok and rep.eligibility["eligible"] is True
    assert rep.eligibility["grid_cell_coverage"] == {"median": 1.0, "min": 1.0} and rep.eligibility["target_region"] == [0.0, 0.125, 1.0, 0.875]
    assert rep.rules_checked == {"image": {"temporal": None, "spatial": True}}
    # all tokens in one cell → 0.25 < 0.5
    one_cell = grid_pos(1, 2, 2, t_centres=[-1.0])
    one_cell[:, 2:] = 0.3
    _grid_write(split_dir, one_cell)
    rep = validate_features(split_dir, bench=gb, write_meta=False)
    assert rep.eligibility["missing"] == ["grid_cell_coverage"] and rep.eligibility["grid_cell_coverage"]["median"] == 0.25
    assert "over the target region" in rep.eligibility["reason"]
    # tokens only in the left strip, outside the scored region, count for nothing
    left = grid_pos(1, 4, 4, t_centres=[-1.0], region=(0.0, 0.0, 1.0, 0.1))
    _grid_write(split_dir, left)
    rep = validate_features(split_dir, bench=gb, write_meta=False)
    assert rep.eligibility["grid_cell_coverage"]["median"] == 0.0 and rep.eligibility["eligible"] is False
    # missing axes: pos_mode none
    write_features(split_dir, N=4, pos_mode="none", describe=make_describe(pos_mode="none", streams=("image",), inp=IMG_DESC["input"]),
                   media_fn=lambda row: {"image": dict(IMG_MEDIA)}, used_fn=lambda row, m: {"image": {}})
    rep = validate_features(split_dir, bench=gb, write_meta=False)
    assert rep.eligibility["missing"][:2] == ["pos.y", "pos.x"]


def test_eligibility_box_streams(split_dir: Path) -> None:
    bb = FakeBench(
        readout="box",
        requires=requires(pos=("y", "x"), streams=("video", "audio"), grid={"h": 2, "w": 2, "min_cell_coverage": 0.25}),
        readout_params={"grid": 2, "visual_stream": "video", "conditioning_stream": "audio", "shuffled_control": True, "shuffle_seed": 1},
        streams={"video": "video", "audio": "audio"},
    )
    write_features(split_dir, N=4, pos=grid_pos(1, 2, 2))
    rep = validate_features(split_dir, bench=bb, write_meta=False)
    assert rep.ok and rep.eligibility["missing"] == ["stream:audio"]
    assert "stream:audio" in rep.summary_lines()[-1]
    pos = np.concatenate([grid_pos(1, 2, 2, stream=0), np.array([[1.0, 0.25, -1, -1], [1.0, 0.75, -1, -1]], dtype=np.float32)])
    write_features(split_dir, N=6, pos=pos, describe=make_describe(streams=("video", "audio")))
    rep = validate_features(split_dir, bench=bb, write_meta=False)
    assert rep.ok and rep.eligibility["eligible"] is True and rep.eligibility["stream_tags"] == ["audio", "video"]
    write_features(split_dir, N=4, pos=grid_pos(1, 2, 2), describe=make_describe(streams=("video", "audio")))
    rep = validate_features(split_dir, bench=bb, write_meta=False)
    assert rep.eligibility["missing"] == ["stream:audio"]


# ============================================================================= W_*


def test_W_SIDE_EFFECT(split_dir: Path) -> None:
    write_features(split_dir, side_effects=["/home/u/.cache/triton/x.json"])
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok and codes(rep)["warnings"] == ["W_SIDE_EFFECT", "W_SIDE_EFFECT"] and "triton" in rep.warnings[0].message


def test_W_COVERAGE(split_dir: Path) -> None:
    write_features(split_dir, missing=(3, 17))
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and codes(rep)["warnings"] == ["W_COVERAGE"]
    assert rep.coverage["present"] == 22 and rep.coverage["missing"] == 2 and rep.coverage["nominal"] == 24
    meta = read_meta(split_dir)
    assert meta["coverage"]["present"] == 22 and [s["n_written"] for s in meta["shards"]] == [11, 11]


def test_W_MEDIA_FP(split_dir: Path) -> None:
    write_features(split_dir, media_fingerprint="old")
    rep = validate_features(split_dir, bench=FakeBench(media_fingerprint="new"), write_meta=False)
    assert rep.ok and codes(rep)["warnings"] == ["W_MEDIA_FP"]
    rep = validate_features(split_dir, bench=FakeBench(media_fingerprint="old"), write_meta=False)
    assert codes(rep)["warnings"] == []


def test_W_FP16_RANGE(split_dir: Path) -> None:
    write_features(split_dir, feat_fn=lambda row, cid, N, D: default_feat(row, cid, N, D) * 3000.0)
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok and codes(rep)["warnings"] == ["W_FP16_RANGE"]

    def mostly_zero(row, cid, N, D):
        f = default_feat(row, cid, N, D)
        f[:, : D * 3 // 4] = 0.0
        return f

    write_features(split_dir, feat_fn=mostly_zero)
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert "zero fraction = 0.75" in [f.message for f in rep.warnings if f.code == "W_FP16_RANGE"][0]
    write_features(split_dir, dtype="float32", describe=make_describe(dtype="float32"), feat_fn=lambda row, cid, N, D: default_feat(row, cid, N, D) * 3000.0)
    assert codes(validate_features(split_dir, bench=FakeBench(), write_meta=False))["warnings"] == []


def test_W_CONSTANT_CLIP(split_dir: Path) -> None:
    def feat(row, cid, N, D):
        f = default_feat(row, cid, N, D)
        if row in (4, 5):
            f[:] = 0.0
        if row == 9:
            f[:] = f[0]
        return f

    write_features(split_dir, feat_fn=feat, media_key_fn=lambda row, cid: "shared" if row in (4, 5) else None)
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    w = [f for f in rep.warnings if f.code == "W_CONSTANT_CLIP"]
    assert len(w) == 1 and w[0].message.startswith("2 clip(s)")


def test_W_SIZE(split_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_features(split_dir)
    monkeypatch.setenv("YAVIH_RAM_BUDGET_GB", "0.0000001")
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok and codes(rep)["warnings"] == ["W_SIZE"] and rep.backend_hint == "memmap"
    monkeypatch.delenv("YAVIH_RAM_BUDGET_GB")
    assert validate_features(split_dir, bench=FakeBench(), write_meta=False).backend_hint == "ram"


def test_W_WEIGHTS_UNFINGERPRINTED(split_dir: Path) -> None:
    """A null weights_sha256 is a warning only: nothing marked or forked."""
    write_features(split_dir, describe=make_describe(weights=None))
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and codes(rep)["warnings"] == ["W_WEIGHTS_UNFINGERPRINTED"]
    assert "weights_ref" in rep.warnings[0].message
    meta = read_meta(split_dir)
    assert meta["model"] == "synth-model" and "~" not in meta["model"]
    assert CODES["W_WEIGHTS_UNFINGERPRINTED"] == "warning"


def test_W_SOURCE_SHA_when_shards_come_from_different_bridge_files(split_dir: Path) -> None:
    write_features(split_dir)
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok and "W_SOURCE_SHA" not in codes(rep)["warnings"]
    edit_sidecar(split_dir, 1, lambda sc: sc["bridge"].__setitem__("source_sha256", "9" * 64))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok and "W_SOURCE_SHA" in codes(rep)["warnings"] and "E_DESCRIBE_MISMATCH" not in codes(rep)["errors"]


# ============================================================================= finalize


# ============================================================================= parallel shard scan


def _nan_every_fifth(row, cid, N, D):
    f = default_feat(row, cid, N, D)
    if row % 5 == 0:
        f[0, 1] = np.nan
    return f


def _per_row_grid(shard: int, row: int) -> np.ndarray:
    return grid_pos(1, 4, 4, t_centres=[-1.0], region=(0.0, 0.0, 1.0, 0.3 + 0.7 * ((row % 4) / 3)))


_HALF_SHORT = lambda row: {"video": dict(VIDEO_MEDIA, duration_s=5.0 if row % 2 else 10.0)}  # noqa: E731
_U8 = make_describe(pos_mode="per_sample", inp={"video": video_input(temporal={"rule": "uniform", "n": 8, "span": "full"})})
_LIN = make_describe(inp={"video": video_input(temporal={"rule": "index_linspace", "n": 4})})
_BAD_TIMES = lambda row, m: {"video": {"times": [0.0, 1.0, 2.0, 3.0 + 9.0 * (row % 3 == 0)]}}  # noqa: E731

#: (fixture kwargs, bench kwargs, validate kwargs): multi-shard sets with findings past the per-code cap, waivers (describe
#: and operator), canaries, missing and twice-written rows, constant and per-sample pos, eligibility, unchecked rules.
_PARALLEL_CASES: dict[str, tuple[dict, dict, dict]] = {
    "clean_missing_canaries": (dict(n_rows=48, n_shards=4, missing=(3, 17, 30, 47), canaries=1), dict(n_rows={"train": 48}), {}),
    "capped_findings": (dict(n_rows=96, n_shards=4, feat_fn=_nan_every_fifth, used_fn=_BAD_TIMES), dict(n_rows={"train": 96}), {}),
    "waived_capped": (dict(n_rows=96, n_shards=5, used_fn=_BAD_TIMES, describe=make_describe(waive={"E_RULE_MISMATCH": "fixed window"})),
                      dict(n_rows={"train": 96}), dict(waive={"E_TIMES_OUT_OF_MEDIA": "reads past the end"})),
    "canary_drift_f32": (dict(n_rows=48, n_shards=3, canaries=2, canary_noise=1e-7, dtype="float32", describe=make_describe(dtype="float32")),
                         dict(n_rows={"train": 48}), {}),
    "canary_errors": (dict(n_rows=48, n_shards=3, canaries=2, canary_noise=0.1,
                           canary_used_fn=lambda row, used: {"video": {"times": [t + 0.04 for t in used["video"]["times"]]}}),
                      dict(n_rows={"train": 48}), {}),
    "duplicates_constant": (dict(n_rows=40, n_shards=4, feat_fn=lambda row, cid, N, D: default_feat(row % 3, "x", N, D) * (row % 7 != 0)),
                            dict(n_rows={"train": 40}), dict(sample_rows=17)),
    "dead_rows_waived_star": (dict(n_rows=48, n_shards=4, feat_fn=_feat_dead, describe=make_describe(waive={"*": "register tokens"})),
                              dict(n_rows={"train": 48}), {}),
    "fp16_range": (dict(n_rows=36, n_shards=3, feat_fn=lambda row, cid, N, D: default_feat(row, cid, N, D) * (1000.0 + row * 37.0)),
                   dict(n_rows={"train": 36}), {}),
    "rows_twice_and_missing": (dict(n_rows=32, n_shards=4, duplicate_rows=(5, 21), missing=(9,)), dict(n_rows={"train": 32}), {}),
    "const_pos_differs": (dict(n_rows=32, n_shards=4, pos=lambda shard, row: grid_pos(1, 2, 2) * (1.0 if shard % 2 == 0 else 0.5)),
                          dict(n_rows={"train": 32}), {}),
    "seq_per_sample_pos": (dict(n_rows=48, n_shards=4, N=8, pos_mode="per_sample", pos=lambda shard, row: grid_pos(8, 1, 1), describe=_U8,
                                media_fn=_HALF_SHORT, used_fn=_uniform_used(8)), dict(n_rows={"train": 48}, **SEQ_BENCH), dict(sample_rows=20)),
    "grid_per_sample_pos": (dict(n_rows=40, n_shards=4, N=16, pos_mode="per_sample", pos=_per_row_grid,
                                 describe=make_describe(pos_mode="per_sample", streams=("image",), inp=IMG_DESC["input"]),
                                 media_fn=lambda row: {"image": dict(IMG_MEDIA)}, used_fn=lambda row, m: {"image": {}}),
                            dict(GRID_BENCH, n_rows={"train": 40}), {}),
    "unchecked_rule": (dict(n_rows=48, n_shards=4, describe=_LIN, pos=grid_pos(4, 1, 1, t_centres=[0.0, 0.332, 0.664, 0.996]),
                            media_fn=lambda row: {"video": dict(VIDEO_MEDIA, n_frames=None) if row % 3 else dict(VIDEO_MEDIA)},
                            used_fn=lambda row, m: {"video": {"times": [0.0, 3.32, 6.64, 9.96]}}), dict(n_rows={"train": 48}), dict(verify_sha=True)),
}


@pytest.mark.parametrize("case", sorted(_PARALLEL_CASES))
def test_parallel_scan_report_equals_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """Shards scanned by 4 threads give the report of the one-shard-at-a-time scan, byte for byte: every scan runs on
    private state and is merged in shard order (findings replayed through the caps and waivers, stats folded)."""
    import threading

    from yavih.io import validate as v

    spec_kw, bench_kw, val_kw = _PARALLEL_CASES[case]
    split_dir = tmp_path / "run" / "features" / "train"
    write_features(split_dir, **spec_kw)
    threads: list[str] = []
    real = v._scan_shard_isolated

    def spy(sh, static):
        threads.append(threading.current_thread().name)
        return real(sh, static)

    monkeypatch.setattr(v, "_scan_shard_isolated", spy)
    reports = {}
    for workers in (1, 4):
        monkeypatch.setenv("YAVIH_VALIDATE_WORKERS", str(workers))
        threads.clear()
        reports[workers] = json.dumps(validate_features(split_dir, bench=FakeBench(**bench_kw), write_meta=False, **val_kw).to_json(), sort_keys=True)
        assert threads and all(t.startswith("yavih-validate") == (workers > 1) for t in threads)
    assert reports[1] == reports[4]
    rep = json.loads(reports[4])
    assert rep["coverage"]["present"] > 0


def test_scan_shard_isolated_is_picklable(split_dir: Path) -> None:
    """A shard scan's inputs and result pickle (a process pool could run it), and the result folds into the split's
    accumulators exactly like a scan that never left the process."""
    import pickle

    from yavih.io import validate as v

    write_features(split_dir, n_rows=24, n_shards=2, canaries=1, pos_mode="per_sample", pos=_per_row_grid,
                   describe=make_describe(pos_mode="per_sample"))
    sidecars = sorted(split_dir.glob("shard-*.json"))
    shards = [v._Shard(index=i, sidecar_path=p, sidecar=json.loads(p.read_text()), tar_path=p.with_suffix(".tar")) for i, p in enumerate(sidecars)]
    describe = shards[0].sidecar["bridge"]["describe"]
    static = v._ScanStatic(N=4, D=8, dtype=np.dtype("float16"), pos_mode="per_sample", describe=describe, kinds={"video": "video"},
                           verify_sha=True, sampled=frozenset(range(0, 24, 2)), elig=v._EligSpec(grid_hw=(2, 2)))
    assert pickle.loads(pickle.dumps(static)) == static
    scan = v._scan_shard_isolated(shards[1], static)
    back = pickle.loads(pickle.dumps(scan))
    assert back.index == 1 and back.findings == scan.findings and back.pos_blob == scan.pos_blob

    def folded(s):
        fx, st, blobs, unchecked = v._Findings(), v._Stats(4, 8, set(static.sampled), static.elig), {}, {}
        v._merge_scan(s, fx, st, blobs, unchecked)
        return ([f.to_json() for f in fx.errors + fx.warnings + fx.notes], st.present_rows, st.clip_ids, st.max_abs, st.n_sampled_seen,
                st.sum.tolist(), st.sumsq.tolist(), st.dup_hash, st.cell_cov, sorted(st.pos_axes), st.canary_pairs, blobs, unchecked)

    assert folded(back) == folded(scan)
    assert folded(scan)[4] == len([r for r in range(12, 24) if r % 2 == 0])


def test_validate_workers_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from yavih import resources
    from yavih.io.validate import VALIDATE_WORKERS_ENV, validate_workers

    sh = lambda nbytes: SimpleNamespace(sidecar={"tar": {"bytes": nbytes}})  # noqa: E731
    monkeypatch.setattr(resources, "available_cpus", lambda env=None: 8)
    monkeypatch.setattr(resources, "ram_budget_bytes", lambda env=None: 10 * 2**30)
    assert validate_workers([sh(1)] * 3, env={}) == 3  # min(n_shards, cpus)
    assert validate_workers([sh(1)] * 20, env={}) == 8
    assert validate_workers([sh(4 * 2**30)] * 20, env={}) == 2  # RAM budget // largest shard tar
    assert validate_workers([], env={}) == 1
    assert validate_workers([sh(1)] * 20, env={VALIDATE_WORKERS_ENV: "1"}) == 1
    assert validate_workers([sh(1)] * 2, env={VALIDATE_WORKERS_ENV: "6"}) == 6
    for bad in ("0", "many"):
        with pytest.raises(ValueError, match=VALIDATE_WORKERS_ENV):
            validate_workers([sh(1)], env={VALIDATE_WORKERS_ENV: bad})


def test_finalize_rewrites_only_when_changed(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    meta = {"format_version": 4, "finalized": "t1", "x": 1, "validation": {"ok": True, "validated": "t1"}}
    p = finalize(d, meta)
    first = p.stat().st_mtime_ns
    time.sleep(0.01)
    finalize(d, {"format_version": 4, "finalized": "t2", "x": 1, "validation": {"ok": True, "validated": "t2"}})
    assert p.stat().st_mtime_ns == first  # timestamps are volatile
    finalize(d, {"format_version": 4, "finalized": "t3", "x": 2, "validation": {"ok": True, "validated": "t3"}})
    assert p.stat().st_mtime_ns != first
    assert read_meta(d)["x"] == 2
    assert not list(d.glob("*.tmp"))


def test_read_meta_refuses_newer_format(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "_meta.json").write_text(json.dumps({"format_version": 5}))
    with pytest.raises(FormatError):
        read_meta(d)
    (d / "_meta.json").write_text(json.dumps({"format_version": "1"}))
    with pytest.raises(FormatError):
        read_meta(d)
    (d / "_meta.json").write_text(json.dumps([1]))
    with pytest.raises(FormatError):
        read_meta(d)


@pytest.mark.parametrize("old", [FEATURE_FORMAT_VERSION - 1])
def test_older_format_meta_is_never_reused(split_dir: Path, old: int) -> None:
    write_features(split_dir)
    assert validate_features(split_dir, bench=FakeBench()).ok
    meta = read_meta(split_dir)
    meta["format_version"] = old
    (split_dir / "_meta.json").write_text(json.dumps(meta))
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok and "reused_meta" not in rep.shards and read_meta(split_dir)["format_version"] == FEATURE_FORMAT_VERSION


def test_build_meta_from_sidecars_alone(split_dir: Path) -> None:
    write_features(split_dir, canaries=1, missing=(2,))
    sidecars = [json.loads((split_dir / f"shard-{i:06d}.json").read_text()) for i in (1, 0)]  # unsorted on purpose
    meta = build_meta(split_dir, sidecars, eligibility={"readout": "clf", "eligible": True}, validation={"ok": True})
    assert [s["file"] for s in meta["shards"]] == ["shard-000000.tar", "shard-000001.tar"]
    assert meta["coverage"] == {"nominal": 24, "present": 23, "missing": 1, "ids_sha256": meta["coverage"]["ids_sha256"]}
    assert meta["canary"] == {"max_rel_l2": 0.0, "n": 2, "tolerance": 1e-5}
    assert meta["rules_checked"] == {} and "decoders" not in meta
    assert meta["features"]["bytes"] == 23 * 4 * 8 * 2
    assert meta["fingerprint"] == meta_fingerprint(sidecars)
    with pytest.raises(ValueError):
        build_meta(split_dir, [], eligibility={}, validation={})


def test_meta_passes_the_benchmark_block_through(split_dir: Path) -> None:
    """External-package pins or the media-info census digest, when the writer records them in the sidecar's
    ``benchmark`` block, reach ``_meta.json`` unchanged (validate never imports external packages)."""
    write_features(split_dir)
    pins = [{"name": "vggsounder", "pip": "vggsounder==1.0", "version_pin": "1.0", "used_for": "auc"}]
    for i in range(2):
        edit_sidecar(split_dir, i, lambda sc: sc["benchmark"].update(external=pins, media_info_sha256="c" * 64))
    assert validate_features(split_dir, bench=FakeBench()).ok
    b = read_meta(split_dir)["benchmark"]
    assert b["external"] == pins and b["media_info_sha256"] == "c" * 64


# ============================================================================= inspect / diff / stage


def test_inspect_features(split_dir: Path) -> None:
    def feat(row, cid, N, D):
        f = default_feat(row, cid, N, D)
        f[:, 0] = 0.0
        f[2] = 1.0
        return f

    write_features(split_dir, feat_fn=feat)
    info = inspect_features(split_dir)
    assert (info["N"], info["D"], info["dtype"], info["pos_mode"], info["n_present"], info["n_sampled"]) == (4, 8, "float16", "constant", 24, 24)
    assert info["near_constant_dims"]["indices_first50"] == [0]
    assert info["dead_rows"]["positions_first50"] == [2]
    assert len(info["per_dim_std"]["hist_counts"]) == 16 and info["mean_abs"] > 0
    assert info["pos"]["mode"] == "constant" and info["pos"]["n_tokens"] == 4 and info["pos"]["stream_counts"] == {"video": 4}
    json.dumps(info)
    assert inspect_features(split_dir, sample_rows=4)["n_sampled"] == 4
    with pytest.raises(FileNotFoundError):
        inspect_features(split_dir.parent / "nothing_here")


def test_diff_features(tmp_path: Path) -> None:
    a = tmp_path / "a" / "features" / "train"
    b = tmp_path / "b" / "features" / "train"
    f32 = dict(dtype="float32", describe=make_describe(dtype="float32"))
    write_features(a, **f32)
    write_features(b, n_shards=3, **f32)
    d = diff_features(a, b)
    assert d["verdict"] == "identical" and d["rows_compared"] == 24 and d["max_abs"] == 0.0 and d["max_rel_l2"] == 0.0
    write_features(b, feat_fn=lambda row, cid, N, D: default_feat(row, cid, N, D) * (1 + 1e-5), missing=(3,), **f32)
    d = diff_features(a, b)
    assert d["verdict"] == "within_tol" and d["rows_compared"] == 23 and d["only_in_a"] == 1 and 0 < d["max_rel_l2"] < 1e-3
    write_features(b, feat_fn=lambda row, cid, N, D: default_feat(row, cid, N, D) * 1.5, **f32)
    d = diff_features(a, b, rel_tol=0.1)
    assert d["verdict"] == "differs" and d["worst_row"] is not None
    write_features(b, D=9, **f32)
    assert diff_features(a, b)["verdict"] == "incompatible"
    write_features(b, split="train", missing=tuple(range(24)), **f32)
    assert diff_features(a, b)["verdict"] == "empty"


def test_stage_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run" / "features"
    write_features(root / "train", split="train")
    write_features(root / "val", split="val")
    validate_features(root / "train", bench=FakeBench())
    (root / "train" / "shard-000009.tar.tmp").write_bytes(b"junk")
    scratch = tmp_path / "scratch"
    staged = stage_features(root, scratch, unit_id="abc123def456")
    assert staged.parent.name.startswith("abc123def456.") and staged.name == "features"
    assert sorted(p.name for p in (staged / "train").iterdir()) == ["_meta.json", "shard-000000.json", "shard-000000.tar", "shard-000001.json", "shard-000001.tar"]
    assert (staged / "val" / "shard-000001.tar").read_bytes() == (root / "val" / "shard-000001.tar").read_bytes()
    assert validate_features(staged / "train", bench=FakeBench(), write_meta=False).ok
    staged2 = stage_features(root / "val", scratch)
    assert staged2.parent.name.startswith("run.") and (staged2 / "val" / "shard-000000.json").exists()

    # threaded copy: the same tree, byte for byte, copied by up to STAGE_THREADS threads; no tmp left behind
    import shutil as _shutil
    import threading

    from yavih.io import stage as stage_mod

    names: set[str] = set()
    real_copy = _shutil.copyfile

    def tracking_copy(a, b):
        names.add(threading.current_thread().name)
        return real_copy(a, b)

    monkeypatch.setattr("yavih.io.stage.shutil.copyfile", tracking_copy)
    staged3 = stage_features(root, tmp_path / "scratch3", unit_id="t")
    src_files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    assert sorted(p.relative_to(staged3) for p in staged3.rglob("*") if p.is_file()) == src_files
    assert all((staged3 / rel).read_bytes() == (root / rel).read_bytes() for rel in src_files)
    assert names and all(n.startswith("yavih-stage") for n in names) and stage_mod.STAGE_THREADS == 4

    # a stop request before a later file: E_PREEMPTED, every copy in flight finished, no tmp left
    polls = {"n": 0}

    def stop_after_two() -> bool:
        polls["n"] += 1
        return polls["n"] > 2

    with pytest.raises(StageError) as ei:
        stage_features(root, tmp_path / "scratch4", unit_id="s", should_stop=stop_after_two)
    assert ei.value.code == "E_PREEMPTED"
    assert not list((tmp_path / "scratch4").rglob("*.tmp"))

    # a failing copy is raised (the first failure in file order); files not started are not copied
    def failing_copy(a, b):
        if Path(a).name == "shard-000000.tar":
            raise OSError("disk on fire")
        return real_copy(a, b)

    monkeypatch.setattr("yavih.io.stage.shutil.copyfile", failing_copy)
    with pytest.raises(OSError, match="disk on fire"):
        stage_features(root, tmp_path / "scratch5", unit_id="f")
    assert not list((tmp_path / "scratch5").rglob("*.tmp"))

    monkeypatch.setattr("yavih.io.stage.shutil.disk_usage", lambda p: SimpleNamespace(free=10, total=10, used=0))
    with pytest.raises(StageError) as ei:
        stage_features(root, scratch)
    assert ei.value.exit_code == 3 and ei.value.code == "E_SCRATCH_SPACE"


def test_cli_features_validate_json(split_dir: Path) -> None:
    import contextlib
    import io as _io

    from yavih import cli

    write_features(split_dir)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["--json", "features", "validate", str(split_dir)])
    env = json.loads(buf.getvalue())
    assert rc == 0 and env["status"] == "ok" and env["data"]["ok"] is True and env["data"]["missing_shards"] == []
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["--json", "features", "inspect", str(split_dir)])
    assert rc == 0 and json.loads(buf.getvalue())["data"]["N"] == 4
    edit_sidecar(split_dir, 0, lambda sc: sc["features"].__setitem__("N", 7))
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["--json", "features", "validate", str(split_dir)])
    env = json.loads(buf.getvalue())
    assert rc == 2 and env["status"] == "error" and {e["code"] for e in env["errors"]} >= {"E_DESCRIBE_MISMATCH"}


# ----------------------------------------------------------------------------- the REAL writer


def test_real_feature_writer_validates_and_member_nbytes_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A split written by the real ``FeatureWriter`` with member provenance (``media`` + ``used``) passes
    validate with ``_meta.json`` written; a wrong sidecar ``members[].nbytes`` is reported."""
    from yavih.io.writer import FeatureWriter

    describe = make_describe(dtype="float32", batch_size=4, pos_mode="constant")
    bridge = SimpleNamespace(name="synth", dtype="float32", pos_mode="constant", batch_size=4, args={"layer": 12})
    bench = FakeBench(n_rows={"train": 23}, media_fingerprint="mfp")
    bench.spec.protocol_version = 2
    split_dir = tmp_path / "run" / "features" / "train"
    from yavih.io.shards import plan_shards

    for i, (lo, hi) in enumerate(plan_shards(23, 4, 4)):
        w = FeatureWriter(tmp_path / "run", "train", shard=(i, 4), rows=(lo, hi), bench=bench, bridge=bridge, describe=describe, source_sha256="s" * 64)
        for b in range(lo, hi, 4):
            rows = [r for r in range(b, min(b + 4, hi)) if r != 5]
            if 5 in range(b, min(b + 4, hi)):
                w.record_missing(5, "train_0005", "MediaMissing: video")
            if not rows:
                continue
            media = {"video": dict(VIDEO_MEDIA)}
            meta = [{"media_key": None, "media": media, "used": expected_used(describe, media)} for _ in rows]
            feats = np.stack([default_feat(r, f"train_{r:04d}", 4, 8).astype(np.float32) for r in rows])
            w.write(rows, [f"train_{r:04d}" for r in rows], feats, grid_pos(1, 2, 2), meta=meta)
        close = w.close
        import inspect as _inspect

        params = _inspect.signature(close).parameters
        kwargs = {"timing": {}, "env": {}, "canary_stats": {"max_rel_l2": 0.0, "max_abs_diff": 0.0, "n": 0, "tolerance": 1e-5}}
        if "side_effects" in params:
            kwargs["side_effects"] = {"writes_outside_run_dir": []}
        if "decoders" in params:
            kwargs["decoders"] = {}
        close(**kwargs)
    rep = validate_features(split_dir, bench=bench)
    assert rep.ok, [f.to_json() for f in rep.errors]
    assert rep.coverage["present"] == 22 and rep.coverage["nominal"] == 23
    meta = read_meta(split_dir)
    assert meta["coverage"]["present"] == 22 and meta["validation"]["ok"] is True and meta["fingerprint"]
    edit_sidecar(split_dir, 0, lambda sc: sc["members"][0].__setitem__("nbytes", 1) if sc["members"] else None)
    edit_sidecar(split_dir, 1, lambda sc: sc["members"][0].__setitem__("nbytes", 1))
    (split_dir / "_meta.json").unlink()
    rep = validate_features(split_dir, bench=bench)
    assert any(f.code == "E_MEMBERS" and "nbytes" in f.message for f in rep.errors), codes(rep)


# ----------------------------------------------------------------------------- pos t, unchecked rows, fp16 canaries


def test_pos_t_must_lie_in_the_span_the_bridge_consumed(split_dir: Path) -> None:
    """t is a fraction of the MEDIA duration. A bridge reading [1, 9] s of a 10 s clip that writes
    grid_pos's default window-relative t = (k + 0.5)/T passes every rule check yet misplaces every token."""
    span = {"rule": "uniform", "n": 4, "span": [1.0, 9.0]}
    desc = make_describe(inp={"video": video_input(temporal=span)})
    used = lambda row, m: {"video": {"times": [round(t * 25) / 25 for t in rule_times(span, m["video"])]}}  # noqa: E731
    write_features(split_dir, describe=desc, used_fn=used, pos=grid_pos(4, 1, 1))  # window-relative t: 0.125 … 0.875
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert any(f.code == "E_RULE_MISMATCH" and "pos t of" in f.message and "outside the media span" in f.message for f in rep.errors)
    # media coordinates: the used times divided by duration_s
    tc = [t / DUR for t in rule_times(span, VIDEO_MEDIA)]
    write_features(split_dir, describe=desc, used_fn=used, pos=grid_pos(4, 1, 1, t_centres=tc))
    assert validate_features(split_dir, bench=FakeBench(), write_meta=False).ok


def test_pos_t_audio_span_and_image_tokens(split_dir: Path) -> None:
    audio_in = {"temporal": {"rule": "span", "start_s": 2.0, "length_s": 4.0, "pad": "zeros"},
                "front_end": {"sample_rate": 16000, "resample": "none", "features": "fbank"}, "norm": dict(NORM), "decoder": dict(DECODER)}
    desc = make_describe(streams=("audio",), inp={"audio": audio_in})
    media = lambda row: {"audio": dict(AUDIO_MEDIA)}  # noqa: E731
    used = lambda row, m: {"audio": {"samples": [[32000, 96000]], "pad": [0, 0]}}  # noqa: E731
    write_features(split_dir, describe=desc, media_fn=media, used_fn=used, pos=grid_pos(4, 1, 1, t_centres=[0.25, 0.35, 0.45, 0.55]))
    assert validate_features(split_dir, write_meta=False).ok
    write_features(split_dir, describe=desc, media_fn=media, used_fn=used, pos=grid_pos(4, 1, 1))  # 1.25 s … 8.75 s: outside [2, 6]
    assert has(validate_features(split_dir, write_meta=False), "E_RULE_MISMATCH")
    img = make_describe(streams=("image",), inp={"image": {"spatial": dict(FULL_FRAME), "norm": dict(NORM), "decoder": dict(DECODER)}})
    write_features(split_dir, describe=img, media_fn=lambda row: {"image": dict(IMG_MEDIA)}, used_fn=lambda row, m: {"image": {}}, pos=grid_pos(1, 2, 2))
    rep = validate_features(split_dir, write_meta=False)
    assert any(f.code == "E_RULE_MISMATCH" and "image has no time axis" in f.message for f in rep.errors)


def test_unevaluable_rule_is_counted_as_unchecked(split_dir: Path) -> None:
    """A checkable rule whose fact is null on a row is not silently 'checked' (W_RULE_UNCHECKED + rules line)."""
    lin = make_describe(inp={"video": video_input(temporal={"rule": "index_linspace", "n": 4})})
    no_nf = lambda row: {"video": dict(VIDEO_MEDIA, n_frames=None) if row % 2 else dict(VIDEO_MEDIA)}  # noqa: E731
    used = lambda row, m: {"video": {"times": [0.0, 3.32, 6.64, 9.96]}}  # noqa: E731
    write_features(split_dir, describe=lin, media_fn=no_nf, used_fn=used, pos=grid_pos(4, 1, 1, t_centres=[0.0, 0.332, 0.664, 0.996]))
    rep = validate_features(split_dir, bench=FakeBench(), write_meta=False)
    assert rep.ok, rep.errors
    w = [f for f in rep.warnings if f.code == "W_RULE_UNCHECKED"]
    assert w and "12 row(s)" in w[0].message
    assert rep.rules_checked["video"]["temporal_unchecked_rows"] == 12
    assert "index_linspace (unchecked on 12 rows)" in rules_line(rep.rules_checked)
    assert not any(f.code == "N_RULE_CUSTOM" for f in rep.notes)  # a checkable rule is not an opt-out


def test_custom_audio_next_to_a_checked_video_rule(split_dir: Path) -> None:
    """Opting out per stream; the other stream's rule is still checked; the note names the stream."""
    av = make_describe(streams=("video",), inp={"video": video_input(), "audio": audio_input(temporal={"rule": "custom", "description": "streams the whole file in 1 s chunks"})})
    used = lambda row, m: {**expected_used(av, m), "audio": {"samples": [[0, 16000]], "pad": [0, 0]}}  # noqa: E731
    write_features(split_dir, describe=av, used_fn=used)
    rep = validate_features(split_dir, bench=FakeBench(streams={"video": "video", "audio": "audio"}), write_meta=False)
    assert rep.ok, rep.errors
    assert codes(rep)["warnings"] == []
    notes = [f for f in rep.notes if f.code == "N_RULE_CUSTOM"]
    assert len(notes) == 1 and notes[0].message.startswith("audio: temporal rule custom — streams the whole file")
    assert rep.rules_checked == {"video": {"temporal": "uniform", "spatial": True}, "audio": {"temporal": "custom", "spatial": None}}
    assert rules_line(rep.rules_checked, rep.describe) == "rules checked: video uniform + spatial; audio custom (opted out: streams the whole file in 1 s chunks)"
    # and an out-of-media audio range under the custom rule still fails
    write_features(split_dir, describe=av, used_fn=lambda row, m: {**expected_used(av, m), "audio": {"samples": [[0, 10**9]], "pad": [0, 0]}})
    assert has(validate_features(split_dir, bench=FakeBench(streams={"video": "video", "audio": "audio"}), write_meta=False), "E_TIMES_OUT_OF_MEDIA")


def test_rel_l2_stored_ignores_one_fp16_ulp() -> None:
    from yavih.io.validate import rel_l2_stored

    rng = np.random.default_rng(0)
    o32 = rng.standard_normal((196, 768)).astype(np.float32)
    c32 = (o32.astype(np.float64) * (1 + 1e-6)).astype(np.float32)
    o16, c16 = o32.astype(np.float16), c32.astype(np.float16)
    plain = float(np.linalg.norm(c16.astype(np.float64) - o16.astype(np.float64)) / np.linalg.norm(o16.astype(np.float64)))
    assert plain > 1e-5  # one-ulp flips alone exceed the default tolerance
    assert rel_l2_stored(c16, o16) == 0.0
    big = (o32.astype(np.float64) * (1 + 1e-2)).astype(np.float16)  # real batch dependence: many ulps
    assert rel_l2_stored(big, o16) > 1e-3
    assert rel_l2_stored(c32, o32) == pytest.approx(1e-6, rel=1e-2)


# ----------------------------------------------------------------------------- supersedes_input (folder changelog)

OLD_IH = hashlib.sha256(b"an earlier input side").hexdigest()


@pytest.fixture
def synth_history():
    """A synthetic 'synth' changelog: the current hashes, and an earlier input hash whose features stay valid."""
    from yavih import hash_history

    hash_history.register("synth", (), [{"version": 2, "protocol_hash": PH, "input_hash": IH, "supersedes_input": [OLD_IH]}])
    yield
    hash_history.reset()


def test_sidecar_with_a_superseded_input_hash_is_accepted(split_dir: Path, synth_history) -> None:
    """Features whose recorded input_hash the folder changelog lists under ``supersedes_input`` stay valid."""
    write_features(split_dir, input_hash=OLD_IH)
    rep = validate_features(split_dir, bench=FakeBench())
    assert rep.ok, rep.errors
    assert read_meta(split_dir)["benchmark"]["input_hash"] == OLD_IH  # recorded as written, never rewritten
    assert validate_features(split_dir, bench=FakeBench()).shards.get("reused_meta") is True


def test_sidecar_with_an_unknown_input_hash_is_refused(split_dir: Path, synth_history) -> None:
    write_features(split_dir, input_hash="9" * 64)
    rep = validate_features(split_dir, bench=FakeBench())
    assert not rep.ok and codes(rep)["errors"].count("E_HASH_INPUT") == 2


def test_superseded_and_current_input_hashes_agree_across_shards(split_dir: Path, synth_history) -> None:
    write_features(split_dir, input_hash=OLD_IH)
    edit_sidecar(split_dir, 1, lambda sc: sc["benchmark"].update({"input_hash": IH}))
    assert validate_features(split_dir, bench=FakeBench()).ok
    assert "E_HASH_INPUT" not in codes(validate_features(split_dir, bench=None, write_meta=False))["errors"]
