"""Shared builders for the plan tests (no tests here; the name keeps it under tests/test_plan*.py).

Everything is synthetic: a fake ``BenchmarkInfo`` installed through the ``yavih.plan.units.resolve_benchmark``
hook (``Benchmark.load`` and ``ProbeConfig.pinned`` are not needed), a ``profile.json`` consistent
with ``yavih.hashing``, and a tiny python script standing in for ``yavih.extract`` so ``exec`` can be
exercised with only stdlib in the child.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

from yavih import hashing
from yavih.plan import units as U
from yavih.plan.units import BenchmarkInfo, Unit

PROTO = "a" * 64
PROTO2 = "b" * 64
QH = "c" * 64
MANI = "d" * 64


def make_describe(**over: Any) -> dict:
    d: dict[str, Any] = {
        "weights_sha256": "e" * 64,
        "weights_ref": "/weights/model.pt",
        "dtype": "float16",
        "batch_size": 4,
        "streams": ["video"],
        "pos_mode": "constant",
        "model": "synthmodel",
        "reading": "v",
        "layer": 12,
        "pooling": "none",
        "views": 1,
        "input": {"video": {"temporal": {"rule": "uniform", "n": 16, "span": "full"},
                            "spatial": {"resize": {"short": 224}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True},
                            "norm": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "source": "synth/recipe@0 transforms.py"},
                            "decoder": {"lib": "pyav", "version": "14.3.0"}}},
        "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
        "args": {"layer": 12},
    }
    d.update(over)
    return d


def make_profile(bench: str = "synth_clf", *, model: str = "synthmodel", reading: str = "v", args: dict | None = None, N: int = 16,
                 D: int = 32, pos_mode: str = "constant", streams: list[str] | None = None, clips_per_s: float = 2.0,
                 describe_over: dict | None = None) -> dict:
    args = {"layer": 12} if args is None else args
    streams = ["video"] if streams is None else streams
    d = make_describe(args=dict(args), pos_mode=pos_mode, streams=list(streams), model=model, reading=reading,
                      **(describe_over or {}))
    return {
        "bench": bench,
        "args": dict(args),
        "describe": d,
        "describe_sha256": hashing.describe_sha256(d),
        "model": d["model"],  # names copied from describe, never computed
        "reading": d["reading"],
        "N": N,
        "D": D,
        "pos_mode": pos_mode,
        "streams": list(streams),
        # the hard checks bridge check passed
        "invariance": {"rerun_bit_identical": True, "batch_rel_l2": 0.0, "order_rel_l2": 0.0, "tolerance": 1e-5, "tolerance_reason": None},
        "rules_checked": {"video": {"temporal": "uniform", "spatial": True}},
        "clips_per_s": clips_per_s,
        "decode_workers": 3,
        "peak_gpu_gb": 1.5,
        "yavih_version": "0.1.0",
        "created": "2026-10-07T00:00:00Z",
        "warnings": [],
        "notes": [],
        "used": {"video": {"times": [0.3]}},  # extra bridge-check keys are allowed
    }


def make_info(name: str = "synth_clf", *, readout: str = "clf", n_train: int = 12, n_eval: int = 6,
              requires_pos: tuple[str, ...] = (), requires_streams: tuple[str, ...] = (),
              protocol_hash: str = PROTO, probe_hash: str = QH, probe_minutes_ref: float = 1.0,
              aliases: tuple[str, ...] = ()) -> BenchmarkInfo:
    return BenchmarkInfo(
        name=name, readout=readout, protocol_hash=protocol_hash, manifest_hash=MANI,
        roles={"train": ["train"], "eval": ["val"], "floor": [], "folds": []},
        n_rows={"train": n_train, "val": n_eval},
        requires_pos=requires_pos, requires_streams=requires_streams, stream_names=("video",),
        probe_minutes_ref=probe_minutes_ref, probe_hash=probe_hash, names=(name, *aliases),
    )


def install_benchmarks(monkeypatch, infos: dict[str, BenchmarkInfo]) -> None:
    by_name: dict[str, BenchmarkInfo] = {}
    for info in infos.values():
        for n in info.names or (info.name,):
            by_name[n] = info

    def fake_resolve(ref: str) -> BenchmarkInfo:
        if ref not in by_name:
            from yavih.benchmark import UnknownBenchmark

            raise UnknownBenchmark(ref)
        return by_name[ref]

    monkeypatch.setattr(U, "resolve_benchmark", fake_resolve)


def write_matrix(root: Path, entries: dict[str, dict], *, out: str = "runs", seeds: list[int] | None = None,
                 shard_minutes: int = 30, canaries: int = 8, defaults: dict | None = None,
                 extra_top: dict | None = None, name: str = "matrix.yaml") -> Path:
    m: dict[str, Any] = {"schema_version": 1, "out": out, "seeds": [0] if seeds is None else seeds,
                         "shard_minutes": shard_minutes, "canaries": canaries, "bridges": entries}
    if defaults is not None:
        m["defaults"] = defaults
    if extra_top:
        m.update(extra_top)
    p = root / name
    p.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
    return p


def write_profile(root: Path, rel: str, profile: dict) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    return p


def standard_matrix(tmp_path: Path, monkeypatch, *, seeds: list[int] | None = None, shard_minutes: int = 30,
                    clips_per_s: float = 2.0, n_train: int = 12, n_eval: int = 6, launcher: str | None = None,
                    extra_entries: dict | None = None, infos: dict[str, BenchmarkInfo] | None = None) -> Path:
    """One clf entry over the synthetic benchmark; returns the matrix path (matrix dir = tmp_path/m)."""
    root = tmp_path / "m"
    root.mkdir(exist_ok=True)
    infos = infos or {"synth_clf": make_info(n_train=n_train, n_eval=n_eval)}
    install_benchmarks(monkeypatch, infos)
    write_profile(root, "bridges/synth.profile.json", make_profile("synth_clf", clips_per_s=clips_per_s))
    entries = {"synth-clf": {"script": "bridges/synth.py", "benchmark": "synth_clf", "args": {"layer": 12},
                             "profile": "bridges/synth.profile.json"}}
    if extra_entries:
        entries.update(extra_entries)
    defaults = {"env": {"HF_HUB_OFFLINE": "1"}, "cpus": 4}
    if launcher:
        defaults["launcher"] = launcher
    return write_matrix(root, entries, seeds=seeds, shard_minutes=shard_minutes, defaults=defaults)


# ----------------------------------------------------------------------------- DONE markers


def write_done_extract(unit: Unit, *, protocol_hash: str | None = None, model: str | None = None,
                       reading: str | None = None, describe_sha256: str | None = None, input_hash: str | None = None) -> Path:
    """A sidecar: ``input_hash`` is the identity, ``protocol_hash`` provenance. Passing only
    ``protocol_hash`` writes it as the input hash too (a stand-in for "another benchmark state")."""
    p = Path(U.resolve_path(unit, unit.done))
    p.parent.mkdir(parents=True, exist_ok=True)
    ih = input_hash or protocol_hash or unit.input_hash or unit.protocol_hash
    rec = {"format_version": 4, "benchmark": {"input_hash": ih, "protocol_hash": protocol_hash or unit.protocol_hash, "manifest_hash": MANI},
           "model": model or unit.model, "reading": reading or unit.reading, "split": unit.split,
           "bridge": {"describe_sha256": describe_sha256 or unit.describe_sha256},
           "shard": {"index": unit.shard[0], "n": unit.shard[1], "rows": unit.extra.get("rows", [0, 0]), "batch_size": 4}}
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def write_done_probe(unit: Unit, *, status: str = "ok", probe_hash: str | None = None,
                     describe_sha256: str | None = None, model: str | None = None, reading: str | None = None) -> Path:
    """A probe DONE marker."""
    p = Path(U.resolve_path(unit, unit.done))
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.parent / "result.json").write_text(json.dumps({"schema_version": 3, "status": status, "timing": {"wall_s": 12.0}}), encoding="utf-8")
    p.write_text(json.dumps({"result_sha256": "0" * 64, "protocol_hash": unit.protocol_hash,
                             "probe_hash": probe_hash or unit.probe_hash, "eval_hash": unit.eval_hash,
                             "model": model or unit.model, "reading": reading or unit.reading,
                             "describe_sha256": describe_sha256 or unit.describe_sha256,
                             "features_fingerprint": None, "yavih_version": "0.1.0", "created": "now"}), encoding="utf-8")
    return p


def write_last(unit: Unit, **over: Any) -> Path:
    logs = Path(U.resolve_path(unit, unit.out)) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rec = {"id": unit.id, "kind": unit.kind, "status": "done", "exit_code": 0, "wall_s": 1.0, "host": "h", "gpu": None,
           "slurm_job_id": None, "inputs_hash": unit.inputs_hash, "inputs_scheme": U.INPUTS_SCHEME, "needs": dict(unit.needs),
           "started": "2026-10-07T00:00:00Z", "ended": "2026-10-07T00:00:01Z"}
    rec.update(over)
    p = logs / f"last.{unit.id}.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- fake extract child


FAKE_SCRIPT = textwrap.dedent(
    '''
    """Stand-in for `python -m yavih.extract` (stdlib only).

    argv: mode done_path protocol_hash model/reading i n [sentinel]
    modes: ok | noop | exit1 | exit2 | exit3 | exit4 | sigterm | sleep | env
    """
    import json, os, signal, sys, time

    mode, done, ph, mr, i, n = sys.argv[1:7]
    model, reading = mr.split("/")
    sentinel = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != "-" else None

    def write_done():
        os.makedirs(os.path.dirname(done), exist_ok=True)
        rec = {"format_version": 4, "benchmark": {"input_hash": ph, "protocol_hash": ph}, "model": model, "reading": reading,
               "bridge": {"describe_sha256": "f" * 64}, "shard": {"index": int(i), "n": int(n)}}
        tmp = done + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, done)

    print("child start", mode, flush=True)
    if sentinel:
        with open(sentinel, "w") as fh:
            fh.write(str(os.getpid()))
    if mode == "ok":
        write_done()
        sys.exit(0)
    if mode == "env":
        write_done()
        print("UNIT=" + os.environ.get("YAVIH_UNIT_ID", "") + " DL=" + os.environ.get("YAVIH_DEADLINE", "-")
              + " HF=" + os.environ.get("HF_HUB_OFFLINE", "-") + " CVD=" + os.environ.get("CUDA_VISIBLE_DEVICES", "-")
              + " ARGV=" + " ".join(sys.argv[8:]), flush=True)
        sys.exit(0)
    if mode == "noop":
        sys.exit(0)
    if mode.startswith("exit"):
        sys.exit(int(mode[4:]))
    if mode == "sigterm":
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        sys.exit(0)
    if mode == "sleep":
        def on_term(signum, frame):
            print("child got signal", signum, flush=True)
            sys.exit(4)
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGUSR1, on_term)
        for _ in range(600):
            time.sleep(0.05)
        sys.exit(0)
    sys.exit(7)
    '''
)


def write_fake_script(root: Path) -> Path:
    p = root / "fake_extract.py"
    p.write_text(FAKE_SCRIPT, encoding="utf-8")
    return p


def fake_unit(root: Path, script: Path, *, mode: str = "ok", uid: str | None = None, kind: str = "extract",
              split: str = "train", shard: tuple[int, int] = (0, 1), depends_on: list[str] | None = None,
              launcher: str | None = None, env: dict | None = None, minutes: int = 10, sentinel: Path | None = None,
              extra_argv: list[str] | None = None, protocol_hash: str = PROTO, model: str = "fake", reading: str = "v",
              expected_status: str = "ok", probe_hash: str | None = None, seed: int | None = None) -> Unit:
    """A plan unit whose command is the fake script; paths are relative to ``root`` (the unit's cwd). Its recorded
    ``describe_sha256`` is ``"f" * 64`` (what the fake child writes into the sidecar)."""
    runs = "runs"
    run_dir = os.path.join(runs, "synth_clf", model, reading)
    if kind == "extract":
        out = os.path.join(run_dir, "features", split)
        done = os.path.join(out, f"shard-{shard[0]:06d}.json")
        cmd = ["python", str(script), mode, done, protocol_hash, f"{model}/{reading}", str(shard[0]), str(shard[1])]
        cmd.append(str(sentinel) if sentinel is not None else "-")  # argv[7]: sentinel slot, "-" = none
        cmd += extra_argv or []
        sh: list[int] | None = list(shard)
        sp: str | None = split
    elif kind == "probe":
        out = run_dir
        qh = probe_hash or QH
        done = os.path.join(run_dir, "probe", hashing.eval_hash(protocol_hash, qh)[:12], f"seed{seed or 0}", "DONE")
        cmd = ["python", "-m", "yavih.cli", "probe", "run", run_dir, "--seeds", str(seed or 0)]
        sh, sp = None, None
    else:
        out = os.path.join(runs, "_report")
        done = os.path.join(out, "DONE")
        cmd = ["python", "-m", "yavih.cli", "report", "build", os.path.join(runs, "*")]
        sh, sp = None, None
    uid = uid or hashing.unit_id(kind=kind, benchmark="synth_clf", model=model, reading=reading, split=sp, shard=sh,
                                 protocol_hash=protocol_hash, probe_hash=probe_hash if kind == "probe" else None,
                                 seed=seed)
    needs = {"gpu": 1 if kind != "report" else 0, "cpus": 4, "mem_gb": 32, "scratch_gb": 0, "minutes": minutes}
    qh_u = (probe_hash or QH) if kind == "probe" else None
    u = Unit(id=uid, kind=kind, benchmark="synth_clf", bridge="fake", model=model, reading=reading, split=sp, shard=sh,
             seed=seed, protocol_hash=protocol_hash, probe_hash=qh_u,
             eval_hash=hashing.eval_hash(protocol_hash, qh_u) if qh_u else None,
             depends_on=depends_on or [], needs=needs, cmd=cmd, launcher=launcher, env=env or {}, cwd=str(root),
             out=out, done=done, inputs_hash="", expected_status=expected_status,
             describe_sha256="f" * 64 if kind != "report" else None)
    u.inputs_hash = hashing.inputs_hash(cmd=U.identity_cmd(cmd), protocol_hash=protocol_hash, probe_hash=u.probe_hash,
                                        model=model, reading=reading, describe_sha256="f" * 64)
    return u


def python_launcher() -> str:
    """The test interpreter as a launcher (basename matches ``python*``)."""
    return sys.executable
