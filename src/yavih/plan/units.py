"""Unit records and ``plan()``; re-exports ``PLAN_SCHEMA_VERSION``.

Kinds: ``extract`` (one per ``(entry, split in roles, shard)``), ``probe`` (one per ``(entry, seed)``,
``depends_on`` = its extract ids), ``report`` (one). Plan-time ``expected_status: not_applicable``
when the profile already shows ineligibility against ``requires``. Units are written
topologically sorted and grouped by ``(kind, needs.gpu, needs.minutes bucket)``.

A split whose shard layout is frozen (``features/<split>/_layout.json``) is planned with exactly that layout;
``inputs_hash`` covers only what a unit computes (``identity_cmd``: no host paths, launcher, env, cwd, worker counts
or ``needs``); ``write_plan`` stores ``cwd`` relative to the plan file and ``read_plan(..., root=)`` re-anchors it
(``exec --root``); ``needs.gpu`` is 0 for a CPU-only profile (``peak_gpu_gb == 0``).

``sample=N`` (``plan build --sample N``) plans a capped run: per role split the first ``min(N, rows)`` manifest rows
(manifest order, not class-balanced) as ONE extract unit ``--rows 0:rows``, probe units with ``--no-results``, under the
separate runs root ``<out>-sample<N>`` (:func:`sample_root`), so a capped run never shares a run directory, a DONE
marker, an index or a report with the full run; unit ids are derived from the full-run ids and ``N``, so they never
collide with them.

Module scope is stdlib + ``yavih.versions`` + ``yavih.hashing`` (both stdlib). The
benchmark (``yavih.benchmark``, numpy) and the pinned ``ProbeConfig`` (``yavih.probes.config``,
torch-free) are reached through ``resolve_benchmark``, a module-level hook, so a controller env
without the probe extra can still plan and tests can plan against a synthetic ``BenchmarkInfo``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from yavih import hashing
from yavih.io.sidecar import identity_value
from yavih.plan import estimate
from yavih.plan.matrix import (
    Matrix,
    MatrixEntry,
    check_profile,
    is_generic_profile,
    load_matrix,
    load_profile,
)
from yavih.versions import PLAN_SCHEMA_VERSION, READOUTS

__all__ = [
    "Unit",
    "BenchmarkInfo",
    "PlanError",
    "plan",
    "build_units",
    "read_plan",
    "write_plan",
    "resolve_benchmark",
    "benchmark_info",
    "resolve_path",
    "identity_cmd",
    "INPUTS_SCHEME",
    "check_done",
    "last_record",
    "is_stale",
    "load_calibration",
    "PLAN_SCHEMA_VERSION",
    "UNIT_KINDS",
    "KIND_ORDER",
    "is_dense",
    "NOT_APPLICABLE_FILE",
    "REPORT_KEY",
    "W_NO_CENSUS",
    "uncensused_frame_streams",
    "sample_root",
]

#: ``plan build`` warning: a video stream guarantees ``n_frames`` and the site has no media-info census.
W_NO_CENSUS = "W_NO_CENSUS"

UNIT_KINDS: frozenset[str] = frozenset({"extract", "probe", "report"})
#: Topological layer of each kind (extract → probe → report).
KIND_ORDER: dict[str, int] = {"extract": 0, "probe": 1, "report": 2}
#: Readouts whose head needs token positions (``requires.pos``): ``N == 1`` or ``pos_mode: none``
#: in the profile rules them out at plan time.
def is_dense(readout: str) -> bool:
    """Whether the readout needs per-token media positions (``Readout.dense``, ``yavih.readouts.registry``)."""
    from yavih.readouts import registry

    try:
        return bool(registry.get(readout).dense)
    except registry.UnknownScorer:
        return False
NOT_APPLICABLE_FILE = "NOT_APPLICABLE.json"
#: Placeholder for the report unit's benchmark/bridge/model/reading fields (it spans every entry).
REPORT_KEY = "_report"
LOGS_DIR = "logs"
#: ``logs/last.json[inputs_scheme]``: the version of the ``inputs_hash`` scheme (over ``identity_cmd``). A record of
#: another scheme is never called stale on that ground alone.
INPUTS_SCHEME = 2
#: ``cmd`` flags whose values say where or with how many workers a unit runs, not what it computes.
#: ``--waive`` (an operator's validation waivers) is recorded in the cmd but changes nothing the unit computes on
#: valid features: it never makes a DONE stale.
_LOGGED_FLAGS = frozenset({"--out", "--bridge", "--expect-profile", "--results", "--decode-workers", "--waive"})
#: ``(group, command)`` whose first positional is a host path (``eval run <run_dir>``, ``report build <glob>``).
_PATH_POSITIONAL = (("eval", "run"), ("report", "build"))


class PlanError(ValueError):
    """A plan cannot be built or read (bad unit record, unknown kind, newer schema) — exit 1."""

    code = "E_PLAN"
    exit_code = 1


# ----------------------------------------------------------------------------- unit record


@dataclass
class Unit:
    """One line of ``plan.jsonl``.

    Three keys extend the basic unit record: ``profile``
    (the entry's ``profile.json`` path, which a plan-time ``not_applicable`` probe needs for
    ``bridge.describe``), ``not_applicable`` (``{"missing": [...], "reason": str}`` or ``null``) and
    ``describe_sha256`` (the profile's consistency digest,
    serialized so ``exec`` reading ``plan.jsonl`` can run the DONE checks; ``null`` on the report unit).

    ``model`` + ``reading`` name the configuration; ``eval_hash`` is set on probe units, ``null`` otherwise;
    ``input_hash``: an extract unit's id, ``inputs_hash`` and DONE check
    key on it, so editing ``eval.py`` or bumping a metric pin re-plans the probes and never the extraction.
    """

    id: str
    kind: str
    benchmark: str
    bridge: str
    model: str
    reading: str
    split: str | None
    shard: list[int] | None
    seed: int | None
    protocol_hash: str
    probe_hash: str | None
    eval_hash: str | None
    depends_on: list[str]
    needs: dict[str, Any]  # {gpu, cpus, mem_gb, scratch_gb, minutes}
    cmd: list[str]
    launcher: str | None
    env: dict[str, str]
    cwd: str
    out: str
    done: str
    inputs_hash: str
    expected_status: str = "ok"  # "ok" | "not_applicable"
    schema_version: int = PLAN_SCHEMA_VERSION
    profile: str | None = None
    not_applicable: dict[str, Any] | None = None
    describe_sha256: str | None = None
    #: The benchmark's ``input_hash`` (plan schema 3): what an extract unit's id, ``inputs_hash`` and DONE
    #: check key on (a scoring edit re-plans no extract unit); recorded on probe units too.
    input_hash: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        return d

    @classmethod
    def from_json(cls, d: dict) -> Unit:
        if not isinstance(d, dict):
            raise PlanError("a unit record must be a JSON object")
        sv = d.get("schema_version")
        if not isinstance(sv, int) or isinstance(sv, bool) or sv > PLAN_SCHEMA_VERSION:
            raise PlanError(f"unit schema_version {sv!r} is newer than this yavih knows ({PLAN_SCHEMA_VERSION}); upgrade yavih")
        if sv < PLAN_SCHEMA_VERSION:
            # unit ids of an older schema are not this yavih's; the features they wrote stay valid (input_hash)
            raise PlanError(f"unit schema_version {sv} is older than this yavih's ({PLAN_SCHEMA_VERSION}): rebuild the plan with "
                            "`yavih plan build` (finished shards are kept: they are checked by input_hash)")
        if d.get("kind") not in UNIT_KINDS:
            raise PlanError(f"unit kind {d.get('kind')!r} must be one of {sorted(UNIT_KINDS)}")
        if d.get("expected_status", "ok") not in ("ok", "not_applicable"):
            raise PlanError(f"unit expected_status {d.get('expected_status')!r} must be ok|not_applicable")
        required = {"id", "kind", "benchmark", "bridge", "model", "reading", "protocol_hash", "depends_on", "needs",
                    "cmd", "cwd", "out", "done", "inputs_hash"}
        missing = sorted(required - set(d))
        if missing:
            raise PlanError(f"unit record lacks {missing}")
        fields = {f for f in cls.__dataclass_fields__ if f != "extra"}
        known = {k: v for k, v in d.items() if k in fields}
        extra = {k: v for k, v in d.items() if k not in fields}
        u = cls(**{**{"split": None, "shard": None, "seed": None, "probe_hash": None, "eval_hash": None, "launcher": None,
                      "env": {}}, **known})
        u.extra = extra
        if u.shard is not None:
            u.shard = [int(u.shard[0]), int(u.shard[1])]
        return u

    @property
    def run_dir(self) -> str:
        """``<out>/<bench>/<model>/<reading>`` for extract/probe units (the features/probe parent)."""
        return self.extra.get("run_dir") or (os.path.dirname(os.path.dirname(self.out)) if self.kind == "extract" else self.out)


def resolve_path(unit: Unit, path: str) -> str:
    """A relative unit path is relative to ``unit.cwd`` (the matrix directory), wherever exec/status run."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(unit.cwd, path))


def identity_cmd(cmd: list[str]) -> list[str]:
    """The part of a unit's ``cmd`` that says what it computes: the interpreter, the values of
    ``--out`` / ``--bridge`` / ``--expect-profile`` / ``--results`` / ``--decode-workers`` and the run-directory
    positional of ``eval run`` / ``report build`` are dropped. Those are host paths (a plan built on the host must not
    be stale in a container mounting the project at ``/workspace``) and worker counts (``exec --cpus`` may change
    them); the configuration they point at is covered by ``describe_sha256`` and the protocol / probe hashes."""
    toks = list(cmd[1:])  # cmd[0] is the interpreter the launcher replaces
    out: list[str] = []
    skip = False
    for j, t in enumerate(toks):
        if skip:
            skip = False
            continue
        if t in _LOGGED_FLAGS:
            skip = True
            continue
        if j >= 2 and (toks[j - 2], toks[j - 1]) in _PATH_POSITIONAL:
            continue
        out.append(t)
    return out


# ----------------------------------------------------------------------------- benchmark view


@dataclass
class BenchmarkInfo:
    """What ``plan`` needs from a loaded ``Benchmark`` plus the pinned ``probe_hash``.

    Built by ``benchmark_info(bench)``; tests construct it directly. ``names`` holds the canonical name
    and every alias so ``profile.bench`` may use either.
    """

    name: str
    readout: str
    protocol_hash: str
    manifest_hash: str
    roles: dict[str, list[str]]
    n_rows: dict[str, int]
    requires_pos: tuple[str, ...]
    requires_streams: tuple[str, ...]
    stream_names: tuple[str, ...]
    probe_minutes_ref: float
    probe_hash: str
    names: tuple[str, ...] = ()
    #: The benchmark's input side: extract units are keyed and checked by it. ``None`` (a hand-built
    #: info in a test) falls back to ``protocol_hash``.
    input_hash: str | None = None
    #: Full lr-grid fits per probe seed: 2 when ``readout_params.shuffled_control`` (estimate.py docstring).
    probe_fits: int = 1
    #: Whether the scorer's unit asks for a GPU (``Scorer.gpu``: probes do, a stateless or direct scorer says).
    needs_gpu: bool = True
    #: Video streams whose input contract guarantees ``n_frames`` while this site has no media-info census
    #: (``W_NO_CENSUS``): every extract unit then probes each clip's header, a demuxing pass for webm / mkv.
    uncensused_frame_streams: tuple[str, ...] = ()

    def role_splits(self) -> list[str]:
        """Manifest split values the extraction covers: ``train + eval + floor + folds``, in
        role order, each once. ``unused`` splits are never read."""
        out: list[str] = []
        for role in ("train", "eval", "floor", "folds"):
            for s in self.roles.get(role) or []:
                if s not in out:
                    out.append(s)
        return out

    def train_splits(self) -> list[str]:
        """Splits the probe may train on: ``train`` plus every fold (each fold is a train split in all
        but one rotation), for ``bytes_train``."""
        out = list(self.roles.get("train") or [])
        for s in self.roles.get("folds") or []:
            if s not in out:
                out.append(s)
        return out


def benchmark_info(bench: Any) -> BenchmarkInfo:
    """Project a ``yavih.benchmark.Benchmark`` onto ``BenchmarkInfo`` and compute the pinned configuration hash
    (``Scorer.hash_config(Scorer.pinned_config(bench))``: ``probe_hash`` of the pinned ``ProbeConfig`` for a probe
    scorer, the constant ``scorer_config_hash`` otherwise)."""
    from yavih.readouts import registry  # torch-free

    spec = bench.spec
    roles = bench.roles()
    splits = []
    for role in ("train", "eval", "floor", "folds"):
        splits.extend(roles.get(role) or [])
    n_rows = {s: int(bench.n_rows(s)) for s in dict.fromkeys(splits)}
    ref = spec.budget.probe_minutes_ref
    if spec.readout not in ref:
        raise PlanError(f"benchmark {bench.name}: budget.probe_minutes_ref lacks readout {spec.readout!r}")
    scorer = registry.get(spec.readout)
    rp = getattr(spec, "readout_params", None) or {}
    return BenchmarkInfo(
        name=bench.name,
        readout=spec.readout,
        protocol_hash=bench.protocol_hash,
        input_hash=getattr(bench, "input_hash", None),
        manifest_hash=bench.manifest_hash,
        roles={k: list(v or []) for k, v in roles.items()},
        n_rows=n_rows,
        requires_pos=tuple(spec.requires.pos),
        requires_streams=tuple(spec.requires.streams),
        stream_names=tuple(spec.streams),
        probe_minutes_ref=float(ref[spec.readout]),
        probe_hash=scorer.hash_config(scorer.pinned_config(bench)),
        names=(bench.name, *spec.aliases),
        needs_gpu=scorer.gpu,
        probe_fits=probe_fits_for(rp),
        uncensused_frame_streams=uncensused_frame_streams(bench),
    )


def uncensused_frame_streams(bench: Any) -> tuple[str, ...]:
    """Video streams of ``bench`` whose ``input_contract.info`` guarantees ``n_frames`` when the site has no media-info
    census (``$YAVIH_HOME/media_info/<name>-<manifest_hash12>.jsonl.gz``); ``()`` when the census exists. Only the
    file's presence is checked (the table itself is not loaded)."""
    from yavih.sample import media_info_path

    try:
        if media_info_path(bench.name, bench.manifest_hash).is_file():
            return ()
        spec = bench.spec
        info = spec.input_contract.info
        return tuple(name for name, s in spec.streams.items() if s.kind == "video" and "n_frames" in (info.get(name) or ()))
    except (AttributeError, OSError, TypeError):  # a duck-typed benchmark without the surface: nothing to say
        return ()


def probe_fits_for(readout_params: Any) -> int:
    """Full fits one probe seed trains: the main fit, plus a second full lr-grid fit for the shuffled
    control (``readout_params.shuffled_control``, ``readouts/box.py``)."""
    try:
        return 2 if readout_params.get("shuffled_control") else 1
    except AttributeError:
        return 1


def resolve_benchmark(ref: str) -> BenchmarkInfo:
    """Hook: ``Benchmark.load(ref)`` → ``BenchmarkInfo``. Tests monkeypatch this name on the module."""
    from yavih.benchmark import Benchmark

    return benchmark_info(Benchmark.load(ref))


# ----------------------------------------------------------------------------- DONE / staleness


def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _get(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _unit_dsha(unit: Unit) -> str | None:
    """The profile's ``describe_sha256`` the unit was planned with (record key, else the in-memory extra)."""
    v = unit.describe_sha256 if unit.describe_sha256 is not None else unit.extra.get("describe_sha256")
    return v if isinstance(v, str) and v else None


def check_done(unit: Unit) -> str | None:
    """The unit's DONE marker exists *and its recorded hashes equal the
    unit's*. Returns ``"done"`` / ``"not_applicable"`` or ``None`` (missing or mismatching → not done).

    extract: ``shard-{i}.json`` with equal ``benchmark.input_hash`` (which folds the manifest hash in; read through
    the folder changelog's ``supersedes_input``), ``model``, ``reading``, ``bridge.describe_sha256`` (= the profile's), ``shard.index``
    and ``shard.n``; or ``NOT_APPLICABLE.json`` of the same ``model``, ``reading`` and
    ``describe_sha256`` in the split dir. probe: ``DONE`` with equal ``protocol_hash``, ``probe_hash``,
    ``model``, ``reading`` and ``describe_sha256`` — a reconfigured bridge (same model name, another
    configuration) is re-extracted and re-probed; ``features_fingerprint`` is unknown at plan time and is the
    probe's own gate. report: ``DONE`` present.

    A unit without a recorded ``describe_sha256`` (a hand-written record) compares the other keys only.
    """
    done_path = resolve_path(unit, unit.done)
    dsha = _unit_dsha(unit)
    if unit.kind == "extract":
        rec = _load_json(done_path)
        if rec is not None:
            want_ih = unit.input_hash or unit.protocol_hash
            ok = (
                identity_value(rec, "benchmark.input_hash") == want_ih
                and rec.get("model") == unit.model
                and rec.get("reading") == unit.reading
                and (dsha is None or identity_value(rec, "bridge.describe_sha256") == dsha)
                and unit.shard is not None
                and _get(rec, "shard.index") == unit.shard[0]
                and _get(rec, "shard.n") == unit.shard[1]
            )
            if ok:
                return "done"
        na = _load_json(os.path.join(resolve_path(unit, unit.out), NOT_APPLICABLE_FILE))
        if (na is not None and na.get("status") == "not_applicable" and na.get("model") == unit.model
                and na.get("reading") == unit.reading
                and (dsha is None or na.get("describe_sha256") == dsha)):
            return "not_applicable"
        return None
    if unit.kind == "probe":
        rec = _load_json(done_path)
        if rec is None:
            return None
        ok = (
            rec.get("protocol_hash") == unit.protocol_hash
            and rec.get("probe_hash") == unit.probe_hash
            and rec.get("model") == unit.model
            and rec.get("reading") == unit.reading
            and (dsha is None or rec.get("describe_sha256") == dsha)
        )
        if not ok:
            return None
        if unit.expected_status == "not_applicable":
            return "not_applicable"
        res = _load_json(os.path.join(os.path.dirname(done_path), "result.json"))
        if res is not None and res.get("status") == "not_applicable":
            return "not_applicable"
        return "done"
    return "done" if os.path.exists(done_path) else None


def last_record(unit: Unit) -> dict | None:
    """The unit's ``logs/last.json``. Several units share one ``out`` (shards of a
    split, seeds of a run), so exec also writes ``logs/last.<id>.json``; that per-unit file wins and the
    shared ``last.json`` counts only when its ``id`` is this unit's."""
    logs = os.path.join(resolve_path(unit, unit.out), LOGS_DIR)
    rec = _load_json(os.path.join(logs, f"last.{unit.id}.json"))
    if rec is not None:
        return rec
    rec = _load_json(os.path.join(logs, "last.json"))
    if rec is not None and rec.get("id") == unit.id:
        return rec
    return None


def is_stale(unit: Unit) -> bool:
    """A DONE marker whose ``logs/last.json.inputs_hash`` differs from the unit's is stale. The hash
    covers what the unit computes only (``identity_cmd``), so a moved project, another launcher or env, a calibrated
    ``needs`` or another worker count is never staleness. A record of another ``inputs_scheme`` is not compared:
    the DONE marker's own hashes decide."""
    rec = last_record(unit)
    if rec is None or "inputs_hash" not in rec or rec.get("inputs_scheme") != INPUTS_SCHEME:
        return False
    return rec["inputs_hash"] != unit.inputs_hash


# ----------------------------------------------------------------------------- calibration


def load_calibration(out_root: str) -> dict:
    """``<out>/_calibration.json`` written by ``plan calibrate``; ``{}`` when absent."""
    from yavih.plan.calibrate import CALIBRATION_FILE

    rec = _load_json(os.path.join(out_root, CALIBRATION_FILE))
    return rec or {}


def _unit_factor(calibration: dict, unit_id: str) -> float:
    units = calibration.get("units") or {}
    ent = units.get(unit_id)
    if not isinstance(ent, dict):
        return 1.0
    f = ent.get("minutes", 1.0)
    return float(f) if isinstance(f, (int, float)) and not isinstance(f, bool) and f > 0 else 1.0


# ----------------------------------------------------------------------------- building


def _arg_tokens(args: dict[str, Any]) -> list[str]:
    """``--arg k=v`` pairs, sorted by key for determinism; bools in lower case so the bridge's
    ``ArgSpec(type=bool)`` parser and ``describe()["args"]`` agree with the yaml spelling."""
    toks: list[str] = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, bool):
            v = "true" if v else "false"
        toks += ["--arg", f"{k}={v}"]
    return toks


def _not_applicable(info: BenchmarkInfo, profile: dict) -> dict | None:
    """Plan-time ineligibility: a dense readout against ``N == 1`` or ``pos_mode: none``, or a
    ``requires.streams`` name absent from ``profile.streams``."""
    missing: list[str] = []
    reasons: list[str] = []
    if is_dense(info.readout) or info.requires_pos:
        if profile["N"] == 1:
            missing += [f"pos.{a}" for a in info.requires_pos] or ["pos"]
            reasons.append(f"pooled features (N == 1) cannot serve the {info.readout} readout")
        elif profile["pos_mode"] == "none":
            missing += [f"pos.{a}" for a in info.requires_pos] or ["pos"]
            reasons.append(f"pos_mode none: the {info.readout} readout needs token positions")
    for s in info.requires_streams:
        if s not in profile["streams"]:
            missing.append(f"stream.{s}")
            reasons.append(f"required stream {s!r} is not among the bridge's streams {list(profile['streams'])}")
    if not missing:
        return None
    return {"missing": list(dict.fromkeys(missing)), "reason": "; ".join(reasons), "decided_by": "plan"}


def sample_root(out: str, n: int) -> str:
    """The runs root of ``plan build --sample N``: ``<out>-sample<N>`` beside the matrix ``out`` (a separate tree:
    its run directories, DONE markers, ``_index.jsonl`` and ``_report/`` never mix with the full run's)."""
    base = os.path.normpath(out)
    if os.path.basename(base) in ("", ".", ".."):
        return os.path.join(base, f"sample{int(n)}")
    return f"{base}-sample{int(n)}"


def _sampled_id(uid: str, n: int | None) -> str:
    """A capped run's unit id: derived from the full-run id and ``N`` (deterministic, never equal to a full-run id)."""
    return uid if n is None else hashing.hash12(hashing.sha256_canon({"unit": uid, "sample": int(n)}))


def _entry_units(matrix: Matrix, entry: MatrixEntry, info: BenchmarkInfo, profile: dict, *,
                 shard_minutes: int, calibration: dict, ram_budget_bytes: int, sample: int | None = None) -> list[Unit]:
    # model and reading come from the profile (checked against its describe by
    # check_profile); nothing bridge-side beyond these two names enters a path, an id or a key.
    model: str = profile["model"]
    reading: str = profile["reading"]
    describe = profile["describe"]
    dsha: str = profile["describe_sha256"]
    batch_size = int(describe.get("batch_size", 1))
    generic = is_generic_profile(profile)
    if generic:  # a declared output schema: the bytes per row bridge check measured on the smoke set
        bpr = int((profile.get("outputs") or {}).get("bytes_per_row") or 1)
    else:
        bpr = estimate.bytes_per_row(profile["N"], profile["D"], describe["dtype"], profile["pos_mode"])
    cps = float(profile["clips_per_s"])
    gpu = estimate.gpu_for_profile(profile)
    mult = calibration.get("multipliers") or {}
    out_root = sample_root(matrix.out, sample) if sample is not None else matrix.out
    run_dir = os.path.join(out_root, info.name, model, reading)
    # the rows each split contributes: all of them, or the first N of a capped run (manifest order)
    split_rows = {s: (min(int(sample), int(r)) if sample is not None else int(r)) for s, r in info.n_rows.items()}
    ih = info.input_hash or info.protocol_hash  # extract units follow the input side only
    na = None if generic else _not_applicable(info, profile)  # a generic set's eligibility is the scorer's (validate)
    units: list[Unit] = []
    extract_ids: list[str] = []

    if na is None:
        for split in info.role_splits():
            rows = split_rows[split]
            out = os.path.join(run_dir, "features", split)
            if sample is not None:  # one shard over the first `rows` rows; the layout is frozen with `sample`
                n, ranges = 1, [(0, rows)]
            else:
                n, ranges = _split_layout(matrix, entry, info, profile, split, out, rows=rows, batch_size=batch_size, cps=cps,
                                          shard_minutes=shard_minutes, bpr=bpr, max_rows_per_shard=matrix.rows_cap(entry))
            for i, (lo, hi) in enumerate(ranges):
                uid = _sampled_id(hashing.unit_id(kind="extract", benchmark=info.name, model=model, reading=reading, split=split,
                                                  shard=[i, n], protocol_hash=None, input_hash=ih, probe_hash=None, seed=None), sample)
                needs = estimate.extract_needs(hi - lo, clips_per_s=cps, cpus=entry.cpus, multipliers=mult, gpu=gpu)
                needs["minutes"] = int(math.ceil(needs["minutes"] * _unit_factor(calibration, uid)))
                cmd = [
                    "python", "-m", "yavih.extract",
                    "--bench", entry.benchmark, "--bridge", entry.script, "--split", split,
                    # `--out` is the RUN directory (features land in `<run_dir>/features/<split>`):
                    # `run_extract` lands shards in `<out>/features/<split>/`, so the unit's `done` path
                    # below is reachable only when the run dir, not the runs root, is passed.
                    *(["--rows", f"{lo}:{hi}"] if sample is not None else ["--shard", f"{i}/{n}"]), "--out", run_dir,
                    *_arg_tokens(entry.args),
                    "--canaries", str(matrix.canaries),
                    "--decode-workers", str(max(1, entry.cpus - 1)),
                    # --expect-profile alone is the configuration-drift check (E_CONFIG_DRIFT)
                    "--expect-profile", entry.profile,
                    *(["--allow-loopback"] if entry.allow_loopback else []),
                ]
                u = Unit(
                    id=uid, kind="extract", benchmark=info.name, bridge=entry.key, model=model, reading=reading,
                    split=split, shard=[i, n], seed=None, protocol_hash=info.protocol_hash, probe_hash=None,
                    eval_hash=None, depends_on=[], needs=needs, cmd=cmd, launcher=entry.launcher, env=dict(entry.env),
                    cwd=matrix.root, out=out, done=os.path.join(out, f"shard-{i:06d}.json"), inputs_hash="",
                    expected_status="ok", profile=entry.profile, not_applicable=None, describe_sha256=dsha, input_hash=ih,
                )
                u.inputs_hash = hashing.inputs_hash(cmd=identity_cmd(cmd), protocol_hash=None, input_hash=ih, probe_hash=None,
                                                    model=model, reading=reading, describe_sha256=dsha)
                u.extra = {"describe_sha256": dsha, "rows": [lo, hi], "run_dir": run_dir}
                units.append(u)
                extract_ids.append(uid)

    bytes_train = 0 if na else sum(split_rows.get(s, 0) for s in info.train_splits()) * bpr
    qh = info.probe_hash
    eh = hashing.eval_hash(info.protocol_hash, qh)  # the comparability key names the probe directory
    for seed in matrix.seeds:
        uid = _sampled_id(hashing.unit_id(kind="probe", benchmark=info.name, model=model, reading=reading, split=None, shard=None,
                                          protocol_hash=info.protocol_hash, probe_hash=qh, seed=seed), sample)
        if na is None:
            needs = estimate.probe_needs(bytes_train=bytes_train, N=1 if generic else profile["N"], D=1 if generic else profile["D"],
                                         probe_minutes_ref=info.probe_minutes_ref, seeds=1,
                                         ram_budget_bytes=ram_budget_bytes, multipliers=mult, cpus=entry.cpus,
                                         fits=info.probe_fits, gpu=gpu if info.needs_gpu else 0)
        else:
            # Writing a planned not_applicable result is a few file writes: no GPU, token wall.
            needs = {"gpu": 0, "cpus": 1, "mem_gb": 4, "scratch_gb": 0, "minutes": 5}
        needs["minutes"] = int(math.ceil(needs["minutes"] * _unit_factor(calibration, uid)))
        cmd = ["python", "-m", "yavih.cli", "eval", "run", run_dir, "--seeds", str(seed)]
        if sample is not None:  # a capped run's result never reaches the results store
            cmd += ["--no-results"]
        elif matrix.results:  # matrix `results`; exec's in-process probe reads it back from cmd
            cmd += ["--results", matrix.results]
        for code, reason in sorted(entry.waive.items()):  # matrix `waive` → the probe's validate prologue
            cmd += ["--waive", f"{code}={reason}"]
        done = os.path.join(run_dir, "probe", hashing.hash12(eh), f"seed{seed}", "DONE")
        u = Unit(
            id=uid, kind="probe", benchmark=info.name, bridge=entry.key, model=model, reading=reading, split=None,
            shard=None, seed=seed, protocol_hash=info.protocol_hash, probe_hash=qh, eval_hash=eh,
            depends_on=list(extract_ids), needs=needs, cmd=cmd, launcher=None, env=dict(entry.env), cwd=matrix.root,
            out=run_dir, done=done, inputs_hash="", expected_status="ok" if na is None else "not_applicable",
            profile=entry.profile, not_applicable=na, describe_sha256=dsha, input_hash=ih,
        )
        u.inputs_hash = hashing.inputs_hash(cmd=identity_cmd(cmd), protocol_hash=u.protocol_hash, probe_hash=qh, model=model,
                                            reading=reading, describe_sha256=dsha)
        u.extra = {"describe_sha256": dsha, "bytes_train": bytes_train, "run_dir": run_dir}
        units.append(u)
    return units


def _split_layout(matrix: Matrix, entry: MatrixEntry, info: BenchmarkInfo, profile: dict, split: str, out: str, *, rows: int,
                  batch_size: int, cps: float, shard_minutes: int, bpr: int,
                  max_rows_per_shard: int | None = estimate.MAX_ROWS_PER_SHARD) -> tuple[int, list[tuple[int, int]]]:
    """``(n, ranges)`` of one split: the split's frozen layout when one exists for this configuration
    (``features/<split>/_layout.json``, or the layout its finished shards carry when that file is absent), else the matrix
    ``shards`` pin, else the shard-count estimate (``n_shards_for``: time, 2 GiB and ``max_rows_per_shard`` caps). A pin disagreeing with a frozen layout is refused: re-sharding a split
    means moving it aside first."""
    from yavih.io.layout import frozen_layout, layout_identity

    split_dir = out if os.path.isabs(out) else os.path.join(matrix.root, out)
    ident = layout_identity(manifest_hash=info.manifest_hash, input_hash=info.input_hash or info.protocol_hash, model=profile["model"],
                            reading=profile["reading"], describe_sha256=profile["describe_sha256"])
    frozen = frozen_layout(split_dir, ident, n_rows=rows) if os.path.isdir(split_dir) else None
    pinned = entry.pinned_shards(split)
    if frozen is not None:
        n = int(frozen["n"])
        if pinned is not None and int(pinned) != n:
            raise PlanError(f"entry {entry.key!r}: shards pins {split} to {pinned} but its layout is frozen at n={n} "
                            f"({split_dir}); move that split directory aside to re-shard it, or drop the pin")
        ranges = frozen.get("ranges")
        if ranges:
            return n, [(int(a), int(b)) for a, b in ranges]
        return n, estimate.shard_ranges(rows, int(frozen.get("batch_size") or batch_size), n)
    if pinned is not None:
        n = int(pinned)
    else:
        n = estimate.n_shards_for(rows, clips_per_s=cps, shard_minutes=shard_minutes, bytes_per_row=bpr, max_rows_per_shard=max_rows_per_shard)
    return n, estimate.shard_ranges(rows, batch_size, n)


def _report_unit(matrix: Matrix, probe_ids: list[str], calibration: dict, out_root: str | None = None) -> Unit:
    # The probe set is folded into the report's protocol_hash so adding a bridge to the matrix makes the
    # old report stale under --pending (inputs_hash does not cover depends_on).
    ph = hashing.sha256_canon(sorted(probe_ids))
    uid = hashing.unit_id(kind="report", benchmark=REPORT_KEY, model=REPORT_KEY, reading=REPORT_KEY, split=None,
                          shard=None, protocol_hash=ph, probe_hash=None, seed=None)
    out_root = matrix.out if out_root is None else out_root
    out = os.path.join(out_root, "_report")
    needs = {"gpu": 0, "cpus": 2, "mem_gb": 8, "scratch_gb": 0, "minutes": 30}
    needs["minutes"] = int(math.ceil(needs["minutes"] * _unit_factor(calibration, uid)))
    cmd = ["python", "-m", "yavih.cli", "report", "build", os.path.join(out_root, "*"), "--fmt", "md",
           "--out", os.path.join(out, "report.md")]
    u = Unit(
        id=uid, kind="report", benchmark=REPORT_KEY, bridge=REPORT_KEY, model=REPORT_KEY, reading=REPORT_KEY, split=None,
        shard=None, seed=None, protocol_hash=ph, probe_hash=None, eval_hash=None, depends_on=list(probe_ids), needs=needs,
        cmd=cmd, launcher=None, env={}, cwd=matrix.root, out=out, done=os.path.join(out, "DONE"), inputs_hash="",
        expected_status="ok", profile=None, not_applicable=None, describe_sha256=None,
    )
    u.inputs_hash = hashing.inputs_hash(cmd=identity_cmd(cmd), protocol_hash=ph, probe_hash=None, model=REPORT_KEY,
                                        reading=REPORT_KEY, describe_sha256="")
    u.extra = {"describe_sha256": "", "run_dir": out}
    return u


def _order(units: list[Unit]) -> list[Unit]:
    """Topological (kind layers) then grouped by ``(gpu, minutes bucket)``; a stable sort keeps the
    matrix/split/shard order inside a group so the plan is deterministic."""
    return sorted(units, key=lambda u: (KIND_ORDER[u.kind], -int(u.needs.get("gpu", 0)),
                                        estimate.minutes_bucket(u.needs.get("minutes", 0))))


def _no_census_warning(info: BenchmarkInfo) -> dict:
    streams = list(info.uncensused_frame_streams)
    return {"code": W_NO_CENSUS, "where": info.name, "message": (
        f"{info.name}: no media-info census on this site and stream(s) {streams} guarantee n_frames: every extract unit probes "
        f"each clip's header first (a demuxing pass per webm / mkv clip); run `yavih bench verify {info.name} --census` once "
        "(shared by every bridge on this benchmark). The plan is unchanged.")}


def build_units(matrix: Matrix, *, shard_minutes: int | None = None, ram_budget_bytes: int | None = None,
                warnings: list[dict] | None = None, sample: int | None = None) -> list[Unit]:
    """Every unit of the matrix (no filtering): matrix refusals, unit records, topological order. ``warnings`` (when
    given) collects ``W_NO_CENSUS`` findings, one per benchmark; they never change a unit. ``sample=N``: the capped
    run of ``plan build --sample N`` (module docstring)."""
    shard_minutes = int(shard_minutes or matrix.shard_minutes)
    if shard_minutes < 1:
        raise PlanError("--shard-minutes must be >= 1")
    if sample is not None and (isinstance(sample, bool) or int(sample) < 1):
        raise PlanError(f"--sample must be >= 1, got {sample!r}")
    if ram_budget_bytes is None:
        ram_budget_bytes = _ram_budget_bytes()
    out_root = matrix.out if os.path.isabs(matrix.out) else os.path.join(matrix.root, matrix.out)
    calibration = load_calibration(out_root)
    units: list[Unit] = []
    seen_ids: dict[str, str] = {}
    seen_runs: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    infos: dict[str, BenchmarkInfo] = {}
    for entry in matrix.entries:
        if entry.benchmark not in infos:
            infos[entry.benchmark] = resolve_benchmark(entry.benchmark)
            if infos[entry.benchmark].uncensused_frame_streams and warnings is not None:
                warnings.append(_no_census_warning(infos[entry.benchmark]))
        info = infos[entry.benchmark]
        if info.readout not in READOUTS:
            raise PlanError(f"benchmark {info.name}: unknown readout {info.readout!r}")
        ppath = entry.profile if os.path.isabs(entry.profile) else os.path.join(matrix.root, entry.profile)
        profile = load_profile(ppath)
        check_profile(profile, entry, bench_names=info.names or (info.name,))
        # the run directory is <out>/<benchmark>/<model>/<reading>/ (canonical benchmark name, so
        # an alias collides too); two entries there would overwrite each other's features and results. Compared
        # case-insensitively: on a case-insensitive filesystem (macOS default, some SMB/NFS exports) `ModelA` and `modela`
        # are one directory, and each entry's extract units would move the other's shards to _stale/.
        run_key = (info.name, str(profile["model"]).casefold(), str(profile["reading"]).casefold())
        if run_key in seen_runs:
            first_key, first_model, first_reading = seen_runs[run_key]
            same_case = (first_model, first_reading) == (profile["model"], profile["reading"])
            raise PlanError(
                f"entries {first_key!r} and {entry.key!r} are both {first_model}/{first_reading}"
                f"{'' if same_case else ' vs ' + profile['model'] + '/' + profile['reading'] + ' (equal ignoring case: one directory on a case-insensitive filesystem)'}"
                f" on {info.name}; give one of them another model name or reading (describe()['model'] / ['reading'])"
            )
        seen_runs[run_key] = (entry.key, profile["model"], profile["reading"])
        for u in _entry_units(matrix, entry, info, profile, shard_minutes=shard_minutes,
                              calibration=calibration, ram_budget_bytes=ram_budget_bytes, sample=sample):
            if u.id in seen_ids:
                raise PlanError(
                    f"entries {seen_ids[u.id]!r} and {entry.key!r} plan the same unit {u.id} "
                    f"({u.kind} {u.benchmark} {u.model}/{u.reading}): two matrix entries resolve to one (benchmark, model, reading)"
                )
            seen_ids[u.id] = entry.key
            units.append(u)
    probe_ids = [u.id for u in units if u.kind == "probe"]
    units.append(_report_unit(matrix, probe_ids, calibration, out_root=sample_root(matrix.out, sample) if sample is not None else None))
    return _order(units)


def _ram_budget_bytes() -> int:
    """The RAM budget: ``yavih.resources.ram_budget_bytes`` (stdlib, so planning needs no numpy and no SLURM;
    a malformed ``YAVIH_RAM_BUDGET_GB`` is an error here too, never silently the default)."""
    from yavih.resources import ram_budget_bytes

    return int(ram_budget_bytes())


def plan(
    matrix_path: str,
    *,
    pending: bool = False,
    kind: str | None = None,
    shard_minutes: int | None = None,
    warnings: list[dict] | None = None,
    sample: int | None = None,
) -> list[Unit]:
    """``yavih plan build``. ``pending`` emits only units whose ``done`` is missing or stale. ``warnings`` collects the
    plan's warnings (``W_NO_CENSUS``). ``sample=N`` (``--sample N``) plans the capped run under ``<out>-sample<N>``."""
    if kind is not None and kind not in UNIT_KINDS:
        raise PlanError(f"--kind must be one of {sorted(UNIT_KINDS)}, got {kind!r}")
    matrix = load_matrix(matrix_path)
    units = build_units(matrix, shard_minutes=shard_minutes, warnings=warnings, sample=sample)
    if pending:
        units = [u for u in units if check_done(u) is None or is_stale(u)]
    if kind is not None:
        units = [u for u in units if u.kind == kind]
    return units


# ----------------------------------------------------------------------------- jsonl io


def read_plan(path: str, *, root: str | None = None) -> list[Unit]:
    """Units of ``path``. A relative ``cwd`` (what ``write_plan`` stores) is anchored at the plan file's directory;
    ``root`` (``exec --root``, ``plan status --root``) replaces every unit's ``cwd`` instead: the directory the plan's
    relative paths mean on this machine (a container that mounts the project at ``/workspace``). Absolute paths
    inside a unit (an absolute matrix ``out``) are used as they are."""
    if not os.path.isfile(path):
        raise PlanError(f"plan file not found: {path}")
    base = os.path.dirname(os.path.abspath(path))
    units: list[Unit] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise PlanError(f"{path}:{lineno}: not JSON: {e}") from e
            u = Unit.from_json(rec)
            if root is not None:
                u.cwd = os.path.abspath(os.path.expanduser(root))
            elif not os.path.isabs(u.cwd):
                u.cwd = os.path.normpath(os.path.join(base, u.cwd))
            units.append(u)
    ids = [u.id for u in units]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise PlanError(f"{path}: duplicate unit ids {dup}")
    return units


def write_plan(units: list[Unit], path: str) -> None:
    """One canonical JSON object per line (sorted keys, tmp + rename so a reader never sees a torn file). An absolute
    ``cwd`` is stored relative to the plan file's directory (``read_plan`` re-anchors it), so a plan written inside the
    project keeps working when the project is mounted elsewhere."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for u in units:
            rec = u.to_json()
            if os.path.isabs(rec["cwd"]):
                rec["cwd"] = os.path.relpath(rec["cwd"], d)
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            fh.write("\n")
    os.replace(tmp, path)
