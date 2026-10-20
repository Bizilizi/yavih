"""The lm-evaluation-harness style results store: one directory per model NAME
(``describe()["model"]``) under the results root holding every reading and benchmark of it:
``results_<timestamp>.json`` per probe invocation (or a ``report --join``, or merged by ``report build
--merge-results``) and ``samples_<benchmark>_<reading>_<timestamp>.jsonl`` with ``--log-samples``. Entries are
keyed ``<benchmark>:<reading>`` (a join ``<benchmark>:a+v+av``, an older entry kept by a merge
``<benchmark>:<reading>@<eval_hash12>``); readers take ``benchmark``, ``reading`` and ``eval_hash`` from
``configs[key]``, never from the key. A merge reads files of ``RESULTS_FILE_VERSION`` only
(:func:`upgrade_results_file`). Re-exports ``RESULTS_FILE_VERSION``.

Derived records: the per-unit ``result.json`` under ``runs/`` stays the scheduler's atomic, DONE-marked
record; these files are written after it (tmp + ``os.replace``) and are never read to
decide anything.stdlib only (the metric registry, itself stdlib-only, is imported
lazily for ``higher_is_better`` / ``floor_of``).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import os
import platform
import socket
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yavih import versions as _versions
from yavih.versions import __version__

__all__ = [
    "RESULTS_FILE_VERSION",
    "TIMESTAMP_FORMAT",
    "timestamp",
    "git_hash",
    "metric_info",
    "benchmark_entry",
    "results_document",
    "write_invocation",
    "write_json_atomic",
    "merge_results",
    "write_join",
    "read_results_files",
    "upgrade_results_file",
    "entry_key",
    "join_reading",
]

#: Results-store file schema; defined in ``yavih.versions`` (core), re-exported here.
RESULTS_FILE_VERSION: int = _versions.RESULTS_FILE_VERSION
#: The routing order of a cross-reading join's entry key (``<benchmark>:a+v+av``).
_JOIN_ORDER = ("a", "v", "av")
#: ``<timestamp>`` = UTC ``%Y-%m-%dT%H-%M-%S.%f`` (filesystem-safe, lm-eval's form).
TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S.%f"
#: Metrics whose smaller value is better when the registry cannot say.
_LOWER_IS_BETTER = frozenset({"absrel", "rmse", "mae_seconds"})


# --------------------------------------------------------------------------- small helpers


def timestamp(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc).strftime(TIMESTAMP_FORMAT)


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_hash() -> str | None:
    """The running yavih's source commit: ``yavih.buildinfo.commit()`` — the build
    stamp, else ``git rev-parse HEAD`` (+``+dirty``) of the checkout it runs from, else ``None``; the same value the
    sidecars record as ``yavih.commit``."""
    stamp = getattr(_versions, "GIT_HASH", None)
    return str(stamp) if stamp else None


def _plain(x: Any) -> Any:
    """JSON-able copy of a spec block (dataclass, namespace, mapping, tuple)."""
    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_plain(v) for v in x]
    if hasattr(x, "__dict__"):
        return {str(k): _plain(v) for k, v in vars(x).items() if not str(k).startswith("_")}
    return str(x)


def _finite(x: Any) -> Any:
    """Strict JSON: non-finite floats become ``null``."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def write_json_atomic(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_finite(obj), fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- metric facts


def metric_info(names: Iterable[str], *, floors: Sequence[str] = (), probe_metrics: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """``{name: {higher_is_better, kind, floor_of}}`` from ``MetricSpec``, overlaid by a folder's
    ``METRICS``; names the registry does not know fall back to the metric-table conventions (``n_*`` are
    disclosures, ``absrel/rmse/mae_seconds`` lower is better) and ``floor_of`` to the name stem
    (``report.compare.floor_name_for``)."""
    names = list(names)
    try:
        from yavih.metrics import registry
    except Exception:  # noqa: BLE001 - the report stays usable without the metric package
        registry = None  # type: ignore[assignment]
    out: dict[str, dict[str, Any]] = {}
    for name in [*names, *floors]:
        spec = (probe_metrics or {}).get(name)
        if spec is None and registry is not None:
            try:
                spec = registry.resolve(name)
            except Exception:  # noqa: BLE001 - unknown (e.g. a folder metric): conventions below
                spec = None
        kind = getattr(spec, "kind", None) or ("disclosure" if name.startswith("n_") else ("floor" if name.endswith("_floor") else "metric"))
        if spec is not None and hasattr(spec, "higher_is_better"):
            hib = spec.higher_is_better
        else:
            hib = None if kind == "disclosure" else (name not in _LOWER_IS_BETTER)
        out[name] = {"higher_is_better": hib, "kind": kind, "floor_of": tuple(getattr(spec, "floor_of", ()) or ())}
    # floor ↔ metric: the floor's declared floor_of, else the name stem
    from yavih.report.compare import floor_name_for

    floor_by_metric: dict[str, str] = {}
    for f in floors:
        for m in out.get(f, {}).get("floor_of", ()):
            if m in names:
                floor_by_metric.setdefault(m, f)
    for m in names:
        if m not in floor_by_metric:
            f = floor_name_for(m, names, list(floors))
            if f is not None:
                floor_by_metric[m] = f
    for m in names:
        out[m]["floor"] = floor_by_metric.get(m)
    return out


# --------------------------------------------------------------------------- per-benchmark entry


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and not math.isfinite(v))


def _mean(values: list[Any]) -> Any:
    """Mean over seeds (bools: any; equal ints: the int; equal-length lists: element-wise); a missing or
    non-finite seed value is ignored."""
    vals = [v for v in values if _present(v)]
    if not vals:
        return None
    if all(isinstance(v, bool) for v in vals):
        return any(vals)
    if all(_num(v) for v in vals):
        if all(isinstance(v, int) for v in vals) and len(set(vals)) == 1:
            return vals[0]
        return float(statistics.fmean(float(v) for v in vals))
    if all(isinstance(v, list) for v in vals) and len({len(v) for v in vals}) == 1:
        return [_mean([v[i] for v in vals]) for i in range(len(vals[0]))]
    return vals[0]


def _std(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if _num(v) and _present(v)]
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


def _seed(r: Mapping[str, Any]) -> int:
    return int((r.get("probe") or {}).get("seed") or 0)


def _spec_of(bench: Any) -> Any:
    return getattr(bench, "spec", None) if bench is not None else None


def _headline(spec: Any, metric_names: Sequence[str], floor_names: Sequence[str], bench_name: str) -> str | None:
    h = getattr(getattr(spec, "metrics", None), "headline", None)
    if h:
        return str(h)
    from yavih.report.compare import _bench_meta, _pick_headline

    return _pick_headline(list(metric_names), list(floor_names), _bench_meta(bench_name))


def benchmark_entry(results: Sequence[Mapping[str, Any]], *, bench: Any = None, result_paths: Sequence[str | None] = ()) -> dict[str, Any]:
    """The five per-entry blocks of one results file from the seeds' ``result.json`` records of
    ONE ``(benchmark, eval_hash, model, reading)``: ``{"name", "reading", "key", "results", "configs", "version",
    "higher_is_better", "n_samples", "external", "bridge", "probe", "model", "env"}``. ``configs`` carries the
    entry's ``benchmark``, ``reading``, ``eval_hash`` and its own provenance (``probe``, ``bridge``,
    ``describe_sha256``, ``features_fingerprint``): one model directory holds several readings, i.e. several
    bridges."""
    if not results:
        raise ValueError("benchmark_entry needs at least one result.json record")
    results = sorted(results, key=_seed)
    first = results[0]
    bmeta = first.get("benchmark") or {}
    name = str(bmeta.get("name"))
    spec = _spec_of(bench)
    ok = [r for r in results if r.get("status") == "ok"]
    probe_mod = None
    if bench is not None:
        try:
            from yavih.readouts.base import folder_probe

            probe_mod = folder_probe(bench)
        except Exception:  # noqa: BLE001 - facts only; a broken probe.py was refused by the probe already
            probe_mod = None

    if ok:
        metric_names = list((ok[0].get("metrics") or {}).keys())
        floor_names = list((ok[0].get("floors") or {}).keys())
        control_names = list((ok[0].get("controls") or {}).keys())
        info = metric_info(metric_names, floors=floor_names, probe_metrics=getattr(probe_mod, "METRICS", None))
        block: dict[str, Any] = {}
        for m in metric_names:
            vals = [(r.get("metrics") or {}).get(m) for r in ok]
            block[m] = _mean(vals)
            if len(ok) > 1:
                sd = _std(vals)
                if sd is not None and not (all(isinstance(v, int) for v in vals) and len(set(vals)) == 1):
                    block[f"{m}_std"] = sd
            ci = (ok[0].get("ci95") or {}).get(m)  # the clip-level interval of the lowest seed
            if ci is not None:
                block[f"{m}_ci95"] = list(ci)
            f = info[m].get("floor")
            if f is not None:
                fv = _mean([(r.get("floors") or {}).get(f) for r in ok])
                if fv is not None:
                    block[f"{m}_floor"] = fv
        for c in control_names:
            block[c] = _mean([(r.get("controls") or {}).get(c) for r in ok])
        cov = (ok[0].get("coverage") or {}).get("eval") or {}
        nominal = cov.get("nominal")
        block.update({
            "status": "ok",
            "headline": _headline(spec, metric_names, floor_names, name),
            "n_seeds": len(ok),
            "coverage_eval": (float(cov.get("present", 0)) / float(nominal)) if nominal else None,
        })
        hib = {m: info[m]["higher_is_better"] for m in metric_names if info[m]["higher_is_better"] is not None}
    else:
        block = {"status": str(first.get("status")), "reason": first.get("reason")}
        hib = {}

    cov_all = first.get("coverage") or {}

    def present(role: str) -> int | None:
        c = cov_all.get(role)
        if isinstance(c, Mapping):
            return int(c.get("present", 0))
        if role == "eval" and isinstance(cov_all.get("folds"), Mapping):
            return int(sum(int((b or {}).get("present", 0)) for b in cov_all["folds"].values()))
        return None

    roles: dict[str, Any] = {}
    splits = getattr(spec, "splits", None)
    for role in ("train", "eval", "floor", "folds"):
        names = list(getattr(splits, role, None) or ()) if splits is not None else []
        n = present("eval" if role == "folds" else role)
        if names or (n is not None and role != "folds"):
            roles[role] = {"splits": names, "n": n}
    metrics_cfg = getattr(spec, "metrics", None)

    def names_of(attr: str, block: str) -> list[str]:
        declared = getattr(metrics_cfg, attr, None)
        if declared is not None:
            return list(declared)
        return list((ok[0].get(block) or {}).keys()) if ok else []

    reading = str(first.get("reading") or "")
    bridge = first.get("bridge") if isinstance(first.get("bridge"), Mapping) else {}
    configs = {
        "benchmark": name,
        "reading": reading,
        "eval_hash": first.get("eval_hash"),
        "readout": bmeta.get("readout"),
        "readout_version": bmeta.get("readout_version"),
        "protocol_version": bmeta.get("protocol_version"),
        "protocol_hash": bmeta.get("protocol_hash"),
        "probe_hash": (first.get("probe") or {}).get("hash"),
        "manifest_hash": bmeta.get("manifest_hash"),
        "folder_source": bmeta.get("folder_source"),  # provenance: the folder's probe.py / reader.py digests
        "roles": roles,
        "metrics": {
            "headline": getattr(metrics_cfg, "headline", None) or block.get("headline"),
            "list": names_of("list", "metrics"),
            "floors": names_of("floors", "floors"),
            "controls": names_of("controls", "controls"),
        },
        "input_contract": _plain(getattr(spec, "input_contract", None)),
        "output_contract": _plain(getattr(spec, "output_contract", None)),
        "external": _plain(getattr(spec, "external", None)) or [],
        # per-entry probe provenance: survives --merge-results, which keeps one file-level config
        "probe": {"config": (first.get("probe") or {}).get("config"), "seeds": [_seed(r) for r in results],
                  "standard": all(bool(r.get("standard")) for r in results)},
        "bridge": first.get("bridge"),
        "describe_sha256": bridge.get("describe_sha256"),
        "features_fingerprint": (first.get("features") or {}).get("fingerprint"),
        "unit_results": [str(p) for p in result_paths if p],
    }
    # validation waivers recorded with the number (marked by the report, never a penalty)
    from yavih.report.compare import waived_of

    waived = waived_of(results)
    if waived:
        configs["waived"] = waived
        block["waived"] = [w["code"] for w in waived]
    n_samples = {k: v for k, v in (("train", present("train")), ("eval", present("eval")), ("floor", present("floor"))) if v is not None}
    external: dict[str, str] = {}
    for r in results:
        external.update({str(k): str(v) for k, v in (r.get("external") or {}).items()})
    return {
        "name": name,
        "reading": reading,
        "key": entry_key(name, reading),
        "results": block,
        "configs": configs,
        "version": bmeta.get("protocol_version"),
        "higher_is_better": hib,
        "n_samples": n_samples,
        "external": external,
        "bridge": first.get("bridge"),
        "probe": {"config": (first.get("probe") or {}).get("config"), "seeds": [_seed(r) for r in results],
                  "standard": all(bool(r.get("standard")) for r in results)},
        "model": first.get("model"),
        "env": first.get("env") or {},
    }


def entry_key(benchmark: str, reading: str, eval_hash: str | None = None) -> str:
    """``<benchmark>:<reading>``, ``<benchmark>:<reading>@<eval_hash12>`` for an older entry a merge
    keeps. Flat (lm-eval compatible) and never parsed: readers use ``configs[key]``."""
    key = f"{benchmark}:{reading}"
    return f"{key}@{str(eval_hash)[:12]}" if eval_hash else key


def _env(first_env: Mapping[str, Any]) -> dict[str, Any]:
    torch = sys.modules.get("torch")
    cuda = getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None
    return {
        "host": first_env.get("host") or socket.gethostname(),
        "python": first_env.get("python") or platform.python_version(),
        "torch": first_env.get("torch"),
        "cuda": cuda,
        "gpu": first_env.get("gpu"),
        "slurm_job_id": first_env.get("slurm_job_id") or os.environ.get("SLURM_JOB_ID"),
    }


def results_document(entries: Sequence[Mapping[str, Any]], *, keys: Sequence[str] | None = None, extra_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the results file from :func:`benchmark_entry` outputs (``keys`` = the ``results`` key per
    entry, default ``<benchmark>:<reading>``). File-level ``config.bridge`` / ``config.probe`` are those of the
    entries when they all agree (always for one entry), else ``null`` (see ``configs[key]``)."""
    if not entries:
        raise ValueError("results_document needs at least one entry")
    keys = list(keys) if keys is not None else [str(e.get("key") or entry_key(str(e["name"]), str(e.get("reading") or ""))) for e in entries]
    first = entries[0]
    external: dict[str, str] = {}
    for e in entries:
        external.update(e.get("external") or {})
    config = {
        "model": first.get("model"),
        "bridge": _agreed([e.get("bridge") for e in entries]),
        "probe": _agreed([e.get("probe") for e in entries]),
        "external": external,
    }
    config.update(dict(extra_config or {}))
    return {
        "results_version": RESULTS_FILE_VERSION,
        "results": {k: e["results"] for k, e in zip(keys, entries, strict=True)},
        "configs": {k: e["configs"] for k, e in zip(keys, entries, strict=True)},
        "versions": {k: e["version"] for k, e in zip(keys, entries, strict=True)},
        "higher_is_better": {k: e["higher_is_better"] for k, e in zip(keys, entries, strict=True)},
        "n_samples": {k: e["n_samples"] for k, e in zip(keys, entries, strict=True)},
        "config": config,
        "git_hash": git_hash(),
        "yavih_version": __version__,
        "date": _utc_iso(),
        "env": _env(first.get("env") or {}),
    }


def _agreed(values: Sequence[Any]) -> Any:
    """The common value of ``values`` (compared as canonical JSON), else ``None``."""
    if not values:
        return None
    canon = {json.dumps(v, sort_keys=True, default=str) for v in values}
    return values[0] if len(canon) == 1 else None


def write_invocation(
    results: Sequence[Mapping[str, Any]],
    *,
    results_root: str | os.PathLike,
    bench: Any = None,
    result_paths: Sequence[str | None] = (),
    samples: Sequence[Mapping[str, Any]] | None = None,
    now: _dt.datetime | None = None,
) -> dict[str, Path | None]:
    """One probe invocation's records (one benchmark, one ``model``, one ``reading``) → ``<results_root>/<model>/
    results_<timestamp>.json`` with the entry ``<benchmark>:<reading>`` and, when ``samples`` is not ``None``,
    ``samples_<benchmark>_<reading>_<timestamp>.jsonl`` (one JSON line per scored row, seed and fold). Returns
    ``{"results_file", "samples_file"}``."""
    results = list(results)
    if not results:
        raise ValueError("write_invocation needs at least one result.json record")
    pairs = {(str(r.get("model") or ""), str(r.get("reading") or "")) for r in results}
    if len(pairs) != 1:
        raise ValueError(f"one invocation writes one model's reading, got {sorted(pairs)}")
    model, reading = pairs.pop()
    if not model or not reading:
        raise ValueError("result.json records carry no model/reading; re-run the probe")
    entry = benchmark_entry(results, bench=bench, result_paths=list(result_paths) or [None] * len(results))
    doc = results_document([entry])
    ts = timestamp(now)
    out_dir = Path(results_root) / model
    results_file = write_json_atomic(out_dir / f"results_{ts}.json", doc)
    samples_file = None
    if samples is not None:
        samples_file = out_dir / f"samples_{entry['name']}_{reading}_{ts}.jsonl"
        tmp = samples_file.with_name(f"{samples_file.name}.tmp.{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            for line in samples:
                fh.write(json.dumps(_finite(line), ensure_ascii=False, allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, samples_file)
    return {"results_file": results_file, "samples_file": samples_file}


# --------------------------------------------------------------------------- results-file helpers


def join_reading(readings: Sequence[str]) -> str:
    """A join's reading tag ``a+v+av`` (routing order) from the readings it joined."""
    rs = [str(r) for r in readings]
    return "+".join([r for r in _JOIN_ORDER if r in rs] + [r for r in rs if r not in _JOIN_ORDER])


def upgrade_results_file(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A results file of ``RESULTS_FILE_VERSION`` (a deep copy). A file of another version raises ``ValueError``:
    a newer one needs a newer yavih, an older one is re-written by ``yavih report`` from its ``result.json``
    records."""
    out = json.loads(json.dumps(doc))
    version = out.get("results_version")
    if not isinstance(version, int) or isinstance(version, bool) or version > RESULTS_FILE_VERSION:
        raise ValueError(f"results_version {version!r} is newer than RESULTS_FILE_VERSION {RESULTS_FILE_VERSION} (or not an int)")
    if version < RESULTS_FILE_VERSION:
        raise ValueError(f"results_version {version} is older than this yavih reads ({RESULTS_FILE_VERSION}); "
                         "re-run `yavih report` over its result.json records")
    return out


# --------------------------------------------------------------------------- merge (report build --merge-results)


def read_results_files(model_dir: str | os.PathLike) -> list[tuple[Path, dict]]:
    """Every readable ``results_*.json`` of a model directory of ``RESULTS_FILE_VERSION``, oldest first (by its
    ``<timestamp>``); a file of another version is skipped with a notice (:func:`upgrade_results_file`)."""
    out: list[tuple[Path, dict]] = []
    for p in sorted(Path(model_dir).glob("results_*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"[yavih.report] skipping unreadable {p}: {e}", file=sys.stderr)
            continue
        if not (isinstance(doc, dict) and isinstance(doc.get("results"), dict)):
            continue
        try:
            out.append((p, upgrade_results_file(doc)))
        except ValueError as e:
            print(f"[yavih.report] skipping {p}: {e}", file=sys.stderr)
    return out


def _entry_standard(doc: Mapping[str, Any], key: str) -> bool:
    """Whether a results-file entry was scored under the pinned probe: the entry's own ``configs[key].probe``,
    else the file-level ``config.probe`` (one invocation = one entry)."""
    cfg = (doc.get("configs") or {}).get(key) or {}
    probe = cfg.get("probe") if isinstance(cfg.get("probe"), Mapping) else None
    if probe is None:
        probe = (doc.get("config") or {}).get("probe") or {}
    return bool(probe.get("standard", False))


def _merge_model(files: Sequence[tuple[Path, dict]]) -> dict[str, Any]:
    """``report build --merge-results``: the newest entry per ``(benchmark, reading, eval_hash)`` wins (one entry, so
    one configuration and one feature set: never averaged across them); entries of one ``(benchmark, reading)``
    with different ``eval_hash``es are all kept: the newest STANDARD entry of the newest ``protocol_version``
    under ``<benchmark>:<reading>`` (a non-standard, e.g. ``fast()``, entry holds it only when no standard one
    exists) and the others under ``<benchmark>:<reading>@<eval_hash12>``. ``benchmark``, ``reading`` and
    ``eval_hash`` come from ``configs[key]``, never from the key. Each entry keeps its own provenance
    (``configs[key]``: ``probe``, ``bridge``, ``describe_sha256``, ``features_fingerprint``)."""
    newest: dict[tuple[str, str, str], tuple[int, Path, dict, str]] = {}
    for order, (path, doc) in enumerate(files):  # oldest first: a later file overwrites
        for key in doc["results"]:
            cfg = (doc.get("configs") or {}).get(key) or {}
            k = (str(cfg.get("benchmark") or key.split(":", 1)[0]), str(cfg.get("reading") or ""), str(cfg.get("eval_hash") or ""))
            newest[k] = (order, path, doc, key)
    by_entry: dict[tuple[str, str], list[tuple[tuple[str, str, str], tuple[int, Path, dict, str]]]] = {}
    for k, v in newest.items():
        by_entry.setdefault((k[0], k[1]), []).append((k, v))
    out = {"results": {}, "configs": {}, "versions": {}, "higher_is_better": {}, "n_samples": {}}
    external: dict[str, str] = {}
    merged_from: list[str] = []
    latest_doc: dict | None = None
    latest_order = -1
    mains: list[str] = []
    for (bench, reading), items in sorted(by_entry.items()):
        def rank(item):
            _k, (order, _p, doc, key) = item
            pv = (doc.get("versions") or {}).get(key)
            return (_entry_standard(doc, key), pv if isinstance(pv, int) else -1, order)

        items = sorted(items, key=rank, reverse=True)
        for i, ((_b, _r, eh), (order, path, doc, key)) in enumerate(items):
            out_key = entry_key(bench, reading, None if i == 0 else (eh or "unknown"))
            if i == 0:
                mains.append(out_key)
            for block in out:
                if key in (doc.get(block) or {}):
                    out[block][out_key] = doc[block][key]
            external.update((doc.get("config") or {}).get("external") or {})
            if str(path) not in merged_from:
                merged_from.append(str(path))
            if order > latest_order:
                latest_order, latest_doc = order, doc
    cfg = dict((latest_doc or {}).get("config") or {})
    cfg["external"] = external
    cfg["merged_from"] = sorted(merged_from)  # the files whose entries the merge kept
    # the file-level probe / bridge blocks describe the main-key entries only when they agree; else see configs[key]
    cfg["probe"] = _agreed([(out["configs"].get(k) or {}).get("probe") for k in mains])
    cfg["bridge"] = _agreed([(out["configs"].get(k) or {}).get("bridge") for k in mains])
    return {
        "results_version": RESULTS_FILE_VERSION,
        **out,
        "config": cfg,
        "git_hash": git_hash(),
        "yavih_version": __version__,
        "date": _utc_iso(),
        "env": (latest_doc or {}).get("env") or {},
    }


def merge_results(results_root: str | os.PathLike, models: Iterable[str] | None = None, *, now: _dt.datetime | None = None) -> list[Path]:
    """Write one merged ``results_<timestamp>.json`` per model directory that holds two or more results files
    (one file has nothing to merge). ``models`` restricts the model names; default every directory under the
    root. Returns the written paths."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    dirs = [root / m for m in models] if models is not None else sorted(p for p in root.iterdir() if p.is_dir())
    written: list[Path] = []
    ts = timestamp(now)
    for d in dirs:
        files = read_results_files(d)
        if len(files) < 2:
            continue
        merged = _merge_model(files)
        written.append(write_json_atomic(d / f"results_{ts}.json", merged))
    return written


# --------------------------------------------------------------------------- joins (report --join)


def write_join(results_root: str | os.PathLike, model: str, *, benchmark: str, joined: Mapping[str, Any],
               refs: Mapping[str, Mapping[str, Any]], unit_results: Mapping[str, str] | None = None,
               external: Mapping[str, str] | None = None, warnings: Sequence[Mapping[str, Any]] = (),
               now: _dt.datetime | None = None) -> Path:
    """``<results_root>/<model>/results_<timestamp>.json`` for a cross-reading join: the entry
    ``<benchmark>:a+v+av`` (the readings in routing order) holds the joined numbers; its ``configs[key]`` carries
    ``benchmark``, ``reading`` (``a+v+av``), ``eval_hash``, ``joined: {reading: {describe_sha256, unit_results}}``
    instead of ``bridge``, and ``warnings`` (``W_JOIN_WEIGHTS``)."""
    first = next(iter(refs.values())) if refs else {}
    bmeta = first.get("benchmark") or {}
    reading = join_reading(list(refs))
    key = entry_key(benchmark, reading)
    doc = {
        "results_version": RESULTS_FILE_VERSION,
        "results": {key: {**_plain(dict(joined)), "status": "ok"}},
        "configs": {key: {
            "benchmark": benchmark, "reading": reading, "eval_hash": first.get("eval_hash"),
            "readout": bmeta.get("readout"), "readout_version": bmeta.get("readout_version"),
            "protocol_version": bmeta.get("protocol_version"), "protocol_hash": bmeta.get("protocol_hash"),
            "probe_hash": (first.get("probe") or {}).get("hash"), "manifest_hash": bmeta.get("manifest_hash"),
            "probe": {"config": (first.get("probe") or {}).get("config"), "seeds": sorted({_seed(r) for r in refs.values()}),
                      "standard": all(bool(r.get("standard")) for r in refs.values())},
            "joined": {rd: {"describe_sha256": (r.get("bridge") or {}).get("describe_sha256"),
                            "unit_results": [str((unit_results or {}).get(rd))] if (unit_results or {}).get(rd) else []}
                       for rd, r in refs.items()},
            "warnings": [dict(w) for w in warnings],
        }},
        "versions": {key: bmeta.get("protocol_version")},
        "higher_is_better": {key: {}},
        "n_samples": {key: {}},
        "config": {
            "model": model,
            "bridge": None,
            "probe": None,
            "external": dict(external or {}),
        },
        "git_hash": git_hash(),
        "yavih_version": __version__,
        "date": _utc_iso(),
        "env": _env(first.get("env") or {}),
    }
    return write_json_atomic(Path(results_root) / model / f"results_{timestamp(now)}.json", doc)
