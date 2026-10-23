"""The model registry: recorded scores next to the hash they were produced under, comparable without rerunning.

A registry is a text-only directory (its own git repository, never the yavih repository and never a project folder):

```
<registry>/
  entries/<benchmark>/<eval_hash12>/<org>/<model>/<reading>.json   one entry = one comparable score
  protocols/<benchmark>/<protocol_hash12>.json                      the hashed payload (``bench hash --explain``) + the
                                                                    folder's conformance fixture, copied on first use
```

An entry (``REGISTRY_ENTRY_VERSION``) is written by :func:`export` from a results-store file
(``results_<timestamp>.json``, [results](results.md)): nothing is typed by hand but the ``org`` namespace. It carries
the comparability key and its parts (``identity``), who was measured (``model``), the numbers with their seeds
(``score``), what was scored (``population``), the conformance result the exporting yavih obtained on the folder's
fixture (``conformance``) and how to diagnose a disagreement (``provenance``). :func:`check` is what the registry
repository's CI runs on every pull request; :func:`records` turns entries into report records so that
``yavih report build --registry DIR`` compares local results with the registry under the one comparability rule
(same benchmark, same ``eval_hash``, both standard; coverage at or above the folder's minimum; differences below a
metric's recorded tolerance are ties).

stdlib + yavih core; the report integration imports ``yavih.report`` lazily.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yavih import hashing
from yavih.versions import REGISTRY_ENTRY_VERSION, __version__

__all__ = [
    "REGISTRY_ENTRY_VERSION",
    "ENTRIES_DIR",
    "PROTOCOLS_DIR",
    "RegistryError",
    "Finding",
    "RegistryReport",
    "entry_id",
    "entry_path",
    "protocol_path",
    "entry_from_results",
    "protocol_payload",
    "export",
    "check",
    "records",
    "read_entries",
]

ENTRIES_DIR = "entries"
PROTOCOLS_DIR = "protocols"
_ORG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

E_SCHEMA = "E_REGISTRY_SCHEMA"
E_ID = "E_REGISTRY_ID"
E_HASH = "E_REGISTRY_HASH"
E_PROTOCOL = "E_REGISTRY_PROTOCOL"
E_CONFORMANCE = "E_REGISTRY_CONFORMANCE"
W_NONSTANDARD = "W_REGISTRY_NONSTANDARD"
W_COVERAGE = "W_REGISTRY_COVERAGE"
W_UNVERIFIED = "W_REGISTRY_UNVERIFIED"


class RegistryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    entry: str | None = None


@dataclass
class RegistryReport:
    root: str
    n_entries: int = 0
    n_protocols: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "ok": self.ok, "n_entries": self.n_entries, "n_protocols": self.n_protocols,
                "errors": [{"code": f.code, "message": f.message, "entry": f.entry} for f in self.errors],
                "warnings": [{"code": f.code, "message": f.message, "entry": f.entry} for f in self.warnings]}


# --------------------------------------------------------------------------- ids and paths


def entry_id(benchmark: str, eval_hash: str, org: str, model: str, reading: str) -> str:
    return f"{benchmark}/{hashing.hash12(eval_hash)}/{org}/{model}/{reading}"


def entry_path(root: str | os.PathLike, entry: Mapping[str, Any]) -> Path:
    i = entry["identity"]
    return Path(root) / ENTRIES_DIR / i["benchmark"] / hashing.hash12(i["eval_hash"]) / entry["model"]["org"] / entry["model"]["model"] / f"{i['reading']}.json"


def protocol_path(root: str | os.PathLike, benchmark: str, protocol_hash: str) -> Path:
    return Path(root) / PROTOCOLS_DIR / benchmark / f"{hashing.hash12(protocol_hash)}.json"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- building an entry


def _seed_records(unit_results: Sequence[str], *, base: Path | None) -> list[dict[str, Any]]:
    """The per-seed ``result.json`` records a results-file entry names (``configs[key].unit_results``), where they
    are readable: as given, else relative to ``base``."""
    out: list[dict[str, Any]] = []
    for p in unit_results:
        candidates = [Path(p)] + ([base / p] if base is not None else [])
        for c in candidates:
            if c.is_file():
                try:
                    r = json.loads(c.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    break
                if isinstance(r, dict) and r.get("status") == "ok":
                    out.append(r)
                break
    return out


def _conformance_block(bench: Any | None, cfg: Mapping[str, Any], cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the exporting yavih can vouch for: the folder's fixture at the entry's protocol hash. ``cache`` (keyed by
    folder and protocol hash) runs the fixture once per benchmark across the entries of one export."""
    if bench is None:
        return {"passed": None, "skipped": False, "reason": f"no benchmark folder named {cfg.get('benchmark')!r} resolves here"}
    if bench.protocol_hash != cfg.get("protocol_hash"):
        return {"passed": None, "skipped": False,
                "reason": f"the resolvable folder computes protocol_hash {hashing.hash12(bench.protocol_hash)}… (version "
                          f"{bench.spec.protocol_version}); the entry was scored under {hashing.hash12(str(cfg.get('protocol_hash') or ''))}…"}
    from yavih import conformance

    ckey = f"{bench.folder}:{bench.protocol_hash}"
    if cache is not None and ckey in cache:
        return copy.deepcopy(cache[ckey])
    rep = conformance.check(bench)
    exp = conformance.expected_path(bench)
    block = {
        "passed": bool(rep.ok and not rep.skipped), "skipped": bool(rep.skipped), "protocol_version": rep.protocol_version,
        "compared": rep.compared, "tolerance": conformance.tolerances(bench),
        "expected_sha256": _sha256_file(exp) if exp.is_file() else None,
        "checked_with": {"yavih_version": __version__},
        "reason": None if rep.ok else "; ".join(f.message for f in rep.findings)[:500],
    }
    if cache is not None:
        cache[ckey] = copy.deepcopy(block)
    return block


def entry_from_results(doc: Mapping[str, Any], key: str, *, org: str, source: str | os.PathLike, bench: Any | None = None,
                       unit_base: str | os.PathLike | None = None, conformance_cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """One registry entry from entry ``key`` of a results-store document (``RESULTS_FILE_VERSION`` 2);
    ``conformance_cache`` is shared across the entries of one export (see :func:`_conformance_block`)."""
    if not _ORG_RE.match(org):
        raise RegistryError(E_SCHEMA, f"org {org!r}: [A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}")
    cfg = dict((doc.get("configs") or {}).get(key) or {})
    res = dict((doc.get("results") or {}).get(key) or {})
    if not cfg or not res:
        raise RegistryError(E_SCHEMA, f"{source}: no entry {key!r}")
    if res.get("status") != "ok":
        raise RegistryError(E_SCHEMA, f"{source}: entry {key!r} has status {res.get('status')!r}; only an ok score is registered")
    benchmark, reading = str(cfg["benchmark"]), str(cfg["reading"])
    if "+" in reading:
        raise RegistryError(E_SCHEMA, f"{source}: entry {key!r} is a join; a registry entry is one reading")
    model = str(cfg.get("model") or (doc.get("config") or {}).get("model") or "")
    if not _MODEL_RE.match(model):
        raise RegistryError(E_SCHEMA, f"{source}: model name {model!r} outside [A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}")
    probe = dict(cfg.get("probe") or {})
    bridge = dict(cfg.get("bridge") or {})
    metrics_cfg = dict(cfg.get("metrics") or {})
    names = list(metrics_cfg.get("list") or [])
    floor_names = list(metrics_cfg.get("floors") or [])
    control_names = list(metrics_cfg.get("controls") or [])
    seeds_rec = _seed_records(list(cfg.get("unit_results") or []), base=Path(unit_base) if unit_base is not None else None)
    seeds: list[dict[str, Any]] = [{"seed": int((r.get("probe") or {}).get("seed", 0)), "metrics": r.get("metrics") or {},
                                    "floors": r.get("floors") or {}, "controls": r.get("controls") or {}, "ci95": r.get("ci95") or {},
                                    "created": r.get("created")} for r in sorted(seeds_rec, key=lambda r: int((r.get("probe") or {}).get("seed", 0)))]
    if not seeds:
        seeds = [{"seed": int(s), "metrics": None} for s in (probe.get("seeds") or [])]
    first = seeds_rec[0] if seeds_rec else {}
    eval_hash = str(cfg.get("eval_hash") or hashing.eval_hash(str(cfg["protocol_hash"]), str(cfg["probe_hash"])))
    identity = {
        "benchmark": benchmark, "reading": reading, "readout": cfg.get("readout"), "readout_version": cfg.get("readout_version"),
        "protocol_version": cfg.get("protocol_version"), "protocol_hash": cfg.get("protocol_hash"),
        "input_hash": bench.input_hash if bench is not None and bench.protocol_hash == cfg.get("protocol_hash") else None,
        "scoring_hash": bench.scoring_hash if bench is not None and bench.protocol_hash == cfg.get("protocol_hash") else None,
        "manifest_hash": cfg.get("manifest_hash"), "probe_hash": cfg.get("probe_hash"), "eval_hash": eval_hash,
        "standard": bool(probe.get("standard", False)),
    }
    describe = dict(bridge.get("describe") or {})
    coverage = first.get("coverage") if isinstance(first.get("coverage"), dict) else None
    if coverage is None:
        roles = dict(cfg.get("roles") or {})
        coverage = {role: ({"present": v.get("n"), "nominal": None, "ids_sha256": None} if isinstance(v, dict) else None) for role, v in roles.items()}
        if res.get("coverage_eval") is not None and isinstance(coverage.get("eval"), dict):
            coverage["eval"]["fraction"] = res["coverage_eval"]
    return {
        "registry_version": REGISTRY_ENTRY_VERSION,
        "id": entry_id(benchmark, eval_hash, org, model, reading),
        "identity": identity,
        "model": {
            "name": f"{org}/{model}", "org": org, "model": model, "bridge": bridge.get("name"),
            "weights_ref": describe.get("weights_ref"), "weights_sha256": describe.get("weights_sha256"),
            "describe_sha256": bridge.get("describe_sha256") or cfg.get("describe_sha256"), "source_sha256": bridge.get("source_sha256"),
            "describe": describe,
        },
        "score": {
            "headline": res.get("headline") or metrics_cfg.get("headline"),
            "metrics": {m: res.get(m) for m in names},
            "ci95": {m: res[f"{m}_ci95"] for m in names if f"{m}_ci95" in res},
            "std": {m: res[f"{m}_std"] for m in names if f"{m}_std" in res},
            "floors": {m: res[f"{m}_floor"] for m in names if f"{m}_floor" in res},
            "floor_names": floor_names,
            "controls": {c: res.get(c) for c in control_names},
            "n_seeds": int(res.get("n_seeds") or len(seeds) or 1),
            "seeds": seeds,
            "higher_is_better": dict((doc.get("higher_is_better") or {}).get(key) or {}),
            "waived": list(first.get("waived") or []),
        },
        "population": {"coverage": coverage, "features_fingerprint": cfg.get("features_fingerprint")},
        "conformance": _conformance_block(bench, cfg, conformance_cache),
        "provenance": {
            "yavih_version": doc.get("yavih_version"), "commit": doc.get("git_hash"), "date": doc.get("date"),
            "folder_source": cfg.get("folder_source"), "external": dict(cfg.get("external") or {}) if isinstance(cfg.get("external"), dict) else cfg.get("external"),
            "probe_config": probe.get("config"), "results_file": str(source), "results_file_sha256": _sha256_file(Path(source)) if Path(source).is_file() else None,
            "unit_results": list(cfg.get("unit_results") or []), "env": doc.get("env"),
            "exported_with": {"yavih_version": __version__}, "replaces": None,
        },
    }


def protocol_payload(bench: Any) -> dict[str, Any]:
    """``protocols/<benchmark>/<protocol_hash12>.json``: the hashed payload and the fixture, self-contained."""
    from yavih import conformance
    from yavih.versions import writer_stamp

    return {
        "benchmark": bench.name, "protocol_version": int(bench.spec.protocol_version), "protocol_hash": bench.protocol_hash,
        "input_hash": bench.input_hash, "scoring_hash": bench.scoring_hash, "manifest_hash": bench.manifest_hash,
        "min_eval_coverage": float(bench.spec.splits.min_eval_coverage),
        "metrics": {"headline": bench.spec.metrics.headline, "list": list(bench.spec.metrics.list), "floors": list(bench.spec.metrics.floors),
                    "controls": list(bench.spec.metrics.controls), "tolerance": conformance.tolerances(bench)},
        "explain": bench.hashes.explain(),
        "expected": conformance.read_expected(bench),
        "folder_source": dict(bench.folder_source),
        "supersedes": [h for e in bench.spec.changelog for h in (e.supersedes or ())],
        "written_with": writer_stamp(),
    }


# --------------------------------------------------------------------------- export


def _results_files(source: str | os.PathLike) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise RegistryError(E_SCHEMA, f"{p}: not a results file or a results root")
    files = sorted(p.glob("*/results_*.json")) or sorted(p.glob("results_*.json"))
    if not files:
        raise RegistryError(E_SCHEMA, f"{p}: no results_*.json under it (a results root holds <model>/results_*.json)")
    return files


def _load_bench(name: str, cache: dict[str, Any]) -> Any | None:
    if name in cache:
        return cache[name]
    try:
        from yavih.benchmark import Benchmark

        cache[name] = Benchmark.load(name)
    except Exception:  # noqa: BLE001 - an unresolvable or broken folder: the entry records that nothing could be verified
        cache[name] = None
    return cache[name]


def _write_json(path: Path, obj: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def export(source: str | os.PathLike, *, org: str, out: str | os.PathLike, include_nonstandard: bool = False,
           models: Iterable[str] | None = None) -> dict[str, Any]:
    """Write registry entries under ``out`` from every ok, single-reading entry of the results files under
    ``source`` (a results root ``<root>/<model>/results_*.json`` or one file), plus the protocol payload of every
    benchmark whose folder resolves here at the entry's hash. A non-standard probe is skipped unless
    ``include_nonstandard``; a resubmission replaces the file and records the previous one's sha256 under
    ``provenance.replaces``. Returns ``{"entries": [paths], "protocols": [paths], "skipped": [{key, reason}]}``."""
    from yavih.report.store import upgrade_results_file

    if not _ORG_RE.match(org):
        raise RegistryError(E_SCHEMA, f"org {org!r}: [A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}")
    out_root = Path(out)
    want = set(models) if models is not None else None
    bench_cache: dict[str, Any] = {}
    conformance_cache: dict[str, Any] = {}  # one fixture check per benchmark folder and hash
    written: list[str] = []
    protocols: list[str] = []
    skipped: list[dict[str, str]] = []
    for path in _results_files(source):
        try:
            doc = upgrade_results_file(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            skipped.append({"key": str(path), "reason": f"unreadable: {e}"})
            continue
        unit_base = path.parent.parent.parent if path.parent.parent.name else None  # <root>/<model>/file -> the directory above the results root
        for key, cfg in sorted((doc.get("configs") or {}).items()):
            res = (doc.get("results") or {}).get(key) or {}
            model = str(cfg.get("model") or (doc.get("config") or {}).get("model") or "")
            if want is not None and model not in want:
                continue
            if res.get("status") != "ok":
                skipped.append({"key": f"{path}:{key}", "reason": f"status {res.get('status')!r}"})
                continue
            if "+" in str(cfg.get("reading") or ""):
                skipped.append({"key": f"{path}:{key}", "reason": "a join entry"})
                continue
            if not bool((cfg.get("probe") or {}).get("standard", False)) and not include_nonstandard:
                skipped.append({"key": f"{path}:{key}", "reason": "non-standard probe (pass --include-nonstandard to register it as such)"})
                continue
            bench = _load_bench(str(cfg.get("benchmark")), bench_cache)
            try:
                entry = entry_from_results(doc, key, org=org, source=path, bench=bench, unit_base=unit_base,
                                           conformance_cache=conformance_cache)
            except RegistryError as e:
                skipped.append({"key": f"{path}:{key}", "reason": e.message})
                continue
            dest = entry_path(out_root, entry)
            if dest.is_file():
                entry["provenance"]["replaces"] = _sha256_file(dest)
            written.append(str(_write_json(dest, entry)))
            if bench is not None and bench.protocol_hash == cfg.get("protocol_hash"):
                pp = protocol_path(out_root, bench.name, bench.protocol_hash)
                if not pp.is_file():
                    protocols.append(str(_write_json(pp, protocol_payload(bench))))
    return {"entries": written, "protocols": protocols, "skipped": skipped}


# --------------------------------------------------------------------------- check


_ENTRY_KEYS = frozenset({"registry_version", "id", "identity", "model", "score", "population", "conformance", "provenance"})
_IDENTITY_KEYS = frozenset({"benchmark", "reading", "protocol_version", "protocol_hash", "probe_hash", "eval_hash", "standard"})


def read_entries(root: str | os.PathLike) -> list[tuple[Path, dict[str, Any]]]:
    """Every ``entries/**/*.json`` under ``root`` (unreadable ones raise ``RegistryError``), sorted by path."""
    base = Path(root) / ENTRIES_DIR
    out: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(base.rglob("*.json")) if base.is_dir() else []:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RegistryError(E_SCHEMA, f"{p}: {e}") from e
        out.append((p, d))
    return out


def _recompute_protocol_hash(explain: Mapping[str, Any]) -> str | None:
    """``protocol_hash`` from a ``bench hash --explain`` payload with stdlib only: what a third party recomputes."""
    try:
        schema = explain["schema"]
        sides = {}
        for side in ("input", "scoring"):
            digests = {k: hashlib.sha256(c["canon"].encode("utf-8")).hexdigest() for k, c in explain[side]["components"].items()}
            sides[side] = hashing.sha256_canon({"schema": schema, "part": side, "components": digests})
        return hashing.sha256_canon({"schema": schema, "input_hash": sides["input"], "scoring_hash": sides["scoring"]})
    except (KeyError, TypeError, AttributeError):
        return None


def check(root: str | os.PathLike) -> RegistryReport:
    """The registry repository's CI check: every entry parses, its id matches its path, its hashes agree, its
    protocol payload is present and recomputes the hash, its conformance passed (nothing skipped or unverified is
    accepted), the tolerance is recorded; a non-standard probe or low coverage is a warning (shown as such, never
    ranked)."""
    rep = RegistryReport(root=str(root))
    try:
        entries = read_entries(root)
    except RegistryError as e:
        rep.findings.append(Finding("error", e.code, e.message))
        return rep
    rep.n_entries = len(entries)
    pdir = Path(root) / PROTOCOLS_DIR
    rep.n_protocols = sum(1 for _ in pdir.rglob("*.json")) if pdir.is_dir() else 0
    payloads: dict[Path, dict[str, Any] | None] = {}
    for path, e in entries:
        rel = str(path.relative_to(Path(root)))
        if not isinstance(e, dict) or not _ENTRY_KEYS <= set(e) or not isinstance(e.get("identity"), dict) or not _IDENTITY_KEYS <= set(e["identity"]):
            rep.findings.append(Finding("error", E_SCHEMA, f"{rel}: not a registry entry (keys {sorted(_ENTRY_KEYS)}, identity {sorted(_IDENTITY_KEYS)})", rel))
            continue
        if int(e.get("registry_version", -1)) > REGISTRY_ENTRY_VERSION:
            rep.findings.append(Finding("error", E_SCHEMA, f"{rel}: registry_version {e['registry_version']} is newer than this yavih ({REGISTRY_ENTRY_VERSION})", rel))
            continue
        i, m = e["identity"], e.get("model") or {}
        want_id = entry_id(str(i["benchmark"]), str(i["eval_hash"]), str(m.get("org", "")), str(m.get("model", "")), str(i["reading"]))
        if e.get("id") != want_id:
            rep.findings.append(Finding("error", E_ID, f"{rel}: id {e.get('id')!r} != {want_id!r} (benchmark/eval_hash12/org/model/reading)", rel))
        try:
            if entry_path(root, e).resolve() != path.resolve():
                rep.findings.append(Finding("error", E_ID, f"{rel}: filed at the wrong path (expected {entry_path(root, e).relative_to(Path(root))})", rel))
        except (KeyError, TypeError):
            rep.findings.append(Finding("error", E_ID, f"{rel}: model block lacks org / model", rel))
        ph, qh, eh = i.get("protocol_hash"), i.get("probe_hash"), i.get("eval_hash")
        if not (isinstance(ph, str) and isinstance(qh, str) and isinstance(eh, str)) or hashing.eval_hash(ph, qh) != eh:
            rep.findings.append(Finding("error", E_HASH, f"{rel}: eval_hash is not sha256(canon({{protocol_hash, probe_hash}}))", rel))
            continue
        pp = protocol_path(root, str(i["benchmark"]), ph)
        if pp not in payloads:
            try:
                payloads[pp] = json.loads(pp.read_text(encoding="utf-8")) if pp.is_file() else None
            except (OSError, ValueError):
                payloads[pp] = None
        payload = payloads[pp]
        if payload is None:
            rep.findings.append(Finding("error", E_PROTOCOL, f"{rel}: no protocol payload {pp.relative_to(Path(root))} (export from a yavih that resolves the folder at this hash)", rel))
        else:
            got = _recompute_protocol_hash(payload.get("explain") or {})
            if got != ph or payload.get("protocol_hash") != ph:
                rep.findings.append(Finding("error", E_PROTOCOL, f"{rel}: the protocol payload does not recompute protocol_hash {hashing.hash12(ph)}…", rel))
            tol = ((payload.get("metrics") or {}).get("tolerance") or {})
            if not tol:
                rep.findings.append(Finding("error", E_SCHEMA, f"{rel}: the protocol payload records no tolerance", rel))
            cov = ((e.get("population") or {}).get("coverage") or {}).get("eval") or {}
            frac = None
            if isinstance(cov, dict):
                if cov.get("present") is not None and cov.get("nominal"):
                    frac = float(cov["present"]) / float(cov["nominal"])
                elif cov.get("fraction") is not None:
                    frac = float(cov["fraction"])
            if frac is not None and frac < float(payload.get("min_eval_coverage") or 0.0):
                rep.findings.append(Finding("warning", W_COVERAGE, f"{rel}: eval coverage {frac:.3f} below the folder's minimum {payload.get('min_eval_coverage')}: shown as partial, never ranked", rel))
        c = e.get("conformance") or {}
        if c.get("passed") is not True:
            rep.findings.append(Finding("error", E_CONFORMANCE, f"{rel}: conformance not passed ({'skipped' if c.get('skipped') else c.get('reason') or 'absent'}): an entry is accepted only with the folder's fixture passed by the exporting yavih", rel))
        if not i.get("standard"):
            rep.findings.append(Finding("warning", W_NONSTANDARD, f"{rel}: non-standard probe: comparable with nothing, shown as (nonstandard)", rel))
        if not (e.get("score") or {}).get("metrics"):
            rep.findings.append(Finding("error", E_SCHEMA, f"{rel}: no score.metrics", rel))
    return rep


# --------------------------------------------------------------------------- report records


def records(root: str | os.PathLike, *, accepted_only: bool = True) -> list[dict[str, Any]]:
    """Index-like records (``{"path", "result", "registry": {...}}``) for ``yavih.report.compare.build_report``: one
    ``result.json``-shaped record per seed when the entry carries per-seed numbers, else one carrying the aggregate
    (``registry.n_seeds`` keeps the seed count honest). With ``accepted_only`` entries whose conformance did not
    pass are left out (what :func:`check` refuses)."""
    from yavih.report.compare import floor_name_for

    out: list[dict[str, Any]] = []
    for path, e in read_entries(root):
        i, m, s, prov = e["identity"], e["model"], e["score"], e.get("provenance") or {}
        if accepted_only and (e.get("conformance") or {}).get("passed") is not True:
            continue
        names = list((s.get("metrics") or {}).keys())
        floor_names = list(s.get("floor_names") or [])
        floors = {floor_name_for(mn, names, floor_names) or f"{mn}_floor": v for mn, v in (s.get("floors") or {}).items()}
        coverage = (e.get("population") or {}).get("coverage") or {}
        base = {
            "schema_version": 4, "status": "ok", "reason": None, "standard": bool(i.get("standard")), "eval_hash": i["eval_hash"],
            "benchmark": {"name": i["benchmark"], "protocol_version": i.get("protocol_version"), "readout": i.get("readout"),
                          "readout_version": i.get("readout_version"), "manifest_hash": i.get("manifest_hash"), "protocol_hash": i["protocol_hash"],
                          "folder_source": prov.get("folder_source")},
            "model": str(m.get("name") or f"{m.get('org')}/{m.get('model')}"), "reading": i["reading"],
            "bridge": {"name": m.get("bridge"), "describe": m.get("describe") or {}, "describe_sha256": m.get("describe_sha256"), "source_sha256": m.get("source_sha256")},
            "features": {"fingerprint": (e.get("population") or {}).get("features_fingerprint"), "dtype": (m.get("describe") or {}).get("dtype"),
                         "N": None, "D": None, "pos_mode": None, "bytes": None, "backend": None, "canary_max_rel_l2": None, "batch_tolerance": None, "rules_checked": None},
            "chance": {}, "coverage": coverage, "not_applicable": None, "timing": {}, "env": prov.get("env") or {}, "created": str(prov.get("date") or ""),
            "warnings": [], "external": prov.get("external") or {}, "waived": list(s.get("waived") or []),
            "registry": {"id": e.get("id"), "org": m.get("org"), "n_seeds": int(s.get("n_seeds") or 1), "path": str(path),
                         # what the report's ties read when the folder does not resolve here
                         "tolerance": dict((e.get("conformance") or {}).get("tolerance") or {}),
                         "higher_is_better": dict(s.get("higher_is_better") or {})},
        }
        per_seed = [sd for sd in (s.get("seeds") or []) if isinstance(sd.get("metrics"), dict) and sd["metrics"]]
        if per_seed:
            for sd in per_seed:
                r = dict(base, probe={"hash": i["probe_hash"], "seed": int(sd.get("seed", 0)), "config": prov.get("probe_config")},
                         metrics=sd["metrics"], ci95=sd.get("ci95") or {}, floors=sd.get("floors") or {}, controls=sd.get("controls") or {})
                r["registry"] = dict(base["registry"], n_seeds=1)
                r["created"] = str(sd.get("created") or base["created"])
                out.append({"path": str(path), "mtime_ns": 0, "size": 0, "result": r})
        else:
            seed0 = int(((s.get("seeds") or [{}])[0]).get("seed", 0))
            r = dict(base, probe={"hash": i["probe_hash"], "seed": seed0, "config": prov.get("probe_config")},
                     metrics=dict(s.get("metrics") or {}), ci95=dict(s.get("ci95") or {}), floors=floors, controls=dict(s.get("controls") or {}))
            out.append({"path": str(path), "mtime_ns": 0, "size": 0, "result": r})
    return out
