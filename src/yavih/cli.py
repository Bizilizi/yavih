"""``yavih`` command line: an argparse mirror of the Python primitives with ``--json``.

Every command accepts ``--json`` (also ``YAVIH_JSON=1``): stdout is exactly one object
``{"status": "ok"|"error"|"skipped"|"not_applicable", "data": {...}, "errors": [Finding],
"warnings": [Finding], "paths": {name: path}}``; logs go to stderr. Exit codes: see ``EXIT_*`` below.

The CLI is thin: parse → call the library function → ``Envelope`` → exit code. Core imports only at
module scope; extras (torch, decoders) are imported lazily inside the handlers or by the
modules a subcommand dispatches to.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import socket
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yavih.versions import __version__

__all__ = [
    "main",
    "build_parser",
    "Envelope",
    "CliError",
    "exit_code_for",
    "EXIT_OK",
    "EXIT_USAGE",
    "EXIT_CONTRACT",
    "EXIT_NOT_READY",
    "EXIT_DEADLINE",
    "EXIT_INTERNAL",
    "STATUSES",
    "EXTRAS",
]

# Exit codes
EXIT_OK = 0  # done, skipped, not_applicable recorded
EXIT_USAGE = 1  # usage / configuration error
EXIT_CONTRACT = 2  # contract violation the agent must fix
EXIT_NOT_READY = 3  # data missing, dependency DONE missing, decoder missing
EXIT_DEADLINE = 4  # deadline reached; nothing partial; resumable
EXIT_INTERNAL = 5  # internal / unexpected failure

STATUSES: frozenset[str] = frozenset({"ok", "error", "skipped", "not_applicable"})

#: pip extras → (import names, pip hint); ``doctor`` reports each block. There is no extra
#: for an original-author package: ``doctor --bench B`` reports the
#: benchmark's ``external`` block instead.
EXTRAS: dict[str, tuple[tuple[str, ...], str]] = {
    "core": (("numpy", "yaml"), "pip install 'numpy>=1.24' 'pyyaml>=6' (an install with --no-deps assumes both)"),
    "media": (("av", "soundfile", "PIL", "scipy"), "pip install av soundfile pillow scipy (= the [media] extra; optional)"),
    "probe": (("torch",), "pip install torch (the probe env; = the [probe] extra)"),
}
_DECODER_KINDS: tuple[str, ...] = ("video", "audio", "image")
_SKILL_FILES_SKIP = {"__pycache__", ".gitkeep"}


class CliError(Exception):
    """A CLI-level failure with an exit code (``EXIT_*``) and an ``E_*`` code."""

    def __init__(self, code: str, message: str, exit_code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


@dataclass
class Envelope:
    """The ``--json`` envelope. ``text`` is the human rendering printed without ``--json``."""

    status: str = "ok"
    data: Any = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    exit_code: int = EXIT_OK
    text: str | None = None

    def to_json(self) -> dict:
        if self.status not in STATUSES:
            raise ValueError(f"envelope status must be one of {sorted(STATUSES)}, got {self.status!r}")
        return {
            "status": self.status,
            "data": _jsonable(self.data),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "paths": dict(self.paths),
        }


def _finding(code: str, message: str, where: str | None = None) -> dict:
    return {"code": code, "message": message, "where": where}


def _jsonable(x: Any) -> Any:
    """JSON-safe copy: Paths → str, tuples/sets → lists, numpy scalars/arrays → Python (duck-typed)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "tolist") and callable(x.tolist):
        return _jsonable(x.tolist())
    if hasattr(x, "item") and callable(x.item) and not isinstance(x, (str, bytes)):
        try:
            return x.item()
        except Exception:  # noqa: BLE001
            return str(x)
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x


def exit_code_for(exc: BaseException) -> tuple[int, str]:
    """Map an exception to ``(exit_code, E_code)``. Modules may carry their own
    ``exit_code``/``code`` attributes (``ExtractError``, ``ProbeRefused``, ``CliError``); otherwise the
    class decides: schema/config/usage → 1, contract (validation, manifest, describe, format) → 2,
    not-ready (missing data/decoder/dependency) → 3, deadline → 4, anything else (incl. a stub's
    ``NotImplementedError``) → 5."""
    code = getattr(exc, "code", None)
    ec = getattr(exc, "exit_code", None)
    if isinstance(ec, int) and isinstance(code, str):
        return ec, code
    # Lazy imports keep the CLI's module scope core-only and tolerant of partially built modules.
    from yavih.benchmark import DataNotReady, ManifestError, SchemaError, UnknownBenchmark
    from yavih.bridge import DescribeError
    from yavih.config import ConfigError
    from yavih.io.format import FeatureShapeError, FeatureValueError, FormatError
    from yavih.io.validate import ValidationError
    from yavih.media import DecoderMissing, MediaMissing

    if isinstance(exc, NotImplementedError):
        return EXIT_INTERNAL, "E_NOT_IMPLEMENTED"
    if isinstance(exc, (SchemaError, ConfigError, UnknownBenchmark, argparse.ArgumentError)):
        return EXIT_USAGE, "E_USAGE"
    if isinstance(exc, (ValidationError, ManifestError, DescribeError, FormatError, FeatureShapeError, FeatureValueError)):
        return EXIT_CONTRACT, "E_CONTRACT"
    if isinstance(exc, (DataNotReady, DecoderMissing, MediaMissing)):
        return EXIT_NOT_READY, "E_NOT_READY"
    if isinstance(exc, TimeoutError):
        return EXIT_DEADLINE, "E_DEADLINE"
    if isinstance(exc, (KeyboardInterrupt,)):
        return EXIT_DEADLINE, "E_INTERRUPTED"
    return EXIT_INTERNAL, "E_INTERNAL"


# ----------------------------------------------------------------------------- shared helpers

Handler = Callable[[argparse.Namespace], Envelope]


def _kv_pairs(items: Sequence[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or ():
        if "=" not in item:
            raise CliError("E_USAGE", f"expected k=v, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _shard(text: str) -> tuple[int, int]:
    try:
        i, n = text.split("/", 1)
        shard = (int(i), int(n))
    except ValueError as e:
        raise CliError("E_USAGE", f"--shard expects i/n, got {text!r}") from e
    if not (0 <= shard[0] < shard[1]):
        raise CliError("E_USAGE", f"--shard i/n needs 0 <= i < n, got {text!r}")
    return shard


def _deadline(spec: str | None) -> float | None:
    """``--deadline`` through the one shared parser, ``yavih.deadline.parse_deadline``."""
    from yavih.deadline import parse_deadline

    return parse_deadline(spec)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute-or-key access (spec dataclasses or raw yaml mappings)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _raw(bench: Any) -> dict:
    """The parsed ``benchmark.yaml`` of a loaded ``Benchmark`` (``spec.raw``), as a plain dict."""
    spec = getattr(bench, "spec", None)
    raw = getattr(spec, "raw", None)
    if isinstance(raw, dict) and raw:
        return raw
    if isinstance(spec, dict):
        return spec
    import yaml

    with open(Path(bench.folder) / "benchmark.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_bench(ref: str, *, data: dict | None = None, fetch: bool = False) -> Any:
    from yavih.benchmark import Benchmark

    return Benchmark.load(ref, data=data, fetch=fetch)


def _bench_roles(bench: Any) -> dict[str, list[str]]:
    roles = bench.roles()
    return {k: list(v or []) for k, v in roles.items()}


def _data_roots(bench: Any) -> dict[str, dict[str, Any]]:
    """``{name: {path, exists, source, env_var, config_key, prepare_only, error}}`` for ``doctor``,
    resolved by ``yavih.config``; an unresolvable root is reported (``path: None``, ``error``), never raised."""
    from yavih.config import resolve_data_sources

    raw = _raw(bench)
    roots = (raw.get("data") or {}).get("roots") or {}
    spec = getattr(bench, "spec", None)
    aliases = tuple(getattr(spec, "aliases", None) or raw.get("aliases") or ())
    recs = resolve_data_sources(bench.name, roots, getattr(bench, "data_overrides", None), aliases=aliases, strict_missing=False)
    out: dict[str, dict[str, Any]] = {}
    for name, rec in recs.items():
        r = roots.get(name)
        path = rec.get("value")
        out[name] = {"path": path, "exists": bool(path) and os.path.exists(path), "source": rec.get("source"),
                     "env_var": rec.get("env_var"), "config_key": rec.get("config_key"), "error": rec.get("error"),
                     "prepare_only": bool(r.get("prepare_only")) if isinstance(r, dict) else False}
    return out


def _root_findings(bench_name: str, roots: dict[str, dict[str, Any]], *, edit_hint: str) -> tuple[list[dict], list[dict]]:
    """``(warnings, notes)`` for data roots that do not resolve or do not exist; each names the config key and env
    var that set the root. A missing ``prepare_only`` root is a note (only ``prepare/`` reads it)."""
    warnings: list[dict] = []
    notes: list[dict] = []
    for name, root in roots.items():
        how = f"set {root['config_key']} in {edit_hint}, export {root['env_var']}, or pass --data {bench_name}.{name}=PATH"
        if root.get("error"):
            warnings.append(_finding("W_DATA_ROOT", f"data root {name!r} does not resolve: {root['error']}", where=name))
        elif not root["exists"]:
            msg = f"data root {name!r} does not exist: {root['path']} (from {root['source']}); {how}"
            if root.get("prepare_only"):
                notes.append(_finding("N_DATA_ROOT_PREPARE", msg + " (only prepare/ reads this root)", where=name))
            else:
                warnings.append(_finding("W_DATA_ROOT", msg, where=name))
    return warnings, notes


def _read_streams(sample: Any) -> dict[str, dict]:
    """Read every stream's stored bytes and facts once (``yavih.sample.fetch`` per stream):
    ``{stream: {ok, kind, bytes, info, error}}``. Nothing is decoded: what the reader hands over is the file."""
    out: dict[str, dict] = {}
    for name, stream in sample.streams.items():
        rec: dict[str, Any] = {"kind": stream.kind, "ok": True, "bytes": None, "info": None, "error": None}
        try:
            rec["bytes"] = len(stream.bytes())
            rec["info"] = dict(stream.info)
        except Exception as e:  # noqa: BLE001 - a read failure is a reported fact, not a crash
            rec["ok"] = False
            rec["error"] = f"{e.__class__.__name__}: {e}"
        out[name] = rec
    return out


def _skill_root() -> Path:
    """The packaged ``yavih/skill/yavih/`` directory via ``importlib.resources``."""
    from importlib import resources

    return Path(str(resources.files("yavih") / "skill" / "yavih"))


def _iter_skill_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part in _SKILL_FILES_SKIP for part in p.relative_to(root).parts):
            yield p


def _import_ok(names: Iterable[str]) -> list[str]:
    import importlib.util

    return [n for n in names if importlib.util.find_spec(n) is None]


def _dist_info_mtime(dist_name: str) -> float | None:
    from importlib import metadata

    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return None
    path = getattr(dist, "_path", None)
    try:
        return os.path.getmtime(str(path)) if path is not None else None
    except OSError:
        return None


def _torch_block() -> dict[str, Any]:
    """``{version, cuda, untouched}`` for ``doctor``. ``untouched`` = yavih's install did not
    (re)install torch: torch is absent, or its dist-info predates yavih's. The model env must keep its own
    torch (``pip install --no-deps yavih``)."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return {"version": None, "cuda": None, "untouched": True}
    import torch  # lazy: [probe] extra / the model env's own torch

    torch_mtime = _dist_info_mtime("torch")
    yavih_mtime = _dist_info_mtime("yavih")
    untouched = True
    if torch_mtime is not None and yavih_mtime is not None:
        untouched = torch_mtime <= yavih_mtime
    return {"version": str(torch.__version__), "cuda": getattr(torch.version, "cuda", None), "untouched": bool(untouched)}


def _gpu_block() -> list[dict[str, Any]]:
    import importlib.util

    gpus: list[dict[str, Any]] = []
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({"index": i, "name": props.name, "memory_gb": round(props.total_memory / 2**30, 1)})
        return gpus
    exe = shutil.which("nvidia-smi")
    if exe:
        import subprocess

        try:
            out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        for i, line in enumerate(line for line in out.splitlines() if line.startswith("GPU")):
            gpus.append({"index": i, "name": line.split(":", 1)[1].split("(")[0].strip(), "memory_gb": None})
    return gpus


def _to_numpy(x: Any) -> Any:
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


# ----------------------------------------------------------------------------- doctor


def _external_block(bench: Any) -> dict[str, dict[str, Any]]:
    """``{name: {ok, version, version_pin, pip}}`` for ``benchmark.yaml[external]`` (``doctor``): the
    authors' packages a folder's ``probe.py`` assert-imports; never yavih dependencies."""
    from yavih.external import status

    return {str(_get(e, "name")): status(str(_get(e, "name")), str(_get(e, "pip")), str(_get(e, "version_pin")))
            for e in list(_get(bench.spec, "external") or [])}


def _doctor_split(bench: Any) -> str | None:
    """The split ``doctor`` reads one row from: the first split with rows among eval, train, folds, floor
    (a folds-only benchmark such as esc50 has no eval/train split)."""
    roles = _bench_roles(bench)
    for role in ("eval", "train", "folds", "floor"):
        for split in roles.get(role) or []:
            try:
                if bench.n_rows(split) > 0:
                    return split
            except Exception:  # noqa: BLE001 - a broken manifest split is reported by bench verify
                continue
    return None


_VERSION_RE = re.compile(r"version (\S+)")


def _tool_pin_check(path: str, name: str, pin: Any, *, timeout: float) -> tuple[str | None, str | None]:
    """``(found, problem)`` for one ``materialize.tool.pins`` entry: run its ``probe`` command (``{tool}`` = the tool's
    path) and take the first match of ``match`` in its output."""
    import subprocess

    cmd = [str(a).replace("{tool}", path) for a in _get(pin, "probe") or []]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"{name}: probe failed ({e})"
    m = re.search(str(_get(pin, "match")).encode(), (proc.stdout or b"") + (proc.stderr or b""))
    found = m.group(0).decode(errors="replace") if m else None
    want = str(_get(pin, "value"))
    return found, None if found == want else f"{name} {found or 'unavailable'} != pinned {want!r}"


def _tools_block(bench: Any, *, timeout: float = 30.0) -> dict[str, dict[str, Any]]:
    """``input_contract.materialize.tool``: the program on this host against its ``version_pin``, and every
    ``tool.pins`` entry (a further build the recipe's bytes depend on, e.g. ffmpeg's libx264) through the probe
    command and regex the yaml declares, so a host whose build differs from the pin is reported. Empty when the
    benchmark materializes nothing."""
    import subprocess

    ic = _get(bench.spec, "input_contract")
    mat = _get(ic, "materialize") if ic is not None else None
    tool = _get(mat, "tool") if mat else None
    if not tool:
        return {}
    name = str(_get(tool, "name"))
    pin = _get(tool, "version_pin")
    rec: dict[str, Any] = {"path": shutil.which(name), "version": None, "version_pin": pin, "ok": False, "problem": None}
    if rec["path"] is None:
        rec["problem"] = f"{name} is not on PATH (pin {pin})"
        return {name: rec}
    try:
        out = subprocess.run([name, "-version"], capture_output=True, text=True, timeout=timeout).stdout
        m = _VERSION_RE.search(out.splitlines()[0] if out else "")
        rec["version"] = m.group(1) if m else None
    except (OSError, subprocess.SubprocessError) as e:
        rec["problem"] = f"`{name} -version` failed: {e}"
        return {name: rec}
    problems = []
    if pin is not None and rec["version"] != str(pin):
        problems.append(f"{name} {rec['version']} at {rec['path']} != pinned {pin}")
    pins = dict(_get(tool, "pins") or {})
    if pins:
        rec["pins"] = {}
        for pname, p in pins.items():
            found, problem = _tool_pin_check(rec["path"], pname, p, timeout=timeout)
            rec["pins"][pname] = {"found": found, "pin": _get(p, "value"), "ok": problem is None}
            if problem:
                problems.append(problem)
    rec["ok"] = not problems
    rec["problem"] = "; ".join(problems) or None
    return {name: rec}


def cmd_doctor(ns: argparse.Namespace) -> Envelope:
    """``yavih doctor [--bench B] [--selftest]``: extras, torch, decoders, decode sample, data
    roots, GPUs, launcher, decode-worker default, ``$YAVIH_HOME`` and the runs root."""
    from yavih import config as _cfg
    from yavih.extract import default_decode_workers
    from yavih.media import decoder_info

    warnings: list[dict] = []
    data: dict[str, Any] = {}
    # the configuration (files, site, every value with its source) is a finding, never an abort
    settings = None
    try:
        settings = _cfg.load()
    except _cfg.ConfigError as e:
        warnings.append(_finding("W_CONFIG", f"{e}"))
        try:
            settings = _cfg.load(lenient=True)
        except _cfg.ConfigError as e2:
            warnings.append(_finding("W_CONFIG", f"even leniently the configuration does not load: {e2}"))
    if settings is not None:
        for f in settings.files:
            if f.layer == "project" and not f.exists:
                warnings.append(_finding("W_CONFIG_MISSING", f"{f.how} names {f.path}, which does not exist (no project config is read)"))
    data["notes"] = []
    for extra, (mods, hint) in EXTRAS.items():
        missing = _import_ok(mods)
        data[extra] = {"ok": not missing, "missing": missing, "pip_hint": hint}
        if missing:
            warnings.append(_finding("W_EXTRA_MISSING", f"[{extra}] extra incomplete: missing {missing}; {hint}"))
    data["torch"] = _torch_block()
    if not data["torch"]["untouched"]:
        warnings.append(_finding("W_TORCH_TOUCHED", "torch dist-info is newer than yavih's: pip may have replaced the model env's torch (install with --no-deps)"))
    decoders: dict[str, Any] = {}
    for kind in _DECODER_KINDS:
        try:
            decoders[kind] = decoder_info(kind)
        except Exception as e:  # noqa: BLE001 - doctor reports, it does not crash on one backend
            decoders[kind] = None
            warnings.append(_finding("W_DECODER", f"decoder_info({kind!r}) failed: {e.__class__.__name__}: {e}"))
    data["decoders"] = decoders  # the optional yavih.media.decode helpers on this host (a bridge may use its own stack)
    data["media_read"] = None
    data["media_info"] = None
    data["data_roots"] = {}
    data["external"] = {}
    data["tools"] = {}
    bench = _load_bench(ns.bench) if ns.bench else None  # an unknown name stays a usage error (exit 1)
    if bench is not None:
        data["external"] = _external_block(bench)
        for name, ext in data["external"].items():
            if not ext["ok"]:
                warnings.append(_finding("W_EXTERNAL", f"{name}: installed {ext['version'] or 'nothing'}, the protocol pins {ext['version_pin']}; "
                                         f"install it independently (needed in the probe env only): {ext['pip']}", where=name))
        data["benchmark"] = {"name": bench.name, "protocol_hash": bench.protocol_hash, "input_hash": bench.input_hash,
                             "readout": _get(bench.spec, "readout")}
        try:
            data["data_roots"] = _data_roots(bench)
        except _cfg.ConfigError as e:  # an unknown configured root of this benchmark (strict schema)
            warnings.append(_finding("W_CONFIG", str(e), where=bench.name))
            data["data_roots"] = {}
        w, n = _root_findings(bench.name, data["data_roots"], edit_hint=settings.edit_hint() if settings is not None else "the config")
        warnings += w
        data["notes"] += n
        split = _doctor_split(bench)
        # read one row's stored bytes and facts (the reader never decodes; the bridge does)
        read_info: dict[str, Any] = {"clip_id": None, "ok": False, "split": split, "streams": {}}
        if split is None:
            # no split with manifest rows to read: warn instead of reporting a silent ok=False
            read_info["error"] = "no eval, train, fold or floor split with manifest rows to read"
            warnings.append(_finding("W_MEDIA_READ", read_info["error"], where=bench.name))
        else:
            try:
                sample = next(iter(bench.reader(split, rows=range(0, 1))))
                streams = _read_streams(sample)
                read_info.update(clip_id=sample.clip_id, streams=streams, ok=all(s["ok"] for s in streams.values()))
                for sname, s in streams.items():
                    if not s["ok"]:
                        warnings.append(_finding("W_MEDIA_READ", f"{sname}: {s['error']}", where=sample.clip_id))
            except Exception as e:  # noqa: BLE001
                read_info["error"] = f"{e.__class__.__name__}: {e}"
                warnings.append(_finding("W_MEDIA_READ", read_info["error"], where=split))
        data["media_read"] = read_info
        table = None
        try:
            table = bench.media_info()
        except Exception as e:  # noqa: BLE001 - a broken census is reported, not fatal
            warnings.append(_finding("W_MEDIA_INFO", f"media-info table unreadable: {e.__class__.__name__}: {e}", where=bench.name))
        data["media_info"] = {"present": table is not None, "sha256": getattr(table, "sha256", None), "path": getattr(table, "path", None)}
        # without a census every row's facts come from a live header probe (PyAV / soundfile / pillow, else
        # ffprobe); a stream kind with none of them fails every row with DecoderMissing (extract refuses at start).
        from yavih.media.info import header_backends

        kinds = sorted({str(_get(s, "kind", s)) for s in (_get(bench.spec, "streams") or {}).values()})
        data["media_info"]["header_backends"] = {k: header_backends(k) for k in kinds}
        lacking = [k for k, b in data["media_info"]["header_backends"].items() if not b]
        if table is None and lacking:
            warnings.append(_finding("W_HEADER_BACKEND", f"no media-info census and no header backend for stream kind(s) {lacking}: every "
                                     "row would fail DecoderMissing; pip install av soundfile pillow (or 'yavih[media]'), put ffprobe on "
                                     f"PATH, or copy a census made elsewhere (`yavih bench verify {bench.name} --census`) into "
                                     "$YAVIH_HOME/media_info/", where=bench.name))
        data["tools"] = _tools_block(bench)
        for name, t in data["tools"].items():
            if not t["ok"]:
                warnings.append(_finding("W_TOOL_PIN", f"{name}: {t['problem']}; the prepare step that uses it refuses "
                                         "any other build. Run doctor where that step will run (on a compute node: "
                                         "`srun … yavih doctor --bench " + bench.name + "`), or bring the pinned build "
                                         "(binary + its ldd closure, or a container); see the benchmark's setup.md", where=name))
    data["gpu"] = _gpu_block()
    data["launcher"] = sys.executable
    try:
        data["decode_workers_default"] = int(default_decode_workers())
    except _cfg.ConfigError:  # already reported as W_CONFIG: the lenient view's value
        from yavih import resources

        data["decode_workers_default"] = resources.describe(settings)["decode_workers"]["value"] if settings is not None else None
    except Exception as e:  # noqa: BLE001
        data["decode_workers_default"] = None
        warnings.append(_finding("W_INTERNAL", f"default_decode_workers failed: {e}"))
    data["yavih_home"] = str(_cfg.yavih_home())
    data["runs_root"] = str(settings.runs_root()) if settings is not None else None
    data["results_root"] = str(settings.results_root()) if settings is not None else None
    if settings is not None:
        try:
            data["config"] = settings.show(bench)
        except _cfg.ConfigError as e:
            warnings.append(_finding("W_CONFIG", str(e)))
            data["config"] = settings.show()
    data["yavih_version"] = __version__
    data["python"] = sys.version.split()[0]
    data["host"] = socket.gethostname()
    # where this yavih comes from, and whether an installed skill copy is older than the packaged skill
    from yavih.shipped import installed_from, installed_skills, skill_stamp

    data.update(installed_from())
    try:
        data["skill"] = {"packaged": skill_stamp(), "installed": installed_skills(SKILL_TARGETS)}
    except OSError as e:  # pragma: no cover - an unreadable package directory
        data["skill"] = {"packaged": None, "installed": [], "error": str(e)}
    for inst in data["skill"]["installed"]:
        if inst["stale"]:
            had = (inst.get("stamp") or {}).get("yavih_version")
            why = f"installed by yavih {had}, its files differ from this yavih's skill" if had else "installed without a stamp (an older yavih)"
            warnings.append(_finding("W_SKILL_STALE", f"the {inst['target']} skill at {inst['path']} is stale ({why}); re-run "
                                     f"`yavih skill install --target {inst['target']}`", where=inst["path"]))
    if ns.selftest:
        data["selftest"] = _selftest()
        if not data["selftest"].get("ok"):
            warnings.append(_finding("W_SELFTEST", data["selftest"].get("error") or "selftest failed"))
    return Envelope(data=data, warnings=warnings, text=_doctor_text(data))


def _config_text(cfg: dict[str, Any], *, indent: str = "  ") -> list[str]:
    """``config show`` lines: the files read, the site, every value with its source, a benchmark's roots."""
    lines = []
    for f in cfg.get("files") or []:
        lines.append(f"{indent}config {f['layer']:<8} {f['path']}  ({f['how']}{'' if f['exists'] else ', absent'})")
    if not cfg.get("files"):
        lines.append(f"{indent}config   (no file)")
    lines.append(f"{indent}site     {cfg.get('site') or '-'}" + (f"  ({cfg['site_source']})" if cfg.get("site_source") else "")
                 + ("  [lenient]" if cfg.get("lenient") else ""))
    vals = cfg.get("values") or {}
    width = max((len(k) for k in vals), default=10)
    for key, rec in vals.items():
        lines.append(f"{indent}{key:<{width}}  {rec.get('value')}  ({rec.get('source')})")
    b = cfg.get("bench")
    if b:
        lines.append(f"{indent}benchmark {b['name']} ({b['folder']})")
        for key, r in b["roots"].items():
            state = r.get("error") or ("exists" if r.get("exists") else "MISSING") + (" (prepare only)" if r.get("prepare_only") else "")
            lines.append(f"{indent}  {key:<14} {r.get('value')}  ({r.get('source')})  {state}")
            lines.append(f"{indent}  {'':<14} set via {r.get('config_key')} | {r.get('env_var')} | --data {b['name']}.{key}=PATH")
    return lines


def cmd_config_show(ns: argparse.Namespace) -> Envelope:
    """``yavih config show [--bench B] [--json]``: every configuration value with its source (cli / env /
    file:line / yaml default / default), the files read and the active site; with ``--bench`` the benchmark's
    roots, whether each exists, and the key and env var that set it."""
    from yavih import config as _cfg

    settings = _cfg.load()
    bench = _load_bench(ns.bench) if ns.bench else None
    data = settings.show(bench)
    warnings = [_finding(w["code"], w["message"]) for w in data.get("warnings") or []]
    return Envelope(data=data, warnings=warnings, text="\n".join(_config_text(data, indent="")) + "\n")


def cmd_config_get(ns: argparse.Namespace) -> Envelope:
    """``yavih config get KEY``: one resolved value on stdout, for shell scripts (``prepare/*.sh``). KEY is
    ``runs.root``, ``results.root``, ``home.root``, ``scratch.root``, ``data.root``, ``datasets.<name>``,
    ``resources.<key>`` or ``<benchmark>.<root>`` (a name or alias). Exit 1 when it does not resolve."""
    from yavih import config as _cfg
    from yavih import resources

    settings = _cfg.load()
    key = str(ns.key)
    head, _, tail = key.partition(".")
    if not tail:
        raise CliError("E_USAGE", f"config get expects <table>.<key> or <benchmark>.<root>, got {key!r}")
    if key in ("runs.root", "results.root", "home.root", "scratch.root", "data.root"):
        value = {"runs.root": lambda: settings.runs_root(), "results.root": lambda: settings.results_root(),
                 "home.root": lambda: settings.yavih_home(), "scratch.root": lambda: settings.scratch_root(),
                 "data.root": lambda: settings.data_root()}[key]()
        v = settings.get(*key.split("."))
        source = v.source if v else "default"
        if value is None:
            raise _cfg.ConfigError(f"{key} is not configured (set [data] root in {settings.edit_hint()} or YAVIH_DATA__ROOT)")
    elif head == "datasets":
        value = settings.dataset(tail)
        v = settings.get("datasets", tail)
        source = v.source if v else "${data.root}/" + tail
        if value is None:
            raise _cfg.ConfigError(f"[datasets] {tail} is not configured and there is no [data] root to default it from")
    elif head == "resources":
        recs = resources.describe(settings)
        if tail not in recs:
            raise CliError("E_USAGE", f"unknown resource {tail!r}; known: {sorted(recs)}")
        value, source = recs[tail]["value"], recs[tail]["source"]
    else:
        bench = _load_bench(head)
        raw = _raw(bench)
        recs = settings.resolve_data_sources(bench.name, (raw.get("data") or {}).get("roots") or {}, aliases=tuple(raw.get("aliases") or ()))
        if tail not in recs:
            raise CliError("E_USAGE", f"benchmark {bench.name!r} declares no data root {tail!r}; declared: {sorted(recs)}")
        value, source = recs[tail]["value"], recs[tail]["source"]
    return Envelope(data={"key": key, "value": value, "source": source}, text=f"{value}\n")


def _doctor_text(d: dict[str, Any]) -> str:
    lines = [f"yavih {d['yavih_version']} on {d['host']} (python {d['python']})"]
    if d.get("installed_from"):
        lines.append(f"  installed   {d['installed_from']} at {d.get('package_dir')}")
    for inst in (d.get("skill") or {}).get("installed") or []:
        lines.append(f"  skill       {inst['target']} {inst['path']} {'STALE (yavih skill install)' if inst['stale'] else 'current'}")
    for extra in EXTRAS:
        blk = d[extra]
        lines.append(f"  {extra:<11} {'ok' if blk['ok'] else 'missing ' + ','.join(blk['missing']) + '  (' + blk['pip_hint'] + ')'}")
    t = d["torch"]
    lines.append(f"  torch       {t['version']} cuda={t['cuda']} untouched={t['untouched']}")
    for kind, info in d["decoders"].items():
        lines.append(f"  helper decoder {kind:<6} {str(info.get('lib')) + ' ' + str(info.get('version')) if info else 'none (optional: yavih[media])'}")
    for name, root in d["data_roots"].items():
        state = "exists" if root["exists"] else ("UNRESOLVED" if root.get("error") else "MISSING") + (" (prepare only)" if root.get("prepare_only") else "")
        lines.append(f"  root {name:<12} {root['path']}  {state}" + (f"  ({root['source']})" if root.get("source") else ""))
    for name, ext in (d.get("external") or {}).items():
        lines.append(f"  external {name:<10} {ext.get('version') or 'NOT INSTALLED'} (pin {ext.get('version_pin')}){'' if ext.get('ok') else '  install: ' + str(ext.get('pip'))}")
    if d.get("media_read"):
        s = d["media_read"]
        lines.append(f"  media_read  {s.get('clip_id')} ok={s.get('ok')} split={s.get('split')}" + (f"  ({s['error']})" if s.get("error") else ""))
    for name, t in (d.get("tools") or {}).items():
        pins = "".join(f"  {pn} {pv.get('found')} (pin {pv.get('pin')})" for pn, pv in (t.get("pins") or {}).items())
        lines.append(f"  tool {name:<10} {t.get('version') or 'NOT FOUND'} (pin {t.get('version_pin')}){pins}" + ("" if t.get("ok") else "  MISMATCH"))
    if d.get("media_info"):
        mi = d["media_info"]
        lines.append(f"  media_info  {'present ' + str(mi.get('sha256'))[:12] if mi.get('present') else 'absent (yavih bench verify <name> --census)'}")
    lines.append(f"  gpu         {[g['name'] for g in d['gpu']] or 'none'}")
    lines.append(f"  launcher    {d['launcher']}")
    lines.append(f"  decode_workers_default {d['decode_workers_default']}")
    if d.get("config"):
        lines.extend(_config_text({k: v for k, v in d["config"].items() if k != "bench"}))
    else:
        lines.append(f"  yavih_home  {d['yavih_home']}")
        lines.append(f"  runs_root   {d['runs_root']}")
        if d.get("results_root"):
            lines.append(f"  results     {d['results_root']}")
    if "selftest" in d:
        lines.append(f"  selftest    {json.dumps(_jsonable(d['selftest']), sort_keys=True)}")
    return "\n".join(lines) + "\n"


def _selftest() -> dict[str, Any]:
    """``doctor --selftest``: run ``yavih.bridges.examples.hf_reference`` on ``SELFTEST_N`` ESC-50 clips: decodes, encodes, checks finiteness / shape and the pinned number when one exists."""
    import numpy as np

    from yavih.bridge import parse_args, validate_describe
    from yavih.bridges.examples import hf_reference as ref

    out: dict[str, Any] = {"bench": ref.SELFTEST_BENCH, "n": ref.SELFTEST_N, "ok": False, "reference_acc": ref.REFERENCE_ACC, "tolerance": ref.REFERENCE_TOL}
    try:
        bench = _load_bench(ref.SELFTEST_BENCH)
        view = bench.view()  # a bridge receives the BenchmarkView, never the Benchmark
        cls = ref.HFReferenceBridge
        bridge = cls(view, parse_args(cls.ARGS, {}))
        describe = bridge.describe()
        validate_describe(describe, bridge, view)
        split = _bench_roles(bench)["eval"][0] if _bench_roles(bench).get("eval") else _bench_roles(bench)["folds"][0]
        samples = list(bench.reader(split, rows=range(0, ref.SELFTEST_N), batch_size=cls.batch_size))
        feats = []
        t0 = time.time()
        for i in range(0, len(samples), cls.batch_size):
            f, _, _ = bridge.encode(samples[i : i + cls.batch_size])
            feats.append(_to_numpy(f))
        arr = np.concatenate(feats)
        out.update(N=int(arr.shape[1]), D=int(arr.shape[2]), clips_per_s=round(len(samples) / max(time.time() - t0, 1e-9), 3), finite=bool(np.isfinite(arr).all()))
        out["ok"] = bool(out["finite"]) and arr.shape[0] == len(samples)
        out["note"] = "pinned accuracy check needs a probe run; REFERENCE_ACC is compared by the e2e test" if ref.REFERENCE_ACC is not None else "no pinned REFERENCE_ACC yet"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{e.__class__.__name__}: {e}"
    return out


# ----------------------------------------------------------------------------- skill


#: ``skill install --target``: ``(default destination, form)``; ``dir`` copies the skill directory (SKILL.md +
#: references/), ``file`` writes one document (the SKILL.md body under the runtime's own header).
SKILL_TARGETS: dict[str, tuple[str, str]] = {
    "claude": (".claude/skills/yavih", "dir"),
    "codex": (".codex/skills/yavih", "dir"),
    "cursor": (".cursor/rules/yavih.mdc", "file"),
    "agents-md": ("AGENTS.md", "file"),
}
_AGENTS_BEGIN, _AGENTS_END = "<!-- yavih skill: begin -->", "<!-- yavih skill: end -->"


def _skill_parts(root: Path) -> tuple[str, str]:
    """``(description, body)`` of the packaged SKILL.md: the frontmatter's description and the text after it."""
    text = (root / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    if not m:
        return "", text
    import yaml

    meta = yaml.safe_load(m.group(1)) or {}
    return " ".join(str(meta.get("description") or "").split()), text[m.end():].lstrip("\n")


def _skill_document(root: Path, target: str, stamp: dict | None = None) -> str:
    """The single-file form of the skill: a Cursor rule (frontmatter + body) or a marked ``AGENTS.md`` block, with
    the install stamp as an HTML comment (``yavih doctor``: ``W_SKILL_STALE``)."""
    from yavih.shipped import stamp_comment

    desc, body = _skill_parts(root)
    pointer = ("\nThe references named above (`references/*.md`) ship with yavih: print one with `yavih skill show --ref <name>` "
               "(bridge, cli, data, readouts, slurm) or all with `yavih skill show --all`.\n")
    if stamp:
        pointer += stamp_comment(stamp) + "\n"
    if target == "cursor":
        return f"---\ndescription: {json.dumps(desc)}\nglobs:\nalwaysApply: false\n---\n\n{body.rstrip()}\n{pointer}"
    return f"{_AGENTS_BEGIN}\n{body.rstrip()}\n{pointer}{_AGENTS_END}\n"


def cmd_skill_install(ns: argparse.Namespace) -> Envelope:
    """Install the packaged skill where an agent runtime reads it: ``--target claude`` (default; the directory to
    ``.claude/skills/yavih``), ``codex`` (the directory to ``.codex/skills/yavih``), ``cursor`` (one rule file
    ``.cursor/rules/yavih.mdc``) or ``agents-md`` (a marked block in ``AGENTS.md``, replaced in place on
    re-install, the rest of the file kept). ``--path`` overrides the destination."""
    root = _skill_root()
    if not root.is_dir():
        raise CliError("E_SKILL_MISSING", f"packaged skill directory not found: {root}", EXIT_INTERNAL)
    from yavih.shipped import SKILL_STAMP_FILE, skill_stamp

    stamp = skill_stamp(root)
    target = getattr(ns, "target", None) or "claude"
    default, form = SKILL_TARGETS[target]
    dest = Path(getattr(ns, "path", None) or default)
    files: list[str] = []
    warnings = []
    if form == "dir":
        dest.mkdir(parents=True, exist_ok=True)
        for src in _iter_skill_files(root):
            rel = src.relative_to(root)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            files.append(rel.as_posix())
        if "SKILL.md" not in files:
            warnings.append(_finding("W_SKILL_INCOMPLETE", "the packaged skill has no SKILL.md"))
        # the stamp `yavih doctor` compares with the packaged skill (W_SKILL_STALE after a yavih upgrade)
        (dest / SKILL_STAMP_FILE).write_text(json.dumps(stamp, sort_keys=True) + "\n", encoding="utf-8")
        files.append(SKILL_STAMP_FILE)
    else:
        doc = _skill_document(root, target, stamp)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if target == "agents-md" and dest.is_file():
            old = dest.read_text(encoding="utf-8")
            if _AGENTS_BEGIN in old and _AGENTS_END in old:
                head, rest = old.split(_AGENTS_BEGIN, 1)
                doc = head + doc.rstrip("\n") + rest.split(_AGENTS_END, 1)[1]
            else:
                doc = old.rstrip("\n") + "\n\n" + doc
        dest.write_text(doc, encoding="utf-8")
        files.append(dest.name)
    data = {"path": str(dest), "target": target, "files": files, "stamp": stamp}
    return Envelope(data=data, warnings=warnings, paths={"skill": str(dest)},
                    text=f"installed the yavih skill for {target}: {len(files)} file(s) at {dest}\n")


def cmd_skill_show(ns: argparse.Namespace) -> Envelope:
    """Print ``SKILL.md``; ``--all`` appends every ``references/*.md``."""
    root = _skill_root()
    path = root / "SKILL.md"
    if not path.is_file():
        raise CliError("E_SKILL_MISSING", f"SKILL.md is not packaged at {path}", EXIT_INTERNAL)
    if getattr(ns, "ref", None):
        from yavih.shipped import skill_reference

        rpath = skill_reference(ns.ref)
        rtext = rpath.read_text(encoding="utf-8")
        return Envelope(data={"path": str(rpath), "text": rtext, "references": [f"references/{rpath.name}"]}, paths={"skill": str(rpath)},
                        text=rtext if rtext.endswith("\n") else rtext + "\n")
    text = path.read_text(encoding="utf-8")
    refs: list[str] = []
    if getattr(ns, "all", False):
        for ref in sorted((root / "references").glob("*.md")):
            refs.append(f"references/{ref.name}")
            text = text.rstrip("\n") + f"\n\n---\n\n<!-- references/{ref.name} -->\n\n" + ref.read_text(encoding="utf-8")
    return Envelope(data={"path": str(path), "text": text, "references": refs}, paths={"skill": str(path)},
                    text=text if text.endswith("\n") else text + "\n")


# ----------------------------------------------------------------------------- bench


def _bench_refs() -> list[str]:
    """Packaged benchmark names, then folders under ``$YAVIH_HOME/benchmarks/``, on ``$YAVIH_BENCH_PATH`` and in
    ``yavih.benchmarks`` entry-point packs (the resolution order)."""
    from yavih.benchmark import _external_folders, packaged_benchmarks
    from yavih.config import yavih_home

    names = list(packaged_benchmarks())
    home = yavih_home() / "benchmarks"
    if home.is_dir():
        for p in sorted(home.iterdir()):
            if (p / "benchmark.yaml").is_file() and p.name not in names:
                names.append(p.name)
    names += [n for n in _external_folders() if n not in names]
    return names


def _status_word(bench: Any) -> str:
    """``bench list`` status: ``"missing" | "partial 61%" | "verified"``; an unresolvable data root
    or absent data counts as ``missing``."""
    from yavih.benchmark import DataNotReady
    from yavih.config import ConfigError

    try:
        return str(bench.data_status().status_word())
    except (ConfigError, DataNotReady, FileNotFoundError):
        return "missing"


def _bench_row(bench: Any) -> dict[str, Any]:
    raw = _raw(bench)
    budget = raw.get("budget") or {}
    return {
        "name": bench.name,
        "aliases": list(raw.get("aliases") or []),
        "readout": raw.get("readout"),
        "clips": dict(budget.get("clips") or {}),
        "media_gb": budget.get("media_gb"),
        "source_kind": (raw.get("source") or {}).get("kind"),
        "obtain": dict(raw["obtain"]) if isinstance(raw.get("obtain"), dict) else None,
        "status": _status_word(bench),
    }


def _suite_refs(suite_ref: str) -> tuple[Any, list[str], dict[str, dict[str, Any]], list[str]]:
    """``bench list --suite S``: ``(suite, packaged benchmark names in suite order, {name: {capability, readings}},
    pending names)``."""
    from yavih.suites import load_suite

    suite = load_suite(suite_ref)
    facts: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for cap, e in suite.entries():
        if e.status != "packaged":
            pending.append(e.benchmark)
            continue
        rec = facts.setdefault(e.benchmark, {"capability": cap, "readings": []})
        rec["readings"] += [r for r in e.readings if r not in rec["readings"]]
        if e.join:
            rec["readings"].append(str(e.join_reading))
    return suite, list(facts), facts, pending


def cmd_bench_list(ns: argparse.Namespace) -> Envelope:
    """``yavih bench list [--fmt text|md] [--suite S]``: every discoverable benchmark, or with ``--suite`` the
    suite's packaged benchmarks in suite order with their ``capability`` and ``readings``."""
    rows = []
    warnings = []
    suite, refs, facts, pending = (None, _bench_refs(), {}, [])
    if getattr(ns, "suite", None):
        suite, refs, facts, pending = _suite_refs(ns.suite)
    if getattr(ns, "fmt", "text") == "md":
        # the generated table of the docs (yavih.doctables), from each folder's benchmark.yaml; no data scan
        from yavih.benchmark import resolve_folder
        from yavih.doctables import render

        folders = {ref: f for ref in refs if (f := resolve_folder(ref)) is not None}
        return Envelope(data={"names": list(folders)}, text=render("readme", folders) + "\n")
    for ref in refs:
        try:
            row = _bench_row(_load_bench(ref))
        except Exception as e:  # noqa: BLE001 - one broken folder must not hide the others
            warnings.append(_finding("W_BENCH_LOAD", f"{ref}: {e.__class__.__name__}: {e}", where=ref))
            continue
        if suite is not None:
            row.update(facts[ref])
        rows.append(row)
    head = f"{'capability':<14}{'readings':<16}" if suite is not None else ""
    lines = [f"{head}{'name':<16}{'readout':<8}{'clips':<28}{'media_gb':>9}  {'source':<15}{'obtain':<17}status"]
    for r in rows:
        clips = ", ".join(f"{k}={v}" for k, v in r["clips"].items())
        tier = (r.get("obtain") or {}).get("tier") or "-"
        pre = f"{r['capability']:<14}{','.join(r['readings']):<16}" if suite is not None else ""
        lines.append(f"{pre}{r['name']:<16}{str(r['readout']):<8}{clips:<28}{str(r['media_gb']):>9}  {str(r['source_kind']):<15}{tier:<17}{r['status']}")
    if suite is not None:
        lines.insert(0, f"suite {suite.name} ({suite.origin}): {suite.description}")
        if pending:
            lines.append(f"pending (no packaged folder yet): {', '.join(pending)}")
    lines.append("obtain: auto = fetchable unattended; manual-download = a person downloads it from an official portal; "
                 "needs-licence = a human signs a licence for their account; needs-data-owner = ask the person who "
                 "holds it; site-only = staged at the origin site only (`yavih bench info <name>` says how)")
    return Envelope(data=rows, warnings=warnings, text="\n".join(lines) + "\n")


def cmd_bench_info(ns: argparse.Namespace) -> Envelope:
    """README text + ``{roles: {role: {nominal, present}}, floors, metrics, source, budget, history}``."""
    bench = _load_bench(ns.name)
    readme_path = Path(bench.folder) / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    if ns.readme_only:
        return Envelope(data={"name": bench.name, "readme": readme}, paths={"readme": str(readme_path), "folder": str(bench.folder)}, text=readme)
    if getattr(ns, "setup", False):
        # setup.md is the agent's first read per benchmark (the file `yavih bench show <b> setup` prints)
        setup_path = Path(bench.folder) / "setup.md"
        if not setup_path.is_file():
            raise CliError("E_NOT_FOUND", f"{bench.name}: no setup.md in {bench.folder}", EXIT_USAGE)
        setup = setup_path.read_text(encoding="utf-8")
        return Envelope(data={"name": bench.name, "setup": setup}, paths={"setup": str(setup_path), "folder": str(bench.folder)}, text=setup)
    raw = _raw(bench)
    roles = _bench_roles(bench)
    present_by_split: dict[str, Any] = {}
    try:
        status = bench.data_status()
        present_by_split = {s: v.get("present") for s, v in (status.per_split or {}).items()}
    except Exception:  # noqa: BLE001 - presence is optional information here
        present_by_split = {}
    role_table = {}
    for role, splits in roles.items():
        if not splits:
            continue
        nominal = sum(int(bench.n_rows(s)) for s in splits)
        present = sum(int(present_by_split[s]) for s in splits if present_by_split.get(s) is not None) if present_by_split else None
        role_table[role] = {"splits": list(splits), "nominal": nominal, "present": present}
    metrics = raw.get("metrics") or {}
    data = {
        "name": bench.name,
        "title": raw.get("title"),
        "readout": raw.get("readout"),
        "protocol_version": raw.get("protocol_version"),
        "protocol_hash": bench.protocol_hash,
        "input_hash": bench.input_hash,
        "scoring_hash": bench.scoring_hash,
        "manifest_hash": bench.manifest_hash,
        "readme": readme,
        "roles": role_table,
        "floors": list(metrics.get("floors") or []),
        "metrics": {"headline": metrics.get("headline"), "list": list(metrics.get("list") or []), "controls": list(metrics.get("controls") or []), "chance": dict(metrics.get("chance") or {})},
        "source": raw.get("source"),
        "obtain": raw.get("obtain"),
        "budget": raw.get("budget"),
        "external": list(raw.get("external") or []),
        "history": list(raw.get("changelog") or []),
    }
    summary = json.dumps(_jsonable({k: v for k, v in data.items() if k != "readme"}), indent=2, sort_keys=True)
    ob = data["obtain"] if isinstance(data["obtain"], dict) else {}
    obtain_line = f"obtain: {ob.get('tier')} — {ob.get('how')}\n" if ob else ""
    return Envelope(data=data, paths={"folder": str(bench.folder)},
                    text=readme + ("\n" if readme and not readme.endswith("\n") else "") + obtain_line + summary + "\n")


def cmd_bench_path(ns: argparse.Namespace) -> Envelope:
    """``yavih bench path <name>``: the benchmark folder (``FOLDER=$(yavih bench path <b>)``), where ``setup.md``,
    ``bridge_template.py`` and ``prepare/`` live in a pip install as in a checkout."""
    if getattr(ns, "what", None):
        # one shipped file's absolute path (sbatch "$(yavih bench path k400 prepare/01_download_torrent.sh)")
        from yavih.shipped import bench_file

        path = bench_file(ns.name, ns.what)
        return Envelope(data={"name": ns.name, "what": ns.what, "path": str(path), "folder": str(path.parent)},
                        paths={"file": str(path)}, text=f"{path}\n")
    bench = _load_bench(ns.name)
    folder = Path(bench.folder).resolve()
    return Envelope(data={"name": bench.name, "folder": str(folder)}, paths={"folder": str(folder)}, text=f"{folder}\n")


def cmd_bench_files(ns: argparse.Namespace) -> Envelope:
    """``yavih bench files <b>``: every shipped file of the folder with its absolute path, kind and size
    (``yavih.shipped.bench_files``), the same from a wheel install and a checkout."""
    from yavih.shipped import bench_files, bench_folder

    folder = bench_folder(ns.name)
    rows = bench_files(ns.name)
    lines = [f"{folder}"] + [f"  {r['kind']:<9}{r['size']:>11}  {r['rel']}" for r in rows]
    lines.append(f"print one: yavih bench show {ns.name} <readme|setup|yaml|template|probe|classes|prepare/<file>|<path>>; "
                 f"run a prepare step: yavih bench prepare {ns.name} <step>")
    return Envelope(data={"name": ns.name, "folder": str(folder), "files": rows}, paths={"folder": str(folder)}, text="\n".join(lines) + "\n")


def cmd_bench_show(ns: argparse.Namespace) -> Envelope:
    """``yavih bench show <b> <what>``: print one shipped file (a short name or a path inside the folder;
    ``yavih.shipped.bench_file_text``, path-traversal safe)."""
    from yavih.shipped import bench_file, bench_file_text

    path = bench_file(ns.name, ns.what)
    text = bench_file_text(ns.name, ns.what)
    return Envelope(data={"name": ns.name, "what": ns.what, "path": str(path), "text": text}, paths={"file": str(path)},
                    text=text if text.endswith("\n") else text + "\n")


def cmd_bench_prepare(ns: argparse.Namespace) -> Envelope:
    """``yavih bench prepare <b> [<step> [-- args…]]``: without a step, list the folder's numbered prepare steps; with
    one (number, name or file name), run it with the right interpreter (this python for ``.py``, ``bash`` for ``.sh``)
    in the current directory, with ``FOLDER`` and ``YAVIH_BENCH`` set. The exit code is the step's own."""
    from yavih.shipped import bench_folder, prepare_steps, run_prepare

    args = list(getattr(ns, "script_args", None) or [])
    if args and args[0] == "--":
        args = args[1:]
    if not ns.step:
        if args:
            raise CliError("E_USAGE", f"arguments {args} given without a step; `yavih bench prepare {ns.name} <step> -- <args>`")
        folder = bench_folder(ns.name)
        steps = prepare_steps(ns.name)
        lines = [f"{ns.name}: {len(steps)} prepare step(s) in {folder / 'prepare'} (setup.md says when and with which variables)"]
        lines += [f"  {st['step']}  {st['file']:<34} {st['interpreter']:<7} {st['doc']}" for st in steps]
        lines.append(f"run one: yavih bench prepare {ns.name} <step> [-- args]   (FOLDER and YAVIH_BENCH are set for it)")
        return Envelope(data={"name": ns.name, "folder": str(folder), "steps": steps}, paths={"folder": str(folder)}, text="\n".join(lines) + "\n")
    sys.stdout.flush()
    rc, rec = run_prepare(ns.name, ns.step, args)
    if rc != 0:
        return Envelope(status="error", data=rec, errors=[_finding("E_PREPARE_FAILED", f"{rec['file']} exited {rc}", rec["path"])],
                        paths={"script": rec["path"]}, exit_code=max(1, min(int(rc), 255)) if rc > 0 else EXIT_INTERNAL)
    return Envelope(data=rec, paths={"script": rec["path"]}, text=None)


def cmd_bench_fetch(ns: argparse.Namespace) -> Envelope:
    """Build/download the manifest into ``$YAVIH_HOME`` and verify its sha: ``Benchmark.load(ref, fetch=True)``."""
    if getattr(ns, "refresh", False):
        from yavih.benchmark import bench_fetch

        bench_fetch(ns.name, refresh=True)
    bench = _load_bench(ns.name, fetch=True)
    data = {"name": bench.name, "folder": str(bench.folder), "manifest_hash": bench.manifest_hash, "manifest_sha_ok": True}
    return Envelope(data=data, paths={"folder": str(bench.folder)}, text=f"{bench.name}: manifest verified at {bench.folder}\n")


def cmd_bench_verify(ns: argparse.Namespace) -> Envelope:
    """``yavih bench verify <name> [--census] [--decode-sample 32] [--fingerprint-sample 256]``.

    The ONE implementation is ``yavih.benchmark.bench_verify``: presence scan, ``media_fingerprint``, a read of
    the stored bytes of a seeded sample (``read``), an optional decode of the same rows as a media-health check
    (``decode_sample``, never a protocol step), derived / target sha256s, the curation sums, the
    external packages, and with ``--census`` the media-info table. It writes ``data_status.json``. This
    handler only shapes that dict into the ``--json`` envelope."""
    from yavih.benchmark import bench_verify

    data = bench_verify(ns.name, census=bool(ns.census), decode_sample=int(ns.decode_sample), fingerprint_sample=int(ns.fingerprint_sample))
    warnings: list[dict] = []
    per_split = {s: dict(v) for s, v in (data.get("per_split") or {}).items()}
    for s, v in per_split.items():
        nominal, present = int(v.get("nominal", 0) or 0), int(v.get("present", 0) or 0)
        if nominal and present < nominal:
            warnings.append(_finding("W_COVERAGE", f"split {s}: {present}/{nominal} present ({present / nominal:.1%})", where=s))
    read = dict(data.get("read") or {})
    if int(read.get("failed", 0) or 0):
        warnings.append(_finding("W_MEDIA_READ", f"{read['failed']} of {int(read.get('ok', 0)) + int(read['failed'])} sampled rows could not be read"))
    dec = dict(data.get("decode_sample") or {})
    if int(dec.get("failed", 0) or 0):
        warnings.append(_finding("W_DECODE_SAMPLE", f"{dec['failed']} of {int(dec.get('ok', 0)) + int(dec['failed'])} sampled rows failed to decode (a media-health check)"))
    curation_ok = data.get("curation_invariants_ok")
    if curation_ok is False:
        warnings.append(_finding("W_CURATION", "upstream.curation sums do not hold against the frozen manifest"))
    if data.get("targets_sha_ok") is False:
        warnings.append(_finding("W_TARGETS", "derived targets missing or their sha256 differs from benchmark.yaml: " + "; ".join(data.get("targets_errors") or [])))
    if data.get("derived_sha_ok") is False:
        warnings.append(_finding("W_DERIVED", "derived files missing or their sha256 differs from benchmark.yaml: " + "; ".join(data.get("derived_errors") or [])))
    for name, ext in (data.get("external") or {}).items():
        if not ext.get("ok"):
            warnings.append(_finding("W_EXTERNAL", f"{name}: installed {ext.get('version') or 'nothing'}, pinned {ext.get('version_pin')}; needed in the probe env: {ext.get('pip')}", where=name))
    name = data.get("name", ns.name)
    mi = dict(data.get("media_info") or {})
    lines = [f"{name}: manifest sha ok; labels ok; curation {'ok' if curation_ok else curation_ok}; targets_sha_ok={data.get('targets_sha_ok')}; "
             f"derived_sha_ok={data.get('derived_sha_ok')}; status {data.get('status')}"]
    for s, v in per_split.items():
        lines.append(f"  {s:<8} present {v.get('present')}/{v.get('nominal')}  missing_first10={v.get('missing_first10')}")
    lines.append(f"  read sample ok={read.get('ok')} failed={read.get('failed')}" + (f" ({read['skipped']})" if read.get("skipped") else ""))
    lines.append(f"  decode sample ok={dec.get('ok')} failed={dec.get('failed')} lib={dec.get('lib')}" + (f" ({dec['skipped']})" if dec.get("skipped") else ""))
    lines.append(f"  media_info {'present rows=' + str(mi.get('rows')) + ' sha256=' + str(mi.get('sha256')) if mi.get('present') else 'absent (run with --census)'}")
    lines.append(f"  media_fingerprint {data.get('media_fingerprint')}")
    lines.append(f"  data_status {data.get('data_status_path')}")
    paths = {"data_status": str(data.get("data_status_path") or "")}
    if mi.get("path"):
        paths["media_info"] = str(mi["path"])
    return Envelope(data=data, warnings=warnings, paths=paths, text="\n".join(lines) + "\n")


def cmd_bench_lint(ns: argparse.Namespace) -> Envelope:
    """``yavih bench lint <name|path>... [--all]``: the folder contract (``yavih.lint``) on packaged or out-of-tree
    folders. Errors exit 2 (a contract violation the folder author fixes); ``TODO`` markers are warnings."""
    from yavih.benchmark import UnknownBenchmark
    from yavih.lint import lint

    refs = list(ns.refs or [])
    if ns.all:
        refs += [r for r in _bench_refs() if r not in refs]
    if not refs:
        raise CliError("E_USAGE", "bench lint needs a benchmark name or path (or --all)")
    skip = tuple(ns.skip or ())
    reports, errors, warnings, lines = [], [], [], []
    for ref in refs:
        try:
            rep = lint(ref, skip=skip)
        except UnknownBenchmark as e:
            raise CliError("E_USAGE", str(e), EXIT_USAGE) from e
        except ValueError as e:
            raise CliError("E_USAGE", str(e), EXIT_USAGE) from e
        reports.append(rep.to_dict())
        for f in rep.errors:
            errors.append(_finding(f.code, f.message, where=f"{rep.name}/{f.file}" if f.file else rep.name))
        for f in rep.warnings:
            warnings.append(_finding(f.code, f.message, where=f"{rep.name}/{f.file}" if f.file else rep.name))
        lines.append(f"{rep.name}: {'ok' if rep.ok else 'FAIL'} ({len(rep.errors)} errors, {len(rep.warnings)} warnings)  {rep.folder}")
        for f in rep.findings:
            lines.append(f"  {f.severity:<7} {f.code:<14} {f.file or '-'}: {f.message}")
    ok = not errors
    return Envelope(status="ok" if ok else "error", data=reports, errors=errors, warnings=warnings,
                    exit_code=EXIT_OK if ok else EXIT_CONTRACT, text="\n".join(lines) + "\n")


def cmd_bench_new(ns: argparse.Namespace) -> Envelope:
    """``yavih bench new <name> [--readout R] [--like B] [--dir D]``: a scaffold folder that passes ``bench lint``
    (``yavih.scaffold``), with ``TODO`` wherever the new benchmark must supply a fact."""
    from yavih.lint import lint_folder
    from yavih.scaffold import ScaffoldError, new_benchmark

    try:
        folder = new_benchmark(ns.name, readout=ns.readout, like=ns.like, dir=ns.dir)
    except ScaffoldError as e:
        raise CliError("E_USAGE", str(e), EXIT_USAGE) from e
    rep = lint_folder(folder)
    todo = sum(int(f.message.split()[0]) for f in rep.warnings if f.code == "W_TODO")
    lines = [f"{ns.name}: scaffold written to {folder}",
             f"  bench lint: {'ok' if rep.ok else 'FAIL'} ({len(rep.errors)} errors, {todo} TODO markers to fill)",
             "  next: replace the manifest (make_manifest.py), protocol and contracts in benchmark.yaml, fill every TODO,",
             f"        then `yavih bench lint {folder}` and `python -m pytest {folder}/tests -q`"]
    return Envelope(data={"name": ns.name, "folder": str(folder), "lint": rep.to_dict(), "todo": todo}, paths={"folder": str(folder)},
                    status="ok" if rep.ok else "error", exit_code=EXIT_OK if rep.ok else EXIT_CONTRACT, text="\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- bridge check


def cmd_bridge_check(ns: argparse.Namespace) -> Envelope:
    """``yavih bridge check <bridge.py> --bench B [--n 32] [--arg k=v] [--decode-workers N] [--out profile.json]``.

    The ONE implementation is ``yavih.extract.bridge_check``: the same bridge preflight
    (describe → validate_describe → E_TRAIN_MODE → precision enforcement → describe_sha256) and the same
    prefetch/encode engine as ``run_extract``, so ``clips_per_s`` is measured exactly as extraction will
    run at the plan's decode-worker count. This handler only shapes the profile into the ``--json``
    envelope. The text names ``model``/``reading`` and the run
    directory the plan will use, and lists the profile's ``notes`` (``N_RULE_CUSTOM``: a declared opt-out,
    never a warning)."""
    from yavih.extract import bridge_check
    from yavih.io.validate import rules_line

    profile = bridge_check(
        bridge=ns.bridge,
        bench=ns.bench,
        n=max(1, int(ns.n)),
        args=_kv_pairs(ns.arg),
        decode_workers=ns.decode_workers,
        out=ns.out,
        synthetic=bool(getattr(ns, "synthetic", False)),
        **({"allow_loopback": True} if getattr(ns, "allow_loopback", False) else {}),
    )
    warnings = [dict(w) for w in (profile.get("warnings") or [])]
    paths = {"profile": str(ns.out)} if ns.out else {}
    if profile.get("status") == "not_applicable":
        return Envelope(status="not_applicable", data=profile, warnings=warnings, paths=paths, text=f"not applicable: {profile.get('reason')}\n")

    def num(x: Any) -> str:
        return "n/a" if x is None else f"{float(x):.3g}"

    inv = dict(profile.get("invariance") or {})
    model, reading = profile.get("model"), profile.get("reading")
    bench_name = profile.get("bench") or ns.bench  # bridge_check records the canonical name (it names the run dir)
    dsha = profile.get("describe_sha256")
    text = (
        f"{model} reading={reading}: N={profile.get('N')} D={profile.get('D')} dtype={profile.get('dtype')} "
        f"pos_mode={profile.get('pos_mode')} streams={profile.get('streams')}\n"
        f"  invariance: rerun_bit_identical={inv.get('rerun_bit_identical')} batch_rel_l2={num(inv.get('batch_rel_l2'))} "
        f"order_rel_l2={num(inv.get('order_rel_l2'))} tolerance={num(inv.get('tolerance'))}"
        + (f" ({inv.get('tolerance_reason')})" if inv.get("tolerance_reason") else "") + "\n"
        f"  {rules_line(dict(profile.get('rules_checked') or {}), profile.get('describe'))}\n"
        f"  clips_per_s={profile.get('clips_per_s')} at decode_workers={profile.get('decode_workers')} "
        f"(n={profile.get('n')}, split={profile.get('split')})\n"
        f"  describe_sha256={str(dsha)[:12] if dsha else None} (consistency only, never identity)\n"
        f"  run directory: runs/{bench_name}/{model}/{reading}/\n"
    )
    if profile.get("synthetic"):
        text += ("  SYNTHETIC: checked on tiny generated media standing in for the benchmark; no profile written. Stage the data and "
                 "rerun without --synthetic (with --out) for the profile plan needs.\n")
    for n_ in profile.get("notes") or []:
        if isinstance(n_, dict):
            text += f"  note {n_.get('code')}: {n_.get('message')}\n"
    for w in warnings:
        text += f"  {w.get('code')}: {w.get('message')}\n"
    return Envelope(data=profile, warnings=warnings, paths=paths, text=text)


# ----------------------------------------------------------------------------- extract / features / probe / report / plan / exec


def cmd_bridge_new(ns: argparse.Namespace) -> Envelope:
    """``yavih bridge new <bench> --model M [--reading R] [--out DIR|FILE] [--force]``: the folder's
    ``bridge_template.py`` copied to ``bridges/<model>_<bench>.py`` with the model name filled in
    (``yavih.shipped.bridge_new``); every other TODO stays, so ``bridge check`` refuses it until it is filled."""
    from yavih.shipped import bridge_new

    rec = bridge_new(ns.bench, model=ns.model, reading=ns.reading, out=ns.out, force=bool(ns.force))
    text = (f"wrote {rec['path']} (from {rec['template']}; filled: {', '.join(rec['filled']) or 'nothing'})\n"
            f"next: read the contract (`{rec['next'][0]}`), fill every TODO from the model's own code, then `{rec['next'][1]}`\n")
    return Envelope(data=rec, paths={"bridge": rec["path"], "template": rec["template"]}, text=text)


def cmd_extract(ns: argparse.Namespace) -> Envelope:
    from yavih.extract import run_extract

    summary = run_extract(
        bench=ns.bench,
        bridge=ns.bridge,
        split=ns.split,
        shard=_shard(ns.shard),
        out=ns.out,
        args=_kv_pairs(ns.arg),
        canaries=ns.canaries,
        decode_workers=ns.decode_workers,
        read_ahead=ns.read_ahead,
        deadline=_deadline(ns.deadline),
        expect_profile=ns.expect_profile,
        **({"allow_loopback": True} if getattr(ns, "allow_loopback", False) else {}),
    )
    status = summary.get("status", "ok") if isinstance(summary, dict) else "ok"
    return Envelope(status=status if status in STATUSES else "ok", data=summary, text=json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")


def _split_dirs(features_dir: Path) -> list[Path]:
    """A split directory itself, or every split subdirectory of a ``features/`` parent."""
    from yavih.io.format import META_FILE, NOT_APPLICABLE_FILE

    def is_split_dir(p: Path) -> bool:
        return (p / META_FILE).is_file() or (p / NOT_APPLICABLE_FILE).is_file() or any(p.glob("shard-*.json")) or any(p.glob("shard-*.tar")) or any(p.glob("shard-*.tmp"))

    if is_split_dir(features_dir):
        return [features_dir]
    subs = sorted(p for p in features_dir.iterdir() if p.is_dir() and is_split_dir(p)) if features_dir.is_dir() else []
    if not subs:
        raise CliError("E_NOT_FOUND", f"{features_dir} is neither a split directory nor a features/ parent with split directories", EXIT_USAGE)
    return subs


def cmd_features_validate(ns: argparse.Namespace) -> Envelope:
    """``yavih features validate <dir>``: one ``ValidationReport`` per split (exit 2 if any fails)."""
    from yavih.io.validate import eligibility_line, rules_line, validate_features
    from yavih.waivers import parse_cli_waivers

    root = Path(ns.features_dir)
    if not root.exists():
        raise CliError("E_NOT_FOUND", f"no such directory: {root}", EXIT_USAGE)
    waive = parse_cli_waivers(getattr(ns, "waive", None))  # E_WAIVE_INVALID, exit 1, before any split is read
    reports = []
    errors: list[dict] = []
    warnings: list[dict] = []
    lines: list[str] = []
    for d in _split_dirs(root):
        report = validate_features(d, verify_sha=ns.verify_sha, write_meta=not ns.no_write_meta, **({"waive": waive} if waive else {}))
        rj = report.to_json()
        rj["dir"] = str(d)
        reports.append(rj)
        errors += [dict(f.to_json(), where=f.where or str(d)) for f in report.errors]
        warnings += [dict(f.to_json(), where=f.where or str(d)) for f in report.warnings]
        cov = report.coverage or {}
        n_missing = len(report.missing_shards)
        elig = report.eligibility or {}
        lines.append(
            f"{report.split}: {'ok' if report.ok else 'FAILED'}  coverage {cov.get('present')}/{cov.get('nominal')}  "
            f"N={report.N} D={report.D} {report.dtype} pos={report.pos_mode}  missing shards: {n_missing}"
            + (f" ({' '.join(report.missing_shards)})" if n_missing else "")
        )
        if getattr(report, "rules_checked", None):
            lines.append("  " + rules_line(report.rules_checked, getattr(report, "describe", None)))  # names opt-outs
        for n_ in getattr(report, "notes", None) or []:
            if getattr(n_, "code", None) == "N_RULE_CUSTOM":  # a declared opt-out, a note, never a warning
                lines.append(f"  note {n_.code}: {n_.message}")
        if elig:
            # eligibility line; `eligible: null` (benchmark not loadable here) prints as "undecided"
            lines.append("  " + eligibility_line(elig))
        for w in getattr(report, "waived", None) or []:  # every waiver in force, with what it waived
            lines.append(f"  waived {w['code']} ({w['source']}, {w['count']} finding(s)): {w['reason']}")
    ok = all(r["ok"] for r in reports)
    data = reports[0] if len(reports) == 1 else reports
    return Envelope(status="ok" if ok else "error", data=data, errors=errors, warnings=warnings, exit_code=EXIT_OK if ok else EXIT_CONTRACT, text="\n".join(lines) + "\n")


def cmd_features_inspect(ns: argparse.Namespace) -> Envelope:
    from yavih.io.inspect import inspect_features

    data = inspect_features(ns.features_dir)
    return Envelope(data=data, text=json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n")


def cmd_features_diff(ns: argparse.Namespace) -> Envelope:
    from yavih.io.diff import diff_features

    data = diff_features(ns.a, ns.b)
    return Envelope(data=data, text=json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n")


def cmd_features_stage(ns: argparse.Namespace) -> Envelope:
    """``yavih features stage <features/> <scratch>``: copy shards + sidecars + ``_meta.json`` (and
    ``NOT_APPLICABLE.json``) to ``<scratch>/<unit_id|run>.<host>.<pid>/features/<split>/`` and print the
    staged directory. The printed path is the one ``probe run --scratch`` takes: the probe
    looks for ``<scratch>/features/`` (one copy routine for the package lives in
    ``yavih.io.stage``)."""
    from yavih.io.stage import stage_features

    root = Path(ns.features_dir)
    if not root.exists():
        raise CliError("E_NOT_FOUND", f"no such directory: {root}", EXIT_USAGE)
    unit = os.environ.get("YAVIH_UNIT_ID") or None
    try:
        features_dest = stage_features(root, ns.scratch, unit_id=unit)
    except FileNotFoundError as e:
        raise CliError("E_NOT_FOUND", str(e), EXIT_USAGE) from e
    dest = features_dest.parent
    splits: dict[str, dict[str, Any]] = {}
    total = 0
    for d in sorted(p for p in features_dest.iterdir() if p.is_dir()):
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        nbytes = sum((d / f).stat().st_size for f in files)
        total += nbytes
        splits[d.name] = {"files": files, "bytes": nbytes}
    data = {"staged": str(dest), "features": str(features_dest), "splits": splits, "bytes": total}
    return Envelope(data=data, paths={"staged": str(dest), "features": str(features_dest)}, text=f"{dest}\n")


def cmd_probe_run(ns: argparse.Namespace) -> Envelope:
    from yavih.readouts import probe
    from yavih.waivers import parse_cli_waivers

    seeds = tuple(int(s) for s in ns.seeds.split(",") if s.strip()) if ns.seeds else (0,)
    extra: dict[str, Any] = {}
    if getattr(ns, "bench", None):
        extra["bench"] = ns.bench  # a folder/ref for a benchmark that `_meta.json`'s name cannot resolve
    if getattr(ns, "out", None):
        extra["out"] = ns.out  # read-only features: mirror them into OUT, write _meta.json and probe/ there
    if getattr(ns, "device", None):
        extra["device"] = ns.device  # cpu | cuda | cuda:N (default: cuda when available)
    # The same prologue as a planned probe unit ("its prologue runs validate for every split"):
    # validate re-checks every split (cheap when _meta.json still matches the sidecars) before the probe
    # trusts `_meta.json`; `--no-validate` keeps the raw library behaviour.
    # results store: --results DIR overrides the root, --no-results skips the file.
    results = probe(
        ns.run_dir,
        seeds=seeds,
        deadline=ns.deadline,
        scratch=ns.scratch,
        allow_low_coverage=ns.allow_low_coverage,
        loader_workers=ns.loader_workers,
        log_samples=bool(getattr(ns, "log_samples", False)),
        results_root=getattr(ns, "results", None),
        write_results=not getattr(ns, "no_results", False),
        validate_first=not getattr(ns, "no_validate", False),
        waive=parse_cli_waivers(getattr(ns, "waive", None)) or None,
        **extra,
    )
    data = []
    lines = []
    headlines: dict[str, Any] = {}
    for r in results:
        res = dict(r.result or {})
        bname = (res.get("benchmark") or {}).get("name")
        hl_name = _headline_name(bname, getattr(ns, "bench", None), headlines)
        # `probe run` data: the comparability key and the row key (eval_hash, model, reading)
        rec = {"status": r.status, "seed": r.seed, "path": str(r.path) if r.path else None,
               "benchmark": bname, "eval_hash": res.get("eval_hash"), "model": res.get("model"), "reading": res.get("reading"),
               "headline": {"name": hl_name, "value": r.metrics.get(hl_name)} if hl_name else None,
               "metrics": r.metrics, "floors": dict(res.get("floors") or {}), "reason": r.reason}
        data.append(rec)
        key = f"{bname}/{rec['model']}/{rec['reading']} eval_hash={str(rec['eval_hash'] or '')[:12]}"
        lines.append(f"seed {r.seed}: {r.status}  {key}  metrics={json.dumps(_jsonable(r.metrics), sort_keys=True)}  floors={json.dumps(_jsonable(rec['floors']), sort_keys=True)}" + (f"  ({r.reason})" if r.reason else ""))
    paths: dict[str, Any] = {}
    results_files = sorted({str(r.results_file) for r in results if getattr(r, "results_file", None)})
    samples_files = sorted({str(r.samples_file) for r in results if getattr(r, "samples_file", None)})
    if results_files:
        paths["results_file"] = results_files[0] if len(results_files) == 1 else results_files
        lines.append(f"results: {', '.join(results_files)}")
    if samples_files:
        paths["samples_files"] = samples_files
        lines.append(f"samples: {', '.join(samples_files)}")
    statuses = {r.status for r in results}
    status = "not_applicable" if statuses and statuses <= {"not_applicable"} else "ok"
    return Envelope(status=status, data=data, paths=paths, text="\n".join(lines) + "\n")


def _headline_name(bench_name: str | None, bench_ref: str | None, cache: dict[str, Any]) -> str | None:
    """The benchmark's ``metrics.headline`` for the ``probe run`` summary; ``None`` when the
    benchmark cannot be loaded here (the number is still in ``metrics``)."""
    ref = bench_ref or bench_name
    if not ref:
        return None
    if ref not in cache:
        try:
            from yavih.benchmark import Benchmark

            cache[ref] = getattr(Benchmark.load(ref).spec.metrics, "headline", None)
        except Exception:  # noqa: BLE001 - a display nicety, never a failure of the probe
            cache[ref] = None
    return cache[ref]


def cmd_report_build(ns: argparse.Namespace) -> Envelope:
    from yavih.report import report

    join = None
    if ns.join:
        kind, *pairs = ns.join
        join = {kind: _kv_pairs(pairs)}
    rep = report(ns.globs, fmt=ns.fmt, seeds=ns.seeds, allow_coverage_diff=ns.allow_coverage_diff, join=join, detail=ns.detail,
                 results_root=getattr(ns, "results", None), merge_results=bool(getattr(ns, "merge_results", False)),
                 **({"suite": ns.suite} if getattr(ns, "suite", None) else {}))
    paths: dict[str, Any] = {}
    if rep.merged:
        paths["merged"] = list(rep.merged)
    for kind, joined in (rep.joins or {}).items():
        if isinstance(joined, dict) and joined.get("path"):
            paths[f"join_{kind}"] = joined["path"]
    if ns.out:
        out = Path(ns.out)
        if out.is_dir() or str(ns.out).endswith(("/", os.sep)):  # a directory: write <dir>/report.<fmt>
            out = out / f"report.{ns.fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(rep.text)
        paths["out"] = str(out)
    # the report's warnings (W_BRIDGE_CHANGED, W_FEATURES_DIFFER, W_NONSTANDARD, …) are also the
    # envelope's Findings (where = the column id), so a --json caller sees them without parsing `data`.
    warnings = [_finding(str(w.get("code")), str(w.get("message", "")), w.get("column") or w.get("where"))
                for w in (getattr(rep, "warnings", None) or []) if isinstance(w, dict) and w.get("code")]
    return Envelope(data=rep.to_json(), warnings=warnings, paths=paths, text=rep.text if rep.text.endswith("\n") else rep.text + "\n")


def cmd_plan_build(ns: argparse.Namespace) -> Envelope:
    from yavih.plan.units import plan, write_plan

    warnings: list[dict] = []  # W_NO_CENSUS: advice only, the units are the same
    units = plan(ns.matrix, pending=ns.pending, kind=ns.kind, shard_minutes=ns.shard_minutes, warnings=warnings)
    for w in warnings:
        sys.stderr.write(f"[yavih] {w['code']}: {w['message']}\n")
    paths = {}
    sha: str | None = None
    if ns.out:
        write_plan(units, ns.out)
        paths["plan"] = ns.out
        import hashlib

        if os.path.isfile(ns.out):  # for `exec --plan-sha` (a pending file rebuilt after submission is refused)
            with open(ns.out, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
    if ns.format == "count":
        data: dict[str, Any] = {"count": len(units)}
        if sha:
            data["sha256"] = sha
        return Envelope(data=data, paths=paths, warnings=warnings, text=f"{len(units)}\n")
    records = [u.to_json() for u in units]
    text = "" if ns.out else "".join(json.dumps(_jsonable(r), sort_keys=True) + "\n" for r in records)
    return Envelope(data=records, paths=paths, warnings=warnings, text=text or f"wrote {len(units)} unit(s) to {ns.out}\n")


def cmd_plan_init(ns: argparse.Namespace) -> Envelope:
    """``yavih plan init --suite S --model M --launcher L [--out matrix.yaml] [--bridges-dir DIR] [--force]``: one
    bridge file per suite benchmark (``yavih.shipped.bridge_new``; an existing one is kept) and a ``matrix.yaml``
    skeleton with one entry per suite ``(benchmark, reading)`` (``yavih.plan.skeleton``). The text lists the
    ``bridge check`` command of every entry, then the ``plan build`` command."""
    from yavih.plan.skeleton import plan_init

    rec = plan_init(ns.suite, model=ns.model, launcher=ns.launcher, out=ns.out, bridges_dir=ns.bridges_dir, force=bool(ns.force))
    lines = [f"wrote {rec['path']}: suite {rec['suite']}, model {rec['model']}, {len(rec['entries'])} entries"]
    for b in rec["bridges"]:
        lines.append(f"  bridge {b['path']} " + ("(exists, kept)" if b["action"] == "kept" else f"(created from {b['benchmark']}'s bridge_template.py)"))
    for p in rec["pending"]:
        lines.append(f"  pending {p['benchmark']} [{', '.join(p['readings'])}] ({p['capability']}): no packaged folder, not planned")
    lines.append("next: fill every TODO in each bridge (`yavih bench show <benchmark> setup` has the contract), then on the staged data:")
    lines += [f"  {c}" for c in rec["next"]]
    for j in rec["joins"]:
        lines.append(f"after probing: `yavih suite missing {rec['suite']} --model {rec['model']}` prints the "
                     f"`yavih report build --join {j['benchmark']} ...` command that scores {j['benchmark']}:{j['reading']}")
    paths = {"matrix": rec["path"], "bridges": [b["path"] for b in rec["bridges"]]}
    return Envelope(data=rec, paths=paths, text="\n".join(lines) + "\n")


def cmd_plan_status(ns: argparse.Namespace) -> Envelope:
    """Text mode prints a summary (totals, per kind, the units that need attention with the last line of
    their message); ``--json`` carries every unit row."""
    from yavih.plan.status import format_status, plan_status

    data = plan_status(ns.plan, **({"root": ns.root} if getattr(ns, "root", None) else {}))
    return Envelope(data=data, text=format_status(data, plan_path=ns.plan))


def cmd_plan_calibrate(ns: argparse.Namespace) -> Envelope:
    from yavih.plan.calibrate import calibrate

    data = calibrate(ns.runs_root)
    return Envelope(data=data, text=json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n")


def cmd_plan_run(ns: argparse.Namespace) -> Envelope:
    """``yavih plan run <plan.jsonl> --local [--slots K]``: the pending units in dependency order on this machine
    (``yavih.plan.local``). There is no SLURM mode: yavih never writes or submits sbatch."""
    if not ns.local:
        raise CliError("E_USAGE", "plan run needs --local (it runs the plan on this machine); yavih never writes or submits "
                       "sbatch: for SLURM use the array templates in references/slurm.md")
    from yavih.plan.local import run_local

    res = run_local(ns.plan, slots=ns.slots, deadline=ns.deadline, deadline_margin=ns.deadline_margin, root=ns.root,
                    cpus=ns.cpus, deps=ns.deps)
    exit_code = int(res["exit_code"])
    env_status = "ok" if exit_code == 0 else "error"
    errors = [] if exit_code == 0 else [_finding(f"E_UNIT_{str(res['status']).upper()}",
                                                f"plan run --local: {res['counts']}, blocked {len(res['blocked'])}, "
                                                f"not started {len(res['not_started'])} (exit {exit_code}); see `yavih plan status`")]
    text = (f"{res['total']} units: {res['already_done']} already done, ran {len(res['units'])} "
            f"({', '.join(f'{k} {v}' for k, v in sorted(res['counts'].items())) or 'none'}), blocked {len(res['blocked'])}, "
            f"not started {len(res['not_started'])}\n")
    return Envelope(status=env_status, data=res, errors=errors, exit_code=exit_code, text=text)


def cmd_exec(ns: argparse.Namespace) -> Envelope:
    from yavih.plan.exec import exec_unit

    # pass only the options that were given
    extra = {k: v for k, v in (("deps", ns.deps), ("unit_range", ns.range), ("root", ns.root), ("cpus", ns.cpus),
                               ("plan_sha", ns.plan_sha)) if v is not None}
    last = exec_unit(ns.plan, index=ns.index, unit_id=ns.id, deadline=ns.deadline, deadline_margin=ns.deadline_margin,
                     slots=ns.slots, **extra)
    status = last.get("status", "done")
    env_status = {"done": "ok", "skipped": "skipped", "not_applicable": "not_applicable"}.get(status, "error")
    exit_code = int(last.get("exit_code", EXIT_OK))
    errors = [] if env_status != "error" else [_finding(f"E_UNIT_{status.upper()}", f"unit {last.get('id')} ended with status {status} (exit {exit_code})")]
    return Envelope(status=env_status, data=last, errors=errors, exit_code=exit_code, text=json.dumps(_jsonable(last), indent=2, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------- suites


def cmd_suite_list(ns: argparse.Namespace) -> Envelope:
    """``yavih suite list``: every suite (shipped, and project ones from ``yavih.suites.project_suites_dir``), each
    with its ``origin``; a project suite of a shipped name ``shadows`` it; an invalid file is listed with its error
    (``W_SUITE_INVALID``)."""
    from yavih.suites import list_suites, project_suites_dir

    recs = list_suites()
    warnings = [_finding("W_SUITE_INVALID", r["error"], where=r["path"]) for r in recs if r.get("error")]
    lines = [f"{'name':<16}{'origin':<10}description"]
    for r in recs:
        desc = r["description"] or f"INVALID: {r['error']}"
        lines.append(f"{r['name']:<16}{r['origin'] + ('*' if r['shadows'] else ''):<10}{desc}")
    lines.append(f"project suites: {project_suites_dir()} (* = shadows the shipped suite of that name)")
    return Envelope(data=recs, warnings=warnings, text="\n".join(lines) + "\n")


def cmd_suite_show(ns: argparse.Namespace) -> Envelope:
    """``yavih suite show <name|file.yaml>``: the suite's groups, benchmarks, readings and joins."""
    from yavih.suites import load_suite

    suite = load_suite(ns.name)
    lines = [f"suite {suite.name} ({suite.origin}: {suite.path})", f"  {suite.description}"]
    for g in suite.groups:
        lines.append(f"  {g.capability}")
        for e in g.entries:
            flags = [f for f, on in (("join " + str(e.join_reading), e.join), ("optional", not e.required),
                                     ("pending", e.status == "pending")) if on]
            lines.append(f"    {e.benchmark:<14}[{', '.join(e.readings)}]" + (f"  {'; '.join(flags)}" if flags else ""))
    return Envelope(data=suite.to_json(), paths={"suite": str(suite.path)}, text="\n".join(lines) + "\n")


def cmd_suite_new(ns: argparse.Namespace) -> Envelope:
    """``yavih suite new <name> [--from default] [--out DIR|FILE] [--force]``: a copy of a suite to edit, written to
    the project suites directory (``yavih.suites.new_suite``)."""
    from yavih.suites import new_suite

    rec = new_suite(ns.name, from_=ns.from_, out=ns.out, force=bool(ns.force))
    text = (f"wrote {rec['path']} (a copy of suite {rec['from']}); edit its groups, then `yavih suite show {rec['path']}` "
            "validates it\n")
    return Envelope(data=rec, paths={"suite": rec["path"]}, text=text)


def cmd_suite_missing(ns: argparse.Namespace) -> Envelope:
    """``yavih suite missing <name> --model M [<glob>...] [--results DIR]``: per suite ``(benchmark, reading)`` of
    model M ``done`` / ``missing`` / ``pending``, per join ``done`` / ``join-missing`` / ``join-unknown``, from the
    DONE-marked results under the globs (default: the runs root) and the results store; exit 0 either way."""
    from yavih import config as _cfg
    from yavih.report.index import build_index, runs_root_of
    from yavih.suites import completeness, completeness_json, load_suite

    suite = load_suite(ns.name)
    globs = list(ns.globs) or [str(_cfg.runs_root())]
    records = build_index(globs)
    live = [r for r in records if not r.get("moved_aside")]
    root = runs_root_of([Path(r["path"]) for r in live]) or (Path(globs[0]) if not ns.globs else None)
    results = ns.results
    if results is None:
        try:
            results = _cfg.results_root(root)
        except _cfg.ConfigError:
            results = None
    rows = completeness(suite, records, models=[ns.model], results_root=results, runs_root=root)
    data = completeness_json(suite, rows, results_root=results)
    data["model"] = ns.model
    lines = [f"suite {suite.name} for model {ns.model}: " + ", ".join(f"{k} {v}" for k, v in data["counts"].items())]
    for r in rows:
        label = f"{r['benchmark']}:{r['reading']}"
        note = f" ({r['note']})" if r.get("note") and r["note"] != "join" else ""
        lines.append(f"  {r['capability']:<14}{label:<24}{r['status']}{note}" + ("" if r.get("required", True) else " [optional]"))
        if r.get("hint"):
            lines.append(f"    {r['hint']}")
    return Envelope(data=data, text="\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """The command table."""
    def config_flags(parser: argparse.ArgumentParser, *, results: bool = True) -> None:
        # The configuration layer of every command (yavih.config): before or after the subcommand. SUPPRESS so a
        # subparser does not reset what the main parser saw; distinct dests so `probe run --results` keeps its own.
        parser.add_argument("--config", dest="cfg_config", default=argparse.SUPPRESS, metavar="FILE",
                            help="project config file (beats ./yavih.toml discovery; exported as YAVIH_CONFIG)")
        parser.add_argument("--runs-root", dest="cfg_runs_root", default=argparse.SUPPRESS, metavar="DIR",
                            help="runs root (beats YAVIH_RUNS and [runs] root)")
        if results:
            parser.add_argument("--results", dest="cfg_results", default=argparse.SUPPRESS, metavar="DIR",
                                help="results-store root (beats YAVIH_RESULTS and [results] root)")
        parser.add_argument("--data", dest="cfg_data", action="append", default=argparse.SUPPRESS, metavar="BENCH.ROOT=PATH",
                            help="data root override, repeatable (also root=PATH, datasets.NAME=PATH)")
        parser.add_argument("--lenient", dest="cfg_lenient", action="store_true", default=argparse.SUPPRESS,
                            help="unknown config keys / YAVIH_* variables / stale benchmark copies warn instead of failing")

    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: a subparser that did not see --json must not reset the flag a parent parser set
    # (argparse applies subparser defaults onto the shared namespace).
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit exactly one JSON envelope on stdout (also YAVIH_JSON=1)",
    )
    # `probe run` and `report build` define their own --results (same meaning, read by the handler)
    common_nr = argparse.ArgumentParser(add_help=False, parents=[common])
    config_flags(common_nr, results=False)
    common_r = argparse.ArgumentParser(add_help=False, parents=[common])
    config_flags(common_r)

    p = argparse.ArgumentParser(prog="yavih", description="yavih: frozen-feature benchmark harness", parents=[common_r])
    p.add_argument("--version", action="version", version=f"yavih {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # doctor
    sp = sub.add_parser("doctor", parents=[common_r], help="environment, decoders, data roots, GPU, configuration; --selftest runs the reference bridge")
    sp.add_argument("--bench", default=None)
    sp.add_argument("--selftest", action="store_true")
    sp.set_defaults(func=cmd_doctor)

    # config
    sp = sub.add_parser("config", parents=[common_r], help="show the configuration with the source of every value (yavih.config)")
    csub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    csub.required = True
    s = csub.add_parser("show", parents=[common_r])
    s.add_argument("--bench", default=None, metavar="B", help="also resolve benchmark B's data roots")
    s.set_defaults(func=cmd_config_show)
    s = csub.add_parser("get", parents=[common_r], help="print one resolved value (for shell scripts)")
    s.add_argument("key", metavar="<bench>.<root>|runs.root|data.root|datasets.<name>|...")
    s.set_defaults(func=cmd_config_get)

    # suite
    sp = sub.add_parser("suite", parents=[common_r], help="named benchmark sets: list / show / new / missing (yavih.suites)")
    susub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    susub.required = True
    s = susub.add_parser("list", parents=[common_r], help="every suite with its origin (shipped / project)")
    s.set_defaults(func=cmd_suite_list)
    s = susub.add_parser("show", parents=[common_r], help="a suite's capabilities, benchmarks, readings and joins")
    s.add_argument("name", metavar="<name|file.yaml>")
    s.set_defaults(func=cmd_suite_show)
    s = susub.add_parser("new", parents=[common_r], help="copy a suite into the project suites directory to edit")
    s.add_argument("name")
    s.add_argument("--from", dest="from_", default="default", metavar="SUITE", help="the suite copied (default: default)")
    s.add_argument("--out", default=None, metavar="DIR|FILE", help="destination (default: <project suites dir>/<name>.yaml)")
    s.add_argument("--force", action="store_true", help="overwrite an existing destination file")
    s.set_defaults(func=cmd_suite_new)
    s = susub.add_parser("missing", parents=[common_nr], help="which suite (benchmark, reading) pairs and joins a model still lacks")
    s.add_argument("name", metavar="<name|file.yaml>")
    s.add_argument("globs", nargs="*", metavar="<glob>", help="where the results are (default: the runs root)")
    s.add_argument("--model", required=True, help="the model NAME (describe()['model'])")
    s.add_argument("--results", default=None, metavar="DIR", help="results-store root, where joins are found")
    s.set_defaults(func=cmd_suite_missing)

    # skill
    sp = sub.add_parser("skill", parents=[common_r], help="install or show the agent skill")
    ssub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    ssub.required = True
    s = ssub.add_parser("install", parents=[common_r])
    s.add_argument("--target", choices=sorted(SKILL_TARGETS), default="claude",
                   help="agent runtime: claude / codex (skill directory), cursor (.mdc rule), agents-md (a block in AGENTS.md)")
    s.add_argument("--path", default=None,
                   help="destination; default per target: " + ", ".join(f"{k} {v[0]}" for k, v in SKILL_TARGETS.items()))
    s.set_defaults(func=cmd_skill_install)
    s = ssub.add_parser("show", parents=[common_r])
    g = s.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="append every references/*.md")
    g.add_argument("--ref", default=None, metavar="NAME", help="print one reference instead: bridge, cli, data, readouts, slurm")
    s.set_defaults(func=cmd_skill_show)

    # bench
    sp = sub.add_parser("bench", parents=[common_r], help="list / info / fetch / verify benchmarks")
    bsub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    bsub.required = True
    s = bsub.add_parser("list", parents=[common_r])
    s.add_argument("--fmt", choices=["text", "md"], default="text", help="md: the markdown table the docs embed (no data scan)")
    s.add_argument("--suite", default=None, metavar="S", help="only the suite's benchmarks, with their capability and readings")
    s.set_defaults(func=cmd_bench_list)
    s = bsub.add_parser("info", parents=[common_r])
    s.add_argument("name")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--readme-only", action="store_true")
    g.add_argument("--setup", action="store_true", help="print setup.md, the agent's first read")
    s.set_defaults(func=cmd_bench_info)
    s = bsub.add_parser("path", parents=[common_r], help="print the benchmark folder (FOLDER=$(yavih bench path <name>)) or one shipped file's path")
    s.add_argument("name")
    s.add_argument("what", nargs="?", default=None, help="optional: a short name or a path inside the folder (as bench show)")
    s.set_defaults(func=cmd_bench_path)

    s = bsub.add_parser("files", parents=[common_r], help="every shipped file of a benchmark folder: absolute path, kind, size")
    s.add_argument("name")
    s.set_defaults(func=cmd_bench_files)

    s = bsub.add_parser("show", parents=[common_r], help="print one shipped file: readme|setup|yaml|template|probe|classes|prepare/<file>|<path>")
    s.add_argument("name")
    s.add_argument("what", metavar="readme|setup|yaml|template|probe|classes|prepare/<file>|<path>")
    s.set_defaults(func=cmd_bench_show)

    s = bsub.add_parser("prepare", parents=[common_r], help="list a benchmark's prepare steps, or run one (FOLDER and YAVIH_BENCH set)")
    s.add_argument("name")
    s.add_argument("step", nargs="?", default=None, help="number (01), name (download) or file (01_download.sh); omit to list")
    s.add_argument("script_args", nargs=argparse.REMAINDER, metavar="-- args", help="passed to the step after --")
    s.set_defaults(func=cmd_bench_prepare)

    s = bsub.add_parser("fetch", parents=[common_r])
    s.add_argument("name")
    s.add_argument("--refresh", action="store_true", help="overwrite locally edited files of the $YAVIH_HOME copy with the packaged ones")
    s.set_defaults(func=cmd_bench_fetch)
    s = bsub.add_parser("verify", parents=[common_r])
    s.add_argument("name")
    s.add_argument("--census", action="store_true", help="probe every present file and write the media-info table")
    s.add_argument("--decode-sample", type=int, default=32)
    s.add_argument("--fingerprint-sample", type=int, default=256)
    s.set_defaults(func=cmd_bench_verify)
    s = bsub.add_parser("lint", parents=[common_r], help="check a benchmark folder against the folder contract (layout, yaml, docs, imports)")
    s.add_argument("refs", nargs="*", metavar="name|path")
    s.add_argument("--all", action="store_true", help="every discoverable benchmark folder")
    s.add_argument("--skip", action="append", default=[], metavar="CHECK", help="skip a check by id (e.g. imports)")
    s.set_defaults(func=cmd_bench_lint)
    s = bsub.add_parser("new", parents=[common_r], help="scaffold a benchmark folder that passes bench lint (TODO where facts are needed)")
    s.add_argument("name")
    s.add_argument("--readout", default=None, help="the readout (clf, seq, grid, box or a registered one); picks the default --like")
    s.add_argument("--like", default=None, metavar="BENCH", help="the folder whose manifest, reader and template are copied (default per readout)")
    s.add_argument("--dir", default=None, metavar="D", help="parent directory (default: the first $YAVIH_BENCH_PATH entry, else .)")
    s.set_defaults(func=cmd_bench_new)

    # bridge
    sp = sub.add_parser("bridge", parents=[common_r], help="smoke-test a bridge and write its profile")
    brsub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    brsub.required = True
    s = brsub.add_parser("check", parents=[common_r])
    s.add_argument("bridge")
    s.add_argument("--bench", required=True)
    s.add_argument("--n", type=int, default=32)
    s.add_argument("--arg", action="append", default=[], metavar="k=v")
    s.add_argument("--decode-workers", type=int, default=None)
    s.add_argument("--out", default=None, metavar="profile.json")
    s.add_argument("--synthetic", action="store_true",
                   help="run the checks on a tiny generated stand-in of the benchmark (its streams and facts), before any data is staged; writes no profile")
    s.add_argument("--allow-loopback", action="store_true",
                   help="let encode() connect to 127.0.0.0/8 / ::1 (a local inference server); recorded in the profile")
    s.set_defaults(func=cmd_bridge_check)
    s = brsub.add_parser("new", parents=[common_r], help="copy a benchmark's bridge_template.py to bridges/<model>_<bench>.py with the model name filled in")
    s.add_argument("bench", metavar="<bench>")
    s.add_argument("--model", required=True, help="the model's NAME (every bridge of this model uses it; results are filed under it)")
    s.add_argument("--reading", default=None, help="a | v | av (fills a literal reading, or the default of a reading arg)")
    s.add_argument("--out", default=None, metavar="DIR|FILE", help="destination (default bridges/<model>_<bench>.py)")
    s.add_argument("--force", action="store_true", help="overwrite an existing destination file")
    s.set_defaults(func=cmd_bridge_new)

    # extract
    sp = sub.add_parser("extract", parents=[common_r], help="extract one shard (python -m yavih.extract)")
    sp.add_argument("--bench", required=True)
    sp.add_argument("--bridge", required=True)
    sp.add_argument("--split", required=True)
    sp.add_argument("--shard", required=True, metavar="i/n")
    sp.add_argument("--out", required=True, metavar="DIR")
    sp.add_argument("--arg", action="append", default=[], metavar="k=v")
    sp.add_argument("--canaries", type=int, default=8)
    sp.add_argument("--decode-workers", type=int, default=None)
    sp.add_argument("--read-ahead", type=int, default=2, help="batches whose stored bytes are read ahead")
    sp.add_argument("--deadline", default=None, metavar="slurm|<epoch_s>")
    sp.add_argument("--expect-profile", default=None, metavar="profile.json",
                    help="the bridge-check profile: skip finished shards before loading the model; E_CONFIG_DRIFT on drift")
    sp.add_argument("--allow-loopback", action="store_true",
                    help="let encode() connect to 127.0.0.0/8 / ::1 (a local inference server); recorded in the sidecar")
    sp.set_defaults(func=cmd_extract)

    # features
    sp = sub.add_parser("features", parents=[common_r], help="validate / inspect / diff / stage feature sets")
    fsub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    fsub.required = True
    s = fsub.add_parser("validate", parents=[common_r])
    s.add_argument("features_dir", metavar="<features_dir or features/>")
    s.add_argument("--verify-sha", action="store_true")
    s.add_argument("--no-write-meta", action="store_true")
    s.add_argument("--waive", action="append", default=[], metavar="CODE=reason",
                   help="waive a waivable check for these features, with a reason (repeatable; all=reason = every waivable check); "
                        "recorded in _meta.json waived (source cli)")
    s.set_defaults(func=cmd_features_validate)
    s = fsub.add_parser("inspect", parents=[common_r])
    s.add_argument("features_dir")
    s.set_defaults(func=cmd_features_inspect)
    s = fsub.add_parser("diff", parents=[common_r])
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_features_diff)
    s = fsub.add_parser("stage", parents=[common_r])
    s.add_argument("features_dir", metavar="<features/>")
    s.add_argument("scratch")
    s.set_defaults(func=cmd_features_stage)

    # probe
    sp = sub.add_parser("probe", parents=[common_r], help="train and score the standardized readout")
    psub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    psub.required = True
    s = psub.add_parser("run", parents=[common_nr])
    s.add_argument("run_dir")
    s.add_argument("--seeds", default="0", metavar="0,1,2")
    s.add_argument("--deadline", default=None, metavar="slurm|<epoch_s>")
    s.add_argument("--scratch", default=None, metavar="DIR")
    s.add_argument("--allow-low-coverage", action="store_true")
    s.add_argument("--loader-workers", type=int, default=None)
    s.add_argument("--bench", default=None, metavar="<ref|dir>", help="benchmark folder/ref when _meta.json's name is neither packaged nor under $YAVIH_HOME/benchmarks")
    s.add_argument("--no-validate", action="store_true", help="skip the validate prologue and trust _meta.json as is")
    s.add_argument("--waive", action="append", default=[], metavar="CODE=reason",
                   help="waive a waivable check in the validate prologue, with a reason (repeatable; all=reason; as features validate --waive)")
    s.add_argument("--log-samples", action="store_true", help="also write samples_<benchmark>_<timestamp>.jsonl, one line per scored row")
    s.add_argument("--out", default=None, metavar="DIR",
                   help="write _meta.json and probe/ under DIR, features mirrored as symlinks: probes a read-only feature set")
    s.add_argument("--device", default=None, metavar="cpu|cuda|cuda:N", help="training device (default: cuda when available, else cpu)")
    rg = s.add_mutually_exclusive_group()
    rg.add_argument("--results", default=None, metavar="DIR",
                    help="results-store root (default: YAVIH_RESULTS / config [results] root / results/ next to the runs root of a "
                         "standard <runs root>/<benchmark>/<model>/<reading> run dir; any other layout needs --results)")
    rg.add_argument("--no-results", action="store_true", help="do not write the results-store file")
    s.set_defaults(func=cmd_probe_run)

    # report
    sp = sub.add_parser("report", parents=[common_r], help="comparability-checked tables with floors")
    rsub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    rsub.required = True
    s = rsub.add_parser("build", parents=[common_nr])
    s.add_argument("globs", nargs="+", metavar="<glob>")
    s.add_argument("--fmt", choices=["md", "csv", "json"], default="md")
    s.add_argument("--out", default=None, metavar="PATH")
    s.add_argument("--seeds", choices=["aggregate", "list"], default="aggregate")
    s.add_argument("--allow-coverage-diff", action="store_true")
    s.add_argument("--detail", action="store_true")
    s.add_argument("--join", nargs="+", default=None, metavar="BENCH READING=RUN",
                   help="call the benchmark folder's probe.py join() over the named readings")
    s.add_argument("--results", default=None, metavar="DIR", help="results-store root for --join / --merge-results")
    s.add_argument("--merge-results", action="store_true", help="merge every model's results_*.json into one results_<timestamp>.json")
    s.add_argument("--suite", default=None, metavar="S", help="append the suite's completeness per model (done / missing / join-missing)")
    s.set_defaults(func=cmd_report_build)

    # plan
    sp = sub.add_parser("plan", parents=[common_r], help="build / status / calibrate / run (--local) unit plans")
    plsub = sp.add_subparsers(dest="subcommand", metavar="<subcommand>")
    plsub.required = True
    s = plsub.add_parser("init", parents=[common_r], help="write a matrix.yaml skeleton and the bridge stubs for one model on a suite")
    s.add_argument("--suite", required=True, metavar="S", help="the suite (yavih suite list)")
    s.add_argument("--model", required=True, help="the model NAME (every bridge of this model uses it; results are filed under it)")
    s.add_argument("--launcher", required=True, metavar="PYTHON", help="the model environment's python (matrix defaults.launcher)")
    s.add_argument("--out", default="matrix.yaml", metavar="matrix.yaml", help="the matrix file (default ./matrix.yaml)")
    s.add_argument("--bridges-dir", default="bridges", metavar="DIR", help="bridge files and profiles, relative to the matrix (default bridges)")
    s.add_argument("--force", action="store_true", help="overwrite an existing matrix file (bridge files are always kept)")
    s.set_defaults(func=cmd_plan_init)
    s = plsub.add_parser("build", parents=[common_r])
    s.add_argument("matrix", metavar="<matrix.yaml>")
    s.add_argument("--out", default=None, metavar="plan.jsonl")
    s.add_argument("--pending", action="store_true")
    s.add_argument("--kind", choices=["extract", "probe", "report"], default=None)
    s.add_argument("--shard-minutes", type=int, default=None)
    s.add_argument("--format", choices=["jsonl", "count"], default="jsonl")
    s.set_defaults(func=cmd_plan_build)
    s = plsub.add_parser("status", parents=[common_r])
    s.add_argument("plan", metavar="<plan.jsonl>")
    s.add_argument("--root", default=None, metavar="DIR", help="the directory the plan's relative paths mean here (as exec --root)")
    s.set_defaults(func=cmd_plan_status)
    s = plsub.add_parser("run", parents=[common_r], help="run the plan's pending units on this machine (--local)")
    s.add_argument("plan", metavar="<plan.jsonl>")
    s.add_argument("--local", action="store_true", help="required: run here (there is no SLURM mode; see references/slurm.md)")
    s.add_argument("--slots", type=int, default=1, help="units at a time, slot g on the g-th visible GPU (default 1)")
    s.add_argument("--deadline", default=None, metavar="none|+S|<dur>|<epoch>|slurm")
    s.add_argument("--deadline-margin", type=float, default=900.0)
    s.add_argument("--root", default=None, metavar="DIR", help="the directory the plan's relative paths mean here")
    s.add_argument("--cpus", type=int, default=None, help="CPUs per unit (decode workers = cpus - 1)")
    s.add_argument("--deps", default=None, metavar="plan.jsonl", help="full plan for dependency lookup (when <plan.jsonl> is a pending file)")
    s.set_defaults(func=cmd_plan_run)
    s = plsub.add_parser("calibrate", parents=[common_r])
    s.add_argument("runs_root", metavar="<runs_root>")
    s.set_defaults(func=cmd_plan_calibrate)

    # exec
    sp = sub.add_parser("exec", parents=[common_r], help="run one plan unit")
    sp.add_argument("plan", metavar="<plan.jsonl>")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, default=None)
    g.add_argument("--id", default=None)
    g.add_argument("--range", default=None, metavar="a:b", help="units a … b-1 in plan order (with --slots: K at a time)")
    sp.add_argument("--deadline", default=None, metavar="none|+S|<dur>|<epoch>|slurm")
    sp.add_argument("--deadline-margin", type=float, default=900.0,
                    help="--deadline slurm = SLURM_JOB_END_TIME - margin s; keep it above the sbatch --signal lead (600)")
    sp.add_argument("--slots", type=int, default=None)
    sp.add_argument("--deps", default=None, metavar="plan.jsonl",
                    help="full plan used only to look up dependencies (needed when <plan.jsonl> is a --pending file)")
    sp.add_argument("--root", default=None, metavar="DIR", help="the directory the plan's relative paths mean here (a container mount)")
    sp.add_argument("--cpus", type=int, default=None, help="CPUs per unit (decode workers = cpus - 1); default: needs.cpus, clamped to the usable CPUs")
    sp.add_argument("--plan-sha", default=None, metavar="HEX", help="refuse unless the plan file's sha256 starts with HEX (plan build --format count --json data.sha256)")
    sp.set_defaults(func=cmd_exec)

    return p


# ----------------------------------------------------------------------------- entry point


def _emit(env: Envelope, as_json: bool, *, out: Any = None) -> int:
    if as_json:
        out = sys.stdout if out is None else out
        out.write(json.dumps(env.to_json(), sort_keys=True, ensure_ascii=False) + "\n")
        out.flush()
        return env.exit_code
    for w in env.warnings:
        sys.stderr.write(f"[yavih] warning {w['code']}: {w['message']}" + (f" ({w['where']})" if w.get("where") else "") + "\n")
    for e in env.errors:
        sys.stderr.write(f"[yavih] error {e['code']}: {e['message']}" + (f" ({e['where']})" if e.get("where") else "") + "\n")
    if env.text:
        sys.stdout.write(env.text)
        sys.stdout.flush()
    if env.status != "ok":
        sys.stderr.write(f"[yavih] status {env.status} (exit {env.exit_code})\n")
    return env.exit_code


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Under ``--json`` stdout must be exactly one envelope, but a command runs third-party code
    in-process (a bridge's model constructor, ``describe``/``encode``, a library) that may ``print``. While
    the command runs, Python-level ``sys.stdout`` and, when it is a real descriptor, fd 1 itself point at
    stderr; the envelope is written to the saved stdout afterwards. The fd
    swap also covers C-level writes and inherited child descriptors; it is skipped (Python-level only) when
    stdout has no usable descriptor, e.g. under a capturing test harness."""
    saved_fd: int | None = None
    try:
        sys.stdout.flush()
    except (OSError, ValueError):  # pragma: no cover - closed stdout
        pass
    try:
        if sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2:
            saved_fd = os.dup(1)
            os.dup2(2, 1)
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        saved_fd = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        try:
            sys.stderr.flush()
        except (OSError, ValueError):  # pragma: no cover
            pass
        if saved_fd is not None:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)


@contextlib.contextmanager
def _config_layer(ns: argparse.Namespace) -> Iterator[None]:
    """Install the CLI configuration layer (``--config/--runs-root/--results/--data/--lenient``) for this
    command and export it to the environment, so the children a command starts (``exec`` → ``extract`` /
    ``probe run``) resolve the same roots. Both are undone afterwards (``main`` may run in-process)."""
    from yavih import config

    results = getattr(ns, "cfg_results", None)
    if results is None and isinstance(getattr(ns, "results", None), str):
        results = ns.results  # `probe run --results` / `report build --results`: the same root
    saved = {k: os.environ.get(k) for k in os.environ if k.startswith("YAVIH_")}
    try:
        config.set_cli(config_file=getattr(ns, "cfg_config", None), runs_root=getattr(ns, "cfg_runs_root", None), results=results,
                       data=getattr(ns, "cfg_data", None), lenient=getattr(ns, "cfg_lenient", None))
        layer = config.cli_overrides()
        if layer.get("config_file"):
            os.environ["YAVIH_CONFIG"] = layer["config_file"]
        if layer.get("lenient"):
            os.environ["YAVIH_LENIENT"] = "1"
        for key, value in (layer.get("entries") or {}).items():
            if key == ("runs", "root"):
                os.environ["YAVIH_RUNS"] = value
            elif key == ("results", "root"):
                os.environ["YAVIH_RESULTS"] = value
            elif key == ("data", "root"):
                os.environ[f"{config.DATA_ENV_PREFIX}ROOT"] = value
            elif key[0] == "datasets":
                os.environ[f"{config.DATASETS_ENV_PREFIX}{key[1].upper()}"] = value
            elif key[0] == "data":
                os.environ[config.data_env_var(key[1], key[2])] = value
        yield
    finally:
        config.set_cli()
        for k in [k for k in os.environ if k.startswith("YAVIH_")]:
            if k not in saved:
                del os.environ[k]
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _wants_json(argv: Sequence[str]) -> bool:
    return "--json" in argv or os.environ.get("YAVIH_JSON", "") == "1"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, dispatch, wrap the result in the ``--json`` envelope, map exceptions to exit codes."""
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        ns = parser.parse_args(argv)
    except SystemExit as e:  # argparse already printed usage/help to stdout/stderr
        code = e.code if isinstance(e.code, int) else EXIT_USAGE
        if code == 0:
            return EXIT_OK
        if _wants_json(argv):  # keep the "exactly one object on stdout" contract for usage errors too
            env = Envelope(status="error", errors=[_finding("E_USAGE", f"invalid arguments: {' '.join(argv)}")], exit_code=EXIT_USAGE)
            return _emit(env, True)
        return EXIT_USAGE
    as_json = bool(getattr(ns, "json", False)) or os.environ.get("YAVIH_JSON", "") == "1"
    ns.json = as_json
    real_stdout = sys.stdout
    from yavih.config import drain_warnings

    drain_warnings()
    try:
        with _config_layer(ns):
            if as_json:
                with _stdout_to_stderr():
                    env = ns.func(ns)
            else:
                env = ns.func(ns)
    except BaseException as exc:  # noqa: BLE001 - the CLI boundary maps everything to an exit code
        reason = getattr(exc, "reason", None)
        if exc.__class__.__name__ == "NotApplicable" and isinstance(reason, str):
            env = Envelope(status="not_applicable", data={"reason": reason}, exit_code=EXIT_OK, text=f"not applicable: {reason}\n")
        else:
            exit_code, code = exit_code_for(exc)
            msg = str(exc) or exc.__class__.__name__
            if exit_code == EXIT_INTERNAL and not as_json:
                traceback.print_exc(file=sys.stderr)
            where = getattr(exc, "where", None)
            env = Envelope(status="error", data={}, errors=[_finding(code, msg, where if isinstance(where, str) else None)], exit_code=exit_code)
    if as_json:  # library warnings (yavih.config.warn_once) already went to stderr; --json also carries them
        seen = {(w.get("code"), w.get("message")) for w in env.warnings}
        env.warnings += [_finding(w["code"], w["message"], w.get("where")) for w in drain_warnings() if (w["code"], w["message"]) not in seen]
    return _emit(env, as_json, out=real_stdout)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
