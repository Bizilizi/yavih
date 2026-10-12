"""Shipped files: one command surface for what the wheel carries (benchmark folders, bridge templates, prepare
scripts, the agent skill), identical from a wheel install and a source checkout.

An agent never searches ``site-packages`` or a repository for these files: ``yavih bench files|show|prepare``,
``yavih bridge new`` and ``yavih skill show`` reach them (this module is their backend, re-exported as
``yavih.benchmark.bench_files`` / ``bench_file_text``). Paths are resolved through the benchmark resolution order
(``yavih.benchmark.resolve_folder``: ``$YAVIH_HOME/benchmarks`` → packaged → ``$YAVIH_BENCH_PATH`` → entry-point
packs → alias), never through the current directory, and every relative path is checked to stay inside the folder.

stdlib only (plus ``yavih.benchmark`` for the folder resolution); torch-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "FILE_KINDS",
    "SHORT_NAMES",
    "SKILL_STAMP_FILE",
    "ShippedFileError",
    "bench_folder",
    "bench_files",
    "bench_file",
    "bench_file_text",
    "prepare_steps",
    "prepare_step",
    "run_prepare",
    "bridge_new",
    "default_bridge_path",
    "skill_root",
    "skill_files",
    "skill_stamp",
    "skill_reference",
    "installed_from",
    "installed_skills",
]

#: ``kind`` of a shipped folder file (``yavih bench files``).
FILE_KINDS: tuple[str, ...] = ("readme", "setup", "yaml", "template", "probe", "manifest", "classes", "prepare", "test", "other")
#: ``yavih bench show <b> <what>``: the short names and the file each one means.
SHORT_NAMES: dict[str, str] = {
    "readme": "README.md",
    "setup": "setup.md",
    "yaml": "benchmark.yaml",
    "template": "bridge_template.py",
    "probe": "probe.py",
    "classes": "classes.csv",
    "manifest": "manifest.csv.gz",
}
_KIND_OF_NAME: dict[str, str] = {v: k for k, v in SHORT_NAMES.items()}
#: Never part of the shipped surface: bytecode, maintainer notes (not in the wheel), hidden files (fetch stamps).
_SKIP_PARTS = frozenset({"__pycache__", "notes"})
#: A numbered prepare step: ``prepare/01_download.sh``, ``prepare/03_materialize.py``.
_STEP_RE = re.compile(r"^(\d+)_([A-Za-z0-9_.-]+)\.(sh|py)$")
#: The stamp ``yavih skill install`` writes next to an installed skill directory (``W_SKILL_STALE`` in doctor).
SKILL_STAMP_FILE = ".yavih_skill.json"
_STAMP_RE = re.compile(r"<!-- yavih skill stamp: (\{.*?\}) -->")


class ShippedFileError(ValueError):
    """A shipped-file request that cannot be served (exit 1): unknown file, a path outside the folder, a binary file
    for ``show``, an unknown prepare step, an existing ``bridge new`` destination."""

    exit_code = 1

    def __init__(self, message: str, code: str = "E_NOT_FOUND") -> None:
        super().__init__(message)
        self.code = code


# ----------------------------------------------------------------------------- benchmark folders


def bench_folder(name: str) -> Path:
    """The folder of benchmark ``name`` (name, alias or folder path), resolved like ``Benchmark.load`` resolves it
    but without loading the manifest or the configuration, so the documents stay readable when data roots or the
    config do not resolve. Raises ``yavih.benchmark.UnknownBenchmark`` (exit 1)."""
    from yavih.benchmark import UnknownBenchmark, resolve_folder

    folder = resolve_folder(str(name))
    if folder is None:
        from yavih.benchmark import packaged_benchmarks

        raise UnknownBenchmark(f"unknown benchmark {name!r}; packaged: {', '.join(packaged_benchmarks())} (`yavih bench list`)")
    return Path(folder).resolve()


def _kind(rel: Path) -> str:
    parts = rel.parts
    if parts[0] == "prepare":
        return "prepare"
    if parts[0] == "tests":
        return "test"
    if len(parts) == 1 and parts[0] in _KIND_OF_NAME:
        return _KIND_OF_NAME[parts[0]]
    return "other"


def _visible(rel: Path) -> bool:
    return not any(p in _SKIP_PARTS or p.startswith(".") for p in rel.parts) and rel.suffix != ".pyc"


def bench_files(name: str) -> list[dict[str, Any]]:
    """Every shipped file of the benchmark folder: ``[{rel, path, kind, size}]`` sorted by ``rel`` (absolute
    ``path``; ``kind`` ∈ :data:`FILE_KINDS`). Maintainer ``notes/``, bytecode and hidden files are not listed: a
    wheel does not carry them, and the listing is the same from a wheel and a checkout."""
    folder = bench_folder(name)
    out: list[dict[str, Any]] = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(folder)
        if not _visible(rel):
            continue
        out.append({"rel": rel.as_posix(), "path": str(p), "kind": _kind(rel), "size": p.stat().st_size})
    return out


def bench_file(name: str, what: str) -> Path:
    """The absolute path of one shipped file: ``what`` is a short name (:data:`SHORT_NAMES`) or a path relative to
    the folder (``prepare/01_download.sh``, ``tests/test_folder.py``). A path that leaves the folder (``..``, an
    absolute path, a symlink out), names a maintainer note or does not exist is refused (``ShippedFileError``)."""
    folder = bench_folder(name)
    want = str(what).strip()
    rel = SHORT_NAMES.get(want.lower(), want)
    if not rel or os.path.isabs(rel) or rel.startswith("~"):
        raise ShippedFileError(f"{what!r}: give a short name ({', '.join(SHORT_NAMES)}) or a path relative to the folder", "E_PATH")
    path = (folder / rel).resolve()
    try:
        inside = path.relative_to(folder)
    except ValueError:
        raise ShippedFileError(f"{what!r} resolves outside the benchmark folder {folder}", "E_PATH") from None
    if not _visible(inside):
        raise ShippedFileError(f"{what!r} is not a shipped file (maintainer notes, bytecode and hidden files are not)", "E_PATH")
    if not path.is_file():
        files = ", ".join(f["rel"] for f in bench_files(name))
        raise ShippedFileError(f"{Path(folder).name}: no shipped file {rel!r}; the folder holds: {files}")
    return path


def bench_file_text(name: str, what: str) -> str:
    """The text of one shipped file (:func:`bench_file`); a binary file (``manifest.csv.gz``) is refused with its
    path (read it with ``yavih.benchmark.Benchmark``, not as text)."""
    path = bench_file(name, what)
    data = path.read_bytes()
    if path.suffix == ".gz" or b"\x00" in data[:4096]:
        raise ShippedFileError(f"{path.name} is binary ({len(data)} B at {path}); `yavih bench files` lists it, the Benchmark API reads it", "E_BINARY")
    return data.decode("utf-8")


# ----------------------------------------------------------------------------- prepare steps


def _first_doc_line(path: Path) -> str:
    """One line describing a prepare script: the module docstring's first line (.py) or the first comment after the
    shebang (.sh)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
    except OSError:
        return ""
    if path.suffix == ".py":
        text = "\n".join(lines)
        m = re.search(r'^[ \t]*[rbuRBU]*("""|\'\'\')\s*(.*?)\s*$', text, flags=re.M)
        return m.group(2).rstrip('"\'').strip() if m else ""
    for ln in lines:
        s = ln.strip()
        if s.startswith("#!") or not s or set(s) <= {"#", "-", "=", " "}:
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip()
        break
    return ""


def prepare_steps(name: str) -> list[dict[str, Any]]:
    """The numbered prepare steps of the folder (``prepare/<NN>_<name>.{sh,py}``) in order: ``[{step, name, file,
    path, interpreter, doc}]``; ``interpreter`` is ``bash`` or ``python`` (this interpreter, ``sys.executable``)."""
    folder = bench_folder(name)
    out: list[dict[str, Any]] = []
    pdir = folder / "prepare"
    if not pdir.is_dir():
        return out
    for p in sorted(pdir.iterdir()):
        m = _STEP_RE.match(p.name)
        if not m or not p.is_file():
            continue
        out.append({"step": m.group(1), "name": m.group(2), "file": p.name, "path": str(p),
                    "interpreter": "python" if m.group(3) == "py" else "bash", "doc": _first_doc_line(p)[:120]})
    return out


def prepare_step(name: str, step: str) -> dict[str, Any]:
    """One step by number (``1``, ``01``), by name (``download``), by file name (``01_download.sh``) or by any file
    under ``prepare/`` (a helper such as ``derive_targets.py``)."""
    steps = prepare_steps(name)
    s = str(step).strip()
    for st in steps:
        if s.isdigit() and int(s) == int(st["step"]):
            return st
        if s in (st["file"], st["name"], f"{st['step']}_{st['name']}", f"prepare/{st['file']}"):
            return st
    rel = s if s.startswith("prepare/") else f"prepare/{s}"
    try:
        path = bench_file(name, rel)
    except ShippedFileError:
        path = None
    if path is not None and path.suffix in (".sh", ".py"):
        return {"step": None, "name": path.stem, "file": path.name, "path": str(path),
                "interpreter": "python" if path.suffix == ".py" else "bash", "doc": _first_doc_line(path)}
    listing = "; ".join(f"{st['step']} {st['file']}" for st in steps) or "none"
    raise ShippedFileError(f"{name}: no prepare step {step!r} (steps: {listing}; `yavih bench prepare {name}` lists them)", "E_PREPARE_STEP")


def run_prepare(name: str, step: str, args: Sequence[str] = (), *, env: Mapping[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    """Run one prepare step with the right interpreter (``sys.executable`` for ``.py``, ``bash`` for ``.sh``), in the
    current directory, with the caller's environment plus ``FOLDER=<folder>`` and ``YAVIH_BENCH=<name>``.
    Returns ``(exit_code, record)``; stdin/stdout/stderr are inherited."""
    st = prepare_step(name, step)
    folder = bench_folder(name)
    interp = [sys.executable] if st["interpreter"] == "python" else ["bash"]
    cmd = [*interp, st["path"], *[str(a) for a in args]]
    child_env = dict(os.environ if env is None else env)
    child_env["FOLDER"] = str(folder)
    bench_name = _bench_name(folder)
    child_env["YAVIH_BENCH"] = bench_name
    rc = subprocess.call(cmd, env=child_env)
    return int(rc), {**st, "bench": bench_name, "folder": str(folder), "cmd": cmd, "exit_code": int(rc)}


# ----------------------------------------------------------------------------- bridge new

_MODEL_LINE_RE = re.compile(r'("model":\s*)"TODO-model"')
_NAME_LINE_RE = re.compile(r'^(\s*name\s*=\s*)"TODO-model([^"]*)"', flags=re.M)
_READING_LITERAL_RE = re.compile(r'("reading":\s*)"([^"]+)"')
_READING_ARG_RE = re.compile(r'("reading":\s*ArgSpec\(.*?default=)"([^"]+)"(.*?\))', flags=re.S)


def default_bridge_path(bench: str, model: str) -> Path:
    """``bridges/<model>_<bench>.py`` (the layout SKILL.md and the matrix examples use)."""
    return Path("bridges") / f"{model}_{bench}.py"


def bridge_new(bench: str, *, model: str, reading: str | None = None, out: str | os.PathLike | None = None,
               force: bool = False) -> dict[str, Any]:
    """Copy the folder's ``bridge_template.py`` to ``bridges/<model>_<bench>.py`` (``out``: a directory or a file),
    with ``describe()["model"]`` (and the optional file label ``Bridge.name``) filled where the template has its
    ``TODO-model`` placeholder, the reading filled when given (a literal ``"reading"`` value, or the default of a
    ``reading`` argument), and a header naming the commands that print the contract. Everything else stays a
    ``TODO``: ``bridge check`` keeps refusing the copy (``E_TEMPLATE_UNFILLED``: ``weights_ref``, ``norm``) until the
    model's own code is filled in. Refuses to overwrite an existing file unless ``force``."""
    from yavih.bridge import MODEL_RE, READING_RE

    if not isinstance(model, str) or not MODEL_RE.match(model):
        raise ShippedFileError(f"--model {model!r} must match [A-Za-z0-9_.-]{{1,64}} and start with a letter or digit "
                               "(the name every bridge of this model uses; results are filed under it)", "E_ARG")
    if reading is not None and not READING_RE.match(str(reading)):
        raise ShippedFileError(f"--reading {reading!r} must match [A-Za-z0-9_-]{{1,32}} (a | v | av across benchmarks)", "E_ARG")
    folder = bench_folder(bench)
    bench_name = _bench_name(folder)
    src = folder / "bridge_template.py"
    if not src.is_file():
        raise ShippedFileError(f"{bench_name}: the folder {folder} ships no bridge_template.py")
    text = src.read_text(encoding="utf-8")
    filled: list[str] = []
    text, n = _MODEL_LINE_RE.subn(lambda m: f'{m.group(1)}"{model}"', text)
    if n:
        filled.append("describe()['model']")
    text, n = _NAME_LINE_RE.subn(lambda m: f'{m.group(1)}"{(model + m.group(2))[:64]}"', text)
    if n:
        filled.append("Bridge.name")
    if reading is not None:
        text, n_arg = _READING_ARG_RE.subn(lambda m: _reading_arg(m, str(reading)), text)
        n_lit = 0
        if not n_arg:
            text, n_lit = _READING_LITERAL_RE.subn(lambda m: f'{m.group(1)}"{reading}"', text)
        if n_arg:
            filled.append("ARGS['reading'].default")
        elif n_lit:
            filled.append("describe()['reading']")
    dest = Path(out) if out is not None else default_bridge_path(bench_name, model)
    if out is not None and (dest.is_dir() or str(out).endswith(("/", os.sep))):
        dest = dest / default_bridge_path(bench_name, model).name
    if dest.exists() and not force:
        raise ShippedFileError(f"{dest} exists; pass --force to overwrite it (or --out another path)", "E_EXISTS")
    header = (f"# yavih bridge for model {model!r} on benchmark {bench_name!r}: a copy of that folder's bridge_template.py "
              "(`yavih bridge new`).\n"
              f"# Read the contract first: `yavih bench show {bench_name} setup` (Input / Output contract, Gotchas).\n"
              "# Fill every TODO from the model's own code (weights_ref, the input rules, the norm constants and their source);\n"
              f"# then `yavih bridge check {dest.as_posix()} --bench {bench_name} --synthetic`, and `--out <profile.json>` on the staged data.\n")
    text = header + text
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return {"bench": bench_name, "model": model, "reading": reading, "path": str(dest.resolve()), "template": str(src),
            "filled": filled, "next": [f"yavih bench show {bench_name} setup",
                                       f"yavih bridge check {dest} --bench {bench_name} --synthetic"]}


def _reading_arg(m: re.Match, reading: str) -> str:
    head, tail = m.group(1), m.group(3)
    cm = re.search(r"choices=\(([^)]*)\)", tail)
    if cm and f'"{reading}"' not in cm.group(1):
        inner = cm.group(1).rstrip().rstrip(",")
        tail = tail[: cm.start(1)] + f'{inner}, "{reading}"' + tail[cm.end(1):]
    return f'{head}"{reading}"{tail}'


def _bench_name(folder: Path) -> str:
    try:
        import yaml

        d = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
        if isinstance(d, dict) and d.get("name"):
            return str(d["name"])
    except Exception:  # noqa: BLE001 - the folder name is the fallback
        pass
    return folder.name


# ----------------------------------------------------------------------------- the skill

_SKILL_SKIP = frozenset({"__pycache__", ".gitkeep"})


def skill_root() -> Path:
    """The packaged ``yavih/skill/yavih/`` directory (``importlib.resources``)."""
    from importlib import resources

    return Path(str(resources.files("yavih") / "skill" / "yavih"))


def skill_files(root: Path | None = None) -> list[Path]:
    root = skill_root() if root is None else root
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.name != SKILL_STAMP_FILE
            and not any(part in _SKILL_SKIP for part in p.relative_to(root).parts)]


def skill_stamp(root: Path | None = None) -> dict[str, str]:
    """``{yavih_version, sha256}``: sha256 over the packaged skill files (relative path + bytes, sorted). ``skill
    install`` records it; ``doctor`` compares an installed copy's stamp with the packaged one (``W_SKILL_STALE``)."""
    from yavih.versions import __version__

    root = skill_root() if root is None else root
    h = hashlib.sha256()
    for p in skill_files(root):
        rel = p.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "big") + rel)
        data = p.read_bytes()
        h.update(len(data).to_bytes(8, "big") + data)
    return {"yavih_version": __version__, "sha256": h.hexdigest()}


def skill_reference(ref: str) -> Path:
    """``references/<ref>.md`` of the packaged skill (``yavih skill show --ref bridge``)."""
    root = skill_root() / "references"
    stem = Path(str(ref)).name
    stem = stem[:-3] if stem.endswith(".md") else stem
    path = root / f"{stem}.md"
    if not path.is_file():
        names = ", ".join(p.stem for p in sorted(root.glob("*.md")))
        raise ShippedFileError(f"no skill reference {ref!r}; the skill ships: {names}")
    return path


def stamp_comment(stamp: Mapping[str, str]) -> str:
    """The stamp line embedded in a single-file skill (Cursor rule, ``AGENTS.md`` block)."""
    return f"<!-- yavih skill stamp: {json.dumps(dict(stamp), sort_keys=True)} -->"


def read_stamp(path: Path) -> dict | None:
    """The stamp of an installed skill (a directory's stamp file or a single file's stamp comment), else ``None``."""
    try:
        if path.is_dir():
            f = path / SKILL_STAMP_FILE
            return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
        if path.is_file():
            m = _STAMP_RE.search(path.read_text(encoding="utf-8", errors="replace"))
            return json.loads(m.group(1)) if m else None
    except (OSError, ValueError):
        return None
    return None


def installed_skills(targets: Mapping[str, tuple[str, str]], base: Path | None = None) -> list[dict[str, Any]]:
    """Installed copies of the skill at the default target paths under ``base`` (default: the current directory):
    ``[{target, path, stamp, stale}]``. A copy without a stamp (installed by an older yavih) counts as stale."""
    base = Path.cwd() if base is None else base
    current = skill_stamp()
    out: list[dict[str, Any]] = []
    for target, (rel, form) in targets.items():
        p = base / rel
        if form == "dir":
            if not (p / "SKILL.md").is_file():
                continue
        elif not p.is_file() or (target == "agents-md" and "yavih skill: begin" not in p.read_text(encoding="utf-8", errors="replace")):
            continue
        stamp = read_stamp(p)
        stale = stamp is None or stamp.get("sha256") != current["sha256"]
        out.append({"target": target, "path": str(p), "stamp": stamp, "stale": stale})
    return out


# ----------------------------------------------------------------------------- where this yavih comes from


def installed_from() -> dict[str, Any]:
    """``{installed_from: wheel|editable|checkout, package_dir, dist_version}``: ``editable`` = a ``pip install -e``
    (PEP 660 ``direct_url.json`` says so), ``checkout`` = imported from a source tree without an installed
    distribution of it (``PYTHONPATH=src``), ``wheel`` = an installed package (its files live in site-packages)."""
    import yavih

    pkg = Path(yavih.__file__).resolve().parent
    kind = "wheel"
    version = None
    try:
        from importlib import metadata

        dist = metadata.distribution("yavih")
        version = dist.version
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        located = Path(str(dist.locate_file(""))).resolve()
        if (info.get("dir_info") or {}).get("editable"):
            kind = "editable"
        elif not _within(pkg, located):
            kind = "checkout"  # an installed dist exists, but this import comes from elsewhere (PYTHONPATH=src)
    except Exception:  # noqa: BLE001 - no distribution metadata: imported from a source tree
        kind = "checkout" if (pkg.parent.name == "src" and (pkg.parents[1] / "pyproject.toml").is_file()) else "wheel"
    return {"installed_from": kind, "package_dir": str(pkg), "dist_version": version}


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
