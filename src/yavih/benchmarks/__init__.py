"""Packaged benchmark folders: ``yavih/benchmarks/<name>/`` with ``README.md``,
``benchmark.yaml``, ``manifest.csv.gz``, ``classes.csv``, an optional ``probe.py``, ``bridge_template.py``,
``make_manifest.py``, ``prepare/``, ``tests/``.

``Benchmark.load`` finds them through ``importlib.resources`` (the wheel test: every
``benchmark.yaml`` must be reachable this way). The helpers here are stdlib-only and never parse
a yaml; a folder counts as a benchmark iff it holds a ``benchmark.yaml``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["packaged_root", "packaged_names", "packaged_folder", "external_folders", "BENCHMARK_YAML", "BENCH_PATH_ENV",
           "ENTRY_POINT_GROUP"]

BENCHMARK_YAML = "benchmark.yaml"
#: Colon-separated directories of out-of-tree benchmark folders (``config.ENV_VARS``).
BENCH_PATH_ENV = "YAVIH_BENCH_PATH"
#: Entry-point group of pip-installed benchmark packs: ``name = "pkg.module"``, a package whose directory is a
#: benchmark folder or holds benchmark folders (``importlib.resources``; data files, never imported as code).
ENTRY_POINT_GROUP = "yavih.benchmarks"


def packaged_root() -> Path:
    """Filesystem path of ``yavih/benchmarks``. The wheel is installed as directories (never
    imported from a zip), so the ``importlib.resources`` traversable is a real path; a zip
    import would need ``as_file`` staging, which the packaging does not need."""
    root = resources.files(__name__)
    path = Path(str(root))
    if not path.is_dir():
        raise FileNotFoundError(f"packaged benchmarks root is not a directory: {root!r} (zip import unsupported)")
    return path


def packaged_names() -> list[str]:
    """Sorted names of the shipped folders that hold a ``benchmark.yaml``."""
    try:
        root = packaged_root()
    except FileNotFoundError:
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / BENCHMARK_YAML).is_file())


def packaged_folder(name: str) -> Path | None:
    """The shipped folder for ``name`` or ``None``; a name is matched only as a plain folder name
    (no path separators, so a ref like ``../x`` can never escape the package)."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        return None
    try:
        root = packaged_root()
    except FileNotFoundError:
        return None
    folder = root / name
    return folder if (folder / BENCHMARK_YAML).is_file() else None


def _folders_in(root: Path) -> list[Path]:
    """``root`` itself when it is a benchmark folder, else its sub-directories that are (sorted)."""
    if (root / BENCHMARK_YAML).is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / BENCHMARK_YAML).is_file())


def _entry_point_roots() -> list[tuple[str, Path]]:
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return []
    try:
        eps = list(entry_points(group=ENTRY_POINT_GROUP))
    except TypeError:  # pragma: no cover - Python < 3.10 API
        eps = list(entry_points().get(ENTRY_POINT_GROUP, []))  # type: ignore[attr-defined]
    out: list[tuple[str, Path]] = []
    for ep in eps:
        try:
            out.append((f"entry point {ep.name}", Path(str(resources.files(ep.value.split(":", 1)[0])))))
        except Exception as e:  # noqa: BLE001 - a broken pack must name itself, not break discovery
            import warnings

            warnings.warn(f"benchmark entry point {ep.name!r} ({ep.value}) cannot be read: {type(e).__name__}: {e}", stacklevel=2)
    return out


def external_folders(env: dict | None = None) -> dict[str, Path]:
    """``name -> folder`` of the out-of-tree benchmark folders, in resolution order: each directory of
    ``$YAVIH_BENCH_PATH`` (colon-separated, first wins), then each ``yavih.benchmarks`` entry-point pack. A name is
    the folder's directory name; the first occurrence wins. Packaged names are not filtered here (the caller
    decides precedence: ``yavih.benchmark.resolve_folder`` never lets these shadow a packaged folder)."""
    import os

    env = os.environ if env is None else env
    out: dict[str, Path] = {}
    roots = [("YAVIH_BENCH_PATH", Path(p).expanduser()) for p in str(env.get(BENCH_PATH_ENV, "") or "").split(os.pathsep) if p]
    for _origin, root in [*roots, *_entry_point_roots()]:
        for folder in _folders_in(root):
            out.setdefault(folder.name, folder)
    return out
