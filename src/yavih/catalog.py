"""The catalog of benchmark folders: the index of the folders in the yavih repository and the commands that bring
one folder into a project.

The wheel ships the core only. A benchmark folder reaches a user as one directory fetched from the yavih repository
at a pinned commit:

- **The index** (``catalog/index.json`` on the repository's ``main`` branch) lists every folder with its metadata at
  its newest revision and ``revisions``: for each distinct ``(core, protocol_version)`` of the folder's history, the
  newest commit carrying it, with the folder's path and git tree id at that commit (newest first). It is generated
  from git history by ``python -m yavih.catalog --write`` (CI regenerates it on every push to ``main``) and is
  reproducible: ``--check`` regenerates it at the commit it names and compares.
- **Fetch** (``yavih bench fetch <name>[@<commit>]``) reads the index, picks the newest revision whose ``core:``
  specifier the installed core satisfies (an older core gets an older revision; yavih never upgrades itself: when no
  revision fits it prints the pip line and stops), downloads the repository archive at that commit (cached per
  commit under ``$YAVIH_HOME/catalog/archives``), extracts the folder into the benchmarks directory (the project's
  ``yavih-bench/benchmarks/``; ``$YAVIH_HOME/benchmarks/`` outside a project), verifies its git tree id against
  the index, and records the revision in the lock file next to that directory (``yavih-bench/benchmarks.lock``, to
  commit with the project; the fetched folders are listed in the directory's ``.gitignore``).
- **Update** (``yavih bench update``) shows what the newest compatible revision changes (commit, ``protocol_version``,
  ``protocol_hash``, ``core``) and refetches only with ``--yes``. Nothing is ever updated silently, and a fetched
  folder edited in place is never overwritten without ``--refresh``.
- **Offline**: ``--from DIR`` (a yavih checkout: its git history, or a directory holding ``catalog/index.json`` and
  the folders) or ``--from FILE`` (a repository archive ``.tar.gz``, as GitHub serves one per commit).

The git tree id is the identity of a folder's content: the same id git prints for ``git rev-parse
<commit>:benchmarks/<name>``, computed here from the files (``tree_id``), so a fetched folder is verified without git.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

__all__ = [
    "CATALOG_VERSION",
    "LOCK_VERSION",
    "DEFAULT_REPO",
    "DEFAULT_URL",
    "DEFAULT_ARCHIVE",
    "INDEX_PATH",
    "LOCK_NAME",
    "CatalogError",
    "tree_id",
    "build_index",
    "load_index",
    "cached_index",
    "select_revision",
    "benchmarks_dir",
    "lock_path",
    "read_lock",
    "lock_entry",
    "origin",
    "fetch",
    "update",
    "main",
]

#: Bumped when the index schema changes (``catalog/index.json``).
CATALOG_VERSION = 1
#: Bumped when the lock file schema changes (``benchmarks.lock``).
LOCK_VERSION = 1
#: The repository the default index and archives come from.
DEFAULT_REPO = "Bizilizi/yavih"
#: The index on the repository's ``main`` branch (``[catalog] url``).
DEFAULT_URL = f"https://raw.githubusercontent.com/{DEFAULT_REPO}/main/catalog/index.json"
#: The repository archive at one commit (``[catalog] archive``; ``{repo}`` and ``{commit}`` are substituted).
DEFAULT_ARCHIVE = "https://codeload.github.com/{repo}/tar.gz/{commit}"
#: The index inside the repository.
INDEX_PATH = "catalog/index.json"
#: The lock file, next to the benchmarks directory it describes.
LOCK_NAME = "benchmarks.lock"
#: Where a folder lived in the repository: ``benchmarks/<name>`` (current), ``src/yavih/benchmarks/<name>`` (before
#: the wheel shipped the core only).
_FOLDER_PARENTS = ("benchmarks", "src/yavih/benchmarks")
#: Files a checkout or a run may add next to a folder's tracked files; never part of its tree id.
_IGNORED_NAMES = frozenset({"__pycache__", ".DS_Store", ".pytest_cache"})
_IGNORED_SUFFIXES = (".pyc", ".pyo")
_IGNORED_RELPATHS = frozenset({"assets/header.html"})  # python -m yavih.headers --html-only (git-ignored)
_GITIGNORE_BEGIN = "# yavih bench fetch: folders fetched from the catalog (yavih-bench/benchmarks.lock records them)"
_GITIGNORE_END = "# end of yavih bench fetch"
_TIMEOUT_S = 60


class CatalogError(Exception):
    """A catalog, fetch or lock problem; ``code`` and ``exit_code`` are read by the CLI."""

    def __init__(self, code: str, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


# ----------------------------------------------------------------------------- tree ids


def _ignored(rel: PurePosixPath) -> bool:
    return (any(part in _IGNORED_NAMES for part in rel.parts) or rel.name.endswith(_IGNORED_SUFFIXES)
            or rel.as_posix() in _IGNORED_RELPATHS)


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's object id, not security


def _tree_entries(folder: Path, rel: PurePosixPath) -> bytes | None:
    entries: list[tuple[bytes, bytes, bytes]] = []
    for p in folder.iterdir():
        r = rel / p.name
        if _ignored(r):
            continue
        if p.is_symlink():
            mode, oid = b"120000", _blob_id(os.readlink(p).encode())
        elif p.is_dir():
            body = _tree_entries(p, r)
            if body is None:  # git does not record an empty directory
                continue
            mode, oid = b"40000", hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()  # noqa: S324
        else:
            mode = b"100755" if os.stat(p).st_mode & 0o111 else b"100644"
            oid = _blob_id(p.read_bytes())
        # git orders tree entries by name, a sub-tree's name compared as if it ended with "/"
        key = p.name.encode() + (b"/" if mode == b"40000" else b"")
        entries.append((key, mode + b" " + p.name.encode() + b"\0", bytes.fromhex(oid)))
    if not entries:
        return None
    entries.sort(key=lambda e: e[0])
    return b"".join(head + oid for _, head, oid in entries)


def tree_id(folder: str | os.PathLike) -> str:
    """The git tree id of ``folder``'s content (``git rev-parse <commit>:<path>`` of a committed folder): file modes
    (executable or not), names and bytes; ``__pycache__``, ``*.pyc`` and the git-ignored header preview excluded."""
    body = _tree_entries(Path(folder), PurePosixPath())
    body = body or b""
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()  # noqa: S324


# ----------------------------------------------------------------------------- the index (generation)


def _git(repo: Path, *args: str, binary: bool = False) -> Any:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise CatalogError("E_CATALOG_GIT", f"git {' '.join(args)} failed in {repo}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout if binary else proc.stdout.decode()


def _git_ok(repo: Path, *args: str) -> str | None:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=300)
    return proc.stdout.decode().strip() if proc.returncode == 0 else None


def _folder_names(repo: Path, commit: str) -> list[str]:
    """Folder names at ``commit``: the directories under ``benchmarks/`` (or the older ``src/yavih/benchmarks/``)
    that hold a ``benchmark.yaml``."""
    names: set[str] = set()
    for parent in _FOLDER_PARENTS:
        out = _git_ok(repo, "ls-tree", "--name-only", f"{commit}:{parent}")
        for n in (out or "").splitlines():
            if _git_ok(repo, "cat-file", "-e", f"{commit}:{parent}/{n}/benchmark.yaml") is not None:
                names.add(n)
        if names:
            break
    return sorted(names)


def _yaml_at(repo: Path, commit: str, path: str) -> Mapping[str, Any] | None:
    raw = _git_ok(repo, "show", f"{commit}:{path}/benchmark.yaml")
    if raw is None:
        return None
    try:
        d = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    return d if isinstance(d, Mapping) else None


def _path_at(repo: Path, commit: str, name: str) -> str | None:
    for parent in _FOLDER_PARENTS:
        p = f"{parent}/{name}"
        if _git_ok(repo, "cat-file", "-e", f"{commit}:{p}/benchmark.yaml") is not None:
            return p
    return None


def _newest_protocol_hash(d: Mapping[str, Any]) -> str | None:
    log = d.get("changelog")
    if isinstance(log, list) and log and isinstance(log[-1], Mapping):
        h = log[-1].get("protocol_hash")
        return str(h) if h else None
    return None


def _metadata(d: Mapping[str, Any], name: str, path: str) -> dict[str, Any]:
    """What ``bench list`` shows before a folder is fetched, read from the raw yaml (no load: a folder for another
    core is described too)."""
    budget = d.get("budget") if isinstance(d.get("budget"), Mapping) else {}
    obtain = d.get("obtain") if isinstance(d.get("obtain"), Mapping) else {}
    header = d.get("header") if isinstance(d.get("header"), Mapping) else {}
    return {
        "name": name,
        "title": d.get("title"),
        "header_title": header.get("title"),
        "aliases": list(d.get("aliases") or []),
        "capability": list(d.get("capability") or []),
        "scorer": d.get("scorer"),
        "obtain": obtain.get("tier"),
        "media_gb": budget.get("media_gb"),
        "clips": dict(budget.get("clips") or {}),
        "protocol_version": d.get("protocol_version"),
        "protocol_hash": _newest_protocol_hash(d),
        "core": d.get("core"),
        "path": path,
    }


def build_index(repo: str | os.PathLike, *, commit: str = "HEAD", repo_name: str = DEFAULT_REPO) -> dict[str, Any]:
    """The index of the folders of the git repository ``repo`` at ``commit``, from its history alone (the working
    tree is never read, so the index is reproducible from the commit it names). A revision without ``core:`` (a
    folder older than the catalog) is not listed: no core that can fetch could load it."""
    repo = Path(repo)
    head = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    out: dict[str, Any] = {}
    for name in _folder_names(repo, head):
        paths = [f"{parent}/{name}" for parent in _FOLDER_PARENTS]
        commits = _git(repo, "log", "--format=%H", head, "--", *paths).split()
        revisions: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        newest: tuple[Mapping[str, Any], str] | None = None
        for c in commits:
            path = _path_at(repo, c, name)
            if path is None:  # the commit that removed or moved it away
                continue
            d = _yaml_at(repo, c, path)
            if d is None or not d.get("core"):
                continue
            if newest is None:
                newest = (d, path)
            key = (d.get("core"), d.get("protocol_version"))
            if key in seen:
                continue
            seen.add(key)
            revisions.append({"commit": c, "path": path, "tree": _git(repo, "rev-parse", f"{c}:{path}").strip(),
                              "core": d.get("core"), "protocol_version": d.get("protocol_version"),
                              "protocol_hash": _newest_protocol_hash(d)})
        if newest is None:
            continue
        out[name] = {**_metadata(newest[0], name, newest[1]), "revisions": revisions}
    return {"catalog_version": CATALOG_VERSION, "repo": repo_name, "commit": head, "benchmarks": out}


def _dumps(index: Mapping[str, Any]) -> str:
    return json.dumps(index, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


# ----------------------------------------------------------------------------- the index (reading)


def _home() -> Path:
    from yavih import config

    try:
        return config.load().yavih_home()
    except config.ConfigError:
        return config.yavih_home()


def _setting(key: str, default: str) -> str:
    from yavih import config

    try:
        v = config.load().get("catalog", key)
    except config.ConfigError:
        v = None
    return str(v.value) if v is not None else default


def _download(url: str, dest: Path) -> Path:
    """``url`` (``https://``, ``http://`` or ``file://``) into ``dest`` (atomically)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".part{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as r, open(tmp, "wb") as fh:  # noqa: S310 - configured URL
            shutil.copyfileobj(r, fh)
        os.replace(tmp, dest)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise CatalogError("E_CATALOG_UNREACHABLE", f"cannot download {url}: {e} (offline? `yavih bench fetch <name> --from "
                           "<yavih checkout or archive>`; or set [catalog] url / archive)", exit_code=3) from e
    return dest


def _check_index(index: Any, where: str) -> dict[str, Any]:
    if not isinstance(index, Mapping) or not isinstance(index.get("benchmarks"), Mapping):
        raise CatalogError("E_CATALOG_INDEX", f"{where} is not a yavih catalog index", exit_code=2)
    v = index.get("catalog_version")
    if v != CATALOG_VERSION:
        raise CatalogError("E_CATALOG_INDEX", f"{where}: catalog_version {v!r}; this yavih reads {CATALOG_VERSION} "
                           "(a newer index: pip install --no-deps -U yavih)", exit_code=3)
    return dict(index)


def cached_index() -> dict[str, Any] | None:
    """The last index downloaded (``$YAVIH_HOME/catalog/index.json``), ``None`` when there is none. No network."""
    p = _home() / "catalog" / "index.json"
    try:
        return _check_index(json.loads(p.read_text(encoding="utf-8")), str(p))
    except (OSError, ValueError, CatalogError):
        return None


def load_index(source: str | os.PathLike | None = None, *, offline: bool = False) -> dict[str, Any]:
    """The index from ``source`` (a yavih checkout or a directory holding ``catalog/index.json``, a repository
    archive, an index file) or, without one, from ``[catalog] url`` (cached under ``$YAVIH_HOME/catalog``; the cache
    is used when the download fails, and alone with ``offline``)."""
    if source is not None:
        return _index_of_source(Path(source))
    url = _setting("url", DEFAULT_URL)
    cache = _home() / "catalog" / "index.json"
    if not offline:
        try:
            _download(url, cache)
        except CatalogError:
            if cached_index() is None:
                raise
            from yavih import config

            config.warn_once("W_CATALOG_CACHED", f"the catalog index at {url} is unreachable: using the cached copy {cache}")
    idx = cached_index()
    if idx is None:
        raise CatalogError("E_CATALOG_UNREACHABLE", f"no catalog index cached at {cache} (fetch one online, or --from <checkout>)", exit_code=3)
    return idx


def _is_git_checkout(d: Path) -> bool:
    return (d / ".git").exists()


def _index_of_source(src: Path) -> dict[str, Any]:
    if src.is_dir():
        if _is_git_checkout(src) and any((src / p).is_dir() for p in _FOLDER_PARENTS):
            return build_index(src)  # the checkout's own history: current even on a branch the published index lags
        idx = src / INDEX_PATH
        if idx.is_file():
            return _check_index(json.loads(idx.read_text(encoding="utf-8")), str(idx))
        raise CatalogError("E_CATALOG_SOURCE", f"{src} is neither a yavih git checkout nor a directory holding {INDEX_PATH}")
    if src.is_file() and _is_archive(src):
        with tarfile.open(src) as tf:
            member = next((m for m in tf.getmembers() if m.isfile() and _strip_top(m.name) == INDEX_PATH), None)
            if member is None:
                raise CatalogError("E_CATALOG_SOURCE", f"{src} holds no {INDEX_PATH}")
            fh = tf.extractfile(member)
            assert fh is not None
            return _check_index(json.loads(fh.read().decode("utf-8")), f"{src}:{INDEX_PATH}")
    if src.is_file():
        return _check_index(json.loads(src.read_text(encoding="utf-8")), str(src))
    raise CatalogError("E_CATALOG_SOURCE", f"--from {src}: no such directory or file")


# ----------------------------------------------------------------------------- choosing a revision


def _resolve_name(index: Mapping[str, Any], ref: str) -> str:
    benches = index["benchmarks"]
    if ref in benches:
        return ref
    for name, e in benches.items():
        if ref in (e.get("aliases") or []):
            return name
    raise CatalogError("E_CATALOG_UNKNOWN", f"no benchmark {ref!r} in the catalog; known: {sorted(benches)}")


def select_revision(entry: Mapping[str, Any], *, version: str | None = None, commit: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(revision, core_status)``: the newest revision whose ``core`` the installed ``version`` satisfies (a
    development build takes the newest, reported), or the revision at ``commit`` (a prefix). No revision fits:
    ``E_CORE_VERSION`` (exit 3) with the pip line of the newest one; yavih never upgrades itself."""
    from yavih.benchmark import core_status

    revs = list(entry.get("revisions") or [])
    if not revs:
        raise CatalogError("E_CATALOG_INDEX", f"{entry.get('name')}: the index lists no revision", exit_code=2)
    if commit is not None:
        hit = [r for r in revs if str(r["commit"]).startswith(commit)]
        if len(hit) != 1:
            raise CatalogError("E_CATALOG_UNKNOWN", f"{entry.get('name')}@{commit}: {'ambiguous' if hit else 'not a listed revision'}; "
                               f"listed: {[str(r['commit'])[:12] for r in revs]}")
        return dict(hit[0]), core_status(hit[0].get("core"), version)
    first_dev: tuple[dict[str, Any], dict[str, Any]] | None = None
    for r in revs:
        st = core_status(r.get("core"), version)
        if st["ok"] is not False:
            return dict(r), st
        if st["dev"] and first_dev is None:
            first_dev = (dict(r), st)
    if first_dev is not None:
        return first_dev
    newest = revs[0]
    st = core_status(newest.get("core"), version)
    raise CatalogError("E_CORE_VERSION", f"{entry.get('name')}: no revision fits yavih {st['version']} (newest needs "
                       f"{newest.get('core')}). Install a core that does: pip install --no-deps \"yavih{newest.get('core')}\" "
                       "(yavih never upgrades itself)", exit_code=3)


# ----------------------------------------------------------------------------- where fetched folders live


def benchmarks_dir() -> Path:
    """The directory fetched folders go to: the project's benchmarks directory (``yavih-bench/benchmarks``, or
    ``[project] benchmarks``), else ``$YAVIH_HOME/benchmarks``."""
    from yavih import config

    d = config.project_benchmarks_dir()
    return d if d is not None else _home() / "benchmarks"


def lock_path(bench_dir: Path) -> Path:
    """The lock file of a benchmarks directory: next to it (``yavih-bench/benchmarks.lock``)."""
    return bench_dir.parent / LOCK_NAME


def read_lock(path: Path) -> dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"lock_version": LOCK_VERSION, "benchmarks": {}}
    except (OSError, ValueError) as e:
        raise CatalogError("E_LOCK", f"{path} is not a yavih lock file: {e}", exit_code=2) from e
    if not isinstance(d, Mapping) or not isinstance(d.get("benchmarks"), Mapping):
        raise CatalogError("E_LOCK", f"{path} is not a yavih lock file", exit_code=2)
    return {"lock_version": d.get("lock_version", LOCK_VERSION), "benchmarks": dict(d["benchmarks"])}


def _write_lock(path: Path, lock: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"lock_version": LOCK_VERSION, "benchmarks": dict(sorted(lock["benchmarks"].items()))},
                              indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def lock_entry(folder: str | os.PathLike) -> dict[str, Any] | None:
    """The lock record of a fetched ``folder`` (``<dir>/<name>`` with ``<dir>/../benchmarks.lock``), else ``None``."""
    folder = Path(folder)
    p = lock_path(folder.parent)
    if not p.is_file():
        return None
    try:
        e = read_lock(p)["benchmarks"].get(folder.name)
    except CatalogError:
        return None
    return dict(e) if isinstance(e, Mapping) else None


def origin(folder: str | os.PathLike) -> dict[str, Any]:
    """``{repo, commit, tree}`` of a folder: the lock's repository and commit for a fetched folder (``None``
    otherwise) and the tree id of its current content. A fetched folder whose content no longer matches its lock is
    reported once (``W_FETCHED_MODIFIED``)."""
    folder = Path(folder)
    tree = tree_id(folder)
    e = lock_entry(folder)
    if e is None:
        return {"repo": None, "commit": None, "tree": tree}
    if e.get("tree") and e["tree"] != tree:
        from yavih import config

        config.warn_once("W_FETCHED_MODIFIED", f"{folder} was edited after `yavih bench fetch` (tree {tree[:12]}, locked "
                         f"{str(e['tree'])[:12]} at {str(e.get('commit'))[:12]}); results record the edited tree. "
                         f"`yavih bench fetch {folder.name} --refresh` restores it", where=str(folder))
    return {"repo": e.get("repo"), "commit": e.get("commit"), "tree": tree}


def _update_gitignore(bench_dir: Path, names: Sequence[str]) -> None:
    """Keep the managed block of ``<bench_dir>/.gitignore`` listing the fetched folders (a project's own folders in
    the same directory stay tracked)."""
    p = bench_dir / ".gitignore"
    text = p.read_text(encoding="utf-8") if p.is_file() else ""
    lines = text.splitlines()
    if _GITIGNORE_BEGIN in lines and _GITIGNORE_END in lines:
        i, j = lines.index(_GITIGNORE_BEGIN), lines.index(_GITIGNORE_END)
        lines = lines[:i] + lines[j + 1:]
    block = [_GITIGNORE_BEGIN, *[f"/{n}/" for n in sorted(names)], _GITIGNORE_END] if names else []
    while lines and not lines[-1].strip():
        lines.pop()
    new = "\n".join([*lines, *([""] if lines and block else []), *block]) + ("\n" if lines or block else "")
    if new != text:
        bench_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(new, encoding="utf-8")


# ----------------------------------------------------------------------------- materializing a folder


def _is_archive(p: Path) -> bool:
    return p.name.endswith((".tar.gz", ".tgz", ".tar"))


def _strip_top(name: str) -> str:
    """``<top>/<rest>`` (an archive's single top directory) → ``<rest>``."""
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else ""


def _safe_members(tf: tarfile.TarFile, path: str, *, strip: bool) -> Iterator[tuple[tarfile.TarInfo, PurePosixPath]]:
    """The regular files and symlinks of ``path`` in an archive, with their path relative to the folder; anything
    that would leave the folder (absolute names, ``..``, links pointing outside) is refused."""
    prefix = PurePosixPath(path)
    for m in tf.getmembers():
        name = _strip_top(m.name) if strip else m.name
        rel = PurePosixPath(name)
        try:
            sub = rel.relative_to(prefix)
        except ValueError:
            continue
        if sub.as_posix() in ("", ".") or m.isdir():
            continue
        if rel.is_absolute() or ".." in sub.parts:
            raise CatalogError("E_CATALOG_ARCHIVE", f"archive member {m.name!r} leaves the folder", exit_code=2)
        if m.issym():
            target = PurePosixPath(m.linkname)
            if target.is_absolute() or ".." in (sub.parent / target).parts:
                raise CatalogError("E_CATALOG_ARCHIVE", f"archive link {m.name!r} -> {m.linkname!r} leaves the folder", exit_code=2)
        elif not m.isfile():
            continue
        yield m, sub


def _extract(tf: tarfile.TarFile, path: str, dest: Path, *, strip: bool) -> int:
    n = 0
    for m, sub in _safe_members(tf, path, strip=strip):
        out = dest / sub
        out.parent.mkdir(parents=True, exist_ok=True)
        if m.issym():
            os.symlink(m.linkname, out)
        else:
            fh = tf.extractfile(m)
            assert fh is not None
            out.write_bytes(fh.read())
            os.chmod(out, 0o755 if m.mode & 0o111 else 0o644)
        n += 1
    return n


def _archive_for(commit: str, repo: str) -> Path:
    """The repository archive at ``commit`` (downloaded once; cached under ``$YAVIH_HOME/catalog/archives``)."""
    cache = _home() / "catalog" / "archives" / f"{commit}.tar.gz"
    if not cache.is_file():
        url = _setting("archive", DEFAULT_ARCHIVE).format(repo=repo, commit=commit)
        _download(url, cache)
    return cache


def _materialize(rev: Mapping[str, Any], *, repo: str, source: Path | None, dest: Path) -> str:
    """Write the folder of ``rev`` into the empty directory ``dest``; returns where it came from."""
    path, commit = str(rev["path"]), str(rev["commit"])
    if source is not None and source.is_dir() and _is_git_checkout(source):
        data = _git(source, "archive", "--format=tar", commit, path, binary=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            n = _extract(tf, path, dest, strip=False)
        where = f"git {source}@{commit[:12]}"
    elif source is not None and source.is_dir():
        src = source / path
        if not (src / "benchmark.yaml").is_file():
            raise CatalogError("E_CATALOG_SOURCE", f"{source} holds no {path}")
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*_IGNORED_NAMES, "*.pyc", "*.pyo"))
        n = 1
        where = str(source)
    else:
        archive = source if source is not None and source.is_file() else _archive_for(commit, repo)
        with tarfile.open(archive) as tf:
            n = _extract(tf, path, dest, strip=True)
        where = str(archive) if source is not None else f"{_setting('archive', DEFAULT_ARCHIVE).format(repo=repo, commit=commit)}"
    if n == 0:
        raise CatalogError("E_CATALOG_SOURCE", f"{path} at {commit[:12]} is empty in {where}", exit_code=2)
    return where


def _archive_revision(entry: Mapping[str, Any], archive: Path) -> dict[str, Any]:
    """The revision a repository archive holds: the one whose tree id equals the archive's folder."""
    with tempfile.TemporaryDirectory(prefix="yavih-fetch-") as tmp:
        for rev in entry.get("revisions") or []:
            d = Path(tmp) / str(rev["commit"])
            d.mkdir()
            with tarfile.open(archive) as tf:
                if _extract(tf, str(rev["path"]), d, strip=True) and tree_id(d) == rev["tree"]:
                    return dict(rev)
    raise CatalogError("E_CATALOG_TREE", f"{archive} holds no revision of {entry.get('name')} the index lists", exit_code=2)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch(ref: str, *, source: str | os.PathLike | None = None, refresh: bool = False, dest_dir: str | os.PathLike | None = None,
          version: str | None = None, index: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fetch the folder ``ref`` (a name or alias, ``name@<commit>`` for a listed revision) from the catalog into
    ``dest_dir`` (default :func:`benchmarks_dir`), verify it and record it in the lock. Returns ``{name, folder,
    commit, tree, core, protocol_version, protocol_hash, status, core_status, source, lock}`` with ``status``
    ``fetched``, ``refreshed`` or ``unchanged`` (already there at that revision)."""
    name_ref, _, commit = str(ref).partition("@")
    src = Path(source).expanduser() if source is not None else None
    if index is None:
        index = load_index(src)
    name = _resolve_name(index, name_ref)
    entry = index["benchmarks"][name]
    if src is not None and src.is_file() and _is_archive(src) and not commit:
        rev = _archive_revision(entry, src)
        from yavih.benchmark import core_status

        st = core_status(rev.get("core"), version)
        if st["ok"] is False and not st["dev"]:
            raise CatalogError("E_CORE_VERSION", f"{name} in {src} needs yavih core {rev.get('core')}; this is yavih {st['version']}: "
                               f"pip install --no-deps \"yavih{rev.get('core')}\"", exit_code=3)
    else:
        rev, st = select_revision(entry, version=version, commit=commit or None)
    bench_dir = Path(dest_dir) if dest_dir is not None else benchmarks_dir()
    folder = bench_dir / name
    lockp = lock_path(bench_dir)
    lock = read_lock(lockp)
    old = lock["benchmarks"].get(name)
    status = "fetched"
    if folder.exists():
        current = tree_id(folder)
        locked = isinstance(old, Mapping) and old.get("tree") == current
        if current == rev["tree"] and not refresh:
            status = "unchanged"
        elif not locked and not refresh:
            raise CatalogError("E_FETCH_MODIFIED", f"{folder} exists and is not the folder the lock records (edited, or not "
                               f"fetched): fetching would overwrite it. Re-run with --refresh to replace it", exit_code=1)
        else:
            status = "refreshed"
    if status != "unchanged":
        bench_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}.fetch-", dir=bench_dir))
        try:
            where = _materialize(rev, repo=str(index.get("repo") or DEFAULT_REPO), source=src, dest=staging / name)
            got = tree_id(staging / name)
            if got != rev["tree"]:
                raise CatalogError("E_CATALOG_TREE", f"{name}@{str(rev['commit'])[:12]} from {where}: tree {got} != {rev['tree']} "
                                   "(the index lists another content: a corrupt download or a wrong source)", exit_code=2)
            from yavih.benchmark import Benchmark

            Benchmark.from_dir(staging / name)  # the manifest's sha256 and the schema, before the folder is installed
            if folder.exists():
                shutil.rmtree(folder)
            os.replace(staging / name, folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    else:
        where = (old or {}).get("source") if isinstance(old, Mapping) else None
    record = {"repo": index.get("repo") or DEFAULT_REPO, "commit": rev["commit"], "path": rev["path"], "tree": rev["tree"],
              "core": rev.get("core"), "protocol_version": rev.get("protocol_version"), "protocol_hash": rev.get("protocol_hash"),
              "source": where, "fetched_at": (old or {}).get("fetched_at") if status == "unchanged" and isinstance(old, Mapping) else _now()}
    if old != record:
        lock["benchmarks"][name] = record
        _write_lock(lockp, lock)
    _update_gitignore(bench_dir, list(lock["benchmarks"]))
    return {"name": name, "folder": str(folder), "status": status, "commit": rev["commit"], "tree": rev["tree"],
            "core": rev.get("core"), "protocol_version": rev.get("protocol_version"), "protocol_hash": rev.get("protocol_hash"),
            "core_status": st, "source": where, "lock": str(lockp)}


def update(names: Sequence[str] | None = None, *, yes: bool = False, source: str | os.PathLike | None = None,
           dest_dir: str | os.PathLike | None = None, version: str | None = None) -> list[dict[str, Any]]:
    """For each locked folder (or ``names``): the newest compatible revision against the locked one. Without
    ``yes`` nothing changes; with it an unmodified folder is refetched (an edited one is reported, ``modified``)."""
    bench_dir = Path(dest_dir) if dest_dir is not None else benchmarks_dir()
    lock = read_lock(lock_path(bench_dir))["benchmarks"]
    wanted = list(names) if names else sorted(lock)
    index = load_index(Path(source).expanduser() if source is not None else None)
    out: list[dict[str, Any]] = []
    for n in wanted:
        name = _resolve_name(index, n)
        cur = lock.get(name)
        if not isinstance(cur, Mapping):
            out.append({"name": name, "status": "not_fetched"})
            continue
        rev, st = select_revision(index["benchmarks"][name], version=version)
        changes = {k: {"locked": cur.get(k), "available": rev.get(k)} for k in ("commit", "core", "protocol_version", "protocol_hash")
                   if cur.get(k) != rev.get(k)}
        folder = bench_dir / name
        modified = folder.exists() and tree_id(folder) != cur.get("tree")
        rec: dict[str, Any] = {"name": name, "locked": cur.get("commit"), "available": rev["commit"], "changes": changes,
                               "modified": modified, "core_status": st}
        if rev["tree"] == cur.get("tree"):
            rec["status"] = "up_to_date"
        elif not yes:
            rec["status"] = "update_available"
        elif modified:
            rec["status"] = "modified"
        else:
            r = fetch(f"{name}@{rev['commit']}", source=source, refresh=True, dest_dir=bench_dir, version=version, index=index)
            rec["status"] = "updated"
            rec["folder"] = r["folder"]
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- generator CLI


def _repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    from yavih.benchmarks import checkout_root

    root = checkout_root()
    if root is None:
        raise SystemExit("python -m yavih.catalog runs in a yavih source checkout (or pass --repo DIR)")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yavih.catalog", description="Generate or check catalog/index.json from the "
                                 "repository's git history (the index `yavih bench fetch` reads).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="write catalog/index.json for HEAD")
    g.add_argument("--check", action="store_true", help="exit 1 when catalog/index.json differs from a regeneration at the commit it names")
    ap.add_argument("--repo", default=None, help="the repository (default: this source checkout)")
    ap.add_argument("--commit", default="HEAD", help="the commit to index (--write; default HEAD)")
    ns = ap.parse_args(argv)
    repo = _repo_root(ns.repo)
    target = repo / INDEX_PATH
    if ns.check:
        try:
            have = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"stale: {target} is unreadable ({e})")
            return 1
        want = build_index(repo, commit=str(have.get("commit") or "HEAD"))
        if _dumps(want) != _dumps(have):
            print(f"stale: {target} differs from the index regenerated at {str(have.get('commit'))[:12]}")
            return 1
        print(f"ok: {target} ({len(want['benchmarks'])} benchmarks at {want['commit'][:12]})")
        return 0
    index = build_index(repo, commit=ns.commit)
    if ns.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        old = target.read_text(encoding="utf-8") if target.is_file() else None
        new = _dumps(index)
        if old != new:
            target.write_text(new, encoding="utf-8")
        print(f"{'wrote' if old != new else 'unchanged'}: {target} ({len(index['benchmarks'])} benchmarks at {index['commit'][:12]})")
        return 0
    sys.stdout.write(_dumps(index))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
