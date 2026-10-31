"""``yavih features stage``: copy shards + sidecars + ``_meta.json`` of a ``features/``
tree to ``<scratch>/<unit_id|run>.<host>.<pid>/`` and return the staged path for
``eval run --scratch``.

Staging is a plain copy on purpose: the probe memmaps shards by sidecar offsets, so a
byte-identical copy on local disk is all it needs, and ``_meta.json`` travels along so the probe
reads the same eligibility verdict. Free space is pre-flighted (exit 3 "not ready"
when short); cleanup of the staged copy is the probe's job (its atexit/SIGTERM handlers).
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

from yavih.io.format import META_FILE, NOT_APPLICABLE_FILE
from yavih.io.validate import _SIDECAR_RE, _TAR_RE, host_tag, is_split_dir, split_dirs

__all__ = ["stage_features", "StageError", "staged_name", "STAGE_HEADROOM_BYTES"]

#: Free-space headroom kept on the scratch filesystem (local scratch is not RAM-charged, so a few GiB suffice).
STAGE_HEADROOM_BYTES: int = 1 * 1024**3
#: Files copied concurrently (network storage serves several streams faster than one).
STAGE_THREADS: int = 4


class StageError(OSError):
    """Scratch cannot hold the features (exit 3, not ready)."""

    exit_code = 3

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def staged_name(features_root: Path, unit_id: str | None) -> str:
    """``<unit_id|run>.<host>.<pid>`` where ``run`` is the run directory's name."""
    run = features_root.parent.name if features_root.name == "features" else features_root.name
    return f"{unit_id or run or 'run'}.{host_tag()}"


def _staged_files(split_dir: Path) -> list[Path]:
    out = []
    for child in sorted(split_dir.iterdir()):
        if child.name in (META_FILE, NOT_APPLICABLE_FILE) or _SIDECAR_RE.match(child.name) or _TAR_RE.match(child.name):
            out.append(child)
    return out


def stage_features(
    features_dir: str | os.PathLike,
    scratch: str | os.PathLike,
    *,
    unit_id: str | None = None,
    splits: list[str] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Copy ``<features_dir>/<split>/{shard-*.tar, shard-*.json, _meta.json, NOT_APPLICABLE.json}``
    to ``<scratch>/<staged_name>/features/<split>/``; returns the staged ``features/`` path.
    ``features_dir`` may also be a single split directory. ``.tmp`` files are never copied.
    ``should_stop`` is polled before each file (the probe's SIGTERM flag): a pending stop
    raises ``StageError("E_PREEMPTED")`` so the caller removes the partial copy instead of leaving it
    to the 36 h sweep.

    Up to :data:`STAGE_THREADS` files are copied at once, each to ``<name>.tmp`` then ``os.replace``d (a staged file
    is complete or absent). The first failure cancels the files not yet started and is raised once the copies in
    flight have finished, so nothing writes into the staged tree after this returns or raises."""
    src = Path(features_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"{src}: not a directory")
    scratch_root = Path(scratch)
    scratch_root.mkdir(parents=True, exist_ok=True)
    if is_split_dir(src):
        sources = [src]
        features_root = src.parent
    else:
        sources = split_dirs(src)
        features_root = src
        if not sources:
            raise FileNotFoundError(f"{src}: no split directories to stage")
    if splits is not None:
        sources = [s for s in sources if s.name in set(splits)]

    files = [(s, f) for s in sources for f in _staged_files(s)]
    need = sum(f.stat().st_size for _, f in files)
    free = shutil.disk_usage(scratch_root).free
    if need + STAGE_HEADROOM_BYTES > free:
        raise StageError(
            "E_SCRATCH_SPACE",
            f"staging needs {need / 1024**3:.2f} GiB (+{STAGE_HEADROOM_BYTES / 1024**3:.0f} GiB headroom) but {scratch_root} has {free / 1024**3:.2f} GiB free",
        )

    dest_root = scratch_root / staged_name(features_root, unit_id) / "features"
    failed = threading.Event()

    def copy_one(split_dir: Path, f: Path) -> None:
        if failed.is_set():
            return  # another file failed: start nothing new
        if should_stop is not None and should_stop():
            raise StageError("E_PREEMPTED", f"stop requested while staging {f.name}; partial copy under {dest_root.parent}")
        dest_dir = dest_root / split_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest_dir / (f.name + ".tmp")
        shutil.copyfile(f, tmp)
        if tmp.stat().st_size != f.stat().st_size:
            tmp.unlink(missing_ok=True)
            raise StageError("E_SCRATCH_SPACE", f"short copy of {f} to {dest_dir}")
        os.replace(tmp, dest_dir / f.name)

    if not files:
        return dest_root
    with ThreadPoolExecutor(max_workers=min(STAGE_THREADS, len(files)), thread_name_prefix="yavih-stage") as pool:
        futures = [pool.submit(copy_one, split_dir, f) for split_dir, f in files]
        try:
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            abort = any(fut.exception() is not None for fut in done)
        except BaseException:
            abort = True
            raise
        finally:
            if abort:
                failed.set()
                for fut in futures:
                    fut.cancel()
    # leaving the pool waited for every copy in flight: raise the first failure in file order
    for fut in futures:
        if not fut.cancelled() and fut.exception() is not None:
            raise fut.exception()  # type: ignore[misc]
    return dest_root
