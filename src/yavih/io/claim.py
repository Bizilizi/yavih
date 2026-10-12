"""One writer per shard: ``shard-{i:06d}.claim`` lease files.

Two writers can land on one shard: a requeue after a node failure while the old task still runs, the stale-running
heuristic resubmitting a slow unit, ``--index`` into a pending file rebuilt while an array was queued, a pod
restart. Without a claim, one could rename the other's partial tar into place and write its sidecar. A writer:

1. creates the claim with ``O_EXCL`` semantics (a hard link of a complete file, else ``O_CREAT|O_EXCL``), holding
   ``{token, host, pid, started, lease_s}``; a heartbeat thread refreshes its mtime every ``lease_s / 4``;
2. treats an existing claim as live unless its writer is provably gone: same host and the pid is dead, or the
   heartbeat is older than ``lease_s`` (default 900 s, ``YAVIH_CLAIM_LEASE_S``). A live claim → ``ClaimHeld``
   (exit 3: the shard is being written elsewhere; resubmit later). A stale one is taken over by renaming it aside
   (atomic; a racing taker that renamed a fresh claim puts it back);
3. writes its tar to a per-writer tmp (``<tar>.<host>.<pid>.<thread>.tmp``) and checks it still holds the claim
   before the tar rename and before the sidecar (``ClaimLost``, exit 4, when another writer took over meanwhile);
4. removes the claim when the shard is done or abandoned.

Only files, renames and mtimes: no ``fcntl``/``flock``, so it works on NFS and Lustre like the rest of the
format. A heartbeat from a host with a clock skewed by more than ``lease_s`` defeats staleness detection; keep
clocks within minutes (NTP) or raise the lease. stdlib only.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

__all__ = ["Claim", "ClaimHeld", "ClaimLost", "acquire_claim", "claim_name", "read_claim", "claim_is_stale",
           "DEFAULT_LEASE_S", "CLAIM_SUFFIX"]

CLAIM_SUFFIX = ".claim"
DEFAULT_LEASE_S = 900.0


class ClaimHeld(RuntimeError):
    """Another live writer holds the shard: not ready (exit 3), the unit is resubmitted later."""

    code = "E_SHARD_CLAIMED"
    exit_code = 3


class ClaimLost(RuntimeError):
    """The claim was taken over while this writer ran (it looked stale): nothing is published (exit 4, resumable)."""

    code = "E_CLAIM_LOST"
    exit_code = 4


def claim_name(index: int) -> str:
    return f"shard-{int(index):06d}{CLAIM_SUFFIX}"


def _lease_s(lease_s: float | None) -> float:
    if lease_s is not None:
        return float(lease_s)
    raw = os.environ.get("YAVIH_CLAIM_LEASE_S")
    try:
        return float(raw) if raw else DEFAULT_LEASE_S
    except ValueError:
        return DEFAULT_LEASE_S


def read_claim(path: str | os.PathLike) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def claim_is_stale(path: str | os.PathLike, *, lease_s: float | None = None, now: float | None = None,
                   host: str | None = None) -> bool:
    """A claim whose writer is provably gone: same host and dead pid, or no heartbeat for ``lease_s``. An
    unreadable claim (torn by a crash between create and write) counts by its mtime alone."""
    now = time.time() if now is None else now
    host = socket.gethostname() if host is None else host
    try:
        mtime = os.stat(path).st_mtime
    except FileNotFoundError:
        return True
    rec = read_claim(path) or {}
    pid = rec.get("pid")
    if rec.get("host") == host and isinstance(pid, int) and pid > 0 and pid != os.getpid() and not _pid_alive(pid):
        return True
    lease = rec.get("lease_s") if isinstance(rec.get("lease_s"), (int, float)) else _lease_s(lease_s)
    return now - mtime > float(lease)


class Claim:
    """A held shard claim; ``verify()`` before publishing, ``release()`` when done (also on abort)."""

    def __init__(self, path: Path, token: str, lease_s: float) -> None:
        self.path = path
        self.token = token
        self.lease_s = lease_s
        self.took_over: dict | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name=f"yavih-claim-{path.name}", daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        period = max(1.0, self.lease_s / 4.0)
        while not self._stop.wait(period):
            if not self.held():
                return
            try:
                os.utime(self.path)
            except OSError:
                return

    def held(self) -> bool:
        rec = read_claim(self.path)
        return rec is not None and rec.get("token") == self.token

    def verify(self) -> None:
        """Raise ``ClaimLost`` when another writer owns the claim now."""
        if not self.held():
            rec = read_claim(self.path) or {}
            raise ClaimLost(f"{self.path.name}: the claim was taken over by host {rec.get('host')} pid {rec.get('pid')} while this "
                            "writer ran (its heartbeat looked stale); nothing published, resubmit")

    def release(self) -> None:
        self._stop.set()
        if self.held():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def to_json(self) -> dict[str, Any]:
        return {"file": self.path.name, "token": self.token, "lease_s": self.lease_s, "took_over": self.took_over}


def _create(path: Path, rec: dict) -> bool:
    """Create ``path`` holding ``rec`` iff absent; ``False`` when it exists."""
    tmp = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError:  # no hard links on this filesystem: O_EXCL on the final name
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, sort_keys=True)
        return True
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def acquire_claim(split_dir: str | os.PathLike, index: int, *, lease_s: float | None = None) -> Claim:
    """Claim shard ``index`` of ``split_dir`` (module docstring). Raises ``ClaimHeld`` when a live writer has it."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / claim_name(index)
    lease = _lease_s(lease_s)
    token = uuid.uuid4().hex
    rec = {"token": token, "host": socket.gethostname(), "pid": os.getpid(), "started": time.time(), "lease_s": lease}
    took_over: dict | None = None
    for _ in range(3):
        if _create(path, rec):
            claim = Claim(path, token, lease)
            claim.took_over = took_over
            return claim
        held = read_claim(path) or {}
        if not claim_is_stale(path, lease_s=lease):
            raise ClaimHeld(
                f"shard {index} of {split_dir} is being written by host {held.get('host')} pid {held.get('pid')} (claim "
                f"{path.name}, heartbeat within {lease:.0f} s): another task runs this unit; nothing done, resubmit later"
            )
        # take the stale claim over: rename it aside (atomic, one renamer wins), then check it was the stale one
        aside = path.with_name(f"{path.name}.stale.{token[:8]}")
        try:
            os.rename(path, aside)
        except FileNotFoundError:
            continue  # someone else moved it first: retry the create
        got = read_claim(aside) or {}
        if held.get("token") is not None and got.get("token") != held.get("token"):
            # a racing taker created a fresh claim between our read and our rename: put it back and yield
            try:
                os.link(aside, path)
            except OSError:
                pass
            try:
                aside.unlink()
            except FileNotFoundError:
                pass
            raise ClaimHeld(f"shard {index} of {split_dir}: another writer took the stale claim over first; resubmit later")
        took_over = {"host": got.get("host"), "pid": got.get("pid"), "started": got.get("started")}
        try:
            aside.unlink()
        except FileNotFoundError:
            pass
    raise ClaimHeld(f"shard {index} of {split_dir}: could not claim it after 3 attempts (concurrent writers); resubmit later")
