"""``probe_ckpt.pt`` (the deadline checkpoint): ``{state_dict, optimizer, scheduler, epoch, rng, best_state,
best_val, patience_left}`` + the hashes it was written under; resume only if the hashes match.


The file is written atomically (tmp + ``os.replace``) so a SIGTERM during the write can never leave a
truncated checkpoint that a resume would then trust."""

from __future__ import annotations

import os
from typing import Any

import torch

__all__ = ["save_ckpt", "load_ckpt", "rng_state", "set_rng_state", "CKPT_FILE", "CKPT_SCHEMA"]

CKPT_FILE = "probe_ckpt.pt"
CKPT_SCHEMA = "yavih-probe-ckpt-1"


def rng_state() -> dict[str, Any]:
    """CPU + every CUDA generator state (the whole ``fork_rng`` scope), so a resumed epoch draws the
    same permutation and the same dropout-free noise as the uninterrupted run would have."""
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_ckpt(path: str, state: dict, *, hashes: dict) -> None:
    """Write ``state`` (torch-serialisable) with the ``hashes`` it belongs to."""
    payload = {"schema": CKPT_SCHEMA, "hashes": dict(hashes), "state": state}
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str, *, hashes: dict) -> dict | None:
    """``None`` when absent or when the recorded hashes differ (never resume a different protocol)."""
    if not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - a corrupt checkpoint is "no checkpoint", the fit restarts
        return None
    if not isinstance(payload, dict) or payload.get("schema") != CKPT_SCHEMA:
        return None
    if payload.get("hashes") != dict(hashes):
        return None
    return payload["state"]
