"""Probe heads and training. ``[probe]`` extra: this package MAY import
torch at module scope, except ``config.py`` which stays torch-free so ``probe_hash``
parity holds in every env.

Only ``config`` (and numpy-only ``floors``) are imported eagerly; the torch-backed names below are
resolved lazily (PEP 562) so ``from yavih.probes import ProbeConfig`` works in the model env without
torch, while ``yavih.probes.train_head`` still reads naturally in the probe env."""

from __future__ import annotations

import importlib
from typing import Any

from yavih.probes.config import (
    HEAD_FAMILIES,
    LOSSES,
    PROBE_SCHEMA,
    ProbeConfig,
    parse_layout,
    testing_enabled,
)
from yavih.probes.floors import floor_outputs

_LAZY: dict[str, str] = {
    "build_attentive_probe": "yavih.probes.attentive",
    "build_dense_query_head": "yavih.probes.dense",
    "build_dense_av_head": "yavih.probes.dense",
    "sincos_embed": "yavih.probes.dense",
    "per_sample_loss": "yavih.probes.losses",
    "FitResult": "yavih.probes.trainer",
    "ProbeDeadline": "yavih.probes.trainer",
    "ProbeInterrupted": "yavih.probes.trainer",
    "build_head_for": "yavih.probes.trainer",
    "micro_batch_size": "yavih.probes.trainer",
    "predict": "yavih.probes.trainer",
    "train_head": "yavih.probes.trainer",
    "save_ckpt": "yavih.probes.ckpt",
    "load_ckpt": "yavih.probes.ckpt",
}

__all__ = [
    "ProbeConfig",
    "PROBE_SCHEMA",
    "HEAD_FAMILIES",
    "LOSSES",
    "parse_layout",
    "testing_enabled",
    "floor_outputs",
    *sorted(_LAZY),
]


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'yavih.probes' has no attribute {name!r}")
    return getattr(importlib.import_module(mod), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
