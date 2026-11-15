"""``av_syncbench``: the folder's own synthetic stand-in and test bridges for ``yavih bench trial --synthetic``.

The pairs, the stand-in folder and the bridges are ``sync_synth.py`` (next to this file): the stand-in keeps this
folder's name, scorer, metrics, protocol, fields and output contract over 3.2 s synthetic videos whose frames and AAC
track follow one on/off pattern, each paired with a perturbed wav of every (task, level) group; the oracle bridge
embeds each stream's segments on its own, the random bridge embeds noise.
Module scope: stdlib, numpy, yaml, yavih core; writing the media needs PyAV.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


def _sibling(name: str):
    """A file next to this one, imported by path (keyed on its content, like yavih loads eval.py)."""
    path = Path(__file__).resolve().with_name(name)
    data = path.read_bytes()
    mod_name = f"yavih_folder_{path.parent.name}_{path.stem}_{hashlib.sha256(data).hexdigest()[:12]}"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
    return sys.modules[mod_name]


synth = _sibling("sync_synth.py")
BRIDGES = {"oracle": "OracleSyncBridge", "random": "RandomSyncBridge"}


def make_stand_in(work, real, *, n_eval, n_train):
    """The stand-in folder (``sync_synth.make_stand_in``) over ``max(n_eval, 40)`` synthetic pairs."""
    return synth.make_stand_in(work, real, n_eval=n_eval, n_train=n_train)


def write_bridge(path, kind, *, D, bench_folder):
    """The stand-in's ``oracle`` / ``random`` bridge file; ``D`` is ignored (the oracle's space has 10 dimensions)."""
    if kind not in BRIDGES:
        raise ValueError(f"write_bridge: kind must be one of {sorted(BRIDGES)}, got {kind!r}")
    return synth.bridge_source(path, BRIDGES[kind], kind=kind, synth=Path(bench_folder) / "sync_synth.py")
