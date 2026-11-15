"""``av_syncbench``: the folder's own training-free pairwise scorer and its metrics.

The scorer (the mean diagonal cosine of per-segment embeddings, the strict original-beats-perturbed decision, the
task and level accuracies, the conformance population) is ``pairwise.py``, next to this file, copied and transcribed
from the AV-SyncBench authors' evaluation toolkit (https://github.com/fgt7t6g/AV-SyncBench @ 82e1353; provenance in
its docstring). Headline ``acc_mean``: the mean of the offset, jitter, speed and timbre accuracies; chance 0.5.
Protocol keys: ``segment_seconds`` (the segment grid, handed to the bridge as a fact) and ``max_segments`` (at most
this many common segments are scored, chosen uniformly). Module scope: stdlib, numpy, yavih core.
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


pairwise = _sibling("pairwise.py")
SCORER = pairwise.make_scorer(Path(__file__).resolve().parent.name, module=__name__)
METRICS = pairwise.make_metrics(SCORER.name)
PROTOCOL_KEYS = pairwise.PROTOCOL_KEYS

__all__ = ["SCORER", "METRICS", "PROTOCOL_KEYS", "pairwise"]
