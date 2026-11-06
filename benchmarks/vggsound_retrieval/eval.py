"""``vggsound_retrieval``: the folder's own training-free retrieval scorer and its metrics.

The scorer and every metric (ranks, R@k, the median rank, the subset metrics, the conformance population) are
``retrieval.py``, next to this file. This file names the folder's subset: ``*_cavmae`` are the 1 545 rows flagged
``cavmae_sync_5pc`` (5 per class, the retrieval datafile of CAV-MAE Sync @ 0add2e7, the subset the CAV-MAE and
CAV-MAE Sync papers report on) ranked among themselves (``readout_params.subsets: {cavmae: cavmae_sync_5pc}``). Same
definitions as the full-set metrics; Wilson intervals over the subset's queries for the R@k.
Module scope: stdlib, numpy, yavih core.
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


retrieval = _sibling("retrieval.py")
SUBSET = "cavmae"
SCORER = retrieval.make_scorer(Path(__file__).resolve().parent.name, module=__name__)
METRICS = retrieval.make_metrics(SCORER.name, subsets=(SUBSET,))

__all__ = ["SCORER", "METRICS", "SUBSET", "retrieval"]
