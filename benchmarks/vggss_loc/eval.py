"""``vggss_loc``: the folder's own training-free localisation scorer and its metrics.

The scorer (heatmaps from the shared-space tokens, the VGG-SS cIoU pipeline, the floor, the two controls, the
conformance population) is ``loc.py``, next to this file. Metrics: ``ciou@0.5`` (the headline, the VGG-SS
evaluator's cIoU@0.5), ``auc``, ``ciou_mean``, ``n_empty_gt``; floor ``ciou_floor`` (the leave-one-out mean
ground-truth mask); controls ``ciou_shuffled`` and ``audio_blind`` (the headline must beat the shuffled-audio control
by 3 McNemar σ, else the report voids the result). Module scope: stdlib, numpy, yavih core.
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


loc = _sibling("loc.py")
SCORER = loc.make_scorer(Path(__file__).resolve().parent.name, module=__name__)
METRICS = loc.make_metrics(SCORER.name)

__all__ = ["SCORER", "METRICS", "loc"]
