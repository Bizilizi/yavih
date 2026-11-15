"""``vggsounder_retrieval``: the folder's own training-free retrieval scorer and its metrics.

Instance-level ranks, R@k and the median rank are ``retrieval.py`` (byte-identical to the copy in
``vggsound_retrieval``); the class-level measures (``cls_map_*``, ``cls_r@k_*``, ``cls_n_no_relevant``), copied from
DCMH's ``calc_map`` and Roth et al.'s ``c_recall@k``, and the driver that computes both are ``class_level.py``. A
class-level relevant clip shares at least one audible-and-visible class (``readout_params.classes_column``) with the
query; the query's own clip is not in its class-level gallery. No subsets (``readout_params.subsets: {}``).
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
class_level = _sibling("class_level.py")
SCORER = class_level.make_scorer(Path(__file__).resolve().parent.name, module=__name__, retrieval=retrieval)
METRICS = {**retrieval.make_metrics(SCORER.name), **class_level.make_metrics(SCORER.name)}

__all__ = ["SCORER", "METRICS", "retrieval", "class_level"]
