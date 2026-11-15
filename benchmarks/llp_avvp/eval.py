"""``llp_avvp``: the folder's own scorer and metrics (weakly-supervised audio-visual video parsing on LLP).

The scorer is ``avvp.py`` (next to this file): the LLP authors' evaluation (``avvp_eval_metrics.py``, their
``utils/eval_metrics.py`` copied byte for byte, called per video as their ``main_avvp.py::eval`` does) and their
readout (``han.py``, the HAN / MMIL network of ``nets/net_audiovisual.py``, trained with ``main_avvp.py``'s recipe
on the bridge's per-second pooled features). Source: https://github.com/YapengTian/AVVP-ECCV20 at
``fde56118f2c2e7c19188dfefa2569a37b70054fe`` (GPL-3.0); the authors ship no pip package, so nothing is imported
from them. Metrics, in percent as the authors print them: segment-level and event-level F-scores of the audio,
visual and audio-visual events (``seg_a`` … ``evt_av``), Type@AV (``seg_type_av``, the headline and the authors'
model-selection criterion; ``evt_type_av``) and Event@AV (``seg_event_av``, ``evt_event_av``).
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


avvp = _sibling("avvp.py")
PROTOCOL_KEYS = frozenset({"num_segments", "segment_seconds"})
SCORER = avvp.make_scorer("llp_avvp", module=__name__)
METRICS = avvp.make_metrics("llp_avvp")

__all__ = ["SCORER", "METRICS", "PROTOCOL_KEYS", "avvp"]
