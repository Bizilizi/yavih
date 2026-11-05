"""``unav100``: the folder's own temporal-detection scorer and its metrics.

The scorer (head, loss, targets, decoding, the ActivityNet detection AP, the floor, the conformance population) is
``detection.py``, next to this file and byte-identical in every folder that detects timed events (a candidate for
the core). This file names the folder's event encoding (``start_s:end_s:class``, the UnAV-100
annotation in seconds) and its metric names: ``map_avg`` (mean AP over tIoU 0.1:0.1:0.9, the UnAV-100 authors'
average), ``map@0.5``, ``map_tiou`` (per threshold), ``n_events`` and the floor ``map_avg_floor``. The UnAV-100
authors' evaluation (ActivityNet's ``eval_detection.py``) is not pip-installable; this is a transcription of it.
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


detection = _sibling("detection.py")
EVENTS_FORMAT = "start_s:end_s:class"
PROTOCOL_KEYS = frozenset({"tiou_thresholds"})
SCORER = detection.make_scorer("unav100", module=__name__, events_format=EVENTS_FORMAT)
METRICS = detection.make_metrics("unav100", mean="map_avg", per_tiou="map_tiou", at={"map@0.5": 4}, floor="map_avg_floor",
                                 n_events="n_events")

__all__ = ["SCORER", "METRICS", "PROTOCOL_KEYS", "EVENTS_FORMAT", "detection"]
