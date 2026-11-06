"""``perception_test_action_loc``: the folder's own temporal-detection scorer and its metrics.

The scorer (head, loss, targets, decoding, the ActivityNet detection AP, the floor, the conformance population) is
``detection.py``, next to this file and byte-identical in every folder that detects timed events (a candidate for
the core). This file names the folder's event encoding (``class:start_us:end_us``, the
Perception Test annotation in integer microseconds) and its metric names. The AP is a transcription of the scorer the
Perception Test localisation baseline ships (``libs/utils/metrics.py`` of
github.com/ptchallenge-workshop/actionformer_release_PT @ 10684a4ae965a4a389c425e2fe8d8729ad6d6281), which is not
pip-installable; README "## In yavih" lists the one documented difference. Module scope: stdlib, numpy, yavih core.
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
EVENTS_FORMAT = "class:start_us:end_us"
PROTOCOL_KEYS = frozenset({"tiou_thresholds"})
SCORER = detection.make_scorer("perception_test_action_loc", module=__name__, events_format=EVENTS_FORMAT)
METRICS = detection.make_metrics("perception_test_action_loc", mean="tmap", per_tiou="tmap_per_tiou", floor="tmap_floor", n_events="n_events")

__all__ = ["SCORER", "METRICS", "PROTOCOL_KEYS", "EVENTS_FORMAT", "detection"]
