"""``llp_avvp``: the event intervals the README figures draw (``python -m yavih.previews llp_avvp``).

A scored (val / test) row's annotated audio and visual events as ``(start_s, end_s, "<class> · heard")`` and
``"<class> · seen"`` bands; a class heard and seen over the same seconds is one audio-visual event and shows as both.
Never hashed. Module scope: stdlib and yavih core.
"""

from __future__ import annotations

from typing import Any


def intervals(bench: Any, split: str, extra: dict) -> list[tuple[float, float, str]]:
    """``[(start_s, end_s, label)]`` of one row from its ``audio_events`` / ``visual_events`` cells
    (``onset:offset:class``); events with ``offset <= onset`` mark nothing and are skipped."""
    names = [c["name"].replace("_", " ") for c in bench.classes()]
    out: list[tuple[float, float, str]] = []
    for column, how in (("audio_events", "heard"), ("visual_events", "seen")):
        cell = (extra or {}).get(column) or ""
        for tok in filter(None, str(cell).split(";")):
            a, b, c = (int(x) for x in tok.split(":"))
            if b > a:
                out.append((float(a), float(b), f"{names[c]} · {how}"))
    return out
