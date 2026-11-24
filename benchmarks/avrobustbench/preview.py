"""``avrobustbench`` README figure kind: one clip's frame under eight conditions (never hashed).

``avrobustbench_conditions`` lays out the frame at ``options.time_s`` of 8 rows (one clip: ``clean`` and seven of its
corrupted versions) as the core ``images`` page does (2 x 4 labelled tiles, ``images.html``), each captioned with
the row's condition; the title names the clip's class. Module scope: stdlib and yavih core.
"""

from __future__ import annotations

import html

from yavih.previews import Component, RenderContext


def _fill(ctx: RenderContext) -> dict:
    t = float(ctx.options["time_s"])
    tiles = []
    for i in range(len(ctx)):
        frame, _ = ctx.frames(i, [t])
        cond = ctx.extra(i).get("condition") or ctx.clip_id(i).rpartition("#")[2]
        tiles.append(f'<div class="tile"><img src="{ctx.data_uri(frame[0], 448)}"><div class="cap">{html.escape(cond)}</div></div>')
    labels = ", ".join(ctx.labels(0))
    return {"title": f"{labels}: one clip, clean and corrupted (severity 5)",
            "chips": [("a", "video frame, 224 px"), ("", f"frame at {t:g} s")],
            "body": '<div class="grid">' + "".join(tiles) + "</div>"}


COMPONENTS = [Component("avrobustbench_conditions", 1, (1600, 620), 8, "images.html", _fill, options={"time_s": 4.0})]
