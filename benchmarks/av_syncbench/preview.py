"""``av_syncbench``: the README figure kind of the folder's own (``python -m yavih.previews av_syncbench``; never hashed).

``av_syncbench_pair``: one pair on one time axis: a strip of frames of the original video, then the waveform and mel
spectrogram of its own soundtrack and of the perturbed soundtrack, so the shift, speed change, jitter or timbre edit
is visible against the pictures. At most ``seconds`` (5 s) of media. Module scope: stdlib, numpy, yavih core.
"""

from __future__ import annotations

import numpy as np

from yavih.previews import MAX_SECONDS, Component, _chip, _cut, _esc, _fmt_t, _frame_tiles, _ticks

WIDTH = 1500


def _describe(extra: dict) -> str:
    task, level, direction = extra.get("task"), extra.get("level"), extra.get("direction")
    if task == "offset":
        return f"audio {'delayed' if direction == 'delay' else 'advanced'} by {level}"
    if task == "speed":
        return f"audio played at {level}"
    if task == "jitter":
        return f"audio locally jittered ({level})"
    return f"{str(task).replace('_', ' ')} edited to {level}"


def _fill(ctx) -> dict:
    from yavih.media.decode import audio

    secs = min(float(ctx.options["seconds"]), MAX_SECONDS)
    dur = min(float(ctx.duration(0) or secs), secs)
    n = int(ctx.options["n"])
    strip = _frame_tiles(ctx, 0, [round(dur * (k + 0.5) / n, 3) for k in range(n)], max_side=400)
    y, rate = ctx.audio(0, seconds=dur)
    e, erate = audio(ctx.sample(0).streams["edited_audio"])
    e = _cut(e, erate, 0.0, dur)
    pad = lambda x, r: np.concatenate([x, np.zeros(max(0, int(round(dur * r)) - len(x)), dtype=np.float32)])  # noqa: E731
    y, e = pad(y, rate), pad(e, erate)
    what = _describe(ctx.extra(0))
    rows = []
    for label, sig, r, cls in (("original soundtrack", y, rate, "a"), (f"perturbed: {what}", e, erate, "b")):
        rows.append(f'<div class="sig"><span class="chip {cls}">{_esc(label)}</span>{ctx.wave_svg(sig, WIDTH, 56)}'
                    f'<img class="spec" src="{ctx.mel_uri(sig, r, columns=WIDTH)}"></div>')
    return {"title": f"which soundtrack belongs? {what}",
            "chips": [("a", "video + audio"), _chip(str(ctx.extra(0).get("category", "")).replace("_", " ")), _chip(_fmt_t(dur))],
            "body": f'<div class="stack"><div class="strip">{strip}</div>{"".join(rows)}</div>{_ticks(dur)}'}


CSS = ".sig .spec{height:84px!important}.sig .chip{align-self:flex-start}"
COMPONENTS = [Component("av_syncbench_pair", 1, (1600, 780), 1, "av.html", _fill, css=CSS, options={"n": 6, "seconds": MAX_SECONDS})]
