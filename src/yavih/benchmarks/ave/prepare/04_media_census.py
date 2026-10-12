#!/usr/bin/env python
"""AVE step 4/5 (optional, read-only) — the folder's media census over the frozen manifest.

Per distinct video (3 232, val included), read the tar member through index.json and record container and
track geometry (fps, size, durations, sample rate, channels), the decoded audio sample count and first pts,
and the video packet pts. Nothing is cut or padded by the benchmark, so this is a fact check, not a protocol
step: ``n_errors`` must be 0 and ``short_clip_rows`` (rows with any track shorter than ``clip_seconds`` 10.0 s)
must equal ``manifest.upstream.curation.short_clips`` (74; make_manifest.py
``--short-clips``). It is NOT the site media-info table the reader uses: that is written by
``yavih bench verify ave --census``.

Note: AAC tracks decode to a few hundred samples more or fewer than the header duration × rate (priming /
padding; on AVE +344 for 2 091 videos, -680 for 857); ``a_samples`` here is the DECODED count.

Expected: 0 errors, 74 rows with a track < 10 s (68 audio 9.73-9.99 s, 6 video 9.96-9.99 s); about a minute
with 24 threads.

    python -I "$FOLDER/prepare/04_media_census.py" "$FOLDER/manifest.csv.gz" /path/to/ave/index.json census.jsonl [workers]

Needs PyAV + numpy. Writes only the jsonl you name and prints a JSON summary on stdout.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np

from yavih.media.tarindex import (
    open_tar_index,  # needs yavih importable (pip install -e / the wheel)
)


def census(tar, key: str) -> dict:
    data = tar.read(key)
    out: dict = {"key": key, "bytes": len(data)}
    with av.open(io.BytesIO(data)) as c:
        out["container_duration"] = c.duration / 1e6 if c.duration is not None else None
        vs = c.streams.video[0] if c.streams.video else None
        as_ = c.streams.audio[0] if c.streams.audio else None
        if vs is not None:
            out["fps"] = float(vs.average_rate) if vs.average_rate else None
            out["width"], out["height"] = vs.codec_context.width, vs.codec_context.height
            out["v_duration"] = float(vs.duration * vs.time_base) if vs.duration is not None else None
            out["v_frames_meta"] = int(vs.frames or 0)
        if as_ is not None:
            out["sample_rate"] = int(as_.codec_context.sample_rate or as_.rate)
            out["channels"] = int(as_.codec_context.channels or 1)
            out["a_duration_meta"] = float(as_.duration * as_.time_base) if as_.duration is not None else None
    # decoded video pts (packets only: pts of every video packet, cheaper than decoding)
    with av.open(io.BytesIO(data)) as c:
        if c.streams.video:
            vs = c.streams.video[0]
            pts = sorted(float(p.pts * p.time_base) for p in c.demux(vs) if p.pts is not None)
            out["v_n_packets"] = len(pts)
            out["v_first_pts"] = pts[0] if pts else None
            out["v_last_pts"] = pts[-1] if pts else None
    # decoded audio length (the reader's notion: samples from t = 0)
    if out.get("sample_rate"):
        with av.open(io.BytesIO(data)) as c:
            st = c.streams.audio[0]
            n = 0
            first = None
            for fr in c.decode(st):
                if first is None and fr.pts is not None:
                    first = float(fr.pts * fr.time_base)
                n += fr.samples
            out["a_samples"] = n
            out["a_first_pts"] = first
    return out


def main() -> int:
    man, index, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    with gzip.open(man, "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    keys = sorted({r["rel_media"] for r in rows if r["split"] in ("train", "test")} | {r["rel_media"] for r in rows})
    tar = open_tar_index(index)
    t = time.time()
    results = []
    with ThreadPoolExecutor(workers) as pool, open(out_path, "w") as fo:
        for res in pool.map(lambda k: _safe(tar, k), keys):
            results.append(res)
            fo.write(json.dumps(res) + "\n")
    dt = time.time() - t
    ok = [r for r in results if "error" not in r]
    by_key = {r["key"]: r for r in ok}
    def arr(k):
        return np.array([r[k] for r in ok if r.get(k) is not None], dtype=float)
    summary = {
        "videos": len(keys),
        "errors": [r for r in results if "error" in r][:10],
        "n_errors": sum("error" in r for r in results),
        "seconds": round(dt, 1),
        "no_audio": sum(1 for r in ok if not r.get("sample_rate")),
        "container_duration_lt_10": sum(1 for r in ok if (r.get("container_duration") or 0) < 10.0),
        "container_duration_min_max": [float(arr("container_duration").min()), float(arr("container_duration").max())],
        "v_duration_min_max": [float(arr("v_duration").min()), float(arr("v_duration").max())],
        "a_seconds_min_max": [float((arr("a_samples") / arr("sample_rate")).min()), float((arr("a_samples") / arr("sample_rate")).max())],
        "fps_values": sorted({round(r["fps"], 3) for r in ok if r.get("fps")}),
        "sample_rates": sorted({r["sample_rate"] for r in ok if r.get("sample_rate")}),
        "channels": sorted({r["channels"] for r in ok if r.get("channels")}),
        "width_min_max": [int(arr("width").min()), int(arr("width").max())],
        "height_min_max": [int(arr("height").min()), int(arr("height").max())],
        "v_first_pts_max": float(arr("v_first_pts").max()),
        "a_first_pts_min_max": [float(arr("a_first_pts").min()), float(arr("a_first_pts").max())],
        "bytes_total": int(arr("bytes").sum()),
        "short_clip_rows": sum(1 for r in rows if _short(by_key.get(r["rel_media"]))),
    }
    print(json.dumps(summary, indent=1))
    return 0


def _short(c: dict | None) -> bool:
    """Any stream shorter than the nominal 10 s clip (decoded audio samples / video stream duration)."""
    if not c:
        return False
    return (c.get("a_samples", 0) / c["sample_rate"] < 10.0 if c.get("sample_rate") else True) or (c.get("v_duration") or 0) < 10.0


def _safe(tar, k):
    try:
        return census(tar, k)
    except Exception as exc:  # noqa: BLE001
        return {"key": k, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    sys.exit(main())
