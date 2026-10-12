"""vggsounder — step 4 (optional, measurement only): header facts of every frozen row's media.

Reads the container headers of the 32 008 mp4 and 32 008 wav files the frozen manifest touches —
nothing is decoded — and prints a summary of the facts the input contract relies on
(`MediaInfo` keys): durations, frame rates, frame counts, frame sizes, sample rates, sample counts, channel
counts. It is how the numbers in setup.md ("## Input contract", "## Gotchas") and `curation.short_clips` were measured and how
`input_contract.info` was checked to be non-null on every row.

It is NOT the site census: `yavih bench verify vggsounder --census` writes the media-info table that
`RawStream.info` is filled from. This script exists so the folder's claims can be re-measured
without core, e.g. after a re-download.

Usage (needs PyAV + soundfile, i.e. `yavih[media]`; ~25 min over NFS with 32 threads):

    python "$(yavih bench path vggsounder)/prepare/04_media_facts.py" --root "$(yavih config get vggsounder.media_root)" \
        --out /tmp/vggsounder_media_facts.jsonl.gz > /tmp/vggsounder_media_facts.summary.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLIP_SECONDS = 10.0


def video_facts(path: str) -> dict:
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        if s.duration is not None and s.time_base is not None:
            dur = float(s.duration * s.time_base)
        elif c.duration is not None:
            dur = c.duration / 1e6
        else:
            dur = None
        return {
            "duration_s": dur,
            "fps": float(s.average_rate) if s.average_rate else None,
            "n_frames": int(s.frames) or None,
            "width": int(s.codec_context.width),
            "height": int(s.codec_context.height),
            "rotation": (s.metadata or {}).get("rotate"),
            "container": "mp4",
            "codec": s.codec_context.name,
            "audio_tracks": len(c.streams.audio),
            "bytes": Path(path).stat().st_size,
        }


def audio_facts(path: str) -> dict:
    import soundfile as sf

    i = sf.info(path)
    return {
        "duration_s": i.frames / i.samplerate,
        "sample_rate": int(i.samplerate),
        "n_samples": int(i.frames),
        "channels": int(i.channels),
        "container": "wav",
        "subtype": i.subtype,
        "bytes": Path(path).stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, required=True, help="VGGSound root (video/, audio/)")
    p.add_argument("--out", type=Path, default=None, help="optional per-row jsonl.gz")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="first N rows only (smoke)")
    a = p.parse_args(argv)

    with gzip.open(HERE / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if a.limit:
        rows = rows[: a.limit]

    def one(r: dict) -> dict:
        out = {"clip_id": r["clip_id"], "split": r["split"]}
        for stream, col, fn in (
            ("video", "rel_video", video_facts),
            ("audio", "rel_audio", audio_facts),
        ):
            try:
                out[stream] = fn(str(a.root / r[col]))
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                out[stream] = {"error": f"{type(exc).__name__}: {exc}"}
        return out

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        facts = list(ex.map(one, rows))
    wall = time.time() - t0

    if a.out:
        with gzip.open(a.out, "wt", encoding="utf-8") as fh:
            for f in facts:
                fh.write(json.dumps(f, sort_keys=True) + "\n")

    summ: dict = {"rows": len(facts), "wall_s": round(wall, 1)}
    for split in sorted({f["split"] for f in facts}):
        fs = [f for f in facts if f["split"] == split]
        s: dict = {"rows": len(fs)}
        for stream in ("video", "audio"):
            ok = [f[stream] for f in fs if "error" not in f[stream]]
            s[f"{stream}_errors"] = [
                (f["clip_id"], f[stream]["error"]) for f in fs if "error" in f[stream]
            ]
            durs = sorted(x["duration_s"] for x in ok if x["duration_s"] is not None)
            s[f"{stream}_duration"] = {
                "null": sum(x["duration_s"] is None for x in ok),
                "min": durs[0] if durs else None,
                "median": durs[len(durs) // 2] if durs else None,
                "max": durs[-1] if durs else None,
                "lt_10s": sum(d < CLIP_SECONDS for d in durs),
                "lt_9_5s": sum(d < 9.5 for d in durs),
                "lt_8s": sum(d < 8.0 for d in durs),
            }
            for key in (
                "fps",
                "n_frames",
                "width",
                "height",
                "sample_rate",
                "n_samples",
                "channels",
            ):
                if any(key in x for x in ok):
                    s.setdefault(f"{stream}_null", {})[key] = sum(x.get(key) is None for x in ok)
        vids = [f["video"] for f in fs if "error" not in f["video"]]
        auds = [f["audio"] for f in fs if "error" not in f["audio"]]
        s["video_sizes_top"] = collections.Counter(
            f"{v['height']}x{v['width']}" for v in vids
        ).most_common(12)
        s["bytes"] = {"video": sum(v["bytes"] for v in vids), "audio": sum(x["bytes"] for x in auds)}
        s["video_portrait"] = sum(v["height"] > v["width"] for v in vids)
        s["video_fps_top"] = collections.Counter(
            round(v["fps"], 3) for v in vids if v["fps"]
        ).most_common(8)
        s["video_rotation_tagged"] = sum(bool(v.get("rotation")) for v in vids)
        s["video_codecs"] = collections.Counter(v["codec"] for v in vids).most_common()
        s["video_audio_tracks"] = collections.Counter(v["audio_tracks"] for v in vids).most_common()
        s["audio_rates"] = collections.Counter(x["sample_rate"] for x in auds).most_common()
        s["audio_channels"] = collections.Counter(x["channels"] for x in auds).most_common()
        s["audio_subtypes"] = collections.Counter(x["subtype"] for x in auds).most_common()
        s["either_lt_10s"] = sum(
            1
            for f in fs
            if any(
                "error" not in f[st]
                and f[st]["duration_s"] is not None
                and f[st]["duration_s"] < CLIP_SECONDS
                for st in ("video", "audio")
            )
        )
        summ[split] = s
    json.dump(summ, sys.stdout, indent=1, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
