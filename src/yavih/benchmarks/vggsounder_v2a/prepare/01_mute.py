#!/usr/bin/env python
"""vggsounder_v2a — step 1: write the muted copy of every frozen video (``input_contract.materialize``; setup.md).

The clip's soundtrack is the label (the reference the generations are scored against), so it cannot reach the
bridge: for every manifest row this script runs the pinned recipe of ``benchmark.yaml`` ``input_contract.materialize``
(kind ``strip_audio``: ``ffmpeg -i <media_root>/<rel_video> -an -c:v copy -map_metadata -1``) and writes
``<derived_root>/<rel_media>``. ``-c:v copy`` copies the stored video stream: no frame is decoded or re-encoded.
Every output is verified with ffprobe (one video stream, no audio stream, the source's first decoded frame time and
its decoded frames, or one more at the end: an mp4 edit list may let the copy show a trailing frame the source hid,
after the scored seconds) before it is renamed into place; an existing verified output is kept (resumable).

Refusals: an ffmpeg build other than ``tool.version_pin`` (exit 2); a missing source file or a failed verification
(the row is listed, exit 1). Usage (``FOLDER=$(yavih bench path vggsounder_v2a)``)::

    python "$FOLDER/prepare/01_mute.py" --rows 32        # a validation run on the first 32 rows
    python "$FOLDER/prepare/01_mute.py" --workers 16     # every row

Roots: ``--source-root`` (default ``yavih config get vggsounder_v2a.media_root``) and ``--out-root`` (default
``yavih config get vggsounder_v2a.derived_root``). Needs ffmpeg / ffprobe on PATH; stdlib + pyyaml + yavih.config.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

FOLDER = Path(__file__).resolve().parents[1]


def _spec() -> dict:
    return yaml.safe_load((FOLDER / "benchmark.yaml").read_text(encoding="utf-8"))


def _ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return out.split()[2]


def _probe(path: Path) -> dict:
    """Stream counts and the video's DECODED frame timestamps (``best_effort_timestamp_time``): the container's
    ``nb_frames`` header and packet counts are unreliable in the VGGSound mp4s (edit lists hide packets)."""
    streams = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)],
                                        capture_output=True, text=True, check=True).stdout).get("streams", [])
    ts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time",
                         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.split()
    times = [float(t.strip(",")) for t in ts if t.strip(",") not in ("", "N/A")]
    return {"video": sum(s.get("codec_type") == "video" for s in streams), "audio": sum(s.get("codec_type") == "audio" for s in streams),
            "frames": len(times), "first": times[0] if times else None}


def _verified(src: dict, out: dict) -> bool:
    """One video stream, no audio stream, the source's first frame time, and the source's decoded frames (an mp4 edit
    list may let the copy show one frame more at the end, after the scored seconds)."""
    return (out["audio"] == 0 and out["video"] == 1 and out["first"] is not None and src["first"] is not None
            and abs(out["first"] - src["first"]) < 1e-6 and src["frames"] <= out["frames"] <= src["frames"] + 1)


def _mute(src: Path, dst: Path, args: list[str]) -> str | None:
    """None when ``dst`` holds a verified muted copy of ``src``, else the reason."""
    if not src.is_file():
        return f"source missing: {src}"
    want = _probe(src)
    if dst.is_file() and _verified(want, _probe(dst)):
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp.mp4")
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), *args, str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"ffmpeg failed: {r.stderr.strip()[:200]}"
    got = _probe(tmp)
    if not _verified(want, got):
        tmp.unlink(missing_ok=True)
        return f"verification failed: source {want}, output {got}"
    os.replace(tmp, dst)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=None, help="only the first N manifest rows (a validation run)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    spec = _spec()
    mat = spec["input_contract"]["materialize"]
    pin = mat["tool"]["version_pin"]
    have = _ffmpeg_version()
    if have != pin:
        print(f"ffmpeg {have} != the pinned {pin} (benchmark.yaml input_contract.materialize.tool.version_pin)", file=sys.stderr)
        sys.exit(2)
    from yavih.config import folder_root

    src_root = Path(a.source_root or folder_root(FOLDER, "media_root"))
    out_root = Path(a.out_root or folder_root(FOLDER, "derived_root"))
    rows = list(csv.DictReader(io.StringIO(gzip.decompress((FOLDER / "manifest.csv.gz").read_bytes()).decode("utf-8"))))
    if a.rows:
        rows = rows[: a.rows]
    col = mat["source_columns"]["video"]
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        results = list(pool.map(lambda r: (r["clip_id"], _mute(src_root / r[col], out_root / r["rel_media"], list(mat["tool"]["args"]))), rows))
    failed = [(c, why) for c, why in results if why is not None]
    print(json.dumps({"rows": len(rows), "ok": len(rows) - len(failed), "failed": len(failed), "first_failures": failed[:5], "out_root": str(out_root)}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
