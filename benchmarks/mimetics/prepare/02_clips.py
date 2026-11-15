#!/usr/bin/env python
"""mimetics step 2 — download the 713 Mimetics clips from YouTube and trim them, as the release README says.

The release README: "mimetics_v1.0.csv ... contains the URLs and timestamps of the YouTube clips, following the
Kinetics format. You can thus use similar script such as
https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics to download and trim the videos." This step
is that crawler's ``download_clip`` (copied below) run over ``<media_root>/release/mimetics_v1.0.csv`` (step 01).

Copied from: ActivityNet Kinetics crawler, https://github.com/activitynet/ActivityNet, commit
82304fa9f4a7219421df13dae6347a445c4cf5bc, ``Crawler/Kinetics/download.py`` (``download_clip``,
``construct_video_filename``), MIT License (Copyright (c) 2017 Fabian Caba H.; the licence text is at the end of this
file). Changes: ``youtube-dl`` (unmaintained) -> ``yt-dlp`` (``$YTDLP``, else ``yt-dlp`` on PATH, else
``python -m yt_dlp``); the command lists run without a shell; one download per YouTube video, then every clip of that
video is trimmed from it (560 videos hold the 713 clips); joblib -> a thread pool; the clips are written flat as
``<media_root>/clips/<clip_id>.mp4`` (``clip_id`` = ``<youtube_id>_<start:06d>_<end:06d>``, the names of the
release's ``mimetics_v1.0_objectsceneannot.txt``); a per-clip report is written; the trim adds the
bitstream filter ``-bsf:v setts=pts=PTS-STARTPTS:dts=DTS-STARTPTS`` (ffmpeg >= 4.4), which shifts the encoded
packets' timestamps so the first video frame of every clip is at pts 0 and changes no frame (without it about one
clip in five starts at 0.016-0.047 s, the first source frame after ``time_start``, and a frame rule evaluated from 0
misreads it), and a clip whose first video frame is still not at 0 is reported ``trim_failed`` and not staged. The yt-dlp format
(``-f mp4``) and the rest of the ffmpeg trim (``-ss <start> -t <end - start> -c:v libx264 -c:a copy``, output seeking)
are the crawler's.

YouTube videos disappear: a clip whose video is gone is reported ``unavailable`` and is simply not staged
(``yavih bench verify mimetics`` counts the coverage). Re-running skips clips already staged.

Usage: yavih bench prepare mimetics 02 [-- --root ROOT --workers 4 --limit N]
Needs yt-dlp (``pip install yt-dlp==2026.10.26``), ffmpeg >= 4.4 with libx264, and ffprobe.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL_BASE = "https://www.youtube.com/watch?v="


def ytdlp_command() -> list[str]:
    if os.environ.get("YTDLP"):
        return os.environ["YTDLP"].split()
    exe = shutil.which("yt-dlp")
    return [exe] if exe else [sys.executable, "-m", "yt_dlp"]


def construct_video_filename(row, clips_dir, trim_format="%06d"):
    """Given a dataset row, this function constructs the output filename for a given video (crawler, flat dir)."""
    basename = "%s_%s_%s.mp4" % (row["youtube_id"], trim_format % int(row["time_start"]), trim_format % int(row["time_end"]))
    return os.path.join(clips_dir, basename)


def download_video(video_identifier, tmp_dir, num_attempts=5):
    """The crawler's download half of ``download_clip``: ``(path | None, log)``."""
    assert isinstance(video_identifier, str), "video_identifier must be string"
    assert len(video_identifier) == 11, "video_identifier must have length 11"
    tmp_filename = os.path.join(tmp_dir, "%s.%%(ext)s" % uuid.uuid4())
    command = ytdlp_command() + ["--quiet", "--no-warnings", "-f", "mp4", "-o", tmp_filename, URL_BASE + video_identifier]
    attempts = 0
    while True:
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return None, err.output.decode("utf-8", "replace").strip()
        else:
            break
    found = glob.glob("%s*" % tmp_filename.split(".")[0])
    return (found[0], "Downloaded") if found else (None, "yt-dlp wrote no file")


def trim_clip(tmp_filename, output_filename, start_time, end_time):
    """The crawler's trim half of ``download_clip``: ``(status, log)``."""
    part = output_filename + ".part.mp4"
    command = ["ffmpeg", "-y", "-i", tmp_filename, "-ss", str(start_time), "-t", str(end_time - start_time),
               "-c:v", "libx264", "-c:a", "copy",
               "-bsf:v", "setts=pts=PTS-STARTPTS:dts=DTS-STARTPTS",  # yavih: the first video frame at pts 0
               "-threads", "1", "-loglevel", "panic", part]
    try:
        subprocess.check_output(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        if os.path.exists(part):
            os.remove(part)
        return False, err.output.decode("utf-8", "replace").strip() or "ffmpeg failed"
    if not os.path.exists(part) or os.path.getsize(part) == 0:
        return False, "ffmpeg wrote no output"
    start = video_start(part)
    if start is None or abs(start) > 1e-3:
        os.remove(part)
        return False, f"first video frame at {start} s, not 0"
    os.replace(part, output_filename)
    return True, "Downloaded"


def video_start(path):
    """``start_time`` of the first video stream (ffprobe), or None when it cannot be read."""
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=start_time", "-of", "csv=p=0", path], stderr=subprocess.STDOUT)
        return float(out.decode().strip().splitlines()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def do_video(ytid, rows, clips_dir, tmp_root):
    """Download one YouTube video once and trim every clip of it; one status tuple per clip."""
    out = []
    todo = []
    for row in rows:
        fn = construct_video_filename(row, clips_dir)
        clip_id = os.path.basename(fn)[: -len(".mp4")]
        if os.path.exists(fn):
            out.append((clip_id, "exists", ""))
        else:
            todo.append((clip_id, fn, int(row["time_start"]), int(row["time_end"])))
    if not todo:
        return out
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
        src, log = download_video(ytid, tmp_dir)
        if src is None:
            return out + [(c, "unavailable", log.splitlines()[-1] if log else "") for c, *_ in todo]
        for clip_id, fn, s, e in todo:
            ok, log = trim_clip(src, fn, s, e)
            out.append((clip_id, "downloaded" if ok else "trim_failed", "" if ok else log[-300:]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="media root (default: yavih config get mimetics.media_root)")
    ap.add_argument("--workers", type=int, default=4, help="parallel YouTube downloads (default 4)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (a sample; 0 = all)")
    args = ap.parse_args(argv)
    root = args.root
    if root is None:
        root = subprocess.check_output(["yavih", "config", "get", "mimetics.media_root"], text=True).strip()
    root = Path(root)
    csv_path = root / "release" / "mimetics_v1.0.csv"
    if not csv_path.is_file():
        print(f"[mimetics/02] missing {csv_path}: run `yavih bench prepare mimetics 01` first", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("[mimetics/02] ffmpeg is not on PATH", file=sys.stderr)
        return 2
    try:
        subprocess.check_output(ytdlp_command() + ["--version"], stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        print("[mimetics/02] yt-dlp not found: pip install yt-dlp==2026.10.26 (or set $YTDLP)", file=sys.stderr)
        return 2
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    videos: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        videos.setdefault(r["youtube_id"], []).append(r)
    items = list(videos.items())[: args.limit or None]
    clips_dir = root / "clips"
    tmp_root = root / "tmp"
    clips_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    status: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(do_video, y, rs, str(clips_dir), str(tmp_root)) for y, rs in items]
        for i, fut in enumerate(as_completed(futs), 1):
            status.extend(fut.result())
            if i % 25 == 0 or i == len(futs):
                print(f"[mimetics/02] {i}/{len(futs)} videos", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    status.sort()
    report = root / "download_report.csv"
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["clip_id", "status", "message", "checked_utc"])
        w.writerows([(c, s, m.replace("\n", " "), stamp) for c, s, m in status])
    counts: dict[str, int] = {}
    for _, s, _ in status:
        counts[s] = counts.get(s, 0) + 1
    staged = sum(1 for p in clips_dir.glob("*.mp4") if not p.name.endswith(".part.mp4"))
    print(f"[mimetics/02] {len(status)} clips of {len(items)} videos: {counts}; {staged} staged under {clips_dir}; report {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ActivityNet Kinetics crawler licence (https://github.com/activitynet/ActivityNet/blob/82304fa9f4a7219421df13dae6347a445c4cf5bc/Crawler/Kinetics/LICENSE):
#
# MIT License
#
# Copyright (c) 2017 Fabian Caba H.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
