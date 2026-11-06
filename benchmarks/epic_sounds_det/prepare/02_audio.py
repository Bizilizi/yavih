#!/usr/bin/env python
"""epic_sounds_det step 2/3: stage ``audio/<video_id>.m4a`` for every manifest row (unattended, resumable).

Per recording: download the EPIC-KITCHENS-100 MP4 from data.bris.ac.uk (public HTTPS, no account; HTTP Range
resume), check its md5 against ``annotations/epic_kitchens_md5.csv`` (step 01), copy its AAC audio track into an
``.m4a`` WITHOUT re-encoding (PyAV remux: the packets are the stored bytes), check that the sha256 over the
output's audio packet payloads equals the source's, rename into place and delete the MP4 (``--keep-mp4`` keeps it
under ``mp4/``). A file already in ``audio/`` is skipped. ``audio/_staged.jsonl`` gets one line per staged file
(video_id, mp4 md5, mp4 bytes, packet sha256, m4a sha256, PyAV version): a site record, never shipped.

Why audio only and why a remux (setup.md "## Obtain the data"): the 632 labelled recordings are 1.11 TB of 1080p
MP4 (up to 13.9 GB each) and yavih reads a row's whole stored file into memory before ``prepare``; the AAC
track of all 632 is ~6 GB. The upstream EPIC-SOUNDS audio tooling also starts from the MP4's audio track.

    yavih bench prepare epic_sounds_det 02                          # every row, 4 parallel downloads
    yavih bench prepare epic_sounds_det 02 -- --split validation --jobs 8
    yavih bench prepare epic_sounds_det 02 -- --videos P03_26,P26_30 # a sample
    yavih bench prepare epic_sounds_det 02 -- --plan                 # print what would be fetched, fetch nothing

Disk: ``--tmp`` (default ``$ROOT/mp4_tmp``) holds at most ``--jobs`` MP4s at a time (<= 14 GB each).
Time: ~1.9 MB/s per connection from data.bris.ac.uk as measured on 2026-10-29: 1.11 TB is ~7 days on one
connection; the server accepts parallel connections (``--jobs``).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FOLDER = Path(os.environ.get("FOLDER") or Path(__file__).resolve().parents[1])
BASE_55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/"   # epic_downloader.py epic_55_base_url
BASE_100 = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/"  # epic_downloader.py epic_100_base_url
MD5_SHA256 = "3f9c31899ff5807c8f0fed5d26a8fe62300f3f4c40dc3eea767345212c2e83c7"
CHUNK = 1 << 22


def log(msg: str) -> None:
    print(f"[02_audio {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def manifest_rows() -> list[dict[str, str]]:
    with gzip.open(FOLDER / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def video_index(md5_csv: Path) -> dict[str, tuple[str, str]]:
    """video_id -> (url, md5) from the pinned EPIC-KITCHENS md5 list."""
    got = hashlib.sha256(md5_csv.read_bytes()).hexdigest()
    if got != MD5_SHA256:
        raise SystemExit(f"{md5_csv}: sha256 {got} != pinned {MD5_SHA256} (run step 01)")
    out = {}
    with open(md5_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            p = r["file_remote_path"]
            if p.endswith(".MP4"):
                out[Path(p).stem] = ((BASE_55 if r["version"] == "55" else BASE_100) + p, r["md5"])
    return out


def download(url: str, dest: Path, md5: str, tries: int = 8) -> int:
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as resp:
                if have and resp.status != 206:   # server ignored the range: restart
                    have = 0
                with open(part, "ab" if have else "wb") as fh:
                    while block := resp.read(CHUNK):
                        fh.write(block)
            h = hashlib.md5()
            with open(part, "rb") as fh:
                for block in iter(lambda: fh.read(CHUNK), b""):
                    h.update(block)
            if h.hexdigest() != md5:
                part.unlink()
                raise OSError(f"md5 {h.hexdigest()} != {md5}")
            part.rename(dest)
            return dest.stat().st_size
        except OSError as e:  # URLError, timeouts, md5 mismatch
            log(f"{dest.name}: attempt {attempt}/{tries} failed: {e}")
            time.sleep(min(300, 10 * attempt))
    raise SystemExit(f"{url}: giving up after {tries} attempts")


def packet_sha(path: Path) -> tuple[str, int]:
    import av

    h, n = hashlib.sha256(), 0
    with av.open(str(path)) as c:
        for pkt in c.demux(c.streams.audio[0]):
            if pkt.size:
                h.update(bytes(pkt))
                n += 1
    return h.hexdigest(), n


def remux_audio(src: Path, dst: Path) -> None:
    import av

    with av.open(str(src)) as inp:
        a = inp.streams.audio[0]
        with av.open(str(dst), "w", format="ipod", container_options={"fflags": "+bitexact"}) as out:
            o = out.add_stream_from_template(a)
            for pkt in inp.demux(a):
                if pkt.dts is None:
                    continue
                pkt.stream = o
                out.mux(pkt)


def stage(vid: str, url: str, md5: str, root: Path, tmp: Path, keep: bool) -> dict:
    final = root / "audio" / f"{vid}.m4a"
    mp4 = tmp / f"{vid}.MP4"
    t0 = time.time()
    nbytes = mp4.stat().st_size if mp4.exists() else download(url, mp4, md5)
    t1 = time.time()
    part = final.with_name(final.name + ".part.m4a")
    remux_audio(mp4, part)
    src_sha, n_src = packet_sha(mp4)
    out_sha, n_out = packet_sha(part)
    if (src_sha, n_src) != (out_sha, n_out):
        part.unlink()
        raise SystemExit(f"{vid}: remuxed audio packets differ from the MP4's ({n_out} vs {n_src} packets)")
    part.rename(final)
    if keep:
        (root / "mp4").mkdir(exist_ok=True)
        mp4.rename(root / "mp4" / mp4.name)
    else:
        mp4.unlink()
    import av

    rec = {"video_id": vid, "mp4_md5": md5, "mp4_bytes": nbytes, "audio_packets": n_src, "audio_packets_sha256": src_sha,
           "m4a_bytes": final.stat().st_size, "m4a_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
           "pyav": av.__version__, "download_s": round(t1 - t0, 1), "remux_s": round(time.time() - t1, 1)}
    log(f"{vid}: {nbytes / 1e9:.2f} GB MP4 -> {rec['m4a_bytes'] / 1e6:.1f} MB m4a ({n_src} AAC packets) in {time.time() - t0:.0f} s")
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=None, help="the staged root (default: the configured epic_sounds_det.media_root)")
    p.add_argument("--split", choices=("train", "validation", "all"), default="all")
    p.add_argument("--videos", default=None, help="comma-separated video ids (a subset of the manifest)")
    p.add_argument("--jobs", type=int, default=4, help="parallel downloads")
    p.add_argument("--tmp", type=Path, default=None, help="where MP4s live while their audio is copied (default $ROOT/mp4_tmp)")
    p.add_argument("--keep-mp4", action="store_true", help="keep each MP4 under $ROOT/mp4/ (1.11 TB for all rows)")
    p.add_argument("--plan", action="store_true", help="print the download plan (count, URLs) and exit")
    args = p.parse_args(argv)
    if args.root is None:
        from yavih import config

        try:
            args.root = config.folder_root(FOLDER, "media_root", None)
        except config.ConfigError as e:
            raise SystemExit(f"pass --root or configure epic_sounds_det.media_root: {e}") from None
    root: Path = args.root
    idx = video_index(root / "annotations" / "epic_kitchens_md5.csv")
    rows = [r for r in manifest_rows() if args.split == "all" or r["split"] == args.split]
    if args.videos:
        want = set(args.videos.split(","))
        rows = [r for r in rows if r["clip_id"] in want]
        if len(rows) != len(want):
            raise SystemExit(f"not manifest rows: {sorted(want - {r['clip_id'] for r in rows})}")
    (root / "audio").mkdir(parents=True, exist_ok=True)
    todo = [r["clip_id"] for r in rows if not (root / "audio" / f"{r['clip_id']}.m4a").exists()]
    log(f"{len(rows)} rows selected, {len(rows) - len(todo)} already staged, {len(todo)} to fetch")
    if args.plan:
        out = io.StringIO()
        for v in todo:
            out.write(f"{v}\t{idx[v][0]}\t{idx[v][1]}\n")
        sys.stdout.write(out.getvalue())
        return 0
    tmp = args.tmp or (root / "mp4_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max(1, args.jobs)) as ex, open(root / "audio" / "_staged.jsonl", "a") as rec:
        for r in ex.map(lambda v: stage(v, *idx[v], root, tmp, args.keep_mp4), todo):
            rec.write(json.dumps(r, sort_keys=True) + "\n")
            rec.flush()
    log("AUDIO_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
