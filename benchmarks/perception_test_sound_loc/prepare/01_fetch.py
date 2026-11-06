#!/usr/bin/env python
"""Fetch everything this benchmark reads from the public Perception Test bucket, unattended and resumable.

1. ``annotations/``: the two challenge annotation zips the manifest was frozen from (sha256-pinned; 1.1 MB), so
   ``make_manifest.py --check`` can re-derive the manifest on this site.
2. ``videos/<video_id>.mp4`` and ``audios/<video_id>.wav`` for every manifest row, taken out of the official split zips
   (``train_videos.zip`` 28.4 GB, ``valid_videos.zip`` 75.4 GB, ``train_audios.zip`` 13.2 GB, ``valid_audios.zip``
   35.5 GB) WITHOUT downloading the zips: the zip's central directory is read through HTTP range requests, its
   listing (``name,size,crc32`` per member, in directory order) must hash to the pinned sha256 below, and each needed
   member is streamed out of the remote zip (only its own bytes) and checked against its listed size and CRC-32
   before it is moved into place. The bucket publishes no per-file sha256, so size + CRC-32 under a pinned listing is
   the pin. 7 368 videos + 7 368 wavs, 149 GB on disk.

A file already in place with the listed size is kept (``--verify`` re-reads it and checks its CRC-32 too), so an
interrupted run resumes. ``--first N`` fetches only the first ``N`` manifest rows of each split (what ``yavih plan
build --sample N`` reads); ``--only videos|audios`` one kind; ``--from-zips DIR`` extracts from zips you downloaded
yourself (same pins). The root is ``--root``, else the configured ``media_root`` of this benchmark
(``yavih config get <benchmark>.media_root``; the default ``${datasets.perception_test}`` is shared by
``perception_test_sound_loc`` and ``perception_test_action_loc``: a second run skips what the first fetched).

Run as ``yavih bench prepare <benchmark> 01 [-- --first 32 --workers 8]``. stdlib + yavih core only.
Exit 0 done; 2 a pin or CRC mismatch (nothing is moved into place); 3 network or disk failure.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FOLDER = Path(os.environ.get("FOLDER") or Path(__file__).resolve().parent.parent)
BENCH = os.environ.get("YAVIH_BENCH") or FOLDER.name
BASE = "https://storage.googleapis.com/dm-perception-test/zip_data"

#: the annotation zips make_manifest.py reads (this benchmark's task), sha256
ANNOTATIONS = {
    "sound": {
        "challenge_sound_localisation_train_annotations.zip": "2f94960db329d18f70ffdbfebc0a72e47476bd8ac42c81d7fecae2d7877f0a9e",
        "challenge_sound_localisation_valid_annotations.zip": "4f0e602ba8f97313aca7d84805334a73edd94fa6f56d80b945954cd32327c141",
    },
    "action": {
        "challenge_action_localisation_train_annotations.zip": "61304d4a2577c9fac449cdcffaa9060cc8d9985063c9aeae72776f605f0785ea",
        "challenge_action_localisation_valid_annotations.zip": "93235a985bc9f47aa9c6d4e6ae52332424e6a4514d49ffc9de7c0dde7a1b3d85",
    },
}
#: (split, kind) -> (zip, its byte size, sha256 of its listing "name,size,crc32\n" header + one line per member)
MEDIA_ZIPS = {
    ("train", "videos"): ("train_videos.zip", 28404087974, "96e7a0a43566b2966dad9a6b95098c44e39d0bcd59ef568245745b1d4f557839"),
    ("valid", "videos"): ("valid_videos.zip", 75375143328, "fd3c6075efe8fff2a2d30fe59ae84bd1a6f62f1d72e63429fe32e00b269fb13f"),
    ("train", "audios"): ("train_audios.zip", 13222906156, "be812d4bb270c4575a88eb4e9ccbf3aded7930301c242626d63661a274fedeba"),
    ("valid", "audios"): ("valid_audios.zip", 35506467738, "3091f5c6adc02ed181ac6e482fb4f0798c34d08923473bc8b49e71929e1c04f0"),
}
COLUMN = {"videos": "rel_video", "audios": "rel_audio"}


class PinError(Exception):
    pass


class HTTPRangeFile(io.RawIOBase):
    """A read-only seekable view of a remote file through HTTP ``Range`` requests (retried)."""

    def __init__(self, url: str, size: int | None = None) -> None:
        self.url = url
        if size is None:
            with _open(urllib.request.Request(url, method="HEAD")) as r:
                size = int(r.headers["Content-Length"])
        self.size = int(size)
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b) -> int:
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + n) - 1
        with _open(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _open(req: urllib.request.Request, tries: int = 6):
    delay = 2.0
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if k == tries - 1:
                raise
            print(f"[{BENCH} 01] retry {k + 1} after {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def listing_sha256(zf: zipfile.ZipFile) -> str:
    text = "name,size,crc32\n" + "".join(f"{i.filename},{i.file_size},{i.CRC:08x}\n" for i in zf.infolist() if not i.is_dir())
    return hashlib.sha256(text.encode()).hexdigest()


def media_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    from yavih import config

    try:
        return Path(config.folder_root(FOLDER, "media_root", None))
    except config.ConfigError as e:
        raise SystemExit(f"pass --root or configure {BENCH}.media_root: {e}") from None


def manifest_rows(first: int | None) -> dict[str, list[dict[str, str]]]:
    with gzip.open(FOLDER / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        out.setdefault(r["split"], []).append(r)
    if first is not None:
        out = {s: v[:first] for s, v in out.items()}
    return out


def fetch_annotations(root: Path, task: str) -> None:
    d = root / "annotations"
    d.mkdir(parents=True, exist_ok=True)
    for name, want in ANNOTATIONS[task].items():
        dest = d / name
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == want:
            continue
        with _open(urllib.request.Request(f"{BASE}/{name}")) as r:
            data = r.read()
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise PinError(f"{name}: sha256 {got}, expected {want}")
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        print(f"[{BENCH} 01] annotations/{name} ok", file=sys.stderr)


def crc32_file(path: Path) -> int:
    c = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            c = zlib.crc32(chunk, c)
    return c & 0xFFFFFFFF


def fetch_zip_members(root: Path, zip_name: str, size: int, want_listing: str, members: list[str], *, workers: int,
                      verify: bool, from_zips: Path | None) -> tuple[int, int, int]:
    """``(fetched, kept, bytes)`` for ``members`` of one zip."""
    local = from_zips / zip_name if from_zips else None

    def open_zip() -> zipfile.ZipFile:
        if local is not None:
            return zipfile.ZipFile(local)
        return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(f"{BASE}/{zip_name}", size), buffer_size=8 << 20))

    zf = open_zip()
    got = listing_sha256(zf)
    if got != want_listing:
        raise PinError(f"{zip_name}: central-directory listing sha256 {got}, expected {want_listing} (the upstream zip changed)")
    info = {i.filename: i for i in zf.infolist()}
    absent = [m for m in members if m not in info]
    if absent:
        raise PinError(f"{zip_name}: {len(absent)} manifest members are not in the zip, e.g. {absent[:3]}")
    todo, kept = [], 0
    for m in members:
        dest = root / m
        if dest.is_file() and dest.stat().st_size == info[m].file_size and (not verify or crc32_file(dest) == info[m].CRC):
            kept += 1
        else:
            todo.append(m)
    print(f"[{BENCH} 01] {zip_name}: listing pinned ok; {len(members)} needed, {kept} in place, {len(todo)} to fetch", file=sys.stderr)
    local_zf = threading.local()

    def one(m: str) -> int:
        z = getattr(local_zf, "zf", None)
        if z is None:
            z = local_zf.zf = open_zip()
        dest = root / m
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        crc, n = 0, 0
        with z.open(m) as src, open(tmp, "wb") as out:  # ZipExtFile also checks the CRC at EOF
            for chunk in iter(lambda: src.read(1 << 22), b""):
                crc = zlib.crc32(chunk, crc)
                n += len(chunk)
                out.write(chunk)
        if n != info[m].file_size or (crc & 0xFFFFFFFF) != info[m].CRC:
            tmp.unlink(missing_ok=True)
            raise PinError(f"{zip_name}:{m}: size/CRC-32 {n}/{crc & 0xFFFFFFFF:08x}, expected {info[m].file_size}/{info[m].CRC:08x}")
        tmp.replace(dest)
        return n

    done, nbytes, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(one, m): m for m in todo}
        for f in as_completed(futs):
            nbytes += f.result()
            done += 1
            if done % 100 == 0 or done == len(todo):
                rate = nbytes / max(1e-6, time.time() - t0) / 1e6
                print(f"[{BENCH} 01] {zip_name}: {done}/{len(todo)} fetched ({nbytes / 1e9:.1f} GB, {rate:.0f} MB/s)", file=sys.stderr)
    return done, kept, nbytes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=None, help="the media root (default: the configured media_root)")
    p.add_argument("--first", type=int, default=None, help="only the first N manifest rows of each split")
    p.add_argument("--only", choices=("videos", "audios"), default=None, help="fetch one kind only")
    p.add_argument("--workers", type=int, default=8, help="parallel member downloads (default 8)")
    p.add_argument("--verify", action="store_true", help="re-check the CRC-32 of files already in place")
    p.add_argument("--from-zips", type=Path, default=None, help="extract from locally downloaded split zips in DIR instead")
    p.add_argument("--skip-annotations", action="store_true")
    args = p.parse_args(argv)
    task = "sound" if "sound" in BENCH else "action"
    root = media_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        if not args.skip_annotations:
            fetch_annotations(root, task)
        rows = manifest_rows(args.first)
        total = [0, 0, 0]
        for (split, kind), (zip_name, size, listing) in MEDIA_ZIPS.items():
            if args.only and kind != args.only:
                continue
            members = [r[COLUMN[kind]] for r in rows.get(split, [])]
            if not members:
                continue
            got = fetch_zip_members(root, zip_name, size, listing, members, workers=args.workers, verify=args.verify, from_zips=args.from_zips)
            total = [a + b for a, b in zip(total, got, strict=True)]
    except PinError as e:
        print(f"[{BENCH} 01] error E_PIN: {e}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, OSError) as e:
        print(f"[{BENCH} 01] error E_FETCH: {type(e).__name__}: {e} (rerun to resume)", file=sys.stderr)
        return 3
    print(f"[{BENCH} 01] done: {total[0]} fetched ({total[2] / 1e9:.2f} GB), {total[1]} already in place, under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
