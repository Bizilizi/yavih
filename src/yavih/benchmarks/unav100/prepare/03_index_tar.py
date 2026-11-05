#!/usr/bin/env python
"""UnAV-100 step 3/4 — index ``unav_raw_videos.tar`` IN PLACE (one sequential tar-header pass, never extracts).

Writes the v1 index yavih's ``layout: tar_index`` reads (``yavih.media.tarindex``)::

    {"version": 1, "shards": ["unav_raw_videos.tar"], "members": {"video/<ytid>.mp4": [0, offset_data, size]}}

Member rules: regular files whose name ends in ``raw_videos/<ytid>.mp4`` (the upstream tar stores them under the
authors' path ``ssd3/gengtiantian/audio_visual_data/unav100/raw_videos/``) with ``ytid`` ``[A-Za-z0-9_-]{1,32}`` are
indexed, keyed ``video/<ytid>.mp4`` (the manifest's ``rel_media``); a repeated name keeps the LAST occurrence (what
``tar -x`` would do); a link onto a media name is a hard error; anything else is counted and skipped. The full tar must
hold exactly the 10 790 videos of the manifest (``--allow-partial`` relaxes that for a sample tar).

    yavih bench prepare unav100 03                                   # ROOT = `yavih config get unav100.media_root`
    yavih bench prepare unav100 03 -- --root /path/to/datasets/unav100

Header walk: a few seconds with a warm page cache, minutes cold on network storage. Stdlib + yavih config.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path

TAR_FILENAME = "unav_raw_videos.tar"
EXPECTED_MEMBERS = 10790
_MP4_RE = re.compile(r"(?:^|/)raw_videos/([^/]+)\.mp4$")
_YTID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def index_tar(tar_path: Path) -> tuple[dict[str, list[int]], dict[str, int]]:
    members: dict[str, list[int]] = {}
    stats = {"duplicates": 0, "directories": 0, "nonregular": 0, "other": 0}
    with tarfile.open(tar_path, "r:") as tar:
        while True:
            info = tar.next()
            if info is None:
                break
            tar.members.clear()  # keep RSS flat
            if not info.isfile():
                if info.isdir():
                    stats["directories"] += 1
                elif (info.islnk() or info.issym()) and _MP4_RE.search(info.name):
                    raise SystemExit(f"member {info.name!r} is a link; an offset index cannot reproduce tar -x for links")
                else:
                    stats["nonregular"] += 1
                continue
            m = _MP4_RE.search(info.name)
            if m is None or not _YTID_RE.fullmatch(m.group(1)):
                stats["other"] += 1
                continue
            rel = f"video/{m.group(1)}.mp4"
            if rel in members:
                stats["duplicates"] += 1
            members[rel] = [0, int(info.offset_data), int(info.size)]
    return members, stats


def _default_root() -> Path:
    if os.environ.get("ROOT"):
        return Path(os.environ["ROOT"])
    from yavih import config

    folder = Path(os.environ.get("FOLDER") or Path(__file__).resolve().parent.parent)
    try:
        return config.folder_root(folder, "media_root", None)
    except config.ConfigError as e:
        raise SystemExit(f"pass --root or configure unav100.media_root: {e}") from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=None, help="dir holding unav_raw_videos.tar (index.json is written there)")
    p.add_argument("--allow-partial", action="store_true", help="index a sample tar holding fewer than the 10 790 videos")
    a = p.parse_args(argv)
    root = a.root or _default_root()
    tar_path = (root / TAR_FILENAME).resolve()
    if not tar_path.is_file():
        raise SystemExit(f"{tar_path} not found: run step 02 first")
    members, stats = index_tar(tar_path)
    if len(members) != EXPECTED_MEMBERS and not a.allow_partial:
        raise SystemExit(f"{len(members)} mp4 members indexed, expected {EXPECTED_MEMBERS} (stats {stats})")
    out = root / "index.json"
    _atomic_write(out, json.dumps({"version": 1, "shards": [TAR_FILENAME], "members": dict(sorted(members.items()))}, separators=(",", ":")))
    print(f"[03_index_tar] wrote {out}: {len(members)} members, shard {TAR_FILENAME!r}, stats {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
