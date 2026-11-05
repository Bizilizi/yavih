#!/usr/bin/env python
"""AVE step 3/5 — index ``AVE_Datset.tar`` IN PLACE (one sequential tar-header pass, never extracts).

Writes the v1 index yavih's ``layout: tar_index`` reads (``yavih.media.tarindex``)::

    {"version": 1, "shards": ["<shard>"], "members": {"video/<ytid>.mp4": [0, offset_data, size]}}

Member rules: only regular files matching ``(^|/)AVE/<ytid>.mp4`` with ytid ``[A-Za-z0-9_-]{1,32}`` are indexed,
keyed ``video/<ytid>.mp4``; a repeated name keeps the LAST occurrence (what ``tar -x`` would do; the real tar has 0
duplicates); a link onto a media name is a hard error; directories / other members (Annotations.txt, ReadMe.txt, …)
are counted and skipped.

``--shard-path relative`` (default) stores the bare file name, which ``yavih.media.tarindex`` resolves next to
``index.json``; ``absolute`` stores the tar's absolute path (the index then breaks when the data moves).
``--relocate INDEX`` rewrites an existing index's shards to bare names (offsets untouched). ``make_manifest.py`` pins
the member set (sorted ``{member: [offset, size]}``), not the file, so either shard path freezes the same manifest.

    python -I "$FOLDER/prepare/03_index_tar.py" --root /path/to/datasets/ave [--shard-path relative]
    python -I "$FOLDER/prepare/03_index_tar.py" --relocate /path/to/datasets/ave/index.json

Header walk: about a second with a warm page cache, ~0.5–5 min cold on network storage. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path

TAR_FILENAME = "AVE_Datset.tar"  # upstream typo is real
EXPECTED_TAR_BYTES = 13_353_553_920
EXPECTED_MEMBERS = 4097
_MP4_RE = re.compile(r"(?:^|/)AVE/([^/]+)\.mp4$")
_YTID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def index_tar(tar_path: Path) -> tuple[dict[str, list[int]], dict[str, int]]:
    members: dict[str, list[int]] = {}
    stats = {"duplicates": 0, "directories": 0, "nonregular": 0, "other": 0}
    with tarfile.open(tar_path, "r") as tar:
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
            members[rel] = [0, int(info.offset_data), int(info.size)]  # offset_data is authoritative
    return members, stats


def relocate(index_path: Path) -> None:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["shards"] = [os.path.basename(s) for s in payload["shards"]]
    _atomic_write(index_path, json.dumps(payload, separators=(",", ":")))
    print(f"[03_index_tar] relocated {index_path}: shards -> {payload['shards']} ({len(payload['members'])} members, offsets unchanged)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=None, help="dir holding AVE_Datset.tar (index.json is written there)")
    p.add_argument("--shard-path", choices=("absolute", "relative"), default="relative")
    p.add_argument("--relocate", type=Path, default=None, help="rewrite an existing index.json's shards to bare names and exit")
    a = p.parse_args(argv)
    if a.relocate:
        relocate(a.relocate)
        return 0
    if a.root is None:
        p.error("--root is required")
    tar_path = (a.root / TAR_FILENAME).resolve()
    if tar_path.stat().st_size != EXPECTED_TAR_BYTES:
        raise SystemExit(f"{tar_path} is {tar_path.stat().st_size} B, expected {EXPECTED_TAR_BYTES} (truncated download walks cleanly — refusing)")
    members, stats = index_tar(tar_path)
    if len(members) != EXPECTED_MEMBERS:
        raise SystemExit(f"{len(members)} mp4 members indexed, expected {EXPECTED_MEMBERS}")
    shard = str(tar_path) if a.shard_path == "absolute" else TAR_FILENAME
    out = a.root / "index.json"
    _atomic_write(out, json.dumps({"version": 1, "shards": [shard], "members": members}, separators=(",", ":")))
    print(f"[03_index_tar] wrote {out}: {len(members)} members, shard {shard!r}, stats {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
