#!/usr/bin/env python
"""Freeze the ``audiocaps_t2a`` manifest from the official AudioCaps test captions and the pinned OpenSound audio
mirror (README.md, "Re-freezing the manifest").

Inputs (read only), under ``--root`` (the ``source_root`` of benchmark.yaml; prepare/01 downloads both):

- ``upstream/test.csv``: the official AudioCaps test captions (github.com/cdjkim/audiocaps, commit
  ``80fb3ade6d400d4311354a360db25da051a48cc1``, sha256-pinned): 4 875 captions, five per clip, 975 clips;
- ``opensound-b29b324/data/test-*.parquet``: the OpenSound/AudioCaps Hugging Face dataset at revision
  ``b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded`` (one row per caption that survived on YouTube, with the clip's
  wav bytes: 24 kHz, mono, stereo or 5.1).

Rows = the official test clips (``youtube_id``, ``start_time``) the mirror holds audio for, in the order of their
first caption in ``test.csv``. A clip whose caption rows in the mirror carry different audio bytes is excluded (the
reference would be ambiguous), and so is a clip whose recording lasts less than ``AUDIO_LENGTH`` (10 s, the scored
window, read from the wav header): every reference then covers the window every generation fills, and av-benchmark's
batched extraction, which truncates but never pads, sees equal lengths. Each row keeps the clip's FIRST caption in ``test.csv`` order (one prompt per clip),
that caption's ``audiocap_id``, the path the staged reference wav gets under ``derived_root`` and its sha256.
``clip_id`` is ``<youtube_id>_<start_time:06d>`` (VGGSound's form: av-benchmark pairs files by this name).

Outputs, byte-identical on re-run: ``manifest.csv.gz`` and a printed JSON block with every value ``benchmark.yaml``
needs. ``--check`` compares against the committed file instead of writing (exit 1 on drift). pyarrow is needed here
(freeze time only).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
CAPTIONS = "upstream/test.csv"
CAPTIONS_SHA256 = "b91c4b7ded2f4f6e7db5b9c4983dc1e1dca3d556f505b61e3fd65cac7e1c638a"
MIRROR = "opensound-b29b324/data"
MIRROR_FILES = 41
AUDIO_LENGTH = 10.0
HEADER = ["clip_id", "split", "caption", "audiocap_id", "rel_audio", "audio_sha256"]


def clip_id(youtube_id: str, start: int) -> str:
    return f"{youtube_id}_{int(start):06d}"


def mirror_audio(root: Path) -> tuple[dict[str, set[str]], dict[str, float]]:
    """``{clip_id: {sha256 of the wav bytes of each of its caption rows}}`` over the mirror's test parquet files, and
    ``{clip_id: duration in seconds}`` from the wav header of its first row."""
    import io as _io

    import pyarrow.parquet as pq
    import soundfile as sf

    files = sorted((root / MIRROR).glob("test-*.parquet"))
    if len(files) != MIRROR_FILES:
        sys.exit(f"{root / MIRROR}: {len(files)} test parquet files, expected {MIRROR_FILES} (prepare/01)")
    out: dict[str, set[str]] = {}
    seconds: dict[str, float] = {}
    for f in files:
        t = pq.read_table(f, columns=["youtube_id", "start_time", "audio"])
        for y, s, a in zip(t.column("youtube_id").to_pylist(), t.column("start_time").to_pylist(), t.column("audio").to_pylist(), strict=True):
            cid = clip_id(y, s)
            out.setdefault(cid, set()).add(hashlib.sha256(a["bytes"]).hexdigest())
            if cid not in seconds:
                seconds[cid] = float(sf.info(_io.BytesIO(a["bytes"])).duration)
    return out, seconds


def freeze(root: Path) -> tuple[bytes, dict]:
    raw = (root / CAPTIONS).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != CAPTIONS_SHA256:
        sys.exit(f"{root / CAPTIONS}: sha256 {got} != pinned {CAPTIONS_SHA256}")
    caps = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    first: dict[str, dict[str, str]] = {}
    for r in caps:
        first.setdefault(clip_id(r["youtube_id"], int(r["start_time"])), r)
    audio, seconds = mirror_audio(root)
    out, absent, ambiguous, short = [], 0, 0, 0
    for cid, r in first.items():
        shas = audio.get(cid)
        if not shas:
            absent += 1
            continue
        if len(shas) != 1:
            ambiguous += 1
            continue
        if seconds[cid] < AUDIO_LENGTH:
            short += 1
            continue
        out.append([cid, "test", r["caption"].strip(), r["audiocap_id"], f"audio/{cid}.wav", next(iter(shas))])
    data = csv_bytes(HEADER, out)
    facts = {
        "manifest": {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(out)},
        "upstream": {"captions_sha256": CAPTIONS_SHA256, "captions": len(caps), "clips": len(first)},
        "curation": {
            "total_upstream": len(first),
            "per_split": {"test": {"listed": len(out), "staged": len(out), "missing": 0}},
            "excluded": {"not_in_mirror": absent, "mirror_audio_ambiguous": ambiguous, "shorter_than_audio_length": short},
        },
    }
    return data, facts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="the source_root: upstream/test.csv and opensound-b29b324/data/")
    ap.add_argument("--check", action="store_true", help="compare with the committed manifest instead of writing")
    args = ap.parse_args()
    data, facts = freeze(args.root)
    path = HERE / "manifest.csv.gz"
    if args.check:
        same = path.is_file() and gzip.decompress(path.read_bytes()) == data
        print(json.dumps({"ok": same, **facts}, indent=1))
        sys.exit(0 if same else 1)
    write_if_changed(path, gzip_bytes(data))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
