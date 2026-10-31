#!/usr/bin/env python
"""audiocaps_t2a step 2/3 — write each frozen clip's reference wav from the OpenSound mirror into ``derived_root``.

For every manifest row, the wav bytes of the clip's caption rows in ``source_root/opensound-b29b324/data/test-*.parquet``
are written unchanged to ``derived_root/<rel_audio>`` and checked against the row's ``audio_sha256`` (the manifest
froze them; a mismatch means another mirror revision). Written to a temporary name and renamed into place; an existing
file with the right sha256 is kept (resumable). Prints a JSON summary; exit 1 on any mismatch or missing clip.

    yavih bench prepare audiocaps_t2a 02
    yavih bench prepare audiocaps_t2a 02 -- --source-root /path/to/audiocaps --out-root /path/to/derived

Needs pyarrow (prepare time only).
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
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
MIRROR = "opensound-b29b324/data"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    from yavih.config import folder_root

    src = Path(a.source_root or folder_root(FOLDER, "source_root"))
    out = Path(a.out_root or folder_root(FOLDER, "derived_root"))
    rows = list(csv.DictReader(io.StringIO(gzip.decompress((FOLDER / "manifest.csv.gz").read_bytes()).decode("utf-8"))))
    want = {r["clip_id"]: r for r in rows}
    done: dict[str, str] = {}
    for cid, r in want.items():
        dst = out / r["rel_audio"]
        if dst.is_file() and _sha(dst.read_bytes()) == r["audio_sha256"]:
            done[cid] = "kept"
    bad: list[tuple[str, str]] = []
    for f in sorted((src / MIRROR).glob("test-*.parquet")):
        t = pq.read_table(f, columns=["youtube_id", "start_time", "audio"])
        for y, s, au in zip(t.column("youtube_id").to_pylist(), t.column("start_time").to_pylist(), t.column("audio").to_pylist(), strict=True):
            cid = f"{y}_{int(s):06d}"
            r = want.get(cid)
            if r is None or cid in done:
                continue
            data = au["bytes"]
            if _sha(data) != r["audio_sha256"]:
                bad.append((cid, f"sha256 {_sha(data)[:12]} != frozen {r['audio_sha256'][:12]}"))
                done[cid] = "bad"
                continue
            dst = out / r["rel_audio"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            done[cid] = "written"
    missing = sorted(set(want) - set(done))
    counts = {k: sum(1 for v in done.values() if v == k) for k in ("kept", "written", "bad")}
    print(json.dumps({"rows": len(want), **counts, "missing": len(missing), "first_missing": missing[:5], "first_bad": bad[:5],
                      "out_root": str(out)}, indent=1))
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
