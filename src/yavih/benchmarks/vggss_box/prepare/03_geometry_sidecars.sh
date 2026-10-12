#!/usr/bin/env bash
# vggss_box — step 3 (optional): the two geometry sidecars the freeze reads, shipped in this folder's prepare/ and
# pinned by sha256 in make_manifest.py:
#
#   frame_sizes.json      188 103 B  sha256 4ac9cf7250149b2fd39c59c0f01156c59e4c7fa2474099f55c25a21131edb37f
#       {clip_id: [width, height]} via ffprobe (stream=width,height), the 5 154 non-leaked clips
#   video_geometry.json   214 602 B  sha256 48cbfb8933225b2c18bce3b6c2b214d75ab062b7a01f6771a478f7feb6896fad
#       {clip_id: [num_frames, avg_fps]} via torchcodec, the 5 154 non-leaked clips
#
# frame_sizes.json gives the manifest's frame_w/frame_h; video_geometry.json gives the split draw's duration gate
# and the short-video count. This script verifies the sidecars and, as an independent check against what yavih's
# decoder sees, compares PyAV's coded frame size of every frozen row with frame_w/frame_h in the manifest
# (expect all rows equal). A mismatch means the staged media are not the frozen clips (e.g. a re-download):
# check that the boxes still describe them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

(cd "$HERE" && sha256sum -c --strict <<'SHA'
4ac9cf7250149b2fd39c59c0f01156c59e4c7fa2474099f55c25a21131edb37f  frame_sizes.json
48cbfb8933225b2c18bce3b6c2b214d75ab062b7a01f6771a478f7feb6896fad  video_geometry.json
SHA
)

"$PYTHON" - <<'PY'
import collections
from concurrent.futures import ThreadPoolExecutor
import av
from yavih import Benchmark
b = Benchmark.load("vggss_box")
root = b.resolved_roots()["media_root"]
jobs = []
for split in ("train", "eval"):
    m = b.manifest(split)
    jobs += [(m.clip_id[i], m.media[i]["video"], m.extra[i]["frame_w"], m.extra[i]["frame_h"]) for i in range(len(m))]
def probe(j):
    with av.open(f"{root}/{j[1]}") as c:
        s = c.streams.video[0]
        return j[0], (s.codec_context.width, s.codec_context.height) == (j[2], j[3]), (s.metadata or {}).get("rotate")
with ThreadPoolExecutor(16) as ex:
    out = list(ex.map(probe, jobs))
bad = [o[0] for o in out if not o[1]]
print(f"rows {len(out)}  frame size == manifest frame_w/frame_h: {len(out) - len(bad)}  rotate tags {collections.Counter(o[2] for o in out)}")
if bad:
    raise SystemExit(f"geometry mismatch on {len(bad)} rows, first10 {bad[:10]}: not the frozen media; check the boxes still describe it")
PY
