#!/usr/bin/env bash
# unav100_causal — step 1: the pinned Hugging Face revision (3 annotation parquets + 13 video archive parts, 66 GB).
# Sizes are always checked; sha256 of the 13 parts too unless SKIP_SHA=1 (hashing 66 GB takes a while).
set -euo pipefail
yavih bench snapshot unav100_causal
ROOT="$(yavih config get unav100_causal.media_root)"
python - "$ROOT" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
parts = {f"unav_raw_videos.tar.gz.part_{i:02d}": (5368709120, h) for i, h in enumerate([
    "aa8dc6798fc8b99d69533b4ebd068a0f659532d13ff82a21a527db5fbdd37f65", "d549709123379d9c9f3b63d2aabd31a09f72c0004c50178e0a8b509896122af0",
    "2f1fae736a5d567519bb6edfa2c79f97571923a6dbb679fb0350adf8df59aeb0", "021ef07d239e735af94a76e46421350788850b2a3b5bfb44b8124b1205cfd6ba",
    "89c2097a6bab34441302d2f2db65d4e504468a9f7ade713bef4698c6d6223546", "f56ad30d4e4bee762f8c9577689a909e93a267785f7e4452fafe5d987e106a1a",
    "25cea1db67f92036241fcb84a4da430be3b79d9a49a620252cab20f18d141d37", "feb5c3f1268c8125d3d9012ef0c37f4e8dc6ddff0493922e3d6dbbc71d7c7677",
    "064ca3707ed4b234ae935413d4f95dc7bc9e55d716df8f39a00576f40dd142d1", "6b5f72f5892e36aa808c357a365cabe826bafa28438cd308e7596977265f0a9a",
    "5ddca8ddc839cdf71a6f0d0c8e67bc14400bf8aaf1e9a8f646f7bfba9eefcc69", "815520a0b00649d77f22e6a4abc18b1e971691d6da355ba7bd951e6cbc45c600",
    "d38371218b5f1dcf2babd039b882608ecf4cd24a6863b4f6c6a39dc03eb788d3"])}
parts["unav_raw_videos.tar.gz.part_12"] = (1351094873, parts["unav_raw_videos.tar.gz.part_12"][1])
bad = []
for name, (size, sha) in parts.items():
    p = os.path.join(root, name)
    if not os.path.isfile(p) or os.path.getsize(p) != size:
        bad.append(f"{name}: missing or wrong size")
        continue
    if os.environ.get("SKIP_SHA") != "1":
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            bad.append(f"{name}: sha256 differs")
print("\n".join(bad) if bad else f"13 parts ok under {root}")
sys.exit(1 if bad else 0)
PY
