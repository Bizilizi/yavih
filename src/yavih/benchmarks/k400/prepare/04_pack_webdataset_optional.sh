#!/usr/bin/env bash
# k400 step 4 (optional) — tar shards for filesystems with an inode quota: the 33 492 frozen files (20.2 GB) packed
# into ~1 GiB tars plus the index yavih's `layout: tar_index` reads:
#   index.json  {"version": 1, "shards": ["k400-000000.tar", ...], "members": {rel_media: [shard_idx, offset, size]}}
# (shard names are relative to the index's directory, so the directory can be moved or copied as a whole).
#
# With WDS_DIR/index.json absent it packs from the staged tree (ROOT, default the configured media_root); with it
# present it only verifies that every shard exists and every frozen rel_media is a member. Packing again into a
# fresh WDS_DIR is the way to change the shard size (SHARD_BYTES, default 1 GiB).
#
# The reader hands a member over in place (RawStream.open() / bytes(), os.pread; no PyAV needed by yavih). The bridge
# decodes through RawStream.open(), so a decoder that wants a file PATH must accept a file object instead (PyAV does;
# the ffmpeg CLI cannot read a member at an offset).
#
# CAUTION: `streams` is hashed, so the tar_index copy has a DIFFERENT protocol_hash from the packaged dir layout
# although every handed-over byte is identical; `report` never merges numbers from the two layouts. Extract every
# reading of a comparison from the same layout.
#
# Usage: yavih bench prepare k400 04 -- WDS_DIR [ROOT]
set -euo pipefail
HERE="${FOLDER:-$(cd "$(dirname "$0")/.." && pwd)}"
WDS="${1:?usage: yavih bench prepare k400 04 -- WDS_DIR [ROOT]}"
ROOT="${2:-$(yavih config get k400.media_root)}"
PY="${PYTHON:-python3}"

"$PY" - "$HERE/manifest.csv.gz" "$WDS" "$ROOT" "${SHARD_BYTES:-1073741824}" <<'PY'
import csv, gzip, io, json, os, sys, tarfile
man, wds, root, shard_bytes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rels = [r["rel_media"] for r in csv.DictReader(io.StringIO(gzip.decompress(open(man, "rb").read()).decode("utf-8"), newline=""))]
index_path = os.path.join(wds, "index.json")
if not os.path.exists(index_path):
    os.makedirs(wds, exist_ok=True)
    shards, members, tf, size = [], {}, None, 0
    for rel in rels:
        if tf is None or size >= shard_bytes:
            if tf is not None:
                tf.close()
            shards.append(f"k400-{len(shards):06d}.tar")
            tf, size = tarfile.open(os.path.join(wds, shards[-1]), "w", format=tarfile.PAX_FORMAT), 0
        p = os.path.join(root, rel)
        ti = tf.gettarinfo(p, arcname=rel)
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = 0
        with open(p, "rb") as fh:
            tf.addfile(ti, fh)
        # addfile leaves tf.offset after the data padded to 512-byte blocks: the data starts that far back
        members[rel] = [len(shards) - 1, tf.offset - -(-ti.size // 512) * 512, ti.size]
        size += ti.size
    if tf is not None:
        tf.close()
    tmp = index_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "shards": shards, "members": members}, fh)
    os.replace(tmp, index_path)
    print(f"[k400/04] packed {len(members)} members into {len(shards)} shards under {wds}")
idx = json.load(open(index_path, encoding="utf-8"))
missing_shards = [s for s in idx["shards"] if not os.path.isfile(os.path.join(wds, s))]
missing = [r for r in rels if r not in idx["members"]]
print(f"[k400/04] index v{idx['version']}: {len(idx['shards'])} shards ({len(missing_shards)} missing), "
      f"{len(idx['members'])} members; frozen rows missing from the index: {len(missing)} (expect 0) {missing[:5]}")
sys.exit(1 if missing or missing_shards else 0)
PY

cat <<MSG
To read the shards, make a stamped copy of the folder and change only the stream's storage:
  yavih bench fetch k400            # copies the folder to \$YAVIH_HOME/benchmarks/k400
  # in that copy's benchmark.yaml:
  #   streams.video.layout: tar_index
  #   data.tar_index: index.json
  #   data.roots.media_root.default: "$WDS"     (or YAVIH_DATA__K400__MEDIA_ROOT=$WDS)
  yavih bench verify k400
NOTE: the layout change forks protocol_hash (streams are hashed); keep one layout per comparison.
MSG
