#!/usr/bin/env bash
# audioset20k step 2/2 — check every file against the release's SHA256SUMS, check the three pinned release files
# against benchmark.yaml `source.sha256`, then let yavih verify presence, labels and curation and write the
# media-info census (setup.md "## Prepare & verify"). Hashing 66 GB takes ~10 min on NFS; SKIP_SUMS=1 skips it.
#
#   DEST=/path/to/datasets/audioset20k-wds-v1 bash 02_verify.sh   # default: `yavih config get audioset20k.media_root`
set -euo pipefail
DEST="${DEST:-$(yavih config get audioset20k.media_root)}"
YAVIH=${YAVIH:-yavih}
fail=0
pin() { # pin <file> <sha256>
  local got; got=$(sha256sum "$DEST/$1" | cut -d' ' -f1)
  if [ "$got" = "$2" ]; then echo "ok $1"; else echo "BAD SHA256 $1 $got (want $2)"; fail=1; fi
}
pin SHA256SUMS a6f22393b4f813af63e241b4a407771e0d98ade764a8399c4dd70951861b177c
pin index.json 6976d509de988883a7e00783cec5dc438c1d38517d877518212a7ec88317b441
pin clips.csv  96e308e4e01647dafded79207663837b82e5e124371dbcf3d752bbfeea99fbad
if [ "${SKIP_SUMS:-0}" != 1 ]; then
  (cd "$DEST" && sha256sum -c --quiet SHA256SUMS) && echo "ok SHA256SUMS (65 files)" || fail=1
fi
[ "$fail" = 0 ] || exit 1
YAVIH_DATA__AUDIOSET20K__MEDIA_ROOT=$DEST "$YAVIH" bench verify audioset20k --census
