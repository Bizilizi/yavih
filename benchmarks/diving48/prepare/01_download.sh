#!/usr/bin/env bash
# diving48 step 1/3 — download the official files from the Nextcloud shares linked on
# http://www.svcl.ucsd.edu/projects/resound/dataset.html (no account, no agreement) into the media root, and check
# every pinned size and sha256:
#
#   Diving48_rgb.tar.gz      10 287 167 940 B   rgb/<vid_name>.mp4 for every clip of the 2018 release
#   Diving48_V2_train.json        1 586 934 B   15 027 entries (V2, 2020-10-30 update)
#   Diving48_V2_test.json           207 545 B    1 970 entries
#   Diving48_vocab.json               3 094 B   48 classes
#
#   yavih bench prepare diving48 01                         # into $(yavih config get diving48.media_root)
#   ROOT=/path/to/datasets/diving48 yavih bench prepare diving48 01
#   CONNECTIONS=4 yavih bench prepare diving48 01           # the archive as 4 parallel byte ranges
#
# Needs curl and sha256sum. By default the archive comes over one connection, resumed (`curl -C -`) after a drop
# and on the next run. The share serves HTTP ranges: CONNECTIONS=N fetches the archive as N byte ranges in
# parallel, each resumed until complete, then concatenated (several times faster when one connection is slow).
# A file already present with the pinned size and sha256 is not downloaded again. Hashing the archive reads 10 GB.
set -euo pipefail
ROOT="${ROOT:-$(yavih config get diving48.media_root)}"
CONNECTIONS="${CONNECTIONS:-1}"
NC=https://nextcloud.nrp-nautilus.io/s
mkdir -p "$ROOT"
cd "$ROOT"

# name  share  bytes  sha256
FILES="
Diving48_V2_train.json Z3CfTwcnTtp5d4p 1586934 62bcff1537b24162b1c9be49f376b8f9b6fccd222734f10d4a5d545b38c00e93
Diving48_V2_test.json mfr5pfaSNp5Zx8X 207545 658fe0d84ebea61414548bb69a996d668d9f249a82919e276ffc0f424d9371a3
Diving48_vocab.json oWjtCkbw9G4qdSy 3094 cc4238904e3ab274ae3ab36d26a09afd41e8c7891aa1f7c773d43fecb9388696
Diving48_rgb.tar.gz eqKMRFHqNCrP77L 10287167940 RGB_SHA256
"

ok() { [ -f "$1" ] && [ "$(stat -c %s "$1")" = "$2" ] && echo "$3  $1" | sha256sum -c --quiet - >/dev/null 2>&1; }

ranged() {  # name url bytes: CONNECTIONS parallel byte ranges, each resumed until complete, then concatenated
  local name=$1 url=$2 bytes=$3 n=$CONNECTIONS chunk i
  chunk=$(( (bytes + n - 1) / n ))
  for i in $(seq 0 $((n - 1))); do
    (
      a=$((i * chunk)); b=$(( (i + 1) * chunk - 1 )); [ "$b" -ge "$bytes" ] && b=$((bytes - 1)); want=$((b - a + 1))
      for _ in $(seq 1 100); do
        have=$(stat -c %s "$name.part$i" 2>/dev/null || echo 0)
        [ "$have" -ge "$want" ] && exit 0
        curl -sSL --retry 5 -r $((a + have))-"$b" "$url" >> "$name.part$i" || sleep 5
      done
      exit 1
    ) &
  done
  wait
  for i in $(seq 0 $((n - 1))); do cat "$name.part$i"; done > "$name.tmp"
  rm -f "$name".part*
  mv "$name.tmp" "$name"
}

echo "$FILES" | while read -r name share bytes sha; do
  [ -n "$name" ] || continue
  [ "$sha" = RGB_SHA256 ] && sha=$RGB_SHA256
  if ok "$name" "$bytes" "$sha"; then echo "=== $name present: size and sha256 ok"; continue; fi
  url="$NC/$share/download"
  echo "=== $(date -Is) downloading $name ($bytes B)"
  if [ "$bytes" -gt 100000000 ] && [ "$CONNECTIONS" -gt 1 ]; then
    rm -f "$name"
    ranged "$name" "$url" "$bytes"
  else
    for _ in $(seq 1 50); do
      [ -f "$name" ] && [ "$(stat -c %s "$name")" -ge "$bytes" ] && break
      curl -sSL --retry 5 -C - -o "$name" "$url" || sleep 5
    done
  fi
  ok "$name" "$bytes" "$sha" || { echo "FAIL: $name: size $(stat -c %s "$name" 2>/dev/null || echo 0) B or its sha256 differs from the pinned $bytes B / $sha; delete it and rerun" >&2; exit 1; }
  echo "=== $name ok"
done
echo "=== $(date -Is) OK: the three JSON files and the RGB archive match their pinned sha256 in $ROOT"
