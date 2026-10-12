#!/usr/bin/env bash
# k400 step 1 — download the Kinetics-400 "_256" release from Academic Torrents (setup.md "## Obtain the data").
#
#   torrent  "Kinetics400 Dataset: The Kinetics Human Action Video Dataset", uploaded 2020-11-07
#            https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26
#            infohash 184d11318372f70018cf9a72ef867e2fb9ce1d26, 157.31 GB, 2 files, no web seeds, few
#            seeders: expect several hours.
#   file 1   kinetics400.zip             144172278021 B  (train_256/ + test_256/)
#   file 2   kinetics400_validation.zip   13139875063 B  (val_256/)
#   sha256   see EXPECTED below (aria2 also verifies the torrent's piece hashes).
#
# It is the compressed height-256 re-encode (portrait clips 256x144), NOT the ~450 GB CVDF release
# (github.com/cvdfoundation/kinetics-dataset): a CVDF copy is a different media set (files, durations,
# containers), so the frozen manifest's paths do not match it and using it is a new protocol_version.
# Never scrape YouTube for K400.
#
# Runs as plain bash or as a CPU batch job. aria2 >= 1.37 (e.g. conda install -c conda-forge aria2;
# ARIA2=/path/to/aria2c if it is not on PATH). Phase 1 fetches the val zip alone (file 2, 12 GiB) so the eval
# split lands first; phase 2 the rest.
#
# Usage:  yavih bench prepare k400 01     (or: DIR=/path sbatch prepare/01_download_torrent.sh)
#         default DIR: $(yavih config get data.root)/k400_download
#SBATCH --job-name=k400-torrent
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00
#SBATCH --output=k400_torrent-%j.log
set -u

DIR="${DIR:-$(yavih config get data.root)/k400_download}" || exit 1
ARIA2="${ARIA2:-$(command -v aria2c || true)}"
[ -n "$ARIA2" ] || { echo "[k400/01] aria2c not on PATH; set ARIA2=/path/to/aria2c"; exit 1; }
TORRENT="${TORRENT:-$DIR/kinetics400.torrent}"
mkdir -p "$DIR"
if [ ! -s "$TORRENT" ]; then
  # Fetch the .torrent (1 500 770 B) first, on a machine with web access:
  #   curl -L -o "$TORRENT" https://academictorrents.com/download/184d11318372f70018cf9a72ef867e2fb9ce1d26.torrent
  # or use the magnet link magnet:?xt=urn:btih:184d11318372f70018cf9a72ef867e2fb9ce1d26 with aria2.
  echo "[k400/01] missing $TORRENT: curl -L -o \"$TORRENT\" https://academictorrents.com/download/184d11318372f70018cf9a72ef867e2fb9ce1d26.torrent"; exit 1
fi

EXTRA_TRACKERS='udp://tracker.opentrackr.org:1337/announce,udp://open.tracker.cl:1337/announce,udp://tracker.torrent.eu.org:451/announce,udp://exodus.desync.com:6969/announce,udp://tracker.openbittorrent.com:6969/announce'
COMMON="--continue=true --seed-time=0 --max-connection-per-server=8 --dir=$DIR \
  --enable-dht=true --bt-tracker=$EXTRA_TRACKERS --listen-port=6881-6999 \
  --dht-listen-port=6881-6999 --summary-interval=60 --console-log-level=notice \
  --bt-max-peers=100 --file-allocation=none"
echo "=== $(date -Is) phase 1: val zip only (file 2) on $(hostname) ==="
# shellcheck disable=SC2086
$ARIA2 $COMMON --select-file=2 "$TORRENT"; rc=$?
echo "=== $(date -Is) phase 1 exit $rc ==="
if [ $rc -eq 0 ]; then
  echo "=== $(date -Is) phase 2: full torrent (train zip) ==="
  # shellcheck disable=SC2086
  $ARIA2 $COMMON --select-file=1,2 "$TORRENT"; rc=$?
  echo "=== $(date -Is) phase 2 exit $rc ==="
fi

# The zips land in $DIR/kinetics400_dataset/. Expected sizes:
cd "$DIR/kinetics400_dataset" || exit 1
for f in kinetics400_validation.zip:13139875063 kinetics400.zip:144172278021; do
  n=${f%%:*}; want=${f##*:}; got=$(stat -c %s "$n" 2>/dev/null || echo 0)
  echo "[k400/01] $n size $got (expect $want)"; [ "$got" = "$want" ] || rc=3
done
# EXPECTED sha256 (sha256sum -c takes about an hour for 157 GB; SKIP_SHA=1 skips it):
#   33a69475d026e94b3a2f3ed708152b7beaf04a520b4ee455de0c1c883900483b  kinetics400_validation.zip
#   335dd19846e60078204b492c725003cfcbbb3f0bbff4dca525916785bd60e30f  kinetics400.zip
[ "${SKIP_SHA:-0}" = 1 ] || sha256sum -c --strict <<'SUMS' || rc=3
33a69475d026e94b3a2f3ed708152b7beaf04a520b4ee455de0c1c883900483b  kinetics400_validation.zip
335dd19846e60078204b492c725003cfcbbb3f0bbff4dca525916785bd60e30f  kinetics400.zip
SUMS
exit $rc
