#!/usr/bin/env bash
# mimetics step 1 — the official Mimetics v1.0 release (NAVER LABS Europe, one 21 869 B zip): download it, check its
# sha256, unpack it into <media_root>/release/. Usage: yavih bench prepare mimetics 01 [-- ROOT]
# (default ROOT: $(yavih config get mimetics.media_root)). Idempotent: a verified zip is not downloaded again.
set -euo pipefail
ROOT="${1:-$(yavih config get mimetics.media_root)}"
URL="https://europe.naverlabs.com/wp-content/uploads/2019/12/Mimetics_release_v1.0.zip"
ZIP_SHA256="6ddc9c98c3f720baa5c70f1dc6cc05b80f1a16deb679a728082a698afb8680ab"
CSV_SHA256="78b32467a248c24b7082d9a29d9eb0bf6eeb4462ed8abda290d2036018115044"

mkdir -p "$ROOT/release"
ZIP="$ROOT/release/Mimetics_release_v1.0.zip"
sha() { sha256sum "$1" | cut -d' ' -f1; }
if [ ! -f "$ZIP" ] || [ "$(sha "$ZIP")" != "$ZIP_SHA256" ]; then
  curl -fsSL --retry 3 -o "$ZIP.part" "$URL"
  mv "$ZIP.part" "$ZIP"
fi
got="$(sha "$ZIP")"
if [ "$got" != "$ZIP_SHA256" ]; then
  echo "[mimetics/01] $ZIP: sha256 $got != pinned $ZIP_SHA256 (a different release; refusing)" >&2
  exit 1
fi
unzip -o -q "$ZIP" -d "$ROOT/release"
got="$(sha "$ROOT/release/mimetics_v1.0.csv")"
if [ "$got" != "$CSV_SHA256" ]; then
  echo "[mimetics/01] mimetics_v1.0.csv: sha256 $got != pinned $CSV_SHA256" >&2
  exit 1
fi
echo "[mimetics/01] release ok: $ROOT/release/mimetics_v1.0.csv (713 clips of 50 Kinetics-400 classes)"
