#!/usr/bin/env python
"""avrobustbench — step 2 (re-freezing only): the VGGSound-2C clip list, from the authors' Hugging Face release.

VGGSOUND-2C is the VGGSound test clips the authors could download (14 046 of the 15 446 ``split == test`` rows of
``vggsound.csv``). The list exists only as the member names of the released ``vgg_data/clean_vgg.zip`` (one
``clean_vgg/<ytid>_<start>.mp4`` per clip) in the dataset ``sakshamsingh1/av_robust_data`` at the pinned revision
below. This step writes those member names, in the zip's central-directory order, one per line, to
``<derived_root>/meta/clean_vgg.members.txt`` and checks the file against the sha256 that ``make_manifest.py`` pins.

By default only the zip's central directory is read, through HTTP range requests (a few MB of the 3.0 GB file;
stdlib only). ``--zip PATH`` reads a local copy instead, after checking its sha256 against the release's LFS
digest. Nothing else of the release is used: the benchmark's media are made at prepare time from the VGGSound
corpus (step 4), never downloaded.

Usage (``FOLDER=$(yavih bench path avrobustbench)``)::

    yavih bench prepare avrobustbench 02                      # ranged read of the remote central directory
    yavih bench prepare avrobustbench 02 -- --zip clean_vgg.zip   # or a downloaded copy of the zip
    yavih bench prepare avrobustbench 02 -- --out FILE          # write elsewhere (default: <derived_root>/meta/)
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
NAME = "avrobustbench"
REPO = "sakshamsingh1/av_robust_data"
REVISION = "42b5e0789ccd1a1545a9e2a4c68e6ecb499541bb"
MEMBER_ZIP = "vgg_data/clean_vgg.zip"
ZIP_BYTES = 3004035238
ZIP_SHA256 = "74ca5d49efa3dc340695a16f2176d81082e868e48ba2ca24ad493ab863d48b3e"  # the release's LFS oid
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{MEMBER_ZIP}"
#: sha256 of the written list (make_manifest.py MEMBERS_SHA256 pins the same value)
MEMBERS_SHA256 = "d8a5754e7a2824848209986f363dc92bc5802ea10c4b44d431f5e5a2583c244a"
OUT_NAME = "clean_vgg.members.txt"


def _range(url: str, lo: int, hi: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}", "User-Agent": "yavih-prepare"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != hi - lo + 1:
        raise SystemExit(f"range {lo}-{hi}: got {len(data)} bytes (the server ignored the Range header?)")
    return data


def remote_members(url: str = URL, size: int = ZIP_BYTES) -> list[str]:
    """Member names of the remote zip in central-directory order (zip64 aware), read through range requests."""
    tail_len = min(size, 65536 + 22 + 76)
    tail = _range(url, size - tail_len, size - 1)
    i = tail.rfind(b"PK\x06\x06")  # zip64 end of central directory record
    if i >= 0:
        (_, _, _, _, _, _, _, n, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", tail[i:i + 56])
    else:
        j = tail.rfind(b"PK\x05\x06")
        if j < 0:
            raise SystemExit(f"{url}: no end-of-central-directory record in the last {tail_len} bytes")
        (_, _, _, _, n, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[j:j + 22])
    cd = _range(url, cd_off, cd_off + cd_size - 1)
    names, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        names.append(cd[p + 46:p + 46 + fl].decode("utf-8"))
        p += 46 + fl + el + cl
    if len(names) != n:
        raise SystemExit(f"{url}: central directory lists {len(names)} entries, the end record says {n}")
    return names


def local_members(path: Path) -> list[str]:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != ZIP_SHA256:
        raise SystemExit(f"{path}: sha256 {h.hexdigest()} != the release's {ZIP_SHA256}")
    with zipfile.ZipFile(path) as z:
        return [i.filename for i in z.infolist()]


def default_out() -> Path:
    from yavih import config

    try:
        return Path(config.folder_root(FOLDER, "derived_root")) / "meta" / OUT_NAME
    except config.ConfigError as e:
        raise SystemExit(f"pass --out or configure the derived_root of {NAME} ({e})") from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", type=Path, default=None, help="a local copy of vgg_data/clean_vgg.zip (sha256-checked)")
    ap.add_argument("--out", type=Path, default=None, help=f"output file (default: <derived_root>/meta/{OUT_NAME})")
    ns = ap.parse_args(argv)
    names = local_members(ns.zip) if ns.zip else remote_members()
    data = "".join(f"{n}\n" for n in names).encode("utf-8")
    got = hashlib.sha256(data).hexdigest()
    if got != MEMBERS_SHA256:
        raise SystemExit(f"member list sha256 {got} != pinned {MEMBERS_SHA256} ({len(names)} entries)")
    out = ns.out or default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    print(f"{out}: {len(names)} entries, sha256 {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
