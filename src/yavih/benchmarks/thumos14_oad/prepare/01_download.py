"""thumos14_oad — step 1: the THUMOS'14 temporal annotations and the 413 annotated videos, from crcv.ucf.edu.

The server sends its certificate without the intermediate CA, so a plain TLS client cannot verify it. This step fetches
the issuer certificate the server certificate names (its Authority Information Access URL), checks that it chains to a
system root (``openssl verify``) and uses the system roots plus that certificate: verification stays on. Then:

* ``annotations/TH14_Temporal_annotations_{validation,test}.zip`` and ``annotations/Class_Index_Detection.txt``
  (sha256-pinned below), unzipped into ``annotations/{validation,test}/``;
* every video an annotation file names (200 validation, 213 test) as ``videos/{validation,test}/<name>.mp4``, one by
  one with resume; with ``--sha`` each file is checked against the folder's ``video_sha256.csv``.

    yavih bench prepare thumos14_oad 01 -- [--workers 4] [--sha]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://www.crcv.ucf.edu/THUMOS14"
FILES = {
    "TH14_Temporal_annotations_validation.zip": (f"{BASE}/Validation_set/TH14_Temporal_annotations_validation.zip",
                                                 "6a86e27d21c86be31112a1bf96318c7810a462cbff07ce14a29446f302eb6c05"),
    "TH14_Temporal_annotations_test.zip": (f"{BASE}/test_set/TH14_Temporal_annotations_test.zip",
                                           "8a9f2270b90fc300c0bcf4edc8394af7874b83acf4aa4af054ae065b5120749d"),
    "Class_Index_Detection.txt": (f"{BASE}/Class%20Index_Detection.txt", "d0a1711d537c6837c43ddbde943aaa8c648e524ed48dd86e6561b5d3d280e98f"),
}
VIDEO_URL = {"validation": f"{BASE}/Validation_set/videos/{{}}.mp4", "test": f"{BASE}/test_set/TH14_test_set_mp4/{{}}.mp4"}
ANN_DIR = {"validation": "annotation", "test": "TH14_Temporal_Annotations_Test/annotations/annotation"}
SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"


def ca_bundle(tmp: Path) -> str:
    """The system roots plus the server's issuer certificate (fetched from its AIA URL and verified against them)."""
    import ssl

    pem = ssl.get_server_certificate(("www.crcv.ucf.edu", 443))
    (tmp / "server.pem").write_text(pem)
    out = subprocess.run(["openssl", "x509", "-in", str(tmp / "server.pem"), "-noout", "-ext", "authorityInfoAccess"],
                         capture_output=True, text=True, check=True).stdout
    url = next(line.split("URI:", 1)[1].strip() for line in out.splitlines() if "CA Issuers" in line)
    der = urllib.request.urlopen(url, timeout=60).read()  # a certificate: its trust comes from the verification below
    (tmp / "issuer.der").write_bytes(der)
    subprocess.run(["openssl", "x509", "-inform", "DER", "-in", str(tmp / "issuer.der"), "-out", str(tmp / "issuer.pem")], check=True)
    subprocess.run(["openssl", "verify", "-CAfile", SYSTEM_CA, str(tmp / "issuer.pem")], check=True, capture_output=True)
    bundle = tmp / "bundle.pem"
    bundle.write_text(Path(SYSTEM_CA).read_text() + (tmp / "issuer.pem").read_text())
    return str(bundle)


def fetch(url: str, dest: Path, ctx) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        req = urllib.request.Request(url, method="HEAD")
        size = int(urllib.request.urlopen(req, context=ctx, timeout=120).headers["Content-Length"])
        if dest.stat().st_size == size:
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, context=ctx, timeout=600) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    os.replace(part, dest)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def names(root: Path, split: str) -> list[str]:
    d = root / "annotations" / split / ANN_DIR[split]
    return sorted({line.split()[0] for f in d.glob("*.txt") for line in f.read_text().splitlines() if line.strip()})


def main(argv: list[str] | None = None) -> int:
    import ssl

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None, help="default: yavih config get thumos14_oad.media_root")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sha", action="store_true", help="check every video against video_sha256.csv")
    ns = ap.parse_args(argv)
    root = ns.root or Path(subprocess.run(["yavih", "config", "get", "thumos14_oad.media_root"], capture_output=True, text=True,
                                          check=True).stdout.strip())
    with tempfile.TemporaryDirectory() as t:
        ctx = ssl.create_default_context(cafile=ca_bundle(Path(t)))
        ann = root / "annotations"
        for name, (url, want) in FILES.items():
            p = ann / name
            fetch(url, p, ctx)
            if sha256(p) != want:
                sys.exit(f"{p}: sha256 differs from the pinned {want}")
        for split in ("validation", "test"):
            with zipfile.ZipFile(ann / f"TH14_Temporal_annotations_{split}.zip") as z:
                z.extractall(ann / split)
            inner = next((ann / split).rglob("*ViperXGTF.zip"), None)  # the raw Viper files: not needed
            if inner is not None:
                inner.unlink()
        todo = [(split, n) for split in ("validation", "test") for n in names(root, split)]
        with ThreadPoolExecutor(ns.workers) as ex:
            list(ex.map(lambda sn: fetch(VIDEO_URL[sn[0]].format(sn[1]), root / "videos" / sn[0] / f"{sn[1]}.mp4", ctx), todo))
    print(f"{len(todo)} videos under {root / 'videos'}")
    if ns.sha:
        here = Path(os.environ.get("FOLDER") or Path(__file__).resolve().parents[1])
        want = dict(line.split(",") for line in (here / "video_sha256.csv").read_text().split()[1:])
        bad = [f"{s}/{n}" for s, n in todo if sha256(root / "videos" / s / f"{n}.mp4") != want.get(f"{s}/{n}")]
        print("sha256 ok" if not bad else f"{len(bad)} videos differ: {bad[:5]}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
