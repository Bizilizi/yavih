"""Re-freeze ``manifest.csv.gz`` / ``classes.csv`` of llp_avvp byte-identically from the public upstream files.

This docstring is the re-freeze recipe (``yavih bench show llp_avvp make_manifest`` prints the file, ``--help`` this
text); keep it current with the code.

Inputs (read-only): TODO: the public upstream annotation files, their pinned revision and sha256s.

Outputs, byte-identical on re-run: TODO: ``manifest.csv.gz`` (``yavih.freeze.gzip_bytes``), ``classes.csv``
(``yavih.freeze.csv_bytes``), written with ``yavih.freeze.write_if_changed``; the ``manifest.upstream`` block (hashes,
``upstream_files``, ``curation``) printed for ``benchmark.yaml``.

Curation: TODO: what is excluded and why, the leakage audit, the seeded train draw
(``yavih.freeze.class_balanced_subsample``), the counts recorded under ``manifest.upstream.curation``.

Usage::

    python "$(yavih bench path llp_avvp)/make_manifest.py" --check   # re-derive and compare, write nothing
    python "$(yavih bench path llp_avvp)/make_manifest.py"           # re-freeze (in a git checkout)
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="re-derive and compare with the committed files; write nothing")
    ap.parse_args(argv)
    print("TODO: make_manifest.py of llp_avvp is not written yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
