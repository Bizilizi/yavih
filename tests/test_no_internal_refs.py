"""yavih is an independent harness: no tracked file names the project it was developed alongside, a development
diary, or the maintainer's sites (cluster names, mount points, home directories)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: Assembled from parts so this file does not match itself.
PATTERN = "|".join([
    "av-?" + "joint", "av_" + "joint", "av" + "joint", "zver" + "ev", "atcre" + "mers", "mc" + "ml", "jupi" + "ter",
    "/stor" + "age/", "/home/" + "wiss", "dog" + "food", "review " + "round",
])
#: The only allowed hits: the copyright holder, the package author, and the VGGSounder paper's author list.
ALLOWED: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("LICENSE", re.compile(r"^Copyright \(c\) \d{4} Daniil Zverev$")),
    ("pyproject.toml", re.compile(r'^authors = \[\{ name = "Daniil Zverev" \}\]$')),
    ("src/yavih/benchmarks/vggsounder/README.md", re.compile(r"zverev\w*2025vggsounder|Daniil Zverev and |Zverev et al\.")),
    ("src/yavih/benchmarks/vggsounder/benchmark.yaml", re.compile(r"^\s+venue: .*Zverev")),
)


def _allowed(path: str, line: str) -> bool:
    return any(path == p and rx.search(line) for p, rx in ALLOWED)


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_no_internal_references_in_tracked_files():
    proc = subprocess.run(["git", "grep", "-n", "-I", "-i", "-E", PATTERN, "--", ".", ":(exclude)tests/test_no_internal_refs.py"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr
    hits = []
    for row in proc.stdout.splitlines():
        path, _, rest = row.partition(":")
        _, _, line = rest.partition(":")
        if not _allowed(path, line):
            hits.append(row[:200])
    assert not hits, "internal references in tracked files:\n" + "\n".join(hits[:50])
