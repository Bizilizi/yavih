"""Tables in docs/ generated from code constants: ``python tests/docs_tables.py --write`` (``--check`` exits 1 on drift).

A generated table sits between two marker lines::

    <!-- yavih:docs-table <kind> begin (generated from ...: python tests/docs_tables.py --write) -->
    ...
    <!-- yavih:docs-table <kind> end -->

``tests/test_docs.py`` fails when a committed block differs from what this module renders. docs/ is repository
documentation (not shipped), so this lives with the tests rather than in the package.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
_BEGIN = re.compile(r"^<!-- yavih:docs-table (?P<kind>[a-z_]+) begin\b.*-->\s*$")


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def env_table() -> str:
    """Every environment variable yavih reads (``yavih.config.ENV_VARS``), in declaration order."""
    from yavih.config import ENV_VARS

    lines = ["| variable | meaning |", "|---|---|"]
    lines += [f"| `{name}` | {_cell(text)} |" for name, text in ENV_VARS.items()]
    return "\n".join(lines)


RENDERERS: dict[str, Callable[[], str]] = {"env": env_table}
#: Which page holds which generated block.
PAGES: dict[str, str] = {"env": "configuration.md"}


def replace_blocks(text: str) -> str:
    """``text`` with every generated block re-rendered; unknown kinds raise."""
    out: list[str] = []
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        m = _BEGIN.match(lines[i].rstrip("\n"))
        out.append(lines[i])
        i += 1
        if not m:
            continue
        kind = m.group("kind")
        end = f"<!-- yavih:docs-table {kind} end -->"
        j = i
        while j < len(lines) and lines[j].strip() != end:
            j += 1
        if j == len(lines):
            raise ValueError(f"docs-table {kind}: no end marker")
        out.append(RENDERERS[kind]() + "\n")
        out.append(lines[j])
        i = j + 1
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    write = "--write" in argv
    stale = []
    for page in PAGES.values():
        path = DOCS / page
        old = path.read_text(encoding="utf-8")
        new = replace_blocks(old)
        if new != old:
            stale.append(page)
            if write:
                path.write_text(new, encoding="utf-8")
    if stale and not write:
        print("stale generated docs tables: " + ", ".join(stale) + " (python tests/docs_tables.py --write)")
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
