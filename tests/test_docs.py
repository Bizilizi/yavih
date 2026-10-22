"""The repository documentation (docs/, README.md) against the code: generated tables are current, the validation
code table and the CLI page cover the code, every link resolves, and nothing refers to the removed documents.

docs/ is repository documentation and is not shipped in the wheel; shipped files link to it by GitHub URL only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pytest

import docs_tables

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
PKG = REPO / "src" / "yavih"
REPO_URL = "https://github.com/Bizilizi/yavih/blob/main/"
PAGES = sorted(DOCS.glob("*.md"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- generated tables


@pytest.mark.parametrize("kind", sorted(docs_tables.PAGES))
def test_generated_docs_tables_are_current(kind: str) -> None:
    page = DOCS / docs_tables.PAGES[kind]
    text = _read(page)
    assert f"<!-- yavih:docs-table {kind} begin" in text, f"{page.name} lost its {kind} block"
    assert docs_tables.replace_blocks(text) == text, f"{page.name}: stale {kind} table (python tests/docs_tables.py --write)"


# ----------------------------------------------------------------------------- validation codes


def _code_rows(text: str) -> dict[str, tuple[str, str]]:
    """``{code: (level, waivable)}`` from rows ``| `CODE` | level | yes/no | meaning |``."""
    rows: dict[str, tuple[str, str]] = {}
    for code, level, waivable in re.findall(r"^\| `([EWN]_[A-Z0-9_]+)` \| (error|warning|note) \| (yes|no) \|", text, flags=re.M):
        assert code not in rows, f"{code} documented twice"
        rows[code] = (level, waivable)
    return rows


def test_validation_codes_match_the_documented_table() -> None:
    """Every ``yavih.io.validate.CODES`` entry has a row in docs/validation.md with its level, and the waivable column
    is ``yes`` exactly for ``yavih.waivers.WAIVABLE``; the never-waivable codes are documented as ``no``."""
    from yavih.io.validate import CODES
    from yavih.waivers import NOT_WAIVABLE, WAIVABLE

    rows = _code_rows(_read(DOCS / "validation.md"))
    for code, level in CODES.items():
        assert code in rows, f"docs/validation.md has no row for {code}"
        assert rows[code][0] == level, f"{code}: documented {rows[code][0]}, code says {level}"
    for code, (_, waivable) in rows.items():
        assert waivable == ("yes" if code in WAIVABLE else "no"), f"{code}: waivable column says {waivable}"
    for code in (*WAIVABLE, *NOT_WAIVABLE):
        assert code in rows, f"docs/validation.md has no row for {code}"
    assert rows["N_CANARY_DRIFT"] == ("note", "no") and rows["N_RULE_CUSTOM"][0] == "note"


# ----------------------------------------------------------------------------- CLI


def _command_paths(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                path = (*prefix, name)
                out.append(path)
                out.extend(_command_paths(sub, path))
    return out


def test_every_cli_command_is_documented() -> None:
    from yavih.cli import build_parser

    text = _read(DOCS / "cli.md")
    paths = _command_paths(build_parser())
    assert len(paths) > 20
    missing = [" ".join(p) for p in paths if f"yavih {' '.join(p)}" not in text]
    assert not missing, f"docs/cli.md does not document: {missing}"


def test_exit_codes_and_envelope_are_documented() -> None:
    from yavih import cli

    text = _read(DOCS / "cli.md")
    for key in ("status", "data", "errors", "warnings", "paths"):
        assert f"`{key}`" in text, key
    for status in cli.STATUSES:
        assert f"`{status}`" in text, status
    for code in (cli.EXIT_OK, cli.EXIT_USAGE, cli.EXIT_CONTRACT, cli.EXIT_NOT_READY, cli.EXIT_DEADLINE, cli.EXIT_INTERNAL):
        assert re.search(rf"^\| `?{code}`? \|", text, flags=re.M), f"exit code {code} has no row in docs/cli.md"


# ----------------------------------------------------------------------------- configuration examples

#: The first line of a ```toml block that is not a yavih configuration file (a plugin's pyproject.toml).
_NOT_YAVIH_CONFIG = "(not a yavih configuration file)"


def _toml_blocks() -> list[tuple[str, str]]:
    """Every fenced ```toml block (indented ones included) in docs/*.md, README.md and the shipped skill pages."""
    pages = PAGES + [REPO / "README.md"] + sorted((PKG / "skill").rglob("*.md"))
    out = []
    for page in pages:
        text = _read(page)
        for m in re.finditer(r"^( *)```toml\n(.*?)^\1```", text, re.M | re.S):
            body = re.sub(rf"^{m.group(1)}", "", m.group(2), flags=re.M)
            if _NOT_YAVIH_CONFIG in body.splitlines()[0]:
                continue
            out.append((f"{page.relative_to(REPO)}:{text[: m.start()].count(chr(10)) + 1}", body))
    return out


_TOML_BLOCKS = _toml_blocks()


@pytest.mark.parametrize("body", [b for _, b in _TOML_BLOCKS], ids=[w for w, _ in _TOML_BLOCKS])
def test_toml_examples_load_under_the_strict_schema(body: str, tmp_path: Path) -> None:
    """A copyable configuration example parses as TOML and passes the strict schema (a key written after a table
    header belongs to that table, so a top-level ``launcher`` below ``[project]`` is refused)."""
    import tomllib

    from yavih import config

    data = tomllib.loads(body)
    config.load(config=data, env={"GROUP": "/group", "HOME": str(tmp_path)}, cwd=tmp_path, lenient=False)


def test_toml_examples_are_found() -> None:
    assert len(_TOML_BLOCKS) >= 4


# ----------------------------------------------------------------------------- links


_LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)\)|<a [^>]*href=\"([^\"]+)\"|<img [^>]*src=\"([^\"]+)\"")


def _anchors(text: str) -> set[str]:
    """GitHub heading anchors of a markdown text (lowercase, punctuation dropped, spaces to dashes, duplicates -1, -2)."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^#{1,6} +(.*?) *#*$", line)
        if not m:
            continue
        slug = re.sub(r"[^\w\- ]", "", m.group(1).strip().lower().replace("`", "")).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        out.add(slug if n == 0 else f"{slug}-{n}")
    return out


def _links(text: str) -> list[str]:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    return [next(g for g in m.groups() if g) for m in _LINK.finditer(text)]


def _check_target(source: Path, target: Path, anchor: str | None) -> str | None:
    if not target.exists():
        return f"{source.relative_to(REPO)}: {target} does not exist"
    if anchor and target.suffix == ".md" and anchor not in _anchors(_read(target)):
        return f"{source.relative_to(REPO)}: no heading #{anchor} in {target.relative_to(REPO)}"
    return None


@pytest.mark.parametrize("path", [*PAGES, REPO / "README.md"], ids=lambda p: str(p.relative_to(REPO)))
def test_every_link_resolves(path: Path) -> None:
    """Relative links (and GitHub links into this repository's main branch) name an existing file and heading."""
    problems = []
    for link in _links(_read(path)):
        if link.startswith(REPO_URL):
            rel, _, anchor = link[len(REPO_URL):].partition("#")
            target = REPO / rel
        elif link.startswith("https://github.com/Bizilizi/yavih/tree/main/"):
            rel, _, anchor = link[len("https://github.com/Bizilizi/yavih/tree/main/"):].partition("#")
            target = REPO / rel
        elif re.match(r"^[a-z][a-z0-9+.-]*:", link):
            continue
        elif link.startswith("#"):
            target, anchor = path, link[1:]
        else:
            rel, _, anchor = link.partition("#")
            target = (path.parent / rel).resolve()
        problem = _check_target(path, target, anchor or None)
        if problem:
            problems.append(problem)
    assert not problems, "\n".join(problems)


def test_every_docs_page_is_in_the_index() -> None:
    index = _read(DOCS / "README.md")
    for page in PAGES:
        if page.name != "README.md":
            assert f"]({page.name})" in index, f"docs/README.md does not link {page.name}"


# ----------------------------------------------------------------------------- removed documents, shipped links


#: SPEC.md, DESIGN.md, IMPLEMENTATION.md and CHANGELOG.md were replaced by docs/; nothing may refer to them.
_REMOVED = re.compile(r"\bSPEC\b|\bDESIGN(\.md\b| \")|Philosophy v3|\bIMPLEMENTATION\b|\bCHANGELOG\b")
#: Lines that may name the retired documents.
_ALLOWED_MENTIONS: dict[str, tuple[str, ...]] = {
    # intended: `yavih bench lint` still refuses a folder that cites the retired documents (E_CITATION)
    "src/yavih/lint.py": ("the retired SPEC / DESIGN documents", r'\bSPEC\b|DESIGN(\.md| \")|Philosophy v3|IMPLEMENTATION'),
}


def _tracked_text_files() -> list[Path]:
    try:
        names = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    out = []
    for n in names:
        p = REPO / n
        if p.suffix in {".gz", ".png", ".npy", ".tar", ".jpg", ".wav", ".mp4"} or not p.is_file():
            continue
        out.append(p)
    return out


def test_nothing_refers_to_the_removed_documents() -> None:
    for name in ("SPEC.md", "DESIGN.md", "IMPLEMENTATION.md", "CHANGELOG.md"):
        assert not (REPO / name).exists(), name
    problems = []
    for path in _tracked_text_files():
        rel = str(path.relative_to(REPO))
        if rel == "tests/test_docs.py":
            continue
        try:
            text = _read(path)
        except UnicodeDecodeError:
            continue
        for allowed in _ALLOWED_MENTIONS.get(rel, ()):
            text = text.replace(allowed, "")
        for k, line in enumerate(text.splitlines(), 1):
            if _REMOVED.search(line):
                problems.append(f"{rel}:{k}: {line.strip()[:120]}")
    assert not problems, "\n".join(problems)


def test_shipped_files_link_docs_by_url_only() -> None:
    """Everything under src/yavih ships in the wheel and docs/ does not: a docs page is named by its GitHub URL."""
    problems = []
    for path in PKG.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".yaml", ".yml", ".sh", ".txt", ".toml"} or "__pycache__" in path.parts:
            continue
        for k, line in enumerate(_read(path).splitlines(), 1):
            for m in re.finditer(r"(?<![\w/.-])docs/[\w-]+\.md", line):
                if not line[: m.start()].endswith(REPO_URL):
                    problems.append(f"{path.relative_to(REPO)}:{k}: {line.strip()[:120]}")
    assert not problems, "\n".join(problems)
