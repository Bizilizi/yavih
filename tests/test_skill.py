"""The agent skill: ``yavih/skill/yavih/SKILL.md`` (≤ 250 lines, nine steps with stop conditions, setup.md
as the first read per benchmark, the guardrails as a rule → checking-code table, "When blocked", the operational rules) and its
``references/*.md``; every ``yavih <group> [<cmd>]`` the skill mentions parses with the real argparse tree;
``references/data.md`` stays in sync with the frozen ``benchmark.yaml``s; ``yavih skill install`` / ``show``
serve the package data."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from pathlib import Path

import pytest

from yavih import cli

REPO = Path(__file__).resolve().parents[1]
SKILL_ROOT = Path(cli.__file__).resolve().parent / "skill" / "yavih"
SKILL_MD = SKILL_ROOT / "SKILL.md"
REFERENCES = ("bridge", "data", "readouts", "slurm", "cli")
# the skill's step table: "<i> <step>" in the first column
GROUPS = ("init", "doctor", "config", "skill", "bench", "bridge", "extract", "features", "eval", "report", "registry", "plan", "exec", "suite")
# CLI flags the skill documents
DOCUMENTED_FLAGS = {
    ("bench", "list"): ("--catalog",),
    ("bench", "fetch"): ("--from", "--refresh"),
    ("bench", "update"): ("--yes", "--from"),
    ("suite", "show"): ("--fetch", "--from"),
    ("plan", "init"): ("--no-fetch", "--from"),
    ("bench", "info"): ("--setup",),
    ("bench", "verify"): ("--census",),
    ("eval", "run"): ("--log-samples", "--results", "--no-results"),
    ("report", "build"): ("--results", "--merge-results"),
}
DOCUMENTED_EXTRACT_FLAGS = ("--read-ahead",)

_CMD_RE = re.compile(r"`?\byavih\s+(?:--json\s+)?(" + "|".join(GROUPS) + r")\b(?:\s+([a-z]+))?")


def _skill_texts() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in [SKILL_MD, *(SKILL_ROOT / "references" / f"{r}.md" for r in REFERENCES)]}


def _section(text: str, heading: str) -> str:
    """Body of the ``## heading…`` section (up to the next ``## ``)."""
    m = re.search(rf"^## {re.escape(heading)}[^\n]*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    assert m, f"section '## {heading}' missing"
    return m.group(1)


# ------------------------------------------------------------------------------------------- SKILL.md shape


def test_frontmatter_parses_with_name_and_description():
    """``skill install`` (the Cursor rule, the AGENTS.md block) and agents' skill discovery read the frontmatter."""
    import yaml

    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    head = yaml.safe_load(text.split("---", 2)[1])
    assert head["name"] == "yavih" and isinstance(head["description"], str) and head["description"].strip()


def _resolve(dotted: str) -> object:
    """Import the longest module prefix of ``dotted`` and walk the rest as attributes."""
    import importlib

    parts = dotted.split(".")
    for k in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:k]))
        except ImportError:
            continue
        for attr in parts[k:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


def test_every_checker_the_guardrails_table_names_exists():
    """Every ``yavih.…`` function the skill's guardrails table names as a checker resolves to a callable."""
    section = _section(SKILL_MD.read_text(encoding="utf-8"), "Guardrails")
    names = set(re.findall(r"`(yavih\.[A-Za-z0-9_.]+)`", section))
    assert names
    for name in sorted(names):
        assert callable(_resolve(name)), name


def test_shipped_skill_cites_nothing_an_installed_user_cannot_open():
    """The repository docs/ pages and notes/ are not in the wheel: the skill names them by GitHub URL only."""
    from yavih.lint import UNOPENABLE

    for name, text in _skill_texts().items():
        for line in text.splitlines():
            if "github.com/Bizilizi/yavih" not in line:
                assert not UNOPENABLE.search(line), f"{name}: {line.strip()[:140]!r}"

def test_exploring_a_benchmark_block_runs():
    """Every command of the "Exploring a benchmark" block works on a packaged benchmark (no checkout needed)."""
    skill = SKILL_MD.read_text(encoding="utf-8")
    block = skill[skill.index("Exploring a benchmark:"):]
    block = block[block.index("```bash") + 7: block.index("```", block.index("```bash") + 7)]
    for cmd in ("suite list", "bench list", "bench files <b>", "bench show <b> setup", "bench show <b> readme|yaml|eval",
                "bench show <b> prepare/<file>", "bench prepare <b> [<step>]", "bench path <b> [<file>]", "skill show --ref"):
        assert f"yavih {cmd}" in block, cmd
    for argv in (["suite", "list"], ["bench", "list"], ["bench", "files", "esc50"], ["bench", "show", "esc50", "setup"],
                 ["bench", "show", "esc50", "readme"], ["bench", "show", "esc50", "yaml"], ["bench", "show", "avsync", "eval"],
                 ["bench", "show", "esc50", "prepare/01_download.sh"], ["bench", "prepare", "esc50"],
                 ["bench", "path", "esc50", "prepare/01_download.sh"], ["skill", "show", "--ref", "bridge"]):
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli.main(["--json", *argv]) == 0, argv


# ------------------------------------------------------------------------------------------- references


def test_bridge_md_has_a_row_per_temporal_rule():
    """The closed temporal vocabulary (``yavih.media.rules.TEMPORAL_RULES``): one row each in references/bridge.md."""
    from yavih.media.rules import TEMPORAL_RULES

    text = (SKILL_ROOT / "references" / "bridge.md").read_text(encoding="utf-8")
    for rule in TEMPORAL_RULES:
        assert re.search(rf"^\| `{rule}` \|", text, flags=re.M), f"bridge.md: temporal rule {rule!r} has no row"


def test_cli_md_documents_flags_and_codes():
    text = (SKILL_ROOT / "references" / "cli.md").read_text(encoding="utf-8")
    for flags in DOCUMENTED_FLAGS.values():
        for f in flags:
            assert f in text, f"cli.md: {f} missing"
    for n in (*DOCUMENTED_EXTRACT_FLAGS, "YAVIH_RESULTS", "E_USED", "E_NETWORK", "E_RERUN_DRIFT", "E_BATCH_INVARIANCE",
              "E_TIMES_OUT_OF_MEDIA", "E_RULE_MISMATCH", "W_SIDE_EFFECT", "W_NO_PREPROCESSING_SOURCE",
              "W_LOOSE_TOLERANCE", "W_RULE_UNCHECKED", "E_EXTERNAL_MISSING", "E_EXTERNAL_VERSION", "MediaUndecodable",
              # hashing, report and join codes
              "N_RULE_CUSTOM", "E_CONFIG_DRIFT", "W_WEIGHTS_UNFINGERPRINTED", "W_BRIDGE_CHANGED",
              "E_JOIN_MODEL", "W_JOIN_WEIGHTS", "<results_root>/<model>/results_<timestamp>.json",
              "samples_<benchmark>_<reading>_<timestamp>.jsonl", "eval_hash"):
        assert n in text, f"cli.md: {n!r} missing"


# ------------------------------------------------------------------------------------------- CLI agreement


def _parse_help(argv: list[str]) -> int:
    parser = cli.build_parser()
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([*argv, "--help"])
    return int(exc.value.code or 0)


def test_the_cli_reference_synopsis_lists_every_top_level_command():
    """The synopsis block at the top of references/cli.md names exactly the parser's top-level commands."""
    text = (SKILL_ROOT / "references" / "cli.md").read_text(encoding="utf-8")
    block = text.split("```", 2)[1]
    listed = {line.split()[0] for line in block.splitlines()[1:] if line.startswith("  ") and not line.startswith("    ")}
    sub = next(a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    assert listed == set(sub.choices), f"synopsis {sorted(listed)} != parser {sorted(sub.choices)}"
    assert set(GROUPS) == set(sub.choices)


def test_every_mentioned_command_parses_with_argparse():
    texts = _skill_texts()
    seen: set[tuple[str, ...]] = set()
    for text in texts.values():
        for m in _CMD_RE.finditer(text):
            group, sub = m.group(1), m.group(2)
            argv = [group] if sub is None else [group, sub]
            if sub is not None:
                # a prose word after a group ("yavih plan status reads", "yavih exec plan.jsonl") is not a subcommand
                probe = cli.build_parser()
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        probe.parse_args([group, sub, "--help"])
                except SystemExit as e:
                    if e.code not in (0, None):
                        argv = [group]
            seen.add(tuple(argv))
    assert seen, "the skill mentions no yavih commands?"
    for argv in sorted(seen):
        assert _parse_help(list(argv)) == 0, f"`yavih {' '.join(argv)} --help` does not parse"
    # the nine steps' commands are among them
    for argv in (("doctor",), ("bench", "list"), ("bench", "info"), ("bench", "verify"), ("bridge", "check"),
                 ("plan", "build"), ("plan", "status"), ("features", "validate"), ("eval", "run"),
                 ("report", "build"), ("exec",), ("skill", "install"), ("skill", "show")):
        assert argv in seen, f"skill never mentions `yavih {' '.join(argv)}`"


def test_extract_module_flags_quoted_in_skill_exist():
    """``python -m yavih.extract`` flags named in the skill are real."""
    from yavih.extract import _build_parser

    known = {a for action in _build_parser()._actions for a in action.option_strings}
    text = "\n".join(_skill_texts().values())
    for flag in ("--bench", "--bridge", "--split", "--shard", "--out", "--arg", "--canaries", "--decode-workers",
                 "--deadline", "--expect-profile"):
        assert flag in known and flag in text


def _option_strings(argv: tuple[str, ...]) -> set[str]:
    parser = cli.build_parser()
    for name in argv:
        sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
        parser = sub.choices[name]
    return {o for a in parser._actions for o in a.option_strings}


def test_flags_documented_in_skill_exist_in_cli():
    """CLI flags the skill tells agents to use must exist in the CLI (SKILL/CLI drift fails)."""
    from yavih.extract import _build_parser

    missing = [f"yavih {' '.join(argv)} {f}" for argv, flags in DOCUMENTED_FLAGS.items() for f in flags if f not in _option_strings(argv)]
    known = {a for action in _build_parser()._actions for a in action.option_strings}
    missing += [f"python -m yavih.extract {f}" for f in DOCUMENTED_EXTRACT_FLAGS if f not in known]
    assert not missing, f"SKILL documents flags the CLI lacks: {missing}"


def test_skill_install_and_show(tmp_path, capsys):
    rc = cli.main(["--json", "skill", "install", "--path", str(tmp_path / "yavih")])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and len(out) == 1
    env = json.loads(out[0])
    assert env["status"] == "ok" and not env["warnings"], env
    files = set(env["data"]["files"])
    assert "SKILL.md" in files and {f"references/{r}.md" for r in REFERENCES} <= files
    assert (tmp_path / "yavih" / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD.read_text(encoding="utf-8")

    rc = cli.main(["skill", "show"])
    shown = capsys.readouterr().out
    assert rc == 0 and shown.startswith("---\nname: yavih")


# ------------------------------------------------------------------------------------------- data.md sync


# ----------------------------------------------------------------------------- documentation examples and gotchas


def _yaml_blocks(text: str) -> list[str]:
    return re.findall(r"```yaml\n(.*?)```", text, flags=re.S)


@pytest.mark.parametrize("doc", ["references/slurm.md", "docs/quickstart.md"])
def test_matrix_examples_parse_with_plan_build(tmp_path, doc):
    """The ``matrix.yaml`` examples of the skill (``references/slurm.md`` "matrix.yaml", which SKILL.md points to) and
    docs/quickstart.md are accepted by ``plan build``'s parser."""
    from yavih.plan.matrix import load_matrix

    path = SKILL_ROOT / doc if doc.startswith("references/") else REPO / doc
    if not path.is_file():
        pytest.skip(f"{doc} not available")
    blocks = [b for b in _yaml_blocks(path.read_text(encoding="utf-8")) if "schema_version" in b and "bridges:" in b]
    assert blocks, f"{doc}: no matrix.yaml example found"
    for i, b in enumerate(blocks):
        f = tmp_path / f"matrix{i}.yaml"
        f.write_text(b, encoding="utf-8")
        m = load_matrix(str(f))
        assert m.entries, doc


# ------------------------------------------------------------------------------------------- the complete code list


def _code_constants() -> dict[str, list[str]]:
    """Every ``E_*`` / ``W_*`` / ``N_*`` string literal in the yavih sources (folder tests excluded): ``{code: files}``."""
    pkg = SKILL_ROOT.parents[1]
    out: dict[str, list[str]] = {}
    for path in sorted(pkg.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        for m in re.finditer(r"""["']([EWN]_[A-Z][A-Z0-9_]*[A-Z0-9])["']""", path.read_text(encoding="utf-8")):
            out.setdefault(m.group(1), []).append(str(path.relative_to(pkg)))
    return out


def test_every_code_the_sources_define_is_in_the_cli_reference_codes():
    """``yavih skill show --ref cli`` is the one complete list (docs/cli.md says so): every code literal in the sources
    appears in its "Codes" section."""
    codes = _section((SKILL_ROOT / "references" / "cli.md").read_text(encoding="utf-8"), "Codes")
    constants = _code_constants()
    assert len(constants) > 150
    missing = sorted(c for c in constants if not re.search(rf"(?<![A-Z0-9_]){c}(?![A-Z0-9_])", codes))
    assert not missing, "codes missing from references/cli.md ## Codes: " + ", ".join(f"{c} ({constants[c][0]})" for c in missing)


def test_the_cli_reference_environment_lists_every_variable():
    """references/cli.md "Environment" names every variable of ``yavih.config.ENV_VARS`` (the strict configuration
    refuses any other ``YAVIH_*`` name)."""
    from yavih.config import ENV_VARS

    env = _section((SKILL_ROOT / "references" / "cli.md").read_text(encoding="utf-8"), "Environment")
    missing = [name for name in ENV_VARS if f"`{name}" not in env]
    assert not missing, f"references/cli.md ## Environment lacks {missing}"
    assert "`YAVIH_DATA__<BENCH>__<ROOT>" in env and "`YAVIH_DATASETS__<NAME>`" in env


def _top_level_keys(shape: str) -> set[str]:
    """The keys at depth 1 of a ``{a: {...}, b?, c: [..] | null}`` shape string (``?`` stripped)."""
    keys: set[str] = set()
    depth, cur = 0, ""
    for ch in shape.strip()[1:-1] + ",":
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if depth == 0 and ch == ",":
            keys.add(re.split(r"[:\s]", cur.strip(), maxsplit=1)[0].rstrip("?"))
            cur = ""
        elif depth == 0 or ch not in ",":
            cur += ch
    return {k for k in keys if k}


def test_the_config_show_row_matches_the_envelope_keys(tmp_path, monkeypatch):
    """references/cli.md documents ``config show``'s data with exactly the keys ``Settings.show`` returns (``bench``
    with ``--bench``), and the ``project`` block with exactly its keys."""
    from yavih import config
    from yavih.benchmark import Benchmark

    text = (SKILL_ROOT / "references" / "cli.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| `yavih config show` |"))
    m = re.search(r"\| `(\{project: .*?\}\}\}\})`;", row)
    assert m, "the config show row's data shape"
    shape = m.group(1).replace("\\|", "|")
    (tmp_path / ".git").mkdir()
    (tmp_path / "yavih-bench").mkdir()
    (tmp_path / "yavih-bench" / "yavih.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.delenv("YAVIH_CONFIG", raising=False)
    shown = config.load(refresh=True, cwd=tmp_path).show(Benchmark.load("esc50"))
    config.load(refresh=True)
    assert _top_level_keys(shape) == set(shown)
    project_shape = re.search(r"project: (\{[^}]*\})", shape).group(1)
    assert _top_level_keys(project_shape) == set(shown["project"])


def test_repository_urls_in_the_shipped_skill_resolve():
    """Every ``https://github.com/Bizilizi/yavih/blob/main/<path>`` the shipped skill names is a file of this
    repository (the wheel ships no docs/: the skill links them by URL)."""
    if not (REPO / "docs").is_dir():
        pytest.skip("not a source checkout")
    missing = sorted({m for text in _skill_texts().values()
                      for m in re.findall(r"https://github\.com/Bizilizi/yavih/blob/main/([\w./-]+?)(?:#[\w-]+)?(?=[\s)`'\"]|$)", text)
                      if not (REPO / m).exists()})
    assert not missing, missing


@pytest.mark.parametrize("ref", ["references/cli.md", "docs/cli.md"])
def test_catalog_flags_sit_in_their_command_rows(ref):
    """The flags of the catalog commands are documented in each command's own row of both CLI references."""
    path = SKILL_ROOT / ref if ref.startswith("references/") else REPO / ref
    if not path.is_file():
        pytest.skip(f"{ref} not available")
    lines = path.read_text(encoding="utf-8").splitlines()
    for argv in (("bench", "list"), ("bench", "fetch"), ("bench", "update"), ("suite", "show"), ("plan", "init")):
        row = next((ln for ln in lines if ln.startswith(f"| `yavih {' '.join(argv)}` |")), None)
        assert row is not None, f"{ref}: no row for yavih {' '.join(argv)}"
        for flag in DOCUMENTED_FLAGS[argv]:
            assert flag in row, f"{ref}: yavih {' '.join(argv)} row lacks {flag}"
