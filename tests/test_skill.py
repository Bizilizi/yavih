"""The agent skill: ``yavih/skill/yavih/SKILL.md`` (≤ 250 lines, nine steps with stop conditions, setup.md
as the first read per benchmark, the guardrails as a rule → checking-code table, "When blocked", the operational rules) and its
``references/*.md``; every ``yavih <group> [<cmd>]`` the skill mentions parses with the real argparse tree;
``references/data.md`` stays in sync with the frozen ``benchmark.yaml``s; ``yavih skill install`` / ``show``
serve the package data."""

from __future__ import annotations

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
STEPS = ("install", "setup.md", "prepare", "bridge", "smoke", "extract", "validate", "probe", "report")
GROUPS = ("doctor", "skill", "bench", "bridge", "extract", "features", "probe", "report", "plan", "exec", "suite")
# the operational rules that follow the guardrail checklist, carried verbatim (whitespace-normalized)
OPERATIONAL_RULES = (
    "Install with `--no-deps`; never let pip touch the model env's torch; install original-author packages "
    "independently, exactly as setup.md says.",
    "Read setup.md before anything else for every benchmark; the input and output contracts there are what the "
    "bridge must comply with.",
    "Default dtype float16; switch to float32 when `bridge check` warns about range.",
    "Many small shards, never one unit per GPU; resubmit from `plan status`.",
    "`validate` before `probe`; read the floor before the metric; a CI overlapping the floor, `floor_violated` or "
    "`audio_blind` is not a result.",
    "Write all readings (audio, video, av) of an audio-visual model; never present a joint number alone.",
    "One model, one `model` name in every bridge; two configurations you want side by side (another layer, another "
    "checkpoint) are two names. Numbers compare only at equal `eval_hash`; the bridge's configuration is recorded, "
    "not compared.",
    "Opting out of the rule vocabulary is fine when it does not fit: declare `{\"rule\": \"custom\", \"description\": "
    "\"<one line>\"}`; validate notes it (`N_RULE_CUSTOM`) and nothing is greyed.",
    "Evaluate a comparison on one cluster; cross-hardware probe noise is ~1 pt.",
)
# CLI flags the skill documents
DOCUMENTED_FLAGS = {
    ("bench", "info"): ("--setup",),
    ("bench", "verify"): ("--census",),
    ("probe", "run"): ("--log-samples", "--results", "--no-results"),
    ("report", "build"): ("--results", "--merge-results"),
}
DOCUMENTED_EXTRACT_FLAGS = ("--read-ahead",)

_CMD_RE = re.compile(r"`?\byavih\s+(?:--json\s+)?(" + "|".join(GROUPS) + r")\b(?:\s+([a-z]+))?")


def _norm(text: str) -> str:
    return " ".join(text.split())


def _skill_texts() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in [SKILL_MD, *(SKILL_ROOT / "references" / f"{r}.md" for r in REFERENCES)]}


def _section(text: str, heading: str) -> str:
    """Body of the ``## heading…`` section (up to the next ``## ``)."""
    m = re.search(rf"^## {re.escape(heading)}[^\n]*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    assert m, f"section '## {heading}' missing"
    return m.group(1)


# ------------------------------------------------------------------------------------------- SKILL.md shape


def test_skill_md_exists_and_is_short():
    assert SKILL_MD.is_file(), f"{SKILL_MD} must be packaged"
    lines = SKILL_MD.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 250, f"SKILL.md has {len(lines)} lines; the cap is 250"


def test_frontmatter_name_and_description():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    head = text.split("---", 2)[1]
    assert re.search(r"^name:\s*yavih\s*$", head, re.M)
    assert "description:" in head
    for trigger in ("yavih", "bridge", "frozen", "setup.md", "probing", "probe", "benchmark suite", "benchmark my model",
                    "video", "audio-visual", "`yavih bench list`"):
        assert trigger in head, f"description must trigger on {trigger!r}"


def test_frontmatter_enumerates_no_benchmark():
    """The benchmark set grows by adding folders: the description triggers on `yavih bench list`, never on a list of
    names that would go stale (no packaged name, alias or header display name)."""
    import yaml

    from yavih.benchmarks import packaged_folder, packaged_names

    head = SKILL_MD.read_text(encoding="utf-8").split("---", 2)[1]
    for name in packaged_names():
        y = yaml.safe_load((packaged_folder(name) / "benchmark.yaml").read_text(encoding="utf-8"))
        for word in (name, *(y.get("aliases") or []), (y.get("header") or {}).get("name") or name):
            assert not re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", head, flags=re.I), \
                f"the SKILL.md description names benchmark {word!r}"


def test_nine_steps_present_in_order():
    text = SKILL_MD.read_text(encoding="utf-8")
    pos = []
    for i, step in enumerate(STEPS, start=1):
        m = re.search(rf"^\|\s*{i}\s+{re.escape(step)}\s*\|", text, flags=re.M)
        assert m, f"step row '{i} {step}' missing from the journey table"
        pos.append(m.start())
    assert pos == sorted(pos)


def test_step_rows_carry_commands_and_stop_conditions():
    """Step table: each step names its command and what must be seen before moving on."""
    text = SKILL_MD.read_text(encoding="utf-8")
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\|\s*\d\s+(\S+)\s*\|[^\n]*$", text, flags=re.M)}
    need = {
        "install": ("--no-deps", "yavih doctor --bench", "untouched", "external"),
        "setup.md": ("yavih bench info <b> --setup", "FIRST", "Dependencies", "Input contract", "Output contract"),
        "prepare": ("yavih bench verify <b> --census", "95 %", "labels_ok", "derived_sha_ok"),
        "bridge": ("bridge_template.py", "preprocessing source", "`model`", "`reading`", "custom", "one-line description",
                   "views: 1", "precision", "(feats, pos, used)", "media coordinates"),
        "smoke": ("yavih bridge check", "bit-identical", "1e-5", "1e-3", "E_RULE_MISMATCH", "E_TIMES_OUT_OF_MEDIA", "E_NETWORK"),
        "extract": ("yavih plan build", "yavih plan status", "pending == 0", "preempted == 0", "failed == 0"),
        "validate": ("yavih features validate runs/<b>/<model>/<reading>/features", "coverage", "N_RULE_CUSTOM",
                     "W_RULE_UNCHECKED", "not_applicable"),
        "probe": ("yavih probe run runs/<b>/<model>/<reading>", "--log-samples", "floor", "floor_violated", "audio_blind",
                  "external"),
        "report": ("yavih report build", "results/<model>/results_<timestamp>.json", "`reading`", "`eval_hash`", "ci95",
                   "floor", "W_BRIDGE_CHANGED"),
    }
    for step, needles in need.items():
        row = rows[step]
        for n in needles:
            assert n in row, f"step '{step}': {n!r} missing from its row"


def test_setup_md_is_the_first_read():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "setup.md" in text.split("## 0. Install", 1)[0], "the intro must point at setup.md before install"
    assert "## The two contracts" in text
    contracts = _section(text, "The two contracts")
    for n in ("Input contract", "Output contract", "input_contract.handed_over", "input_contract.facts",
              "output_contract.requires", "media coordinates", "full-frame"):
        assert n in contracts, n


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


def test_guardrails_table_names_the_checking_code_or_says_advisory():
    """Each guardrail (docs/design.md) is a row saying which code checks it (a real
    ``yavih.…`` function) or that it is advisory; nothing enforced is presented as advice and vice versa."""
    section = _section(SKILL_MD.read_text(encoding="utf-8"), "Guardrails")
    rows = [ln for ln in section.splitlines() if ln.startswith("| ") and not ln.startswith("| rule") and not ln.startswith("|---")]
    assert len(rows) >= 15
    names = set()
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        assert len(cells) == 3, row
        found = re.findall(r"`(yavih\.[A-Za-z0-9_.]+)`", cells[1])
        assert found or "advisory" in cells[1] or "by construction" in cells[1], f"row names no checker: {row}"
        names.update(found)
    for name in sorted(names):
        assert callable(_resolve(name)), name
    text = _norm(section)
    for rule in ("instance-wise", "batch statistics", "padding to the batch maximum", "pretraining recipe", "one deterministic view",
                 "no cross-sample state", "never manifest, labels or split", "inference_mode", "fixed `N` and `D`",
                 "media coordinates", "no network", "outside the run directory", "preprocessing source", "line by line",
                 "legitimate opt-out", "E_TEMPLATE_UNFILLED", "E_CONSTANT_FEATURES"):
        assert rule in text, f"Guardrails table: {rule!r} missing"
    advisory = [r for r in rows if "advisory" in r]
    assert any("pretraining recipe" in r for r in advisory) and any("line by line" in r for r in advisory)


def test_when_blocked_and_shared_bridge_code():
    """A blocked benchmark is recorded and skipped with a to-do list; a model's bridges share a
    ``<model>_common.py`` (its directory is on sys.path under bridge check and extraction)."""
    skill = SKILL_MD.read_text(encoding="utf-8")
    blocked = _norm(_section(skill, "When blocked"))
    for n in ("yavih bench list", "obtain", "manual-download", "needs-licence", "needs-data-owner", "site-only", "not_ready", "Continue with the other benchmarks",
              "--synthetic", "to-do list for the user"):
        assert n in blocked, n
    assert "<model>_common.py" in skill and "sys.path" in skill
    bridge = (SKILL_ROOT / "references" / "bridge.md").read_text(encoding="utf-8")
    assert "<model>_common.py" in bridge and "sys.path" in bridge and "class " in _section(bridge, "7. Many benchmarks, one model")


def test_shipped_skill_cites_nothing_an_installed_user_cannot_open():
    """The repository docs/ pages and notes/ are not in the wheel: the skill names them by GitHub URL only."""
    from yavih.lint import UNOPENABLE

    for name, text in _skill_texts().items():
        for line in text.splitlines():
            if "github.com/Bizilizi/yavih" not in line:
                assert not UNOPENABLE.search(line), f"{name}: {line.strip()[:140]!r}"

def test_contributing_hook_links_the_doc():
    """SKILL.md "Contributing a benchmark": at most six lines naming the clone, ``yavih bench new``, the contributing
    guide and the required trial run, both by GitHub URL (the wheel ships no docs/), and both pages exist."""
    section = _section(SKILL_MD.read_text(encoding="utf-8"), "Contributing a benchmark")
    lines = [ln for ln in section.splitlines() if ln.strip()]
    assert 1 <= len(lines) <= 6, lines
    text = _norm(section)
    url = "https://github.com/Bizilizi/yavih/blob/main/"
    for page in ("docs/contributing-a-benchmark.md", "docs/trial-run.md"):
        assert url + page in text, page
        assert (REPO / page).is_file() or not (REPO / "docs").is_dir(), page
    for n in ("https://github.com/Bizilizi/yavih ", "yavih bench new", "required"):
        assert n in text, n
    assert "## Contributing a benchmark" in SKILL_MD.read_text(encoding="utf-8").split("## References", 1)[0]


def test_exploring_a_benchmark_block_runs():
    """Every command of the "Exploring a benchmark" block works on a packaged benchmark (no checkout needed)."""
    skill = SKILL_MD.read_text(encoding="utf-8")
    block = skill[skill.index("Exploring a benchmark:"):]
    block = block[block.index("```bash") + 7: block.index("```", block.index("```bash") + 7)]
    for cmd in ("suite list", "bench list", "bench files <b>", "bench show <b> setup", "bench show <b> readme|yaml|probe",
                "bench show <b> prepare/<file>", "bench prepare <b> [<step>]", "bench path <b> [<file>]", "skill show --ref"):
        assert f"yavih {cmd}" in block, cmd
    for argv in (["suite", "list"], ["bench", "list"], ["bench", "files", "esc50"], ["bench", "show", "esc50", "setup"],
                 ["bench", "show", "esc50", "readme"], ["bench", "show", "esc50", "yaml"], ["bench", "show", "avsync", "probe"],
                 ["bench", "show", "esc50", "prepare/01_download.sh"], ["bench", "prepare", "esc50"],
                 ["bench", "path", "esc50", "prepare/01_download.sh"], ["skill", "show", "--ref", "bridge"]):
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli.main(["--json", *argv]) == 0, argv


def test_operational_rules_verbatim():
    section = _norm(_section(SKILL_MD.read_text(encoding="utf-8"), "Operational rules"))
    for rule in OPERATIONAL_RULES:
        assert _norm(rule) in section, f"operational rule missing or reworded: {rule!r}"
    raw = _section(SKILL_MD.read_text(encoding="utf-8"), "Operational rules")
    assert raw.count("\n- ") + raw.startswith("- ") == len(OPERATIONAL_RULES)


def test_skill_says_opting_out_of_rules_is_fine():
    """bridge.md "Opting out" — a custom rule with a one-line description is fine when the vocabulary
    does not fit (N_RULE_CUSTOM, a note; out-of-media and position checks still apply); SKILL.md says it in a rule
    and in the bridge step, and its guardrails call it a legitimate opt-out."""
    bridge = (SKILL_ROOT / "references" / "bridge.md").read_text(encoding="utf-8")
    opt = _norm(re.search(r"^### Opting out[^\n]*\n(.*?)(?=^#)", bridge, flags=re.M | re.S).group(1))
    for n in ('{"rule": "custom", "description": "<one line', "N_RULE_CUSTOM", "fine", "never greys",
              "E_TIMES_OUT_OF_MEDIA", "E_POS_RANGE", "E_INPUT"):
        assert n in opt, f"bridge.md Opting out: {n!r} missing"
    row = next(ln for ln in bridge.splitlines() if ln.startswith("| `custom` |"))
    assert "OPT-OUT" in row and "N_RULE_CUSTOM" in row and "W_RULE_UNCHECKED" not in row, row
    skill = SKILL_MD.read_text(encoding="utf-8")
    assert "legitimate opt-out" in _norm(_section(skill, "Guardrails"))
    cli_md = (SKILL_ROOT / "references" / "cli.md").read_text(encoding="utf-8")
    assert "N_RULE_CUSTOM" in cli_md and "never a warning" in _norm(cli_md)


def test_bridge_md_documents_model_and_reading():
    """``model`` is the name results are filed under (one per model; a layer sweep = several
    names), ``reading`` names the run subdirectory and the results key; ``describe()`` is recorded, never identity."""
    text = (SKILL_ROOT / "references" / "bridge.md").read_text(encoding="utf-8")
    row = next(ln for ln in text.splitlines() if ln.startswith("| `model` |"))
    for n in ("[A-Za-z0-9_.-]{1,64}", "runs/<b>/<model>/<reading>/", "results/<model>/", "two names", "W_BRIDGE_CHANGED"):
        assert n in row, f"bridge.md `model` row: {n!r} missing"
    assert "| key | in `describe_sha256` | content |" in text and "| key | identity |" not in text
    for n in ("describe_sha256", "E_CONFIG_DRIFT", "--expect-profile", "W_WEIGHTS_UNFINGERPRINTED", "`notes`"):
        assert n in text, f"bridge.md: {n!r} missing"


# ------------------------------------------------------------------------------------------- references


def test_reference_files_exist_and_nonempty():
    for ref in REFERENCES:
        p = SKILL_ROOT / "references" / f"{ref}.md"
        assert p.is_file(), f"references/{ref}.md missing"
        assert len(p.read_text(encoding="utf-8")) > 500


def test_bridge_md_covers_the_contract():
    """bridge.md: raw input, media coordinates, describe(), used, invariance, worked examples."""
    text = (SKILL_ROOT / "references" / "bridge.md").read_text(encoding="utf-8")
    for n in ("RawStream", "Sample", "BenchmarkView", "def bytes", "def open", "meta[\"facts\"]",
              "media coordinates", "ORIGINAL frame", "describe()", "\"input\"", "temporal", "spatial", "front_end",
              "norm", "decoder", "views", "batch_tolerance", "used", "grid_pos", "t_centres", "region", "crop_region",
              "per_sample", "MediaUndecodable", "NotApplicable", "E_RERUN_DRIFT", "E_BATCH_INVARIANCE",
              "E_RULE_MISMATCH", "E_TIMES_OUT_OF_MEDIA", "yavih.media.decode", "invariance", "profile.json"):
        assert n in text, f"bridge.md: {n!r} missing"
    # the closed temporal vocabulary (docs/bridge.md), one row each
    from yavih.media.rules import TEMPORAL_RULES

    for rule in TEMPORAL_RULES:
        assert re.search(rf"^\| `{rule}` \|", text, flags=re.M), f"bridge.md: temporal rule {rule!r} has no row"
    # one worked example per media kind
    examples = _section(text, "5. Worked examples")
    for kind in ("mp4", "wav", "jpg / png", "audio-visual"):
        assert kind in examples, f"bridge.md worked examples: {kind!r} missing"


def test_readouts_md_covers_readouts_and_results():
    text = (SKILL_ROOT / "references" / "readouts.md").read_text(encoding="utf-8")
    for n in ("media coordinates", "T = 10", "t_bins", "min_cell_coverage", "target region", "0.25", "full-frame",
              "external", "E_EXTERNAL_MISSING", "E_EXTERNAL_VERSION", "vggsounder==1.0", "result.json",
              "\"schema_version\": 3", "\"eval_hash\"", "\"model\"", "\"reading\"", "results_<timestamp>.json",
              "samples_<benchmark>_<reading>_<timestamp>.jsonl", "<benchmark>:<reading>", "\"results_version\": 2",
              "(benchmark, reading, eval_hash)", "W_BRIDGE_CHANGED", "W_FEATURES_DIFFER", "higher_is_better",
              "--merge-results", "floor_of", "not_applicable"):
        assert n in text, f"readouts.md: {n!r} missing"


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


def test_slurm_md_templates_are_generic():
    """slurm.md has Templates A, B and C, each a batch script whose site-specific values are placeholders
    (``<partition>``, ``<account>``, ``$PROJECT``, ``/path/to/…``) for the reader to replace."""
    text = (SKILL_ROOT / "references" / "slurm.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## Template ([A-Z])\b", text, flags=re.M)
    assert headings == ["A", "B", "C"], headings
    for letter in headings:
        block = re.search(r"```bash\n(.*?)```", _section(text, f"Template {letter}"), flags=re.S)
        assert block, f"Template {letter} has no bash block"
        script = block.group(1)
        assert "#SBATCH --partition=<partition>" in script, f"Template {letter}: partition is not a placeholder"
        assert 'cd "$PROJECT"' in script, f"Template {letter}: project directory is not a placeholder"
        for line in script.splitlines():
            m = re.match(r"#+SBATCH --(account|qos)=(\S+)", line)
            assert not m or m.group(2).startswith("<"), f"Template {letter}: {line.strip()!r}"
            for path in re.findall(r"(?<![\w$}])/[A-Za-z][\w.-]*(?:/[\w.${}-]*)*", line.split("#", 1)[0]):
                assert path.startswith(("/path/to/", "/dev/", "/tmp")), f"Template {letter}: site path {path!r}"


# ------------------------------------------------------------------------------------------- CLI agreement


def _parse_help(argv: list[str]) -> int:
    parser = cli.build_parser()
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([*argv, "--help"])
    return int(exc.value.code or 0)


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
                 ("plan", "build"), ("plan", "status"), ("features", "validate"), ("probe", "run"),
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


def test_data_md_rows_match_the_frozen_benchmark_yaml():
    """``references/data.md`` section 1 has one row per packaged folder and quotes that folder's frozen
    ``budget.media_gb``, per-split row counts (``budget.clips``, which ``tests/test_packaged_folders.py`` pins to
    the manifest), readout and aliases. Reads the yaml directly so it does not depend on the loader."""
    import yaml

    from yavih.benchmarks import packaged_folder, packaged_names

    text = (SKILL_ROOT / "references" / "data.md").read_text(encoding="utf-8")
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| ([a-z0-9_]+)\b[^\n]*$", text, flags=re.M)}
    names = packaged_names()
    assert names and set(names) <= set(rows), sorted(set(names) - set(rows))
    for name in names:
        y = yaml.safe_load((packaged_folder(name) / "benchmark.yaml").read_text(encoding="utf-8"))
        row = rows[name]
        assert isinstance(y["budget"]["media_gb"], (int, float)), f"{name}: budget.media_gb is not a number yet"
        assert f"{y['budget']['media_gb']:g} GB = `budget.media_gb`" in row, (name, y["budget"]["media_gb"])
        assert row.split("|")[2].strip().startswith(y["readout"]), (name, y["readout"])
        for split, n_rows in y["budget"]["clips"].items():
            n = f"{n_rows:,}".replace(",", " ")
            assert n in row, (name, split, n)
        for alias in y.get("aliases") or []:
            assert f"`{alias}`" in row, (name, alias)


def test_data_md_points_to_every_setup_md_and_prepare_time_transforms():
    from yavih.benchmarks import packaged_names

    text = (SKILL_ROOT / "references" / "data.md").read_text(encoding="utf-8")
    setup = _section(text, "5. Per-benchmark `setup.md`")
    assert "yavih bench info <b> --setup" in setup
    for name in packaged_names():
        assert re.search(rf"\b{name}\b", setup), f"data.md section 5 does not point to {name}'s setup.md"
    transforms = _section(text, "4. Prepare-time transforms")
    for n in ("avsync", "recipe", "ALAC", "nyuv2_depth", "PNG"):
        assert n in transforms, n


# ----------------------------------------------------------------------------- documentation examples and gotchas


def _yaml_blocks(text: str) -> list[str]:
    return re.findall(r"```yaml\n(.*?)```", text, flags=re.S)


@pytest.mark.parametrize("doc", ["references/slurm.md", "README.md"])
def test_matrix_examples_parse_with_plan_build(tmp_path, doc):
    """The ``matrix.yaml`` examples of the skill (``references/slurm.md`` "matrix.yaml", which SKILL.md points to) and
    README.md are accepted by ``plan build``'s parser."""
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


_NORM_GOTCHA_FOLDERS = ("esc50", "fsd50k", "vggsounder", "ave", "audioset20k")


@pytest.mark.parametrize("name", _NORM_GOTCHA_FOLDERS)
def test_normalization_gotchas_allow_the_models_own_per_clip_recipe(name):
    """A model pretrained with per-clip standardisation over its padded buffer keeps it; the gotchas say so and
    point to ``norm``."""
    from yavih.benchmarks import packaged_folder

    gotchas = _norm(_section((packaged_folder(name) / "setup.md").read_text(encoding="utf-8"), "Gotchas"))
    low = gotchas.lower()
    assert "pretrain" in low and ("per clip" in low or "per-clip" in low), name
    assert re.search(r"(own pretraining recipe|pretrained with per-clip|pretrained with them|model's own pretraining)", gotchas,
                     flags=re.I), name
    assert "`norm`" in gotchas, name


def test_in1k_image_as_video_gotcha_says_top_level():
    """``image_as_video`` is a top-level ``describe()`` key (``input.image`` refuses it)."""
    from yavih.benchmarks import packaged_folder

    gotchas = _norm(_section((packaged_folder("in1k") / "setup.md").read_text(encoding="utf-8"), "Gotchas"))
    assert "TOP-LEVEL `describe()` key" in gotchas and "not inside `input.image`" in gotchas
    template = (packaged_folder("in1k") / "bridge_template.py").read_text(encoding="utf-8")
    assert re.search(r'^ {12}"image_as_video": \{', template, flags=re.M)  # a sibling of "input" in describe()


def test_readme_example_reading_uses_the_benchmarks_vocabulary():
    """The README's vggsounder bridge declares ``reading`` in vggsounder's {a, v, av}."""
    from yavih.benchmarks import packaged_folder

    readme = REPO / "README.md"
    if not readme.is_file():
        pytest.skip("README.md not available")
    m = re.search(r'"reading": "([a-z]+)"', readme.read_text(encoding="utf-8"))
    assert m and m.group(1) == "v"
    assert "`describe()[\"reading\"]` ∈ {`a`, `v`, `av`}" in (packaged_folder("vggsounder") / "setup.md").read_text(encoding="utf-8")
