"""Cross-area agreements: code constants vs the repository docs (docs/) and the skill references, the reading
vocabulary of every template, the documented budgets vs the yamls."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import yaml

import yavih

PKG = Path(yavih.__file__).resolve().parent
REPO = PKG.parents[1]
DOCS = REPO / "docs"
REFS = PKG / "skill" / "yavih" / "references"
BENCH = PKG.parents[1] / "benchmarks"  # the repository folders (a source checkout)


def _doc(page: str) -> str:
    """Text of the repository docs page ``docs/<page>``."""
    return (DOCS / page).read_text()


# ----------------------------------------------------------------------------- one deadline margin


def test_every_deadline_margin_default_clears_the_template_signal_lead() -> None:
    """``--deadline slurm`` must stop a unit on its own clock BEFORE the templates' ``B:TERM@600``: exec, extract
    and probe run share one 900 s default."""
    from yavih import extract
    from yavih.cli import build_parser
    from yavih.plan import exec as plan_exec
    from yavih.readouts import base

    lead = {int(x) for x in re.findall(r"B:TERM@(\d+)", (REFS / "slurm.md").read_text())}
    assert lead == {600}, lead
    margins = {
        "exec": plan_exec.DEFAULT_DEADLINE_MARGIN_S,
        "exec.resolve_deadline": inspect.signature(plan_exec.resolve_deadline).parameters["margin_s"].default,
        "extract": extract.DEFAULT_DEADLINE_MARGIN_S,
        "extract.parse_deadline": inspect.signature(extract.parse_deadline).parameters["margin"].default,
        "probe (readouts.base.resolve_deadline)": inspect.signature(base.resolve_deadline).parameters["margin"].default,
        "cli exec --deadline-margin": build_parser().parse_args(["exec", "p.jsonl", "--index", "0"]).deadline_margin,
    }
    assert set(margins.values()) == {900.0}, margins
    assert min(margins.values()) > max(lead)
    assert "[--deadline-margin 900]" in _doc("execution.md") and "--deps plan.jsonl" in _doc("execution.md")


# ----------------------------------------------------------------------------- the canary-drift note in the skill


# ----------------------------------------------------------------------------- the fps rule keys are documented


def test_fps_rule_optional_keys_are_documented_where_bridge_authors_read() -> None:
    from yavih.media.rules import FPS_DURATION_FROM, FPS_STARTS, TEMPORAL_PARAMS

    optional = TEMPORAL_PARAMS["fps"][1]
    assert optional == {"edge", "phase", "duration_from"}
    for name, text in (("docs/bridge.md", _doc("bridge.md")), ("references/bridge.md", (REFS / "bridge.md").read_text())):
        row = next(line for line in text.splitlines() if line.startswith("| `fps` |") and "phase" in line)
        for word in sorted(optional | FPS_STARTS | FPS_DURATION_FROM):
            assert word in row, f"{name} fps row does not mention {word!r}: {row}"
        assert "(k + phase)/fps" in row, row


# ----------------------------------------------------------------------------- one reading vocabulary


def _template_readings(path: Path) -> set[str]:
    """Every reading a bridge file can declare: literal ``"reading": "x"`` values and ``ArgSpec(choices=...)``."""
    src = path.read_text()
    out = set(re.findall(r'"reading":\s*"([^"]+)"', src))
    for block in re.findall(r'"reading":\s*ArgSpec\((.*?)\)\s*,\s*\n', src, flags=re.S):
        m = re.search(r"choices=\(([^)]*)\)", block)
        assert m, f"{path}: a reading ArgSpec without choices"
        out |= set(re.findall(r'"([^"]+)"', m.group(1)))
    return out


@pytest.mark.parametrize("path", sorted(BENCH.glob("*/bridge_template.py")) + [PKG / "bridges" / "examples" / "hf_reference.py"],
                         ids=lambda p: p.parent.name if p.name == "bridge_template.py" else p.stem)
def test_every_shipped_bridge_declares_a_v_or_av(path: Path) -> None:
    """The report groups a model's results by the ``reading`` string and ``report --join`` routes ``a``/``v``/``av``:
    any other spelling would split one model over two report rows."""
    readings = _template_readings(path)
    allowed = {"a", "v", "av"}
    if path.name == "bridge_template.py":
        import yaml

        fields = yaml.safe_load((path.parent / "benchmark.yaml").read_text(encoding="utf-8")).get("fields") or {}
        if all((f or {}).get("storage") == "cell" for f in fields.values()):
            allowed = {"t"}  # nothing but text is handed over (a caption): the model reads text
        oc = yaml.safe_load((path.parent / "benchmark.yaml").read_text(encoding="utf-8")).get("output_contract") or {}
        if oc.get("shape") == "sequence":  # a streaming benchmark: its real-time readings carry the -rt suffix
            allowed = allowed | {f"{r}-rt" for r in allowed}
    assert readings and readings <= allowed, f"{path}: {readings}"


# ----------------------------------------------------------------------------- `extract --out` is the run dir


# ----------------------------------------------------------------------------- the live-probe fallback


# ----------------------------------------------------------------------------- clip_acc == seg_acc on AVE


# ----------------------------------------------------------------------------- documented budgets == the yamls


@pytest.mark.parametrize("folder", sorted(p.parent.name for p in BENCH.glob("*/setup.md")))
def test_setup_md_probe_budget_matches_the_yaml(folder: str) -> None:
    """A setup.md that quotes ``probe_minutes_ref`` quotes the yaml's value."""
    text = (BENCH / folder / "setup.md").read_text()
    ref = (yaml.safe_load((BENCH / folder / "benchmark.yaml").read_text()).get("budget") or {}).get("probe_minutes_ref") or {}
    quoted = re.findall(r"probe_minutes_ref\.(\w+) (\d+(?:\.\d+)?)", text)
    quoted += [(r, v) for block in re.findall(r"probe_minutes_ref \{([^}]*)\}", text) for r, v in re.findall(r"(\w+): (\d+(?:\.\d+)?)", block)]
    for readout, value in quoted:
        assert float(ref.get(readout, -1)) == float(value), f"{folder}/setup.md says probe_minutes_ref.{readout} {value}, yaml {ref}"


# ----------------------------------------------------------------------------- exec, plan status and doctor in docs/


def test_docs_document_the_exec_status_and_doctor_fields() -> None:
    cli_md = _doc("cli.md")
    exec_row = next(line for line in cli_md.splitlines() if line.startswith("| `yavih exec` |"))
    assert "--deps plan.jsonl" in exec_row and "--deadline-margin 900" in exec_row
    from yavih.plan.status import STATUS_KEYS

    status_doc = _doc("execution.md")
    for key in (*STATUS_KEYS, "stale_running", "by_kind"):
        assert key in status_doc, key
    doctor_row = next(line for line in cli_md.splitlines() if line.startswith("| `yavih doctor` |"))
    assert "tools:" in doctor_row and "W_TOOL_PIN" in doctor_row and "W_MEDIA_READ" in doctor_row
