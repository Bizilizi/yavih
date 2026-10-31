"""Cross-folder invariants of the shipped benchmark folders.

The folder contract itself lives in ``yavih.lint`` (``yavih bench lint``): this module runs
every lint check on every packaged folder (``packaged_names()``, so a new folder is covered without editing this
file), pins the repo-level facts no single folder owns (aliases are unique and resolve, the generated index), and
proves that the lint catches what it claims on a mutated copy. Each folder's own ``tests/test_folder.py`` checks
that folder's facts in depth. Offline; no media needed."""

from __future__ import annotations

import json
import re
import shutil

import pytest
import yaml

from yavih import lint
from yavih.benchmark import Benchmark
from yavih.benchmarks import packaged_folder, packaged_names, packaged_root

NAMES = packaged_names()
CHECK_IDS = [cid for cid, _ in lint.CHECKS]
INDEX_COLUMNS = ["description", "readout", "media", "source"]  # first column: "Benchmark" (task) or "name"
SOURCE_WORDS = {"public_url": "public", "torrent": "torrent", "manual_portal": "manual portal",
                "private_mirror": "private mirror", "derived": "derived"}
_lines_outside_fences = lint.lines_outside_fences


@pytest.fixture(autouse=True)
def _no_site_config(monkeypatch):
    monkeypatch.setenv("YAVIH_CONFIG", "/nonexistent/yavih-config.toml")


def _yaml(name: str) -> dict:
    return yaml.safe_load((packaged_folder(name) / "benchmark.yaml").read_text(encoding="utf-8"))


def test_packaged_set_is_discovered():
    """Every packaged folder is a directory holding benchmark.yaml; the list is never silently empty."""
    assert NAMES and NAMES == sorted(NAMES)
    assert {"esc50", "avsync", "vggsounder"} <= set(NAMES)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("check", CHECK_IDS)
def test_folder_passes_lint_check(name, check):
    """One lint check on one packaged folder: no errors (warnings such as ``W_TODO`` are allowed)."""
    rep = lint.lint_folder(packaged_folder(name), checks=[check])
    assert rep.ok, "\n".join(f"{name}/{f.file}: {f.code}: {f.message}" for f in rep.errors)


def test_aliases_are_unique_and_resolve():
    """Aliases are declared in each folder's yaml, resolve to that folder, are not hashed, and never collide."""
    declared: dict[str, str] = {}
    for name in NAMES:
        for alias in Benchmark.load(name).spec.aliases:
            assert alias not in declared and alias not in NAMES, f"alias {alias!r} of {name} collides"
            declared[alias] = name
    assert "imagenet1k" in declared  # never silently empty
    for alias, name in declared.items():
        b = Benchmark.load(alias)
        assert b.name == name and alias in b.spec.aliases
        assert b.protocol_hash == Benchmark.load(name).protocol_hash  # aliases are not hashed


def test_pinned_revision_folders_are_known():
    """audioset20k (the published Hugging Face release) pins its revision; the lint's setup check enforces that a
    pinned ``source.revision`` appears in "## Obtain the data"."""
    assert [n for n in NAMES if (_yaml(n).get("source") or {}).get("revision")]


# ===================================================================== the index


def _index_groups() -> list[tuple[str, int, list[list[str]]]]:
    """``(capability, count in the heading, rows)`` per ``### <capability> (<n>)`` table of the index."""
    text = (packaged_root() / "README.md").read_text(encoding="utf-8")
    lines = [line for _, line in _lines_outside_fences(text)]
    groups: list[tuple[str, int, list[list[str]]]] = []
    heading: tuple[str, int] | None = None
    for k, line in enumerate(lines):
        m = re.fullmatch(r"### ([a-z-]+) \((\d+)\)", line.strip())
        if m:
            heading = (m.group(1), int(m.group(2)))
            continue
        cells = [c.strip().lower() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and cells[0] in ("benchmark", "name") and cells[1:] == INDEX_COLUMNS:
            assert heading is not None, "an index table must sit under a `### <capability> (<n>)` heading"
            rows = []
            for row in lines[k + 2:]:
                if not row.lstrip().startswith("|"):
                    break
                rows.append([c.strip() for c in row.strip().strip("|").split("|")])
            groups.append((*heading, rows))
            heading = None
    assert groups, "benchmarks/README.md has no | Benchmark | Description | Readout | Media | Source | table"
    return groups


def test_index_table_lists_every_packaged_folder():
    """Grouped by capability (vocabulary order): each packaged folder sits, sorted by name, under every capability
    it declares; links resolve, readout and source agree."""
    from yavih.suites import CAPABILITIES

    groups = _index_groups()
    caps = [cap for cap, _n, _rows in groups]
    assert caps == [c for c in CAPABILITIES if c in caps], f"index groups {caps} are not in CAPABILITIES order"
    seen: dict[str, list[str]] = {}
    for cap, count, rows in groups:
        names = []
        for cells in rows:
            assert len(cells) == 1 + len(INDEX_COLUMNS), cells
            m = re.fullmatch(r"\[([a-z0-9_]+)\]\(([^)]+)\)", cells[0])
            assert m and m.group(2) == f"{m.group(1)}/README.md", f"index row must link the folder README: {cells[0]!r}"
            name = m.group(1)
            names.append(name)
            assert packaged_folder(name) is not None, f"index lists {name!r}, which is not a packaged folder"
            y = _yaml(name)
            assert cap in y["capability"], f"index lists {name} under {cap!r}; it declares {y['capability']}"
            assert cells[2] == y["scorer"], f"index: {name} scorer {cells[2]!r} != {y['scorer']!r}"
            assert SOURCE_WORDS[y["source"]["kind"]] in cells[4].lower(), f"index: {name} source {cells[4]!r} vs {y['source']['kind']}"
            assert cells[1] and cells[3], f"index: {name} needs a description and the media handed over"
            for alias in y.get("aliases") or []:
                assert f"`{alias}`" in cells[1], f"index: {name}'s description must name alias {alias!r}"
            seen.setdefault(name, []).append(cap)
        assert names == sorted(names) and len(names) == count, f"index group {cap}: {names} (heading says {count})"
    assert sorted(seen) == packaged_names(), f"index lists {sorted(seen)} != packaged folders {packaged_names()}"
    for name, caps_of in seen.items():
        assert caps_of == [c for c in CAPABILITIES if c in _yaml(name)["capability"]], f"{name} is under {caps_of}"


def test_index_links_resolve():
    text = (packaged_root() / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)#\s]+)\)", text)
    assert links
    for link in links:
        if "://" in link:
            continue
        assert (packaged_root() / link).is_file(), f"benchmarks/README.md links {link}, which does not exist"
    for name in packaged_names():
        assert f"({name}/setup.md)" in text, f"benchmarks/README.md must point to {name}/setup.md"



def test_generated_benchmark_tables_are_current():
    """The benchmark tables of the repo README, the index and the skill's data.md are generated from the folders
    (``python -m yavih.doctables --write``); a committed block that differs is stale."""
    from yavih import doctables

    docs = doctables.repo_documents()
    assert {"data.md", "readouts.md"} <= {d.name for d in docs} and any(d.parent.name == "benchmarks" for d in docs)
    if (packaged_root().parents[2] / "pyproject.toml").is_file():  # a source checkout: the repo docs too
        assert {"README.md", "benchmark-folder.md"} <= {d.name for d in docs if d.parent.name != "benchmarks"}
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        assert "<!-- yavih:bench-table" in text, f"{doc}: no generated block"
        assert doctables.replace_blocks(text) == text, f"{doc}: generated table is stale; run python -m yavih.doctables --write"


def test_doctables_render_any_folder_set(tmp_path):
    from yavih import doctables

    folders = {"esc50": packaged_folder("esc50")}
    assert "2 000 in folds 1–5 (400 each)" in doctables.render("readme", folders)
    text = "x\n<!-- yavih:bench-table names begin -->\nold\n<!-- yavih:bench-table names end -->\ny\n"
    assert doctables.replace_blocks(text, folders) == "x\n<!-- yavih:bench-table names begin -->\n`esc50`.\n<!-- yavih:bench-table names end -->\ny\n"
    with pytest.raises(ValueError):
        doctables.replace_blocks("<!-- yavih:bench-table names begin -->\n", folders)


def test_doctables_lists_are_derived_from_the_folders():
    """The lists beside the tables (folder facts, readout groups, ``prepare_only`` roots, voiding controls) follow the
    folder set they are rendered for: a new folder needs no hand-written line."""
    from yavih import doctables

    pick = {n: packaged_folder(n) for n in ("avsync", "esc50", "in1k", "nyuv2_depth", "vggss_box")}
    folder = doctables.render("folder", pick).splitlines()
    assert folder[0] == "| benchmark | scorer | label kind | streams | headline | eval.py |" and len(folder) == 2 + len(pick)
    rows = {r.split(" | ")[0].strip("| "): r for r in folder[2:]}
    assert "| clf (5 folds) | single | audio | `acc_mean` |" in rows["esc50"]
    assert rows["in1k (aliases `imagenet1k`, `imagenet`)"].startswith("| in1k (aliases")
    assert "video, audio (materialized)" in rows["avsync"] and rows["avsync"].endswith("| yes |")
    assert "| none (targets) | image | `d1` | yes, `external` |" in rows["nyuv2_depth"]
    assert doctables.render("readout_groups", pick) == ("Packaged benchmarks per readout: `clf` avsync, esc50, in1k · "
                                                        "`grid` nyuv2_depth · `box` vggss_box.")
    assert doctables.render("prepare_roots", pick) == ("Roots marked `prepare_only`: avsync `media_root` and "
                                                       "`curation_root`; nyuv2_depth `geonet_root`; vggss_box "
                                                       "`annotations_root`.")
    assert doctables.render("prepare_roots", {"esc50": pick["esc50"]}) == "No root is marked `prepare_only`."
    assert doctables.render("controls", pick) == ("- `avsync` and `vggss_box` emit controls (`floor_violated`, "
                                                  "`audio_blind`); the report greys out the rows they void.")
    assert doctables.render("controls", {"vggss_box": pick["vggss_box"]}).startswith("- `vggss_box` emits controls (`audio_blind`)")


# ===================================================================== obtain


@pytest.mark.parametrize("name", NAMES)
def test_obtain_tier_is_declared_and_shown(name, capsys):
    """5.3: every folder says who must act before its data can be staged; ``bench list`` / ``bench info`` show it."""
    from yavih import cli

    y = _yaml(name)
    assert set(y["obtain"]) == {"tier", "how"} and y["obtain"]["tier"] in lint.OBTAIN_TIERS and y["obtain"]["how"].strip()
    b = Benchmark.load(name)
    assert b.spec.obtain is not None and b.spec.obtain.tier == y["obtain"]["tier"]
    assert cli.main(["--json", "bench", "info", name]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["obtain"] == y["obtain"]


def test_obtain_is_optional_and_outside_the_protocol_hash(tmp_path):
    """5.3: ``obtain`` is provenance for people and agents: editing or dropping it changes no protocol_hash."""
    import shutil

    src = packaged_folder("esc50")
    want = Benchmark.load("esc50").protocol_hash
    for variant in ("edited", "dropped"):
        dst = tmp_path / variant / "esc50"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        y = yaml.safe_load((dst / "benchmark.yaml").read_text(encoding="utf-8"))
        if variant == "edited":
            y["obtain"] = {"tier": "site-only", "how": "edited"}
        else:
            del y["obtain"]
        (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
        b = Benchmark.load(str(dst))
        assert b.protocol_hash == want and (b.spec.obtain is None) == (variant == "dropped")


# ===================================================================== capability


@pytest.mark.parametrize("name", NAMES)
def test_capability_is_declared_shown_and_matches_the_suites(name, capsys):
    """Every folder declares its capability tags (the suites' vocabulary); ``bench list`` / ``bench info`` show them;
    no shipped suite lists the folder under a capability it does not carry (``W_SUITE_CAPABILITY``)."""
    from yavih import cli
    from yavih.suites import CAPABILITIES

    y = _yaml(name)
    assert y["capability"] and set(y["capability"]) <= set(CAPABILITIES)
    assert Benchmark.load(name).spec.capability == tuple(y["capability"])
    assert cli.main(["--json", "bench", "info", name]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["capability"] == y["capability"]
    rep = lint.lint_folder(packaged_folder(name), checks=["capability"])
    assert not rep.findings, rep.findings


def test_capability_is_strict_and_outside_every_hash(tmp_path, esc50_copy):
    """An unknown capability is a schema error naming the vocabulary; editing or dropping the tags changes no hash;
    a shipped suite group the tags do not cover is a lint warning."""
    from yavih.benchmark import SchemaError

    want = Benchmark.load("esc50")
    for caps in (["visual", "temporal"], None):
        _edit_yaml(esc50_copy, lambda y, caps=caps: y.update({"capability": caps}) if caps else y.pop("capability", None))
        b = Benchmark.load(str(esc50_copy))
        assert (b.protocol_hash, b.input_hash, b.scoring_hash) == (want.protocol_hash, want.input_hash, want.scoring_hash)
    _edit_yaml(esc50_copy, lambda y: y.update({"capability": ["visual", "temporal"]}))
    found = lint.lint_folder(esc50_copy, checks=["capability"]).findings
    assert [(f.code, f.severity) for f in found] == [("W_SUITE_CAPABILITY", "warning")] * 2  # minimal + default: audio
    for bad, word in ((["smell"], "vocabulary is audio, visual"), ([], "non-empty"), ("audio", r"list\[str\]"),
                      (["audio", "audio"], "duplicate")):
        _edit_yaml(esc50_copy, lambda y, bad=bad: y.update({"capability": bad}))
        with pytest.raises(SchemaError, match=word):
            Benchmark.load(str(esc50_copy))
        assert "E_SCHEMA" in _codes(esc50_copy, checks=["yaml"])


# ===================================================================== the lint catches what it claims


@pytest.fixture
def esc50_copy(tmp_path):
    dst = tmp_path / "esc50"
    shutil.copytree(packaged_folder("esc50"), dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _codes(folder, **kw) -> set[str]:
    return {f.code for f in lint.lint_folder(folder, **kw).errors}


def test_lint_copy_is_clean(esc50_copy):
    assert lint.lint_folder(esc50_copy).ok


@pytest.mark.parametrize("mutate, code", [
    (lambda d: (d / "README.md").write_text((d / "README.md").read_text(encoding="utf-8").replace("## Citation", "## Cite"), encoding="utf-8"), "E_README"),
    (lambda d: (d / "setup.md").write_text((d / "setup.md").read_text(encoding="utf-8").replace("## Gotchas", "## Pitfalls"), encoding="utf-8"), "E_SETUP"),
    (lambda d: (d / "setup.md").write_text((d / "setup.md").read_text(encoding="utf-8") + "\nSee docs/bridge.md.\n", encoding="utf-8"), "E_CITATION"),
    # prepare steps run through the one command surface (a wheel install has no checkout to cd into)
    (lambda d: (d / "setup.md").write_text((d / "setup.md").read_text(encoding="utf-8").replace("yavih bench prepare esc50", "bash prepare"), encoding="utf-8"), "E_SETUP"),
    (lambda d: (d / "setup.md").write_text((d / "setup.md").read_text(encoding="utf-8") + "\nsee notes/decisions/x.md\n", encoding="utf-8"), "E_NOTES_LINK"),
    (lambda d: (d / "__init__.py").write_text("", encoding="utf-8"), "E_LAYOUT"),
    (lambda d: (d / "make_manifest.py").unlink(), "E_LAYOUT"),
    (lambda d: (d / "bridge_template.py").write_text(re.sub(r'"reading": "[a-z]+"', '"reading": "x"', (d / "bridge_template.py").read_text(encoding="utf-8")), encoding="utf-8"), "E_TEMPLATE"),
    # the README opens with the header image and carries no changelog / checklist / history section
    (lambda d: (d / "README.md").write_text("# ESC-50\n\n" + (d / "README.md").read_text(encoding="utf-8"), encoding="utf-8"), "E_README"),
    (lambda d: (d / "README.md").write_text((d / "README.md").read_text(encoding="utf-8") + "\n### Changelog\n- v1\n", encoding="utf-8"), "E_README"),
    (lambda d: (d / "setup.md").write_text((d / "setup.md").read_text(encoding="utf-8") + "\n### History\n- v1\n", encoding="utf-8"), "E_SETUP"),
    # an optional reader.py (packaged folders ship none) must import without torch
    (lambda d: (d / "reader.py").write_text("import torch\nfrom yavih.sample import iter_samples\n", encoding="utf-8"), "E_IMPORT"),
])
def test_lint_reports_a_broken_folder(esc50_copy, mutate, code):
    mutate(esc50_copy)
    assert code in _codes(esc50_copy)


def _edit_yaml(d, fn):
    y = yaml.safe_load((d / "benchmark.yaml").read_text(encoding="utf-8"))
    fn(y)
    (d / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_lint_yaml_obtain_budget_and_name(esc50_copy, tmp_path):
    _edit_yaml(esc50_copy, lambda y: y.pop("obtain"))
    assert "E_OBTAIN" in _codes(esc50_copy, checks=["obtain"])
    _edit_yaml(esc50_copy, lambda y: y.pop("capability"))
    assert "E_CAPABILITY" in _codes(esc50_copy, checks=["capability"])
    assert Benchmark.load(str(esc50_copy)).spec.capability == ()  # optional for the loader
    _edit_yaml(esc50_copy, lambda y: y.pop("header"))
    assert "E_HEADER" in _codes(esc50_copy, checks=["header"])
    _edit_yaml(esc50_copy, lambda y: y["budget"]["clips"].update({"1": 399}))
    assert "E_BUDGET" in _codes(esc50_copy, checks=["budget"])
    _edit_yaml(esc50_copy, lambda y: y.update({"scorer": "nope"}))
    assert "E_SCHEMA" in _codes(esc50_copy, checks=["yaml"])
    other = tmp_path / "renamed"
    shutil.copytree(packaged_folder("esc50"), other, ignore=shutil.ignore_patterns("__pycache__"))
    assert "E_NAME" in _codes(other, checks=["yaml"])
    (esc50_copy / "benchmark.yaml").write_text("name: [", encoding="utf-8")
    assert "E_YAML" in _codes(esc50_copy, checks=["yaml"])


def test_lint_todo_is_a_warning(esc50_copy):
    (esc50_copy / "setup.md").write_text((esc50_copy / "setup.md").read_text(encoding="utf-8") + "\nTODO: say more\n", encoding="utf-8")
    rep = lint.lint_folder(esc50_copy)
    assert rep.ok and [f.code for f in rep.warnings] == ["W_TODO"]


def _notes_findings(folder, rel: str) -> list[lint.Finding]:
    return [f for f in lint.lint_folder(folder, checks=["citations"]).findings if f.file == rel]


@pytest.mark.parametrize("rel, line", [
    ("setup.md", "Decisions: `$FOLDER/notes/decisions.md`."),
    ("README.md", "Provenance: notes/provenance.md (this folder)."),
    ("benchmark.yaml", "# decisions in notes/decisions.md (this folder)"),
    ("make_manifest.py", "# extra pin, measured by hand (notes/provenance.md §4)."),
    ("prepare/09_extra.sh", "# see esc50/notes/provenance.md §2"),
    ("tests/test_extra.py", "# order pinned in notes/provenance.md §5"),
    ("setup.md", "`$FOLDER/notes/x.md`; https://github.com/Bizilizi/yavih/blob/main/notes/y.md (the 2024 download)."),
])
def test_lint_flags_a_shipped_file_naming_notes(esc50_copy, rel, line):
    """A folder's notes/ is not in the wheel (pyproject exclude-package-data): no shipped text file may point there,
    wherever it sits (folder, prepare/, tests/) and however the path is spelled ($FOLDER/notes/, <name>/notes/, bare)."""
    p = esc50_copy / rel
    p.write_text((p.read_text(encoding="utf-8") if p.is_file() else "") + "\n" + line + "\n", encoding="utf-8")
    found = _notes_findings(esc50_copy, rel)
    assert [f.code for f in found if line.strip()[:40] in f.message] == ["E_NOTES_LINK"], found  # once, not also E_CITATION
    assert all(f.severity == "error" for f in found)


def test_lint_notes_rule_spares_non_paths_and_repo_urls(esc50_copy):
    """A yaml ``notes:`` key, a ``notes.txt`` file name, a ``*_notes/`` word and a github.com/Bizilizi/yavih URL are
    not links into an unshipped notes/ dir; files under notes/ itself are not read (they do not ship)."""
    rel = "prepare/09_extra.sh"
    (esc50_copy / rel).write_text("# notes: kept\n# writes notes.txt\n# my_notes/x\n"
                                  "# https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/esc50/notes/decisions.md\n",
                                  encoding="utf-8")
    (esc50_copy / "notes").mkdir(exist_ok=True)
    (esc50_copy / "notes" / "extra.md").write_text("see notes/decisions.md\n", encoding="utf-8")
    assert _notes_findings(esc50_copy, rel) == []
    assert not [f for f in lint.lint_folder(esc50_copy, checks=["citations"]).findings
                if (f.file or "").startswith("notes/")]


def test_bench_lint_cli(esc50_copy, capsys):
    from yavih import cli

    assert cli.main(["--json", "bench", "lint", "esc50", str(esc50_copy)]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["status"] == "ok" and [r["ok"] for r in env["data"]] == [True, True]
    (esc50_copy / "README.md").write_text("# nope\n", encoding="utf-8")
    assert cli.main(["--json", "bench", "lint", str(esc50_copy), "--skip", "imports"]) == 2
    env = json.loads(capsys.readouterr().out)
    assert env["status"] == "error" and any(e["code"] == "E_README" for e in env["errors"])
    assert cli.main(["bench", "lint", "no_such_benchmark"]) == 1
