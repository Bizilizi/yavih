"""fleet/check.py: the worker footprint, the work queue and its entries (fleet/worker-brief.md, fleet/integrate.md)."""

from __future__ import annotations

import copy
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fleet_check", REPO / "fleet" / "check.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _entry(name: str, **kw) -> dict:
    e = {"name": name, "title": name, "kind": "benchmark", "survey_class": "A", "survey_reasons": [],
         "route": "core-clf", "text_fields": False, "parents": [], "survey_note": "", "status": "todo",
         "obtain": None, "blocked_by": None, "branch": None}
    e.update(kw)
    return e


def test_the_repository_queue_is_valid_and_reserves_every_folder():
    entries = check.load_queue()
    assert check.check_queue(entries, check.repo_folders()) == []
    shipped = {e["name"] for e in entries if e["status"] == "shipped"}
    assert shipped == check.repo_folders()
    assert check.main(["queue"]) == 0


@pytest.mark.parametrize("mutate, needle", [
    (lambda q: q.append(copy.deepcopy(q[0])), "duplicate of entries[0]"),
    (lambda q: q[0].update(name="Bad-Name"), "name must match"),
    (lambda q: q[0].update(status="shipped"), "does not exist"),
    (lambda q: q[0].update(status="maybe"), "status must be one of"),
    (lambda q: q[0].update(route="core-zzz"), "route must be one of"),
    (lambda q: q[0].update(status="blocked"), "says what unblocks it"),
    (lambda q: q[0].update(status="claimed"), "names its branch, bench/alpha"),
    (lambda q: q[0].update(survey_reasons=["ii"]), "for B2 only"),
    (lambda q: q[0].update(obtain="by-email"), "obtain must be one of"),
    (lambda q: q[0].update(parents=["nosuch"]), "is not a suite entry"),
    (lambda q: q[0].pop("branch"), "missing keys ['branch']"),
    (lambda q: q[1].update(name="beta"), "has no queue entry"),
    (lambda q: q[2].update(components=["alpha"]), "does not list it under parents"),
    (lambda q: q[2].update(status="claimed", branch="bench/suite_x"), "never claimed for a worker"),
    (lambda q: q[2].update(status="review", branch="bench/suite_x"), "never claimed for a worker"),
])
def test_the_queue_check_names_each_problem(mutate, needle):
    q = [_entry("alpha"), _entry("existing", status="shipped"),
         _entry("suite_x", kind="suite", route="suite", components=["existing"])]
    q[1]["parents"] = ["suite_x"]
    assert check.check_queue(q, {"existing"}) == []
    mutate(q)
    problems = check.check_queue(q, {"existing"})
    assert any(needle in p for p in problems), problems


def test_a_queued_folder_a_pull_request_adds_is_allowed_only_when_named():
    q = [_entry("alpha")]
    assert any("already exists" in p for p in check.check_queue(q, {"alpha"}))
    assert check.check_queue(q, {"alpha"}, allow_new=frozenset({"alpha"})) == []


def test_entry_prints_one_entry(capsys):
    assert check.main(["entry", "esc50"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("name: esc50\n") and "status: shipped" in out
    assert check.main(["entry", "no_such_benchmark"]) == 1


def test_ship_marks_existing_folders_and_keeps_the_layout(tmp_path: Path):
    q = tmp_path / "queue.yaml"
    q.write_text("# header comment\nschema_version: 1\nentries:\n"
                 "- name: alpha\n  kind: benchmark\n  status: claimed\n  branch: bench/alpha\n"
                 "- name: beta\n  kind: benchmark\n  status: todo\n"
                 "- name: gamma\n  kind: benchmark\n  status: shipped\n"
                 "- name: alpha_suite\n  kind: suite\n  status: todo\n")
    assert check.ship(q, {"alpha", "gamma", "alpha_suite"}) == ["alpha"]
    text = q.read_text()
    assert text.startswith("# header comment\n")
    assert "- name: alpha\n  kind: benchmark\n  status: shipped\n  branch: bench/alpha\n" in text
    assert "- name: beta\n  kind: benchmark\n  status: todo\n" in text and "  status: todo\n" in text.split("alpha_suite")[1]
    assert check.ship(q, {"alpha", "gamma"}) == []


def test_ship_reads_an_entry_whatever_its_key_order(tmp_path: Path):
    q = tmp_path / "queue.yaml"
    q.write_text("schema_version: 1\nentries:\n"
                 "- status: todo\n  name: alpha\n  kind: benchmark\n"
                 "- name: beta\n  status: claimed\n  survey_note: 'kind: suite, status: todo'\n  kind: benchmark\n"
                 "- kind: suite\n  status: todo\n  name: delta\n")
    assert check.ship(q, {"alpha", "beta", "delta"}) == ["alpha", "beta"]
    text = q.read_text()
    assert text.startswith("schema_version: 1\nentries:\n- status: shipped\n  name: alpha\n")
    assert "- name: beta\n  status: shipped\n  survey_note: 'kind: suite, status: todo'\n" in text
    assert text.endswith("- kind: suite\n  status: todo\n  name: delta\n")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("needs git")
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "benchmarks" / "other").mkdir(parents=True)
    (r / "src" / "core.py").write_text("x = 1\n")
    (r / "benchmarks" / "other" / "benchmark.yaml").write_text("name: other\n")

    def git(*a: str) -> None:
        _git(r, *a)

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-qm", "base")
    git("checkout", "-qb", "bench/toy")
    (r / "benchmarks" / "toy").mkdir()
    (r / "benchmarks" / "toy" / "benchmark.yaml").write_text("name: toy\n")
    git("add", "-A")
    git("commit", "-qm", "toy")
    return r


def _git(r: Path, *a: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=r, check=True, capture_output=True)


def test_footprint_passes_a_folder_only_branch(repo: Path):
    assert check.footprint("bench/toy", "main", "toy", repo=repo) == []


def test_footprint_names_every_path_outside_the_folder(repo: Path):
    (repo / "src" / "core.py").write_text("x = 2\n")
    (repo / "benchmarks" / "other" / "benchmark.yaml").write_text("name: other  # edited\n")
    _git(repo, "commit", "-qam", "core too")
    assert check.footprint("bench/toy", "main", "toy", repo=repo) == ["benchmarks/other/benchmark.yaml", "src/core.py"]


def test_footprint_sees_a_file_moved_out_of_another_folder(repo: Path):
    _git(repo, "mv", "benchmarks/other/benchmark.yaml", "benchmarks/toy/stolen.yaml")
    _git(repo, "commit", "-qm", "move")
    assert check.footprint("bench/toy", "main", "toy", repo=repo) == ["benchmarks/other/benchmark.yaml"]


def test_footprint_refuses_bad_input(repo: Path):
    with pytest.raises(check.CheckError, match="not a folder name"):
        check.footprint("bench/toy", "main", "../src", repo=repo)
    with pytest.raises(check.CheckError, match="changes nothing"):
        check.footprint("main", "main", "toy", repo=repo)
    with pytest.raises(check.CheckError, match="git diff"):
        check.footprint("no-such-branch", "main", "toy", repo=repo)
