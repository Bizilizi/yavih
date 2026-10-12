"""``yavih.report.migrate`` protocol-hash upgrades (the benchmark ``changelog``; docs/hashing.md).

There is no central table: each folder's newest ``changelog`` entry records the ``protocol_hash`` it computes and the
earlier hashes it ``supersedes``; ``report.migrate`` builds its map by scanning the folders (``yavih.hash_history``).
These tests use synthetic folders and hashes; ``tests/test_hashing.py`` checks that every packaged folder records the
hash it computes."""

from __future__ import annotations

import pytest

from yavih import hash_history, hashing
from yavih.report.migrate import PROTOCOL_HASH_UPGRADES, migrate, upgrade_protocol_hash

OLD, NEW, INP, OLD_INP = "a1" * 32, "b2" * 32, "c3" * 32, "d4" * 32


@pytest.fixture
def bench_path(tmp_path, monkeypatch):
    """An out-of-tree folder ``mybench`` on ``$YAVIH_BENCH_PATH`` whose current protocol hash supersedes ``OLD``."""
    folder = tmp_path / "packs" / "mybench"
    folder.mkdir(parents=True)
    (folder / "benchmark.yaml").write_text(
        "name: mybench\nchangelog:\n  - version: 1\n    date: '2026-01-01'\n    reason: x\n    invalidates: false\n"
        f"    protocol_hash: {NEW}\n    input_hash: {INP}\n    supersedes: [{OLD}]\n    supersedes_input: [{OLD_INP}]\n",
        encoding="utf-8")
    monkeypatch.setenv("YAVIH_BENCH_PATH", str(tmp_path / "packs"))
    hash_history.reset()
    yield folder
    monkeypatch.delenv("YAVIH_BENCH_PATH")
    hash_history.reset()


def test_the_map_is_built_from_the_folders_only():
    hist = hash_history.history()
    assert set(PROTOCOL_HASH_UPGRADES) == {h for r in hist.records for h in r.supersedes if h != r.protocol_hash}
    assert not set(PROTOCOL_HASH_UPGRADES) & set(PROTOCOL_HASH_UPGRADES.values())
    for old, new in PROTOCOL_HASH_UPGRADES.items():
        assert hist.owner(old) == hist.owner(new)  # a hash never migrates into another benchmark's column


def test_out_of_tree_folder_brings_its_own_history(bench_path):
    """A folder on $YAVIH_BENCH_PATH is scanned like a packaged one: its ``supersedes`` / ``supersedes_input`` apply."""
    assert upgrade_protocol_hash(OLD) == NEW and PROTOCOL_HASH_UPGRADES[OLD] == NEW
    assert upgrade_protocol_hash(NEW) == NEW
    assert hash_history.effective_input_hash({"input_hash": OLD_INP}) == INP
    assert hash_history.effective_input_hash({"input_hash": INP}) == INP


def test_an_unknown_hash_is_not_upgraded(bench_path):
    other = "e5" * 32
    assert other not in PROTOCOL_HASH_UPGRADES and upgrade_protocol_hash(other) == other


def test_a_superseded_result_lands_in_the_current_column(bench_path):
    """A current-schema result written under a superseded protocol hash moves to the current column (``eval_hash``
    recomputed); in memory only."""
    qh = "c" * 64
    rec = {"schema_version": 3, "model": "m", "reading": "a", "benchmark": {"name": "mybench", "protocol_hash": OLD},
           "probe": {"hash": qh}, "eval_hash": hashing.eval_hash(OLD, qh), "bridge": {"name": "b", "describe": {}}}
    out = migrate(rec)
    assert out["benchmark"]["protocol_hash"] == NEW and out["eval_hash"] == hashing.eval_hash(NEW, qh)
    assert rec["benchmark"]["protocol_hash"] == OLD  # the input record is untouched
