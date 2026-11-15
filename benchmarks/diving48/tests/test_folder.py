"""Folder tests of diving48 (run: ``python -m pytest "$(yavih bench path diving48)/tests" -q -p no:cacheprovider``).

Written by ``yavih bench new`` from ssv2: the tests run on the scaffold as it is, so every value pinned below is
still ssv2's. TODO: pin this folder's own facts once its manifest is frozen (rows per split, the targets and metrics
on hand-checked inputs, the bridge template's ``describe()``) and write the tests that skip with a ``TODO:`` reason.
Test what nothing else checks: ``yavih bench lint`` (run once per folder by the repository) already checks the
layout, the yaml, the hashes, the conformance fixture and the README.md / setup.md structure, so never re-run it here
and never pin documentation wording. ``yavih bench lint`` reports every ``TODO:`` left in this file.
"""

from __future__ import annotations

import re

import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows = F.spec, F.rows

READOUT = 'clf'
HEADLINE = 'acc'
STREAMS = {'video': 'video'}
CLIPS = {'train': 20000, 'val': 24777}
READINGS = ['v']
EXTERNAL = []
HAS_EVAL = False


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)
    assert [e["name"] for e in spec["external"] or []] == EXTERNAL
    assert (HERE / "eval.py").is_file() is HAS_EVAL


# ----------------------------------------------------------------------------- prepare/


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_obtain.sh"]  # TODO: this benchmark's numbered prepare steps
    assert all(re.match(r"^\d+_[A-Za-z0-9_.-]+\.(?:sh|py)$", n) for n in names), names


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values()
              if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    d = cls(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] in READINGS and set(d["input"]) <= set(STREAMS)


# ----------------------------------------------------------------------------- needs this benchmark's data


def test_make_manifest_check():
    pytest.skip("TODO: run make_manifest.py --check on the public upstream files (or a stand-in rebuilt from the "
                "committed manifest) and assert it re-derives manifest.csv.gz byte-identically")


def test_reader_smoke_synthetic_raw():
    pytest.skip("TODO: write synthetic media for a few manifest rows and assert what Benchmark.reader hands over "
                "(streams, info keys, meta, canaries)")
