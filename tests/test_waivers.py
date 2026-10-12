"""Validation waivers (``yavih.waivers``, docs/validation.md): a check that is wrong for a model is waived WITH A REASON —
``describe()["waive"]`` (the bridge) or ``--waive`` / matrix ``waive`` (the operator) — never edited out of yavih. A waived
error is ``W_WAIVED``, recorded (profile, ``_meta.json``, ``result.json``, results store) and marked in the report, never
greyed or excluded. Structural and identity checks and the unfilled-template guard are never waivable."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_extract_fixtures import (
    BatchStatsBridge,
    DropoutBridge,
    FakeBenchmark,
    NumpyBridge,
    OrderBridge,
    WrongTimesBridge,
)
from test_plan_helpers import (
    install_benchmarks,
    make_info,
    make_profile,
    write_matrix,
    write_profile,
)
from yavih import hashing, waivers
from yavih.bridge import DescribeError
from yavih.extract import ExtractError, bridge_check
from yavih.waivers import (
    WAIVABLE,
    WaiveError,
    check_waive,
    describe_waivers,
    parse_cli_waivers,
    waivers_for,
)

# ----------------------------------------------------------------------------- the waivable set


def test_waivable_and_never_waivable_sets():
    assert set(WAIVABLE) == {"E_TIMES_OUT_OF_MEDIA", "E_RULE_MISMATCH", "E_POS_RANGE", "E_BATCH_INVARIANCE", "E_RERUN_DRIFT",
                             "E_CONSTANT_FEATURES", "E_DEAD_ROWS", "E_DUPLICATE_CLIPS", "E_ARGS_UNCONSUMED"}
    for code in ("E_FORMAT", "E_SHARD_SET", "E_RANGE_TILING", "E_MEMBERS", "E_ROWS", "E_TAR_SHA", "E_POS_SHAPE", "E_FINITE",
                 "E_NA_AND_SHARDS", "E_HASH_MANIFEST", "E_HASH_INPUT", "E_DESCRIBE_MISMATCH", "E_TEMPLATE_UNFILLED"):
        assert code in waivers.NOT_WAIVABLE and code not in WAIVABLE
        with pytest.raises(WaiveError) as ei:
            check_waive({code: "a reason"})
        assert "not waivable" in str(ei.value) and ei.value.code == "E_WAIVE_INVALID"
    from yavih.io.validate import CODES

    assert {c for c in WAIVABLE if c in CODES} <= {c for c, lvl in CODES.items() if lvl == "error"}  # validate's waivable errors


def test_check_waive_star_explicit_and_reasons():
    full = check_waive({"*": "under study", "E_DEAD_ROWS": "register tokens"})
    assert set(full) == set(WAIVABLE) and full["E_DEAD_ROWS"] == "register tokens" and full["E_RERUN_DRIFT"] == "under study"
    for bad in ({"E_DEAD_ROWS": " "}, {"E_DEAD_ROWS": 3}, {"W_COVERAGE": "x"}, {"N_RULE_CUSTOM": "x"}, {"E_X": "x"}, "E_DEAD_ROWS", {1: "x"}):
        with pytest.raises(WaiveError):
            check_waive(bad)


def test_cli_waivers_parse():
    assert parse_cli_waivers(["E_DEAD_ROWS=register tokens", "E_RULE_MISMATCH=a=b"]) == {"E_DEAD_ROWS": "register tokens",
                                                                                         "E_RULE_MISMATCH": "a=b"}
    assert set(parse_cli_waivers(["all=checkpoint under study"])) == set(WAIVABLE)
    with pytest.raises(WaiveError) as ei:
        parse_cli_waivers(["E_DEAD_ROWS"])
    assert ei.value.exit_code == 1 and "CODE=reason" in str(ei.value)
    with pytest.raises(WaiveError):
        parse_cli_waivers(["E_HASH_INPUT=no"])


def test_describe_waivers_read_the_waive_mapping():
    assert describe_waivers({"waive": {"E_DEAD_ROWS": "padding"}}) == {"E_DEAD_ROWS": "padding"}
    assert describe_waivers({"model": "m"}) == {} and describe_waivers(None) == {}
    with pytest.raises(WaiveError):
        describe_waivers({"waive": {"E_HASH_INPUT": "no"}})


def test_waivers_for_describe_before_cli():
    w = waivers_for({"waive": {"E_DEAD_ROWS": "padding"}}, {"E_DEAD_ROWS": "cli says", "E_RULE_MISMATCH": "operator"})
    assert w.entries == {"E_DEAD_ROWS": ("padding", "describe"), "E_RULE_MISMATCH": ("operator", "cli")}
    w.record("E_DEAD_ROWS")
    assert w.to_json() == [{"code": "E_DEAD_ROWS", "reason": "padding", "source": "describe", "count": 1},
                           {"code": "E_RULE_MISMATCH", "reason": "operator", "source": "cli", "count": 0}]
    assert w.message("E_RULE_MISMATCH", "m") == "E_RULE_MISMATCH: m — waived: operator (--waive)"


# ----------------------------------------------------------------------------- hashing: a bridge decision, never a protocol


def test_waive_is_in_describe_sha256_and_nowhere_else():
    from test_plan_helpers import make_describe

    d = make_describe()
    assert "waive" not in hashing.DESCRIBE_SHA_EXCLUDED
    assert hashing.describe_sha256(d) != hashing.describe_sha256({**d, "waive": {"E_DEAD_ROWS": "padding"}})
    assert hashing.describe_sha256({**d, "waive": {"E_DEAD_ROWS": "padding"}}) != hashing.describe_sha256({**d, "waive": {"E_DEAD_ROWS": "other"}})


# ----------------------------------------------------------------------------- bridge check


def _with_waive(cls, waive, name):
    class Waiving(cls):
        def describe(self):
            return dict(super().describe(), waive=waive)

    Waiving.name = name
    Waiving.__name__ = f"Waiving{cls.__name__}"
    return Waiving


@pytest.mark.parametrize("cls, codes", [
    (BatchStatsBridge, ("E_BATCH_INVARIANCE",)),
    (OrderBridge, ("E_BATCH_INVARIANCE",)),
    (DropoutBridge, ("E_RERUN_DRIFT", "E_BATCH_INVARIANCE")),  # random noise fails the rerun and both invariance passes
    (WrongTimesBridge, ("E_RULE_MISMATCH",)),
])
def test_bridge_check_waived_hard_check_writes_a_profile_plan_accepts(tmp_path, cls, codes):
    out = tmp_path / "profile.json"
    reason = "a known property of this research model"
    prof = bridge_check(bridge=_with_waive(cls, dict.fromkeys(codes, reason), f"w-{cls.__name__.lower()}"),
                        bench=FakeBenchmark(), n=8, decode_workers=1, out=out)
    assert prof["status"] == "ok" and out.is_file()
    waived = [w for w in prof["warnings"] if w["code"] == "W_WAIVED"]
    assert waived and all(w["message"].split(":")[0] in codes and w["message"].endswith(f"waived: {reason}") for w in waived)
    assert [w["code"] for w in prof["waived"]] == sorted(codes) and all(w["count"] >= 1 and w["source"] == "describe" for w in prof["waived"])
    assert sum(w["count"] for w in prof["waived"]) == len(waived)
    from yavih.plan.matrix import load_profile

    load_profile(str(out))  # plan accepts the profile: the failed hard check is a recorded, waived one


def test_bridge_check_waived_constant_features(tmp_path):
    class Constant(NumpyBridge):
        name = "w-constant"

        def describe(self):
            return dict(super().describe(), waive={"E_CONSTANT_FEATURES": "a random-init baseline", "E_DEAD_ROWS": "random-init baseline"})

        def encode(self, samples):
            feats, pos, used = super().encode(samples)
            return np.zeros_like(feats), pos, used

    prof = bridge_check(bridge=Constant, bench=FakeBenchmark(), n=4, decode_workers=1)
    assert {w["code"]: w["count"] for w in prof["waived"]} == {"E_CONSTANT_FEATURES": 1, "E_DEAD_ROWS": 0}


@pytest.mark.parametrize("waive, needle", [
    ({"E_TEMPLATE_UNFILLED": "later"}, "unfilled-template"),
    ({"E_HASH_INPUT": "x"}, "identity"),
    ({"E_RERUN_DRIFT": ""}, "non-empty reason"),
])
def test_bridge_check_refuses_an_invalid_waiver(waive, needle):
    with pytest.raises(DescribeError) as ei:
        bridge_check(bridge=_with_waive(NumpyBridge, waive, "w-bad"), bench=FakeBenchmark(), n=4, decode_workers=1)
    assert ei.value.code == "E_WAIVE_INVALID" and needle in str(ei.value)


def test_bridge_check_without_waiver_still_refuses():
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=_with_waive(DropoutBridge, {"E_BATCH_INVARIANCE": "a different check"}, "w-drop"), bench=FakeBenchmark(),
                     n=8, decode_workers=1)
    assert ei.value.code == "E_RERUN_DRIFT"


def test_args_unconsumed_is_waivable():
    from yavih.bridge import ArgSpec, validate_describe

    class Dev(NumpyBridge):
        name = "w-dev"
        ARGS = {**NumpyBridge.ARGS, "device": ArgSpec(type=str, default="cpu")}

        def describe(self):
            d = super().describe()
            d["args"] = {k: v for k, v in d["args"].items() if k != "device"}  # not echoed: E_ARGS_UNCONSUMED
            return d

    from test_extract_fixtures import view_of

    inst = Dev(view_of(FakeBenchmark()), {"device": "cpu"})
    d = inst.describe()
    with pytest.raises(DescribeError) as ei:
        validate_describe(d, inst, view_of(FakeBenchmark()))
    assert ei.value.code == "E_ARGS_UNCONSUMED"
    reason = "the device cannot change the features"
    validate_describe(dict(d, waive={"E_ARGS_UNCONSUMED": reason}), inst, view_of(FakeBenchmark()))
    prof = bridge_check(bridge=_with_waive(Dev, {"E_ARGS_UNCONSUMED": reason}, "w-dev2"), bench=FakeBenchmark(), n=4, decode_workers=1)
    assert prof["waived"][0]["count"] == 1 and any(w["message"].startswith("E_ARGS_UNCONSUMED: ") for w in prof["warnings"])


# ----------------------------------------------------------------------------- plan: profile and matrix


def test_plan_profile_with_failed_invariance_needs_the_waiver(tmp_path):
    from yavih.plan.matrix import ProfileError, load_profile

    prof = make_profile()
    prof["invariance"]["batch_rel_l2"] = 0.2
    p = write_profile(tmp_path, "p.json", prof)
    with pytest.raises(ProfileError, match="E_BATCH_INVARIANCE"):
        load_profile(str(p))
    prof = make_profile(describe_over={"waive": {"E_BATCH_INVARIANCE": "bf16 attention differs by batch"}})
    prof["invariance"]["batch_rel_l2"] = 0.2
    load_profile(str(write_profile(tmp_path, "p2.json", prof)))
    bad = make_profile(describe_over={"waive": {"E_FORMAT": "x"}})
    with pytest.raises(ProfileError, match="E_WAIVE_INVALID"):
        load_profile(str(write_profile(tmp_path, "p3.json", bad)))


def test_matrix_waive_reaches_the_probe_units_and_not_their_identity(tmp_path, monkeypatch):
    from yavih.plan import units as U
    from yavih.plan.matrix import MatrixError

    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    write_profile(root, "bridges/s.profile.json", make_profile())
    entry = {"script": "bridges/s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "bridges/s.profile.json"}
    plain = [u for u in U.plan(str(write_matrix(root, {"s": entry}))) if u.kind == "probe"]
    entry_w = dict(entry, waive={"E_DUPLICATE_CLIPS": "digitally silent audio"})
    waived = [u for u in U.plan(str(write_matrix(root, {"s": entry_w}))) if u.kind == "probe"]
    assert "--waive" in waived[0].cmd and "E_DUPLICATE_CLIPS=digitally silent audio" in waived[0].cmd
    assert "--waive" not in plain[0].cmd
    assert waived[0].id == plain[0].id and waived[0].inputs_hash == plain[0].inputs_hash  # an operator's waiver changes no hash
    assert waived[0].protocol_hash == plain[0].protocol_hash and waived[0].eval_hash == plain[0].eval_hash
    with pytest.raises(MatrixError, match="E_WAIVE_INVALID"):
        U.plan(str(write_matrix(root, {"s": dict(entry, waive={"E_ROWS": "x"})})))


def test_exec_passes_the_matrix_waiver_to_the_probe(monkeypatch, tmp_path):
    import yavih.readouts as readouts
    from yavih.plan import exec as plan_exec

    seen = {}
    monkeypatch.setattr(readouts, "probe", lambda run_dir, **kw: seen.update(kw))
    unit = SimpleNamespace(cwd=str(tmp_path), out="runs/b/m/v", seed=0, expected_status="ok", profile=None,
                           cmd=["python", "-m", "yavih.cli", "probe", "run", "runs/b/m/v", "--seeds", "0",
                                "--waive", "E_DEAD_ROWS=register tokens"])
    plan_exec._run_probe(unit, deadline=None, scratch=None)
    assert seen["waive"] == {"E_DEAD_ROWS": "register tokens"} and seen["validate_first"] is True


# ----------------------------------------------------------------------------- result.json, results store, report


WAIVED = [{"code": "E_DUPLICATE_CLIPS", "reason": "digitally silent audio", "source": "describe", "count": 3, "splits": ["test"]}]


def test_result_json_carries_waived_from_the_metas():
    from yavih.readouts.base import _waived_of

    metas = {"train": {"waived": [{"code": "E_DUPLICATE_CLIPS", "reason": "r", "source": "describe", "count": 1}]},
             "test": {"waived": [{"code": "E_DUPLICATE_CLIPS", "reason": "r", "source": "describe", "count": 2},
                                 {"code": "E_DEAD_ROWS", "reason": "p", "source": "cli", "count": 1}]}}
    assert _waived_of(metas) == [{"code": "E_DEAD_ROWS", "reason": "p", "source": "cli", "count": 1, "splits": ["test"]},
                                 {"code": "E_DUPLICATE_CLIPS", "reason": "r", "source": "describe", "count": 3, "splits": ["test", "train"]}]
    assert _waived_of({"x": {}}) == []


def test_report_marks_waived_rows_without_greying_them():
    from test_report import _records, make_result
    from yavih.report.compare import build_report

    a = make_result(model="m1")
    b = dict(make_result(model="m2"), waived=WAIVED)
    from yavih.readouts.base import check_result

    check_result(b)  # `waived` is a known, optional result key
    check_result(a)
    md = build_report(_records(a, b), fmt="md")
    rows = {r["model"]: r for r in md.rows}
    assert rows["m2"]["waived"] and not rows["m1"]["waived"]
    assert rows["m2"]["greyed"] is False and rows["m2"]["flags"] == rows["m1"]["flags"]
    text = md.text
    assert "| m2 † |" in text and "| m1 |" in text
    assert "† m2 v: waived E_DUPLICATE_CLIPS (digitally silent audio; describe, 3 finding(s))" in text
    lm = {r.get("waived") is not None for r in md.lm_eval["m2"]}
    assert lm == {True} and " †" in text.split("# Comparability columns")[0]
    csv_rep = build_report(_records(a, b), fmt="csv")
    lines = csv_rep.text.splitlines()
    assert lines[0].split(",")[-1] == "waived" and any(ln.endswith(",E_DUPLICATE_CLIPS") for ln in lines if ",m2," in ln)


def test_results_store_entry_records_waived():
    from test_report import make_result
    from yavih.report.store import benchmark_entry

    e = benchmark_entry([dict(make_result(), waived=WAIVED)])
    assert e["configs"]["waived"][0]["code"] == "E_DUPLICATE_CLIPS" and e["results"]["waived"] == ["E_DUPLICATE_CLIPS"]
    assert "waived" not in benchmark_entry([make_result()])["results"]


def test_docs_name_every_waivable_code():
    from pathlib import Path

    import yavih

    pkg = Path(yavih.__file__).resolve().parent
    validation_md = (pkg.parents[1] / "docs" / "validation.md").read_text()
    cli_md = (pkg / "skill" / "yavih" / "references" / "cli.md").read_text()
    bridge_md = (pkg / "skill" / "yavih" / "references" / "bridge.md").read_text()
    for code in WAIVABLE:
        assert code in validation_md and code in cli_md
    for needle in ("W_WAIVED", "E_WAIVE_INVALID", "--waive"):
        assert needle in validation_md and needle in cli_md
    assert "VGGSounder" in bridge_md and '"waive"' in bridge_md
    skill = (pkg / "skill" / "yavih" / "SKILL.md").read_text()
    assert "don't edit yavih" in skill and "waive" in skill
    assert json.dumps(sorted(WAIVABLE))  # stable, JSON-able
