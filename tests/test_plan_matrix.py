"""matrix.yaml schema, defaults merge, profile checks and refusals (exit 1)."""

from __future__ import annotations

import json

import pytest

from test_plan_helpers import make_describe, make_profile, write_matrix, write_profile
from yavih import hashing
from yavih.plan.matrix import (
    MatrixError,
    ProfileError,
    check_describe,
    check_profile,
    load_matrix,
    load_profile,
)


def _entry(**over):
    e = {"script": "bridges/x.py", "benchmark": "synth_clf", "profile": "bridges/x.profile.json"}
    e.update(over)
    return e


def test_load_matrix_defaults_merge_and_types(tmp_path):
    p = write_matrix(
        tmp_path,
        {"a": _entry(args={"layer": 12}), "b": _entry(launcher="conda run -n x python", env={"OMP_NUM_THREADS": 2}, cpus=16)},
        defaults={"launcher": "/envs/v/bin/python", "env": {"HF_HUB_OFFLINE": "1"}, "cpus": 8, "args": {"pool": "mean"}},
        seeds=[0, 1],
        shard_minutes=15,
        canaries=4,
    )
    m = load_matrix(str(p))
    assert m.schema_version == 1 and m.seeds == (0, 1) and m.shard_minutes == 15 and m.canaries == 4
    assert m.out == "runs" and m.root == str(tmp_path)
    a, b = m.entries
    assert a.key == "a" and a.launcher == "/envs/v/bin/python" and a.env == {"HF_HUB_OFFLINE": "1"} and a.cpus == 8
    assert a.args == {"pool": "mean", "layer": 12}  # defaults.args merged under the entry's args
    assert b.launcher == "conda run -n x python" and b.cpus == 16
    assert b.env == {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2"}  # scalar env values coerced to str


def test_load_matrix_out_defaults_to_runs_root(tmp_path, runs_dir):
    p = tmp_path / "matrix.yaml"
    p.write_text("schema_version: 1\nbridges:\n  a: {script: s.py, benchmark: k400, profile: p.json}\n")
    m = load_matrix(str(p))
    assert m.out == str(runs_dir)
    assert m.seeds == (0,) and m.shard_minutes == 30 and m.canaries == 8


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda m: m.update({"nope": 1}), "unknown matrix keys"),
        (lambda m: m.update({"schema_version": 2}), "schema_version"),
        (lambda m: m.update({"seeds": []}), "seeds"),
        (lambda m: m.update({"seeds": [0, 0]}), "distinct"),
        (lambda m: m.update({"shard_minutes": 0}), "shard_minutes"),
        (lambda m: m.update({"bridges": {}}), "bridges"),
        (lambda m: m["bridges"]["a"].update({"weird": 1}), "unknown keys"),
        (lambda m: m["bridges"]["a"].pop("profile"), "profile is required"),
        (lambda m: m["bridges"]["a"].pop("script"), "script is required"),
        (lambda m: m["bridges"]["a"].update({"benchmark": "K400!"}), "benchmark"),
        (lambda m: m["bridges"]["a"].update({"cpus": 0}), "cpus"),
        (lambda m: m["bridges"]["a"].update({"env": "x"}), "env must be a mapping"),
        (lambda m: m.update({"defaults": {"gpus": 2}}), "unknown matrix.defaults keys"),
    ],
)
def test_load_matrix_refusals_exit_1(tmp_path, mutate, needle):
    import yaml

    m = {"schema_version": 1, "out": "runs", "bridges": {"a": _entry()}}
    mutate(m)
    p = tmp_path / "matrix.yaml"
    p.write_text(yaml.safe_dump(m))
    with pytest.raises(MatrixError) as ei:
        load_matrix(str(p))
    assert needle in str(ei.value)
    assert MatrixError.exit_code == 1 and ei.value.exit_code == 1


def test_load_matrix_missing_or_non_mapping(tmp_path):
    with pytest.raises(MatrixError):
        load_matrix(str(tmp_path / "none.yaml"))
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(MatrixError):
        load_matrix(str(p))


# ----------------------------------------------------------------------------- profile


def test_load_profile_ok_and_extra_keys_allowed(tmp_path):
    prof = make_profile()
    p = write_profile(tmp_path, "p.json", prof)
    got = load_profile(str(p))
    assert (got["model"], got["reading"]) == ("synthmodel", "v") and "used" in got


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda p: p.pop("clips_per_s"), "lacks keys"),
        (lambda p: p.update({"clips_per_s": 0}), "clips_per_s"),
        (lambda p: p.update({"N": 0}), "N and D"),
        (lambda p: p.update({"pos_mode": "weird"}), "pos_mode"),
        # model and reading are names with charsets (they key paths)
        (lambda p: p.update({"model": "a/b"}), "model"),
        (lambda p: p.update({"model": ".."}), "model"),
        (lambda p: p.update({"model": "x" * 65}), "model"),
        (lambda p: p.update({"reading": "a+v"}), "reading"),
        (lambda p: p.update({"reading": ""}), "reading"),
        (lambda p: p.pop("reading"), "lacks keys"),
        (lambda p: (p.pop("model"), p.pop("reading")), "lacks keys"),
        # the profile must record PASSED hard checks
        (lambda p: p.pop("invariance"), "lacks keys"),
        (lambda p: p.update({"invariance": "ok"}), "invariance must be an object"),
        (lambda p: p["invariance"].pop("order_rel_l2"), "invariance lacks"),
        (lambda p: p["invariance"].update({"rerun_bit_identical": False}), "E_RERUN_DRIFT"),
        (lambda p: p["invariance"].update({"batch_rel_l2": 2e-5}), "E_BATCH_INVARIANCE"),
        (lambda p: p["invariance"].update({"order_rel_l2": None}), "E_BATCH_INVARIANCE"),
        (lambda p: p["invariance"].update({"tolerance": 1e-2}), "tolerance"),
        (lambda p: p["invariance"].update({"tolerance": 1e-4}), "tolerance_reason"),
    ],
)
def test_load_profile_refusals(tmp_path, mutate, needle):
    prof = make_profile()
    mutate(prof)
    p = write_profile(tmp_path, "p.json", prof)
    with pytest.raises(ProfileError) as ei:
        load_profile(str(p))
    assert needle in str(ei.value) and ei.value.exit_code == 1


def test_load_profile_missing_or_bad_json(tmp_path):
    with pytest.raises(ProfileError, match="bridge check"):
        load_profile(str(tmp_path / "nope.json"))
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ProfileError, match="JSON"):
        load_profile(str(p))


def test_check_describe_accepts_valid_describes():
    check_describe(make_describe())
    check_describe(make_describe(batch_tolerance={"rel_l2": 1e-4, "reason": "bf16 attention kernels differ between batch sizes"}))
    check_describe(make_describe(weights_sha256=None, layer="blocks.11", precision={"autocast": "bf16", "matmul_precision": "high", "allow_tf32": True, "deterministic": False}))


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"precision": {"autocast": None, "matmul_precision": None, "allow_tf32": False, "deterministic": True}}, "E_PRECISION_UNDECLARED"),
        ({"precision": {"autocast": None, "matmul_precision": "highest"}}, "precision lacks"),
        ({"weights_sha256": "abc"}, "weights_sha256"),
        ({"dtype": "bfloat16"}, "dtype"),
        ({"views": 3}, "views must be 1"),
        ({"views": "1"}, "wrong type"),
        ({"input": {}}, "describe.input"),
        ({"batch_tolerance": {"rel_l2": 1e-2, "reason": "bf16"}}, "batch_tolerance"),
        ({"batch_tolerance": {"rel_l2": 1e-4, "reason": ""}}, "batch_tolerance"),
        ({"batch_size": "4"}, "wrong type"),
        ({"streams": "video"}, "wrong type"),
        ({"args": {"x": object()}}, "JSON-primitive"),
        # model/reading charsets
        ({"model": "my model"}, "describe.model"),
        ({"model": "."}, "describe.model"),
        ({"model": 3}, "wrong type"),
        ({"reading": "a:v"}, "describe.reading"),
        ({"reading": "r" * 33}, "describe.reading"),
    ],
)
def test_check_describe_refusals(over, needle):
    d = make_describe(**over)
    with pytest.raises(ProfileError) as ei:
        check_describe(d)
    assert needle in str(ei.value)


def test_check_describe_missing_key():
    for key in ("model", "reading"):
        d = make_describe()
        d.pop(key)
        with pytest.raises(ProfileError, match=key):
            check_describe(d)


def test_check_describe_accepts_model_charset_edges():
    """``[A-Za-z0-9_.-]{1,64}`` for model, ``[A-Za-z0-9_-]{1,32}`` for reading, each starting with a letter
    or digit."""
    check_describe(make_describe(model="vjepa2-l_v1.0", reading="av"))
    check_describe(make_describe(model="m" * 64, reading="r" * 32))
    check_describe(make_describe(model="x...", reading="audio_visual-2"))
    check_describe(make_describe(model="0_.-", reading="9_-"))


@pytest.mark.parametrize("model,reading", [("_stale", "a"), ("_report", "a"), ("_x", "a"), (".hidden", "a"), ("...x", "a"), ("-m", "a"),
                                           ("m1", "_stale"), ("m1", "_a"), ("m1", "-a"), ("m" * 65, "a"), ("m1", "r" * 33)])
def test_check_describe_refuses_reserved_and_hidden_names(model, reading):
    """Model and reading are path components. A leading ``_`` would be invisible to the report
    (``report.index`` never enters ``_stale``; the join skips ``_``-directories) or clash with ``_report``; a leading
    ``.`` hides the directory from ``*`` globs; a leading ``-`` reads as an option. Refused at plan time too."""
    with pytest.raises(ProfileError, match="start with a letter or digit"):
        check_describe(make_describe(model=model, reading=reading))


def _matrix_entry(**over):
    from yavih.plan.matrix import MatrixEntry

    kw = {"key": "e", "script": "s.py", "benchmark": "synth_clf", "profile": "p.json", "args": {"layer": 12}}
    kw.update(over)
    return MatrixEntry(**kw)


def test_check_profile_ok_alias_and_defaulted_args():
    prof = make_profile("synth_alias", args={"layer": 12, "pool": "mean"})  # pool: a default filled by ARGS
    check_profile(prof, _matrix_entry(), bench_names=("synth_clf", "synth_alias"))
    check_profile(make_profile(args={"layer": 12}), _matrix_entry(args={"layer": "12"}))  # yaml string spelling


def test_check_profile_refuses_bench_mismatch():
    with pytest.raises(ProfileError, match="profile.bench"):
        check_profile(make_profile("k400"), _matrix_entry())


def test_check_profile_refuses_args_mismatch():
    with pytest.raises(ProfileError, match="profile.args differ"):
        check_profile(make_profile(args={"layer": 11}), _matrix_entry())
    with pytest.raises(ProfileError, match="profile.args differ"):
        check_profile(make_profile(args={}), _matrix_entry())


def test_check_profile_refuses_bad_describe_and_profile_drift():
    prof = make_profile()
    prof["describe"]["precision"]["deterministic"] = None
    with pytest.raises(ProfileError, match="E_PRECISION_UNDECLARED"):
        check_profile(prof, _matrix_entry())
    prof = make_profile()
    prof["describe"]["layer"] = 13  # describe edited after bridge check: describe_sha256 no longer matches
    with pytest.raises(ProfileError, match="describe_sha256"):
        check_profile(prof, _matrix_entry())
    # the profile's model/reading are copies of describe's; plan never computes them
    prof = make_profile()
    prof["model"] = "othermodel"
    with pytest.raises(ProfileError, match="profile.model"):
        check_profile(prof, _matrix_entry())
    prof = make_profile()
    prof["reading"] = "a"
    with pytest.raises(ProfileError, match="profile.reading"):
        check_profile(prof, _matrix_entry())


def test_check_profile_accepts_null_weights_fingerprint():
    """A null ``weights_sha256`` is recorded (W_WEIGHTS_UNFINGERPRINTED), never a gate or a name
    change: the profile plans exactly like a fingerprinted one."""
    prof = make_profile(describe_over={"weights_sha256": None})
    check_profile(prof, _matrix_entry())
    assert (prof["model"], prof["reading"]) == (make_profile()["model"], make_profile()["reading"])


def test_profile_roundtrip_json(tmp_path):
    prof = make_profile()
    p = write_profile(tmp_path, "p.json", prof)
    assert json.loads(p.read_text())["describe_sha256"] == hashing.describe_sha256(prof["describe"])


def test_load_profile_accepts_a_declared_loose_tolerance(tmp_path):
    """A declared tolerance up to 1e-3 with a reason passes when the measured drift is within it."""
    prof = make_profile()
    prof["invariance"].update({"tolerance": 1e-4, "tolerance_reason": "bf16 attention kernels differ between batch sizes", "batch_rel_l2": 3e-5, "order_rel_l2": 1e-5})
    p = write_profile(tmp_path, "p.json", prof)
    assert load_profile(str(p))["invariance"]["tolerance"] == 1e-4
