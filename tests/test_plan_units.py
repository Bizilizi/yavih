"""Unit records: determinism, ids, dependency graph, plan-time not_applicable,
--pending staleness, --kind, jsonl round trip, calibration factors."""

from __future__ import annotations

import json
import os

import pytest

from test_plan_helpers import (
    PROTO,
    QH,
    install_benchmarks,
    make_info,
    make_profile,
    standard_matrix,
    write_done_extract,
    write_done_probe,
    write_last,
    write_matrix,
    write_profile,
)
from yavih import cli, hashing
from yavih.plan import units as U
from yavih.plan.units import PlanError, Unit, plan, read_plan, write_plan


def test_plan_is_deterministic_and_topological(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch, seeds=[0, 1], shard_minutes=1, clips_per_s=0.05)  # 0.05*60 = 3 rows/shard
    a = [u.to_json() for u in plan(str(mp))]
    b = [u.to_json() for u in plan(str(mp))]
    assert a == b
    kinds = [u["kind"] for u in a]
    assert kinds == sorted(kinds, key=lambda k: U.KIND_ORDER[k])
    extracts = [u for u in a if u["kind"] == "extract"]
    probes = [u for u in a if u["kind"] == "probe"]
    reports = [u for u in a if u["kind"] == "report"]
    assert len(reports) == 1 and len(probes) == 2
    # train 12 rows → 4 shards of 3 rows (batch 4 → 3 batches dealt over 4 shards → one empty), val 6 → 2 shards
    assert {(u["split"], u["shard"][1]) for u in extracts} == {("train", 4), ("val", 2)}
    ids = [u["id"] for u in a]
    assert len(set(ids)) == len(ids)
    pos = {u["id"]: i for i, u in enumerate(a)}
    for u in a:
        for d in u["depends_on"]:
            assert pos[d] < pos[u["id"]], "dependencies precede dependents"
    assert sorted(probes[0]["depends_on"]) == sorted(x["id"] for x in extracts)
    assert sorted(reports[0]["depends_on"]) == sorted(p["id"] for p in probes)
    assert {p["seed"] for p in probes} == {0, 1}


def test_unit_record_fields(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch, launcher="/envs/v/bin/python")
    units = plan(str(mp))
    ex = next(u for u in units if u.kind == "extract" and u.split == "train")
    prof = json.loads((tmp_path / "m" / "bridges" / "synth.profile.json").read_text())
    rec = ex.to_json()
    for key in ["schema_version", "id", "kind", "benchmark", "bridge", "model", "reading", "split", "shard", "seed", "protocol_hash",
                "probe_hash", "eval_hash", "depends_on", "needs", "cmd", "launcher", "env", "cwd", "out", "done", "inputs_hash",
                "expected_status", "describe_sha256", "input_hash"]:
        assert key in rec, key
    assert rec["schema_version"] == 3 and rec["expected_status"] == "ok" and rec["probe_hash"] is None and rec["seed"] is None
    # an extract unit is keyed by the benchmark's input side, not by its protocol_hash
    ih = rec["input_hash"]
    assert isinstance(ih, str) and len(ih) == 64
    assert rec["eval_hash"] is None and rec["describe_sha256"] == prof["describe_sha256"]
    assert (rec["model"], rec["reading"]) == (prof["model"], prof["reading"]) == ("synthmodel", "v")
    assert rec["bridge"] == "synth-clf" and rec["benchmark"] == "synth_clf"
    run_dir = os.path.join("runs", "synth_clf", "synthmodel", "v")
    assert rec["launcher"] == "/envs/v/bin/python" and rec["env"] == {"HF_HUB_OFFLINE": "1"} and rec["cwd"] == str(tmp_path / "m")
    assert set(rec["needs"]) == {"gpu", "cpus", "mem_gb", "scratch_gb", "minutes"} and rec["needs"]["cpus"] == 4 and rec["needs"]["mem_gb"] == 32
    cmd = rec["cmd"]
    assert cmd[:3] == ["python", "-m", "yavih.extract"]
    for flag, val in [("--bench", "synth_clf"), ("--bridge", "bridges/synth.py"), ("--split", "train"), ("--shard", "0/1"),
                      ("--out", run_dir), ("--arg", "layer=12"), ("--canaries", "8"),
                      ("--decode-workers", "3"), ("--expect-profile", "bridges/synth.profile.json")]:
        assert val == cmd[cmd.index(flag) + 1], flag
    assert "--expect-model-id" not in cmd and "--weak-identity" not in cmd  # --expect-profile alone
    assert rec["out"] == os.path.join(run_dir, "features", "train")
    # `--out` is the run dir: the child writes `<out>/features/<split>/shard-*.json` = `done`
    assert rec["out"] == os.path.join(cmd[cmd.index("--out") + 1], "features", "train")
    assert rec["done"] == os.path.join(rec["out"], "shard-000000.json")
    # unit identities
    assert rec["id"] == hashing.unit_id(kind="extract", benchmark="synth_clf", model="synthmodel", reading="v", split="train",
                                        shard=[0, 1], protocol_hash=None, input_hash=ih, probe_hash=None, seed=None)
    assert rec["inputs_hash"] == hashing.inputs_hash(cmd=U.identity_cmd(cmd), protocol_hash=None, input_hash=ih, probe_hash=None,
                                                     model="synthmodel", reading="v", describe_sha256=prof["describe_sha256"])
    # host paths and worker counts are logged, never hashed
    ident = U.identity_cmd(cmd)
    assert "--out" not in ident and "--decode-workers" not in ident and "--bridge" not in ident and "--expect-profile" not in ident
    assert ident[:2] == ["-m", "yavih.extract"] and "--canaries" in ident and "--shard" in ident
    pr = next(u for u in units if u.kind == "probe")
    eh = hashing.eval_hash(PROTO, QH)
    assert pr.probe_hash == QH and pr.eval_hash == eh and pr.launcher is None and pr.needs["gpu"] == 1 and pr.needs["minutes"] >= 30
    # probe/<eval_hash12>/seed<k>/DONE
    assert pr.done == os.path.join(run_dir, "probe", eh[:12], "seed0", "DONE")
    assert pr.cmd[:5] == ["python", "-m", "yavih.cli", "probe", "run"] and pr.cmd[5] == run_dir
    assert pr.id == hashing.unit_id(kind="probe", benchmark="synth_clf", model="synthmodel", reading="v", split=None, shard=None,
                                    protocol_hash=PROTO, probe_hash=QH, seed=0)
    rp = next(u for u in units if u.kind == "report")
    assert rp.needs["gpu"] == 0 and rp.done == os.path.join("runs", "_report", "DONE")


def test_shard_minutes_override_changes_shard_count(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch, n_train=120, clips_per_s=1.0)
    one = [u for u in plan(str(mp)) if u.kind == "extract" and u.split == "train"]
    assert len(one) == 1  # 120 rows at 1 clip/s = 2 min < 30
    many = [u for u in plan(str(mp), shard_minutes=1) if u.kind == "extract" and u.split == "train"]
    assert len(many) == 2  # 60 rows per shard-minute
    for u in many:
        assert u.shard[1] == 2


def test_plan_time_not_applicable_dense_readout_on_pooled_bridge(tmp_path, monkeypatch):
    root = tmp_path / "m"
    root.mkdir()
    infos = {"synth_seq": make_info("synth_seq", readout="seq", requires_pos=("t",), requires_streams=("video",))}
    install_benchmarks(monkeypatch, infos)
    write_profile(root, "p.json", make_profile("synth_seq", N=1, pos_mode="none"))
    mp = write_matrix(root, {"e": {"script": "s.py", "benchmark": "synth_seq", "args": {"layer": 12}, "profile": "p.json"}})
    units = plan(str(mp))
    assert [u.kind for u in units] == ["probe", "report"], "no extract units for a planned not_applicable"
    pr = units[0]
    assert pr.expected_status == "not_applicable" and pr.depends_on == [] and pr.needs["gpu"] == 0
    assert pr.not_applicable["decided_by"] == "plan" and "pos.t" in pr.not_applicable["missing"]
    assert pr.profile == "p.json"


def test_plan_time_not_applicable_missing_required_stream(tmp_path, monkeypatch):
    root = tmp_path / "m"
    root.mkdir()
    infos = {"synth_box": make_info("synth_box", readout="box", requires_pos=("y", "x"), requires_streams=("video", "audio"))}
    install_benchmarks(monkeypatch, infos)
    write_profile(root, "p.json", make_profile("synth_box", N=64, pos_mode="constant", streams=["video"]))
    mp = write_matrix(root, {"e": {"script": "s.py", "benchmark": "synth_box", "args": {"layer": 12}, "profile": "p.json"}})
    pr = plan(str(mp), kind="probe")[0]
    assert pr.expected_status == "not_applicable" and pr.not_applicable["missing"] == ["stream.audio"]


def test_clf_on_pooled_bridge_is_applicable(tmp_path, monkeypatch):
    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    write_profile(root, "p.json", make_profile("synth_clf", N=1, pos_mode="none"))
    mp = write_matrix(root, {"e": {"script": "s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"}})
    units = plan(str(mp))
    assert {u.kind for u in units} == {"extract", "probe", "report"}
    assert all(u.expected_status == "ok" for u in units)


def test_plan_refuses_profile_mismatch_and_duplicate_identity(tmp_path, monkeypatch):
    from yavih.plan.matrix import ProfileError

    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    write_profile(root, "p.json", make_profile("k400"))
    mp = write_matrix(root, {"e": {"script": "s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"}})
    with pytest.raises(ProfileError):
        plan(str(mp))
    # two entries, one (benchmark, model, reading) → one run directory → refused
    write_profile(root, "p.json", make_profile("synth_clf"))
    mp = write_matrix(root, {
        "e1": {"script": "s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"},
        "e2": {"script": "s2.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"},
    })
    with pytest.raises(PlanError, match="entries 'e1' and 'e2' are both synthmodel/v on synth_clf"):
        plan(str(mp))


def test_plan_kind_filter_and_unknown_kind(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch)
    assert all(u.kind == "extract" for u in plan(str(mp), kind="extract"))
    assert len(plan(str(mp), kind="report")) == 1
    with pytest.raises(PlanError):
        plan(str(mp), kind="nope")


def test_pending_skips_done_and_reemits_stale(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch)
    units = plan(str(mp))
    assert len(plan(str(mp), pending=True)) == len(units)
    ex = [u for u in units if u.kind == "extract"]
    write_done_extract(ex[0])
    write_last(ex[0])
    pend = plan(str(mp), pending=True)
    assert {u.id for u in pend} == {u.id for u in units} - {ex[0].id}
    # a DONE marker with a different protocol hash is not "done" for this plan
    write_done_extract(ex[1], protocol_hash="9" * 64)
    assert ex[1].id in {u.id for u in plan(str(mp), pending=True)}
    # last.json inputs_hash drift → stale → re-emitted
    write_last(ex[0], inputs_hash="0" * 64, needs={"gpu": 1, "cpus": 99, "mem_gb": 1, "scratch_gb": 0, "minutes": 1})
    assert ex[0].id in {u.id for u in plan(str(mp), pending=True)}
    # ... but a record of another inputs_hash scheme is never stale on that ground
    write_last(ex[0], inputs_hash="0" * 64, inputs_scheme=U.INPUTS_SCHEME + 1)
    assert ex[0].id not in {u.id for u in plan(str(mp), pending=True)}
    # ... and the resource estimate, env or launcher are not part of the hash at all
    write_last(ex[0], needs=dict(ex[0].needs, minutes=ex[0].needs["minutes"] * 3))
    assert ex[0].id not in {u.id for u in plan(str(mp), pending=True)}
    # probe DONE with matching hashes → not pending; mismatching probe_hash → pending
    pr = next(u for u in units if u.kind == "probe")
    write_done_probe(pr)
    assert pr.id not in {u.id for u in plan(str(mp), pending=True)}
    write_done_probe(pr, probe_hash="1" * 64)
    assert pr.id in {u.id for u in plan(str(mp), pending=True)}


def test_pending_not_applicable_marker_counts_as_done(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch)
    ex = [u for u in plan(str(mp)) if u.kind == "extract" and u.split == "val"]
    out = U.resolve_path(ex[0], ex[0].out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "NOT_APPLICABLE.json"), "w") as fh:
        json.dump({"status": "not_applicable", "model": ex[0].model, "reading": ex[0].reading,
                   "describe_sha256": ex[0].describe_sha256, "reason": "x"}, fh)
    assert U.check_done(ex[0]) == "not_applicable"
    assert ex[0].id not in {u.id for u in plan(str(mp), pending=True)}


def test_write_read_plan_roundtrip_and_schema_refusal(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch, seeds=[0, 1])
    units = plan(str(mp))
    p = tmp_path / "plan.jsonl"
    write_plan(units, str(p))
    lines = p.read_text().strip().splitlines()
    assert len(lines) == len(units) and all(json.loads(line)["schema_version"] == 3 for line in lines)
    back = read_plan(str(p))
    assert [u.to_json() for u in back] == [u.to_json() for u in units]
    rec = json.loads(lines[0])
    rec["schema_version"] = 99
    with pytest.raises(PlanError, match="newer"):
        Unit.from_json(rec)
    # an older schema's unit ids are not this yavih's: rebuild the plan
    rec = json.loads(lines[0])
    rec["schema_version"] = rec["schema_version"] - 1
    with pytest.raises(PlanError, match="rebuild"):
        Unit.from_json(rec)
    rec = json.loads(lines[0])
    rec["kind"] = "dance"
    with pytest.raises(PlanError, match="kind"):
        Unit.from_json(rec)
    rec = json.loads(lines[0])
    rec["unknown_future_key"] = 1  # forward-compatible: kept in .extra, not in to_json
    u = Unit.from_json(rec)
    assert u.extra == {"unknown_future_key": 1} and "unknown_future_key" not in u.to_json()
    (tmp_path / "dup.jsonl").write_text(lines[0] + "\n" + lines[0] + "\n")
    with pytest.raises(PlanError, match="duplicate"):
        read_plan(str(tmp_path / "dup.jsonl"))
    with pytest.raises(PlanError):
        read_plan(str(tmp_path / "missing.jsonl"))


def test_calibration_file_scales_minutes(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch)
    base = {u.id: u.needs["minutes"] for u in plan(str(mp))}
    ex_id = next(u.id for u in plan(str(mp)) if u.kind == "extract")
    runs = tmp_path / "m" / "runs"
    runs.mkdir()
    (runs / "_calibration.json").write_text(json.dumps({
        "schema": 1, "multipliers": {"extract_minutes": 2.0, "probe_minutes": 1.0},
        "units": {ex_id: {"minutes": 1.5, "bumped_at": "x", "reason": "exit 4"}},
    }))
    after = {u.id: u.needs["minutes"] for u in plan(str(mp))}
    for uid, m in base.items():
        if uid == ex_id:
            assert after[uid] >= m * 1.5
        elif after[uid] != m:
            assert after[uid] >= m  # the global multiplier only ever raises here


def test_cli_plan_build_count_and_out(tmp_path, monkeypatch, capsys):
    mp = standard_matrix(tmp_path, monkeypatch, seeds=[0, 1])
    rc = cli.main(["--json", "plan", "build", str(mp), "--format", "count"])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and env["status"] == "ok" and env["data"]["count"] == len(plan(str(mp)))
    out = tmp_path / "plan.jsonl"
    rc = cli.main(["--json", "plan", "build", str(mp), "--out", str(out), "--kind", "probe"])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and env["paths"]["plan"] == str(out) and len(env["data"]) == 2
    assert len(read_plan(str(out))) == 2
    rc = cli.main(["plan", "build", str(mp), "--format", "count"])
    assert rc == 0 and capsys.readouterr().out.strip() == str(len(plan(str(mp))))


def test_cli_plan_build_matrix_error_exit_1(tmp_path, capsys):
    p = tmp_path / "matrix.yaml"
    p.write_text("schema_version: 1\nbogus: 1\nbridges: {a: {script: s, benchmark: k400, profile: p}}\n")
    rc = cli.main(["--json", "plan", "build", str(p)])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 1 and env["status"] == "error" and env["errors"][0]["code"] == "E_MATRIX"


def test_benchmark_info_projection_from_a_benchmark_like_object(monkeypatch):
    """``benchmark_info`` reads the ``Benchmark`` surface (``spec``, ``roles()``, ``n_rows``, hashes) and the pinned
    config through ``ProbeConfig.pinned``; a duck-typed stand-in pins the projection."""
    import types

    from yavih.probes import config as pc

    class FakeSpec:
        readout = "clf"
        aliases = ("alias1",)
        budget = types.SimpleNamespace(probe_minutes_ref={"clf": 20})
        requires = types.SimpleNamespace(pos=(), streams=())
        streams = {"video": None}

    class FakeBench:
        name = "fakeb"
        spec = FakeSpec()
        manifest_hash = "1" * 64
        protocol_hash = "2" * 64
        input_hash = "3" * 64

        def roles(self):
            return {"train": ["train"], "eval": ["val"], "floor": [], "folds": None}

        def n_rows(self, split):
            return {"train": 10, "val": 5}[split]

    cfg = pc.ProbeConfig(readout="clf", head={"family": "attentive_probe_v1"}, lr=1e-4, lr_grid=(), weight_decay=1e-4, max_epochs=30,
                         batch_size=256, patience=5, warmup_fraction=0.0, loss="ce")
    monkeypatch.setattr(pc.ProbeConfig, "pinned", staticmethod(lambda readout, bench=None: cfg))
    info = U.benchmark_info(FakeBench())
    assert info.name == "fakeb" and info.names == ("fakeb", "alias1") and info.n_rows == {"train": 10, "val": 5}
    assert info.probe_hash == hashing.probe_hash(cfg) and info.probe_minutes_ref == 20.0
    assert info.role_splits() == ["train", "val"] and info.train_splits() == ["train"]


# ----------------------------------------------------------------------------- hashing, ids and paths


def test_bridge_configuration_never_enters_ids_or_paths(tmp_path, monkeypatch):
    """Changing a ``describe()`` key (tap, rule, precision, dtype, batch size, weights)
    changes ``describe_sha256`` — and with it ``inputs_hash`` (staleness) — but no unit id, no run directory and no
    probe directory."""
    mp = standard_matrix(tmp_path, monkeypatch, seeds=[0, 1])
    before = plan(str(mp))
    variants = [
        {"layer": 7},
        {"weights_sha256": None},
        {"weights_sha256": "1" * 64},
        {"precision": {"autocast": "bf16", "matmul_precision": "high", "allow_tf32": True, "deterministic": False}},
        {"pooling": "mean"},
        {"input": {"video": {"temporal": {"rule": "custom", "description": "every second frame of the first 4 s"},
                             "spatial": {"rule": "custom", "description": "letterbox to 256"}}}},
    ]
    root = tmp_path / "m"
    for over in variants:
        prof = make_profile("synth_clf", describe_over=over)
        assert prof["describe_sha256"] != before[0].describe_sha256
        write_profile(root, "bridges/synth.profile.json", prof)
        after = plan(str(mp))
        assert [u.id for u in after] == [u.id for u in before], over
        assert [(u.out, u.done) for u in after] == [(u.out, u.done) for u in before], over
        assert [u.eval_hash for u in after] == [u.eval_hash for u in before]
        assert all(u.describe_sha256 == prof["describe_sha256"] for u in after if u.kind != "report")
        ex_b = [u.inputs_hash for u in before if u.kind == "extract"]
        ex_a = [u.inputs_hash for u in after if u.kind == "extract"]
        assert all(a != b for a, b in zip(ex_a, ex_b, strict=True)), "describe_sha256 is in inputs_hash (staleness)"


def test_dtype_and_batch_size_never_change_the_run_or_probe_dir(tmp_path, monkeypatch):
    """dtype and batch size change shard plans (bytes per row, batch alignment) but never a path's model part."""
    mp = standard_matrix(tmp_path, monkeypatch)
    before = {u.kind: u for u in plan(str(mp))}
    write_profile(tmp_path / "m", "bridges/synth.profile.json", make_profile("synth_clf", describe_over={"dtype": "float32", "batch_size": 2}))
    after = {u.kind: u for u in plan(str(mp))}
    assert after["probe"].done == before["probe"].done and after["probe"].id == before["probe"].id
    assert os.path.dirname(after["extract"].out) == os.path.dirname(before["extract"].out)


def test_probe_done_of_another_configuration_is_pending(tmp_path, monkeypatch):
    """A probe DONE written under another ``describe_sha256`` (same model name, reconfigured bridge) is
    not done; so is one of another model or reading."""
    mp = standard_matrix(tmp_path, monkeypatch)
    pr = next(u for u in plan(str(mp)) if u.kind == "probe")
    write_done_probe(pr)
    assert U.check_done(pr) == "done" and pr.id not in {u.id for u in plan(str(mp), pending=True)}
    write_done_probe(pr, describe_sha256="7" * 64)
    assert U.check_done(pr) is None and pr.id in {u.id for u in plan(str(mp), pending=True)}
    write_done_probe(pr, reading="a")
    assert U.check_done(pr) is None
    write_done_probe(pr, model="other")
    assert U.check_done(pr) is None
    # the record carries describe_sha256, so exec (which reads plan.jsonl, not the profile) runs the same check
    p = tmp_path / "plan.jsonl"
    write_plan([pr], str(p))
    back = read_plan(str(p))[0]
    assert back.describe_sha256 == pr.describe_sha256 and U.check_done(back) is None
    write_done_probe(pr)
    assert U.check_done(back) == "done"


def test_extract_sidecar_of_another_configuration_is_pending(tmp_path, monkeypatch):
    """A sidecar (or NOT_APPLICABLE.json) of another ``describe_sha256``, model or reading
    does not count as the unit's DONE: ``run_extract`` moves it to ``_stale/`` and re-extracts."""
    mp = standard_matrix(tmp_path, monkeypatch)
    ex = next(u for u in plan(str(mp)) if u.kind == "extract")
    write_done_extract(ex)
    assert U.check_done(ex) == "done"
    write_done_extract(ex, describe_sha256="7" * 64)
    assert U.check_done(ex) is None and ex.id in {u.id for u in plan(str(mp), pending=True)}
    write_done_extract(ex, reading="a")
    assert U.check_done(ex) is None
    write_done_extract(ex, model="other")
    assert U.check_done(ex) is None
    os.remove(U.resolve_path(ex, ex.done))
    na_path = os.path.join(U.resolve_path(ex, ex.out), "NOT_APPLICABLE.json")
    with open(na_path, "w") as fh:
        json.dump({"status": "not_applicable", "model": ex.model, "reading": ex.reading, "describe_sha256": "7" * 64}, fh)
    assert U.check_done(ex) is None


def test_two_readings_of_one_model_plan_side_by_side(tmp_path, monkeypatch):
    """Readings must not collide — ``a`` and ``v`` of one model on one benchmark are two run dirs."""
    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    write_profile(root, "pv.json", make_profile("synth_clf", reading="v"))
    write_profile(root, "pa.json", make_profile("synth_clf", reading="a"))
    mp = write_matrix(root, {
        "ev": {"script": "s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "pv.json"},
        "ea": {"script": "s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "pa.json"},
    })
    probes = [u for u in plan(str(mp)) if u.kind == "probe"]
    assert {u.out for u in probes} == {os.path.join("runs", "synth_clf", "synthmodel", r) for r in ("a", "v")}
    assert len({u.id for u in probes}) == 2 and len({u.eval_hash for u in probes}) == 1


def test_plan_refuses_collision_through_an_alias(tmp_path, monkeypatch):
    """The run directory uses the canonical benchmark name, so an alias entry collides with a canonical one."""
    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info(aliases=("synth_alias",))})
    write_profile(root, "p1.json", make_profile("synth_clf"))
    write_profile(root, "p2.json", make_profile("synth_alias", describe_over={"layer": 3}))
    mp = write_matrix(root, {
        "canon": {"script": "s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p1.json"},
        "alias": {"script": "s.py", "benchmark": "synth_alias", "args": {"layer": 12}, "profile": "p2.json"},
    })
    with pytest.raises(PlanError, match="both synthmodel/v on synth_clf") as ei:
        plan(str(mp))
    assert ei.value.exit_code == 1


@pytest.mark.parametrize("names", [(("ModelA", "v"), ("modela", "v")), (("m1", "A"), ("m1", "a")), (("Vjepa", "AV"), ("vJEPA", "av"))])
def test_plan_refuses_a_collision_that_differs_only_in_case(tmp_path, monkeypatch, names):
    """Model and reading are path components, and on a case-insensitive filesystem `ModelA` and
    `modela` (or readings `A` and `a`) are ONE run directory and one results directory; the two entries' extract units
    would move each other's shards to _stale/. The collision check ignores case."""
    (m_a, r_a), (m_b, r_b) = names
    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    write_profile(root, "p1.json", make_profile("synth_clf", model=m_a, reading=r_a))
    write_profile(root, "p2.json", make_profile("synth_clf", model=m_b, reading=r_b, describe_over={"layer": 3}))
    mp = write_matrix(root, {
        "e1": {"script": "s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p1.json"},
        "e2": {"script": "s2.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p2.json"},
    })
    with pytest.raises(PlanError, match="equal ignoring case") as ei:
        plan(str(mp))
    assert ei.value.exit_code == 1 and f"{m_a}/{r_a}" in str(ei.value) and f"{m_b}/{r_b}" in str(ei.value)
    # names that differ in more than case still plan side by side
    write_profile(root, "p2.json", make_profile("synth_clf", model=m_b + "x", reading=r_b, describe_over={"layer": 3}))
    assert {u.kind for u in plan(str(mp))} == {"extract", "probe", "report"}


def test_cli_plan_build_collision_exit_1(tmp_path, monkeypatch, capsys):
    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    write_profile(root, "p.json", make_profile("synth_clf"))
    mp = write_matrix(root, {
        "e1": {"script": "s.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"},
        "e2": {"script": "s2.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"},
    })
    rc = cli.main(["--json", "plan", "build", str(mp)])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 1 and env["errors"][0]["code"] == "E_PLAN" and "another model name or reading" in env["errors"][0]["message"]


def test_plan_refuses_a_profile_without_model_and_reading(tmp_path, monkeypatch, capsys):
    """A profile without ``model``/``reading`` is refused (exit 1, re-run bridge check)."""
    mp = standard_matrix(tmp_path, monkeypatch)
    prof = make_profile("synth_clf")
    prof.pop("model")
    prof.pop("reading")
    write_profile(tmp_path / "m", "bridges/synth.profile.json", prof)
    rc = cli.main(["--json", "plan", "build", str(mp)])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 1 and env["errors"][0]["code"] == "E_PROFILE" and "bridge check" in env["errors"][0]["message"]


def test_no_census_is_a_warning_that_changes_no_unit(tmp_path, monkeypatch, capsys):
    """``W_NO_CENSUS``: a video stream guarantees ``n_frames`` and the site has no census → one warning per benchmark
    (stderr and the envelope's ``warnings``); the units are those of a site with the census."""
    import dataclasses

    from test_plan_helpers import make_info

    base = make_info()
    mp = standard_matrix(tmp_path, monkeypatch, infos={"synth_clf": base})
    with_census = [u.to_json() for u in plan(str(mp))]
    install_benchmarks(monkeypatch, {"synth_clf": dataclasses.replace(base, uncensused_frame_streams=("video",))})
    warnings: list[dict] = []
    assert [u.to_json() for u in plan(str(mp), warnings=warnings)] == with_census
    assert [w["code"] for w in warnings] == [U.W_NO_CENSUS] and warnings[0]["where"] == "synth_clf"
    assert "bench verify synth_clf --census" in warnings[0]["message"]
    rc = cli.main(["--json", "plan", "build", str(mp)])
    env = json.loads(capsys.readouterr().out)
    assert rc == 0 and [w["code"] for w in env["warnings"]] == [U.W_NO_CENSUS]


def test_uncensused_frame_streams_of_a_packaged_benchmark(tmp_path, monkeypatch):
    from yavih.benchmark import Benchmark
    from yavih.sample import media_info_path

    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "home"))
    ssv2, esc50 = Benchmark.load("ssv2"), Benchmark.load("esc50")
    assert U.uncensused_frame_streams(ssv2) == ("video",)  # webm: n_frames guaranteed, no census here
    assert U.uncensused_frame_streams(esc50) == ()  # audio only
    census = media_info_path(ssv2.name, ssv2.manifest_hash)
    census.parent.mkdir(parents=True)
    census.write_bytes(b"")
    assert U.uncensused_frame_streams(ssv2) == ()
