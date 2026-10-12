"""Execution side: ``--slots`` device masks and slot processes, ``--range``, ``--root``, ``--cpus``, ``--plan-sha``,
``plan run --local``, the frozen shard layout in ``plan``, shard claims, CPU-only plans, usable-CPU detection and the
3.10 TOML fallback."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from test_plan_helpers import (
    PROTO,
    fake_unit,
    install_benchmarks,
    make_info,
    make_profile,
    python_launcher,
    standard_matrix,
    write_done_probe,
    write_fake_script,
    write_matrix,
    write_profile,
)
from yavih import cli
from yavih.plan import exec as X
from yavih.plan import units as U
from yavih.plan.local import run_local
from yavih.plan.units import plan, write_plan


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root, write_fake_script(root)


def _plan(root: Path, *units, name: str = "plan.jsonl") -> str:
    p = root / name
    write_plan(list(units), str(p))
    return str(p)


def _log(rec: dict) -> str:
    return open(rec["log"]).read()


# ----------------------------------------------------------------------------- slots


def test_device_masks_follow_the_inherited_list():
    assert X.device_masks(2, {}) == ["0", "1"]  # nothing inherited: the node is this job's
    assert X.device_masks(2, {"CUDA_VISIBLE_DEVICES": "4,5"}) == ["4", "5"]  # docker --gpus device=4,5 / non-cgroup SLURM
    assert X.device_masks(1, {"CUDA_VISIBLE_DEVICES": "GPU-aa11, GPU-bb22"}) == ["GPU-aa11"]  # Kubernetes UUIDs
    assert X.device_masks(3, {"CUDA_VISIBLE_DEVICES": ""}) == ["", "", ""]  # explicitly no GPU
    with pytest.raises(X.ExecError, match="lists 2 device"):
        X.device_masks(4, {"CUDA_VISIBLE_DEVICES": "4,5"})


def test_slots_index_the_inherited_devices_in_separate_processes(workspace, monkeypatch):
    root, script = workspace
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    us = [fake_unit(root, script, mode="env", shard=(i, 2), launcher=python_launcher(),
                    env={"CUDA_VISIBLE_DEVICES": "9"} if i == 1 else None) for i in range(2)]
    res = X.exec_unit(_plan(root, *us), index=0, slots=2)
    assert res["exit_code"] == 0 and [r["id"] for r in res["slots"]] == [u.id for u in us]
    # slot g gets the g-th INHERITED device, and a unit's env cannot move it onto another slot's GPU
    assert "CVD=5" in _log(res["slots"][0]) and "CVD=7" in _log(res["slots"][1])
    assert [r["gpu"] for r in res["slots"]] == ["5", "7"]
    # each slot is its own `yavih exec --id` process (probe torch globals and devices never cross)
    assert all(f"[yavih] exec {u.id} extract argv=" in _log(r) for u, r in zip(us, res["slots"], strict=True))


def test_probe_slots_are_processes_too(workspace, monkeypatch):
    root, script = workspace
    seen: list[list[str]] = []
    real = subprocess.Popen

    def spy(argv, *a, **kw):
        if "--id" in argv:
            seen.append(list(argv))
        return real(argv, *a, **kw)

    monkeypatch.setattr(X.subprocess, "Popen", spy)
    ex = fake_unit(root, script, mode="ok", launcher=python_launcher())
    pr = fake_unit(root, script, kind="probe", seed=0, depends_on=["000000000000"])  # not ready: exits 3 without torch work
    res = X.exec_unit(_plan(root, ex, pr), index=0, slots=2)
    assert {a[a.index("--id") + 1] for a in seen} == {ex.id, pr.id}
    assert {r["id"]: r["status"] for r in res["slots"]} == {ex.id: "done", pr.id: "not_ready"} and res["exit_code"] == 3


# ----------------------------------------------------------------------------- --range / --cpus / --plan-sha / --root


def test_range_runs_in_order_and_stops_at_the_deadline(workspace):
    root, script = workspace
    us = [fake_unit(root, script, mode="ok", shard=(i, 4), launcher=python_launcher(), minutes=200 if i == 1 else 1) for i in range(4)]
    plan_p = _plan(root, *us)
    res = X.exec_unit(plan_p, unit_range="0:3", deadline=time.time() + 1200)  # unit 1 needs 3000 s of headroom
    assert [r["id"] for r in res["units"]] == [us[0].id, us[1].id]
    assert res["units"][1]["status"] == "deadline" and res["units"][1]["refused_at_start"] is True
    assert res["not_started"] == [us[2].id] and res["exit_code"] == 4
    assert U.check_done(us[3]) is None
    res = X.exec_unit(plan_p, unit_range="2:", slots=2)  # K at a time over a range
    assert res["exit_code"] == 0 and [r["status"] for r in res["units"]] == ["done", "done"]
    for bad in ("3:1", "x", "0:9"):
        with pytest.raises(X.ExecError):
            X.exec_unit(plan_p, unit_range=bad)
    with pytest.raises(X.ExecError):
        X.exec_unit(plan_p, unit_range="0:1", index=0)


def test_cpus_override_rewrites_decode_workers(workspace):
    root, script = workspace
    u = fake_unit(root, script, mode="env", launcher=python_launcher(), extra_argv=["--decode-workers", "3"])
    last = X.exec_unit(_plan(root, u), index=0, cpus=2)
    assert last["status"] == "done" and "ARGV=--decode-workers 1" in _log(last) and last["needs"]["cpus"] == 2
    with pytest.raises(X.ExecError):
        X.exec_unit(_plan(root, u), index=0, cpus=0)


def test_cpus_default_is_clamped_to_the_usable_cpus(workspace, monkeypatch):
    root, script = workspace
    import yavih.resources as R

    monkeypatch.setattr(R, "available_cpus", lambda env=None: 2)
    u = fake_unit(root, script, mode="env", launcher=python_launcher(), extra_argv=["--decode-workers", "3"])  # needs.cpus 4
    last = X.exec_unit(_plan(root, u), index=0)
    assert "ARGV=--decode-workers 1" in _log(last) and last["needs"]["cpus"] == 2


def test_plan_sha_refuses_a_rebuilt_pending_file(workspace):
    root, script = workspace
    u = fake_unit(root, script, mode="ok", launcher=python_launcher())
    plan_p = _plan(root, u)
    sha = hashlib.sha256(Path(plan_p).read_bytes()).hexdigest()
    assert X.exec_unit(plan_p, index=0, plan_sha=sha[:12])["status"] == "done"
    for bad in ("0" * 12, "xyz", "abc"):
        with pytest.raises(X.ExecError):
            X.exec_unit(plan_p, index=0, plan_sha=bad)


def test_plan_paths_are_relative_to_the_plan_file_and_root_re_anchors_them(workspace, tmp_path):
    root, script = workspace
    u = fake_unit(root, script, mode="ok", launcher=python_launcher())
    u.cmd[1] = os.path.relpath(u.cmd[1], root)  # the fake script by a relative path, as a matrix would write it
    plan_p = _plan(root, u)
    assert json.loads(Path(plan_p).read_text())["cwd"] == "."  # stored relative to the plan file
    moved = tmp_path / "workspace-mounted-elsewhere"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    last = X.exec_unit(str(moved / "plan.jsonl"), index=0)
    assert last["status"] == "done" and (moved / u.done).exists() and last["cwd"] == str(moved)
    # a plan file kept outside the project: --root says where the project is on this machine
    elsewhere = tmp_path / "plans"
    elsewhere.mkdir()
    shutil.copy(moved / "plan.jsonl", elsewhere / "plan.jsonl")
    again = X.exec_unit(str(elsewhere / "plan.jsonl"), index=0, root=str(moved))
    assert again["status"] == "skipped"
    assert U.read_plan(str(elsewhere / "plan.jsonl"), root=str(moved))[0].cwd == str(moved)


def test_inputs_hash_ignores_launcher_env_cwd_and_workers(tmp_path, monkeypatch):
    """A plan built on the host is not stale in a container: launcher, env, cwd and worker counts are logged only."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = standard_matrix(tmp_path / "a", monkeypatch)
    b = standard_matrix(tmp_path / "b", monkeypatch, launcher="/envs/other/bin/python")
    ua = {u.id: u for u in plan(str(a))}
    ub = {u.id: u for u in plan(str(b))}
    assert ua.keys() == ub.keys()
    for uid in ua:
        assert ua[uid].cwd != ub[uid].cwd and ua[uid].inputs_hash == ub[uid].inputs_hash
    cmd = next(u for u in ua.values() if u.kind == "extract").cmd
    other = list(cmd)
    other[other.index("--decode-workers") + 1] = "63"  # exec --cpus
    other[other.index("--out") + 1] = "/workspace/runs/x"  # another mount
    assert U.identity_cmd(other) == U.identity_cmd(cmd)


# ----------------------------------------------------------------------------- plan run --local


def test_plan_run_local_runs_dependencies_first_and_blocks_on_failure(workspace, monkeypatch):
    import yavih.readouts as R

    root, script = workspace
    e0 = fake_unit(root, script, mode="ok", shard=(0, 3), launcher=python_launcher())
    e1 = fake_unit(root, script, mode="ok", shard=(1, 3), launcher=python_launcher())
    bad = fake_unit(root, script, mode="exit2", shard=(2, 3), launcher=python_launcher(), split="val")
    pr = fake_unit(root, script, kind="probe", seed=0, depends_on=[e0.id, e1.id])
    pr_blocked = fake_unit(root, script, kind="probe", seed=1, depends_on=[bad.id])
    order: list[str] = []

    def fake_probe(run_dir, *, seeds=(0,), **kw):
        order.append(f"probe{seeds[0]}")
        assert U.check_done(e0) and U.check_done(e1)  # its extract dependencies finished first
        write_done_probe(pr)
        return []

    monkeypatch.setattr(R, "probe", fake_probe)
    plan_p = _plan(root, pr, pr_blocked, e0, e1, bad)  # listed out of order on purpose
    res = run_local(plan_p)
    by = {r["id"]: r["status"] for r in res["units"]}
    assert by == {e0.id: "done", e1.id: "done", bad.id: "failed", pr.id: "done"} and order == ["probe0"]
    assert res["blocked"] == [pr_blocked.id] and res["exit_code"] == 2 and res["already_done"] == 0
    # rerun: everything done is skipped up front, the failure runs again, its dependent stays blocked
    res = run_local(plan_p)
    assert [r["id"] for r in res["units"]] == [bad.id] and res["already_done"] == 3


def test_plan_run_local_with_slots_and_the_cli(workspace, capsys):
    root, script = workspace
    us = [fake_unit(root, script, mode="ok", shard=(i, 3), launcher=python_launcher()) for i in range(3)]
    plan_p = _plan(root, *us)
    rc = cli.main(["--json", "plan", "run", plan_p])  # --local is required: there is no SLURM mode
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 1 and env["errors"][0]["code"] == "E_USAGE" and "sbatch" in env["errors"][0]["message"]
    rc = cli.main(["--json", "plan", "run", plan_p, "--local", "--slots", "2"])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and env["status"] == "ok" and env["data"]["counts"] == {"done": 3}
    assert all(U.check_done(u) == "done" for u in us)


def test_plan_run_local_stops_new_starts_at_the_deadline(workspace):
    root, script = workspace
    us = [fake_unit(root, script, mode="ok", shard=(i, 3), launcher=python_launcher(), minutes=1 if i == 0 else 500) for i in range(3)]
    res = run_local(_plan(root, *us), deadline=time.time() + 1200)
    assert [r["status"] for r in res["units"]] == ["done", "deadline"] and res["not_started"] == [us[2].id]
    assert res["exit_code"] == 4


# ----------------------------------------------------------------------------- frozen layout in plan


def _features_dir(mp: Path, model="synthmodel", reading="v", split="train") -> Path:
    return mp.parent / "runs" / "synth_clf" / model / reading / "features" / split


def _freeze(split_dir: Path, profile: dict, *, n: int, n_rows: int, batch_size: int = 4) -> dict:
    from yavih.io.layout import build_layout, layout_identity

    ident = layout_identity(manifest_hash=make_info().manifest_hash, input_hash=PROTO, model=profile["model"],
                            reading=profile["reading"], describe_sha256=profile["describe_sha256"])
    rec = build_layout(ident, n=n, batch_size=batch_size, n_rows=n_rows)
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "_layout.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def test_plan_follows_a_frozen_layout_whatever_the_shard_minutes(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch, shard_minutes=1, clips_per_s=0.1)  # estimate: 2 train shards
    est = [u for u in plan(str(mp)) if u.kind == "extract" and u.split == "train"]
    assert len(est) == 2
    prof = make_profile("synth_clf", clips_per_s=0.1)
    rec = _freeze(_features_dir(mp), prof, n=3, n_rows=12)
    for minutes in (1, 600):  # another GPU's clips_per_s or another partition's --shard-minutes: same layout
        got = [u for u in plan(str(mp), shard_minutes=minutes) if u.kind == "extract" and u.split == "train"]
        assert [u.shard for u in got] == [[i, 3] for i in range(3)]
        assert [u.extra["rows"] for u in got] == [list(r) for r in rec["ranges"]]
    # another configuration's layout is not this one's
    _freeze(_features_dir(mp), dict(prof, describe_sha256="0" * 64), n=5, n_rows=12)
    assert len([u for u in plan(str(mp)) if u.kind == "extract" and u.split == "train"]) == 2


def test_matrix_shards_pin_and_its_conflict_with_a_frozen_layout(tmp_path, monkeypatch):
    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    prof = make_profile("synth_clf")
    write_profile(root, "p.json", prof)
    entry = {"script": "b.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"}
    mp = write_matrix(root, {"e": dict(entry, shards={"train": 4})})
    tr = [u for u in plan(str(mp)) if u.kind == "extract" and u.split == "train"]
    assert [u.shard[1] for u in tr] == [4] * 4
    mp = write_matrix(root, {"e": dict(entry, shards=2)})
    assert {u.shard[1] for u in plan(str(mp)) if u.kind == "extract"} == {2}
    _freeze(_features_dir(mp), prof, n=3, n_rows=12)
    with pytest.raises(U.PlanError, match="frozen at n=3"):
        plan(str(mp))
    from yavih.plan.matrix import MatrixError

    for bad in (0, "2", {"train": 0}, True):
        with pytest.raises(MatrixError):
            plan(str(write_matrix(root, {"e": dict(entry, shards=bad)})))


def test_matrix_allow_loopback_reaches_the_extract_cmd(tmp_path, monkeypatch):
    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    write_profile(root, "p.json", make_profile("synth_clf"))
    entry = {"script": "b.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"}
    units = plan(str(write_matrix(root, {"e": dict(entry, allow_loopback=True)})))
    assert all("--allow-loopback" in u.cmd for u in units if u.kind == "extract")
    units = plan(str(write_matrix(root, {"e": entry}, defaults={"allow_loopback": True})))
    assert all("--allow-loopback" in u.cmd for u in units if u.kind == "extract")
    units = plan(str(write_matrix(root, {"e": entry})))
    assert not any("--allow-loopback" in u.cmd for u in units)


def test_cpu_only_profile_plans_no_gpu(tmp_path, monkeypatch):
    from yavih.plan.estimate import gpu_for_profile

    assert gpu_for_profile({"peak_gpu_gb": 0.0}) == 0 and gpu_for_profile({"peak_gpu_gb": 0}) == 0
    assert gpu_for_profile({"peak_gpu_gb": None}) == 1 and gpu_for_profile({"peak_gpu_gb": 1.5}) == 1  # unknown: GPU
    root = tmp_path / "m"
    root.mkdir()
    install_benchmarks(monkeypatch, {"synth_clf": make_info()})
    write_profile(root, "p.json", dict(make_profile("synth_clf"), peak_gpu_gb=0.0))
    units = plan(str(write_matrix(root, {"e": {"script": "b.py", "benchmark": "synth_clf", "args": {"layer": 12}, "profile": "p.json"}})))
    assert {u.needs["gpu"] for u in units} == {0}


# ----------------------------------------------------------------------------- claims


def test_claim_is_exclusive_until_released(tmp_path):
    from yavih.io.claim import ClaimHeld, acquire_claim

    c = acquire_claim(tmp_path, 3)
    assert (tmp_path / "shard-000003.claim").exists() and c.held()
    with pytest.raises(ClaimHeld) as ei:
        acquire_claim(tmp_path, 3)
    assert ei.value.exit_code == 3
    c.release()
    assert not (tmp_path / "shard-000003.claim").exists()
    acquire_claim(tmp_path, 3).release()


def test_stale_claims_are_taken_over_and_the_old_writer_notices(tmp_path):
    from yavih.io.claim import ClaimLost, acquire_claim

    c = acquire_claim(tmp_path, 0, lease_s=60)
    p = tmp_path / "shard-000000.claim"
    old = time.time() - 3600
    os.utime(p, (old, old))  # no heartbeat for an hour: the writer is gone
    new = acquire_claim(tmp_path, 0, lease_s=60)
    assert new.took_over and new.took_over["pid"] == os.getpid()
    with pytest.raises(ClaimLost) as ei:
        c.verify()
    assert ei.value.exit_code == 4
    c.release()  # releasing a lost claim never removes the new owner's
    assert p.exists() and new.held()
    new.release()
    # a claim of a dead process on this host is stale at once
    p.write_text(json.dumps({"token": "x", "host": socket.gethostname(), "pid": 2**22 + 7, "lease_s": 3600}))
    acquire_claim(tmp_path, 0).release()


def test_writer_publishes_nothing_once_its_claim_is_lost(tmp_path):
    from test_io_core import _writer, vmeta
    from yavih.io.claim import ClaimLost, acquire_claim

    split = tmp_path / "features" / "train"
    claim = acquire_claim(split, 0, lease_s=60)
    w = _writer(tmp_path, rows=(0, 1))
    w.claim = claim
    w.write([0], ["a"], np.zeros((1, 6, 16), np.float32), meta=vmeta([0]))
    os.utime(claim.path, (time.time() - 3600,) * 2)
    acquire_claim(split, 0, lease_s=60)  # another writer takes the stale claim over
    with pytest.raises(ClaimLost):
        w.close(timing={}, env={}, canary_stats={}, side_effects={})
    assert sorted(p.name for p in split.iterdir()) == ["shard-000000.claim"]  # no tar, no sidecar, no tmp


# ----------------------------------------------------------------------------- resources / toml


def test_available_cpus_sources(tmp_path, monkeypatch):
    from yavih import resources as R

    assert R.available_cpus({"YAVIH_CPUS": "3"}) == 3
    assert R.available_cpus({"SLURM_CPUS_PER_TASK": "5"}) == 5
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    assert R.cgroup_cpu_quota(tmp_path) == 2
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert R.cgroup_cpu_quota(tmp_path) is None
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("150000")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert R.cgroup_cpu_quota(v1) == 2
    for name in ("YAVIH_CPUS", "SLURM_CPUS_PER_TASK", "NCPUS", "PBS_NUM_PPN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(R, "cgroup_cpu_quota", lambda *a, **kw: 1)
    assert R.available_cpus() == 1  # min(affinity, quota) on the host


def test_vendored_toml_reader_reads_a_site_config():
    from yavih import _toml

    text = '''
# site config
[runs]
root = "~/runs"   # comment
[results]
root = '/srv/results'
[data]
root = "/data"
[data.esc50]
media_root = "/data/esc50 \\u00e9"
[scratch]
root = """/scratch/
me"""
[resources]
cpus = 1_6
ratio = 0.5
flags = [true, false,
  "x",]
inline = {a = 1, b.c = "d"}
[[site]]
id = "a"
[[site]]
id = "b"
'''
    cfg = _toml.loads(text)
    assert cfg["runs"]["root"] == "~/runs" and cfg["results"]["root"] == "/srv/results"
    assert cfg["data"]["esc50"]["media_root"] == "/data/esc50 é" and cfg["scratch"]["root"] == "/scratch/\nme"
    assert cfg["resources"] == {"cpus": 16, "ratio": 0.5, "flags": [True, False, "x"], "inline": {"a": 1, "b": {"c": "d"}}}
    assert [s["id"] for s in cfg["site"]] == ["a", "b"]
    if sys.version_info >= (3, 11):
        import tomllib

        assert tomllib.loads(text) == cfg
    for bad in ("a = ", "[a]\n[a]", "a = 1\na = 2", 'a = "x', "a = 2026-10-10"):
        with pytest.raises(_toml.TOMLDecodeError):
            _toml.loads(bad)
