"""Results-store root, ``eval run --out``, ``--allow-loopback`` and ``eval run --device``.

- The results-store root is never guessed from directory depth: ``--results`` → ``YAVIH_RESULTS`` /
  ``[results] root`` → ``results/`` next to the runs root of a STANDARD run directory
  ``<runs root>/<benchmark>/<model>/<reading>``; any other layout is ``E_RESULTS_ROOT`` (exit 1) before training.
- ``eval run --out DIR`` probes a read-only feature set (shards mirrored as symlinks, ``_meta.json`` and
  ``probe/`` written under DIR, the source never touched).
- ``--allow-loopback`` lets ``encode`` reach 127.0.0.0/8 / ::1 (recorded in the sidecar); any other connect
  is still ``E_NETWORK``.
"""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import pytest

from test_extract_fixtures import FakeBenchmark, NetworkBridge, NumpyBridge
from test_readouts import MODEL, READING, clf_cfg, simple_clf_world
from test_readouts import (
    fakes as fakes,  # noqa: F401 - the fixture, re-exported for this module
)
from yavih import cli, extract
from yavih.extract import E_NETWORK, ExtractError, run_extract
from yavih.io.sidecar import read_sidecar
from yavih.readouts import base, probe
from yavih.readouts.base import ProbeRefused, mirror_features, resolve_results_root


@pytest.fixture(autouse=True)
def _no_site_results(monkeypatch):
    monkeypatch.delenv("YAVIH_RESULTS", raising=False)
    monkeypatch.delenv("YAVIH_DEADLINE", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def _tree(d: Path) -> dict[str, float]:
    return {p.relative_to(d).as_posix(): p.lstat().st_mtime for p in sorted(d.rglob("*"))}


# ============================================================================ 3.4 results root


def test_standard_layout_results_next_to_the_runs_root(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)  # tmp/runs/<bench>/<model>/<reading>
    res = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    assert Path(res.results_file).parent == tmp_path / "results" / MODEL


def test_explicit_and_configured_root_win(tmp_path, fakes, monkeypatch):
    bench, run = simple_clf_world(tmp_path)
    res = probe(run, bench=bench, cfg=clf_cfg(3), results_root=tmp_path / "mine")[0]
    assert Path(res.results_file).parent == tmp_path / "mine" / MODEL
    monkeypatch.setenv("YAVIH_RESULTS", str(tmp_path / "env"))
    assert resolve_results_root(run, bench=bench, model=MODEL, reading=READING) == tmp_path / "env"
    # a non-standard layout is fine once the root is configured
    assert resolve_results_root(tmp_path / "a" / "b", bench=bench, model=MODEL, reading=READING) == tmp_path / "env"


def test_non_standard_layout_is_refused_before_training(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    odd = tmp_path / "a" / "b"
    odd.parent.mkdir(parents=True)
    run.rename(odd)
    with pytest.raises(ProbeRefused) as ei:
        probe(odd, bench=bench, cfg=clf_cfg(3))
    assert ei.value.code == "E_RESULTS_ROOT" and ei.value.exit_code == 1 and "--results" in str(ei.value)
    assert fakes == [], "nothing may be trained before the results root is known"
    assert not (odd / "probe").exists()
    # --no-results needs no root; --results makes the same layout acceptable
    assert probe(odd, bench=bench, cfg=clf_cfg(3), write_results=False)[0].status == "ok"
    res = probe(odd, bench=bench, cfg=clf_cfg(3), results_root=tmp_path / "r")[0]
    assert res.skipped and Path(res.results_file).parent == tmp_path / "r" / MODEL


def test_relative_out_dir_is_judged_on_its_absolute_path(tmp_path, fakes, monkeypatch):
    """A relative ``--out`` is resolved to an absolute path before the results root is derived."""
    bench, run = simple_clf_world(tmp_path)
    (tmp_path / "runs" / "x").parent.mkdir(exist_ok=True)
    run.rename(tmp_path / "runs" / "x")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ProbeRefused) as ei:
        probe("runs/x", bench=bench, cfg=clf_cfg(3))
    assert ei.value.code == "E_RESULTS_ROOT"
    assert resolve_results_root(Path("runs") / bench.name / MODEL / READING, bench=bench, model=MODEL, reading=READING) == tmp_path / "results"


def test_results_store_write_failure_is_never_fatal(tmp_path, fakes, monkeypatch, capsys):
    from yavih.report import store

    def boom(*a, **k):
        raise KeyError("broken store")

    monkeypatch.setattr(store, "write_invocation", boom)
    bench, run = simple_clf_world(tmp_path)
    res = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    assert res.status == "ok" and res.results_file is None
    assert "could not write the results store" in capsys.readouterr().err


# ============================================================================ 2.6 eval run --out


def test_probe_out_on_a_read_only_source(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    before = _tree(run)
    out = tmp_path / "scratch" / "probe_out"
    modes = {}
    for p in [run, *run.rglob("*")]:
        modes[p] = p.stat().st_mode
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        res = probe(run, bench=bench, cfg=clf_cfg(3), out=out)[0]
    finally:
        for p, m in sorted(modes.items(), key=lambda kv: len(kv[0].parts)):
            p.chmod(m)
    assert res.status == "ok" and Path(res.path).is_relative_to(out)
    assert _tree(run) == before, "the source run directory must not be written"
    for split in ("train", "val"):
        shard = out / "features" / split / "shard-000000.tar"
        assert shard.is_symlink() and os.readlink(shard) == os.fspath(run / "features" / split / "shard-000000.tar")
        meta = out / "features" / split / "_meta.json"
        assert meta.is_file() and not meta.is_symlink()
    # the results root follows the SOURCE layout (runs/<bench>/<model>/<reading>), not the arbitrary --out
    assert Path(res.results_file).parent == tmp_path / "results" / MODEL


def test_mirror_refresh_adds_new_and_drops_vanished_links(tmp_path):
    src = tmp_path / "src"
    split = src / "features" / "train"
    split.mkdir(parents=True)
    for name in ("shard-000000.tar", "shard-000000.json", "shard-000001.tar.host.1.tmp"):
        (split / name).write_text(name)
    (split / "_stale").mkdir()
    out = mirror_features(src, tmp_path / "out")
    mirrored = out / "features" / "train"
    assert sorted(p.name for p in mirrored.iterdir()) == ["shard-000000.json", "shard-000000.tar"]
    (split / "shard-000000.json").unlink()
    (split / "shard-000001.json").write_text("x")
    (split / "NOT_APPLICABLE.json").write_text("{}")
    mirror_features(src, out)
    assert sorted(p.name for p in mirrored.iterdir()) == ["NOT_APPLICABLE.json", "shard-000000.tar", "shard-000001.json"]


def test_cli_probe_run_passes_out_and_device(monkeypatch, capsys):
    import yavih.readouts as RO

    seen = {}

    def fake_probe(run_dir, **kw):
        seen.update(kw, run_dir=run_dir)
        return []

    monkeypatch.setattr(RO, "probe", fake_probe)
    assert cli.main(["eval", "run", "/r", "--out", "/o", "--device", "cpu"]) == 0
    assert seen["out"] == "/o" and seen["device"] == "cpu" and seen["run_dir"] == "/r"
    seen.clear()
    assert cli.main(["eval", "run", "/r"]) == 0
    assert "out" not in seen and "device" not in seen


def test_probe_device_reaches_the_readout_context(tmp_path, fakes, monkeypatch):
    devices = []
    real = base.ReadoutContext.__init__

    def spy(self, *a, **k):
        real(self, *a, **k)
        devices.append(self.device)

    monkeypatch.setattr(base.ReadoutContext, "__init__", spy)
    bench, run = simple_clf_world(tmp_path)
    probe(run, bench=bench, cfg=clf_cfg(3), device="cpu")
    assert devices == ["cpu"]


# ============================================================================ 2.7 --allow-loopback


class FarNetworkBridge(NetworkBridge):
    """Connects to a non-loopback address inside encode (the hook refuses before any packet leaves)."""

    name = "np-far-network"

    def encode(self, samples):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.01)
            s.connect(("10.255.255.1", 9))
        except OSError:
            pass
        finally:
            s.close()
        return NumpyBridge.encode(self, samples)


def _extract(tmp_path, bridge, **kw):
    return run_extract(bench=FakeBenchmark(), bridge=bridge, split="train", shard=(0, 1), out=tmp_path / "run",
                       decode_workers=2, canaries=0, **kw)


def test_loopback_is_refused_by_default_and_allowed_on_opt_in(tmp_path):
    with pytest.raises(ExtractError) as ei:
        _extract(tmp_path / "a", NetworkBridge)
    assert ei.value.code == E_NETWORK
    summary = _extract(tmp_path / "b", NetworkBridge, allow_loopback=True)
    sc = read_sidecar(summary["sidecar_path"])
    assert sc["side_effects"]["allow_loopback"] is True
    assert any("127.0.0.1" in a for a in sc["side_effects"]["loopback_connects"])
    # without the opt-in nothing extra is recorded
    plain = read_sidecar(_extract(tmp_path / "c", NumpyBridge)["sidecar_path"])
    assert "allow_loopback" not in plain["side_effects"]


def test_allow_loopback_still_refuses_other_addresses(tmp_path):
    with pytest.raises(ExtractError) as ei:
        _extract(tmp_path, FarNetworkBridge, allow_loopback=True)
    assert ei.value.code == E_NETWORK and "10.255.255.1" in str(ei.value)


@pytest.mark.parametrize(("addr", "ok"), [(("127.0.0.1", 80), True), (("127.3.4.5", 1), True), (("::1", 80, 0, 0), True),
                                          (("localhost", 8000), True), (("10.0.0.1", 80), False), (("example.org", 80), False),
                                          (None, False)])
def test_is_loopback(addr, ok):
    assert extract._is_loopback(addr) is ok


def test_allow_loopback_flags_parse():
    assert extract._build_parser().parse_args(["--bench", "b", "--bridge", "p", "--split", "s", "--shard", "0/1", "--out", "o",
                                                "--allow-loopback"]).allow_loopback is True
    p = cli.build_parser()
    assert p.parse_args(["extract", "--bench", "b", "--bridge", "p", "--split", "s", "--shard", "0/1", "--out", "o",
                         "--allow-loopback"]).allow_loopback is True
    assert p.parse_args(["bridge", "check", "x.py", "--bench", "b", "--allow-loopback"]).allow_loopback is True
