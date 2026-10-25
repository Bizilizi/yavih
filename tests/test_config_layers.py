"""The layered configuration of ``yavih.config``.

User file → its site table → project file (``--config`` / ``YAVIH_CONFIG`` / ``./yavih.toml`` upward) → its
site table → env ``YAVIH_*`` → CLI flags; strict schema with ``--lenient``; ``~``/``$VAR`` expansion;
relative paths anchored to their file; ``[datasets]`` / ``${datasets.X}``; ``config show`` provenance;
doctor never aborts on configuration; per-roots status caches; stale ``$YAVIH_HOME`` copies; one resources
helper. Every test uses tmp HOME / XDG directories (the conftest also isolates them).
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

import yavih.cli as cli
from yavih import config, resources
from yavih.benchmark import Benchmark, bench_fetch, data_status_path


@pytest.fixture
def xdg(tmp_path, monkeypatch) -> Path:
    """An isolated user config directory, no project file, the test's own cwd."""
    d = tmp_path / "xdg"
    (d / "yavih").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(d))
    monkeypatch.delenv("YAVIH_CONFIG", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    yield d
    config.set_cli()


def _user(xdg: Path, text: str) -> Path:
    p = xdg / "yavih" / "config.toml"
    p.write_text(text)
    return p


def _json(capsys, argv) -> tuple[int, dict]:
    rc = cli.main([*argv, "--json"])
    return rc, json.loads(capsys.readouterr().out)


# ----------------------------------------------------------------------------- layers


def test_project_file_discovered_upward_and_layered_over_user(xdg, tmp_path, monkeypatch):
    _user(xdg, '[data]\nroot = "/user/data"\n[runs]\nroot = "/user/runs"\n')
    proj = tmp_path / "work"
    (proj / "yavih.toml").write_text('[runs]\nroot = "runs"\n')
    sub = proj / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    s = config.load()
    assert [f.layer for f in s.files] == ["user", "project"] and s.files[1].how == "discovered"
    # the project file wins for runs; its relative path is anchored to the file, not the cwd
    assert s.runs_root() == proj / "runs"
    assert s.get("runs", "root").source == f"file:{proj / 'yavih.toml'}:2"
    # the user file still supplies what the project does not set
    assert s.data_root() == "/user/data"


def test_yavih_config_and_cli_config_name_the_project_file(xdg, tmp_path, monkeypatch):
    _user(xdg, '[runs]\nroot = "/user/runs"\n')
    a = tmp_path / "a.toml"
    a.write_text('[runs]\nroot = "/a/runs"\n')
    b = tmp_path / "b.toml"
    b.write_text('[runs]\nroot = "/b/runs"\n')
    (tmp_path / "work" / "yavih.toml").write_text('[runs]\nroot = "/discovered"\n')
    assert config.runs_root() == Path("/discovered")
    monkeypatch.setenv("YAVIH_CONFIG", str(a))
    assert config.runs_root() == Path("/a/runs")
    config.set_cli(config_file=b)
    assert config.runs_root() == Path("/b/runs") and config.load().files[1].how == "--config"
    config.set_cli(config_file=tmp_path / "missing.toml")
    with pytest.raises(config.ConfigError, match="no such file"):
        config.load()


def test_precedence_env_beats_files_and_cli_beats_env(xdg, monkeypatch):
    _user(xdg, '[runs]\nroot = "/file"\n[results]\nroot = "/file-results"\n[data.esc50]\nmedia_root = "/file/esc50"\n')
    roots = {"media_root": {"default": "${data.root}/esc50"}}
    monkeypatch.setenv("YAVIH_RUNS", "/env")
    monkeypatch.setenv("YAVIH_DATA__ESC50__MEDIA_ROOT", "/env/esc50")
    assert config.runs_root() == Path("/env") and config.resolve_data("esc50", roots) == {"media_root": "/env/esc50"}
    config.set_cli(runs_root="/cli", results="/cli-results", data=["esc50.media_root=/cli/esc50"])
    assert config.runs_root() == Path("/cli") and config.results_root() == Path("/cli-results")
    src = config.resolve_data_sources("esc50", roots)["media_root"]
    assert src["value"] == "/cli/esc50" and src["source"] == "cli"
    assert src["env_var"] == "YAVIH_DATA__ESC50__MEDIA_ROOT" and src["config_key"] == "[data.esc50] media_root"
    # programmatic overrides (Benchmark.load(data=...)) sit above everything
    assert config.resolve_data("esc50", roots, {"media_root": "/arg"}) == {"media_root": "/arg"}


def test_site_tables_by_env_and_hostname_glob(xdg, monkeypatch):
    host = socket.gethostname()
    _user(xdg, f'''[data]
root = "/generic"
[site.alpha]
hosts = ["{host[:2]}*"]
[site.alpha.data]
root = "/alpha"
[site.beta]
hosts = ["no-such-host-*"]
[site.beta.runs]
root = "/beta-runs"
''')
    s = config.load()
    assert s.site == "alpha" and "matches" in s.site_source and s.data_root() == "/alpha"
    assert "[site.alpha]" in s.get("data", "root").source
    monkeypatch.setenv("YAVIH_SITE", "beta")
    s = config.load()
    assert s.site == "beta" and s.data_root() == "/generic" and s.runs_root() == Path("/beta-runs")
    monkeypatch.setenv("YAVIH_SITE", "gamma")
    with pytest.raises(config.ConfigError, match="no config file has a \\[site.gamma\\]"):
        config.load()


def test_project_site_beats_project_generic_and_user(xdg, tmp_path, monkeypatch):
    _user(xdg, '[site.c.runs]\nroot = "/user-site"\n')
    (tmp_path / "work" / "yavih.toml").write_text('[runs]\nroot = "/proj"\n[site.c.runs]\nroot = "/proj-site"\n')
    monkeypatch.setenv("YAVIH_SITE", "c")
    assert config.runs_root() == Path("/proj-site")


def test_expansion_tilde_and_vars(xdg, monkeypatch, tmp_path):
    monkeypatch.setenv("SCRATCHY", str(tmp_path / "s"))
    _user(xdg, '[runs]\nroot = "$SCRATCHY/runs"\n[results]\nroot = "${SCRATCHY}/res"\n[home]\nroot = "~/yh"\n')
    monkeypatch.delenv("YAVIH_HOME", raising=False)
    s = config.load()
    assert s.runs_root() == tmp_path / "s" / "runs" and s.results_root() == tmp_path / "s" / "res"
    assert s.yavih_home() == Path("~/yh").expanduser()
    _user(xdg, '[runs]\nroot = "$NOT_SET_ANYWHERE_42/runs"\n')
    with pytest.raises(config.ConfigError, match="NOT_SET_ANYWHERE_42 is not set"):
        config.load()


def test_datasets_table_shared_vggsound_tree(xdg):
    _user(xdg, '[data]\nroot = "/site"\n')
    for name in ("vggsounder", "vggss_box", "avsync"):
        assert Benchmark.load(name).resolved_roots()["media_root"] == "/site/vggsound"  # ${datasets.vggsound} default
    _user(xdg, '[data]\nroot = "/site"\n[datasets]\nvggsound = "/mirror/vggsound"\n')
    for name in ("vggsounder", "vggss_box", "avsync"):
        assert Benchmark.load(name).resolved_roots()["media_root"] == "/mirror/vggsound"
    # a per-benchmark root still beats the shared dataset; a config value may reference ${datasets.X} too
    _user(xdg, '[datasets]\nvggsound = "/mirror/vggsound"\n[data.vggss_box]\nmedia_root = "${datasets.vggsound}/../vgg-ss"\n')
    assert Benchmark.load("vggss_box").resolved_roots()["media_root"] == "/mirror/vggsound/../vgg-ss"
    assert config.dataset("vggsound") == "/mirror/vggsound"


def test_datasets_do_not_change_protocol_hashes(xdg):
    _user(xdg, '[datasets]\nvggsound = "/elsewhere"\n')
    for name in ("vggsounder", "vggss_box", "avsync"):
        b = Benchmark.load(name)
        rec = b.spec.changelog[-1]  # the hashes the folder records
        assert (b.protocol_hash, b.input_hash) == (rec.protocol_hash, rec.input_hash)


# ----------------------------------------------------------------------------- strict schema


@pytest.mark.parametrize(
    "text, match",
    [
        ('[run]\nroot = "/x"\n', r"unknown table \[run\], did you mean \[runs\]"),
        ('[data.esc50]\nmediaroot = "/x"\n', r"no data root 'mediaroot', did you mean 'media_root'"),
        ('[data.ESC50]\nmedia_root = "/x"\n', r"unknown benchmark 'ESC50', did you mean 'esc50'"),
        ('[results]\nrot = "/x"\n', r"unknown key \[results\] rot"),
        ('[resources]\ncpu = 4\n', r"unknown key \[resources\] cpu, did you mean 'cpus'"),
        ('[runs]\nroot = 3\n', r"must be a string path"),
    ],
)
def test_strict_schema_refuses_typos(xdg, text, match):
    _user(xdg, text)
    with pytest.raises(config.ConfigError, match=match):
        config.load()


def test_unknown_env_vars_and_near_miss_bench_env(xdg, monkeypatch):
    monkeypatch.setenv("YAVIH_DATA__ESC_50__MEDIA_ROOT", "/x")
    with pytest.raises(config.ConfigError, match="unknown benchmark 'esc_50', did you mean 'esc50'"):
        config.load()
    monkeypatch.delenv("YAVIH_DATA__ESC_50__MEDIA_ROOT")
    monkeypatch.setenv("YAVIH_RUN", "/x")
    with pytest.raises(config.ConfigError, match="unknown environment variable YAVIH_RUN, did you mean YAVIH_RUNS"):
        config.load()
    monkeypatch.setenv("YAVIH_RAM_BUDGET_GB", "lots")
    with pytest.raises(config.ConfigError, match="ram_budget_gb"):
        config.load()


def test_lenient_downgrades_schema_errors_to_warnings(xdg, monkeypatch):
    _user(xdg, '[run]\nroot = "/x"\n[runs]\nroot = "/ok"\n')
    monkeypatch.setenv("YAVIH_DECODE_POOL", "process")  # an unknown variable
    with pytest.raises(config.ConfigError) as ei:
        config.load()
    assert "[run]" in str(ei.value) and "YAVIH_DECODE_POOL" in str(ei.value) and "--lenient" in str(ei.value)
    monkeypatch.setenv("YAVIH_LENIENT", "1")
    s = config.load()
    assert s.lenient and s.runs_root() == Path("/ok")
    assert {c for c, _m in s.warnings} >= {"W_CONFIG_UNKNOWN", "W_CONFIG_ENV"}


def test_unknown_benchmark_name_is_a_warning_and_checked_when_loaded(xdg, synthetic_benchmark, monkeypatch):
    sb = synthetic_benchmark
    monkeypatch.setattr(config, "_REGISTERED", {})  # earlier tests in this process may have loaded a synth_clf folder
    _user(xdg, f'[data.{sb.name}]\nmedia_rot = "/x"\n[data.my_private_bench]\nmedia_root = "/p"\n')
    s = config.load()
    assert any(c == "W_CONFIG_UNKNOWN_BENCH" and "my_private_bench" in m for c, m in s.warnings)
    with pytest.raises(config.ConfigError, match="no data root 'media_rot'"):  # validated once the folder is known
        Benchmark.from_dir(sb.folder).resolved_roots()


def test_typical_user_config_shape_is_valid(xdg):
    """A typical user config (shared data root, runs root, per-benchmark roots) loads under the strict schema."""
    _user(xdg, '''[data]
root = "/data/datasets"
[runs]
root = "/scratch/yavih-runs"
[data.in1k]
media_root = "/mirrors/imagenet2012"
[data.nyuv2_depth]
derived_root = "/scratch/yavih-runs/_data/nyuv2_depth"

[data.audioset20k]
media_root = "/data/audioset20k-wds"
''')
    s = config.load()
    assert not s.warnings and s.get("data", "in1k", "media_root").source.endswith("config.toml:6")


def test_alias_tables_map_to_the_canonical_benchmark(xdg):
    _user(xdg, '[data.imagenet]\nmedia_root = "/mirror/in1k"\n')
    assert Benchmark.load("in1k").resolved_roots()["media_root"] == "/mirror/in1k"


# ----------------------------------------------------------------------------- CLI: config show / get / doctor


def test_config_show_json_lists_sources(xdg, capsys, monkeypatch, tmp_path):
    user = _user(xdg, '[data]\nroot = "/site"\n')
    monkeypatch.setenv("YAVIH_RUNS", str(tmp_path / "r"))
    rc, env = _json(capsys, ["--results", str(tmp_path / "res"), "config", "show", "--bench", "esc50"])
    assert rc == 0, env
    d = env["data"]
    assert d["files"][0] == {"path": str(user), "layer": "user", "how": "XDG_CONFIG_HOME", "exists": True}
    assert d["values"]["data.root"] == {"value": "/site", "source": f"file:{user}:2"}
    assert d["values"]["runs.root"]["source"] == "env:YAVIH_RUNS"
    assert d["values"]["results.root"] == {"value": str(tmp_path / "res"), "source": "cli"}
    assert {"resources.ram_budget_gb", "resources.decode_workers", "resources.cpus"} <= set(d["values"])
    root = d["bench"]["roots"]["media_root"]
    assert root["value"] == "/site/esc50" and root["source"].startswith("yaml default") and root["exists"] is False
    assert root["env_var"] == "YAVIH_DATA__ESC50__MEDIA_ROOT"
    # the CLI layer does not outlive the command
    assert config.cli_overrides() == {} and "YAVIH_RESULTS" not in os.environ


def test_config_get_for_shell_scripts(xdg, capsys):
    _user(xdg, '[data]\nroot = "/site"\n[datasets]\nvggsound = "/data/vggsound"\n')
    assert cli.main(["config", "get", "avsync.curation_root"]) == 0
    assert capsys.readouterr().out == "/site/avsync\n"
    assert cli.main(["config", "get", "datasets.vggsound"]) == 0
    assert capsys.readouterr().out == "/data/vggsound\n"
    assert cli.main(["--data", "esc50.media_root=/cli", "config", "get", "esc50.media_root"]) == 0
    assert capsys.readouterr().out == "/cli\n"
    assert cli.main(["config", "get", "esc50.nope"]) == cli.EXIT_USAGE


def test_doctor_with_empty_home_reports_instead_of_aborting(xdg, capsys, monkeypatch, tmp_path):
    """With no config at all, ``doctor --bench esc50`` reports instead of aborting."""
    monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))
    rc, env = _json(capsys, ["doctor", "--bench", "esc50"])
    assert rc == 0, env
    root = env["data"]["data_roots"]["media_root"]
    assert root["path"] is None and "no data root is configured" in root["error"]
    w = [x for x in env["warnings"] if x["code"] == "W_DATA_ROOT"]
    assert w and "YAVIH_DATA__ESC50__MEDIA_ROOT" in w[0]["message"] and "[data.esc50] media_root" in w[0]["message"]
    assert env["data"]["config"]["files"][0]["exists"] is False


def test_doctor_missing_root_names_key_and_env_var(xdg, capsys):
    _user(xdg, '[data]\nroot = "/nonexistent-site"\n')
    rc, env = _json(capsys, ["doctor", "--bench", "nyuv2_depth"])
    assert rc == 0
    msgs = {w["where"]: w["message"] for w in env["warnings"] if w["code"] == "W_DATA_ROOT"}
    assert "YAVIH_DATA__NYUV2_DEPTH__DERIVED_ROOT" in msgs["derived_root"] and "/nonexistent-site/nyuv2" in msgs["derived_root"]
    # geonet_root is read by prepare/ only: a note, not a warning
    assert "geonet_root" not in msgs and any(n["where"] == "geonet_root" for n in env["data"]["notes"])


def test_doctor_reports_an_invalid_config_as_a_finding(xdg, capsys):
    _user(xdg, '[run]\nroot = "/x"\n')
    rc, env = _json(capsys, ["doctor"])
    assert rc == 0 and any(w["code"] == "W_CONFIG" and "[run]" in w["message"] for w in env["warnings"])
    assert cli.main(["config", "show"]) == cli.EXIT_USAGE
    capsys.readouterr()
    assert cli.main(["--lenient", "config", "show"]) == 0
    capsys.readouterr()


# ----------------------------------------------------------------------------- caches and home copies


def test_data_status_is_keyed_by_the_resolved_roots(xdg, synthetic_benchmark, monkeypatch, tmp_path):
    sb = synthetic_benchmark
    b = Benchmark.from_dir(sb.folder)
    st = b.data_status()
    other = tmp_path / "other_site_media"
    other.mkdir()
    b2 = Benchmark.from_dir(sb.folder, data={"media_root": str(other)})
    st2 = b2.data_status()
    assert st.path != st2.path and Path(st.path).is_file() and Path(st2.path).is_file()
    assert Path(st.path) == data_status_path(sb.name, b.resolved_roots())
    assert config.roots_digest({"a": "1"}) != config.roots_digest({"a": "2"})


def test_read_only_home_warns_once(xdg, synthetic_benchmark, monkeypatch, tmp_path, capsys):
    home = tmp_path / "ro_home"
    home.mkdir()
    monkeypatch.setenv("YAVIH_HOME", str(home))
    import yavih.benchmark as B

    def deny(path, payload):
        raise PermissionError(13, "Read-only file system", str(path))

    monkeypatch.setattr(B, "_write_json_atomic", deny)
    config.drain_warnings()
    for _ in range(2):
        Benchmark.from_dir(synthetic_benchmark.folder).data_status()
    warned = [w for w in config.drain_warnings() if w["code"] == "W_HOME_READONLY"]
    assert len(warned) == 1 and "YAVIH_HOME" in warned[0]["message"]
    assert capsys.readouterr().err.count("W_HOME_READONLY") <= 1


def test_stale_home_copy_is_refused_and_refetch_repairs_it(xdg, yavih_home, monkeypatch):
    out = bench_fetch("esc50")
    dest = Path(out["folder"])
    stamp = json.loads((dest / ".yavih_fetch.json").read_text())
    assert stamp["protocol_hash"] == Benchmark.load("esc50").protocol_hash and "benchmark.yaml" in stamp["files"]
    assert Benchmark.load("esc50").folder == dest.resolve()
    # simulate a yavih upgrade: the packaged folder no longer matches what the copy was synced from
    stamp["files"]["reader.py"] = "0" * 64
    (dest / ".yavih_fetch.json").write_text(json.dumps(stamp))
    import yavih.benchmark as B

    B._HOME_CHECKED.clear()
    with pytest.raises(config.ConfigError, match="E_HOME_BENCH_STALE|has changed since"):
        Benchmark.load("esc50")
    monkeypatch.setenv("YAVIH_LENIENT", "1")
    config.drain_warnings()
    Benchmark.load("esc50")
    assert any(w["code"] == "W_HOME_BENCH_STALE" for w in config.drain_warnings())
    monkeypatch.delenv("YAVIH_LENIENT")
    B._HOME_CHECKED.clear()
    bench_fetch("esc50")  # an unmodified cached file is refreshed; the stamp is rewritten
    B._HOME_CHECKED.clear()
    assert Benchmark.load("esc50").folder == dest.resolve()


def test_unstamped_home_copy_warns(xdg, yavih_home):
    bench_fetch("esc50")
    (yavih_home / "benchmarks" / "esc50" / ".yavih_fetch.json").unlink()
    import yavih.benchmark as B

    B._HOME_CHECKED.clear()
    config.drain_warnings()
    Benchmark.load("esc50")
    assert any(w["code"] == "W_HOME_BENCH_UNSTAMPED" for w in config.drain_warnings())


def test_folder_root_for_scripts(xdg, synthetic_benchmark):
    _user(xdg, '[data]\nroot = "/site"\n')
    folder = Benchmark.load("esc50").folder
    assert config.folder_root(folder, "media_root") == Path("/site/esc50")
    assert config.folder_root(folder, "media_root", "/flag") == Path("/flag")
    with pytest.raises(config.ConfigError, match="declares no data root"):
        config.folder_root(folder, "nope")


def test_avsync_curation_root_is_declared():
    b = Benchmark.load("avsync")
    roots = b.spec.raw["data"]["roots"]
    assert roots["curation_root"]["prepare_only"] is True and b.spec.data.roots["curation_root"].prepare_only


# ----------------------------------------------------------------------------- resources


def test_resources_rules_from_env():
    gib = 1024**3
    assert resources.ram_budget_bytes({}) == 16 * gib
    assert resources.ram_budget_bytes({"SLURM_MEM_PER_NODE": "16000"}) == int(0.45 * 16000 * 1024**2)
    assert resources.ram_budget_bytes({"SLURM_MEM_PER_CPU": "1000", "SLURM_CPUS_PER_TASK": "4"}) == int(0.45 * 4000 * 1024**2)
    assert resources.ram_budget_bytes({"YAVIH_RAM_BUDGET_GB": "0"}) == 0
    with pytest.raises(ValueError):
        resources.ram_budget_bytes({"YAVIH_RAM_BUDGET_GB": "lots"})
    assert resources.decode_workers({"SLURM_CPUS_PER_TASK": "8"}) == 7
    assert resources.decode_workers({"NCPUS": "4"}) == 3  # PBS
    assert resources.decode_workers({"YAVIH_CPUS": "3"}) == 2  # an `exec --slots` slot
    assert resources.decode_workers({"YAVIH_DECODE_WORKERS": "5", "SLURM_CPUS_PER_TASK": "64"}) == 5
    assert resources.cpu_limit({"SLURM_CPUS_PER_TASK": "6"}) == (6, "env:SLURM_CPUS_PER_TASK", True)


def test_resources_from_config_table(xdg, monkeypatch):
    for k in ("SLURM_CPUS_PER_TASK", "NCPUS", "PBS_NUM_PPN", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        monkeypatch.delenv(k, raising=False)
    _user(xdg, '[resources]\nram_budget_gb = 2.5\ncpus = 9\n')
    assert resources.ram_budget_bytes() == int(2.5 * 1024**3)
    assert resources.cpu_limit()[:2] == (9, f"file:{xdg / 'yavih' / 'config.toml'}:3")
    assert resources.decode_workers() == 8
    from yavih.io.reader import ram_budget_bytes as reader_budget
    from yavih.plan.units import _ram_budget_bytes as plan_budget

    assert reader_budget() == plan_budget() == int(2.5 * 1024**3)
    monkeypatch.setenv("YAVIH_RAM_BUDGET_GB", "1")
    assert resources.ram_budget_bytes() == 1024**3


def test_cgroup_parsers(tmp_path):
    (tmp_path / "cpu.max").write_text("250000 100000\n")
    (tmp_path / "memory.max").write_text(str(8 * 1024**3) + "\n")
    assert resources.cgroup_cpu_quota(tmp_path) == 3 and resources._cgroup_memory(tmp_path) == 8 * 1024**3
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (tmp_path / "memory.max").write_text("max\n")
    assert resources.cgroup_cpu_quota(tmp_path) is None and resources._cgroup_memory(tmp_path) is None
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "memory").mkdir()
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("400000")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000")
    (v1 / "memory" / "memory.limit_in_bytes").write_text(str(2**63 - 4096))  # v1 "unlimited"
    assert resources.cgroup_cpu_quota(v1) == 4 and resources._cgroup_memory(v1) is None


def test_relative_env_roots_are_anchored_to_the_cwd_at_load(xdg, capsys, monkeypatch, tmp_path):
    """A relative ``YAVIH_RUNS`` / ``YAVIH_RESULTS`` / ``YAVIH_SCRATCH`` (a launcher is a command, never a path) names
    the directory under the cwd the process started in: ``runs_root()`` is absolute, ``config get runs.root`` prints
    that directory and a matrix ``out: null`` lands there too, not under the matrix file's directory."""
    from yavih.plan.matrix import load_matrix

    work = Path.cwd()
    monkeypatch.setenv("YAVIH_RUNS", "runs")
    monkeypatch.setenv("YAVIH_RESULTS", "res")
    monkeypatch.setenv("YAVIH_SCRATCH", "scr")
    monkeypatch.setenv("YAVIH_LAUNCHER", "python")
    s = config.load(refresh=True)
    assert s.runs_root() == work / "runs" and s.runs_root().is_absolute()
    assert s.results_root() == work / "res" and s.configured_scratch_root() == work / "scr"
    assert s.launcher() == "python" and s.runs_root_source() == "env:YAVIH_RUNS"
    rc, env = _json(capsys, ["config", "get", "runs.root"])
    assert rc == 0 and env["data"]["value"] == str(work / "runs")
    sub = work / "matrices"
    sub.mkdir()
    m = sub / "m.yaml"
    m.write_text("schema_version: 1\nbridges:\n  a: {script: s.py, benchmark: k400, profile: p.json}\n", encoding="utf-8")
    assert load_matrix(str(m)).out == str(work / "runs")
    assert config.runs_root(env={"YAVIH_RUNS": "elsewhere"}, config={}) == work / "elsewhere"
