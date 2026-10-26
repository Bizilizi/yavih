"""``yavih.config``: env vars, TOML config, resolve_data precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from yavih import config

ROOTS = {
    "media_root": {"default": "${data.root}/kinetics400", "description": "{train,val}/<class>/*.mp4"},
    "derived_root": {"default": "${data.root}/kinetics400_derived"},
}


def _cfg_file(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text)
    return p


def test_yavih_home_default_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("YAVIH_HOME", raising=False)
    assert config.yavih_home() == Path.home() / ".yavih"
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "h"))
    assert config.yavih_home() == tmp_path / "h"
    assert config.yavih_home({"YAVIH_HOME": "~/x"}) == Path("~/x").expanduser()


def test_config_path_default_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("YAVIH_CONFIG", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert config.config_path() == Path.home() / ".config" / "yavih" / "config.toml"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config.config_path() == tmp_path / "xdg" / "yavih" / "config.toml"
    monkeypatch.setenv("YAVIH_CONFIG", str(tmp_path / "c.toml"))
    assert config.config_path() == tmp_path / "c.toml"


def test_load_config_missing_is_empty_and_malformed_raises(tmp_path):
    assert config.load_config(tmp_path / "nope.toml") == {}
    bad = _cfg_file(tmp_path, "[data\nroot = 1")
    with pytest.raises(config.ConfigError):
        config.load_config(bad)
    good = _cfg_file(tmp_path, '[data]\nroot = "/d"\n[data.in1k]\nmedia_root = "/mirror/in1k"\n[runs]\nroot = "/r"\n')
    assert config.load_config(good) == {"data": {"root": "/d", "in1k": {"media_root": "/mirror/in1k"}}, "runs": {"root": "/r"}}


def test_runs_root_precedence(monkeypatch, tmp_path):
    cfg = _cfg_file(tmp_path, '[runs]\nroot = "/from/config"\n')
    monkeypatch.setenv("YAVIH_CONFIG", str(cfg))
    monkeypatch.delenv("YAVIH_RUNS", raising=False)
    assert config.runs_root() == Path("/from/config")
    monkeypatch.setenv("YAVIH_RUNS", "/from/env")
    assert config.runs_root() == Path("/from/env")
    monkeypatch.delenv("YAVIH_RUNS")
    monkeypatch.setenv("YAVIH_CONFIG", str(tmp_path / "absent.toml"))
    assert config.runs_root() == Path.cwd() / "runs"  # no project: runs under the current directory, absolute


def test_scratch_root_never_uses_tmpdir(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", "/tmp/should-not-be-used")
    monkeypatch.delenv("YAVIH_SCRATCH", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    assert config.scratch_root(run_dir=tmp_path / "run") == tmp_path / "run"
    with pytest.raises(config.ConfigError):
        config.scratch_root()
    monkeypatch.setenv("SLURM_TMPDIR", "/slurm/tmp")
    assert config.scratch_root(run_dir=tmp_path) == Path("/slurm/tmp")
    monkeypatch.setenv("YAVIH_SCRATCH", "/explicit")
    assert config.scratch_root(run_dir=tmp_path) == Path("/explicit")
    # order: YAVIH_SCRATCH → [scratch] root → SLURM_TMPDIR → in place (explicit config beats the site's variable)
    monkeypatch.delenv("YAVIH_SCRATCH")
    cfg = tmp_path / "yavih.toml"
    cfg.write_text('[scratch]\nroot = "/configured"\n', encoding="utf-8")
    monkeypatch.setenv("YAVIH_CONFIG", str(cfg))
    assert config.scratch_root(run_dir=tmp_path) == Path("/configured") == config.configured_scratch_root()
    monkeypatch.setenv("YAVIH_SCRATCH", "/explicit")
    assert config.configured_scratch_root() == Path("/explicit")
    monkeypatch.delenv("YAVIH_SCRATCH")
    monkeypatch.delenv("SLURM_TMPDIR")
    cfg.write_text("", encoding="utf-8")
    assert config.configured_scratch_root() is None  # the probe reads in place (W_SIZE)


def test_data_env_var_name():
    assert config.data_env_var("k400", "media_root") == "YAVIH_DATA__K400__MEDIA_ROOT"
    assert config.data_env_var("nyuv2_depth", "derived_root") == "YAVIH_DATA__NYUV2_DEPTH__DERIVED_ROOT"


def test_resolve_data_default_with_site_root(tmp_path):
    cfg = {"data": {"root": "/site"}}
    out = config.resolve_data("k400", ROOTS, config=cfg, env={})
    assert out == {"media_root": "/site/kinetics400", "derived_root": "/site/kinetics400_derived"}


def test_resolve_data_requires_site_root_when_placeholder_used():
    with pytest.raises(config.ConfigError, match="data root 'media_root'"):
        config.resolve_data("k400", ROOTS, config={}, env={})
    # a literal default needs no site root
    out = config.resolve_data("x", {"r": {"default": "/abs/path"}}, config={}, env={})
    assert out == {"r": "/abs/path"}
    # env YAVIH_DATA__ROOT can supply the site root
    out = config.resolve_data("k400", ROOTS, config={}, env={"YAVIH_DATA__ROOT": "/envsite"})
    assert out["media_root"] == "/envsite/kinetics400"


def test_resolve_data_precedence_kwargs_env_config_default():
    # "kx": not a packaged benchmark (the packaged k400 declares media_root only, and the strict schema
    # refuses a configured root a known benchmark does not declare)
    cfg = {"data": {"root": "/site", "kx": {"media_root": "/cfg/k400", "derived_root": "/cfg/k400_derived"}}}
    env = {"YAVIH_DATA__KX__MEDIA_ROOT": "/env/k400"}
    # config beats default
    out = config.resolve_data("kx", ROOTS, config=cfg, env={})
    assert out == {"media_root": "/cfg/k400", "derived_root": "/cfg/k400_derived"}
    # env beats config
    out = config.resolve_data("kx", ROOTS, config=cfg, env=env)
    assert out == {"media_root": "/env/k400", "derived_root": "/cfg/k400_derived"}
    # kwargs beat env
    out = config.resolve_data("kx", ROOTS, {"media_root": "/kw/k400"}, config=cfg, env=env)
    assert out == {"media_root": "/kw/k400", "derived_root": "/cfg/k400_derived"}
    # None override falls through to the next level
    out = config.resolve_data("kx", ROOTS, {"media_root": None}, config=cfg, env=env)
    assert out["media_root"] == "/env/k400"


def test_known_benchmark_refuses_undeclared_configured_root():
    cfg = {"data": {"root": "/site", "k400": {"derived_root": "/cfg/x"}}}
    with pytest.raises(config.ConfigError, match="no data root 'derived_root'"):
        config.resolve_data("k400", ROOTS, config=cfg, env={})


def test_resolve_data_rejects_unknown_override_and_expands_user():
    with pytest.raises(config.ConfigError, match="unknown data root override"):
        config.resolve_data("k400", ROOTS, {"nope": "/x"}, config={"data": {"root": "/s"}}, env={})
    out = config.resolve_data("k400", ROOTS, {"media_root": "~/m", "derived_root": "/d"}, config={}, env={})
    assert out["media_root"] == str(Path("~/m").expanduser())


def test_resolve_data_reads_real_env_and_config_by_default(monkeypatch, tmp_path):
    cfg = _cfg_file(tmp_path, '[data]\nroot = "/site"\n[data.kx]\nderived_root = "/cfg/derived"\n')
    monkeypatch.setenv("YAVIH_CONFIG", str(cfg))
    monkeypatch.setenv("YAVIH_DATA__KX__MEDIA_ROOT", "/env/media")
    out = config.resolve_data("kx", ROOTS)
    assert out == {"media_root": "/env/media", "derived_root": "/cfg/derived"}


def test_resolve_data_accepts_bare_string_specs():
    out = config.resolve_data("b", {"r": "${data.root}/b"}, config={"data": {"root": "/s"}}, env={})
    assert out == {"r": "/s/b"}


def test_synthetic_fixture_root_resolves(synthetic_benchmark):
    sb = synthetic_benchmark
    out = config.resolve_data(sb.name, sb.spec["data"]["roots"])
    assert Path(out["media_root"]) == sb.media_root
