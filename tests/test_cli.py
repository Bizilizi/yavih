"""The CLI (envelope shape, exit codes, every handler; docs/cli.md) and ``yavih.testing``
(synthetic benchmark folders per readout, tiny media, random/oracle bridges, ``fast()``).

Handlers are exercised against small fakes so the tests stay independent of the benchmark, extract and probe
modules; the
synthetic benchmark folders are checked against the folder contract (docs/benchmark-folder.md) directly (yaml keys, sha256, registry
names, curation sums, decodable media, derived targets).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from yavih import cli
from yavih import testing as T
from yavih.bridge import DESCRIBE_REQUIRED, Bridge
from yavih.hashing import canon
from yavih.metrics import registry as R

ENVELOPE_KEYS = {"status", "data", "errors", "warnings", "paths"}


def run_json(capsys, argv):
    rc = cli.main(["--json", *argv])
    out = capsys.readouterr()
    lines = out.out.strip().splitlines()
    assert len(lines) == 1, f"expected exactly one JSON object on stdout, got {out.out!r}"
    env = json.loads(lines[0])
    assert set(env) == ENVELOPE_KEYS
    assert env["status"] in cli.STATUSES
    for f in env["errors"] + env["warnings"]:
        assert set(f) == {"code", "message", "where"}
    return rc, env


# ============================================================================= envelope / exit codes


def test_usage_error_under_json_is_one_envelope(capsys):
    rc, env = run_json(capsys, ["extract"])
    assert rc == cli.EXIT_USAGE and env["status"] == "error" and env["errors"][0]["code"] == "E_USAGE"
    rc, env = run_json(capsys, ["nope"])
    assert rc == cli.EXIT_USAGE


def test_help_and_version_exit_zero(capsys):
    assert cli.main(["--help"]) == 0
    assert cli.main(["--version"]) == 0
    assert cli.main(["bench", "--help"]) == 0


def test_cli_error_and_not_applicable_mapping(capsys, monkeypatch):
    from yavih.bridge import NotApplicable

    def boom(ns):
        raise cli.CliError("E_X", "bad", cli.EXIT_CONTRACT)

    def na(ns):
        raise NotApplicable("audio-only bridge: no audio stream")

    monkeypatch.setattr(cli, "cmd_plan_status", boom)
    rc, env = run_json(capsys, ["plan", "status", "p.jsonl"])
    assert rc == 2 and env["errors"][0] == {"code": "E_X", "message": "bad", "where": None}
    monkeypatch.setattr(cli, "cmd_plan_status", na)
    # the handler table is looked up at parse time through set_defaults, so patch the parser's binding too
    monkeypatch.setattr(cli, "build_parser", _patched_parser(cli.build_parser, {"cmd_plan_status": na}))
    rc, env = run_json(capsys, ["plan", "status", "p.jsonl"])
    assert rc == 0 and env["status"] == "not_applicable" and "audio-only" in env["data"]["reason"]


def _patched_parser(build, replacements):
    def _build():
        p = build()

        def patch(parser):
            for a in parser._actions:
                if isinstance(a, type(parser._subparsers._group_actions[0])) if parser._subparsers else False:
                    pass
            for action in parser._actions:
                choices = getattr(action, "choices", None)
                if isinstance(choices, dict):
                    for sp in choices.values():
                        func = sp._defaults.get("func")
                        if func is not None and func.__name__ in replacements:
                            sp.set_defaults(func=replacements[func.__name__])
                        patch(sp)

        patch(p)
        return p

    return _build


def test_exit_code_for_known_exceptions():
    from yavih.readouts.base import ProbeRefused

    assert cli.exit_code_for(ProbeRefused("E_COVERAGE", "low")) == (2, "E_COVERAGE")
    assert cli.exit_code_for(TimeoutError("deadline"))[0] == 4
    assert cli.exit_code_for(KeyboardInterrupt())[0] == 4
    assert cli.exit_code_for(cli.CliError("E_NOT_FOUND", "x", 3)) == (3, "E_NOT_FOUND")


def test_envelope_jsonable_data():
    env = cli.Envelope(data={"p": Path("/x"), "a": np.float32(1.5), "arr": np.arange(2), "t": (1, 2), "nan": float("nan")})
    j = env.to_json()["data"]
    assert j == {"p": "/x", "a": 1.5, "arr": [0, 1], "t": [1, 2], "nan": None}
    with pytest.raises(ValueError):
        cli.Envelope(status="weird").to_json()


def test_human_mode_prints_text_and_findings(capsys, monkeypatch):
    def handler(ns):
        return cli.Envelope(data={"x": 1}, warnings=[{"code": "W_X", "message": "careful", "where": "here"}], text="hello\n")

    monkeypatch.setattr(cli, "build_parser", _patched_parser(cli.build_parser, {"cmd_plan_status": handler}))
    rc = cli.main(["plan", "status", "p.jsonl"])
    out = capsys.readouterr()
    assert rc == 0 and out.out == "hello\n" and "W_X" in out.err and "(here)" in out.err


# ============================================================================= doctor


@pytest.fixture
def fake_decoders(monkeypatch):
    import yavih.extract
    import yavih.media

    monkeypatch.setattr(yavih.media, "decoder_info", lambda kind: {"lib": "fake", "version": "1", "libav": None} if kind != "image" else None)
    monkeypatch.setattr(yavih.extract, "default_decode_workers", lambda env=None: 3)


def test_doctor_json_shape(capsys, fake_decoders, yavih_home, runs_dir):
    rc, env = run_json(capsys, ["doctor"])
    assert rc == 0 and env["status"] == "ok"
    d = env["data"]
    for k in ["core", "media", "probe", "external", "torch", "decoders", "media_read", "media_info", "data_roots", "gpu", "launcher",
              "decode_workers_default", "yavih_home", "runs_root", "results_root"]:
        assert k in d, k
    for extra in ("core", "media", "probe"):
        assert set(d[extra]) == {"ok", "missing", "pip_hint"}
    # no extra for an original-author package; without --bench there is no external block to check
    assert "vggsounder" not in d and d["external"] == {}
    assert d["results_root"] == str(runs_dir.absolute().parent / "results")
    assert d["core"]["ok"] is True
    assert set(d["torch"]) == {"version", "cuda", "untouched"} and isinstance(d["torch"]["untouched"], bool)
    assert d["decoders"] == {"video": {"lib": "fake", "version": "1", "libav": None}, "audio": {"lib": "fake", "version": "1", "libav": None}, "image": None}
    assert d["media_read"] is None and d["media_info"] is None and d["data_roots"] == {}
    assert d["launcher"] == sys.executable and d["decode_workers_default"] == 3
    assert d["yavih_home"] == str(yavih_home) and d["runs_root"] == str(runs_dir)
    assert isinstance(d["gpu"], list)


def test_doctor_human_mode(capsys, fake_decoders):
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0 and f"yavih {cli.__version__}" in out and "decoder video" in out and "launcher" in out


class FakeBench:
    """Just enough of a ``Benchmark`` for the handlers: spec dict, roles, n_rows, data_status (faked: presence is not
    what these tests are about) and the real reader of the synthetic folder (RawStreams over its tiny media;
    a ``YAVIH_DATA__SYNTH_CLF__MEDIA_ROOT`` override set by a test is honoured at read time)."""

    def __init__(self, folder: Path, *, present=None):
        self.folder = folder
        self.spec = yaml.safe_load((folder / "benchmark.yaml").read_text())
        self.name = self.spec["name"]
        self.manifest_hash = "m" * 64
        self.protocol_hash = "p" * 64
        self.input_hash = "i" * 64
        self.scoring_hash = "s" * 64
        self.rows = T.read_manifest_rows(folder)
        self._present = present

    def _real(self):
        from yavih.benchmark import Benchmark

        return Benchmark.from_dir(self.folder)

    def roles(self):
        s = self.spec["splits"]
        return {"train": list(s["train"]), "eval": list(s["eval"]), "floor": list(s["floor"]), "folds": list(s["folds"] or [])}

    def n_rows(self, split):
        return sum(r["split"] == split for r in self.rows)

    def reader(self, split, *, shard=None, batch_size=1, rows=None, canaries=0, canary_seed=0):
        return self._real().reader(split, shard=shard, batch_size=batch_size, rows=rows, canaries=canaries, canary_seed=canary_seed)

    def view(self):
        return self._real().view()

    def media_info(self):
        return None

    def data_status(self):
        per = {}
        for s in ("train", "val"):
            n = self.n_rows(s)
            p = n if self._present is None else self._present
            per[s] = {"present": p, "nominal": n, "missing_first10": []}
        st = SimpleNamespace(per_split=per, media_fingerprint="f" * 64, roots={"media_root": "/m"}, path="/home/x/.yavih/data_status/x.json", targets_sha_ok=None, verified="2026-10-07T00:00:00Z")
        st.status_word = lambda: "verified" if self._present is None else "partial 50%"
        return st


@pytest.fixture
def synth_clf(tmp_path):
    return T.make_synthetic_benchmark(tmp_path, "clf", n_train=8, n_eval=4, media="tiny")


@pytest.fixture
def fake_benchmark_module(monkeypatch, synth_clf):
    import yavih.benchmark as B

    benches = {"synth_clf": FakeBench(synth_clf)}

    def load(ref, *, data=None, fetch=False):
        if ref in benches:
            return benches[ref]
        if Path(ref).is_dir():
            return FakeBench(Path(ref))
        from yavih.config import yavih_home

        home = yavih_home() / "benchmarks" / ref
        if (home / "benchmark.yaml").is_file():
            return FakeBench(home)
        raise B.UnknownBenchmark(ref)

    monkeypatch.setattr(B.Benchmark, "load", staticmethod(load))
    monkeypatch.setattr(B, "packaged_benchmarks", lambda: ["synth_clf"])
    return benches


def test_doctor_with_bench(capsys, fake_decoders, fake_benchmark_module, monkeypatch, tmp_path):
    """``doctor --bench``: one row's stored bytes and facts are READ (``media_read``; the reader never
    decodes), the census is reported (``media_info``), data roots resolve, external packages are listed."""
    rc, env = run_json(capsys, ["doctor", "--bench", "synth_clf"])
    assert rc == 0
    d = env["data"]
    assert d["media_read"]["ok"] is True and d["media_read"]["clip_id"] == "val_0000"
    st = d["media_read"]["streams"]["video"]
    assert st["kind"] == "video" and st["bytes"] > 0 and st["info"]["duration_s"] == 10.0 and st["info"]["container"] == "mp4"
    assert {k: d["media_info"][k] for k in ("present", "sha256", "path")} == {"present": False, "sha256": None, "path": None}
    assert "ffprobe" in d["media_info"]["header_backends"]["video"] or "av" in d["media_info"]["header_backends"]["video"]
    assert d["external"] == {}
    assert set(d["data_roots"]) == {"media_root"} and d["data_roots"]["media_root"]["exists"] is True
    assert not any(w["code"] in ("W_DATA_ROOT", "W_MEDIA_READ") for w in env["warnings"])
    # a missing root and an unreadable row are reported, not raised
    monkeypatch.setenv("YAVIH_DATA__SYNTH_CLF__MEDIA_ROOT", str(tmp_path / "nowhere"))
    rc, env = run_json(capsys, ["doctor", "--bench", "synth_clf"])
    assert rc == 0 and env["data"]["data_roots"]["media_root"]["exists"] is False
    assert env["data"]["media_read"]["ok"] is False and {"W_DATA_ROOT", "W_MEDIA_READ"} <= {w["code"] for w in env["warnings"]}


def test_doctor_unknown_bench_exit_1(capsys, fake_decoders, fake_benchmark_module):
    rc, env = run_json(capsys, ["doctor", "--bench", "nope"])
    assert rc == cli.EXIT_USAGE and env["errors"][0]["code"] == "E_USAGE"


# ============================================================================= skill


@pytest.fixture
def fake_skill(tmp_path, monkeypatch):
    root = tmp_path / "skillpkg"
    (root / "references").mkdir(parents=True)
    (root / "SKILL.md").write_text("# yavih skill\nnine steps\n")
    (root / "references" / "cli.md").write_text("cli\n")
    (root / "references" / ".gitkeep").write_text("")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"x")
    monkeypatch.setattr(cli, "_skill_root", lambda: root)
    return root


def test_skill_install_and_show(capsys, fake_skill, tmp_path):
    dest = tmp_path / "proj" / ".claude" / "skills" / "yavih"
    rc, env = run_json(capsys, ["skill", "install", "--path", str(dest)])
    stamp = env["data"].pop("stamp")  # yavih version + sha256 of the skill files (doctor: W_SKILL_STALE)
    assert rc == 0 and env["data"] == {"path": str(dest), "target": "claude", "files": ["SKILL.md", "references/cli.md", ".yavih_skill.json"]}
    assert json.loads((dest / ".yavih_skill.json").read_text()) == stamp and len(stamp["sha256"]) == 64
    assert (dest / "SKILL.md").read_text().startswith("# yavih skill") and env["paths"]["skill"] == str(dest)
    rc, env = run_json(capsys, ["skill", "show"])
    assert rc == 0 and env["data"]["text"].startswith("# yavih skill")
    rc = cli.main(["skill", "show"])
    assert rc == 0 and capsys.readouterr().out == "# yavih skill\nnine steps\n"


def test_skill_install_default_path_uses_cwd(capsys, fake_skill, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc, env = run_json(capsys, ["skill", "install"])
    assert rc == 0 and Path(".claude/skills/yavih/SKILL.md").is_file()


def test_skill_install_targets(capsys, fake_skill, tmp_path, monkeypatch):
    """One skill, four agent runtimes; AGENTS.md keeps the user's text and replaces only yavih's block."""
    (fake_skill / "SKILL.md").write_text("---\nname: yavih\ndescription: >\n  Evaluate a model\n  with yavih.\n---\n\n# yavih skill\nnine steps\n")
    monkeypatch.chdir(tmp_path)
    rc, env = run_json(capsys, ["skill", "install", "--target", "codex"])
    assert rc == 0 and Path(".codex/skills/yavih/SKILL.md").is_file() and Path(".codex/skills/yavih/references/cli.md").is_file()
    rc, env = run_json(capsys, ["skill", "install", "--target", "cursor"])
    rule = Path(".cursor/rules/yavih.mdc").read_text()
    assert rc == 0 and rule.startswith('---\ndescription: "Evaluate a model with yavih."\n') and "# yavih skill" in rule and "yavih skill show --all" in rule
    Path("AGENTS.md").write_text("# my project\nkeep me\n")
    for _ in range(2):  # a re-install replaces the block in place
        rc, env = run_json(capsys, ["skill", "install", "--target", "agents-md"])
        assert rc == 0
    text = Path("AGENTS.md").read_text()
    assert text.startswith("# my project\nkeep me\n") and text.count("<!-- yavih skill: begin -->") == 1 and "nine steps" in text
    assert "name: yavih" not in text  # the frontmatter belongs to skill runtimes only


def test_skill_show_all_appends_references(capsys, fake_skill):
    rc, env = run_json(capsys, ["skill", "show", "--all"])
    assert rc == 0 and env["data"]["references"] == ["references/cli.md"]
    assert env["data"]["text"].startswith("# yavih skill") and env["data"]["text"].rstrip().endswith("cli")


def test_bench_path_prints_the_folder(capsys, fake_benchmark_module):
    rc, env = run_json(capsys, ["bench", "path", "synth_clf"])
    assert rc == 0 and Path(env["data"]["folder"], "setup.md").is_file() and env["paths"]["folder"] == env["data"]["folder"]
    assert cli.main(["bench", "path", "synth_clf"]) == 0 and capsys.readouterr().out.strip() == env["data"]["folder"]
    rc, env = run_json(capsys, ["bench", "path", "no_such_bench"])
    assert rc == 1


def test_python_dash_m_yavih_runs_the_cli():
    proc = subprocess.run([sys.executable, "-m", "yavih", "--version"], capture_output=True, text=True)
    assert proc.returncode == 0 and "yavih" in proc.stdout


def test_skill_show_missing_is_internal_error(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_skill_root", lambda: tmp_path / "nothing")
    rc, env = run_json(capsys, ["skill", "show"])
    assert rc == cli.EXIT_INTERNAL and env["errors"][0]["code"] == "E_SKILL_MISSING"


def test_skill_install_real_package_data(capsys, tmp_path):
    """The real packaged directory copies whatever ships."""
    dest = tmp_path / "s"
    rc, env = run_json(capsys, ["skill", "install", "--path", str(dest)])
    assert rc == 0 and "README.md" in env["data"]["files"] or "SKILL.md" in env["data"]["files"]


# ============================================================================= bench


def test_bench_list(capsys, fake_benchmark_module, yavih_home):
    rc, env = run_json(capsys, ["bench", "list"])
    assert rc == 0 and len(env["data"]) == 1
    row = env["data"][0]
    assert set(row) == {"name", "aliases", "readout", "clips", "media_gb", "source_kind", "obtain", "status"}
    assert row["name"] == "synth_clf" and row["readout"] == "clf" and row["clips"] == {"train": 8, "val": 4} and row["status"] == "verified"
    assert row["source_kind"] == "derived" and row["aliases"] == ["synth_clf_alias"]
    rc = cli.main(["bench", "list"])
    assert rc == 0 and "synth_clf" in capsys.readouterr().out


def test_bench_list_includes_yavih_home_and_tolerates_broken(capsys, fake_benchmark_module, yavih_home, tmp_path):
    home_b = yavih_home / "benchmarks" / "synth_clf_home"
    T.make_synthetic_benchmark(tmp_path / "h", "clf", n_train=2, n_eval=2, name="synth_clf_home")
    home_b.parent.mkdir(parents=True)
    (tmp_path / "h" / "synth_clf_home").rename(home_b)
    broken = yavih_home / "benchmarks" / "broken"
    broken.mkdir()
    (broken / "benchmark.yaml").write_text("name: broken\n")
    rc, env = run_json(capsys, ["bench", "list"])
    assert rc == 0 and [r["name"] for r in env["data"]] == ["synth_clf", "synth_clf_home"]
    assert any(w["code"] == "W_BENCH_LOAD" and w["where"] == "broken" for w in env["warnings"])


def test_bench_info(capsys, fake_benchmark_module):
    rc, env = run_json(capsys, ["bench", "info", "synth_clf"])
    assert rc == 0
    d = env["data"]
    for k in ("readme", "roles", "floors", "metrics", "source", "budget", "external", "history"):
        assert k in d
    assert d["roles"]["train"] == {"splits": ["train"], "nominal": 8, "present": 8}
    assert d["roles"]["eval"]["nominal"] == 4 and d["floors"] == ["acc_floor"] and d["metrics"]["list"] == ["acc", "n_eval"]
    assert d["readme"].startswith("![synth_clf](assets/header.png)") and d["history"][0]["version"] == 1 and d["external"] == []
    rc, env = run_json(capsys, ["bench", "info", "synth_clf", "--readme-only"])
    assert rc == 0 and set(env["data"]) == {"name", "readme"}
    rc = cli.main(["bench", "info", "synth_clf", "--readme-only"])
    assert rc == 0 and capsys.readouterr().out.startswith("![synth_clf](assets/header.png)")
    # setup.md is the agent's first read
    rc, env = run_json(capsys, ["bench", "info", "synth_clf", "--setup"])
    assert rc == 0 and set(env["data"]) == {"name", "setup"} and env["data"]["setup"].startswith("# synth_clf — setup")
    assert env["paths"]["setup"].endswith("setup.md") and env["paths"]["folder"]
    assert cli.main(["bench", "info", "synth_clf", "--setup", "--readme-only"]) == cli.EXIT_USAGE
    capsys.readouterr()


def test_bench_fetch(capsys, fake_benchmark_module):
    rc, env = run_json(capsys, ["bench", "fetch", "synth_clf"])
    assert rc == 0 and env["data"]["manifest_sha_ok"] is True and env["data"]["name"] == "synth_clf"


def _fake_bench_verify(calls: list, *, present: int = 4, failed: int = 0, unreadable: int = 0, curation_ok: bool | None = True, targets_ok: bool | None = None):
    """A ``yavih.benchmark.bench_verify`` stand-in returning the ``bench verify`` dict; the CLI only shapes it."""

    def bench_verify(ref, *, census=False, decode_sample=32, fingerprint_sample=256, data=None):
        calls.append({"ref": ref, "census": census, "decode_sample": decode_sample, "fingerprint_sample": fingerprint_sample})
        per_split = {"train": {"present": 8, "nominal": 8, "missing_first10": []}, "val": {"present": present, "nominal": 4, "missing_first10": [] if present == 4 else ["val_0002", "val_0003"]}}
        failures = [{"clip_id": f"val_{i:04d}", "error": "DecodeError: boom"} for i in range(failed)]
        return {
            "name": ref, "manifest_sha_ok": True, "manifest_hash": "m" * 64, "protocol_hash": "p" * 64, "per_split": per_split,
            "read": {"ok": max(0, decode_sample - unreadable), "failed": unreadable},
            "decode_sample": {"ok": max(0, decode_sample - failed), "failed": failed, "failures": failures, "lib": {"video": "pyav"}},
            "labels_ok": True, "unknown_labels": [],
            "targets_sha_ok": targets_ok, "targets_errors": [] if targets_ok is not False else ["targets/val.npz missing"],
            "derived_sha_ok": None, "derived_errors": [],
            "curation_invariants_ok": curation_ok,
            "media_info": {"present": census, "rows": 12 if census else 0, "sha256": ("c" * 64) if census else None, "path": "/home/x/.yavih/media_info/synth_clf-mmmmmmmmmmmm.jsonl.gz"},
            "external": {}, "media_fingerprint": "f" * 64, "data_roots": {"media_root": "/m"},
            "data_status_path": "/home/x/.yavih/data_status/synth_clf.json", "status": "verified" if present == 4 else "partial 50%",
        }

    return bench_verify


def test_bench_verify(capsys, monkeypatch):
    """``bench verify``: the CLI delegates to ``yavih.benchmark.bench_verify`` (the only
    implementation that fingerprints media, reads and decodes a sample, writes the census and
    ``data_status.json``) and shapes its dict."""
    import yavih.benchmark as B

    calls: list = []
    monkeypatch.setattr(B, "bench_verify", _fake_bench_verify(calls))
    rc, env = run_json(capsys, ["bench", "verify", "synth_clf", "--decode-sample", "4", "--fingerprint-sample", "16"])
    assert rc == 0
    assert calls == [{"ref": "synth_clf", "census": False, "decode_sample": 4, "fingerprint_sample": 16}]  # forwarded
    d = env["data"]
    for k in ("manifest_sha_ok", "per_split", "read", "decode_sample", "labels_ok", "targets_sha_ok", "derived_sha_ok",
              "curation_invariants_ok", "media_info", "external", "media_fingerprint", "data_roots", "data_status_path"):
        assert k in d, k
    assert d["per_split"]["val"] == {"present": 4, "nominal": 4, "missing_first10": []}
    assert d["read"] == {"ok": 4, "failed": 0} and d["decode_sample"]["ok"] == 4 and d["decode_sample"]["failed"] == 0
    assert d["curation_invariants_ok"] is True and d["media_fingerprint"] == "f" * 64 and d["status"] == "verified" and env["warnings"] == []
    assert env["paths"]["data_status"] == d["data_status_path"] and d["media_info"]["present"] is False
    # --census is forwarded and the media-info table reported
    rc, env = run_json(capsys, ["bench", "verify", "synth_clf", "--census"])
    assert rc == 0 and calls[-1]["census"] is True and env["data"]["media_info"]["present"] is True
    assert env["paths"]["media_info"].endswith(".jsonl.gz")
    # partial presence → W_COVERAGE; unreadable rows → W_MEDIA_READ; decode failures → W_DECODE_SAMPLE; targets sha → W_TARGETS
    monkeypatch.setattr(B, "bench_verify", _fake_bench_verify(calls, present=2, failed=2, unreadable=1, targets_ok=False))
    rc, env = run_json(capsys, ["bench", "verify", "synth_clf", "--decode-sample", "2"])
    assert rc == 0 and {w["code"] for w in env["warnings"]} == {"W_COVERAGE", "W_MEDIA_READ", "W_DECODE_SAMPLE", "W_TARGETS"}
    assert env["data"]["decode_sample"]["failed"] == 2 and env["data"]["decode_sample"]["failures"][0]["clip_id"] == "val_0000"
    text = cli.main(["bench", "verify", "synth_clf"])
    out = capsys.readouterr().out
    assert text == 0 and "partial 50%" in out and "read sample" in out and "media_info absent" in out


def test_bench_verify_curation_mismatch_warns(capsys, monkeypatch):
    import yavih.benchmark as B

    monkeypatch.setattr(B, "bench_verify", _fake_bench_verify([], curation_ok=False))
    rc, env = run_json(capsys, ["bench", "verify", "synth_clf", "--decode-sample", "0"])
    assert rc == 0 and env["data"]["curation_invariants_ok"] is False and any(w["code"] == "W_CURATION" for w in env["warnings"])


def test_bench_verify_real_synthetic_benchmark(capsys, yavih_home, tmp_path):
    """The real ``bench_verify`` over a synthetic folder with tiny media: fingerprint computed, data_status
    written with ``verified`` set, so ``bench list`` can say ``verified``."""
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=8, n_eval=4, media="tiny")
    rc, env = run_json(capsys, ["bench", "verify", folder.name, "--census", "--decode-sample", "2", "--fingerprint-sample", "8"])
    assert rc == 0, env
    d = env["data"]
    assert d["manifest_sha_ok"] and d["labels_ok"] and d["curation_invariants_ok"] is True
    assert isinstance(d["media_fingerprint"], str) and len(d["media_fingerprint"]) == 64
    assert d["read"]["ok"] > 0 and d["read"]["failed"] == 0 and d["decode_sample"]["ok"] > 0 and d["decode_sample"]["failed"] == 0
    assert d["status"] == "verified" and d["media_info"]["present"] is True and d["media_info"]["rows"] == 12
    assert Path(d["media_info"]["path"]).is_file() and d["external"] == {} and "materialized_sha_ok" not in d
    status = json.loads(Path(d["data_status_path"]).read_text())
    assert status["media_fingerprint"] == d["media_fingerprint"] and status["verified"] and status["roots"] == d["data_roots"]
    rc, env = run_json(capsys, ["bench", "list"])
    assert rc == 0 and next(r for r in env["data"] if r["name"] == folder.name)["status"] == "verified"


# ============================================================================= bridge check


@pytest.fixture
def fake_bridge_module(monkeypatch):
    import yavih.bridge as Bm
    import yavih.extract as E
    import yavih.media

    def load_bridge_class(path):
        ns: dict = {}
        code = Path(path).read_text()
        exec(compile(code, path, "exec"), ns)  # noqa: S102 - test helper
        cands = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Bridge) and v is not Bridge and v.__module__ == "builtins" or (isinstance(v, type) and issubclass(v, Bridge) and v.__name__ in ("RandomBridge", "OracleBridge", "Flaky", "NA"))]
        return cands[-1]

    def parse_args(spec, raw):
        out = {}
        for k, v in (spec or {}).items():
            out[k] = getattr(v, "default", None) if not isinstance(v, dict) else v.get("default")
        for k, v in dict(raw).items():
            if k not in (spec or {}):
                raise cli.CliError("E_USAGE", f"unknown --arg {k}")
            typ = getattr(spec[k], "type", str)
            out[k] = typ(v)
        return out

    def validate_describe(d, bridge, bench):
        missing = [k for k in DESCRIBE_REQUIRED if k not in d]
        if missing:
            raise Bm.DescribeError(f"missing {missing}")

    monkeypatch.setattr(E, "load_bridge_class", load_bridge_class)
    monkeypatch.setattr(E, "default_decode_workers", lambda env=None: 2)
    monkeypatch.setattr(Bm, "parse_args", parse_args)
    monkeypatch.setattr(Bm, "validate_describe", validate_describe)
    monkeypatch.setattr(yavih.media, "decoder_info", lambda kind: {"lib": "fake", "version": "1", "libav": None})


def test_bridge_check_random_bridge(capsys, fake_benchmark_module, fake_bridge_module, tmp_path):
    """The profile carries ``invariance`` (rerun bit-identical, batch/order rel_l2, tolerance),
    ``rules_checked``, the first row's ``used`` and ``media``; the random bridge declares ``custom`` rules.
    ``model``/``reading`` (names), ``notes`` holds ``N_RULE_CUSTOM`` (the opt-out is never a
    warning), and the text names the run directory the plan will use."""
    bfile = T.write_bridge_file(tmp_path / "random_bridge.py", "random", N=3, D=8, seed=5)
    out = tmp_path / "profile.json"
    rc, env = run_json(capsys, ["bridge", "check", str(bfile), "--bench", "synth_clf", "--n", "4", "--arg", "seed=7", "--out", str(out)])
    assert rc == 0 and env["status"] == "ok", env
    p = env["data"]
    for k in ["describe", "describe_sha256", "model", "reading", "used", "media", "N", "D", "pos_mode", "streams", "invariance",
              "rules_checked", "clips_per_s", "decode_workers", "peak_gpu_gb", "warnings", "notes", "bench", "args", "yavih_version", "created"]:
        assert k in p, k
    assert p["N"] == 3 and p["D"] == 8 and p["pos_mode"] == "constant" and p["streams"] == ["video"]
    assert p["invariance"] == {"rerun_bit_identical": True, "rerun_rel_l2": 0.0, "batch_rel_l2": 0.0, "order_rel_l2": 0.0, "tolerance": 1e-5, "tolerance_reason": None}
    assert p["args"] == {"seed": 7} and p["describe"]["args"] == {"seed": 7}
    assert (p["model"], p["reading"]) == (p["describe"]["model"], p["describe"]["reading"]) == ("yavih-random", "synthetic")
    assert p["used"] == {"video": {"times": []}} and p["media"]["video"]["duration_s"] == 10.0 and p["media"]["video"]["width"] == 16
    assert p["rules_checked"] == {"video": {"temporal": "custom", "spatial": False}} and p["decode_workers"] == 2
    # a custom rule is a legitimate opt-out: a note, never W_RULE_UNCHECKED
    assert not {w["code"] for w in p["warnings"]} & {"W_RULE_UNCHECKED"} and env["warnings"] == p["warnings"]
    assert "N_RULE_CUSTOM" in {n["code"] for n in p["notes"]}
    saved = json.loads(out.read_text())
    assert (saved["model"], saved["reading"], saved["describe_sha256"]) == (p["model"], p["reading"], p["describe_sha256"])
    assert env["paths"]["profile"] == str(out)
    rc = cli.main(["bridge", "check", str(bfile), "--bench", "synth_clf", "--n", "2", "--arg", "seed=7"])
    text = capsys.readouterr().out
    assert rc == 0 and "rerun_bit_identical=True" in text and "rules checked: video custom" in text
    assert "run directory: runs/synth_clf/yavih-random/synthetic/" in text and "note N_RULE_CUSTOM" in text
    assert "@" not in text.splitlines()[0]
    # a different seed changes the configuration digest and nothing of the names (args are recorded,
    # never identity)
    rc, env2 = run_json(capsys, ["bridge", "check", str(bfile), "--bench", "synth_clf", "--n", "2", "--arg", "seed=8"])
    assert env2["data"]["describe_sha256"] != p["describe_sha256"]
    assert (env2["data"]["model"], env2["data"]["reading"]) == (p["model"], p["reading"])


def test_bridge_check_rerun_drift_is_a_hard_failure(capsys, fake_benchmark_module, fake_bridge_module, tmp_path):
    """A bridge whose output changes between two identical passes fails ``E_RERUN_DRIFT`` (exit 2) and
    no profile is written, so ``plan`` cannot schedule it."""
    bfile = tmp_path / "flaky.py"
    bfile.write_text(
        """
import itertools, numpy as np
from yavih.bridge import Bridge
counter = itertools.count()
class Flaky(Bridge):
    name = "flaky"; batch_size = 2; dtype = "float32"; pos_mode = "none"; streams = ("video",)
    ARGS = {}
    def describe(self):
        return {"weights_sha256": "a" * 64, "weights_ref": "r", "dtype": "float32", "batch_size": 2, "streams": ["video"], "pos_mode": "none",
                "reading": "x", "model": "f", "layer": 1, "pooling": "mean", "views": 1, "args": {},
                "input": {"video": {"temporal": {"rule": "custom", "description": "none"}, "spatial": {"rule": "custom", "description": "none"},
                                    "norm": {"source": "none"}, "decoder": {"lib": "none", "version": "0"}}},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True}}
    def encode(self, samples):
        return np.full((len(samples), 2, 4), float(next(counter)), dtype=np.float32), None, [{"video": {"times": []}} for _ in samples]
"""
    )
    out = tmp_path / "flaky.profile.json"
    rc, env = run_json(capsys, ["bridge", "check", str(bfile), "--bench", "synth_clf", "--n", "4", "--out", str(out)])
    assert rc == cli.EXIT_CONTRACT and env["errors"][0]["code"] == "E_RERUN_DRIFT" and not out.exists()


def test_bridge_check_not_applicable(capsys, fake_benchmark_module, fake_bridge_module, tmp_path):
    bfile = tmp_path / "na.py"
    bfile.write_text(
        """
from yavih.bridge import Bridge, NotApplicable
class NA(Bridge):
    name = "na"; ARGS = {}
    def __init__(self, bench, args):
        raise NotApplicable("audio-only bridge: benchmark has no audio stream")
"""
    )
    out = tmp_path / "na.profile.json"
    rc, env = run_json(capsys, ["bridge", "check", str(bfile), "--bench", "synth_clf", "--out", str(out)])
    assert rc == 0 and env["status"] == "not_applicable" and "audio-only" in env["data"]["reason"]
    assert json.loads(out.read_text())["status"] == "not_applicable"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="E_TRAIN_MODE walks torch.nn.Module objects")
def test_bridge_check_train_mode_exit_2(capsys, fake_benchmark_module, fake_bridge_module, tmp_path):
    bfile = T.write_bridge_file(tmp_path / "rb.py", "random")
    # a real torch module left in training mode (nn.Module defaults to training=True), nested in a dict
    code = bfile.read_text() + "\n\nimport torch\n\n_orig = RandomBridge.__init__\ndef _init(self, bench, args):\n    _orig(self, bench, args)\n    self.model = {'inner': torch.nn.Linear(2, 2)}\nRandomBridge.__init__ = _init\n"
    bfile.write_text(code)
    rc, env = run_json(capsys, ["bridge", "check", str(bfile), "--bench", "synth_clf", "--n", "2"])
    assert rc == cli.EXIT_CONTRACT and env["errors"][0]["code"] == "E_TRAIN_MODE"
    assert "model" in env["errors"][0]["message"] and "inner" in env["errors"][0]["message"]


def test_bridge_check_bad_arg_exit_1(capsys, fake_benchmark_module, fake_bridge_module, tmp_path):
    bfile = T.write_bridge_file(tmp_path / "rb.py", "random")
    rc, env = run_json(capsys, ["bridge", "check", str(bfile), "--bench", "synth_clf", "--arg", "nokv"])
    assert rc == cli.EXIT_USAGE
    rc, env = run_json(capsys, ["bridge", "check", str(bfile), "--bench", "synth_clf", "--arg", "layer=3"])
    assert rc == cli.EXIT_USAGE


# ============================================================================= features (validate / stage / inspect / diff)


class FakeFinding:
    def __init__(self, code, message, where=None):
        self.code, self.message, self.where = code, message, where

    def to_json(self):
        return {"code": self.code, "message": self.message, "where": self.where}


def _fake_report(split, ok=True, eligible=True):
    return SimpleNamespace(
        ok=ok,
        errors=[] if ok else [FakeFinding("E_ROWS", "row 3 neither written nor missing")],
        warnings=[FakeFinding("W_COVERAGE", "9/10")],
        notes=[],
        split=split,
        coverage={"nominal": 10, "present": 9, "missing": 1},
        N=4,
        D=16,
        dtype="float32",
        pos_mode="constant",
        bytes=100,
        backend_hint="ram",
        eligibility={"readout": "box", "eligible": eligible, "missing": [] if eligible else ["pos.y", "pos.x"]},
        missing_shards=["--shard 1/2"],
        to_json=lambda: {"ok": ok, "split": split, "coverage": {"nominal": 10, "present": 9}, "N": 4, "D": 16, "dtype": "float32", "pos_mode": "constant", "bytes": 100, "backend_hint": "ram", "eligibility": {"readout": "box", "eligible": eligible, "missing": []}, "missing_shards": ["--shard 1/2"], "errors": [], "warnings": [], "notes": []},
    )


def test_features_validate_parent_iterates_splits(capsys, tmp_path, monkeypatch):
    import yavih.io.validate as V

    feats = tmp_path / "features"
    for s in ("train", "val"):
        (feats / s).mkdir(parents=True)
        (feats / s / "shard-000000.json").write_text("{}")
    calls = []

    def fake_validate(d, *, bench=None, verify_sha=False, sample_rows=512, canary_rel_tol=1e-3, write_meta=True):
        calls.append((Path(d).name, verify_sha, write_meta))
        return _fake_report(Path(d).name, ok=Path(d).name == "train", eligible=False)

    monkeypatch.setattr(V, "validate_features", fake_validate)
    rc, env = run_json(capsys, ["features", "validate", str(feats), "--verify-sha", "--no-write-meta"])
    assert rc == cli.EXIT_CONTRACT and env["status"] == "error"
    assert calls == [("train", True, False), ("val", True, False)]
    assert [r["split"] for r in env["data"]] == ["train", "val"] and env["data"][0]["missing_shards"] == ["--shard 1/2"]
    assert env["errors"][0]["code"] == "E_ROWS" and env["errors"][0]["where"] == str(feats / "val")
    assert len(env["warnings"]) == 2
    # a single split dir → one report, human text has the eligibility line
    rc = cli.main(["features", "validate", str(feats / "train")])
    out = capsys.readouterr().out
    assert rc == 0 and "train: ok" in out and "eligible for box: no (missing: pos.y, pos.x)" in out and "--shard 1/2" in out


def test_features_validate_missing_dir(capsys, tmp_path):
    rc, env = run_json(capsys, ["features", "validate", str(tmp_path / "nope")])
    assert rc == cli.EXIT_USAGE and env["errors"][0]["code"] == "E_NOT_FOUND"
    (tmp_path / "empty").mkdir()
    rc, env = run_json(capsys, ["features", "validate", str(tmp_path / "empty")])
    assert rc == cli.EXIT_USAGE


def test_features_stage(capsys, tmp_path, monkeypatch):
    feats = tmp_path / "run" / "features"
    for s in ("train", "val"):
        d = feats / s
        d.mkdir(parents=True)
        (d / "shard-000000.tar").write_bytes(b"tar" * 10)
        (d / "shard-000000.json").write_text("{}")
        (d / "_meta.json").write_text("{}")
        (d / "shard-000001.tmp").write_bytes(b"partial")
        (d / "logs.txt").write_text("not staged")
    monkeypatch.setenv("YAVIH_UNIT_ID", "abc123")
    scratch = tmp_path / "scratch"
    rc, env = run_json(capsys, ["features", "stage", str(feats), str(scratch)])
    assert rc == 0
    staged = Path(env["data"]["staged"])
    assert staged.parent == scratch and staged.name.startswith("abc123.") and staged.name.endswith(f".{os.getpid()}")
    assert env["paths"]["staged"] == str(staged)
    # the staged copy keeps the features/<split> level: `probe run --scratch <staged>` looks for <staged>/features
    assert sorted(p.name for p in (staged / "features" / "train").iterdir()) == ["_meta.json", "shard-000000.json", "shard-000000.tar"]
    assert env["data"]["splits"]["val"]["bytes"] == 30 + 2 + 2
    rc = cli.main(["features", "stage", str(feats / "val"), str(scratch)])
    out = capsys.readouterr().out.strip()
    assert rc == 0 and Path(out).is_dir() and (Path(out) / "features" / "val" / "shard-000000.tar").is_file()


def test_features_inspect_and_diff_envelopes(capsys, monkeypatch):
    import yavih.io.diff as D
    import yavih.io.inspect as I

    monkeypatch.setattr(I, "inspect_features", lambda d, **kw: {"dims": 4, "dir": d})
    monkeypatch.setattr(D, "diff_features", lambda a, b, **kw: {"rows_compared": 3, "max_abs": 0.0, "max_rel_l2": 0.0, "verdict": "identical"})
    rc, env = run_json(capsys, ["features", "inspect", "x"])
    assert rc == 0 and env["data"]["dims"] == 4
    rc, env = run_json(capsys, ["features", "diff", "a", "b"])
    assert rc == 0 and env["data"]["verdict"] == "identical"


# ============================================================================= probe / report / plan / exec envelopes


def test_probe_run_envelope(capsys, monkeypatch):
    import yavih.readouts as RO

    seen = {}

    def fake_probe(run_dir, *, seeds=(0,), readout=None, cfg=None, deadline=None, scratch=None, allow_low_coverage=False, loader_workers=None,
                   validate_first=False, **kw):
        import inspect

        from yavih.readouts import base

        inspect.signature(base.probe).bind(run_dir, seeds=seeds, deadline=deadline, scratch=scratch, allow_low_coverage=allow_low_coverage,
                                           loader_workers=loader_workers, validate_first=validate_first, **kw)  # the CLI → probe keyword contract
        seen.update(run_dir=run_dir, seeds=seeds, allow=allow_low_coverage, scratch=scratch, validate_first=validate_first,
                    log_samples=kw.get("log_samples"), results_root=kw.get("results_root"), write_results=kw.get("write_results"))
        res = {"benchmark": {"name": "synth_clf"}, "eval_hash": "e" * 64, "model": "m", "reading": "v",
               "metrics": {"acc": 0.9}, "floors": {"acc_floor": 0.2}}
        return [SimpleNamespace(status="ok", seed=s, path=Path(f"/r/seed{s}/result.json"), result=res,
                                metrics={"acc": 0.9}, reason=None, results_file=Path("/res/m/results_T.json"),
                                samples_file=(Path("/res/m/samples_b_v_T.jsonl") if kw.get("log_samples") else None)) for s in seeds]

    import yavih.benchmark as BM

    loads = []

    def fake_load(ref, *a, **k):
        loads.append(ref)
        return SimpleNamespace(name=ref, spec=SimpleNamespace(metrics=SimpleNamespace(headline="acc")))

    monkeypatch.setattr(BM.Benchmark, "load", staticmethod(fake_load))
    monkeypatch.setattr(RO, "probe", fake_probe)
    rc, env = run_json(capsys, ["probe", "run", "/r", "--seeds", "0,1", "--allow-low-coverage", "--scratch", "/s"])
    # the CLI runs the validate prologue by default, like a planned probe unit; the results store is on by default
    assert rc == 0 and seen == {"run_dir": "/r", "seeds": (0, 1), "allow": True, "scratch": "/s", "validate_first": True,
                                "log_samples": False, "results_root": None, "write_results": True}
    # each summary names the comparability key and the row key, and the headline
    assert env["data"][1] == {"status": "ok", "seed": 1, "path": "/r/seed1/result.json", "benchmark": "synth_clf", "eval_hash": "e" * 64,
                              "model": "m", "reading": "v", "headline": {"name": "acc", "value": 0.9},
                              "metrics": {"acc": 0.9}, "floors": {"acc_floor": 0.2}, "reason": None}
    assert loads == ["synth_clf"], "the headline is looked up once per benchmark"
    assert env["paths"] == {"results_file": "/res/m/results_T.json"}
    rc = cli.main(["probe", "run", "/r"])
    assert rc == 0 and f"synth_clf/m/v eval_hash={'e' * 12}" in capsys.readouterr().out
    rc, env = run_json(capsys, ["probe", "run", "/r", "--no-validate"])
    assert rc == 0 and seen["validate_first"] is False
    rc, env = run_json(capsys, ["probe", "run", "/r", "--log-samples", "--results", "/res"])
    assert rc == 0 and seen["log_samples"] is True and seen["results_root"] == "/res"
    assert env["paths"] == {"results_file": "/res/m/results_T.json", "samples_files": ["/res/m/samples_b_v_T.jsonl"]}
    rc, env = run_json(capsys, ["probe", "run", "/r", "--no-results"])
    assert rc == 0 and seen["write_results"] is False
    assert cli.main(["probe", "run", "/r", "--no-results", "--results", "/x"]) != 0  # mutually exclusive: a usage error
    capsys.readouterr()

    def na_probe(run_dir, **kw):
        return [SimpleNamespace(status="not_applicable", seed=0, path=None, result={}, metrics={}, reason="pos missing")]

    monkeypatch.setattr(RO, "probe", na_probe)
    rc, env = run_json(capsys, ["probe", "run", "/r"])
    assert rc == 0 and env["status"] == "not_applicable"

    def refused(run_dir, **kw):
        from yavih.readouts.base import ProbeRefused

        raise ProbeRefused("E_COVERAGE", "eval coverage 0.5 < 0.95")

    monkeypatch.setattr(RO, "probe", refused)
    rc, env = run_json(capsys, ["probe", "run", "/r"])
    assert rc == 2 and env["errors"][0]["code"] == "E_COVERAGE"


def test_report_build_envelope(capsys, monkeypatch, tmp_path):
    import yavih.report as RP

    seen = {}

    def fake_report(globs, *, fmt="md", seeds="aggregate", allow_coverage_diff=False, join=None, detail=False, results_root=None, merge_results=False):
        seen.update(globs=list(globs), fmt=fmt, join=join, detail=detail, results_root=results_root, merge_results=merge_results)
        merged = ["/res/m@x/results_T.json"] if merge_results else []
        joins = {"vggsounder": {"path": "/res/fam/results_T.json"}} if join else {}
        return SimpleNamespace(text="| a |\n", to_json=lambda: {"rows": [1]}, merged=merged, joins=joins)

    monkeypatch.setattr(RP, "report", fake_report)
    out = tmp_path / "r.md"
    rc, env = run_json(capsys, ["report", "build", "runs/*", "more/*", "--fmt", "md", "--out", str(out), "--join", "vggsounder", "a=A", "v=V", "av=AV", "--detail"])
    assert rc == 0 and env["data"] == {"rows": [1]} and out.read_text() == "| a |\n"
    assert env["paths"] == {"out": str(out), "join_vggsounder": "/res/fam/results_T.json"}
    assert seen == {"globs": ["runs/*", "more/*"], "fmt": "md", "join": {"vggsounder": {"a": "A", "v": "V", "av": "AV"}}, "detail": True,
                    "results_root": None, "merge_results": False}
    rc, env = run_json(capsys, ["report", "build", "runs/*", "--merge-results", "--results", "/res"])
    assert rc == 0 and seen["merge_results"] is True and seen["results_root"] == "/res" and env["paths"] == {"merged": ["/res/m@x/results_T.json"]}
    rc = cli.main(["report", "build", "runs/*"])
    assert rc == 0 and capsys.readouterr().out == "| a |\n"
    # the report's warnings are also the envelope's Findings, where = the column id
    warn = {"code": "W_BRIDGE_CHANGED", "column": "b:0123456789ab", "message": "m1 a on b: results from 2 bridge configurations"}
    monkeypatch.setattr(RP, "report", lambda globs, **kw: SimpleNamespace(text="| a |\n", to_json=lambda: {"rows": [1], "warnings": [warn]},
                                                                          merged=[], joins={}, warnings=[warn]))
    rc, env = run_json(capsys, ["report", "build", "runs/*"])
    assert rc == 0 and env["status"] == "ok" and env["data"]["warnings"] == [warn]
    assert env["warnings"] == [{"code": "W_BRIDGE_CHANGED", "message": warn["message"], "where": "b:0123456789ab"}]
    # --out naming an existing directory writes <dir>/report.<fmt> (SKILL step 9) instead of failing E_INTERNAL
    rdir = tmp_path / "reports"
    rdir.mkdir()
    rc, env = run_json(capsys, ["report", "build", "runs/*", "--out", str(rdir)])
    assert rc == 0 and env["paths"]["out"] == str(rdir / "report.md") and (rdir / "report.md").read_text() == "| a |\n"


def test_plan_and_exec_envelopes(capsys, monkeypatch, tmp_path):
    import yavih.plan.calibrate as C
    import yavih.plan.exec as X
    import yavih.plan.status as S
    import yavih.plan.units as U

    units = [SimpleNamespace(to_json=lambda i=i: {"id": f"u{i}", "kind": "extract"}) for i in range(3)]
    monkeypatch.setattr(U, "plan", lambda m, *, pending=False, kind=None, shard_minutes=None: units)
    written = {}
    monkeypatch.setattr(U, "write_plan", lambda us, path: written.update(path=path, n=len(us)))
    rc, env = run_json(capsys, ["plan", "build", "m.yaml", "--out", str(tmp_path / "p.jsonl"), "--pending", "--shard-minutes", "5"])
    assert rc == 0 and len(env["data"]) == 3 and written == {"path": str(tmp_path / "p.jsonl"), "n": 3} and env["paths"]["plan"] == str(tmp_path / "p.jsonl")
    rc, env = run_json(capsys, ["plan", "build", "m.yaml", "--format", "count"])
    assert rc == 0 and env["data"] == {"count": 3}
    rc = cli.main(["plan", "build", "m.yaml", "--format", "count"])
    assert rc == 0 and capsys.readouterr().out == "3\n"
    rc = cli.main(["plan", "build", "m.yaml"])
    assert rc == 0 and len(capsys.readouterr().out.strip().splitlines()) == 3

    monkeypatch.setattr(S, "plan_status", lambda p: {"done": 1, "pending": 2, "not_ready": 0, "preempted": 0, "failed": 0, "not_applicable": 0, "by_kind": {}})
    rc, env = run_json(capsys, ["plan", "status", "p.jsonl"])
    assert rc == 0 and env["data"]["pending"] == 2
    monkeypatch.setattr(C, "calibrate", lambda root: {"multipliers": {"extract": 1.2}})
    rc, env = run_json(capsys, ["plan", "calibrate", "runs"])
    assert rc == 0 and env["data"]["multipliers"]["extract"] == 1.2

    for status, exit_code, env_status in [("done", 0, "ok"), ("skipped", 0, "skipped"), ("not_applicable", 0, "not_applicable"), ("preempted", 4, "error"), ("not_ready", 3, "error"), ("failed", 5, "error")]:
        monkeypatch.setattr(X, "exec_unit", lambda p, *, index=None, unit_id=None, deadline=None, deadline_margin=300.0, slots=None, _s=status, _e=exit_code: {"id": "u0", "status": _s, "exit_code": _e})
        rc, env = run_json(capsys, ["exec", "p.jsonl", "--index", "0"])
        assert (rc, env["status"]) == (exit_code, env_status), status
        if env_status == "error":
            assert env["errors"][0]["code"] == f"E_UNIT_{status.upper()}"
    assert cli.main(["exec", "p.jsonl"]) == cli.EXIT_USAGE  # --index xor --id required


def test_extract_envelope_passes_arguments(capsys, monkeypatch):
    import yavih.extract as E

    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"status": "ok", "rows": {"n_written": 5}}

    monkeypatch.setattr(E, "run_extract", fake_run)
    rc, env = run_json(capsys, ["extract", "--bench", "b", "--bridge", "x.py", "--split", "train", "--shard", "1/4", "--out", "runs", "--arg", "layer=3",
                                "--canaries", "2", "--deadline", "1700000000", "--expect-profile", "p.json"])
    assert rc == 0 and env["data"]["rows"]["n_written"] == 5
    assert seen["shard"] == (1, 4) and seen["args"] == {"layer": "3"} and seen["canaries"] == 2 and seen["deadline"] == 1700000000.0
    assert seen["expect_profile"] == "p.json"

    def skipped(**kw):
        return {"status": "skipped"}

    monkeypatch.setattr(E, "run_extract", skipped)
    rc, env = run_json(capsys, ["extract", "--bench", "b", "--bridge", "x.py", "--split", "train", "--shard", "0/1", "--out", "runs"])
    assert rc == 0 and env["status"] == "skipped"

    def drift(**kw):
        raise E.ExtractError("E_CONFIG_DRIFT", "the bridge's configuration drifted since `bridge check`: keys ['layer'] differ", exit_code=2)

    monkeypatch.setattr(E, "run_extract", drift)
    rc, env = run_json(capsys, ["extract", "--bench", "b", "--bridge", "x.py", "--split", "train", "--shard", "0/1", "--out", "runs",
                                "--expect-profile", "p.json"])
    assert rc == 2 and env["errors"][0]["code"] == "E_CONFIG_DRIFT"


# ============================================================================= yavih.testing: synthetic benchmarks

REQUIRED_TOP_LEVEL_KEYS = {  # benchmark.yaml schema 2
    "schema_version", "name", "aliases", "title", "readout", "protocol_version", "changelog", "source", "budget",
    "manifest", "splits", "streams", "targets", "data", "input_contract", "output_contract", "external", "protocol",
    "metrics", "readout_params", "header",
}
README_H2 = ["Description", "Citation", "In yavih"]  # README.md contract
SETUP_H2 = ["Obtain the data", "Dependencies", "Input contract", "Output contract", "Prepare & verify", "Gotchas"]  # setup.md contract


def _headings(text: str, level: int) -> list[str]:
    out, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("#" * level + " "):
            out.append(line[level + 1 :].strip())
    return out


def _last_yaml_block(text: str, heading: str) -> dict:
    body = text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    return yaml.safe_load(body.rsplit("```yaml\n", 1)[1].split("```", 1)[0])


def _read_gz_csv(path):
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    return raw, list(csv.DictReader(io.StringIO(raw.decode())))


@pytest.mark.parametrize("readout", T.READOUTS)
def test_make_synthetic_benchmark_folder_is_valid(tmp_path, readout):
    folder = T.make_synthetic_benchmark(tmp_path, readout, n_train=12, n_eval=6, media="none")
    assert folder == tmp_path / f"synth_{readout}"
    for f in ("README.md", "setup.md", "benchmark.yaml", "manifest.csv.gz", "reader.py", "bridge_template.py", "make_manifest.py",
              "prepare/01_generate.sh", "tests/test_folder.py"):
        assert (folder / f).is_file(), f
    spec = yaml.safe_load((folder / "benchmark.yaml").read_text())
    assert set(spec) == REQUIRED_TOP_LEVEL_KEYS and spec["schema_version"] == 2 and spec["name"] == folder.name and spec["readout"] == readout
    assert spec["protocol_version"] == 1 and [c["version"] for c in spec["changelog"]] == [1] and spec["external"] == []
    # contracts: handed_over/info name exactly the streams; the output contract is pinned
    ic, oc = spec["input_contract"], spec["output_contract"]
    assert set(ic["handed_over"]) == set(ic["info"]) == set(spec["streams"]) and ic["materialize"] is None
    assert all(f in spec["protocol"] for f in ic["facts"])
    assert oc["shape"] == "fixed" and oc["coordinates"] == "media" and (oc["target_frame"] == "full_frame") == (readout in ("grid", "box"))
    from yavih.benchmark import Benchmark

    assert Benchmark.from_dir(folder).spec.readout == readout  # the schema-2 loader accepts it
    raw, rows = _read_gz_csv(folder / "manifest.csv.gz")
    assert hashlib.sha256(raw).hexdigest() == spec["manifest"]["sha256"] == spec["manifest"]["upstream"]["sha256"]
    assert len(rows) == 18 and len({r["clip_id"] for r in rows}) == 18 and {r["split"] for r in rows} == {"train", "val"}
    assert all(set(r["clip_id"]) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.#:-") for r in rows)
    # declared columns exist, media columns reference declared streams and vice versa
    cols = spec["manifest"]["columns"]
    assert set(cols) <= set(rows[0]) and "clip_id" not in cols and "split" not in cols
    media_cols = {c: v["stream"] for c, v in cols.items() if v["role"] == "media"}
    assert set(media_cols.values()) == set(spec["streams"])
    for sname, st in spec["streams"].items():
        assert st["media_column"] in cols and cols[st["media_column"]]["stream"] == sname
        assert st["root"] in spec["data"]["roots"]
    # classes
    if spec["manifest"]["classes"]:
        cb = (folder / "classes.csv").read_bytes()
        assert hashlib.sha256(cb).hexdigest() == spec["manifest"]["classes_sha256"]
        classes = list(csv.DictReader(io.StringIO(cb.decode())))
        assert [int(c["index"]) for c in classes] == list(range(len(classes))) and len({c["name"] for c in classes}) == len(classes)
        names = {c["name"] for c in classes}
        assert all(r[spec["manifest"]["label"]["column"]] in names for r in rows)
    else:
        assert spec["manifest"]["classes_sha256"] is None and not (folder / "classes.csv").exists()
    # roles cover every split value exactly once
    s = spec["splits"]
    assigned = s["train"] + s["eval"] + s["floor"] + s["unused"] + (s["folds"] or [])
    assert sorted(assigned) == sorted({r["split"] for r in rows})
    # metrics resolve in the registry for this readout; protocol keys declared
    specs = dict(R.resolve_names(spec["metrics"]["list"], readout, kinds=["metric", "disclosure"]))
    specs.update(R.resolve_names(spec["metrics"]["floors"], readout, kinds=["floor"]))
    specs.update(R.resolve_names(spec["metrics"]["controls"], readout, kinds=["control"]))
    for key in R.protocol_keys_for(specs.values()):
        assert key in spec["protocol"], key
    assert spec["metrics"]["headline"] in spec["metrics"]["list"]
    assert set(spec["metrics"]["ci"]) <= set(spec["metrics"]["list"]) and all(v in R.CI_KINDS for v in spec["metrics"]["ci"].values())
    assert set(spec["metrics"]["chance"]) <= set(spec["metrics"]["list"])
    # readout_params keys are the allowed set; requires streams are declared
    from yavih.benchmark import READOUT_PARAM_KEYS

    assert set(spec["readout_params"]) <= READOUT_PARAM_KEYS[readout]
    assert set(spec["output_contract"]["requires"]["streams"]) <= set(spec["streams"])
    for k in ("visual_stream", "conditioning_stream"):
        if k in spec["readout_params"]:
            assert spec["readout_params"][k] in spec["streams"]
    assert (spec["targets"] is not None) == (readout == "grid")
    cur = spec["manifest"]["upstream"]["curation"]
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"] == 18
    assert spec["budget"]["clips"] == {"train": 12, "val": 6}
    # README.md / setup.md in their contract forms; setup.md ends each contract section with a yaml copy
    readme = (folder / "README.md").read_text()
    assert _headings(readme, 1) == [] and _headings(readme, 2) == README_H2
    assert "protocol_version" in readme and spec["metrics"]["headline"] in readme
    setup = (folder / "setup.md").read_text()
    assert _headings(setup, 1) == [f"{folder.name} — setup"] and _headings(setup, 2) == SETUP_H2
    assert _last_yaml_block(setup, "Input contract") == ic and _last_yaml_block(setup, "Output contract") == oc
    assert f"yavih bench verify {folder.name}" in setup and "None." in setup
    # media "none": no media files, roots default to absolute (non-existent) dirs
    media_root = spec["data"]["roots"]["media_root"]["default"]
    assert os.path.isabs(media_root) and not Path(media_root).exists()


def test_make_synthetic_benchmark_is_byte_identical_and_validates_args(tmp_path):
    a = T.make_synthetic_benchmark(tmp_path / "a", "clf", n_train=4, n_eval=2)
    b = T.make_synthetic_benchmark(tmp_path / "a", "clf", n_train=4, n_eval=2)
    assert a == b and (a / "manifest.csv.gz").read_bytes() == (b / "manifest.csv.gz").read_bytes()
    assert (a / "classes.csv").read_bytes() == (b / "classes.csv").read_bytes()
    with pytest.raises(ValueError):
        T.make_synthetic_benchmark(tmp_path, "nope")
    with pytest.raises(ValueError):
        T.make_synthetic_benchmark(tmp_path, "clf", media="huge")
    with pytest.raises(ValueError):
        T.make_synthetic_benchmark(tmp_path, "clf", n_train=0)


def test_synthetic_seq_rows_and_grid_targets(tmp_path):
    seq = T.make_synthetic_benchmark(tmp_path, "seq", n_train=9, n_eval=3)
    spec = yaml.safe_load((seq / "benchmark.yaml").read_text())
    rows = T.read_manifest_rows(seq)
    ns = spec["protocol"]["num_segments"]
    # AVE: all ten seconds scored, 8 of 10 must hold a timed token; labels per second are facts
    assert spec["protocol"]["background_index"] == 4 and spec["output_contract"]["requires"]["t_bins"] == 8 and spec["readout_params"] == {"T": 10}
    assert spec["protocol"]["segments_scored"] == 10 and spec["protocol"]["segment_offset"] == 0
    assert spec["input_contract"]["facts"] == ["num_segments", "segment_seconds"]
    assert all(0 <= int(r["event_start"]) < int(r["event_end"]) <= ns for r in rows)
    assert any(int(r["event_end"]) - int(r["event_start"]) < ns for r in rows) and any(int(r["event_end"]) - int(r["event_start"]) == ns for r in rows)
    assert all(r["label"] != "background" for r in rows)

    grid = T.make_synthetic_benchmark(tmp_path, "grid", n_train=5, n_eval=3)
    gspec = yaml.safe_load((grid / "benchmark.yaml").read_text())
    S = gspec["protocol"]["target_size"]
    for split, n in (("train", 5), ("val", 3)):
        t = gspec["targets"][split]
        path = Path(gspec["data"]["roots"][t["root"]]["default"]) / t["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == t["sha256"] == gspec["data"]["derived"][t["file"]]
        arr = np.load(path)
        assert arr.shape == (n, S, S) and arr.dtype == np.float32
        assert (arr[:, :2, :] == 0).all() and (arr[:, 2:, :] > 0).all() and (arr <= gspec["protocol"]["max_depth"]).all()
        cids = [r["clip_id"] for r in T.read_manifest_rows(grid) if r["split"] == split]
        assert np.array_equal(arr[1], T.synthetic_depth(cids[1], size=S))

    assert gspec["protocol"]["target_region"] == list(T.GRID_TARGET_REGION) and gspec["input_contract"]["facts"] == ["target_region"]
    box = T.make_synthetic_benchmark(tmp_path, "box", n_train=4, n_eval=2)
    bspec = yaml.safe_load((box / "benchmark.yaml").read_text())
    assert bspec["input_contract"]["facts"] == ["annotated_second"] and bspec["protocol"]["annotated_second"] == 5.0
    for r in T.read_manifest_rows(box):
        x1, y1, x2, y2 = (float(v) for v in r["box"].split("|"))
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def test_tiny_media_is_decodable(tmp_path):
    import av
    import soundfile as sf
    from PIL import Image

    seq = T.make_synthetic_benchmark(tmp_path, "seq", n_train=2, n_eval=1, media="tiny")
    spec = yaml.safe_load((seq / "benchmark.yaml").read_text())
    root = Path(spec["data"]["roots"]["media_root"]["default"])
    rows = T.read_manifest_rows(seq)
    for r in rows:
        with av.open(str(root / r["rel_video"])) as c:
            st = c.streams.video[0]
            pts = [float(f.pts * f.time_base) for f in c.decode(st)]
        assert len(pts) == 20 and pts[0] == 0.0 and abs(pts[-1] - 9.5) < 1e-6 and st.width == 16
        info = sf.info(str(root / r["rel_audio"]))
        assert info.samplerate == 8000 and abs(info.duration - 10.0) < 1e-6 and info.channels == 1
    grid = T.make_synthetic_benchmark(tmp_path, "grid", n_train=1, n_eval=1, media="tiny")
    gspec = yaml.safe_load((grid / "benchmark.yaml").read_text())
    img = Image.open(Path(gspec["data"]["roots"]["media_root"]["default"]) / T.read_manifest_rows(grid)[0]["rel_media"])
    assert img.size == (48, 32) and img.mode == "RGB"  # not square: full-frame fractions are exercised
    # re-running does not rewrite existing media and keeps the manifest identical
    before = (root / rows[0]["rel_video"]).stat().st_mtime_ns
    T.make_synthetic_benchmark(tmp_path, "seq", n_train=2, n_eval=1, media="tiny")
    assert (root / rows[0]["rel_video"]).stat().st_mtime_ns == before


# ============================================================================= yavih.testing: bridges


def _real_bench(tmp_path, readout, n_train, n_eval):
    from yavih.benchmark import Benchmark

    folder = T.make_synthetic_benchmark(tmp_path, readout, n_train=n_train, n_eval=n_eval, media="tiny")
    return folder, Benchmark.from_dir(folder)


def _samples(bench, split: str, n: int):
    return list(bench.reader(split, rows=range(0, n)))


def _scrambled(samples):
    from yavih.extract import scramble

    return [scramble(s, k) for k, s in enumerate(samples)]


@pytest.mark.parametrize("readout", T.READOUTS)
def test_random_bridge_is_pure_function_of_the_media(tmp_path, readout):
    """``feat = default_rng(sha256(f"{seed}:{bench}:" + sha256 of every stream's bytes))``; the bridge
    receives the ``BenchmarkView``, declares ``custom`` rules and reports empty ``used``."""
    folder, bench = _real_bench(tmp_path, readout, 6, 3)
    view = bench.view()
    cls = T.random_bridge(bench, N=4, D=16, seed=3)
    assert issubclass(cls, Bridge) and cls.name == "yavih-random" and cls.__module__ == "yavih.testing"
    b = cls(view, {"seed": 3})
    d = b.describe()
    assert set(DESCRIBE_REQUIRED) <= set(d) and d["streams"] == list(b.streams) and d["pos_mode"] == "constant" and d["args"] == {"seed": 3}
    assert d["views"] == 1 and set(d["input"]) == set(bench.spec.streams)
    from yavih.bridge import validate_describe

    validate_describe(d, b, view)
    canon(d)  # JSON primitives only
    samples = _samples(bench, "train", 4)
    feats, pos, used = b.encode(samples)
    assert feats.dtype == np.float32 and feats.shape[0] == 4 and feats.shape[2] == 16 and pos.shape == (feats.shape[1], 4)
    for u in used:
        assert u == {s: ({"times": []} if k == "video" else {"samples": [], "pad": [0, 0]} if k == "audio" else {}) for s, k in view.streams.items()}
    # the random bridge's formula, exactly
    s1 = samples[1]
    digest = "".join(hashlib.sha256(s1.streams[n].bytes()).hexdigest() for n in sorted(s1.streams))
    expect = np.random.default_rng(int.from_bytes(hashlib.sha256(f"3:{bench.name}:{digest}".encode()).digest(), "big")).standard_normal((feats.shape[1], 16))
    assert np.array_equal(feats[1], expect.astype(np.float32))
    # order / batch independent, and blind to row / clip_id / split (bridge check's alone pass); another seed differs
    again, _, _ = b.encode(samples[::-1])
    assert np.array_equal(again[::-1], feats)
    alone = np.stack([b.encode([s])[0][0] for s in _scrambled(samples)])
    assert np.array_equal(alone, feats)
    other = T.random_bridge(bench, N=4, D=16, seed=4)(view, {"seed": 4})
    assert not np.array_equal(other.encode(samples)[0], feats)
    # pos semantics: stream tags index describe()["streams"]; media coordinates in [0,1] or -1
    assert ((pos[:, 1:] >= 0) & (pos[:, 1:] <= 1) | (pos[:, 1:] == -1)).all() and set(pos[:, 0].astype(int)) <= set(range(len(b.streams)))
    req = bench.spec.output_contract.requires
    if "t" in req.pos:
        assert len({int(t * 10) for t in pos[:, 1] if t >= 0}) >= req.t_bins  # scored seconds of a 10 s clip
    if "y" in req.pos:
        g = req.grid
        y0, x0, y1, x1 = bench.spec.target_region
        cells = {(int((y - y0) / (y1 - y0) * g["h"]), int((x - x0) / (x1 - x0) * g["w"])) for _, _, y, x in pos if y >= 0}
        assert len(cells) == g["h"] * g["w"]  # every cell of the target region holds a token
    if readout == "box":
        tags = set(pos[:, 0].astype(int))
        assert tags == {b.streams.index("video"), b.streams.index("audio")}


def test_random_bridge_options():
    cls = T.random_bridge("k400", N=2, D=8, pos_mode="none", dtype="float16", batch_size=7)
    assert cls.pos_mode == "none" and cls.dtype == "float16" and cls.batch_size == 7 and cls._bench_name == "k400"
    with pytest.raises(ValueError):
        T.random_bridge("k400", dtype="bfloat16")
    with pytest.raises(ValueError):
        T.random_bridge("k400", pos_mode="weird")
    with pytest.raises(ValueError):
        T.oracle_bridge("k400", D=4)


def _decode_label(code_row, codebook):
    return int(np.argmax(codebook @ code_row))


@pytest.mark.parametrize("readout", T.READOUTS)
def test_oracle_bridge_encodes_labels(tmp_path, readout):
    """Oracle: features encode the label (found from the media file, so the scrambled identity of
    bridge check's alone pass changes nothing); it decodes under declared, checkable rules and reports the
    decoded pts / sample ranges as ``used``."""
    folder, bench = _real_bench(tmp_path, readout, 8, 4)
    view = bench.view()
    cls = T.oracle_bridge(bench, N=4, D=16)
    b = cls(view, {})
    d = b.describe()
    assert set(DESCRIBE_REQUIRED) <= set(d) and d["weights_sha256"] is not None and len(d["weights_sha256"]) == 64 and d["args"] == {}
    from yavih.bridge import validate_describe

    validate_describe(d, b, view)
    assert all(e["temporal"]["rule"] != "custom" for s, e in d["input"].items() if view.streams[s] != "image")
    canon(d)
    samples = _samples(bench, "val", 4)
    feats, pos, used = b.encode(samples)
    again, _, used2 = b.encode(_scrambled(samples)[1:3])
    assert np.array_equal(again, feats[1:3]) and used2 == used[1:3]  # pure function of the media → bit-identical canaries
    for u in used:
        for sname, kind in view.streams.items():
            if kind == "video":
                # box: the annotated second; seq: one frame per timed token, at its time (pos t = used time / duration_s)
                want = [5.0] if readout == "box" else ([k + 0.5 for k in range(10)] if readout == "seq" else [1.0, 3.0, 5.0, 7.0, 9.0])
                assert u[sname] == {"times": want}
            elif kind == "audio":
                assert u[sname] == {"samples": [[0, 80000]], "pad": [0, 0]}
            else:
                assert u[sname] == {}
    rows = {r["clip_id"]: r for r in T.read_manifest_rows(folder)}
    spec = yaml.safe_load((folder / "benchmark.yaml").read_text())
    if readout == "clf":
        classes = {c["name"]: int(c["index"]) for c in csv.DictReader(io.StringIO((folder / "classes.csv").read_text()))}
        for s, f in zip(samples, feats, strict=True):
            assert all(_decode_label(tok, b.codebook) == classes[rows[s.clip_id]["label"]] for tok in f)
    elif readout == "seq":
        classes = {c["name"]: int(c["index"]) for c in csv.DictReader(io.StringIO((folder / "classes.csv").read_text()))}
        bg = spec["protocol"]["background_index"]
        for s, f in zip(samples, feats, strict=True):
            r = rows[s.clip_id]
            for tok, p in zip(f, pos, strict=True):
                sec = int(p[1] * 10.0)  # absolute second of a 10 s clip
                expect = classes[r["label"]] if int(r["event_start"]) <= sec < int(r["event_end"]) else bg
                assert _decode_label(tok, b.codebook) == expect
    elif readout == "grid":
        S = spec["protocol"]["target_size"]
        y0, x0, y1, x1 = spec["protocol"]["target_region"]
        for s, f in zip(samples, feats, strict=True):
            depth = T.synthetic_depth(s.clip_id, size=S)
            for tok, p in zip(f, pos, strict=True):
                i, j = int((p[2] - y0) / (y1 - y0) * 4), int((p[3] - x0) / (x1 - x0) * 4)  # cell of the target region
                cell = depth[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                d_true = float(cell[cell > 0].mean())
                assert abs(tok[0] - np.log(d_true)) < 0.2 and abs(tok[1] - d_true / 10.0) < 0.2
    elif readout == "box":
        aud = b.streams.index("audio")
        for s, f in zip(samples, feats, strict=True):
            x1, y1, x2, y2 = (float(v) for v in rows[s.clip_id]["box"].split("|"))
            for tok, p in zip(f, pos, strict=True):
                if int(p[0]) == aud:
                    cx, cy, w, h = (tok[:4] + 1) / 2
                    assert abs(cx - (x1 + x2) / 2) < 0.1 and abs(w - (x2 - x1)) < 0.1 and abs(cy - (y1 + y2) / 2) < 0.1 and abs(h - (y2 - y1)) < 0.1
                else:
                    inside = x1 <= p[3] < x2 and y1 <= p[2] < y2
                    assert (tok[0] > 0) == inside
    # rows are not byte-identical to one another (E_DUPLICATE_CLIPS) and tokens vary (E_DEAD_ROWS)
    assert not np.array_equal(feats[0], feats[1]) and feats.std(axis=0).min() > 0


def test_oracle_bridge_unknown_media_raises(tmp_path):
    folder, bench = _real_bench(tmp_path, "clf", 2, 2)
    b = T.oracle_bridge(bench)(bench.view(), {})
    bad = SimpleNamespace(clip_id="ghost", streams={"video": SimpleNamespace(path=str(tmp_path / "elsewhere" / "ghost.mp4"), kind="video")})
    with pytest.raises(KeyError):
        b.encode([bad])


def test_write_bridge_file_defines_single_bridge_subclass(tmp_path):
    for kind in ("random", "oracle"):
        p = T.write_bridge_file(tmp_path / f"{kind}.py", kind, N=3, D=8, seed=9, batch_size=2)
        ns: dict = {}
        exec(compile(p.read_text(), str(p), "exec"), ns)  # noqa: S102
        classes = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Bridge)]
        assert len(classes) == 1 and classes[0].N == 3 and classes[0].D == 8 and classes[0].batch_size == 2
        assert classes[0].__module__ == "builtins" or classes[0].__module__ != "yavih.testing"
        if kind == "random":
            assert classes[0].seed == 9 and classes[0].ARGS["seed"].default == 9
    with pytest.raises(ValueError):
        T.write_bridge_file(tmp_path / "x.py", "weird")


# ============================================================================= fast()


def test_fast_requires_testing_env(monkeypatch):
    from yavih.probes.config import ProbeConfig

    cfg = ProbeConfig(readout="clf", head={"family": "attentive_probe_v1"}, lr=1e-4, lr_grid=(1e-4, 3e-4), weight_decay=1e-4, max_epochs=30, batch_size=256, patience=5, warmup_fraction=0.0, loss="ce")
    f = T.fast(cfg)
    assert f == cfg.fast()  # one definition: ProbeConfig.fast()
    assert f.max_epochs == 25 and f.patience == 25 and f.lr == 3e-3 and f.lr_grid == () and f.batch_size == 16 and f.max_tokens_per_forward == 32768
    assert f.head["family"] == cfg.head["family"] and f.head["self_attn_depth"] == 1 and f.head["num_heads"] == 4
    assert f.loss == "ce" and f.readout == "clf" and f.precision == "fp32" and f.selection == "val_loss"
    assert T.testing_enabled() is True
    monkeypatch.delenv("YAVIH_TESTING")
    assert T.testing_enabled() is False
    with pytest.raises(RuntimeError, match="YAVIH_TESTING"):
        T.fast(cfg)
    monkeypatch.setenv("YAVIH_TESTING", "1")
    with pytest.raises(TypeError):
        T.fast({"lr": 1.0})


def test_import_boundary_testing_module():
    """``yavih.testing`` is importable without torch loaded (it is allowed torch but needs none)."""
    import subprocess

    code = "import sys; sys.modules['torch'] = None; import yavih.testing, yavih.cli; print('ok')"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


# ============================================================================= --json envelope and doctor edge cases


def test_json_envelope_survives_prints_from_command_code(capsys, monkeypatch):
    """Anything a command prints to stdout (a bridge's model constructor, a library) goes to stderr under --json; the
    envelope stays the only stdout line."""

    def noisy(ns):
        print("model geometry: 14x14 patches")
        sys.stdout.write("more noise\n")
        return cli.Envelope(data={"x": 1}, text="human text\n")

    monkeypatch.setattr(cli, "cmd_skill_show", noisy)
    rc, env = run_json(capsys, ["skill", "show"])  # asserts exactly one line on stdout
    assert rc == 0 and env["data"] == {"x": 1}
    rc = cli.main(["--json", "skill", "show"])
    out = capsys.readouterr()
    assert rc == 0 and json.loads(out.out)["data"] == {"x": 1} and "model geometry" in out.err and "more noise" in out.err
    # without --json stdout is untouched
    rc = cli.main(["skill", "show"])
    out = capsys.readouterr().out
    assert rc == 0 and "model geometry" in out and out.endswith("human text\n")
    # an exception after printing still yields one envelope on stdout

    def noisy_fail(ns):
        print("half-built model")
        raise cli.CliError("E_X", "nope", exit_code=2)

    monkeypatch.setattr(cli, "cmd_skill_show", noisy_fail)
    rc, env = run_json(capsys, ["skill", "show"])
    assert rc == 2 and env["errors"][0]["code"] == "E_X"


def test_json_envelope_survives_fd_level_and_child_writes(tmp_path):
    """C-level writes to fd 1 and inherited child stdout are routed to stderr too (real descriptors)."""
    import subprocess

    code = (
        "import os, subprocess, sys\n"
        "from yavih import cli\n"
        "def noisy(ns):\n"
        "    print('py noise')\n"
        "    sys.stdout.flush()\n"
        "    os.write(1, b'fd noise\\n')\n"
        "    subprocess.run([sys.executable, '-c', 'print(\"child noise\")'], check=True)\n"
        "    return cli.Envelope(data={'ok': True})\n"
        "cli.cmd_skill_show = noisy\n"
        "rc = cli.main(['--json', 'skill', 'show'])\n"
        "print('after main')\n"
        "sys.exit(rc)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert json.loads(lines[0])["data"] == {"ok": True} and lines[1:] == ["after main"], proc.stdout  # fd 1 restored
    for noise in ("py noise", "fd noise", "child noise"):
        assert noise in proc.stderr and noise not in proc.stdout


def test_doctor_reads_a_fold_split_and_warns_when_nothing_is_readable(capsys, fake_decoders, fake_benchmark_module, synth_clf):
    """A benchmark with only fold splits: doctor reads a fold, and warns when nothing is readable."""
    bench = fake_benchmark_module["synth_clf"]
    bench.roles = lambda: {"train": [], "eval": [], "floor": [], "folds": ["train", "val"]}
    rc, env = run_json(capsys, ["doctor", "--bench", "synth_clf"])
    mr = env["data"]["media_read"]
    assert rc == 0 and mr["ok"] is True and mr["split"] == "train" and mr["clip_id"].startswith("train_")
    assert not any(w["code"] == "W_MEDIA_READ" for w in env["warnings"])
    bench.roles = lambda: {"train": [], "eval": ["nosuchsplit"], "floor": [], "folds": []}
    rc, env = run_json(capsys, ["doctor", "--bench", "synth_clf"])
    mr = env["data"]["media_read"]
    assert rc == 0 and mr["ok"] is False and mr["split"] is None and "no eval, train, fold or floor split" in mr["error"]
    assert any(w["code"] == "W_MEDIA_READ" for w in env["warnings"])
    rc = cli.main(["doctor", "--bench", "synth_clf"])
    assert "media_read  None ok=False split=None  (no eval" in capsys.readouterr().out


def _fake_ffmpeg(path: Path, version: str, x264: str) -> Path:
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:] == ['-version']:\n"
        f"    print('ffmpeg version {version} Copyright (c) 2000-2021 the FFmpeg developers')\n"
        "    sys.exit(0)\n"
        f"sys.stdout.buffer.write(b'\\x00\\x00\\x01\\x06\\x05' + {x264.encode()!r} + b' - H.264/MPEG-4 AVC codec')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_doctor_reports_a_materializing_benchmarks_pinned_tool(capsys, fake_decoders, fake_benchmark_module, synth_clf, tmp_path):
    """doctor reports a materializing benchmark's pinned tool versions (e.g. ffmpeg 4.4.2 + a libx264 build) against
    the host's. Every pin is declared in the yaml (``materialize.tool.pins``: value, probe command, regex); doctor reads no script."""
    bench = fake_benchmark_module["synth_clf"]
    rc, env = run_json(capsys, ["doctor", "--bench", "synth_clf"])
    assert env["data"]["tools"] == {}  # nothing to materialize → nothing checked
    x264_pin = {"libx264": {"value": "x264 - core 163 r3060 5db6aa6", "match": r"x264 - core \d+ r\d+ [0-9a-f]+",
                            "probe": ["{tool}", "-f", "lavfi", "-c:v", "libx264", "-f", "h264", "-"]}}
    wrong = _fake_ffmpeg(tmp_path / "ffmpeg-wrong", "6.1.1-3ubuntu5", "x264 - core 164 r3108 31e19f9")
    bench.spec["input_contract"]["materialize"] = {"tool": {"name": str(wrong), "version_pin": "4.4.2-0ubuntu0.22.04.1", "pins": x264_pin}}
    rc, env = run_json(capsys, ["doctor", "--bench", "synth_clf"])
    t = env["data"]["tools"][str(wrong)]
    assert rc == 0 and t["ok"] is False and t["version"] == "6.1.1-3ubuntu5"
    assert t["pins"]["libx264"] == {"found": "x264 - core 164 r3108 31e19f9", "pin": "x264 - core 163 r3060 5db6aa6", "ok": False}
    assert "!= pinned 4.4.2" in t["problem"] and "libx264" in t["problem"]
    w = [w for w in env["warnings"] if w["code"] == "W_TOOL_PIN"]
    assert len(w) == 1 and "compute node" in w[0]["message"]
    right = _fake_ffmpeg(tmp_path / "ffmpeg-right", "4.4.2-0ubuntu0.22.04.1", "x264 - core 163 r3060 5db6aa6")
    bench.spec["input_contract"]["materialize"] = {"tool": {"name": str(right), "version_pin": "4.4.2-0ubuntu0.22.04.1", "pins": x264_pin}}
    rc, env = run_json(capsys, ["doctor", "--bench", "synth_clf"])
    assert env["data"]["tools"][str(right)]["ok"] is True and not any(w["code"] == "W_TOOL_PIN" for w in env["warnings"])
    rc = cli.main(["doctor", "--bench", "synth_clf"])
    assert "4.4.2-0ubuntu0.22.04.1 (pin 4.4.2-0ubuntu0.22.04.1)  libx264 x264 - core 163" in capsys.readouterr().out
    bench.spec["input_contract"]["materialize"] = {"tool": {"name": str(tmp_path / "no-ffmpeg"), "version_pin": "4.4.2"}}
    rc, env = run_json(capsys, ["doctor", "--bench", "synth_clf"])
    t = env["data"]["tools"][str(tmp_path / "no-ffmpeg")]
    assert t["ok"] is False and "not on PATH" in t["problem"]
