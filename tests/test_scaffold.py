"""Scaffold-level guarantees: every documented module imports, the CLI tree is complete (docs/cli.md), and the
synthetic fixture is a valid benchmark folder (docs/benchmark-folder.md)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys

import pytest
import yaml

from yavih import cli
from yavih.metrics import registry as R

REQUIRED_TOP_LEVEL_KEYS = {  # benchmark.yaml schema 2; there is no top-level `requires` (it lives in output_contract)
    "schema_version", "name", "aliases", "title", "readout", "protocol_version", "changelog", "source", "budget",
    "manifest", "splits", "streams", "targets", "data", "input_contract", "output_contract", "external", "protocol",
    "metrics", "readout_params",
}


def test_yavih_reexports():
    import yavih

    # the core modules
    for name in ["Benchmark", "BenchmarkView", "Sample", "RawStream", "Bridge", "FeatureWriter", "FeatureReader",
                 "validate_features", "NotApplicable", "MediaUndecodable", "grid_pos", "__version__"]:
        assert hasattr(yavih, name), name
    assert yavih.NotApplicable("x").reason == "x"
    e = yavih.MediaUndecodable(["c1", "c2"], "corrupt")
    assert list(e.clip_ids) == ["c1", "c2"] and "corrupt" in str(e)


def test_core_helpers_and_unknown_benchmark_fails():
    """``plan_shards`` and ``grid_pos`` on small inputs; an unknown benchmark ref is a refusal."""
    import yavih
    from yavih.io.shards import plan_shards

    assert plan_shards(10, 2, 3) == [(0, 2), (2, 6), (6, 10)]  # b_lo = (i*5)//3 → batches 0,1,3
    assert yavih.grid_pos(1, 1, 1, 0).shape == (1, 4)
    with pytest.raises(Exception, match=r"no_such_benchmark|benchmark"):
        yavih.Benchmark.load("no_such_benchmark")  # the resolution order finds nothing


def test_raw_stream_fields_are_read_only(tmp_path):
    from yavih.sample import RawStream, _Source

    # a stored name that cannot occur inside the handle by chance: "a.wav" did whenever the 24-hex link name
    # ended in "a" (1 run in 16, depending on the session's /tmp/pytest-N path)
    f = tmp_path / "label-dog.wav"
    f.write_bytes(b"RIFF....")
    s = RawStream("audio", "audio", src=_Source(("file", str(f))), info={"duration_s": 1.0, "sample_rate": 8000})
    assert s.kind == "audio" and s.name == "audio" and s._real_path == str(f) and s.member is None
    assert os.path.realpath(s.path) == os.path.realpath(f)
    assert re.fullmatch(r"[0-9a-f]{24}\.wav", os.path.basename(s.path)) and "label-dog" not in s.path  # an opaque handle
    assert s.bytes() == b"RIFF...." and s.open().read() == b"RIFF...."
    with pytest.raises(AttributeError):
        s.path = "/elsewhere"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        s.extra_attribute = 1  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        RawStream("text", "text", src=_Source(("file", str(f))))


def test_cli_help_lists_the_full_command_tree(capsys):
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert set(sub.choices) == {"init", "doctor", "skill", "bench", "bridge", "extract", "features", "probe", "report", "plan", "exec", "config", "suite", "registry"}

    def subs(name):
        p = sub.choices[name]
        act = next(a for a in p._actions if a.dest == "subcommand")
        return set(act.choices)

    assert subs("skill") == {"install", "show"}
    # files / show / prepare and bridge new: one command surface for shipped files
    assert subs("bench") == {"list", "info", "path", "fetch", "verify", "lint", "new", "files", "show", "prepare", "trial", "conformance", "hash"}
    assert subs("bridge") == {"check", "new"}
    assert subs("features") == {"validate", "inspect", "diff", "stage"}
    assert subs("probe") == {"run"}
    assert subs("report") == {"build"}
    assert subs("config") == {"show", "get"}
    assert subs("plan") == {"init", "build", "status", "calibrate", "run"}  # run --local
    assert subs("suite") == {"list", "show", "new", "missing"}
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for cmd in ["doctor", "bench", "features", "probe", "report", "plan", "exec"]:
        assert cmd in out


def test_cli_module_help_in_subprocess():
    proc = subprocess.run([sys.executable, "-m", "yavih.cli", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "yavih" in proc.stdout and "features" in proc.stdout


def test_cli_json_envelope_on_error_and_exit_codes(capsys, tmp_path):
    # A failing command returns the --json envelope with status error; a missing plan file is a usage /
    # configuration error (exit 1), printed as exactly one JSON object.
    rc = cli.main(["--json", "plan", "status", str(tmp_path / "no_such_plan.jsonl")])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == cli.EXIT_USAGE and len(out) == 1
    env = json.loads(out[0])
    assert set(env) == {"status", "data", "errors", "warnings", "paths"}
    assert env["status"] == "error" and env["errors"][0]["code"].startswith("E_")
    assert env["errors"][0]["message"]
    # an unexpected internal failure maps to exit 5 with the same envelope
    rc = cli.main(["--json", "plan", "status", str(tmp_path)])  # a directory, not a plan file
    out = capsys.readouterr().out.strip().splitlines()
    assert rc in (cli.EXIT_USAGE, cli.EXIT_INTERNAL) and len(out) == 1 and json.loads(out[0])["status"] == "error"
    # usage errors exit 1 (argparse)
    assert cli.main(["extract"]) == cli.EXIT_USAGE
    assert cli.main(["nope"]) == cli.EXIT_USAGE
    # bad --shard syntax is a usage error from the CLI itself
    rc = cli.main(["--json", "extract", "--bench", "b", "--bridge", "x.py", "--split", "train", "--shard", "3of4", "--out", "o"])
    assert rc == cli.EXIT_USAGE


def test_yavih_json_env_var(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("YAVIH_JSON", "1")
    rc = cli.main(["plan", "status", str(tmp_path / "no_such_plan.jsonl")])
    out = capsys.readouterr().out.strip()
    assert rc == cli.EXIT_USAGE and json.loads(out)["status"] == "error"


def test_exit_code_mapping():
    from yavih.benchmark import DataNotReady, ManifestError, SchemaError
    from yavih.config import ConfigError
    from yavih.extract import ExtractError
    from yavih.io.validate import Finding, ValidationError
    from yavih.media import DecoderMissing

    assert cli.exit_code_for(SchemaError("x"))[0] == 1
    assert cli.exit_code_for(ConfigError("x"))[0] == 1
    assert cli.exit_code_for(ManifestError("x"))[0] == 2
    assert cli.exit_code_for(ValidationError([Finding("E_ROWS", "m")]))[0] == 2
    assert cli.exit_code_for(DataNotReady("x"))[0] == 3
    assert cli.exit_code_for(DecoderMissing("av", "pip install av"))[0] == 3
    assert cli.exit_code_for(ExtractError("E_CONFIG_DRIFT", "m", exit_code=2)) == (2, "E_CONFIG_DRIFT")
    assert cli.exit_code_for(NotImplementedError("stub"))[0] == 5
    assert cli.exit_code_for(RuntimeError("boom"))[0] == 5


def test_validation_codes_match_the_documented_code_table():
    from yavih.io.validate import CODES

    expected = {  # the validation code table (docs/validation.md)
        "E_FORMAT", "E_NA_AND_SHARDS", "E_SHARD_SET", "W_STALE_TMP", "E_RANGE_TILING", "E_HASH_MANIFEST", "E_HASH_INPUT",
        "E_DESCRIBE_MISMATCH", "E_ARGS_UNCONSUMED", "E_TAR_SHA", "E_MEMBERS", "E_ROWS", "E_TIMES_OUT_OF_MEDIA", "E_RULE_MISMATCH",
        "E_FINITE", "E_POS_SHAPE", "E_POS_RANGE", "E_BATCH_INVARIANCE", "E_DEAD_ROWS", "E_DUPLICATE_CLIPS",
        # validation waivers (yavih.waivers)
        "W_WAIVED", "E_WAIVE_INVALID",
        "N_ELIGIBILITY", "N_CANARY_DRIFT", "W_LOOSE_TOLERANCE", "W_RULE_UNCHECKED", "W_COVERAGE", "W_MEDIA_FP", "W_FP16_RANGE",
        "W_CONSTANT_CLIP", "W_SIDE_EFFECT", "W_SIZE", "W_WEIGHTS_UNFINGERPRINTED",
        # a declared custom rule (the bridge's opt-out of the rule vocabulary) is a note, never a warning
        "N_RULE_CUSTOM",
        # a split whose shards were written by different bridge files
        "W_SOURCE_SHA",
    }
    assert CODES["N_RULE_CUSTOM"] == "note"  # opting out is legitimate, never penalized
    assert set(CODES) == expected
    assert all((c.startswith("E_") and lvl == "error") or (c.startswith("W_") and lvl == "warning") or (c.startswith("N_") and lvl == "note")
               for c, lvl in CODES.items())


# ----------------------------------------------------------------------------- the synthetic fixture itself


def test_synthetic_fixture_is_a_valid_benchmark_folder(synthetic_benchmark):
    sb = synthetic_benchmark
    spec = yaml.safe_load(sb.yaml_path.read_text())
    assert set(spec) == REQUIRED_TOP_LEVEL_KEYS
    assert spec["schema_version"] == 2 and spec["name"] == sb.folder.name
    assert set(spec["input_contract"]["handed_over"]) == set(spec["streams"]) == set(spec["input_contract"]["info"])
    with gzip.open(sb.folder / "manifest.csv.gz", "rb") as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == spec["manifest"]["sha256"]
    assert hashlib.sha256((sb.folder / "classes.csv").read_bytes()).hexdigest() == spec["manifest"]["classes_sha256"]
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows) == 18 and len({r["clip_id"] for r in rows}) == 18
    assert {r["split"] for r in rows} == {"train", "val"}
    classes = list(csv.DictReader(io.StringIO(sb.classes_bytes.decode())))
    assert [int(c["index"]) for c in classes] == list(range(3)) and len({c["name"] for c in classes}) == 3
    names = {c["name"] for c in classes}
    assert all(r["label"] in names for r in rows)  # every label cell parses against classes.csv
    # every split value assigned exactly once across roles
    roles = spec["splits"]
    assigned = list(roles["train"]) + list(roles["eval"]) + list(roles["floor"]) + list(roles["unused"]) + list(roles["folds"] or [])
    assert sorted(assigned) == sorted({r["split"] for r in rows})
    # media column declared, stream references resolve, metrics resolve in the registry
    assert spec["streams"]["video"]["media_column"] in spec["manifest"]["columns"]
    R.resolve_names(spec["metrics"]["list"], spec["readout"], kinds=["metric", "disclosure"])
    R.resolve_names(spec["metrics"]["floors"], spec["readout"], kinds=["floor"])
    assert spec["metrics"]["headline"] in spec["metrics"]["list"]
    assert all(v in R.CI_KINDS for v in spec["metrics"]["ci"].values())
    # curation invariant
    cur = spec["manifest"]["upstream"]["curation"]
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"]
    assert all((sb.media_root / r["rel_media"]).exists() for r in rows)
