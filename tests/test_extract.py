"""``run_extract``, the byte read-ahead, the ``used``
contract, ``MediaUndecodable``, the encode audit hook (``E_NETWORK`` / ``W_SIDE_EFFECT``), resumption (keyed on
``model``, ``reading`` and ``describe_sha256``), canaries, ``E_TRAIN_MODE`` / ``E_PRECISION_UNDECLARED`` /
``E_CONFIG_DRIFT``, ``NotApplicable`` →
``NOT_APPLICABLE.json``, SIGTERM/deadline → exit 4 with nothing partial, ``python -m yavih.extract`` and
``bridge check``'s hard rerun / batch / order invariance checks."""

from __future__ import annotations

import importlib.util
import io
import json
import signal
import tarfile
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from test_extract_fixtures import (
    INPUT_HASH,
    MANIFEST_HASH,
    PROTOCOL_HASH,
    AVBridge,
    BadPrecisionBridge,
    BadUsedBridge,
    BatchStatsBridge,
    ClipIdCacheBridge,
    ConstantPosBridge,
    CrashingBridge,
    DropoutBridge,
    FakeBenchmark,
    NetworkBridge,
    NoPosBridge,
    NoPreprocBridge,
    NotApplicableEncodeBridge,
    NotApplicableInitBridge,
    NumpyBridge,
    OrderBridge,
    SigtermBridge,
    TwoTupleBridge,
    UndecodableBridge,
    WritingBridge,
    WrongShapeBridge,
    WrongTimesBridge,
    clip_features,
    describe_of,
    frame_times,
    view_of,
)
from yavih import extract, hashing
from yavih.bridge import ArgError, DescribeError
from yavih.extract import (
    E_BATCH_INVARIANCE,
    E_BRIDGE,
    E_CONFIG_DRIFT,
    E_DEADLINE,
    E_FEATURES,
    E_NETWORK,
    E_ORACLE,
    E_PRECISION_UNDECLARED,
    E_PREEMPTED,
    E_PROFILE,
    E_RERUN_DRIFT,
    E_TRAIN_MODE,
    E_USED,
    W_NO_PREPROCESSING_SOURCE,
    ExtractError,
    bridge_check,
    default_decode_workers,
    load_bridge_class,
    parse_deadline,
    run_extract,
    scramble,
)
from yavih.io.format import NOT_APPLICABLE_FILE, STALE_DIR
from yavih.io.sidecar import read_sidecar
from yavih.io.validate import validate_features

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("YAVIH_DEADLINE", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("YAVIH_TESTING", raising=False)


def _run(tmp_path: Path, bridge=NumpyBridge, bench=None, split="train", shard=(0, 1), **kw):
    bench = bench or FakeBenchmark()
    kw.setdefault("decode_workers", 2)
    return run_extract(bench=bench, bridge=bridge, split=split, shard=shard, out=tmp_path / "run", **kw), bench


def _split_dir(tmp_path: Path, split="train") -> Path:
    return tmp_path / "run" / "features" / split


def _listing(d: Path) -> list[str]:
    """Files under ``d``; the split's frozen ``_layout.json`` (written by the first unit, kept on failure: it is the
    shard plan, not a partial shard) is left out."""
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file() and p.name != "_layout.json") if d.exists() else []


def _member(tmp_path: Path, name: str, split="train"):
    with tarfile.open(_split_dir(tmp_path, split) / "shard-000000.tar") as tf:
        data = tf.extractfile(name).read()
    return json.loads(data) if name.endswith(".json") else np.load(io.BytesIO(data), allow_pickle=False)


# ----------------------------------------------------------------------------- helpers


def test_default_decode_workers_rules():
    assert default_decode_workers({"SLURM_CPUS_PER_TASK": "8"}) == 7
    assert default_decode_workers({"SLURM_CPUS_PER_TASK": "1"}) == 1
    assert default_decode_workers({"SLURM_CPUS_PER_TASK": "64"}) == 16
    n = default_decode_workers({})
    from yavih.resources import available_cpus

    assert 1 <= n <= 16 and n == max(1, min(16, available_cpus({}) // 2))


def test_parse_deadline_forms():
    now = time.time()
    assert parse_deadline(None, {}) is None
    assert parse_deadline("none", {}) is None
    assert abs(parse_deadline("+10", {}) - (now + 10)) < 2
    assert parse_deadline("1700000000", {}) == 1700000000.0
    assert parse_deadline(None, {"YAVIH_DEADLINE": "1700000000"}) == 1700000000.0
    assert parse_deadline("slurm", {"SLURM_JOB_END_TIME": "2000000000"}) == 2000000000.0 - 900.0  # > the 600 s signal lead
    assert parse_deadline("slurm", {}) is None  # outside SLURM: no deadline
    assert abs(parse_deadline("+90m", {}) - (now + 5400)) < 2
    from yavih.deadline import DeadlineError

    with pytest.raises(DeadlineError) as ei:  # inside a job without END_TIME: exit 1, never "now"
        parse_deadline("slurm", {"SLURM_JOB_ID": "1"})
    assert ei.value.exit_code == 1
    with pytest.raises(DeadlineError):
        parse_deadline("tomorrow", {})


def test_load_bridge_class_from_file(tmp_path: Path):
    src = textwrap.dedent(
        """
        import numpy as np
        from yavih.bridge import Bridge
        class Mine(Bridge):
            name = "mine"
            def describe(self): return {}
            def encode(self, samples): return np.zeros((len(samples), 1, 1)), None, [{} for _ in samples]
        """
    )
    f = tmp_path / "mine.py"
    f.write_text(src)
    cls = load_bridge_class(str(f))
    assert cls.__name__ == "Mine" and cls.name == "mine"
    assert extract.source_sha256(f) == hashing.sha256_hex(src.encode())
    (tmp_path / "two.py").write_text(src + "\nclass Other(Mine):\n    name = 'other'\n")
    with pytest.raises(ArgError):
        load_bridge_class(str(tmp_path / "two.py"))
    (tmp_path / "picked.py").write_text(src + "\nclass Other(Mine):\n    name = 'other'\nBRIDGE = Other\n")
    assert load_bridge_class(str(tmp_path / "picked.py")).name == "other"
    (tmp_path / "none.py").write_text("x = 1\n")
    with pytest.raises(ArgError):
        load_bridge_class(str(tmp_path / "none.py"))
    (tmp_path / "broken.py").write_text("raise RuntimeError('import boom')\n")
    with pytest.raises(ArgError):
        load_bridge_class(str(tmp_path / "broken.py"))
    with pytest.raises(ArgError):
        load_bridge_class("no.such.module:X")


def test_scramble_replaces_identity_and_hands_over_new_streams_over_the_same_bytes():
    s = FakeBenchmark()._sample("val", 3)
    t = scramble(s, 5)
    assert (t.row, t.clip_id, t.split) == (10**9 + 5, "bridge-check-5", "bridge-check")
    assert t.meta == s.meta and set(t.streams) == set(s.streams)
    assert t.streams["video"] is not s.streams["video"] and t.streams["video"].bytes() == s.streams["video"].bytes()


# ----------------------------------------------------------------------------- happy path


def test_run_extract_writes_shard_sidecar_and_canaries(tmp_path: Path):
    summary, bench = _run(tmp_path, canaries=8)
    assert summary["status"] == "ok"
    sd = _split_dir(tmp_path)
    assert (sd / "shard-000000.tar").exists() and (sd / "shard-000000.json").exists()
    assert not list(sd.glob("*.tmp"))
    sc = read_sidecar(sd / "shard-000000.json")
    assert sc["shard"] == {"index": 0, "n": 1, "rows": [0, 20], "batch_size": 4}
    assert sc["rows"]["n_range"] == 20 and sc["rows"]["n_written"] == 20 and sc["rows"]["n_missing"] == 0
    assert sc["rows"]["n_canary"] == min(8, 20 // 4) == 5
    assert sc["canary"]["max_rel_l2"] == 0.0 and sc["canary"]["max_abs_diff"] == 0.0 and sc["canary"]["n"] == 5
    assert sc["canary"]["tolerance"] == 1e-5 and "mismatch" not in sc["canary"]
    assert sc["features"] == {**sc["features"], "dtype": "float32", "N": 4, "D": 6, "pos_mode": "per_sample"}
    assert sc["benchmark"]["manifest_hash"] == MANIFEST_HASH and sc["benchmark"]["protocol_hash"] == PROTOCOL_HASH
    assert sc["benchmark"]["input_hash"] == INPUT_HASH and sc["yavih"]["version"] == sc["yavih_version"]
    # the declared names; describe_sha256 is the consistency digest
    assert sc["format_version"] == 4 and sc["model"] == "np-test" and sc["reading"] == "v"
    assert list(sc)[:6] == ["format_version", "yavih_version", "benchmark", "split", "model", "reading"]
    assert sc["bridge"]["describe_sha256"] == hashing.describe_sha256(sc["bridge"]["describe"])
    assert sc["bridge"]["name"] == "np-test"  # a recorded label
    assert len(sc["bridge"]["source_sha256"]) == 64
    assert sc["bridge"]["args"] == {"layer": 12, "scale": 1.0, "flag": False, "mode": "a"}
    assert "decoders" not in sc or sc["decoders"] == {}  # decoders live in describe()["input"]
    assert sc["env"]["decode_workers"] == 2 and sc["env"]["read_ahead"] == 2 and sc["env"]["host"]
    assert sc["timing"]["wall_s"] >= 0 and sc["timing"]["started"].endswith("Z")
    assert sc["timing"]["encode_s"] >= 0 and sc["timing"]["read_s"] >= 0 and "decode_calls" not in sc["timing"]
    assert len(bench.reader_calls) == 1 and bench.reader_calls[0]["canaries"] == 8


def test_members_carry_media_used_and_manifest_media_key(tmp_path: Path):
    summary, bench = _run(tmp_path, canaries=0, bench=FakeBenchmark(shared_media={4: 3}))
    rec = _member(tmp_path, "000000003.json")
    assert rec["row"] == 3 and rec["clip_id"] == "train_0003" and rec["media_key"] == "train/0003.mp4"
    assert rec["media"]["video"]["duration_s"] == 2.0 and rec["media"]["video"]["fps"] == 25.0 and rec["media"]["video"]["width"] == 320
    assert rec["used"] == {"video": {"times": [0.24, 0.76, 1.24, 1.76]}}
    assert "frame_times" not in rec
    assert _member(tmp_path, "000000004.json")["media_key"] == "train/0003.mp4"  # two rows, one media file
    feat = _member(tmp_path, "000000003.feat.npy")
    s = bench._sample("train", 3)
    np.testing.assert_array_equal(feat, clip_features(s.streams["video"].bytes(), 4, 6))
    pos = _member(tmp_path, "000000003.pos.npy")
    np.testing.assert_allclose(pos[:, 1], np.asarray(frame_times(s.streams["video"].info, 4)) / 2.0, rtol=1e-6)
    assert (pos[:, 3] >= 0.125 - 1e-6).all() and (pos[:, 3] <= 0.875 + 1e-6).all()  # the 4:3 frame's centre crop


def test_bridge_sees_media_only_and_gets_its_cpu_budget(tmp_path: Path):
    seen = {}

    class Spy(NumpyBridge):
        name = "np-spy"

        def encode(self, samples):
            seen["bench"] = self.bench
            seen["workers"] = self.decode_workers
            seen["meta"] = [s.meta for s in samples]
            seen["canary"] = [getattr(s, "_canary_of", None) for s in samples]
            return super().encode(samples)

    _run(tmp_path, bridge=Spy, canaries=0, decode_workers=3)
    from yavih.bridge import BenchmarkView

    assert isinstance(seen["bench"], BenchmarkView) and not hasattr(seen["bench"], "manifest")
    assert seen["workers"] == 3
    assert all(m == {"facts": {}} for m in seen["meta"])


def test_sharding_tiles_the_split_and_batches_align(tmp_path: Path):
    bench = FakeBenchmark(n={"train": 22, "val": 8})
    for i in range(3):
        _run(tmp_path, bench=bench, shard=(i, 3), canaries=0)
    sd = _split_dir(tmp_path)
    scs = [read_sidecar(sd / f"shard-{i:06d}.json") for i in range(3)]
    ranges = [tuple(sc["shard"]["rows"]) for sc in scs]
    assert ranges[0][0] == 0 and ranges[-1][1] == 22
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:], strict=False))
    assert all(lo % 4 == 0 for lo, _ in ranges)
    assert sum(sc["rows"]["n_written"] for sc in scs) == 22


def test_empty_shard_writes_sidecar_without_members(tmp_path: Path):
    bench = FakeBenchmark(n={"train": 4, "val": 8})  # 1 batch, 3 shards → the first two are empty (row dealing)
    summary, _ = _run(tmp_path, bench=bench, shard=(0, 3), canaries=8)
    assert summary["status"] == "ok"
    sc = read_sidecar(_split_dir(tmp_path) / "shard-000000.json")
    assert sc["shard"]["rows"][0] == sc["shard"]["rows"][1] and sc["rows"]["n_written"] == 0 and sc["members"] == []


def test_constant_pos_and_pos_none_float16(tmp_path: Path):
    _run(tmp_path, bridge=ConstantPosBridge, canaries=0)
    with tarfile.open(_split_dir(tmp_path) / "shard-000000.tar") as tf:
        assert "_pos.npy" in tf.getnames() and "000000000.pos.npy" not in tf.getnames()
    _run(tmp_path, bridge=NoPosBridge, split="val", canaries=0)
    sc = read_sidecar(_split_dir(tmp_path, "val") / "shard-000000.json")
    assert sc["features"]["pos_mode"] == "none" and sc["features"]["dtype"] == "float16" and sc["features"]["pos_sha256"] is None


def test_av_bridge_reports_audio_ranges(tmp_path: Path):
    bench = FakeBenchmark(streams={"video": "video", "audio": "audio"}, infos={2: {"audio": {"n_samples": 30000, "duration_s": 1.875}}})
    _run(tmp_path, bridge=AVBridge, bench=bench, canaries=0)
    rec = _member(tmp_path, "000000002.json")
    assert rec["used"]["audio"] == {"samples": [[0, 30000]], "pad": [0, 2000]}
    assert rec["media"]["audio"]["n_samples"] == 30000


def test_validate_features_accepts_the_extraction(tmp_path: Path):
    """Cross-module check: the validator over a real extraction, with finalize writing ``_meta.json``."""
    from yavih.io.finalize import read_meta
    from yavih.io.validate import validate_features

    bench = FakeBenchmark()
    _run(tmp_path, bench=bench, shard=(0, 2), canaries=8)
    _run(tmp_path, bench=bench, shard=(1, 2), canaries=8)
    report = validate_features(_split_dir(tmp_path), bench=bench)
    assert report.ok, [f.to_json() for f in report.errors]
    assert report.rules_checked == {"video": {"temporal": "uniform", "spatial": True}}
    meta = read_meta(_split_dir(tmp_path))
    assert meta["coverage"]["present"] == 20 and meta["coverage"]["nominal"] == 20 and meta["validation"]["ok"] is True
    assert meta["canary"]["n"] > 0 and meta["canary"]["tolerance"] == 1e-5 and "decoders" not in meta


# ----------------------------------------------------------------------------- missing rows


def test_missing_rows_recorded_and_bridge_never_called_for_them(tmp_path: Path):
    bench = FakeBenchmark(missing_rows={2, 7})
    summary, _ = _run(tmp_path, bench=bench, canaries=0)
    sc = read_sidecar(_split_dir(tmp_path) / "shard-000000.json")
    assert sc["rows"]["n_missing"] == 2 and sc["rows"]["n_written"] == 18
    assert [m["row"] for m in sc["missing"]] == [2, 7]
    assert sc["missing"][0]["clip_id"] == "train_0002" and sc["missing"][0]["reason"].startswith("MediaMissing: video")
    with tarfile.open(_split_dir(tmp_path) / "shard-000000.tar") as tf:
        assert "000000002.feat.npy" not in tf.getnames()


def test_media_undecodable_becomes_missing_and_the_batch_is_re_encoded(tmp_path: Path):
    calls = []

    class B(UndecodableBridge):
        bad_rows = frozenset({3, 9})

        def encode(self, samples):
            calls.append([s.row for s in samples])
            return super().encode(samples)

    summary, _ = _run(tmp_path, bridge=B, canaries=0)
    sc = read_sidecar(_split_dir(tmp_path) / "shard-000000.json")
    assert [m["row"] for m in sc["missing"]] == [3, 9] and "moov atom" in sc["missing"][0]["reason"]
    assert sc["rows"]["n_written"] == 18
    assert [0, 1, 2, 3] in calls and [0, 1, 2] in calls  # the batch without the undecodable row
    # an exception naming a clip outside the batch is a bridge bug
    from yavih.bridge import MediaUndecodable

    class Liar(NumpyBridge):
        name = "np-liar"

        def encode(self, samples):
            raise MediaUndecodable(["not-in-batch"], "?")

    with pytest.raises(ExtractError) as ei:
        _run(tmp_path / "liar", bridge=Liar, canaries=0)
    assert ei.value.exit_code == 5


def test_bridge_exception_aborts_the_shard_exit_5(tmp_path: Path):
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=CrashingBridge, canaries=0)
    assert ei.value.exit_code == 5 and ei.value.code == E_BRIDGE and "boom" in str(ei.value)
    assert _listing(_split_dir(tmp_path)) == []  # nothing partial: tmp removed, no tar, no sidecar


def test_decoder_missing_is_exit_3(tmp_path: Path):
    from yavih.cli import exit_code_for
    from yavih.media import DecoderMissing

    bench = FakeBenchmark(fail_rows={1: DecoderMissing("av")})
    with pytest.raises(DecoderMissing) as ei:
        _run(tmp_path, bench=bench, canaries=0)
    assert exit_code_for(ei.value)[0] == 3
    assert _listing(_split_dir(tmp_path)) == []


# ----------------------------------------------------------------------------- refusals (exit 2)


@pytest.mark.parametrize("variant", ["missing", "short", "illformed"])
def test_e_used(tmp_path: Path, variant):
    class B(BadUsedBridge):
        pass

    B.variant = variant
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=B, canaries=0)
    assert ei.value.code == E_USED and ei.value.exit_code == 2
    assert _listing(_split_dir(tmp_path)) == []


def test_two_tuple_encode_is_e_used(tmp_path: Path):
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=TwoTupleBridge, canaries=0)
    assert ei.value.code == E_USED and "(feats, pos, used)" in str(ei.value)


def test_views_other_than_one_are_refused(tmp_path: Path):
    class Views(NumpyBridge):
        name = "np-views"

        def describe(self):
            return dict(super().describe(), views=3)

    with pytest.raises(DescribeError) as ei:
        _run(tmp_path, bridge=Views, canaries=0)
    assert ei.value.exit_code == 2
    assert not _split_dir(tmp_path).exists() or _listing(_split_dir(tmp_path)) == []


def test_e_precision_undeclared(tmp_path: Path):
    with pytest.raises(DescribeError) as ei:
        _run(tmp_path, bridge=BadPrecisionBridge, canaries=0)
    assert ei.value.code == E_PRECISION_UNDECLARED and ei.value.exit_code == 2


class _Unfingerprinted(NumpyBridge):
    name = "np-weak"

    def describe(self):
        d = super().describe()
        d["weights_sha256"] = None
        return d


def test_null_weights_fingerprint_is_recorded_not_gated(tmp_path: Path, capsys: pytest.CaptureFixture):
    """A null weights_sha256 extracts (nothing is gated) and is recorded as a warning."""
    summary, bench = _run(tmp_path, bridge=_Unfingerprinted, canaries=0)
    assert summary["status"] == "ok" and summary["model"] == "np-test" and summary["reading"] == "v"
    assert summary["bridge"]["describe"]["weights_sha256"] is None
    assert "W_WEIGHTS_UNFINGERPRINTED" in capsys.readouterr().err
    rep = validate_features(_split_dir(tmp_path), bench=bench, write_meta=False)
    assert rep.ok and [w.code for w in rep.warnings].count("W_WEIGHTS_UNFINGERPRINTED") == 1


def test_wrong_feature_shape_is_exit_2(tmp_path: Path):
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=WrongShapeBridge, canaries=0)
    assert ei.value.code == E_FEATURES and ei.value.exit_code == 2


def test_unknown_arg_is_exit_1(tmp_path: Path):
    with pytest.raises(ArgError) as ei:
        _run(tmp_path, args={"bogus": "1"})
    assert ei.value.exit_code == 1


# ----------------------------------------------------------------------------- audit hook


def test_network_inside_encode_is_e_network(tmp_path: Path):
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=NetworkBridge, canaries=0)
    assert ei.value.code == E_NETWORK and ei.value.exit_code == 2 and "127.0.0.1" in str(ei.value)
    assert _listing(_split_dir(tmp_path)) == []
    # outside encode the hook is disarmed: the same connect attempt is not recorded
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.01)
    try:
        s.connect(("127.0.0.1", 9))
    except OSError:
        pass
    finally:
        s.close()
    assert extract._Audit.armed == 0


def test_writes_outside_the_run_dir_are_w_side_effect(tmp_path: Path, capsys):
    target = tmp_path / "elsewhere.cache"

    class B(WritingBridge):
        pass

    B.target = str(target)
    summary, _ = _run(tmp_path, bridge=B, canaries=0)
    assert summary["status"] == "ok" and target.exists()
    assert "W_SIDE_EFFECT" in capsys.readouterr().err
    se = summary.get("side_effects")
    if se is not None:  # the writer records it in the sidecar
        assert se["writes_outside_run_dir"] == [str(target)]
    # inside the run directory a write is not a side effect
    inside = tmp_path / "run" / "scratch.bin"
    inside.parent.mkdir(parents=True, exist_ok=True)

    class C(WritingBridge):
        name = "np-writer-inside"

    C.target = str(inside)
    capsys.readouterr()
    _run(tmp_path, bridge=C, split="val", canaries=0)
    assert "W_SIDE_EFFECT" not in capsys.readouterr().err


# ----------------------------------------------------------------------------- canaries


def test_canaries_record_used_and_pos_mismatch(tmp_path: Path):
    class Shifty(NumpyBridge):
        """Reports different times when the batch holds a single sample (a batch-dependent frame choice)."""

        name = "np-shifty"

        def encode(self, samples):
            feats, pos, used = super().encode(samples)
            if len(samples) == 1:
                used = [{"video": {"times": [t + 0.04 for t in u["video"]["times"]]}} for u in used]
            return feats, pos, used

    bench = FakeBenchmark(n={"train": 20, "val": 8})
    summary, _ = _run(tmp_path, bridge=Shifty, bench=bench, canaries=1)
    assert summary["canary"]["n"] == 1 and summary["canary"]["mismatch"]


# ----------------------------------------------------------------------------- E_TRAIN_MODE / precision (torch)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not importable")
def test_e_train_mode_names_the_module_and_eval_passes(tmp_path: Path):
    import torch

    class TorchBridge(NumpyBridge):
        name = "torch-test"

        def __init__(self, bench, args=None, *, eval_mode=False):
            super().__init__(bench, args)
            self.model = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Dropout(0.5))
            self.parts = {"aux": [torch.nn.Linear(2, 2)]}
            if eval_mode:
                self.model.eval()
                self.parts["aux"][0].eval()

        def encode(self, samples):
            feats, pos, used = super().encode(samples)
            assert torch.is_inference_mode_enabled()
            return self.model(torch.from_numpy(feats)), pos, used

    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=TorchBridge, canaries=0)
    assert ei.value.code == E_TRAIN_MODE and ei.value.exit_code == 2
    msg = str(ei.value)
    assert "self.model" in msg and "self.parts['aux'][0]" in msg
    assert _listing(_split_dir(tmp_path)) == []

    class EvalBridge(TorchBridge):
        name = "torch-eval"

        def __init__(self, bench, args=None):
            super().__init__(bench, args, eval_mode=True)

    summary, _ = _run(tmp_path, bridge=EvalBridge, canaries=4)
    env = summary["env"]
    assert env["torch"] == torch.__version__
    assert env["precision_live"] == {"matmul_precision": "highest", "allow_tf32": False, "deterministic": True}
    assert summary["canary"]["max_rel_l2"] == 0.0  # eval-mode dropout is the identity


@pytest.mark.skipif(not HAS_TORCH, reason="torch not importable")
def test_precision_enforced_from_describe(tmp_path: Path):
    import torch

    class HighPrec(NumpyBridge):
        name = "np-highprec"

        def describe(self):
            d = super().describe()
            d["precision"] = {"autocast": "bf16", "matmul_precision": "high", "allow_tf32": True, "deterministic": False}
            return d

    try:
        summary, _ = _run(tmp_path, bridge=HighPrec, canaries=0)
        assert summary["env"]["precision_live"] == {"matmul_precision": "high", "allow_tf32": True, "deterministic": False}
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not importable")
def test_e_train_mode_sees_class_level_and_module_level_models(tmp_path: Path):
    import torch

    class ClassModel(NumpyBridge):
        name = "torch-class-level"
        MODEL = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Dropout(0.5))  # training mode by construction

    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=ClassModel, canaries=0)
    assert ei.value.code == E_TRAIN_MODE and "ClassModel.MODEL" in str(ei.value)
    ClassModel.MODEL.eval()
    assert _run(tmp_path, bridge=ClassModel, canaries=0)[0]["status"] == "ok"
    globals()["_MODULE_LEVEL_MODEL"] = torch.nn.Linear(2, 2)
    try:
        with pytest.raises(ExtractError) as ei:
            _run(tmp_path / "m", bridge=ClassModel, canaries=0)
        assert ei.value.code == E_TRAIN_MODE and "_MODULE_LEVEL_MODEL" in str(ei.value)
    finally:
        del globals()["_MODULE_LEVEL_MODEL"]


# ----------------------------------------------------------------------------- NotApplicable


def test_not_applicable_from_init_writes_file_and_status(tmp_path: Path):
    summary, _ = _run(tmp_path, bridge=NotApplicableInitBridge)
    assert summary["status"] == "not_applicable" and "audio" in summary["reason"]
    na = json.loads((_split_dir(tmp_path) / NOT_APPLICABLE_FILE).read_text())
    assert na["status"] == "not_applicable" and na["benchmark"] == "synth_fake" and na["split"] == "train"
    # __init__ raised before describe(): no model/reading without a plan profile
    assert na["model"] is None and na["reading"] is None and na["describe_sha256"] is None
    assert na["bridge"] == {} and na["created"].endswith("Z")
    assert _listing(_split_dir(tmp_path)) == [NOT_APPLICABLE_FILE]


def test_not_applicable_from_first_encode_leaves_no_tmp(tmp_path: Path):
    summary, _ = _run(tmp_path, bridge=NotApplicableEncodeBridge)
    assert summary["status"] == "not_applicable"
    na = json.loads((_split_dir(tmp_path) / NOT_APPLICABLE_FILE).read_text())
    assert na["model"] == "np-test" and na["reading"] == "v" and na["bridge"]["model"] == "np-test"
    assert na["describe_sha256"] == hashing.describe_sha256(na["bridge"])
    assert _listing(_split_dir(tmp_path)) == [NOT_APPLICABLE_FILE]


def test_not_applicable_after_first_encode_is_a_bridge_bug(tmp_path: Path):
    from yavih.bridge import NotApplicable

    class Late(NumpyBridge):
        name = "np-late-na"

        def encode(self, samples):
            if self.encode_calls >= 1:
                raise NotApplicable("too late")
            return super().encode(samples)

    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=Late, canaries=0)
    assert ei.value.exit_code == 5 and ei.value.code == E_BRIDGE


def test_stale_not_applicable_is_moved_aside_when_shards_get_written(tmp_path: Path):
    _run(tmp_path, bridge=NotApplicableInitBridge)
    _run(tmp_path, canaries=0)
    sd = _split_dir(tmp_path)
    assert not (sd / NOT_APPLICABLE_FILE).exists()
    assert list((sd / STALE_DIR).glob(f"*/{NOT_APPLICABLE_FILE}"))


# ----------------------------------------------------------------------------- resumption


def test_second_run_is_skipped_and_touches_nothing(tmp_path: Path):
    bench = FakeBenchmark()
    first, _ = _run(tmp_path, bench=bench, canaries=4)
    sd = _split_dir(tmp_path)
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in sd.iterdir()}
    bridge_calls = 0

    class Counting(NumpyBridge):
        name = "np-test"  # same identity

        def encode(self, samples):
            nonlocal bridge_calls
            bridge_calls += 1
            return super().encode(samples)

    second, _ = _run(tmp_path, bridge=Counting, bench=bench, canaries=4)
    assert second["status"] == "skipped" and bridge_calls == 0
    assert {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in sd.iterdir()} == before
    assert (second["model"], second["reading"]) == (first["model"], first["reading"])
    assert second["bridge"]["describe_sha256"] == first["bridge"]["describe_sha256"]


def _profile_of(tmp_path: Path, bench, bridge=NumpyBridge, **over) -> Path:
    """A minimal `bridge check` profile: describe, model, reading, describe_sha256."""
    d = json.loads(hashing.canon(bridge(view_of(bench), {}).describe()))
    prof = {"describe": d, "model": d["model"], "reading": d["reading"], "describe_sha256": hashing.describe_sha256(d),
            "batch_size": d["batch_size"], "dtype": d["dtype"], "pos_mode": d["pos_mode"]}
    prof.update(over)
    path = tmp_path / f"profile-{len(list(tmp_path.glob('profile-*.json')))}.json"
    path.write_text(json.dumps(prof))
    return path


def test_fast_skip_with_expect_profile_needs_no_model(tmp_path: Path):
    bench = FakeBenchmark()
    first, _ = _run(tmp_path, bench=bench, canaries=0)

    class Exploding(NumpyBridge):
        name = "np-test"

        def __init__(self, bench, args=None):
            raise AssertionError("the model must not be loaded for a planned skip")

    second, _ = _run(tmp_path, bridge=Exploding, bench=bench, canaries=0, expect_profile=str(_profile_of(tmp_path, bench)))
    assert second["status"] == "skipped"
    assert second["bridge"]["describe_sha256"] == first["bridge"]["describe_sha256"]


def test_mismatching_sidecar_moves_to_stale_and_reruns(tmp_path: Path):
    """Another configuration under the SAME model name and reading (another describe_sha256) is stale."""
    first, _ = _run(tmp_path, canaries=0)
    second, _ = _run(tmp_path, canaries=0, args={"layer": "13"})  # configuration change, same model + reading
    assert second["status"] == "ok" and (second["model"], second["reading"]) == (first["model"], first["reading"])
    assert second["bridge"]["describe_sha256"] != first["bridge"]["describe_sha256"]
    sd = _split_dir(tmp_path)
    stale = list((sd / STALE_DIR).iterdir())
    assert len(stale) == 1
    # the earlier configuration's frozen layout goes with its shards
    assert sorted(p.name for p in stale[0].iterdir()) == ["_layout.json", "shard-000000.json", "shard-000000.tar"]
    assert read_sidecar(stale[0] / "shard-000000.json")["bridge"]["describe_sha256"] == first["bridge"]["describe_sha256"]
    assert read_sidecar(sd / "shard-000000.json")["bridge"]["describe_sha256"] == second["bridge"]["describe_sha256"]


def test_a_renamed_model_or_reading_is_stale_too(tmp_path: Path):
    """The resume match keys are model, reading AND describe_sha256 (RESUME_MATCH_KEYS)."""
    first, _ = _run(tmp_path, canaries=0)

    class Renamed(NumpyBridge):
        MODEL_NAME = "np-test-renamed"

    second, _ = _run(tmp_path, bridge=Renamed, canaries=0)
    assert second["status"] == "ok" and second["model"] == "np-test-renamed"
    assert len(list((_split_dir(tmp_path) / STALE_DIR).iterdir())) == 1


def test_a_moved_checkpoint_path_is_the_same_configuration(tmp_path: Path):
    """weights_ref is outside describe_sha256: a path edit neither re-extracts nor counts as drift."""
    bench = FakeBenchmark()
    _run(tmp_path, bench=bench, canaries=0)

    class Moved(NumpyBridge):
        def describe(self):
            return dict(super().describe(), weights_ref="/new/home/of/the/checkpoint.pt")

    second, _ = _run(tmp_path, bridge=Moved, bench=bench, canaries=0, expect_profile=str(_profile_of(tmp_path, bench)))
    assert second["status"] == "skipped"


def test_own_tmp_is_deleted_at_start(tmp_path: Path):
    sd = _split_dir(tmp_path)
    sd.mkdir(parents=True)
    (sd / "shard-000000.tar.tmp").write_bytes(b"garbage")
    (sd / "shard-000001.tar.tmp").write_bytes(b"sibling in progress")
    _run(tmp_path, shard=(0, 2), canaries=0)
    assert not (sd / "shard-000000.tar.tmp").exists()
    assert (sd / "shard-000001.tar.tmp").exists()  # another unit's tmp is not ours to delete


def test_e_config_drift_lists_the_differing_describe_keys(tmp_path: Path):
    """A live describe() whose describe_sha256 differs from the planned profile's is E_CONFIG_DRIFT
    (exit 2), naming the keys; weights_ref / batch_tolerance are never listed; nothing is written."""
    bench = FakeBenchmark()
    d = json.loads(hashing.canon(NumpyBridge(view_of(bench), {}).describe()))
    planned = json.loads(hashing.canon(d))
    planned["args"]["layer"] = 99
    planned["input"]["video"]["temporal"]["n"] = 16
    planned["weights_ref"] = "/elsewhere"  # provenance: must NOT be listed
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"describe": planned, "model": "np-test", "reading": "v", "describe_sha256": hashing.describe_sha256(planned)}))
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bench=bench, canaries=0, expect_profile=str(profile))
    assert ei.value.code == E_CONFIG_DRIFT and ei.value.exit_code == 2
    msg = str(ei.value)
    assert "['args', 'input']" in msg and "weights_ref" not in msg and "bridge check" in msg
    assert _listing(_split_dir(tmp_path)) == []  # never writes a configuration the plan did not describe
    ok, _ = _run(tmp_path, bench=bench, canaries=0, expect_profile=str(_profile_of(tmp_path, bench)))
    assert ok["status"] == "ok"


@pytest.mark.parametrize("key,value", [("model", "another-model"), ("reading", "av")])
def test_e_config_drift_on_model_or_reading(tmp_path: Path, key: str, value: str):
    """A profile planned under another model name or reading is drift even when the describe() body matches."""
    bench = FakeBenchmark()
    profile = _profile_of(tmp_path, bench, **{key: value})
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bench=bench, canaries=0, expect_profile=str(profile))
    assert ei.value.code == E_CONFIG_DRIFT and key in str(ei.value)
    assert _listing(_split_dir(tmp_path)) == []


@pytest.mark.parametrize("content", ["not json", "[]", json.dumps({"describe": {}}), json.dumps({"describe": {"x": 1}})])
def test_expect_profile_unreadable_or_incomplete_is_a_usage_error(tmp_path: Path, content: str):
    profile = tmp_path / "bad-profile.json"
    profile.write_text(content)
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, canaries=0, expect_profile=str(profile))
    assert ei.value.code == E_PROFILE and ei.value.exit_code == 1


def test_not_applicable_from_init_takes_model_and_reading_from_the_profile(tmp_path: Path):
    """With a plan profile, a NotApplicable raised before describe() still records model and reading."""
    bench = FakeBenchmark()
    profile = _profile_of(tmp_path, bench)
    summary, _ = _run(tmp_path, bridge=NotApplicableInitBridge, bench=bench, expect_profile=str(profile))
    na = json.loads((_split_dir(tmp_path) / NOT_APPLICABLE_FILE).read_text())
    assert summary["status"] == "not_applicable" and (na["model"], na["reading"]) == ("np-test", "v")


def _files(d: Path) -> list[str]:
    """Top-level files of a split directory (``_stale/`` and the frozen ``_layout.json`` excluded)."""
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.name != "_layout.json") if d.exists() else []


def test_planned_fast_path_moves_nothing_before_the_drift_check(tmp_path: Path):
    """On disk config A, profile B, live bridge A → E_CONFIG_DRIFT and shard-0 is still in
    place. The pre-load check is read-only (``peek_shard``): only after ``_prepare`` and the drift check does the moving
    resume run, so a shard the live configuration wrote never goes to _stale/ on the profile's word."""
    from yavih.io.sidecar import peek_shard

    bench = FakeBenchmark()
    first, _ = _run(tmp_path, bench=bench, canaries=0)  # config A, the live one
    sd = _split_dir(tmp_path)
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in sd.iterdir()}
    d_b = json.loads(hashing.canon(describe_of(NumpyBridge, bench, {"layer": "13"})))
    profile_b = _profile_of(tmp_path, bench, describe=d_b, describe_sha256=hashing.describe_sha256(d_b))
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bench=bench, canaries=0, expect_profile=str(profile_b))
    assert ei.value.code == E_CONFIG_DRIFT and "args" in str(ei.value)
    assert {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in sd.iterdir()} == before  # shard-0 untouched
    assert not (sd / STALE_DIR).exists()
    assert read_sidecar(sd / "shard-000000.json")["bridge"]["describe_sha256"] == first["bridge"]["describe_sha256"]
    # peek_shard itself is read-only: a mismatch says "run" and moves nothing, a tmp is left for the moving resume
    (sd / "shard-000000.tar.tmp").write_bytes(b"x")
    assert peek_shard(sd, 0, manifest_hash=bench.manifest_hash, input_hash=bench.input_hash, model="np-test", reading="v",
                      describe_sha256=hashing.describe_sha256(d_b), n=1, dtype="float32", pos_mode="per_sample") == "run"
    assert (sd / "shard-000000.tar.tmp").exists() and (sd / "shard-000000.json").exists()
    # the live configuration's own profile still skips without loading the model
    again, _ = _run(tmp_path, bench=bench, canaries=0, expect_profile=str(_profile_of(tmp_path, bench)))
    assert again["status"] == "skipped"


def test_reconfiguration_with_fewer_shards_moves_every_foreign_shard_and_meta(tmp_path: Path):
    """A split re-extracted under another configuration with FEWER shards leaves no foreign
    ``shard-00000k`` for k >= n, no foreign ``_meta.json`` and no foreign ``_layout.json``: the first unit moves them all
    to ONE ``_stale/<utc>/`` before its own resume, and validate accepts the new set without hand clean-up."""
    from yavih.io.validate import validate_features

    bench = FakeBenchmark()
    for i in range(3):
        _run(tmp_path, bench=bench, shard=(i, 3), canaries=0)
    sd = _split_dir(tmp_path)
    assert validate_features(sd, bench=bench).ok and (sd / "_meta.json").exists()
    old = read_sidecar(sd / "shard-000002.json")
    second, _ = _run(tmp_path, bench=bench, shard=(0, 1), canaries=0, args={"layer": "13"})
    assert second["status"] == "ok" and second["shard"]["n"] == 1
    assert second["bridge"]["describe_sha256"] != old["bridge"]["describe_sha256"]
    assert _files(sd) == ["shard-000000.json", "shard-000000.tar"]
    assert json.loads((sd / "_layout.json").read_text())["n"] == 1  # the new configuration froze its own layout
    (stale,) = list((sd / STALE_DIR).iterdir())  # one step, one directory
    assert sorted(p.name for p in stale.iterdir()) == ["_layout.json", "_meta.json"] + [f"shard-00000{k}.{e}" for k in range(3) for e in ("json", "tar")]
    rep = validate_features(sd, bench=bench)
    assert rep.ok, [f.to_json() for f in rep.errors]


def test_a_frozen_layout_refuses_another_shard_count_and_survives_a_rerun(tmp_path: Path):
    """The shard count is not identity. The first unit freezes the split's layout; a unit of
    the same configuration asking for another n (a plan built from another GPU's clips_per_s or another
    --shard-minutes) is refused with E_LAYOUT (exit 1, re-plan) before anything moves, and the finished shards stay."""
    from yavih.io.layout import LayoutError

    bench = FakeBenchmark()
    for i in range(3):
        _run(tmp_path, bench=bench, shard=(i, 3), canaries=0)
    sd = _split_dir(tmp_path)
    layout = json.loads((sd / "_layout.json").read_text())
    assert layout["n"] == 3 and len(layout["ranges"]) == 3 and layout["model"] == "np-test"
    before = _files(sd)
    for profile in (None, str(_profile_of(tmp_path, bench))):  # the plan's fast path and the plain path alike
        with pytest.raises(LayoutError) as ei:
            _run(tmp_path, bench=bench, shard=(0, 1), canaries=0, expect_profile=profile)
        assert ei.value.exit_code == 1 and ei.value.code == "E_LAYOUT" and "plan build" in str(ei.value)
    assert _files(sd) == before and not (sd / STALE_DIR).exists()
    again, _ = _run(tmp_path, bench=bench, shard=(2, 3), canaries=0, expect_profile=str(_profile_of(tmp_path, bench)))
    assert again["status"] == "skipped"


def test_a_split_without_layout_infers_it_from_the_finished_shards(tmp_path: Path):
    """A split whose _layout.json is missing: the layout most of the configuration's finished shards
    carry is inferred and frozen; a straggler of an older shard plan is moved aside; nothing is re-extracted."""
    bench = FakeBenchmark()
    for i in range(2):
        _run(tmp_path, bench=bench, shard=(i, 2), canaries=0)
    sd = _split_dir(tmp_path)
    (sd / "_layout.json").unlink()  # lost or deleted
    first = read_sidecar(sd / "shard-000000.json")
    again, _ = _run(tmp_path, bench=bench, shard=(1, 2), canaries=0)
    assert again["status"] == "skipped"
    layout = json.loads((sd / "_layout.json").read_text())
    assert layout["n"] == 2 and layout["inferred_from_shards"] is True
    assert read_sidecar(sd / "shard-000000.json")["timing"] == first["timing"]  # untouched


def test_a_live_claim_is_not_ready_and_a_dead_writers_claim_is_taken_over(tmp_path: Path):
    """One writer per shard. Another live task's claim → E_SHARD_CLAIMED (exit 3), nothing
    touched; a claim without a heartbeat for the lease is taken over, the dead writer's tmp swept, and the sidecar
    records the takeover."""
    import os as _os

    from yavih.io.claim import ClaimHeld, claim_name

    sd = _split_dir(tmp_path)
    sd.mkdir(parents=True)
    claim = sd / claim_name(0)
    claim.write_text(json.dumps({"token": "t", "host": "another-host", "pid": 1, "lease_s": 900}))
    (sd / "shard-000000.tar.another-host.1.1.tmp").write_bytes(b"the other writer's partial tar")
    with pytest.raises(ClaimHeld) as ei:
        _run(tmp_path, canaries=0)
    assert ei.value.exit_code == 3 and ei.value.code == "E_SHARD_CLAIMED"
    assert (sd / "shard-000000.tar.another-host.1.1.tmp").exists() and not (sd / "shard-000000.json").exists()
    old = time.time() - 7200
    _os.utime(claim, (old, old))  # no heartbeat for two hours: that writer is gone
    ok, _ = _run(tmp_path, canaries=0)
    assert ok["status"] == "ok" and ok["claim"]["took_over"]["host"] == "another-host"
    assert not claim.exists() and not list(sd.glob("*.tmp"))


def test_sweep_keeps_the_same_configurations_siblings_and_a_running_siblings_files(tmp_path: Path):
    """The sweep moves only what another configuration or shard plan wrote: siblings of the same plan stay, and neither
    a tar without a sidecar (a sibling between tar rename and sidecar write) nor another unit's tmp is touched."""
    bench = FakeBenchmark()
    _run(tmp_path, bench=bench, shard=(1, 3), canaries=0)
    sd = _split_dir(tmp_path)
    (sd / "shard-000002.tar").write_bytes(b"in flight")
    (sd / "shard-000002.json.host.1.2.tmp").write_bytes(b"{")
    _run(tmp_path, bench=bench, shard=(0, 3), canaries=0)
    assert _files(sd) == ["shard-000000.json", "shard-000000.tar", "shard-000001.json", "shard-000001.tar",
                          "shard-000002.json.host.1.2.tmp", "shard-000002.tar"]
    assert not (sd / STALE_DIR).exists()


def test_sweep_tolerates_a_sibling_task_moving_the_same_files_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every array task of a reconfigured split sweeps the same foreign shards: a file a sibling moved first is skipped,
    never an error (exit 5) for the unit that lost the race."""
    from yavih.io import sidecar as S

    bench = FakeBenchmark()
    for i in range(2):
        _run(tmp_path, bench=bench, shard=(i, 2), canaries=0)
    sd = _split_dir(tmp_path)
    real_move = S.shutil.move
    lost: list[str] = []

    def racing_move(src, dst):
        if not lost and str(src).endswith("shard-000001.json"):
            lost.append(str(src))
            Path(src).unlink()  # the sibling's move happened between our listing and our move
            raise FileNotFoundError(src)
        return real_move(src, dst)

    monkeypatch.setattr(S.shutil, "move", racing_move)
    second, _ = _run(tmp_path, bench=bench, shard=(0, 1), canaries=0, args={"layer": "13"})
    assert second["status"] == "ok" and lost
    assert _files(sd) == ["shard-000000.json", "shard-000000.tar"]


@pytest.mark.parametrize("na_bridge", ["init", "encode"])
def test_not_applicable_moves_an_earlier_configurations_shards_and_meta_aside(tmp_path: Path, na_bridge: str):
    """Writing NOT_APPLICABLE.json under another configuration moves the earlier
    configuration's shards and its ``_meta.json`` to _stale/, so neither validate (E_NA_AND_SHARDS) nor the probe (a
    stale meta read as this configuration's) sees them."""
    from yavih.io.validate import validate_features

    bench = FakeBenchmark()
    for i in range(2):
        _run(tmp_path, bench=bench, shard=(i, 2), canaries=0)
    sd = _split_dir(tmp_path)
    assert validate_features(sd, bench=bench).ok
    bridge = NotApplicableInitBridge if na_bridge == "init" else NotApplicableEncodeBridge
    summary, _ = _run(tmp_path, bridge=bridge, bench=bench, shard=(0, 2), canaries=0, args={"layer": "13"})
    assert summary["status"] == "not_applicable"
    assert _files(sd) == [NOT_APPLICABLE_FILE] and not (sd / "_layout.json").exists()  # the verdict replaces the shard plan
    moved = sorted(p.name for p in (sd / STALE_DIR).glob("*/*"))
    assert moved == ["_layout.json", "_meta.json", "shard-000000.json", "shard-000000.tar", "shard-000001.json", "shard-000001.tar"], moved
    rep = validate_features(sd, bench=bench)
    assert rep.ok and rep.coverage.get("not_applicable") is True, [f.to_json() for f in rep.errors]


def test_not_applicable_of_the_same_configuration_keeps_its_siblings_but_never_a_meta(tmp_path: Path):
    """A NotApplicable raised at run time by the SAME configuration that wrote sibling shards leaves those shards (they
    are not another configuration's; validate names the conflict, E_NA_AND_SHARDS) but no ``_meta.json``: validate writes
    one only next to shards, so beside NOT_APPLICABLE.json it is never this verdict's."""
    from yavih.io.validate import validate_features

    bench = FakeBenchmark()
    for i in range(2):
        _run(tmp_path, bench=bench, shard=(i, 2), canaries=0)
    sd = _split_dir(tmp_path)
    assert validate_features(sd, bench=bench).ok
    assert describe_of(NotApplicableEncodeBridge, bench) == describe_of(NumpyBridge, bench)  # one configuration
    (sd / "shard-000000.json").unlink()  # shard 0 is re-run (e.g. lost), its media now refused
    (sd / "shard-000000.tar").unlink()
    summary, _ = _run(tmp_path, bridge=NotApplicableEncodeBridge, bench=bench, shard=(0, 2), canaries=0)
    assert summary["status"] == "not_applicable"
    assert _files(sd) == [NOT_APPLICABLE_FILE, "shard-000001.json", "shard-000001.tar"]
    assert [p.name for p in (sd / STALE_DIR).glob("*/*")] == ["_meta.json"]
    assert any(f.code == "E_NA_AND_SHARDS" for f in validate_features(sd, bench=bench).errors)


def test_not_applicable_from_init_with_a_profile_keeps_the_planned_configurations_shards(tmp_path: Path):
    """``__init__`` raised before ``describe()``: the live configuration is unknown, so the planned profile's
    ``describe_sha256`` stands in for it — that configuration's shards stay (validate names the conflict), another
    configuration's go to _stale/, and the ``_meta.json`` goes either way."""
    from yavih.io.validate import validate_features

    bench = FakeBenchmark()
    for i in range(2):
        _run(tmp_path, bench=bench, shard=(i, 2), canaries=0)
    sd = _split_dir(tmp_path)
    assert validate_features(sd, bench=bench).ok
    (sd / "shard-000000.json").unlink()
    (sd / "shard-000000.tar").unlink()
    summary, _ = _run(tmp_path, bridge=NotApplicableInitBridge, bench=bench, shard=(0, 2), canaries=0,
                      expect_profile=str(_profile_of(tmp_path, bench)))
    assert summary["status"] == "not_applicable" and summary["describe_sha256"] is None
    assert _files(sd) == [NOT_APPLICABLE_FILE, "shard-000001.json", "shard-000001.tar"]
    assert [p.name for p in (sd / STALE_DIR).glob("*/*")] == ["_meta.json"]


# ----------------------------------------------------------------------------- deadline / SIGTERM (exit 4)


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGUSR1"])
def test_signal_finishes_batch_then_exits_4_with_nothing_partial(tmp_path: Path, sig):
    class B(SigtermBridge):
        signal_name = sig

    before = signal.getsignal(getattr(signal, sig))
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=B, canaries=0)
    assert ei.value.exit_code == 4 and ei.value.code == E_PREEMPTED and sig in str(ei.value)
    assert _listing(_split_dir(tmp_path)) == []
    assert signal.getsignal(getattr(signal, sig)) == before  # handler restored


def test_signal_lets_the_current_batch_finish(tmp_path: Path):
    seen = []

    class B(SigtermBridge):
        def encode(self, samples):
            out = super().encode(samples)
            seen.append(len(samples))
            return out

    with pytest.raises(ExtractError):
        _run(tmp_path, bridge=B, canaries=0)
    assert seen == [4]  # exactly one full batch, then the shard was abandoned


def test_past_deadline_exits_4_before_reading_anything(tmp_path: Path):
    bench = FakeBenchmark()
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bench=bench, canaries=0, deadline=time.time() - 1)
    assert ei.value.exit_code == 4 and ei.value.code == E_DEADLINE
    assert bench.reader_calls == [] and _listing(_split_dir(tmp_path)) == []


def test_deadline_hit_mid_shard_exits_4(tmp_path: Path):
    class Slow(NumpyBridge):
        name = "np-slow"

        def encode(self, samples):
            time.sleep(0.03)
            return super().encode(samples)

    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, bridge=Slow, canaries=0, deadline=time.time() + 0.05, decode_workers=1, read_ahead=0)
    assert ei.value.exit_code == 4
    assert _listing(_split_dir(tmp_path)) == []


def test_deadline_env_from_exec_is_honoured(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("YAVIH_DEADLINE", str(time.time() - 1))
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, canaries=0)
    assert ei.value.exit_code == 4
    with pytest.raises(ExtractError) as ei:
        _run(tmp_path, canaries=0, deadline="+0")
    assert ei.value.exit_code == 4


# ----------------------------------------------------------------------------- byte read-ahead


def test_read_ahead_reads_every_stream_once_before_the_bridge(tmp_path: Path):
    bench = FakeBenchmark()
    seen_reads = []

    class Spy(NumpyBridge):
        name = "np-readspy"

        def encode(self, samples):
            seen_reads.extend(s.streams["video"].reads for s in samples)  # read-ahead already touched the bytes
            return super().encode(samples)

    summary, _ = _run(tmp_path, bridge=Spy, bench=bench, canaries=0, decode_workers=4, read_ahead=2)
    assert summary["rows"]["n_written"] == 20 and all(n >= 1 for n in seen_reads)
    sc = read_sidecar(_split_dir(tmp_path) / "shard-000000.json")
    assert sc["timing"]["read_s"] >= 0 and sc["env"]["decode_workers"] == 4


def test_read_ahead_zero_still_works(tmp_path: Path):
    summary, _ = _run(tmp_path, canaries=0, read_ahead=0)
    assert summary["rows"]["n_written"] == 20


# ----------------------------------------------------------------------------- bridge check


def test_bridge_check_profile_records_invariance(tmp_path: Path):
    out = tmp_path / "profile.json"
    bench = FakeBenchmark(missing_rows={1})
    prof = bridge_check(bridge=NumpyBridge, bench=bench, n=6, decode_workers=2, out=out)
    assert prof["status"] == "ok" and prof["bench"] == "synth_fake" and prof["split"] == "val" and prof["n"] == 6
    assert prof["N"] == 4 and prof["D"] == 6 and prof["pos_mode"] == "per_sample" and prof["streams"] == ["video"]
    assert prof["invariance"] == {"rerun_bit_identical": True, "rerun_rel_l2": 0.0, "batch_rel_l2": 0.0, "order_rel_l2": 0.0, "tolerance": 1e-5, "tolerance_reason": None}
    assert prof["rules_checked"] == {"video": {"temporal": "uniform", "spatial": True}}
    assert prof["used"] == {"video": {"times": [0.24, 0.76, 1.24, 1.76]}} and prof["media"]["video"]["fps"] == 25.0
    assert prof["clips_per_s"] > 0 and prof["decode_workers"] == 2
    assert any(w["code"] == "W_MISSING_ROWS" for w in prof["warnings"])  # row 1 missing: 6 PRESENT rows were used
    assert not any(w["code"] == W_NO_PREPROCESSING_SOURCE for w in prof["warnings"])
    assert "bit_identical_rerun" not in prof and "decoders" not in prof
    assert json.loads(out.read_text())["invariance"]["rerun_bit_identical"] is True
    # the profile records the declared names and the consistency digest
    assert (prof["model"], prof["reading"]) == ("np-test", "v") and prof["describe_sha256"] == hashing.describe_sha256(prof["describe"])
    assert prof["notes"] == [] and prof["bridge_name"] == "np-test"
    assert not any(w["code"] in ("W_WEIGHTS_UNFINGERPRINTED", "W_RULE_UNCHECKED") for w in prof["warnings"])


class _CustomRules(NumpyBridge):
    """Opts out of both rule vocabularies: legitimate, recorded as a note."""

    name = "np-custom"

    def input_block(self) -> dict:
        blk = super().input_block()
        blk["video"]["temporal"] = {"rule": "custom", "description": "the model's own clip sampler (irregular, deterministic)"}
        blk["video"]["spatial"] = {"rule": "custom", "description": "letterbox to 4:3, then resize"}
        return blk


def test_bridge_check_custom_rule_is_a_note_not_a_warning(tmp_path: Path):
    prof = bridge_check(bridge=_CustomRules, bench=FakeBenchmark(), n=4, out=tmp_path / "p.json")
    assert prof["status"] == "ok"
    assert [n["code"] for n in prof["notes"]] == ["N_RULE_CUSTOM"]
    msg = prof["notes"][0]["message"]
    assert "own clip sampler" in msg and "letterbox" in msg and "checked: out-of-media" in msg
    assert not any(w["code"] in ("W_RULE_UNCHECKED", "W_RULE_CUSTOM") for w in prof["warnings"]), prof["warnings"]
    assert prof["rules_checked"] == {"video": {"temporal": "custom", "spatial": False}}
    assert json.loads((tmp_path / "p.json").read_text())["notes"][0]["code"] == "N_RULE_CUSTOM"


def test_bridge_check_custom_rule_still_checks_out_of_media_times(tmp_path: Path):
    """The opt-out skips the rule comparison only: reported times outside the media still fail."""

    class CustomButOutside(_CustomRules):
        def one(self, s):
            f, p, u = super().one(s)
            return f, p, {"video": {"times": [t + 100.0 for t in u["video"]["times"]]}}

    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=CustomButOutside, bench=FakeBenchmark(), n=4)
    assert ei.value.code == "E_TIMES_OUT_OF_MEDIA"


def test_bridge_check_null_weights_is_w_weights_unfingerprinted(tmp_path: Path):
    prof = bridge_check(bridge=_Unfingerprinted, bench=FakeBenchmark(), n=4)
    assert prof["status"] == "ok" and prof["model"] == "np-test"
    codes = [w["code"] for w in prof["warnings"]]
    assert codes.count("W_WEIGHTS_UNFINGERPRINTED") == 1


@pytest.mark.parametrize(
    "cls, code, where",
    [
        (BatchStatsBridge, E_BATCH_INVARIANCE, "alone"),
        (ClipIdCacheBridge, E_BATCH_INVARIANCE, "alone"),
        (OrderBridge, E_BATCH_INVARIANCE, "order"),
        (DropoutBridge, E_RERUN_DRIFT, "bit-identical"),
    ],
)
def test_bridge_check_hard_checks_fail_and_write_no_profile(tmp_path: Path, cls, code, where):
    out = tmp_path / "profile.json"
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=cls, bench=FakeBenchmark(), n=8, decode_workers=1, out=out)
    assert ei.value.code == code and ei.value.exit_code == 2 and where in str(ei.value)
    assert not out.exists()


def test_bridge_check_declared_tolerance_admits_small_batch_drift(tmp_path: Path):
    class Drift(NumpyBridge):
        name = "np-drift"

        def describe(self):
            return dict(super().describe(), batch_tolerance={"rel_l2": 1e-3, "reason": "bf16 kernels differ between batch sizes"})

        def encode(self, samples):
            feats, pos, used = super().encode(samples)
            return feats * (1.0 + 1e-5 * (len(samples) > 1)), pos, used

    prof = bridge_check(bridge=Drift, bench=FakeBenchmark(), n=8, decode_workers=1)
    inv = prof["invariance"]
    assert inv["tolerance"] == 1e-3 and inv["tolerance_reason"].startswith("bf16") and 0 < inv["batch_rel_l2"] <= 1e-3
    assert any(w["code"] == "W_LOOSE_TOLERANCE" for w in prof["warnings"])
    with pytest.raises(ExtractError) as ei:  # the same drift without a declaration fails at the default 1e-5
        class Undeclared(Drift):
            name = "np-drift-undeclared"

            def describe(self):
                d = super().describe()
                d.pop("batch_tolerance")
                return d

        bridge_check(bridge=Undeclared, bench=FakeBenchmark(), n=8, decode_workers=1)
    assert ei.value.code == E_BATCH_INVARIANCE


def test_bridge_check_network_rule_mismatch_and_oracle(tmp_path: Path, monkeypatch):
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=NetworkBridge, bench=FakeBenchmark(), n=4, decode_workers=1)
    assert ei.value.code == E_NETWORK
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=WrongTimesBridge, bench=FakeBenchmark(), n=4, decode_workers=1)
    assert ei.value.code == "E_RULE_MISMATCH" and ei.value.exit_code == 2

    class FakeOracle(NumpyBridge):
        name = "np-oracle"
        ORACLE = True

    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=FakeOracle, bench=FakeBenchmark(), n=4, decode_workers=1)
    assert ei.value.code == E_ORACLE
    monkeypatch.setenv("YAVIH_TESTING", "1")
    assert bridge_check(bridge=FakeOracle, bench=FakeBenchmark(), n=4, decode_workers=1)["status"] == "ok"


def test_bridge_check_warns_on_missing_preprocessing_source_and_side_effects(tmp_path: Path):
    prof = bridge_check(bridge=NoPreprocBridge, bench=FakeBenchmark(), n=4, decode_workers=1)
    assert any(w["code"] == W_NO_PREPROCESSING_SOURCE for w in prof["warnings"])
    # the warning says where it looked and which layout passes
    msg = next(w["message"] for w in prof["warnings"] if w["code"] == W_NO_PREPROCESSING_SOURCE)
    assert "NoPreprocBridge.__doc__ line 3" in msg and "only template text" in msg and "Expected layout" in msg, msg

    class B(WritingBridge):
        pass

    B.target = str(tmp_path / "side.cache")
    prof = bridge_check(bridge=B, bench=FakeBenchmark(), n=4, decode_workers=1)
    assert any(w["code"] == "W_SIDE_EFFECT" for w in prof["warnings"])


def test_bridge_check_not_applicable(tmp_path: Path):
    out = tmp_path / "p.json"
    prof = bridge_check(bridge=NotApplicableInitBridge, bench=FakeBenchmark(), n=4, out=out)
    assert prof["status"] == "not_applicable" and json.loads(out.read_text())["status"] == "not_applicable"
    prof = bridge_check(bridge=NotApplicableEncodeBridge, bench=FakeBenchmark(), n=4)
    assert prof["status"] == "not_applicable" and (prof["model"], prof["reading"]) == ("np-test", "v")


def test_bridge_check_smoke_set_without_present_rows_is_exit_3():
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=NumpyBridge, bench=FakeBenchmark(n={"train": 4, "val": 3}, missing_rows={0, 1, 2}), n=4)
    assert ei.value.exit_code == 3
    assert "data roots searched" in str(ei.value) and "--synthetic" in str(ei.value)  # names where it looked


# ----------------------------------------------------------------------------- sharing, templates, --synthetic


def test_bridge_file_imports_a_sibling_module_whatever_the_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A model's bridges share one ``<model>_common.py`` next to them: the bridge's directory is on sys.path."""
    d = tmp_path / "bridges"
    d.mkdir()
    (d / "mm_common.py").write_text("from test_extract_fixtures import NumpyBridge\n\nclass Base(NumpyBridge):\n    name = 'mm-base'\n")
    (d / "mm_k400.py").write_text("from mm_common import Base\n\nclass MmK400(Base):\n    name = 'mm-k400'\n")
    monkeypatch.chdir(tmp_path / ".." if (tmp_path / "..").is_dir() else "/")
    monkeypatch.setattr("sys.path", list(__import__("sys").path))
    cls = load_bridge_class(str(d / "mm_k400.py"))
    assert cls.name == "mm-k400" and bridge_check(bridge=str(d / "mm_k400.py"), bench=FakeBenchmark(), n=4)["status"] == "ok"


def test_bridge_import_error_keeps_the_traceback(tmp_path: Path):
    (tmp_path / "broken_bridge.py").write_text("import no_such_module_xyz\n")
    with pytest.raises(ArgError) as ei:
        load_bridge_class(str(tmp_path / "broken_bridge.py"))
    assert "Traceback" in str(ei.value) and "no_such_module_xyz" in str(ei.value) and "broken_bridge.py" in str(ei.value)


class _TodoModel(NumpyBridge):
    MODEL_NAME = "TODO-model"


class _TodoNorm(NumpyBridge):
    def input_block(self) -> dict:
        blk = super().input_block()
        blk["video"]["norm"] = {"mean": None, "std": [0.5], "source": "TODO repo@commit file:line"}
        return blk


class _ConstantFeatures(NumpyBridge):
    def encode(self, samples):
        feats, pos, used = super().encode(samples)
        return np.zeros_like(feats), pos, used


@pytest.mark.parametrize("cls, code, needle", [
    (_TodoModel, "E_TEMPLATE_UNFILLED", "model = 'TODO-model'"),
    (_TodoNorm, "E_TEMPLATE_UNFILLED", "['mean'] null"),
    (_ConstantFeatures, "E_CONSTANT_FEATURES", "same features"),
])
def test_bridge_check_refuses_an_unfilled_template(tmp_path: Path, cls, code, needle):
    out = tmp_path / "p.json"
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=cls, bench=FakeBenchmark(), n=4, decode_workers=1, out=out)
    assert ei.value.code == code and ei.value.exit_code == 2 and needle in str(ei.value) and not out.exists()


def test_unfilled_describe_and_constant_features_units():
    d = describe_of(NumpyBridge)
    assert extract.unfilled_describe(d) == []
    d["input"]["video"]["norm"] = {"source": "none: the model takes raw pixels"}  # no constants is a declaration, not a gap
    assert extract.unfilled_describe(d) == []
    d["weights_ref"] = "TODO: hub id"
    d["input"]["video"]["norm"] = None
    assert extract.unfilled_describe(d) == ["weights_ref = 'TODO: hub id'", "input['video']['norm'] is null"]
    res = extract._PassResult()
    res.by_index[0] = (np.ones((2, 3), np.float32), None, {})
    assert not extract.constant_features(res)  # one clip proves nothing
    res.by_index[1] = (np.ones((2, 3), np.float32), None, {})
    assert extract.constant_features(res)
    res.by_index[2] = (np.full((2, 3), 2.0, np.float32), None, {})
    assert not extract.constant_features(res)


def test_bridge_check_synthetic_stands_in_for_the_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A bridge is smoke-tested before any data is staged, on tiny media with the benchmark's streams and facts."""
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    from yavih import testing

    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "home"))
    real = testing.make_synthetic_benchmark(tmp_path / "real", "box", n_train=2, n_eval=2, media="none", name="realbox")
    seen = {}

    class Probe(testing.random_bridge(None, N=4, D=8)):
        def __init__(self, bench, args=None):
            super().__init__(bench, args)
            seen.update(name=bench.name, streams=dict(bench.streams), facts=dict(bench.facts))

    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=Probe, bench=str(real), n=3, synthetic=True, out=tmp_path / "p.json")
    assert ei.value.code == "E_SYNTHETIC" and ei.value.exit_code == 1
    prof = bridge_check(bridge=Probe, bench=str(real), n=3, synthetic=True, decode_workers=1)
    assert prof["status"] == "ok" and prof["synthetic"] is True and prof["n"] == 3 and prof["bench"] == "realbox"
    assert seen == {"name": "realbox", "streams": {"video": "video", "audio": "audio"}, "facts": {"annotated_second": 5.0}}


@pytest.mark.parametrize("name", ["esc50", "k400", "in1k", "vggsounder", "ave", "nyuv2_depth"])
def test_make_synthetic_like_mirrors_a_packaged_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    pytest.importorskip("PIL")
    from yavih import testing
    from yavih.benchmark import Benchmark

    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("YAVIH_CONFIG", str(tmp_path / "none.toml"))
    real = Benchmark.load(name)
    synth = Benchmark.load(str(testing.make_synthetic_like(tmp_path, real, n_eval=2)))
    rv, sv = real.view(), synth.view()
    assert (sv.name, sv.readout, sv.streams, sv.facts) == (rv.name, rv.readout, rv.streams, rv.facts)
    split = synth.roles()["eval"][0]
    s = next(iter(synth.reader(split, rows=range(0, 1))))
    assert set(s.streams) == set(rv.streams) and all(v.info["bytes"] for v in s.streams.values())


# ----------------------------------------------------------------------------- python -m yavih.extract


def _bridge_file(tmp_path: Path, cls_name: str) -> Path:
    f = tmp_path / "bridge_under_test.py"
    f.write_text(textwrap.dedent(f"""
        from test_extract_fixtures import {cls_name} as _Base
        class Under(_Base):
            pass
        """))
    return f


@pytest.fixture
def fake_load(monkeypatch):
    """``Benchmark.load`` resolves names to FakeBenchmarks (the loader is not under test here)."""
    benches: dict[str, FakeBenchmark] = {}

    def load(cls, ref, *, data=None, fetch=False):
        return benches.setdefault(ref, FakeBenchmark(name=ref))

    from yavih.benchmark import Benchmark

    monkeypatch.setattr(Benchmark, "load", classmethod(load))
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return benches


def test_main_exit_codes_and_json(tmp_path: Path, fake_load, capsys):
    f = _bridge_file(tmp_path, "NumpyBridge")
    base = ["--bench", "fakeb", "--bridge", str(f), "--split", "train", "--decode-workers", "1"]
    out1, out2 = str(tmp_path / "run1"), str(tmp_path / "run2")
    assert extract.main([*base, "--out", out1, "--shard", "0/1", "--canaries", "2", "--read-ahead", "1", "--json"]) == 0
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["status"] == "ok" and env["data"]["rows"]["n_canary"] == 2 and "sidecar_path" in env["paths"]
    assert extract.main([*base, "--out", out1, "--shard", "0/1", "--json"]) == 0  # skipped
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["status"] == "skipped"
    assert extract.main([*base, "--out", out1, "--shard", "0/1", "--arg", "bogus=1", "--json"]) == 1
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["status"] == "error" and env["errors"][0]["code"] == "E_ARG"
    assert extract.main([*base, "--out", out1, "--shard", "2/2"]) == 1  # bad shard
    assert extract.main([*base, "--out", out2, "--shard", "0/1", "--deadline", "+0"]) == 4
    drifted = _profile_of(tmp_path, FakeBenchmark(), args={"layer": 99}, describe_sha256="0" * 64)
    assert extract.main([*base, "--out", out2, "--shard", "0/1", "--expect-profile", str(drifted), "--json"]) == 2
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["errors"][0]["code"] == "E_CONFIG_DRIFT"
    assert _listing(Path(out2) / "features" / "train") == []
    assert extract.main(["--help"]) == 0
    assert extract.main([]) == 1


def test_main_not_applicable_is_exit_0_and_refusals_exit_2_5(tmp_path: Path, fake_load, capsys):
    out = tmp_path / "run"
    f = _bridge_file(tmp_path, "NotApplicableInitBridge")
    assert extract.main(["--bench", "b", "--bridge", str(f), "--split", "val", "--shard", "0/1", "--out", str(out), "--json"]) == 0
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["status"] == "not_applicable" and (out / "features" / "val" / NOT_APPLICABLE_FILE).exists()
    for cls, code in (("BadPrecisionBridge", 2), ("TwoTupleBridge", 2), ("NetworkBridge", 2), ("CrashingBridge", 5)):
        f = _bridge_file(tmp_path, cls)
        assert extract.main(["--bench", "b", "--bridge", str(f), "--split", "train", "--shard", "0/1", "--out", str(out)]) == code, cls
    f = _bridge_file(tmp_path, "SigtermBridge")
    assert extract.main(["--bench", "b", "--bridge", str(f), "--split", "train", "--shard", "0/1", "--out", str(out), "--canaries", "0"]) == 4
    assert _listing(out / "features" / "train") == []


# ----------------------------------------------------------------------------- caches, opaque paths, fp16 canaries


def _synth_clf(tmp_path: Path, n_train: int = 16, n_eval: int = 8):
    from yavih import testing as T
    from yavih.benchmark import Benchmark

    folder = T.make_synthetic_benchmark(tmp_path / "bench", "clf", n_train=n_train, n_eval=n_eval, media="tiny")
    return Benchmark.from_dir(folder)


def _path_cached_batch_mean_bridge(bench, *, cached: bool):
    """Adds the batch mean to its features (forbidden: docs/bridge.md) and, optionally, memoizes them by the
    RawStream paths: such a cache hides the batch dependence from bridge check unless every pass gets fresh opaque paths."""
    from yavih import testing as T

    base = T.random_bridge(bench, N=4, D=16)

    class PathCached(base):  # type: ignore[valid-type, misc]
        name = "path-cached"

        def __init__(self, b, args=None):
            super().__init__(b, args)
            self.cache: dict = {}

        def encode(self, samples):
            keys = [tuple(sorted((n, st.path) for n, st in s.streams.items())) for s in samples]
            miss = [k for k, key in enumerate(keys) if not (cached and key in self.cache)]
            fresh = {}
            if miss:
                f, _, u = super().encode([samples[k] for k in miss])
                f = f + f.mean(axis=0, keepdims=True)
                for j, k in enumerate(miss):
                    fresh[keys[k]] = (f[j], u[j])
                    if cached:
                        self.cache[keys[k]] = (f[j], u[j])
            got = [self.cache[key] if cached and key in self.cache else fresh[key] for key in keys]
            return np.stack([g[0] for g in got]), self._pos.copy(), [g[1] for g in got]

    return PathCached


@pytest.mark.parametrize("cached", [False, True])
def test_bridge_check_a_path_keyed_cache_cannot_hide_batch_dependence(tmp_path: Path, cached: bool):
    bench = _synth_clf(tmp_path)
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=_path_cached_batch_mean_bridge(bench, cached=cached), bench=bench, n=8)
    assert ei.value.code == "E_BATCH_INVARIANCE"


def test_canaries_catch_a_path_keyed_cache(tmp_path: Path):
    bench = _synth_clf(tmp_path, n_train=32)
    run_extract(bench=bench, bridge=_path_cached_batch_mean_bridge(bench, cached=True), split="train", shard=(0, 1), out=tmp_path / "run",
                canaries=4, decode_workers=2)
    rep = validate_features(_split_dir(tmp_path), bench=bench, write_meta=False)
    assert any(f.code == "E_BATCH_INVARIANCE" for f in rep.errors)


def test_fp16_storage_rounding_is_not_batch_dependence(tmp_path: Path):
    """A float32 drift of ~1e-6 between batch compositions (cuBLAS with different batch sizes) passes bridge check;
    after float16 storage it flips some values by one ulp, which the canary check must not call E_BATCH_INVARIANCE."""
    from yavih import testing as T

    bench = _synth_clf(tmp_path, n_train=32)
    base = T.random_bridge(bench, N=64, D=256, dtype="float16", batch_size=4)

    class Drift(base):  # type: ignore[valid-type, misc]
        name = "fp32-drift"

        def encode(self, samples):
            feats = np.stack([self._feat(s) * (1.0 + 1e-6 * k) for k, s in enumerate(samples)]).astype(np.float32)
            return feats, self._pos.copy(), [self._used(s) for s in samples]

    prof = bridge_check(bridge=Drift, bench=bench, n=8)
    assert prof["status"] == "ok" and 0 < prof["invariance"]["batch_rel_l2"] <= 1e-5
    summary = run_extract(bench=bench, bridge=Drift, split="train", shard=(0, 1), out=tmp_path / "run", canaries=4, decode_workers=2)
    assert summary["status"] == "ok"
    sc = read_sidecar(_split_dir(tmp_path) / "shard-000000.json")
    assert sc["canary"]["max_rel_l2"] <= 1e-5
    rep = validate_features(_split_dir(tmp_path), bench=bench, write_meta=False)
    assert rep.ok and not any(f.code == "E_BATCH_INVARIANCE" for f in rep.errors), rep.errors
