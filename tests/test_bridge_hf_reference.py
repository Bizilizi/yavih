"""The shipped ``hf_reference`` bridge. No download: ``_load_model`` is monkeypatched
with a tiny torch module and an extractor stand-in with the HF feature-extractor call shape.

The bridge decodes the stored wav itself, declares its input rules and runs one clip per forward pass (no
padding to the batch maximum), so bridge check's hard invariance checks pass.
"""

from __future__ import annotations

import importlib.util
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from test_extract_fixtures import FakeBenchmark, view_of
from yavih.bridge import NotApplicable, validate_describe
from yavih.bridges.examples import hf_reference as hf

HAS_TORCH = importlib.util.find_spec("torch") is not None


def wav_bytes(split: str, row: int, stream: str, *, sr: int = 16000, seconds: float = 1.0) -> bytes:
    """A real 16-bit PCM WAV (stdlib ``wave``), different per row: what the reader hands over."""
    n = int(sr * seconds)
    x = np.sin(2 * np.pi * (220 + 10 * row) * np.arange(n) / sr) * 0.3
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def audio_bench(**kw) -> FakeBenchmark:
    bench = FakeBenchmark(streams={"audio": "audio"}, payload_fn=wav_bytes, **kw)
    return bench


def test_module_is_cheap_and_declares_the_selftest_constants():
    assert hf.SELFTEST_BENCH == "esc50" and hf.SELFTEST_N == 32
    assert hf.REFERENCE_ACC is None or 0.0 < hf.REFERENCE_ACC <= 1.0
    assert 0 < hf.REFERENCE_TOL < 1
    assert hf.BRIDGE is hf.HFReferenceBridge
    assert hf.HFReferenceBridge.streams == ("audio",)
    assert hf.HFReferenceBridge.ARGS["model"].default == hf.REFERENCE_MODEL


def test_not_applicable_without_an_audio_stream_before_any_import():
    with pytest.raises(NotApplicable) as ei:
        hf.HFReferenceBridge(view_of(FakeBenchmark()), {})
    assert "audio" in ei.value.reason


def test_resample_identity_and_rate_change():
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    assert hf._resample(x, 16000, 16000) is x or np.array_equal(hf._resample(x, 16000, 16000), x)
    if importlib.util.find_spec("scipy") is None:
        pytest.skip("scipy not importable")
    y = hf._resample(np.random.default_rng(0).standard_normal(44100).astype(np.float32), 44100, 16000)
    assert y.dtype == np.float32 and abs(len(y) - 16000) <= 2


@pytest.fixture
def tiny_model(monkeypatch):
    if not HAS_TORCH:
        pytest.skip("torch not importable")
    import torch

    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(1, 5)

        def forward(self, input_values, attention_mask=None, output_hidden_states=False):
            x = input_values.unsqueeze(-1)  # [B, L, 1]
            h = self.proj(x)  # [B, L, 5]
            return SimpleNamespace(last_hidden_state=h, hidden_states=(x.expand(-1, -1, 5), h) if output_hidden_states else None)

    class Extractor:
        sampling_rate = 16000
        do_normalize = True

        def __call__(self, waves, sampling_rate, return_tensors, padding):
            assert padding is False and len(waves) == 1  # one clip per forward: nothing is padded to a batch maximum
            L = max(len(w) for w in waves)
            arr = np.zeros((len(waves), L), dtype=np.float32)
            for i, w in enumerate(waves):
                arr[i, : len(w)] = (w - w.mean()) / (w.std() + 1e-7)
            return {"input_values": torch.from_numpy(arr)}

    loads = []

    def fake_load(model_ref, device="auto"):
        loads.append((model_ref, device))
        torch.manual_seed(0)  # the same "checkpoint" on every load: bridge check builds a second instance
        return torch, Tiny(), Extractor()

    monkeypatch.setattr(hf.HFReferenceBridge, "_load_model", staticmethod(fake_load))
    return loads


def test_describe_validates_and_encode_reports_used(tiny_model):
    bench = audio_bench()
    view = view_of(bench)
    b = hf.HFReferenceBridge(view, {})
    d = b.describe()
    validate_describe(d, b, view)
    assert d["views"] == 1 and d["pooling"] == "mean" and d["reading"] == "a"
    assert d["input"]["audio"]["front_end"]["sample_rate"] == 16000 and d["input"]["audio"]["norm"]["source"]
    assert len(d["weights_sha256"]) == 64 and d["weights_ref"] == hf.REFERENCE_MODEL and d["args"]["model"] == hf.REFERENCE_MODEL
    assert tiny_model == [(hf.REFERENCE_MODEL, "auto")]
    samples = [bench._sample("val", r) for r in range(3)]
    for s in samples:
        s.streams["audio"].info.update(sample_rate=16000, n_samples=16000, duration_s=1.0)
    feats, pos, used = b.encode(samples)
    assert pos is None and tuple(np.asarray(feats).shape) == (3, 1, 5) and len(used) == 3 and "audio" in used[0]
    from yavih import hashing

    d2 = hf.HFReferenceBridge(view, {"layer": "0"}).describe()
    assert hashing.describe_sha256(d2) != hashing.describe_sha256(d)
    assert hashing.describe_sha256(dict(d, weights_ref="/local/copy")) == hashing.describe_sha256(d)


def test_bridge_check_hard_checks_pass_for_the_reference_bridge(tiny_model):
    """The reference bridge must be instance-wise: no padding to the batch maximum."""
    from yavih.extract import bridge_check

    bench = audio_bench(n={"train": 10, "val": 9}, infos={r: {"audio": {"sample_rate": 16000, "n_samples": 16000 - 1000 * (r % 3), "duration_s": (16000 - 1000 * (r % 3)) / 16000}} for r in range(10)})
    prof = bridge_check(bridge=hf.HFReferenceBridge, bench=bench, n=6, decode_workers=1)
    assert prof["invariance"]["rerun_bit_identical"] is True and prof["invariance"]["batch_rel_l2"] <= 1e-5


def test_run_extract_end_to_end_with_the_reference_bridge(tiny_model, tmp_path):
    from yavih.extract import run_extract

    bench = audio_bench(n={"train": 10, "val": 9})
    summary = run_extract(bench=bench, bridge=hf.HFReferenceBridge, split="val", shard=(0, 1), out=tmp_path / "run", canaries=2, decode_workers=1)
    assert summary["status"] == "ok" and summary["features"]["N"] == 1 and summary["features"]["D"] == 5
    assert summary["features"]["pos_mode"] == "none" and summary["rows"]["n_written"] == 9
    assert summary["canary"]["max_rel_l2"] <= 1e-5
    assert summary["env"]["precision_live"] == {"matmul_precision": "highest", "allow_tf32": False, "deterministic": True}


def test_import_error_message_names_the_extras(monkeypatch):
    if not HAS_TORCH:
        pytest.skip("torch not importable")
    import builtins

    real_import = builtins.__import__

    def no_transformers(name, *a, **k):
        if name.startswith("transformers"):
            raise ImportError("no transformers")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_transformers)
    with pytest.raises(ImportError) as ei:
        hf.HFReferenceBridge._load_model("x/y")
    assert "transformers" in str(ei.value) and "yavih[media,probe]" in str(ei.value)
