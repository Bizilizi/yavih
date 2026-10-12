"""Determinism is declared at extraction and recorded at probing.

- ``bridge check``: a bridge declaring ``precision.deterministic: false`` is held to its declared tolerance on the
  rerun pass (``W_RERUN_NONDETERMINISTIC`` + ``invariance.rerun_rel_l2``), a deterministic one to bit identity.
- ``plan``'s profile check accepts a non-bit-identical rerun only under ``deterministic: false`` within tolerance.
- ``precision_scope`` records the ops torch ran without a deterministic kernel; ``determinism_record`` reports them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from test_extract_fixtures import PRECISION_OK, FakeBenchmark, NumpyBridge
from yavih.extract import E_RERUN_DRIFT, W_RERUN_NONDETERMINISTIC, ExtractError, bridge_check
from yavih.plan.matrix import ProfileError, check_invariance


def _noisy(rel: float, *, deterministic: bool) -> type:
    class Noisy(NumpyBridge):
        """Multiplicative noise per encode call: a non-deterministic kernel's footprint.

        Preprocessing source: example/model@abc123 transforms.py::eval_transform
            Resize(224) -> CenterCrop(224) -> x / 255 -> (x - 0.5) / 0.5
        """

        name = "np-noisy"

        def __init__(self, bench, args=None):
            super().__init__(bench, args)
            self.rng = np.random.default_rng(0)

        def describe(self) -> dict:
            d = super().describe()
            d["precision"] = {**PRECISION_OK, "deterministic": deterministic}
            return d

        def encode(self, samples):
            feats, pos, used = super().encode(samples)
            noise = self.rng.standard_normal(feats.shape).astype(np.float32)
            return (feats * (1.0 + rel * noise)).astype(np.float32), pos, used

    return Noisy


def test_nondeterministic_bridge_passes_within_tolerance(tmp_path):
    prof = bridge_check(bridge=_noisy(1e-6, deterministic=False), bench=FakeBenchmark(), n=8, decode_workers=1, out=tmp_path / "p.json")
    inv = prof["invariance"]
    assert inv["rerun_bit_identical"] is False and 0.0 < inv["rerun_rel_l2"] <= inv["tolerance"]
    assert W_RERUN_NONDETERMINISTIC in {w["code"] for w in prof["warnings"]}
    check_invariance(inv, deterministic=False)  # plan accepts it


def test_deterministic_bridge_with_the_same_noise_fails_rerun():
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=_noisy(1e-6, deterministic=True), bench=FakeBenchmark(), n=8, decode_workers=1)
    assert ei.value.code == E_RERUN_DRIFT and "bit-identical" in str(ei.value)


def test_nondeterministic_bridge_above_tolerance_fails_rerun():
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=_noisy(1e-2, deterministic=False), bench=FakeBenchmark(), n=8, decode_workers=1)
    assert ei.value.code == E_RERUN_DRIFT and "declared tolerance" in str(ei.value)


def test_bit_identical_bridge_records_zero_rerun_and_no_warning():
    prof = bridge_check(bridge=NumpyBridge, bench=FakeBenchmark(), n=4, decode_workers=1)
    assert prof["invariance"]["rerun_bit_identical"] is True and prof["invariance"]["rerun_rel_l2"] == 0.0
    assert W_RERUN_NONDETERMINISTIC not in {w["code"] for w in prof["warnings"]}


def _inv(**kw) -> dict:
    base = {"rerun_bit_identical": True, "batch_rel_l2": 0.0, "order_rel_l2": 0.0, "tolerance": 1e-5, "tolerance_reason": None}
    base.update(kw)
    return base


def test_check_invariance_rerun_rules():
    check_invariance(_inv())  # a bit-identical rerun needs no rerun_rel_l2
    check_invariance(_inv(rerun_bit_identical=False, rerun_rel_l2=5e-6), deterministic=False)
    with pytest.raises(ProfileError, match="deterministic is true"):
        check_invariance(_inv(rerun_bit_identical=False, rerun_rel_l2=5e-6), deterministic=True)
    with pytest.raises(ProfileError, match="not within the tolerance"):
        check_invariance(_inv(rerun_bit_identical=False, rerun_rel_l2=5e-4), deterministic=False)
    with pytest.raises(ProfileError, match="not within the tolerance"):
        check_invariance(_inv(rerun_bit_identical=False), deterministic=False)  # nothing recorded


def test_precision_scope_records_nondeterministic_ops(monkeypatch):
    trainer = pytest.importorskip("yavih.probes.trainer")
    monkeypatch.setattr(trainer, "_NONDETERMINISTIC_OPS", set())
    monkeypatch.setattr(trainer, "_ONCE_OPS", set())
    trainer.reset_determinism()
    assert trainer.determinism_record("cpu") == {"mode": "warn_only", "device": "cpu", "nondeterministic_ops": [],
                                                 "bitwise_reproducible": True}
    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        with trainer.precision_scope():
            warnings.warn("upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation, but you set "
                          "'torch.use_deterministic_algorithms(True, warn_only=True)'.", UserWarning, stacklevel=1)
            warnings.warn("Memory Efficient attention defaults to a non-deterministic algorithm. To explicitly enable "
                          "determinism call torch.use_deterministic_algorithms(True, warn_only=False).", UserWarning, stacklevel=1)
            warnings.warn("something unrelated", UserWarning, stacklevel=1)
    # the unrelated warning is re-emitted, the determinism ones are recorded instead of shown
    assert [str(w.message) for w in outer] == ["something unrelated"]
    rec = trainer.determinism_record("cuda")
    assert rec["nondeterministic_ops"] == ["memory_efficient_attention", "upsample_bilinear2d_backward_out_cuda"]
    assert rec["bitwise_reproducible"] is False and rec["device"] == "cuda"

    # per seed: a reset clears the per-call ops; the once-per-process SDPA warning stays attributed to later CUDA fits
    trainer.reset_determinism()
    assert trainer.determinism_record("cuda")["nondeterministic_ops"] == []  # no scope entered since the reset
    with trainer.precision_scope():
        pass
    assert trainer.determinism_record("cuda")["nondeterministic_ops"] == ["memory_efficient_attention"]
    assert trainer.determinism_record("cpu")["nondeterministic_ops"] == []
