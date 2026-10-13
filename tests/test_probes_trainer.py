"""Training loop (docs/readouts.md common rules) — determinism, micro-batching exactness, val split, schedule,
ckpt save/resume, deadline / stop flag, oracle convergence under ``fast()``, floors."""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest
import torch

from yavih.probes.ckpt import CKPT_FILE, load_ckpt, save_ckpt
from yavih.probes.config import ProbeConfig
from yavih.probes.dense import axis_centres
from yavih.probes.floors import floor_outputs
from yavih.probes.losses import per_sample_loss
from yavih.probes.trainer import (
    ProbeDeadline,
    ProbeInterrupted,
    accumulate_batch,
    build_head_for,
    lr_lambda,
    micro_batch_size,
    predict,
    train_head,
    val_split,
)

D, N = 16, 4


class _RowView:
    """Stand-in for ``yavih.io.reader.RowView``: ``.shape`` + ``__getitem__(ndarray) -> [k, N, D]`` (stored fp16)."""

    def __init__(self, arr: np.ndarray) -> None:
        self._a = arr.astype(np.float16)
        self.shape = arr.shape
        self.backend = "ram"
        self.gathers = 0

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        self.gathers += 1
        return self._a[np.asarray(idx)]


def _oracle_clf(n: int, c: int, seed: int = 0, noise: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Separable: token 0 carries a class-specific direction, other tokens are noise."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, c, size=n)
    protos = rng.standard_normal((c, D)).astype(np.float32)
    feats = rng.standard_normal((n, N, D)).astype(np.float32) * noise
    feats[:, 0] += protos[labels]
    return feats, labels.astype(np.int64)


def _cfg_clf(c: int, **over) -> ProbeConfig:
    cfg = ProbeConfig.pinned("clf", num_classes=c).fast()
    from dataclasses import replace

    return replace(cfg, **over) if over else cfg


# ----------------------------------------------------------------------------- helpers


def test_micro_batch_size() -> None:
    assert micro_batch_size(256, 64, 131072) == 256
    assert micro_batch_size(256, 1968, 131072) == 66
    assert micro_batch_size(8, 100000, 131072) == 1
    assert micro_batch_size(8, 0, 131072) == 8


def test_val_split_rules() -> None:
    tr, va = val_split(100, 0.1, seed=3)
    assert len(tr) == 90 and len(va) == 10
    assert sorted(torch.cat([tr, va]).tolist()) == list(range(100))
    tr2, va2 = val_split(100, 0.1, seed=3)
    assert torch.equal(tr, tr2) and torch.equal(va, va2)
    assert not torch.equal(va, val_split(100, 0.1, seed=4)[1])
    tr, va = val_split(3, 0.1, seed=0)
    assert len(va) == 1 and len(tr) == 2  # max(1, round(0.3)) = 1
    tr, va = val_split(2, 0.9, seed=0)
    assert len(va) == 1 and len(tr) == 1  # capped at n - 1
    tr, va = val_split(1, 0.1, seed=0)
    assert len(tr) == 1 and len(va) == 1  # nothing to carve: val = train


def test_lr_lambda_warmup_then_cosine() -> None:
    f = lr_lambda(100, 0.0)
    assert f(0) == 1.0 and math.isclose(f(50), 0.5, abs_tol=1e-9) and f(100) == 0.0
    g = lr_lambda(100, 0.05)
    assert g(0) == 0.2 and g(4) == 1.0  # ceil(5) warmup steps: 1/5 … 5/5
    assert g(5) == 1.0 and g(99) < 1e-3 and g(1000) == 0.0
    assert all(g(i) <= g(i + 1) + 1e-12 for i in range(4))  # monotone warmup


# ----------------------------------------------------------------------------- training


def test_train_clf_deterministic_and_oracle_converges(tmp_path) -> None:
    feats, labels = _oracle_clf(96, 3)
    cfg = _cfg_clf(3)
    view = _RowView(feats)
    r1 = train_head(view, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    r2 = train_head(_RowView(feats), labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    s1, s2 = r1.head.state_dict(), r2.head.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)  # bit-identical given the seed
    assert r1.n_train == 86 and r1.n_val == 10 and r1.selected_lr == cfg.lr
    assert r1.stop in {"patience", "max_epochs"} and 1 <= r1.epochs_run <= cfg.max_epochs
    assert r1.best_epoch >= 1 and len(r1.history) == r1.epochs_run
    assert r1.s_per_epoch > 0 and len(r1.arms) == 1
    assert view.gathers > 0  # features consumed through the row gather, never materialised
    logits = predict(r1.head, view, np.arange(96), device="cpu")
    acc = float((logits.argmax(1).numpy() == labels).mean())
    assert acc >= 0.95, acc
    # a different seed gives a different fit
    r3 = train_head(_RowView(feats), labels, build_head=build_head_for(cfg), cfg=cfg, seed=1, device="cpu")
    assert any(not torch.equal(s1[k], r3.head.state_dict()[k]) for k in s1)


def test_random_features_do_not_beat_chance_by_much() -> None:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((96, N, D)).astype(np.float32)
    labels = rng.integers(0, 3, size=96).astype(np.int64)
    cfg = _cfg_clf(3)
    r = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    hold = rng.standard_normal((60, N, D)).astype(np.float32)
    pred = predict(r.head, hold, np.arange(60), device="cpu").argmax(1).numpy()
    acc = float((pred == rng.integers(0, 3, size=60)).mean())
    # chance 1/3 at n=60: 0.5 is ~2.7 sigma above it (the run is seeded, so this is deterministic);
    # a probe leaking labels would clear it, one at chance cannot
    assert acc <= 0.5, acc


def test_micro_batching_equals_full_batch() -> None:
    """The token cap changes only the accumulation order, not the optimisation."""
    feats, labels = _oracle_clf(48, 3)
    from dataclasses import replace

    cfg_full = replace(_cfg_clf(3), lr=1e-4, max_epochs=2, batch_size=16, max_tokens_per_forward=10**9)
    cfg_micro = replace(cfg_full, max_tokens_per_forward=N * 5)  # 5 rows per forward, batches of 16
    assert micro_batch_size(16, N, cfg_micro.max_tokens_per_forward) == 5
    # same probe hash is NOT required here (max_tokens is pinned, hence hashed); only the numbers
    r_full = train_head(feats, labels, build_head=build_head_for(cfg_full), cfg=cfg_full, seed=0, device="cpu")
    r_micro = train_head(feats, labels, build_head=build_head_for(cfg_micro), cfg=cfg_micro, seed=0, device="cpu")
    # AdamW normalises by sqrt(v), so fp32 summation-order noise in the gradient (~1e-7) shows up as
    # up to ~lr per step in the weights; at lr=1e-4 two epochs stay within 5e-4. The exact contract is
    # on the gradient (next tests).
    s_full, s_micro = r_full.head.state_dict(), r_micro.head.state_dict()
    for k in s_full:
        torch.testing.assert_close(s_full[k], s_micro[k], atol=5e-4, rtol=1e-2)
    for h_full, h_micro in zip(r_full.history, r_micro.history, strict=True):
        assert math.isclose(h_full["train_loss"], h_micro["train_loss"], rel_tol=1e-4)
        assert math.isclose(h_full["val_loss"], h_micro["val_loss"], rel_tol=1e-4)


def _grad_vector(head: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in head.parameters() if p.grad is not None])


@pytest.mark.parametrize("micro", [1, 3, 5, 16])
def test_accumulated_gradient_equals_whole_batch_ce(micro: int) -> None:
    torch.manual_seed(0)
    cfg = _cfg_clf(3)
    head = build_head_for(cfg)(D)
    x = torch.randn(16, N, D)
    y = torch.randint(0, 3, (16,))
    crit = per_sample_loss("ce")
    head.zero_grad()
    (crit(head(x), y).sum() / 16).backward()
    ref = _grad_vector(head).clone()
    head.zero_grad()
    total = accumulate_batch(head, crit, x, y, None, micro=micro, device="cpu")
    torch.testing.assert_close(_grad_vector(head), ref, atol=1e-6, rtol=1e-5)
    assert math.isclose(total, float(crit(head(x), y).sum()), rel_tol=1e-5)


@pytest.mark.parametrize("micro", [1, 4])
def test_accumulated_gradient_equals_whole_batch_box_giou(micro: int) -> None:
    """GIoU does not decompose as a batch mean of anything."""
    torch.manual_seed(0)
    g = 2
    cfg = ProbeConfig.pinned("box", grid=g).fast()
    head = build_head_for(cfg)(D)
    n_tok = g * g + 3
    x = torch.randn(8, n_tok, D)
    yy, xx = np.meshgrid(axis_centres(g).numpy(), axis_centres(g).numpy(), indexing="ij")
    vis = np.stack([np.zeros(g * g), np.full(g * g, 0.5), yy.reshape(-1), xx.reshape(-1)], -1)
    aud = np.stack([np.ones(3), axis_centres(3).numpy(), -np.ones(3), -np.ones(3)], -1)
    pos = torch.from_numpy(np.concatenate([vis, aud]).astype(np.float32))
    boxes = torch.rand(8, 4) * 0.6 + 0.2
    crit = per_sample_loss("box_l1_giou")
    head.zero_grad()
    (crit(head(x, pos), boxes).sum() / 8).backward()
    ref = _grad_vector(head).clone()
    head.zero_grad()
    accumulate_batch(head, crit, x, boxes, pos, micro=micro, device="cpu")
    torch.testing.assert_close(_grad_vector(head), ref, atol=1e-6, rtol=1e-5)


def test_micro_batching_exact_for_box_loss() -> None:
    """GIoU is not a batch mean of anything; the per-sample-sum contract keeps accumulation exact."""
    from dataclasses import replace

    g = 2
    rng = np.random.default_rng(0)
    n = 24
    n_tok = g * g + 3
    feats = rng.standard_normal((n, n_tok, D)).astype(np.float32)
    yy, xx = np.meshgrid(axis_centres(g).numpy(), axis_centres(g).numpy(), indexing="ij")
    vis = np.stack([np.zeros(g * g), np.full(g * g, 0.5), yy.reshape(-1), xx.reshape(-1)], -1)
    aud = np.stack([np.ones(3), axis_centres(3).numpy(), -np.ones(3), -np.ones(3)], -1)
    pos = np.concatenate([vis, aud]).astype(np.float32)
    boxes = rng.uniform(0.2, 0.8, size=(n, 4)).astype(np.float32)
    boxes[:, 2:] = 0.2
    base = replace(ProbeConfig.pinned("box", grid=g).fast(), lr=1e-4, max_epochs=2, batch_size=8)
    cfg_full = replace(base, max_tokens_per_forward=10**9)
    cfg_micro = replace(base, max_tokens_per_forward=n_tok * 3)
    r_full = train_head(feats, boxes, pos=pos, build_head=build_head_for(cfg_full), cfg=cfg_full, seed=0, device="cpu")
    r_micro = train_head(feats, boxes, pos=pos, build_head=build_head_for(cfg_micro), cfg=cfg_micro, seed=0, device="cpu")
    s_full, s_micro = r_full.head.state_dict(), r_micro.head.state_dict()
    for k in s_full:
        torch.testing.assert_close(s_full[k], s_micro[k], atol=5e-4, rtol=1e-2)
    out = predict(r_full.head, feats, np.arange(n), pos=pos, device="cpu")
    assert out.shape == (n, 4)


def test_lr_grid_selection_on_val_loss() -> None:
    from dataclasses import replace

    feats, labels = _oracle_clf(64, 3)
    cfg = replace(_cfg_clf(3), lr_grid=(1e-6, 1e-3), max_epochs=3)
    r = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    assert len(r.arms) == 2 and {a["lr"] for a in r.arms} == {1e-6, 1e-3}
    assert r.selected_lr == 1e-3  # the lr that actually learns wins on val loss
    assert r.best_val == min(a["best_val"] for a in r.arms)


def test_seq_head_training_with_nan_masked_targets() -> None:
    from dataclasses import replace

    rng = np.random.default_rng(0)
    n, T, C = 48, 4, 3
    feats = rng.standard_normal((n, 6, D)).astype(np.float32)
    pos = np.stack([np.zeros(6), axis_centres(6).numpy(), -np.ones(6), -np.ones(6)], -1).astype(np.float32)
    targets = rng.integers(0, C, size=(n, T)).astype(np.float32)
    targets[:, 0] = np.nan  # unscored segment
    cfg = replace(ProbeConfig.pinned("seq", num_classes=C, T=T).fast(), max_epochs=2)
    r = train_head(feats, targets, pos=pos, build_head=build_head_for(cfg, visual_tag=0), cfg=cfg, seed=0, device="cpu")
    out = predict(r.head, feats, np.arange(n), pos=pos, device="cpu")
    assert out.shape == (n, T, C) and torch.isfinite(out).all()
    assert all(math.isfinite(h["train_loss"]) for h in r.history)


def test_grid_head_training_depth() -> None:
    from dataclasses import replace

    rng = np.random.default_rng(0)
    n, g, S = 32, 2, 6
    feats = rng.standard_normal((n, g * g, D)).astype(np.float32)
    yy, xx = np.meshgrid(axis_centres(g).numpy(), axis_centres(g).numpy(), indexing="ij")
    pos = np.stack([np.zeros(g * g), np.full(g * g, 0.5), yy.reshape(-1), xx.reshape(-1)], -1).astype(np.float32)
    depth = rng.uniform(1.0, 5.0, size=(n, S, S)).astype(np.float32)
    depth[:, 0, 0] = 0.0  # invalid pixel (readout convention: NaN or <= 0)
    cfg = replace(ProbeConfig.pinned("grid", grid=(g, g), target_size=S).fast(), max_epochs=2)
    r = train_head(feats, depth, pos=pos, build_head=build_head_for(cfg, visual_tag=0), cfg=cfg, seed=0, device="cpu")
    out = predict(r.head, feats, np.arange(n), pos=pos, device="cpu")
    assert out.shape == (n, S, S) and bool((out > 0).all())


# ----------------------------------------------------------------------------- ckpt / deadline / signals


def test_ckpt_roundtrip_and_hash_gate(tmp_path) -> None:
    p = str(tmp_path / CKPT_FILE)
    save_ckpt(p, {"epoch": 2, "x": torch.ones(2)}, hashes={"probe_hash": "a", "seed": 0})
    st = load_ckpt(p, hashes={"probe_hash": "a", "seed": 0})
    assert st is not None and st["epoch"] == 2 and torch.equal(st["x"], torch.ones(2))
    assert load_ckpt(p, hashes={"probe_hash": "b", "seed": 0}) is None
    assert load_ckpt(p, hashes={"probe_hash": "a", "seed": 1}) is None
    assert load_ckpt(str(tmp_path / "nope.pt"), hashes={}) is None
    (tmp_path / "bad.pt").write_bytes(b"garbage")
    assert load_ckpt(str(tmp_path / "bad.pt"), hashes={}) is None


def test_stop_flag_writes_ckpt_and_resume_is_equivalent(tmp_path) -> None:
    from dataclasses import replace

    feats, labels = _oracle_clf(64, 3)
    cfg = replace(_cfg_clf(3), max_epochs=4, patience=10, batch_size=16)
    build = build_head_for(cfg)
    ref = train_head(feats, labels, build_head=build, cfg=cfg, seed=0, device="cpu")
    ckpt = str(tmp_path / CKPT_FILE)

    # interrupt at the start of epoch 3 (after 2 epochs = 8 optimizer steps, 8 micro-batches)
    class Flag:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> bool:
            self.calls += 1
            return self.calls > 8

    with pytest.raises(ProbeInterrupted) as ei:
        train_head(feats, labels, build_head=build, cfg=cfg, seed=0, device="cpu", ckpt_path=ckpt, stop_flag=Flag())
    assert ei.value.exit_code == 4 and ei.value.partial is not None and ei.value.partial.epochs_run == 2
    saved = load_ckpt(ckpt, hashes={"probe_hash": __import__("yavih.hashing", fromlist=["x"]).probe_hash(cfg), "seed": 0})
    assert saved is not None and saved["epoch"] == 2 and saved["arm"] == 0
    assert set(saved) >= {"state_dict", "optimizer", "scheduler", "epoch", "rng", "best_state", "best_val", "patience_left"}

    resumed = train_head(feats, labels, build_head=build, cfg=cfg, seed=0, device="cpu", ckpt_path=ckpt)
    assert resumed.epochs_run == ref.epochs_run == 4
    s_ref, s_res = ref.head.state_dict(), resumed.head.state_dict()
    for k in s_ref:
        torch.testing.assert_close(s_ref[k], s_res[k], atol=1e-6, rtol=1e-5)
    for h_ref, h_res in zip(ref.history, resumed.history, strict=True):
        assert math.isclose(h_ref["val_loss"], h_res["val_loss"], rel_tol=1e-5)

    # a checkpoint written under another seed is ignored (fresh fit, still 4 epochs)
    other = train_head(feats, labels, build_head=build, cfg=cfg, seed=1, device="cpu", ckpt_path=ckpt)
    assert other.epochs_run == 4


def test_deadline_raises_between_epochs_and_resume_skips_done_arms(tmp_path) -> None:
    from dataclasses import replace

    feats, labels = _oracle_clf(48, 3)
    cfg = replace(_cfg_clf(3), max_epochs=2, lr_grid=(1e-3, 1e-4))
    build = build_head_for(cfg)
    ckpt = str(tmp_path / CKPT_FILE)
    with pytest.raises(ProbeDeadline) as ed:
        train_head(feats, labels, build_head=build, cfg=cfg, seed=0, device="cpu", deadline=time.monotonic() - 1.0,
                   ckpt_path=ckpt)
    assert ed.value.exit_code == 4 and isinstance(ed.value, TimeoutError)
    # nothing ran → no ckpt
    assert load_ckpt(ckpt, hashes={"probe_hash": __import__("yavih.hashing", fromlist=["x"]).probe_hash(cfg), "seed": 0}) is None

    ref = train_head(feats, labels, build_head=build, cfg=cfg, seed=0, device="cpu")
    # arm 0 completes, arm 1 interrupted after its first epoch (2 + 1 epochs = 6 + 3 steps of 1 micro-batch each … use a
    # counter on epochs through the deadline: deadline far away, stop flag after arm 0 + 1 epoch)
    steps_per_epoch = math.ceil(ref.n_train / cfg.batch_size)
    calls = {"n": 0}

    def flag() -> bool:
        calls["n"] += 1
        return calls["n"] > 3 * steps_per_epoch

    with pytest.raises(ProbeInterrupted):
        train_head(feats, labels, build_head=build, cfg=cfg, seed=0, device="cpu", ckpt_path=ckpt, stop_flag=flag)
    resumed = train_head(feats, labels, build_head=build, cfg=cfg, seed=0, device="cpu", ckpt_path=ckpt)
    assert [a["lr"] for a in resumed.arms] == [1e-3, 1e-4]
    assert resumed.selected_lr == ref.selected_lr
    for a_ref, a_res in zip(ref.arms, resumed.arms, strict=True):
        assert math.isclose(a_ref["best_val"], a_res["best_val"], rel_tol=1e-5)
    s_ref, s_res = ref.head.state_dict(), resumed.head.state_dict()
    for k in s_ref:
        torch.testing.assert_close(s_ref[k], s_res[k], atol=1e-6, rtol=1e-5)


def test_train_head_input_validation() -> None:
    cfg = _cfg_clf(3)
    with pytest.raises(ValueError):
        train_head(np.zeros((0, N, D), np.float32), np.zeros(0), build_head=build_head_for(cfg), cfg=cfg, seed=0)
    with pytest.raises(ValueError):
        train_head(np.zeros((4, N, D), np.float32), np.zeros(3), build_head=build_head_for(cfg), cfg=cfg, seed=0)
    with pytest.raises(TypeError):
        train_head(np.zeros((4, N, D), np.float32), np.zeros(4), build_head=build_head_for(cfg), cfg={"lr": 1}, seed=0)


def test_loader_workers_prefetch_gives_same_result() -> None:
    from dataclasses import replace

    feats, labels = _oracle_clf(48, 3)
    cfg = replace(_cfg_clf(3), max_epochs=2)
    a = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu", loader_workers=0)
    b = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu", loader_workers=2)
    for k, v in a.head.state_dict().items():
        assert torch.equal(v, b.head.state_dict()[k])


class _OrderedView(_RowView):
    """A ``_RowView`` that records every gathered index array, in call order."""

    def __init__(self, arr: np.ndarray) -> None:
        super().__init__(arr)
        self.seen: list[list[int]] = []

    def __getitem__(self, idx):
        self.seen.append(np.asarray(idx).tolist())
        return super().__getitem__(idx)


def test_prefetch_depth_two_gives_same_result() -> None:
    """The default loader (one thread, PREFETCH_DEPTH batches ahead) gathers the inline loop's batches in the inline
    loop's order, and the fit is bit-identical."""
    from dataclasses import replace

    from yavih.probes import trainer as tr

    assert tr.PREFETCH_DEPTH == 2
    feats, labels = _oracle_clf(96, 3)
    cfg = replace(_cfg_clf(3), max_epochs=3, batch_size=8)
    va, vb = _OrderedView(feats), _OrderedView(feats)
    a = train_head(va, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu", loader_workers=0)
    b = train_head(vb, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")  # default: prefetch
    assert va.seen == vb.seen and len(va.seen) > 3 * 8
    assert [h["train_loss"] for h in a.history] == [h["train_loss"] for h in b.history]
    assert [h["val_loss"] for h in a.history] == [h["val_loss"] for h in b.history]
    for k, v in a.head.state_dict().items():
        assert torch.equal(v, b.head.state_dict()[k])


def test_gather_fp16_cast_after_h2d_bitwise_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA gathers keep the stored fp16 and upcast after the copy; CPU upcasts in the gather. fp16 -> fp32 is exact,
    so the two orders give the same fp32 rows and the same fit, bit for bit (both orders exercised on CPU)."""
    from dataclasses import replace

    from yavih.probes import trainer as tr

    feats, labels = _oracle_clf(48, 3)
    view = _RowView(feats)
    idx = torch.tensor([5, 0, 47, 5, 12])
    early = tr._gather(view, idx)
    stored = tr._gather(view, idx, keep_dtype=True)
    assert early.dtype == torch.float32 and stored.dtype == torch.float16
    late = tr._to_device(stored, "cpu")
    assert late.dtype == torch.float32 and torch.equal(early, late)
    assert not tr._cast_on_device("cpu") and tr._cast_on_device("cuda:0") and tr._cast_on_device(torch.device("cuda"))

    cfg = replace(_cfg_clf(3), max_epochs=2, batch_size=8)
    a = train_head(_RowView(feats), labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    pa = predict(a.head, _RowView(feats), np.arange(48), device="cpu")
    monkeypatch.setattr(tr, "_cast_on_device", lambda device: True)  # the CUDA order, on CPU
    b = train_head(_RowView(feats), labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    pb = predict(b.head, _RowView(feats), np.arange(48), device="cpu")
    assert a.history == [dict(h, s=a.history[i]["s"]) for i, h in enumerate(b.history)]
    for k, v in a.head.state_dict().items():
        assert torch.equal(v, b.head.state_dict()[k])
    assert torch.equal(pa, pb)


def test_precision_globals_restored() -> None:
    prev = (torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32)
    feats, labels = _oracle_clf(16, 2)
    from dataclasses import replace

    cfg = replace(_cfg_clf(2), max_epochs=1)
    train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    assert (torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32) == prev


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")
def test_train_on_cuda() -> None:
    feats, labels = _oracle_clf(32, 3)
    cfg = _cfg_clf(3)
    r = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cuda")
    assert r.device == "cuda"


# ----------------------------------------------------------------------------- floors


def test_floor_outputs_clf() -> None:
    out = floor_outputs("clf", np.array([0, 2, 2, 1, 2, 0]), 5, protocol={}, label_kind="single", num_classes=4)
    assert out.shape == (5, 4) and (out.argmax(1) == 2).all()
    # ties → smallest index; num_classes inferred from the labels when not given
    out = floor_outputs("clf", np.array([1, 1, 0, 0]), 2, protocol={}, label_kind="single")
    assert out.shape == (2, 2) and (out.argmax(1) == 0).all()
    multi = np.array([[1, 0, np.nan], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    out = floor_outputs("clf", multi, 3, protocol={}, label_kind="multi")
    np.testing.assert_allclose(out[0], [2 / 3, 2 / 3, 0.0], atol=1e-6)
    assert (out == out[0]).all()
    out_sets = floor_outputs("clf", multi, 2, protocol={}, label_kind="multi_sets")
    np.testing.assert_allclose(out_sets[1], out[0], atol=1e-6)


def test_floor_outputs_seq_grid_box() -> None:
    seq = np.array([[0, 1, 28, 28], [2, 28, 28, -1]], dtype=np.int64)
    out = floor_outputs("seq", seq, 3, protocol={}, label_kind="segments", num_classes=29)
    assert out.shape == (3, 4, 29) and (out.argmax(-1) == 28).all()
    seq_nan = seq.astype(np.float32)
    seq_nan[1, 3] = np.nan
    np.testing.assert_array_equal(floor_outputs("seq", seq_nan, 3, protocol={}, label_kind="segments", num_classes=29), out)

    depth = np.array([[[1.0, 2.0], [0.0, 4.0]], [[1.0, 8.0], [0.0, 16.0]]], dtype=np.float32)  # [2, 2, 2]
    out = floor_outputs("grid", depth, 2, protocol={"max_depth": 10.0}, label_kind="none")
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out[0, 0, 0], 1.0, rtol=1e-5)
    np.testing.assert_allclose(out[0, 0, 1], math.exp((math.log(2) + math.log(8)) / 2), rtol=1e-5)  # geometric mean
    np.testing.assert_allclose(out[0, 1, 1], 4.0, rtol=1e-5)  # 16 > max_depth excluded
    # pixel valid nowhere → global mean log-depth
    global_mean = math.exp(np.mean(np.log([1.0, 2.0, 4.0, 1.0, 8.0])))
    np.testing.assert_allclose(out[0, 1, 0], global_mean, rtol=1e-5)

    class Reader:  # TargetReader-like: .shape + .rows(idx)
        shape = depth.shape

        def rows(self, idx):
            return depth[np.asarray(idx)]

    np.testing.assert_allclose(floor_outputs("grid", Reader(), 1, protocol={"max_depth": 10.0}, label_kind="none")[0],
                               out[0])

    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.4, 0.2], [np.nan, 0, 0, 0]], dtype=np.float32)
    out = floor_outputs("box", boxes, 3, protocol={}, label_kind="box")
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[2], [0.4, 0.6, 0.3, 0.2], atol=1e-6)
    with pytest.raises(ValueError):
        floor_outputs("heat", boxes, 1, protocol={}, label_kind="box")


def test_checkpoint_resume_gate_carries_the_fit_identity(tmp_path) -> None:
    """Deadline: a checkpoint resumes only the fit that wrote it. ``ckpt_hashes`` names the fit
    (the readout driver passes the fold/control tag and the features fingerprint), so fold k in the same
    seed directory can never pick up fold 1's finished head; identical hashes DO resume (the deadline path)."""
    feats, labels = _oracle_clf(48, 3)
    other = np.random.default_rng(7).standard_normal(feats.shape).astype(np.float32)
    cfg = _cfg_clf(3)
    ck = str(tmp_path / CKPT_FILE)
    r1 = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu", ckpt_path=ck, ckpt_hashes={"fit": "fold0"})
    assert (tmp_path / CKPT_FILE).exists()
    r2 = train_head(other, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu", ckpt_path=ck, ckpt_hashes={"fit": "fold1"})
    s1, s2 = r1.head.state_dict(), r2.head.state_dict()
    assert any(not torch.equal(s1[k], s2[k]) for k in s1), "a different fit resumed the previous fit's checkpoint"
    # same identity → the finished checkpoint IS resumed (0 further epochs, same weights): why the driver deletes it
    r3 = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu", ckpt_path=ck, ckpt_hashes={"fit": "fold1"})
    assert all(torch.equal(s2[k], r3.head.state_dict()[k]) for k in s2)


# ----------------------------------------------------------------------------- checkpoint cadence


def _count_writes(monkeypatch) -> list[int]:
    """Patch the trainer's ``save_ckpt`` to record the epoch of every disk write (and still write it)."""
    import yavih.probes.trainer as T

    epochs: list[int] = []
    real = T.save_ckpt

    def spy(path, state, *, hashes):
        epochs.append(int(state["epoch"]))
        real(path, state, hashes=hashes)

    monkeypatch.setattr(T, "save_ckpt", spy)
    return epochs


def test_ckpt_reaches_disk_at_arm_end_not_every_epoch(tmp_path, monkeypatch) -> None:
    """Within CKPT_EVERY_S the checkpoint stays in memory and reaches disk once per arm (its last epoch); the fit is unchanged."""
    from dataclasses import replace

    monkeypatch.delenv("YAVIH_PROBE_CKPT_EVERY_S", raising=False)
    feats, labels = _oracle_clf(48, 3)
    cfg = replace(_cfg_clf(3), max_epochs=3, patience=10, lr_grid=(1e-3, 1e-4))
    epochs = _count_writes(monkeypatch)
    res = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu",
                     ckpt_path=str(tmp_path / CKPT_FILE))
    assert epochs == [3, 3], epochs  # one write per arm, at its last epoch
    ref = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    for k, v in ref.head.state_dict().items():
        assert torch.equal(v, res.head.state_dict()[k])

    # 0 = every epoch
    monkeypatch.setenv("YAVIH_PROBE_CKPT_EVERY_S", "0")
    epochs.clear()
    train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=1, device="cpu", ckpt_path=str(tmp_path / "b.pt"))
    assert epochs == [1, 2, 3, 1, 2, 3], epochs  # every epoch; the arm-end flush finds nothing pending


def test_ckpt_every_s_env() -> None:
    from yavih.probes.trainer import CKPT_EVERY_S, ckpt_every_s

    assert ckpt_every_s({}) == CKPT_EVERY_S == 600.0
    assert ckpt_every_s({"YAVIH_PROBE_CKPT_EVERY_S": "0"}) == 0.0
    assert ckpt_every_s({"YAVIH_PROBE_CKPT_EVERY_S": " "}) == CKPT_EVERY_S
    for bad in ("-1", "soon", "nan", "inf"):
        with pytest.raises(ValueError, match="YAVIH_PROBE_CKPT_EVERY_S"):
            ckpt_every_s({"YAVIH_PROBE_CKPT_EVERY_S": bad})


def test_interrupt_flushes_the_last_completed_epoch_held_in_memory(tmp_path, monkeypatch) -> None:
    """A SIGTERM mid-epoch writes the last COMPLETED epoch (from the in-memory copy, not the half-trained head)."""
    from dataclasses import replace

    monkeypatch.delenv("YAVIH_PROBE_CKPT_EVERY_S", raising=False)
    feats, labels = _oracle_clf(64, 3)
    cfg = replace(_cfg_clf(3), max_epochs=4, patience=10, batch_size=16)
    epochs = _count_writes(monkeypatch)
    calls = {"n": 0}

    def flag() -> bool:  # 4 micro-batches per epoch: stop in the middle of epoch 3
        calls["n"] += 1
        return calls["n"] > 10

    with pytest.raises(ProbeInterrupted):
        train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu",
                   ckpt_path=str(tmp_path / CKPT_FILE), stop_flag=flag)
    assert epochs == [2]  # nothing during the fit, then the flush on the way out
    ref = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu")
    resumed = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu",
                         ckpt_path=str(tmp_path / CKPT_FILE))
    assert resumed.epochs_run == ref.epochs_run == 4
    for k, v in ref.head.state_dict().items():
        torch.testing.assert_close(v, resumed.head.state_dict()[k], atol=1e-6, rtol=1e-5)


def test_resume_of_an_arm_stopped_by_patience_trains_nothing_more(tmp_path, monkeypatch) -> None:
    """The arm-end checkpoint of a patience-stopped arm has ``patience_left == 0``; resuming it (a kill between arms)
    must not train one more epoch (before: the loop only tested ``epoch < max_epochs``)."""
    from dataclasses import replace

    monkeypatch.delenv("YAVIH_PROBE_CKPT_EVERY_S", raising=False)
    rng = np.random.default_rng(3)
    feats = rng.standard_normal((40, N, D)).astype(np.float32)
    labels = rng.integers(0, 3, size=40).astype(np.int64)  # noise labels: val loss rises, patience fires
    cfg = replace(_cfg_clf(3), max_epochs=30, patience=1, lr=1e-2)
    ck = str(tmp_path / CKPT_FILE)
    first = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu", ckpt_path=ck)
    assert first.stop == "patience" and first.epochs_run < 30
    saved = load_ckpt(ck, hashes={"probe_hash": __import__("yavih.hashing", fromlist=["x"]).probe_hash(cfg), "seed": 0})
    assert saved is not None and saved["patience_left"] == 0 and saved["epoch"] == first.epochs_run
    again = train_head(feats, labels, build_head=build_head_for(cfg), cfg=cfg, seed=0, device="cpu", ckpt_path=ck)
    assert again.stop == "patience" and again.epochs_run == first.epochs_run and again.best_epoch == first.best_epoch
    for k, v in first.head.state_dict().items():
        assert torch.equal(v, again.head.state_dict()[k])


def test_precision_scope_sets_cublas_workspace_config(monkeypatch) -> None:
    """``precision_scope`` sets CUBLAS_WORKSPACE_CONFIG for deterministic CUDA matmuls; a user value is kept."""
    from yavih.probes.trainer import precision_scope

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        with precision_scope():
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            assert torch.backends.cudnn.benchmark is False and torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is True  # torch globals restored
    finally:
        torch.backends.cudnn.benchmark = bench
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with precision_scope():
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fresh_probe_process_emits_no_cublas_determinism_warning(tmp_path) -> None:
    """In a fresh process (cuBLAS not yet initialised), a CUDA fit emits no CuBLAS determinism warning."""
    import subprocess
    import sys

    script = tmp_path / "fit.py"
    script.write_text(
        "import warnings, numpy as np\n"
        "warnings.simplefilter('always')\n"
        "from yavih.probes.config import ProbeConfig\n"
        "from yavih.probes.trainer import build_head_for, train_head\n"
        "cfg = ProbeConfig.pinned('clf', num_classes=3).fast()\n"
        "rng = np.random.default_rng(0)\n"
        "x = rng.standard_normal((24, 4, 16)).astype(np.float32); y = rng.integers(0, 3, 24)\n"
        "train_head(x, y, build_head=build_head_for(cfg), cfg=cfg, seed=0, device='cuda')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "CuBLAS" not in r.stderr, r.stderr[-2000:]
