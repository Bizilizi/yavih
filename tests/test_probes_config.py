"""``ProbeConfig`` (pinned table, hash, ``fast()``), torch-free config module."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from yavih.hashing import probe_hash
from yavih.probes.config import LOSSES, ProbeConfig, parse_layout

REPO = Path(__file__).resolve().parents[1]

PINNED_TRAIN = {
    "clf": dict(lr=1e-4, lr_grid=(), weight_decay=1e-4, max_epochs=30, batch_size=256, patience=5, warmup_fraction=0.0),
    "seq": dict(lr=3e-4, lr_grid=(1e-4, 3e-4, 1e-3), weight_decay=1e-4, max_epochs=50, batch_size=64, patience=10,
                warmup_fraction=0.05),
    "grid": dict(lr=3e-4, lr_grid=(1e-4, 3e-4, 1e-3), weight_decay=1e-4, max_epochs=50, batch_size=64, patience=10,
                 warmup_fraction=0.05),
    "box": dict(lr=3e-4, lr_grid=(1e-4, 3e-4, 1e-3), weight_decay=1e-4, max_epochs=50, batch_size=64, patience=10,
                warmup_fraction=0.05),
}
PINNED_HEAD = {
    "clf": ("attentive_probe_v1", 3, 1, "none", "point", "ce"),
    "seq": ("dense_query_v1", 3, 1, "sincos(t)+null_embed", "time:8", "ce_seq"),
    "grid": ("dense_query_v1", 3, 1, "sincos(y,x)+null_embed", "grid:14x14", "l1_log"),
    "box": ("dense_av_v1", 2, 2, "sincos(y,x) on binned grid", "box", "box_l1_giou"),
}
RESOLVE = {
    "clf": dict(num_classes=10),
    "seq": dict(num_classes=29, T=8),
    "grid": dict(grid=(14, 14), target_size=224),
    "box": dict(grid=14),
}


@pytest.mark.parametrize("readout", sorted(PINNED_TRAIN))
def test_pinned_table(readout: str) -> None:
    cfg = ProbeConfig.pinned(readout, **RESOLVE[readout])
    for k, v in PINNED_TRAIN[readout].items():
        assert getattr(cfg, k) == v, k
    fam, sd, cd, mp, layout, loss = PINNED_HEAD[readout]
    assert cfg.head["family"] == fam
    assert cfg.head["self_attn_depth"] == sd and cfg.head["cross_attn_depth"] == cd
    assert cfg.head["num_heads"] == 16 and cfg.head["mlp_ratio"] == 4
    assert cfg.head["memory_pos"] == mp and cfg.head["layout"] == layout
    assert cfg.loss == loss
    assert (cfg.schedule, cfg.val_fraction, cfg.selection, cfg.max_tokens_per_forward, cfg.precision) == (
        "cosine", 0.1, "val_loss", 131072, "fp32")
    assert cfg.is_pinned()


def test_pinned_out_and_loss_variants() -> None:
    assert ProbeConfig.pinned("clf", num_classes=7).head["out"] == 7
    assert ProbeConfig.pinned("clf", num_classes=7, label_kind="multi").loss == "bce"
    assert ProbeConfig.pinned("clf", num_classes=7, label_kind="multi_sets").loss == "bce"
    assert ProbeConfig.pinned("clf", num_classes=7, label_kind="single").loss == "ce"
    g = ProbeConfig.pinned("grid", grid=(14, 14), target_size=224).head
    assert g["out"] == 1 and g["upsample"] == 224 and g["activation"] == "exp"
    b = ProbeConfig.pinned("box").head
    assert b["out"] == 4 and b["grid"] == 14 and b["activation"] == "sigmoid"
    assert ProbeConfig.pinned("box", grid=7).head["grid"] == 7


def test_pinned_without_bench_is_placeholder_and_refused_by_parse_layout() -> None:
    cfg = ProbeConfig.pinned("seq")
    assert cfg.head["layout"] == "time:?" and cfg.head["out"] is None
    probe_hash(cfg)  # still hashable (JSON primitives)
    with pytest.raises(ValueError):
        cfg.layout()


def test_pinned_from_bench_like_object() -> None:
    """``bench`` duck-typed: ``spec.manifest.label.kind``, ``spec.manifest.classes``, ``spec.readout_params``,
    ``spec.protocol``, ``classes()`` (the bench fills the pinned config)."""
    from types import SimpleNamespace as NS

    spec = NS(readout="seq", manifest=NS(label=NS(kind="segments"), classes="classes.csv"),
              readout_params={"T": 8}, protocol={"segments_scored": 8})
    bench = NS(spec=spec, classes=lambda: [{"index": i} for i in range(29)])
    cfg = ProbeConfig.pinned("seq", bench)
    assert cfg.head["layout"] == "time:8" and cfg.head["out"] == 29
    spec_g = NS(readout="grid", manifest=NS(label=NS(kind="none"), classes=None),
                readout_params={"h": 14, "w": 14, "out": 1, "visual_stream": "image"},
                protocol={"target_size": 224, "max_depth": 10.0})
    cfg_g = ProbeConfig.pinned("grid", NS(spec=spec_g, classes=lambda: []))
    assert cfg_g.head["layout"] == "grid:14x14" and cfg_g.head["upsample"] == 224
    spec_b = NS(readout="box", manifest=NS(label=NS(kind="box"), classes=None),
                readout_params={"grid": 14, "visual_stream": "video", "conditioning_stream": "audio"}, protocol={})
    assert ProbeConfig.pinned("box", NS(spec=spec_b, classes=lambda: [])).head["grid"] == 14
    spec_c = NS(readout="clf", manifest=NS(label=NS(kind="multi"), classes="classes.csv"), readout_params={}, protocol={})
    cfg_c = ProbeConfig.pinned("clf", NS(spec=spec_c, classes=lambda: [1, 2, 3]))
    assert cfg_c.loss == "bce" and cfg_c.head["out"] == 3


def test_probe_hash_excludes_seed_and_moves_with_head() -> None:
    a = ProbeConfig.pinned("clf", num_classes=10)
    b = ProbeConfig.pinned("clf", num_classes=11)
    assert probe_hash(a) == probe_hash(ProbeConfig.pinned("clf", num_classes=10))
    assert probe_hash(a) != probe_hash(b)
    assert probe_hash(a) != probe_hash(a.fast())
    assert probe_hash(a) == probe_hash(a.to_json())  # dict and dataclass agree (duck typing)
    assert ProbeConfig.from_json(a.to_json()) == a


def test_fast_requires_testing_and_is_not_standard(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ProbeConfig.pinned("clf", num_classes=3)
    fast = cfg.fast()
    assert fast.max_epochs < cfg.max_epochs and fast.lr_grid == () and fast.batch_size <= 32
    assert fast.head["family"] == cfg.head["family"] and fast.head["layout"] == cfg.head["layout"]
    assert fast.head["out"] == cfg.head["out"]
    assert not fast.is_pinned() and cfg.is_pinned()
    monkeypatch.delenv("YAVIH_TESTING")
    with pytest.raises(RuntimeError):
        cfg.fast()


def test_validation() -> None:
    base = ProbeConfig.pinned("clf", num_classes=3)
    with pytest.raises(ValueError):
        ProbeConfig(**{**base.to_json(), "lr_grid": (), "loss": "mse"})
    with pytest.raises(ValueError):
        ProbeConfig(**{**base.to_json(), "lr_grid": (), "precision": "bf16"})
    with pytest.raises(ValueError):
        ProbeConfig(**{**base.to_json(), "lr_grid": (), "head": {"family": "mlp"}})
    with pytest.raises(ValueError):
        ProbeConfig.pinned("heatmap")
    assert ProbeConfig(**{**base.to_json(), "lr_grid": [1e-3]}).lr_grid == (1e-3,)


def test_parse_layout() -> None:
    assert parse_layout("point") == ("point", ())
    assert parse_layout("box") == ("box", ())
    assert parse_layout("time:8") == ("time", (8,))
    assert parse_layout("grid:14x14") == ("grid", (14, 14))
    for bad in ("time", "grid:14", "time:8x8", "blob:3", "point:1"):
        with pytest.raises(ValueError):
            parse_layout(bad)


def test_losses_registry_matches_config() -> None:
    from yavih.probes.losses import LOSSES as IMPL

    assert set(IMPL) == set(LOSSES)


def test_config_module_is_torch_free() -> None:
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import yavih.probes.config as c, yavih.probes.floors as f\n"
        "print(c.ProbeConfig.pinned('clf', num_classes=2).loss)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO),
                         env={"PYTHONPATH": str(REPO / "src"), "YAVIH_TESTING": "1", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ce"
