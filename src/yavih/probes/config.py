"""``ProbeConfig`` and its pinned values; re-exports ``PROBE_SCHEMA``.

Torch-free on purpose: ``yavih.hashing.probe_hash`` consumes this dataclass by attribute and
must agree across envs. The hash payload is the explicit ``probe_hash`` field list, so adding a field here does NOT silently move every digest.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from yavih.versions import PROBE_SCHEMA

if TYPE_CHECKING:  # pragma: no cover
    from yavih.benchmark import Benchmark

__all__ = [
    "ProbeConfig",
    "PROBE_SCHEMA",
    "HEAD_FAMILIES",
    "LOSSES",
    "SCHEDULES",
    "SELECTIONS",
    "PRECISIONS",
    "testing_enabled",
    "parse_layout",
]

#: Head families and losses the built-in trainer implements; a registered probe scorer adds its own (``Scorer.head_families``,
#: ``Scorer.losses``), which ``ProbeConfig`` then accepts too.
HEAD_FAMILIES: frozenset[str] = frozenset({"attentive_probe_v1", "dense_query_v1", "dense_av_v1"})
LOSSES: frozenset[str] = frozenset({"ce", "bce", "ce_seq", "l1_log", "box_l1_giou"})
SCHEDULES: frozenset[str] = frozenset({"cosine"})
SELECTIONS: frozenset[str] = frozenset({"val_loss"})
PRECISIONS: frozenset[str] = frozenset({"fp32"})

#: ``ProbeConfig.fast()`` schedule (test-only): one definition for the package.
FAST_LR: float = 3e-3
FAST_EPOCHS: int = 25
FAST_MAX_BATCH: int = 16
FAST_MAX_TOKENS: int = 32768

#: Default box grid when neither ``bench`` nor ``grid=`` is given (vggss_box's benchmark.yaml ``readout_params``).
DEFAULT_BOX_GRID = 14


def testing_enabled(env: dict | None = None) -> bool:
    """``YAVIH_TESTING=1`` gates non-pinned configs (``fast()``)."""
    env = os.environ if env is None else env
    return env.get("YAVIH_TESTING", "") == "1"


def parse_layout(layout: str) -> tuple[str, tuple[int, ...]]:
    """``"point"`` → ``("point", ())``; ``"time:8"`` → ``("time", (8,))``; ``"grid:14x14"`` → ``("grid", (14, 14))``;
    ``"box"`` → ``("box", ())`` (``head.layout``). A size that is still unresolved (``"time:?"``)
    raises ``ValueError`` so an unfilled pinned config cannot build a head."""
    kind, _, rest = layout.partition(":")
    if kind not in {"point", "time", "grid", "box"}:
        raise ValueError(f"unknown head.layout kind {layout!r}")
    if kind in {"point", "box"}:
        if rest:
            raise ValueError(f"head.layout {layout!r} takes no size")
        return kind, ()
    try:
        shape = tuple(int(p) for p in rest.split("x"))
    except ValueError as e:
        raise ValueError(f"head.layout {layout!r} has an unresolved size; pass bench= or T=/grid= to pinned()") from e
    if (kind == "time" and len(shape) != 1) or (kind == "grid" and len(shape) != 2) or any(s < 1 for s in shape):
        raise ValueError(f"head.layout {layout!r} has a malformed size")
    return kind, shape


def _readout(readout: str) -> Any:
    """The readout's self-description (``yavih.readouts.registry``): pinned training block, head, loss, sizes."""
    from yavih.readouts import registry

    try:
        r = registry.get(readout)
    except registry.UnknownScorer as e:
        raise ValueError(f"unknown scorer {readout!r}; expected one of {sorted(registry.names())}") from e
    if not r.is_probe:
        raise ValueError(f"scorer {readout!r} is kind {r.kind!r}: it trains no probe and has no ProbeConfig")
    return r


def _head_families() -> frozenset[str]:
    from yavih.readouts import registry

    return HEAD_FAMILIES.union(*(r.head_families for r in registry.all_scorers().values()))


def _losses() -> frozenset[str]:
    from yavih.readouts import registry

    return LOSSES.union(*(r.losses for r in registry.all_scorers().values()))


def _from_bench(bench: Benchmark) -> dict[str, Any]:
    """Pull the task sizes a pinned config needs out of a loaded ``Benchmark``: the label kind and
    class count here, the readout's own sizes (``Readout.task_sizes``: T, grid, target size, …) from the readout."""
    spec = bench.spec
    out: dict[str, Any] = {"label_kind": spec.manifest.label.kind}
    if spec.manifest.classes is not None:
        try:
            out["num_classes"] = len(bench.classes())
        except NotImplementedError:  # pragma: no cover - a Benchmark without classes()
            pass
    out.update(_readout(spec.readout).task_sizes(bench))
    return out


@dataclass(frozen=True, kw_only=True)
class ProbeConfig:
    """The probe configuration. ``kw_only`` because fields with defaults precede ``loss``."""

    readout: str
    head: dict  # architecture descriptor (JSON primitives)
    lr: float
    lr_grid: tuple[float, ...]
    weight_decay: float
    max_epochs: int
    batch_size: int
    patience: int
    warmup_fraction: float
    schedule: str = "cosine"
    val_fraction: float = 0.1
    loss: str
    selection: str = "val_loss"
    max_tokens_per_forward: int = 131072
    precision: str = "fp32"

    def __post_init__(self) -> None:
        # Validation here (not in the trainer) so a bad config is refused before it is hashed.
        try:
            _readout(self.readout)
        except ValueError as e:
            raise ValueError(f"ProbeConfig.readout: {e}") from None
        if not isinstance(self.head, dict) or self.head.get("family") not in _head_families():
            raise ValueError(f"ProbeConfig.head.family must be one of {sorted(_head_families())}, got {self.head!r}")
        if self.loss not in _losses():
            raise ValueError(f"ProbeConfig.loss must be one of {sorted(_losses())}, got {self.loss!r}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"ProbeConfig.schedule must be one of {sorted(SCHEDULES)}")
        if self.selection not in SELECTIONS:
            raise ValueError(f"ProbeConfig.selection must be one of {sorted(SELECTIONS)}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"ProbeConfig.precision must be one of {sorted(PRECISIONS)} (fp32)")
        object.__setattr__(self, "lr_grid", tuple(float(x) for x in self.lr_grid))
        object.__setattr__(self, "lr", float(self.lr))
        if self.max_epochs < 1 or self.batch_size < 1 or self.patience < 1 or self.max_tokens_per_forward < 1:
            raise ValueError("ProbeConfig: max_epochs, batch_size, patience, max_tokens_per_forward must be >= 1")
        if not 0.0 <= self.val_fraction < 1.0 or not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("ProbeConfig: val_fraction and warmup_fraction must lie in [0, 1)")

    # ------------------------------------------------------------------ construction

    @staticmethod
    def pinned(
        readout: str,
        bench: Benchmark | None = None,
        *,
        num_classes: int | None = None,
        label_kind: str | None = None,
        T: int | None = None,
        grid: tuple[int, int] | int | None = None,
        target_size: int | None = None,
    ) -> ProbeConfig:
        """The pinned table for ``readout``; ``bench`` fills ``head.out`` (num_classes), ``head.layout``
        (``time:T`` from ``readout_params.T``, ``grid:HxW`` from ``readout_params.h/w``, box ``grid``),
        ``head.upsample`` (``protocol.target_size``) and the single/multi loss choice. The keyword
        arguments fill the same slots without a ``Benchmark`` (tests, custom drivers) and override ``bench``.
        An unfilled slot stays a placeholder that the head builders refuse (``parse_layout``)."""
        r = _readout(readout)
        resolved: dict[str, Any] = {}
        if bench is not None:
            resolved.update(_from_bench(bench))
        for k, v in dict(num_classes=num_classes, label_kind=label_kind, T=T, grid=grid, target_size=target_size).items():
            if v is not None:
                resolved[k] = v
        head = r.head(
            num_classes=resolved.get("num_classes"),
            T=resolved.get("T"),
            grid=resolved.get("grid"),
            target_size=resolved.get("target_size"),
        )
        return ProbeConfig(readout=readout, head=head, loss=r.loss(resolved.get("label_kind")), **dict(r.train))

    def fast(self) -> ProbeConfig:
        """A tiny variant for tests; requires ``YAVIH_TESTING=1`` and makes ``result.standard = false``.
        Keeps the head family and task geometry (``layout``, ``out``) so the
        same drivers run, but shrinks depth/heads and the schedule so a CPU fit takes seconds.

        This is THE definition (``yavih.testing.fast(cfg)`` delegates here):
        depth 1/1, 4 heads, ``lr = max(lr, 3e-3)``, no lr grid, no warmup, ``max_epochs = patience = 25``
        (the oracle bridge must clear every floor, incl. the dense grid head, and the patience must not
        cut the short schedule), ``batch_size ≤ 16``, ``max_tokens_per_forward ≤ 32768``."""
        if not testing_enabled():
            raise RuntimeError("ProbeConfig.fast() requires YAVIH_TESTING=1 (non-pinned configs are test-only)")
        head = dict(self.head)
        head["self_attn_depth"] = 1
        head["cross_attn_depth"] = 1
        head["num_heads"] = 4
        return replace(
            self,
            head=head,
            lr=max(float(self.lr), FAST_LR),
            lr_grid=(),
            max_epochs=FAST_EPOCHS,
            batch_size=min(int(self.batch_size), FAST_MAX_BATCH),
            patience=FAST_EPOCHS,
            warmup_fraction=0.0,
            max_tokens_per_forward=min(int(self.max_tokens_per_forward), FAST_MAX_TOKENS),
        )

    # ------------------------------------------------------------------ queries

    def pinned_like(self) -> ProbeConfig:
        """The pinned config with THIS config's task geometry (classes, layout, grid, upsample, loss kind)."""
        return ProbeConfig.pinned(self.readout, **_readout(self.readout).geometry(self))

    def is_pinned(self) -> bool:
        """``True`` iff this config equals the pinned table for its task → ``result.standard``."""
        try:
            return self == self.pinned_like()
        except (ValueError, KeyError):
            return False

    def layout(self) -> tuple[str, tuple[int, ...]]:
        """``parse_layout(head.layout)``."""
        return parse_layout(str(self.head["layout"]))

    def to_json(self) -> dict[str, Any]:
        """JSON-primitive view for ``result.json[probe.config]``."""
        return {
            "readout": self.readout,
            "head": dict(self.head),
            "lr": self.lr,
            "lr_grid": list(self.lr_grid),
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "warmup_fraction": self.warmup_fraction,
            "schedule": self.schedule,
            "val_fraction": self.val_fraction,
            "loss": self.loss,
            "selection": self.selection,
            "max_tokens_per_forward": self.max_tokens_per_forward,
            "precision": self.precision,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ProbeConfig:
        d = dict(d)
        d["lr_grid"] = tuple(d.get("lr_grid", ()))
        return cls(**d)
