"""The readout registry: every readout describes itself; the core asks the registry.

A readout is one module that exports ``READOUT = Readout(...)`` and its driver ``run_<name>(ctx)``. The built-in
ones are ``yavih.readouts.{clf, seq, grid, box}``; a package adds one through the entry-point group
``yavih.readouts`` (``name = "pkg.module"``, the module exporting ``READOUT``). ``benchmark.yaml`` validation,
``protocol_hash`` (the readout version), the pinned ``ProbeConfig``, the floor outputs, the dense-readout checks
of ``features validate`` and ``plan``, and the metric table all consult :func:`get` / :func:`names` instead of a
hard-coded set, so a new task type touches no core module.

Importing a readout module must not import torch (the descriptor is read in the model env, where
``protocol_hash`` is computed): a driver imports its torch code lazily. This module is stdlib-only.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BUILTIN", "ENTRY_POINT_GROUP", "Readout", "UnknownReadout", "get", "names", "all_readouts", "register", "reset"]

#: The readout modules shipped with yavih, in documentation order.
BUILTIN: tuple[str, ...] = ("yavih.readouts.clf", "yavih.readouts.seq", "yavih.readouts.grid", "yavih.readouts.box")
ENTRY_POINT_GROUP = "yavih.readouts"


class UnknownReadout(KeyError):
    """A ``readout`` name no registered readout declares."""


def _none(*_a: Any, **_k: Any) -> None:
    return None


@dataclass(frozen=True)
class Readout:
    """What the core needs to know about one readout.

    Schema (``Benchmark.load``): ``label_kinds`` the ``manifest.label.kind`` values it scores; ``protocol_keys`` the
    per-benchmark protocol keys its driver consumes; ``param_keys`` the exact ``readout_params`` key set;
    ``targets`` whether labels come from ``benchmark.yaml`` ``targets`` (and only then); ``full_frame`` whether
    ``output_contract.target_frame`` must be ``full_frame``; ``requires_t_bins`` / ``requires_grid`` ``True``
    (required), ``False`` (forbidden) or ``None`` (optional) for ``output_contract.requires``; ``validate_spec``
    the remaining cross-checks ``(readout_params, requires, protocol) -> None`` (raises ``ValueError``).

    Identity: ``version`` enters ``protocol_hash`` on the scoring side: any change to the head, loss,
    floor, metric or protocol of the readout bumps it, and features stay valid. ``input_keys`` names the
    ``readout_params`` / ``protocol`` keys the readout's ``check_outputs`` reads beyond the core ones
    (``hashing.INPUT_READOUT_PARAMS``, ``hashing.INPUT_PROTOCOL_KEYS``): they enter ``input_hash``.

    Probe: ``train`` the pinned training block; ``head(num_classes=, T=, grid=, target_size=)`` the head
    descriptor; ``loss(label_kind)``; ``task_sizes(bench)`` the sizes a pinned config takes from a benchmark;
    ``geometry(cfg)`` the keyword arguments that rebuild a config's pinned twin; ``head_families`` / ``losses``
    the values its configs may carry.

    Scoring: ``floor_outputs(train_targets, n_eval, protocol=, label_kind=, num_classes=)`` the label-only
    constant output scored through the floor metrics; ``dense`` whether features need media positions
    (``features validate``, ``plan``); ``check_outputs(view, requires) -> list[str]`` an optional extra
    output-contract check on a feature set; ``synthetic`` an optional builder of a tiny fixture
    (``yavih.testing``). ``module`` is the module that declares it; its ``run_<name>(ctx)`` is the driver."""

    name: str
    version: int
    label_kinds: frozenset[str]
    module: str
    protocol_keys: frozenset[str] = frozenset()
    param_keys: frozenset[str] = frozenset()
    targets: bool = False
    full_frame: bool = False
    requires_t_bins: bool | None = None
    requires_grid: bool | None = None
    validate_spec: Callable[..., None] = _none
    dense: bool = False
    train: Mapping[str, Any] = field(default_factory=dict)
    head: Callable[..., dict] | None = None
    loss: Callable[[str | None], str] | None = None
    task_sizes: Callable[[Any], dict] = lambda bench: {}
    geometry: Callable[[Any], dict] = lambda cfg: {}
    head_families: frozenset[str] = frozenset()
    losses: frozenset[str] = frozenset()
    floor_outputs: Callable[..., Any] | None = None
    check_outputs: Callable[..., list] | None = None
    synthetic: Callable[..., Any] | None = None
    input_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum() or not self.name.islower():
            raise ValueError(f"Readout name {self.name!r} must match [a-z0-9_]+")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError(f"Readout {self.name}: version must be an int >= 1")
        for f in ("label_kinds", "protocol_keys", "param_keys", "head_families", "losses", "input_keys"):
            object.__setattr__(self, f, frozenset(getattr(self, f)))

    def driver(self) -> Callable[[Any], dict]:
        """``run_<name>`` of the declaring module (imported on call; may import torch)."""
        return getattr(importlib.import_module(self.module), f"run_{self.name}")


_LOCK = threading.RLock()
_REGISTRY: dict[str, Readout] | None = None
_EXTRA: dict[str, Readout] = {}


def _entry_point_modules() -> list[tuple[str, str]]:
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return []
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 API
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    return [(ep.name, ep.value) for ep in eps]


def _from_module(modname: str) -> Readout:
    mod = importlib.import_module(modname)
    r = getattr(mod, "READOUT", None)
    if not isinstance(r, Readout):
        raise TypeError(f"{modname} does not export READOUT = yavih.readouts.registry.Readout(...)")
    return r


def _load() -> dict[str, Readout]:
    global _REGISTRY
    with _LOCK:
        if _REGISTRY is not None:
            return _REGISTRY
        reg: dict[str, Readout] = {}
        for modname in BUILTIN:
            r = _from_module(modname)
            reg[r.name] = r
        for ep_name, value in _entry_point_modules():
            modname = value.split(":", 1)[0]
            try:
                r = _from_module(modname)
            except Exception as e:  # noqa: BLE001 - a broken plugin must name itself, not break the core
                import warnings

                warnings.warn(f"readout entry point {ep_name!r} ({value}) failed to load: {type(e).__name__}: {e}", stacklevel=2)
                continue
            if r.name in reg:
                import warnings

                warnings.warn(f"readout entry point {ep_name!r} redefines readout {r.name!r}; ignored", stacklevel=2)
                continue
            reg[r.name] = r
        for name, r in _EXTRA.items():
            reg.setdefault(name, r)
        _REGISTRY = reg
        return reg


def register(readout: Readout, *, replace: bool = False) -> Readout:
    """Add a readout at run time (tests, an in-process plugin). Entry points are the persistent way."""
    with _LOCK:
        reg = _load()
        if readout.name in reg and not replace:
            raise ValueError(f"readout {readout.name!r} already registered")
        reg[readout.name] = readout
        _EXTRA[readout.name] = readout
        return readout


def reset() -> None:
    """Forget run-time registrations and rescan (tests)."""
    global _REGISTRY
    with _LOCK:
        _REGISTRY = None
        _EXTRA.clear()


def get(name: str) -> Readout:
    try:
        return _load()[name]
    except KeyError:
        raise UnknownReadout(f"unknown readout {name!r}; registered: {names()}") from None


def names() -> tuple[str, ...]:
    return tuple(_load())


def all_readouts() -> dict[str, Readout]:
    return dict(_load())
