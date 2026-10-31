"""The scorer registry: every scorer describes itself; the core asks the registry.

A scorer is one module that exports ``SCORER = Scorer(...)`` and, for the shared kinds, its driver ``run_<name>(ctx)``.
The built-in ones are ``yavih.readouts.{clf, seq, grid, box}`` (the pinned probes); a package adds one through the
entry-point group ``yavih.scorers`` (``name = "pkg.module"``, the module exporting ``SCORER``); a benchmark folder
owns one by exporting ``SCORER`` from its ``eval.py`` (registered under the folder's name by ``Benchmark.load``).
``benchmark.yaml`` validation, ``protocol_hash`` (the scorer version), the pinned configuration and its hash (the
second slot of ``eval_hash``), the floor outputs, eligibility, the dense checks of ``features validate`` and
``plan``, the conformance population and the metric table all consult :func:`get` / :func:`names` instead of a
hard-coded set, so a new task type touches no core module.

Three kinds: ``probe`` trains the shared pinned head (``train`` / ``head`` / ``loss`` required; the configuration
hash is ``hashing.probe_hash`` of a ``ProbeConfig``); ``stateless`` scores stored outputs without training;
``direct`` scores stored predictions. For the last two the configuration is the constant
:class:`ScorerConfig` and its hash :func:`yavih.hashing.scorer_config_hash`.

Importing a scorer module must not import torch (the descriptor is read in the model env, where ``protocol_hash``
is computed): a driver imports its torch code lazily. This module is stdlib-only.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "BUILTIN",
    "ENTRY_POINT_GROUP",
    "SCORER_KINDS",
    "Scorer",
    "ScorerConfig",
    "UnknownScorer",
    "get",
    "names",
    "all_scorers",
    "register",
    "register_folder",
    "reset",
]

#: The scorer modules shipped with yavih, in documentation order.
BUILTIN: tuple[str, ...] = ("yavih.readouts.clf", "yavih.readouts.seq", "yavih.readouts.grid", "yavih.readouts.box")
ENTRY_POINT_GROUP = "yavih.scorers"
#: ``Scorer.kind``: ``probe`` trains the shared pinned head; ``stateless`` scores stored outputs without training;
#: ``direct`` scores stored predictions (a folder's own or an authors' scorer).
SCORER_KINDS: frozenset[str] = frozenset({"probe", "stateless", "direct"})


class UnknownScorer(KeyError):
    """A ``scorer`` name no registered scorer declares."""


def _none(*_a: Any, **_k: Any) -> None:
    return None


@dataclass(frozen=True)
class ScorerConfig:
    """The configuration of a non-probe scorer: a constant. ``to_json()`` is what ``result.json[probe.config]``
    records; :func:`yavih.hashing.scorer_config_hash` hashes ``scorer`` and ``version``."""

    scorer: str
    version: int
    kind: str

    @property
    def readout(self) -> str:
        return self.scorer

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Scorer:
    """What the core needs to know about one scorer.

    Schema (``Benchmark.load``): ``label_kinds`` the ``manifest.label.kind`` values it scores; ``protocol_keys`` the
    per-benchmark protocol keys its driver consumes; ``param_keys`` the exact ``readout_params`` key set;
    ``targets`` whether labels come from ``benchmark.yaml`` ``targets`` (and only then); ``full_frame`` whether
    ``output_contract.target_frame`` must be ``full_frame``; ``requires_t_bins`` / ``requires_grid`` ``True``
    (required), ``False`` (forbidden) or ``None`` (optional) for ``output_contract.requires``; ``validate_spec``
    the remaining cross-checks ``(readout_params, requires, protocol) -> None`` (raises ``ValueError``);
    ``output_keys`` the ``output_contract.schema`` keys a feature set must carry for it.

    Identity: ``version`` enters ``protocol_hash`` on the scoring side: any change to the head, loss,
    floor, metric or protocol of the scorer bumps it, and features stay valid. ``input_keys`` names the
    ``readout_params`` / ``protocol`` keys its eligibility reads beyond the core ones
    (``hashing.INPUT_READOUT_PARAMS``, ``hashing.INPUT_PROTOCOL_KEYS``): they enter ``input_hash``.

    Kind and family: ``kind`` is ``probe`` / ``stateless`` / ``direct``; ``family`` names the scorer whose metrics,
    targets and conformance population this one shares (``None`` = itself).

    Probe (kind ``probe``): ``train`` the pinned training block; ``head(num_classes=, T=, grid=, target_size=)`` the
    head descriptor; ``loss(label_kind)``; ``task_sizes(bench)`` the sizes a pinned config takes from a benchmark;
    ``geometry(cfg)`` the keyword arguments that rebuild a config's pinned twin; ``head_families`` / ``losses``
    the values its configs may carry.

    Scoring: ``score(ctx) -> blocks`` the driver (``None`` = ``run_<name>`` of ``module``); ``floor_outputs(train_targets,
    n_eval, protocol=, label_kind=, num_classes=)`` the label-only constant output scored through the floor
    metrics (``None`` = the scorer declares no floors and the folder's ``metrics.floors`` must be empty); ``dense``
    whether features need media positions (``features validate``, ``plan``); ``eligibility(view) -> (ok, missing)``
    the eligibility verdict on a validated feature set (``None`` = the ``output_contract.requires`` rules);
    ``config_hash(cfg)`` / ``pinned(bench)`` override the per-kind defaults (:meth:`hash_config`,
    :meth:`pinned_config`); ``conformance_population(ctx, tag)`` builds the fixture population
    (``yavih.conformance``); ``needs_gpu`` whether its unit asks the scheduler for a GPU (``None`` = probes do).
    ``module`` is the module that declares it."""

    name: str
    version: int
    label_kinds: frozenset[str]
    module: str
    kind: str = "probe"
    family: str | None = None
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
    input_keys: frozenset[str] = frozenset()
    output_keys: frozenset[str] = frozenset()
    eligibility: Callable[[Any], tuple[bool, list[str]]] | None = None
    score: Callable[[Any], dict] | None = None
    config_hash: Callable[[Any], str] | None = None
    pinned: Callable[[Any], Any] | None = None
    conformance_population: Callable[..., Any] | None = None
    needs_gpu: bool | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum() or not self.name.islower():
            raise ValueError(f"Scorer name {self.name!r} must match [a-z0-9_]+")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError(f"Scorer {self.name}: version must be an int >= 1")
        if self.kind not in SCORER_KINDS:
            raise ValueError(f"Scorer {self.name}: kind must be one of {sorted(SCORER_KINDS)}, got {self.kind!r}")
        for f in ("label_kinds", "protocol_keys", "param_keys", "head_families", "losses", "input_keys", "output_keys"):
            object.__setattr__(self, f, frozenset(getattr(self, f)))
        if self.kind == "probe":
            if self.head is None or self.loss is None or not self.train or not self.head_families:
                raise ValueError(f"Scorer {self.name}: kind 'probe' needs train, head, loss and head_families")
        elif self.head is not None or self.train or self.head_families or self.losses:
            raise ValueError(f"Scorer {self.name}: kind {self.kind!r} trains no probe; train, head, head_families and losses must be empty")
        if self.family is not None and (not self.family or not self.family.replace("_", "").isalnum() or not self.family.islower()):
            raise ValueError(f"Scorer {self.name}: family must match [a-z0-9_]+")

    # ------------------------------------------------------------------ queries

    @property
    def is_probe(self) -> bool:
        return self.kind == "probe"

    @property
    def gpu(self) -> bool:
        """Whether the scorer's unit asks for a GPU (``needs_gpu``; probes by default)."""
        return self.is_probe if self.needs_gpu is None else bool(self.needs_gpu)

    def families(self) -> frozenset[str]:
        """The names a metric may be declared for to serve this scorer: itself and its ``family``."""
        return frozenset({self.name, self.family or self.name})

    def driver(self) -> Callable[[Any], dict]:
        """``score`` when set, else ``run_<name>`` of the declaring module (imported on call; may import torch)."""
        if self.score is not None:
            return self.score
        return getattr(importlib.import_module(self.module), f"run_{self.name}")

    def hash_config(self, cfg: Any) -> str:
        """The second slot of ``eval_hash`` for ``cfg``: ``config_hash`` when set, ``hashing.probe_hash`` for a
        probe, else the constant :func:`yavih.hashing.scorer_config_hash` of this scorer."""
        from yavih import hashing

        if self.config_hash is not None:
            return self.config_hash(cfg)
        if self.is_probe:
            return hashing.probe_hash(cfg)
        return hashing.scorer_config_hash(self.name, self.version)

    def pinned_config(self, bench: Any = None) -> Any:
        """The standard configuration: ``pinned`` when set, ``ProbeConfig.pinned(name, bench)`` for a probe, else
        :class:`ScorerConfig`."""
        if self.pinned is not None:
            return self.pinned(bench)
        if self.is_probe:
            from yavih.probes.config import ProbeConfig  # torch-free

            return ProbeConfig.pinned(self.name, bench)
        return ScorerConfig(scorer=self.name, version=self.version, kind=self.kind)


_LOCK = threading.RLock()
_REGISTRY: dict[str, Scorer] | None = None
_EXTRA: dict[str, Scorer] = {}
_FOLDER: dict[str, Scorer] = {}


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


def _from_module(modname: str) -> Scorer:
    mod = importlib.import_module(modname)
    r = getattr(mod, "SCORER", None)
    if not isinstance(r, Scorer):
        raise TypeError(f"{modname} does not export SCORER = yavih.readouts.registry.Scorer(...)")
    return r


def _load() -> dict[str, Scorer]:
    global _REGISTRY
    with _LOCK:
        if _REGISTRY is not None:
            return _REGISTRY
        reg: dict[str, Scorer] = {}
        for modname in BUILTIN:
            r = _from_module(modname)
            reg[r.name] = r
        for ep_name, value in _entry_point_modules():
            modname = value.split(":", 1)[0]
            try:
                r = _from_module(modname)
            except Exception as e:  # noqa: BLE001 - a broken plugin must name itself, not break the core
                import warnings

                warnings.warn(f"scorer entry point {ep_name!r} ({value}) failed to load: {type(e).__name__}: {e}", stacklevel=2)
                continue
            if r.name in reg:
                import warnings

                warnings.warn(f"scorer entry point {ep_name!r} redefines scorer {r.name!r}; ignored", stacklevel=2)
                continue
            reg[r.name] = r
        for name, r in _EXTRA.items():
            reg.setdefault(name, r)
        for name, r in _FOLDER.items():
            reg.setdefault(name, r)
        _REGISTRY = reg
        return reg


def register(scorer: Scorer, *, replace: bool = False) -> Scorer:
    """Add a scorer at run time (tests, an in-process plugin). Entry points are the persistent way."""
    with _LOCK:
        reg = _load()
        if scorer.name in reg and not replace:
            raise ValueError(f"scorer {scorer.name!r} already registered")
        reg[scorer.name] = scorer
        _EXTRA[scorer.name] = scorer
        return scorer


def register_folder(scorer: Scorer) -> Scorer:
    """Register a benchmark folder's own scorer (``eval.py`` ``SCORER``) under its name. A folder scorer may be
    re-registered (the folder is loaded again); it may never shadow a built-in, entry-point or run-time scorer."""
    with _LOCK:
        reg = _load()
        if scorer.name in reg and scorer.name not in _FOLDER:
            raise ValueError(f"scorer {scorer.name!r} is already a registered scorer; a folder scorer must carry the folder's own name")
        reg[scorer.name] = scorer
        _FOLDER[scorer.name] = scorer
        return scorer


def reset() -> None:
    """Forget run-time and folder registrations and rescan (tests)."""
    global _REGISTRY
    with _LOCK:
        _REGISTRY = None
        _EXTRA.clear()
        _FOLDER.clear()


def get(name: str) -> Scorer:
    try:
        return _load()[name]
    except KeyError:
        raise UnknownScorer(f"unknown scorer {name!r}; registered: {names()}") from None


def names() -> tuple[str, ...]:
    return tuple(_load())


def all_scorers() -> dict[str, Scorer]:
    return dict(_load())
