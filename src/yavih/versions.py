"""Every version constant of yavih.

This module is stdlib-only and torch-free on purpose: ``Benchmark.protocol_hash``
folds ``READOUT_VERSION`` in, and the hash must be computable in
the model env (which has no readout code installed) and in the probe env alike. Other modules re-export these names and
nothing else may define a version.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

UNKNOWN_VERSION: Final[str] = "0.0.0+unknown"


def _package_version() -> str:
    """The installed version: ``yavih/_version.py`` (written by setuptools-scm from the git tag at build / install
    time), else the distribution metadata, else :data:`UNKNOWN_VERSION` (a source tree on ``sys.path``)."""
    try:
        from yavih._version import __version__ as v  # type: ignore[import-not-found]  # generated

        if isinstance(v, str) and v:
            return v
    except Exception:  # noqa: BLE001 - absent in a bare checkout; a broken file is "unknown" too
        pass
    try:
        from importlib.metadata import version

        return version("yavih")
    except Exception:  # noqa: BLE001
        return UNKNOWN_VERSION


#: Package version (PEP 440, from the git tag ``vX.Y.Z``); provenance only, never hashed.
__version__: Final[str] = _package_version()

#: Bumped when the tar layout, member names, sidecar or ``_meta.json`` schema change. Sidecars, ``_meta.json`` and
#: ``NOT_APPLICABLE.json`` carry ``model`` + ``reading``; the sidecar's ``benchmark`` block carries ``input_hash`` and
#: its components, the ``yavih`` block the writer's version and commit. 5: the ``outputs`` block and per-key members
#: ``{row}.{key}.{ext}`` of a declared ``output_contract.schema``; a default-schema shard is written exactly as in 4.
FEATURE_FORMAT_VERSION: Final[int] = 5
#: Formats the readers accept: a format-4 set is a default-schema set and stays usable (no re-extraction).
READABLE_FEATURE_FORMATS: Final[tuple[int, ...]] = (4, 5)

#: Bumped when the ``benchmark.yaml`` schema changes. 3: ``fields:`` (file and cell storage) replaces ``streams:``; hashes unchanged.
BENCHMARK_SCHEMA_VERSION: Final[int] = 3

#: Bumped when the ``result.json`` schema changes (``eval_hash``, ``model``, ``reading``, ...). 4: ``benchmark.folder_source``.
RESULT_SCHEMA_VERSION: Final[int] = 4

#: Bumped when the results-store file schema changes (``results_<timestamp>.json``): one directory per ``model``,
#: entries keyed ``<benchmark>:<reading>``.
RESULTS_FILE_VERSION: Final[int] = 2

#: Bumped when the registry entry schema changes (``yavih.registry``: ``entries/<benchmark>/<eval_hash12>/<org>/<model>/<reading>.json``).
REGISTRY_ENTRY_VERSION: Final[int] = 1
#: Absolute tolerance per metric on its reported scale when a folder's ``metrics.tolerance`` names none: a tenth of a
#: point. Bounds what the conformance fixture accepts as "same protocol" (``yavih.conformance``) and what the report
#: shows as a tie (``yavih.report.compare``); not hashed (it changes no number).
DEFAULT_TOLERANCE: Final[float] = 0.001
#: Bumped when the unit record (``plan.jsonl``) changes. Extract units are keyed by ``input_hash``, probe and
#: report units by ``protocol_hash``.
PLAN_SCHEMA_VERSION: Final[int] = 3

class _ReadoutVersions(Mapping):
    """``{scorer: version}`` read from the scorer registry on first use (each ``yavih/readouts/<name>.py``, an
    entry-point scorer or a folder's ``eval.py`` declares its own ``VERSION``). Any change to a scorer's head,
    loss, floor, metric or protocol bumps its version and thereby forks ``protocol_hash``. Read-only;
    this module stays stdlib-only at import time."""

    def _map(self) -> dict[str, int]:
        from yavih.readouts import registry

        return {name: r.version for name, r in registry.all_scorers().items()}

    def __getitem__(self, key: str) -> int:
        return self._map()[key]

    def __iter__(self):
        return iter(self._map())

    def __len__(self) -> int:
        return len(self._map())

    def __repr__(self) -> str:
        return f"READOUT_VERSION({self._map()!r})"


class _ReadoutNames(Sequence):
    """The registered readout names, in registry order (the built-ins first: clf, seq, grid, box)."""

    def _names(self) -> tuple[str, ...]:
        from yavih.readouts import registry

        return registry.names()

    def __getitem__(self, i):  # type: ignore[override]
        return self._names()[i]

    def __len__(self) -> int:
        return len(self._names())

    def __contains__(self, x: object) -> bool:
        return x in self._names()

    def __eq__(self, other: object) -> bool:
        return tuple(self) == tuple(other) if isinstance(other, (tuple, list, Sequence)) else NotImplemented

    def __hash__(self) -> int:
        return hash(self._names())

    def __repr__(self) -> str:
        return f"READOUTS({self._names()!r})"


#: Per-readout version, from the readout registry.
READOUT_VERSION: Final[Mapping[str, int]] = _ReadoutVersions()

#: The ``probe_hash`` payload schema tag (the configuration hash of a ``probe`` scorer).
PROBE_SCHEMA: Final[str] = "yavih-probe-1"

#: The configuration-hash payload schema tag of a non-probe scorer (``hashing.scorer_config_hash``): the constant
#: ``{schema, scorer, version}`` that fills the second slot of ``eval_hash`` when nothing is trained.
SCORER_SCHEMA: Final[str] = "yavih-scorer-1"

#: The ``protocol_hash`` payload schema tag: ``protocol_hash = H(input_hash, scoring_hash)`` over the benchmark's
#: declaration only (manifest content, typed yaml views, declared versions; no source file since 5). Bumped when
#: either payload's schema or a typed view changes, and by a change to ``yavih.sample.iter_samples`` that changes
#: what is handed over.
PROTOCOL_SCHEMA: Final[str] = "yavih-protocol-5"

#: The ``describe_sha256`` payload schema tag: the digest covers ``describe()`` minus ``weights_ref``,
#: ``batch_tolerance``, ``batch_size`` and every ``input.<stream>.decoder`` (provenance).
DESCRIBE_SCHEMA: Final[str] = "yavih-describe-2"

#: ``GIT_HASH`` — the yavih source commit: ``git rev-parse HEAD`` of the checkout the package runs from,
#: else the commit setuptools-scm stamped into ``yavih/_version.py`` when the installed wheel was built, else ``None``.
#: Resolved lazily on first access (module ``__getattr__``) so importing this module never spawns ``git``.


def __getattr__(name: str):
    if name == "GIT_HASH":
        from yavih import buildinfo

        return buildinfo.commit()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def writer_stamp() -> dict[str, str | None]:
    """``{"version", "commit"}`` of this yavih: recorded by every writer (sidecars, ``_meta.json``) so a hash mismatch
    can name both ends (``E_HASH_INPUT``)."""
    return {"version": __version__, "commit": __getattr__("GIT_HASH")}

#: The readout names yavih knows (the ``benchmark.yaml`` ``readout`` key); equals ``READOUT_VERSION`` keys.
READOUTS: Final[Sequence[str]] = _ReadoutNames()

__all__ = [
    "__version__",
    "FEATURE_FORMAT_VERSION",
    "READABLE_FEATURE_FORMATS",
    "BENCHMARK_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "RESULTS_FILE_VERSION",
    "PLAN_SCHEMA_VERSION",
    "REGISTRY_ENTRY_VERSION",
    "DEFAULT_TOLERANCE",
    "READOUT_VERSION",
    "PROBE_SCHEMA",
    "SCORER_SCHEMA",
    "PROTOCOL_SCHEMA",
    "DESCRIBE_SCHEMA",
    "GIT_HASH",  # noqa: F822 - resolved lazily by the module __getattr__
    "READOUTS",
    "writer_stamp",
]
