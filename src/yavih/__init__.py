"""yavih — AI-native frozen-feature benchmark harness.

Re-exports the core API and exposes the torch-dependent subpackages
(``probes``, ``readouts``, ``report``, ``plan``) as PEP 562 lazy attributes so that
``import yavih`` never pulls torch or any optional extra.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from yavih.benchmark import Benchmark
from yavih.bridge import BenchmarkView, Bridge, MediaUndecodable, NotApplicable, grid_pos
from yavih.io import FeatureReader, FeatureWriter, validate_features
from yavih.sample import RawStream, Sample
from yavih.versions import __version__

__all__ = [
    "Benchmark",
    "BenchmarkView",
    "Sample",
    "RawStream",
    "Bridge",
    "FeatureWriter",
    "FeatureReader",
    "validate_features",
    "NotApplicable",
    "MediaUndecodable",
    "grid_pos",
    "__version__",
    # lazy subpackages (PEP 562)
    "probes",
    "readouts",
    "report",
    "plan",
]

_LAZY_SUBPACKAGES = frozenset({"probes", "readouts", "report", "plan"})

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yavih import plan, probes, readouts, report  # noqa: F401


def __getattr__(name: str) -> Any:
    # Deferred so the model env (which has no torch) can `import yavih`.
    if name in _LAZY_SUBPACKAGES:
        return importlib.import_module(f"yavih.{name}")
    raise AttributeError(f"module 'yavih' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBPACKAGES)
