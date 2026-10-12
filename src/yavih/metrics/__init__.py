"""Metrics. Importing this package never imports scikit-learn or torch: ``REGISTRY`` holds lazy
callables and the metric modules are pure numpy. ``interval`` is the driver's entry point for
``metrics.ci`` → ``result.ci95``."""

from __future__ import annotations

from yavih.metrics.intervals import bootstrap_clip, interval, pooled_binomial, wilson
from yavih.metrics.registry import (
    CI_KINDS,
    PATTERNS,
    REGISTRY,
    LazyFn,
    MetricNotAllowed,
    MetricPattern,
    MetricSpec,
    UnknownMetric,
    expand_sets,
    protocol_keys_for,
    register,
    resolve,
    resolve_names,
)

__all__ = [
    "REGISTRY",
    "PATTERNS",
    "CI_KINDS",
    "MetricSpec",
    "MetricPattern",
    "LazyFn",
    "UnknownMetric",
    "MetricNotAllowed",
    "register",
    "resolve",
    "resolve_names",
    "expand_sets",
    "protocol_keys_for",
    "interval",
    "wilson",
    "bootstrap_clip",
    "pooled_binomial",
]
