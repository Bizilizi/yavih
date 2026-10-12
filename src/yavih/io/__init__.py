"""Feature WebDataset format: writer, reader, sidecars, shard planning, validation,
finalize, inspect, diff. numpy + stdlib ``tarfile`` only.
"""

from __future__ import annotations

from yavih.io.finalize import read_meta
from yavih.io.format import (
    FEATURE_FORMAT_VERSION,
    FeatureMetaError,
    FeatureShapeError,
    FeatureValueError,
    FormatError,
)
from yavih.io.reader import FeatureReader, RowView
from yavih.io.shards import plan_shards
from yavih.io.stage import StageError, stage_features
from yavih.io.validate import Finding, ValidationReport, validate_features
from yavih.io.writer import FeatureWriter

__all__ = [
    "FEATURE_FORMAT_VERSION",
    "FeatureWriter",
    "FeatureReader",
    "RowView",
    "plan_shards",
    "validate_features",
    "ValidationReport",
    "Finding",
    "read_meta",
    "stage_features",
    "StageError",
    "FeatureShapeError",
    "FeatureMetaError",
    "FeatureValueError",
    "FormatError",
]
