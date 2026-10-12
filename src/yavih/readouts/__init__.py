"""Readouts. ``[probe]`` extra: ``clf/seq/grid/box`` MAY import torch at
module scope; ``base`` stays torch-free and imports the readout module lazily once the
refusal checks have passed. Entry point ``probe()``."""

from __future__ import annotations

from yavih.readouts.base import (
    RESULT_SCHEMA_VERSION,
    ProbeInterrupted,
    ProbeRefused,
    ProbeResult,
    ReadoutContext,
    ResultSchemaError,
    check_result,
    probe,
    probe_not_applicable,
    request_stop,
    result_dir,
    write_result,
)

__all__ = [
    "probe",
    "probe_not_applicable",
    "request_stop",
    "ProbeResult",
    "ProbeRefused",
    "ProbeInterrupted",
    "ResultSchemaError",
    "ReadoutContext",
    "RESULT_SCHEMA_VERSION",
    "check_result",
    "result_dir",
    "write_result",
]
