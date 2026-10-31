"""Report: index, the per-model lm-evaluation-harness table and
the comparability columns (keyed by ``eval_hash``; rows ``(model, reading)``), tables, result-schema migration, the results store (``store``) and joins
through a benchmark folder's ``eval.py`` (``join``; the authors' scorer lives only there, reached through
``yavih.external.require``). stdlib only.

``migrate`` is NOT re-exported as a name: it would shadow the ``yavih.report.migrate`` submodule, the
``RESULT_SCHEMA_VERSION`` re-exporter (``from yavih.report import migrate`` must
yield the module)."""

from __future__ import annotations

from yavih.report.compare import (
    Report,
    ReportError,
    build_report,
    column_key,
    comparable,
    lm_eval_tables,
    report,
    row_key,
    select_newest,
)
from yavih.report.index import build_index, iter_moved_aside, iter_results

__all__ = [
    "report",
    "Report",
    "ReportError",
    "build_report",
    "lm_eval_tables",
    "column_key",
    "row_key",
    "select_newest",
    "comparable",
    "build_index",
    "iter_results",
    "iter_moved_aside",
]
