"""Table rendering: md prints view 1 first — per model the lm-evaluation-harness table
``benchmark | reading | version | eval_hash | readout | metric | value | ci95 | floor``
(:func:`render_lm_eval`) — then view 2, the comparability columns ``(benchmark, eval_hash)`` with rows
``(model, reading)`` (``~~value~~`` for greyed rows); csv is the long format of view 2
(``flag=below_floor``; ``eval_hash``, ``model``, ``reading``, ``describe_sha256`` and
``features_fingerprint`` per row); json carries both. ``value (floor)`` and CI per metric; ``mean ± std`` with
``n_seeds``; ``MISSING`` cells. A row of a capped run (``plan build --sample N``) is marked ``partial``: the
``(partial)`` suffix on its values in md (both views; ``partial`` also in view 2's flags), the ``partial`` column and
flag in csv, ``partial: true`` in json. A suite completeness appendix (``report(suite=...)``) is a trailing md section
and part of json; csv stays the one long table of view 2 (as for the superseded, join and warning sections).
stdlib only; renders the duck-typed ``Report`` of :mod:`yavih.report.compare`
(never imports it, to keep the import graph acyclic)."""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

__all__ = ["render", "render_md", "render_csv", "render_json", "render_lm_eval", "render_suite_md", "LM_EVAL_HEADER", "format_value",
           "format_number", "format_coverage", "MISSING", "PARTIAL_MARK"]

#: Columns of the per-model view, in order.
LM_EVAL_HEADER: tuple[str, ...] = ("benchmark", "reading", "version", "eval_hash", "readout", "metric", "value", "ci95", "floor")
_DASH = "—"

MISSING = "MISSING"

_CSV_FIELDS = [
    "benchmark", "eval_hash", "protocol_hash", "probe_hash", "model", "reading", "describe_sha256", "features_fingerprint",
    "bridge_name", "status", "seed", "n_seeds", "coverage_train", "coverage_eval", "metric", "kind", "value", "std", "floor",
    "ci_lo", "ci_hi", "chance", "partial", "flag", "waived",
]
#: The suffix on every value of a capped run's row (``plan build --sample N``).
PARTIAL_MARK = "(partial)"
#: The mark of a row whose features passed validation with a waived check: shown next to the value with a
#: footnote naming the codes and reasons; the row is never greyed, excluded or flagged for it.
WAIVED_MARK = "†"


def waived_text(waived: Any) -> str:
    """``E_DUPLICATE_CLIPS (digitally silent audio; describe, 3 findings)``, ``;``-joined."""
    out = []
    for w in waived or []:
        if isinstance(w, dict) and w.get("code"):
            out.append(f"{w['code']} ({w.get('reason') or ''}; {w.get('source') or 'describe'}, {int(w.get('count') or 0)} finding(s))")
    return "; ".join(out)


def _waived_footnotes(rows: list[dict[str, Any]], label) -> list[str]:
    seen: list[str] = []
    for row in rows:
        text = waived_text(row.get("waived"))
        if text:
            line = f"{WAIVED_MARK} {label(row)}: waived {text}"
            if line not in seen:
                seen.append(line)
    return [f"- {_md_escape(x)}" for x in seen]


def format_number(x: Any) -> str:
    """Compact scalar formatting: ints as ints, floats to 4 significant digits, None as ``""``."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return str(x)
        if x == int(x) and abs(x) >= 1000:
            return str(int(x))
        return f"{x:.4g}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(format_number(v) for v in x) + "]"
    return str(x)


def format_coverage(x: Any) -> str:
    """Coverage fractions always with two decimals (``1.00``, ``0.95``)."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return ""
    return f"{float(x):.2f}"


def format_value(
    value: float | None,
    floor: float | None = None,
    ci: list[float] | None = None,
    *,
    std: float | None = None,
    n_seeds: int | None = None,
) -> str:
    """``value ± std (floor) [lo, hi]`` with the absent parts omitted. ``n_seeds`` is appended as ``×n`` when given and > 1."""
    if value is None:
        return ""
    text = format_number(value)
    if std is not None:
        text += f" ± {format_number(std)}"
    if n_seeds is not None and n_seeds > 1:
        text += f" ×{n_seeds}"
    if floor is not None:
        text += f" ({format_number(floor)})"
    if ci is not None and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        text += f" [{format_number(ci[0])}, {format_number(ci[1])}]"
    return text


def _cell_text(cell: dict[str, Any] | None) -> str:
    if not cell:
        return ""
    return format_value(cell.get("value"), cell.get("floor"), cell.get("ci"), std=cell.get("std"))


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _grey(text: str, greyed: bool) -> str:
    return f"~~{text}~~" if greyed and text else text


def _column_title(col: dict[str, Any]) -> str:
    bits = [f"### {col['benchmark']}"]
    if col.get("readout"):
        bits.append(f"{col['readout']}")
    if col.get("protocol_version") is not None:
        bits.append(f"v{col['protocol_version']}")
    bits.append(f"eval_hash `{str(col['eval_hash'])[:12]}`")
    title = " · ".join(bits)
    flags = col.get("flags") or []
    if flags:
        title += "  **[" + ", ".join(flags) + "]**"
    return title


def render_lm_eval(lm_eval: dict[str, list[dict[str, Any]]], *, show_all: bool = True) -> str:
    """View 1: one ``## <model>`` table per model, lm-eval style; ``eval_hash`` is the
    comparability key (two numbers of one benchmark compare iff it is equal). Without ``show_all`` a metric
    outside its benchmark's ``metrics.table`` (row ``in_table: false``) is left out."""
    out: list[str] = []
    for model, rows in lm_eval.items():
        if not show_all:
            rows = [r for r in rows if r.get("in_table", True)]
        out.append(f"## {_md_escape(model)}")
        out.append("")
        out.append("| " + " | ".join(LM_EVAL_HEADER) + " |")
        out.append("|" + "---|" * len(LM_EVAL_HEADER))
        for row in rows:
            if row.get("status") != "ok":
                value = f"_{row.get('status')}_" + (f" ({_md_escape(str(row.get('reason')))})" if row.get("reason") else "")
                ci = floor = _DASH
            else:
                value = format_number(row.get("value"))
                if row.get("std") is not None:
                    value += f" ± {format_number(row['std'])}"
                if (row.get("n_seeds") or 1) > 1:
                    value += f" (n_seeds {row['n_seeds']})"
                ci = format_number(row["ci95"]) if row.get("ci95") else _DASH
                floor = format_number(row["floor"]) if row.get("floor") is not None else _DASH
            if row.get("waived"):
                value += f" {WAIVED_MARK}"
            if row.get("partial"):
                value += f" {PARTIAL_MARK}"
            metric = str(row.get("metric", _DASH))
            if row.get("headline"):
                metric = f"**{metric}**"
            cells = [_md_escape(str(row.get("benchmark", ""))), _md_escape(str(row.get("reading") or "")), format_number(row.get("version")),
                     f"`{row.get('eval_hash') or ''}`" if row.get("eval_hash") else "", str(row.get("readout") or ""), metric, value, ci, floor]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        notes = _waived_footnotes(rows, lambda r: f"{r.get('benchmark')} {r.get('reading')}")
        if notes:
            out.extend(notes)
            out.append("")
    return "\n".join(out)


def render_md(report) -> str:
    out: list[str] = []
    out.append(f"# yavih report ({report.n_results} results, {len(report.columns)} columns)")
    out.append("")
    if not report.columns:
        out.append("_no result.json with DONE found_")
        out.append("")
    show_all = bool(getattr(report, "show_all", True))
    lm = getattr(report, "lm_eval", None) or {}
    if lm:
        out.append(render_lm_eval(lm, show_all=show_all))
        out.append("# Comparability columns")
        out.append("")
    rows_by_col: dict[str, list[dict[str, Any]]] = {}
    for row in report.rows:
        rows_by_col.setdefault(row["column"], []).append(row)
    for col in report.columns:
        out.append(_column_title(col))
        notes = []
        if col.get("population_differs"):
            notes.append("**population differs** (eval ids_sha256 disagree between rows)")
        if col.get("superseded"):
            notes.append("superseded protocol_version")
        if not col.get("standard"):
            notes.append("non-standard ProbeConfig: not comparable")
        if notes:
            out.append("; ".join(notes))
        out.append("")
        metrics = list(col.get("metrics") or [])
        if not show_all and col.get("table"):
            metrics = [m for m in metrics if m in set(col["table"])]  # metrics.table; the rest under --detail
        controls = list(col.get("controls") or [])
        headline = col.get("headline")
        header = ["model", "reading", "seeds", "cov eval"]
        header += [(f"**{m}**" if m == headline else m) for m in metrics]
        header += controls
        header += ["flags"]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for row in rows_by_col.get(col["id"], []):
            if row.get("missing"):
                cells = [row["model"], row["reading"], "", ""] + [MISSING] * (len(metrics) + len(controls)) + ["missing"]
                out.append("| " + " | ".join(_md_escape(c) for c in cells) + " |")
                continue
            greyed = bool(row.get("greyed"))
            seeds = str(row.get("n_seeds", "")) if row.get("seed") is None else f"seed {row['seed']}"
            cov = row.get("coverage") or {}
            cells = [
                _grey(_md_escape(row["model"]), greyed) + (f" {WAIVED_MARK}" if row.get("waived") else ""),
                _md_escape(row["reading"]),
                seeds,
                format_coverage(cov.get("eval")),
            ]
            if row.get("status") != "ok":
                reason = row.get("reason") or ""
                na = row.get("not_applicable") or {}
                if na.get("missing"):
                    reason = (reason + " " if reason else "") + "missing " + ",".join(map(str, na["missing"]))
                cells += [f"_{row['status']}_" + (f" ({_md_escape(reason)})" if reason else "")] + [""] * (len(metrics) + len(controls) - 1 if metrics or controls else 0)
            else:
                for m in metrics:
                    text = _grey(_cell_text((row.get("metrics") or {}).get(m)), greyed)
                    cells.append(f"{text} {PARTIAL_MARK}" if text and row.get("partial") else text)
                for c in controls:
                    cells.append(format_number((row.get("controls") or {}).get(c)))
            cells.append(", ".join(row.get("flags") or []))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        notes = _waived_footnotes(rows_by_col.get(col["id"], []), lambda r: f"{r.get('model')} {r.get('reading')}")
        if notes:
            out.extend(notes)
            out.append("")
    if report.superseded:
        out.append("## Superseded protocol versions")
        out.append("")
        for s in report.superseded:
            out.append(f"- `{s['benchmark']}` protocol_version {s['protocol_version']} (latest {s['latest']}): "
                       f"{s['n_results']} result(s) under eval_hash {', '.join(h[:12] for h in s['eval_hashes'])}")
        out.append("")
    if getattr(report, "joins", None):
        out.append("## Joins")
        out.append("")
        for kind, data in report.joins.items():
            out.append(f"### {kind}")
            out.append("")
            if isinstance(data, dict):
                out.extend(_render_join_md(data, show_all=show_all))
            out.append("")
    if getattr(report, "detail", None):
        out.append("## Detail: configuration and feature sets per row")
        out.append("")
        for column_id, by_row in report.detail.items():
            out.append(f"### {column_id}")
            out.append("")
            for row_key, d in by_row.items():
                out.extend(_render_row_detail(row_key, d))
    if report.warnings:
        out.append("## Warnings")
        out.append("")
        for w in report.warnings:
            where = f" ({w['column']})" if w.get("column") else ""
            out.append(f"- `{w.get('code', 'W')}`{where}: {w.get('message', '')}")
        out.append("")
    if getattr(report, "suite", None):
        out.extend(render_suite_md(report.suite))
    return "\n".join(out).rstrip("\n") + "\n"


def _join_dump(data: dict[str, Any]) -> list[str]:
    """Every key of a join output: scalars inline, a flat mapping as ``k=v`` pairs, a mapping of mappings
    (``{column: {row: value}}``) as a table."""
    out = []
    for k, v in data.items():
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            cols = list(v)
            rows = list(dict.fromkeys(r for x in v.values() for r in x))
            out += [f"- **{k}**:", "", "| | " + " | ".join(_md_escape(str(c)) for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
            out += ["| " + " | ".join([_md_escape(str(r))] + [format_number(v[c].get(r)) for c in cols]) + " |" for r in rows]
            out.append("")
        elif isinstance(v, dict):
            out.append(f"- **{k}**: " + ", ".join(f"{kk}={format_number(vv)}" for kk, vv in v.items()))
        else:
            out.append(f"- **{k}**: {format_number(v)}")
    return out


def _render_join_md(data: dict[str, Any], *, show_all: bool) -> list[str]:
    """One ``report --join`` section. When the folder's ``join()`` output carries ``table`` (``{"header": [...],
    "rows": [[...]], "notes": [...]}``, the folder's compact view of its numbers), that table plus the join's
    identity lines is shown, and everything else only with ``show_all`` (``--detail``); otherwise every key."""
    results = data.get("results") if isinstance(data.get("results"), dict) else {}
    table = results.get("table") if isinstance(results.get("table"), dict) else None
    if table is None:
        return _join_dump(data)
    out = [f"- **{k}**: {format_number(data.get(k))}" for k in ("model", "reading", "eval_hash", "path") if data.get(k) is not None]
    header = [str(h) for h in table.get("header") or []]
    out += ["", "| " + " | ".join(_md_escape(h) for h in header) + " |", "|" + "---|" * len(header)]
    for row in table.get("rows") or []:
        out.append("| " + " | ".join(_md_escape(format_number(c)) for c in row) + " |")
    notes = [str(n) for n in table.get("notes") or []]
    if notes:
        out += [""] + [f"- {_md_escape(n)}" for n in notes]
    for w in data.get("warnings") or []:
        if isinstance(w, dict):
            out.append(f"- `{w.get('code', 'W')}`: {_md_escape(str(w.get('message', '')))}")
    if show_all:
        out += ["", "Full join output:", ""]
        out += _join_dump({k: v for k, v in data.items() if k != "results"})
        out += _join_dump({k: v for k, v in results.items() if k != "table"})
    return out


def render_suite_md(suite: dict[str, Any]) -> list[str]:
    """The trailing ``## Suite completeness: <name>`` section: one row per suite ``(benchmark, reading)`` (a join
    row reads ``<a+v+av> (join)``), one status column per model, the counts, and the command for every missing
    join."""
    out = [f"## Suite completeness: {suite.get('name')}", ""]
    if suite.get("description"):
        out += [_md_escape(str(suite["description"])) + f" ({suite.get('origin')})", ""]
    rows = list(suite.get("rows") or [])
    models = list(dict.fromkeys(r["model"] for r in rows))
    if not models:
        return out + ["_no model in the report_", ""]
    cells: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in rows:
        key = (str(r["capability"]), str(r["benchmark"]), str(r["reading"]))
        cells.setdefault(key, {})[r["model"]] = str(r["status"]) + (f" ({r['note']})" if r.get("note") and r.get("note") != "join" else "")
    optional = {(str(r["benchmark"]), str(r["reading"])) for r in rows if not r.get("required", True)}
    joins = {(str(r["benchmark"]), str(r["reading"])) for r in rows if r.get("note") == "join"}
    header = ["capability", "benchmark", "reading", *models]
    out.append("| " + " | ".join(_md_escape(h) for h in header) + " |")
    out.append("|" + "---|" * len(header))
    for (cap, bench, rd), by_model in cells.items():
        label = rd + (" (join)" if (bench, rd) in joins else "") + (" (optional)" if (bench, rd) in optional else "")
        out.append("| " + " | ".join(_md_escape(c) for c in [cap, bench, label, *(by_model.get(m, "") for m in models)]) + " |")
    out.append("")
    counts = suite.get("counts") or {}
    if counts:
        out += ["counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()), ""]
    hints = [(r["model"], r["benchmark"], r["reading"], r["hint"]) for r in rows if r.get("hint")]
    for model, bench, rd, hint in hints:
        out.append(f"- {_md_escape(model)} {bench} {rd}: `{hint}`")
    if hints:
        out.append("")
    return out


def _render_row_detail(row_key: str, d: dict[str, Any]) -> list[str]:
    """``--detail`` for one row: the shown ``describe()`` and ``describe_sha256``, the older
    configurations (``W_BRIDGE_CHANGED``) with their ``describe()`` diff, the older feature sets
    (``W_FEATURES_DIFFER``) and the recorded warnings."""
    out = [f"**{_md_escape(row_key)}** — describe `{str(d.get('describe_sha256') or '?')[:12]}`, "
           f"features `{str(d.get('features_fingerprint') or '?')[:12]}`", ""]
    out.append("- describe(): `" + _md_escape(json.dumps(d.get("describe") or {}, sort_keys=True, default=str)) + "`")
    if d.get("rules_checked") is not None:
        out.append("- rules_checked: `" + _md_escape(json.dumps(d["rules_checked"], sort_keys=True, default=str)) + "`")
    if d.get("warnings"):
        out.append("- recorded warnings: " + ", ".join(f"`{w}`" for w in d["warnings"]))
    for oc in d.get("older_configs") or []:
        aside = "; moved aside to probe/_stale/" if oc.get("moved_aside") else ""
        out.append(f"- older configuration `{str(oc.get('describe_sha256') or '?')[:12]}` ({oc.get('created')}; seeds {oc.get('seeds')}; "
                   f"headline {format_number(oc.get('headline'))}{aside}):")
        for key, vals in (oc.get("diff") or {}).items():
            out.append(f"  - `{key}`: shown {_md_escape(json.dumps(vals.get('shown'), sort_keys=True, default=str))}, "
                       f"older {_md_escape(json.dumps(vals.get('other'), sort_keys=True, default=str))}")
    for fs in d.get("older_feature_sets") or []:
        out.append(f"- older feature set `{str(fs.get('fingerprint') or '?')[:12]}` ({fs.get('created')}; seeds {fs.get('seeds')}; "
                   f"headline {format_number(fs.get('headline'))})")
    out.append("")
    return out


def _csv_row(row: dict[str, Any], **extra: Any) -> dict[str, Any]:
    cov = row.get("coverage") or {}
    base = {
        "benchmark": row["benchmark"],
        "eval_hash": row["eval_hash"],
        "protocol_hash": row.get("protocol_hash") or "",
        "probe_hash": row.get("probe_hash") or "",
        "model": row["model"],
        "reading": row["reading"],
        "describe_sha256": row.get("describe_sha256") or "",
        "features_fingerprint": row.get("features_fingerprint") or "",
        "bridge_name": row.get("bridge_name", ""),
        "status": row["status"],
        "seed": "" if row.get("seed") is None else row["seed"],
        "n_seeds": row.get("n_seeds", 0),
        "coverage_train": format_coverage(cov.get("train")),
        "coverage_eval": format_coverage(cov.get("eval")),
        "metric": "", "kind": "", "value": "", "std": "", "floor": "", "ci_lo": "", "ci_hi": "", "chance": "",
        "flag": ";".join(row.get("flags") or []),
        "waived": ";".join(str(w.get("code")) for w in row.get("waived") or [] if isinstance(w, dict)),
        "partial": "true" if row.get("partial") else "false",
    }
    base.update(extra)
    return base


def render_csv(report) -> str:
    """Long format: one line per (row, metric) plus one per control; ``flag`` carries ``below_floor``
    (and any other flag, ``;``-joined); ``MISSING`` rows have ``value=MISSING``."""
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    cols = {c["id"]: c for c in report.columns}
    for row in report.rows:
        col = cols.get(row["column"], {})
        if row.get("missing"):
            w.writerow(_csv_row(row, value=MISSING))
            continue
        if row.get("status") != "ok":
            w.writerow(_csv_row(row, value=row["status"]))
            continue
        for m in col.get("metrics") or list((row.get("metrics") or {}).keys()):
            cell = (row.get("metrics") or {}).get(m) or {}
            ci = cell.get("ci") or [None, None]
            w.writerow(_csv_row(
                row, metric=m, kind="headline" if m == row.get("headline") else "metric",
                value=format_number(cell.get("value")), std=format_number(cell.get("std")),
                floor=format_number(cell.get("floor")), ci_lo=format_number(ci[0]), ci_hi=format_number(ci[1]),
                chance=format_number(cell.get("chance")),
            ))
        for c in col.get("controls") or []:
            w.writerow(_csv_row(row, metric=c, kind="control", value=format_number((row.get("controls") or {}).get(c))))
    return buf.getvalue()


def render_json(report) -> str:
    return json.dumps(report.to_json(), indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def render(report, fmt: str) -> str:
    if fmt == "md":
        return render_md(report)
    if fmt == "csv":
        return render_csv(report)
    if fmt == "json":
        return render_json(report)
    raise ValueError(f"unknown report format {fmt!r} (md | csv | json)")
