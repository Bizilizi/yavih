"""Table rendering: md prints view 1 first — per model the lm-evaluation-harness table
``benchmark | reading | version | eval_hash | readout | metric | value | ci95 | floor``
(:func:`render_lm_eval`) — then view 2, the comparability columns ``(benchmark, eval_hash)`` with rows
``(model, reading)`` (``~~value~~`` for greyed rows); csv is the long format of view 2
(``flag=below_floor``; ``eval_hash``, ``model``, ``reading``, ``describe_sha256`` and
``features_fingerprint`` per row); json carries both. ``value (floor)`` and CI per metric; ``mean ± std`` with
``n_seeds``; ``MISSING`` cells.
stdlib only; renders the duck-typed ``Report`` of :mod:`yavih.report.compare`
(never imports it, to keep the import graph acyclic)."""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

__all__ = ["render", "render_md", "render_csv", "render_json", "render_lm_eval", "LM_EVAL_HEADER", "format_value", "format_number",
           "format_coverage", "MISSING"]

#: Columns of the per-model view, in order.
LM_EVAL_HEADER: tuple[str, ...] = ("benchmark", "reading", "version", "eval_hash", "readout", "metric", "value", "ci95", "floor")
_DASH = "—"

MISSING = "MISSING"

_CSV_FIELDS = [
    "benchmark", "eval_hash", "protocol_hash", "probe_hash", "model", "reading", "describe_sha256", "features_fingerprint",
    "bridge_name", "status", "seed", "n_seeds", "coverage_train", "coverage_eval", "metric", "kind", "value", "std", "floor",
    "ci_lo", "ci_hi", "chance", "flag", "waived",
]
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


def render_lm_eval(lm_eval: dict[str, list[dict[str, Any]]]) -> str:
    """View 1: one ``## <model>`` table per model, lm-eval style; ``eval_hash`` is the
    comparability key (two numbers of one benchmark compare iff it is equal)."""
    out: list[str] = []
    for model, rows in lm_eval.items():
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
    lm = getattr(report, "lm_eval", None) or {}
    if lm:
        out.append(render_lm_eval(lm))
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
                    cells.append(_grey(_cell_text((row.get("metrics") or {}).get(m)), greyed))
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
                for k, v in data.items():
                    if isinstance(v, dict):
                        out.append(f"- **{k}**: " + ", ".join(f"{kk}={format_number(vv)}" for kk, vv in v.items()))
                    else:
                        out.append(f"- **{k}**: {format_number(v)}")
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
    return "\n".join(out).rstrip("\n") + "\n"


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
