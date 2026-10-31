"""Paired differences between two systems on the same clips (``yavih report paired``). numpy only.

Reads two samples files (``eval run --log-samples``) whose lines carry per-clip metric values (a folder scorer's
``prediction = {metric: value}``), keeps the ``eval`` rows of the clips both scored, and reports per metric the two
means, the mean difference (second minus first) and a seeded clip-level percentile bootstrap interval of that
difference (``yavih.metrics.intervals.bootstrap_clip`` over the per-clip differences: the clips are resampled once
for both systems, so the interval accounts for their pairing). A difference whose interval excludes 0 is reported as
``separated``. Distribution-level metrics (FD, KAD, IS) have no per-clip value and are not compared here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["read_samples", "paired", "paired_text", "PairedError"]


class PairedError(ValueError):
    """The two samples files cannot be compared (no per-clip values, several seeds, no common clip)."""


def read_samples(path: str | Path, *, seed: int | None = None) -> dict[str, dict[str, float]]:
    """``{clip_id: {metric: value}}`` from a samples file's ``eval`` lines with dict predictions (one seed)."""
    rows: dict[str, dict[str, float]] = {}
    seeds: set[int] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("role", "eval") != "eval" or not isinstance(rec.get("prediction"), dict):
                continue
            if seed is not None and int(rec.get("seed", 0)) != seed:
                continue
            seeds.add(int(rec.get("seed", 0)))
            rows[str(rec["clip_id"])] = {str(k): (float(v) if isinstance(v, (int, float)) else math.nan) for k, v in rec["prediction"].items()}
    if not rows:
        raise PairedError(f"{path}: no eval line with per-clip metric values (a folder scorer's `eval run --log-samples`)")
    if len(seeds) > 1:
        raise PairedError(f"{path}: lines of seeds {sorted(seeds)}; pick one with --seed")
    return rows


def paired(first: str | Path, second: str | Path, *, metrics: Sequence[str] | None = None, samples: int = 2000, seed: int = 0,
           sample_seed: int | None = None) -> dict[str, Any]:
    """Per-metric paired comparison of ``second`` against ``first`` on their common clips."""
    from yavih.metrics.intervals import bootstrap_clip

    a, b = read_samples(first, seed=sample_seed), read_samples(second, seed=sample_seed)
    common = sorted(set(a) & set(b))
    if not common:
        raise PairedError("the two samples files share no clip")
    names = list(metrics) if metrics else [m for m in a[common[0]] if m in b[common[0]]]
    out: dict[str, Any] = {"first": str(first), "second": str(second), "n_common": len(common), "only_first": len(set(a) - set(b)),
                           "only_second": len(set(b) - set(a)), "bootstrap": {"samples": int(samples), "seed": int(seed)}, "metrics": {}}
    for m in names:
        va = np.asarray([a[c].get(m, math.nan) for c in common], dtype=np.float64)
        vb = np.asarray([b[c].get(m, math.nan) for c in common], dtype=np.float64)
        ok = np.isfinite(va) & np.isfinite(vb)
        if not ok.any():
            raise PairedError(f"metric {m!r}: no clip with a finite value in both files")
        d = vb[ok] - va[ok]
        lo, hi = bootstrap_clip(d, samples=samples, seed=seed)
        out["metrics"][m] = {"first": float(va[ok].mean()), "second": float(vb[ok].mean()), "diff": float(d.mean()),
                             "ci95": [lo, hi], "n": int(ok.sum()), "separated": bool(lo > 0 or hi < 0)}
    return out


def paired_text(res: dict[str, Any]) -> str:
    lines = [f"first:  {res['first']}", f"second: {res['second']}",
             f"common clips: {res['n_common']} (only first {res['only_first']}, only second {res['only_second']}); "
             f"bootstrap {res['bootstrap']['samples']} resamples, seed {res['bootstrap']['seed']}", "",
             "| metric | first | second | second - first | 95 % CI | n | separated |", "|---|---|---|---|---|---|---|"]
    for m, r in res["metrics"].items():
        lines.append(f"| {m} | {r['first']:.4g} | {r['second']:.4g} | {r['diff']:+.4g} | [{r['ci95'][0]:+.4g}, {r['ci95'][1]:+.4g}] | {r['n']} | "
                     f"{'yes' if r['separated'] else 'no'} |")
    return "\n".join(lines) + "\n"
