"""Eligibility of a validated feature set for a scorer: the measured facts (:class:`EligibilityView`) and the
default verdict (:func:`requires_missing`, the ``output_contract.requires`` rules).

``yavih features validate`` measures the facts once over its seeded sample (which position axes and stream tags the
tokens carry, the grid-cell coverage, the scored seconds holding a timed token, the distance to an annotated frame)
and asks the scorer: ``Scorer.eligibility(view)`` when the scorer declares one, else :func:`requires_missing`. The
verdict is written to ``_meta.json`` and the probe never re-derives it. A scorer's hook returns the list of what is
missing (empty = eligible), or ``(missing, reasons)`` with one sentence per reason. Stdlib only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["EligibilityView", "requires_missing", "normalize_verdict"]

#: ``missing`` entries the default verdict explains with a sentence of their own (the rest are listed under "missing:").
EXPLAINED: frozenset[str] = frozenset({"tokens>1", "grid_cell_coverage", "t_bins", "annotated_frame"})


@dataclass(frozen=True)
class EligibilityView:
    """What a scorer may read to decide eligibility: the folder's declaration (``requires``, ``readout_params``,
    ``protocol``, ``output_schema``) and what validate measured on the sample (``N``, ``pos_mode``, ``pos_axes``,
    ``stream_tags``, the medians / mins ``grid_cell_coverage``, ``t_bins`` and ``annotated_frame_dt`` as
    ``{"median", "min"}`` or ``None``, ``output_keys`` the schema keys the feature set carries)."""

    scorer: str
    dense: bool
    N: int
    pos_mode: str
    readout_params: Mapping[str, Any]
    requires: Mapping[str, Any]
    protocol: Mapping[str, Any]
    pos_axes: frozenset[str] = frozenset()
    stream_tags: tuple[str, ...] = ()
    grid_cell_coverage: Mapping[str, float] | None = None
    t_bins: Mapping[str, float] | None = None
    annotated_frame_dt: Mapping[str, float | None] | None = None
    annot_second: float | None = None
    annot_tol: float = 0.0
    annot_stream: str | None = None
    target_region: Sequence[float] | None = None
    output_schema: Mapping[str, Any] | None = None
    output_keys: frozenset[str] = field(default_factory=frozenset)


def requires_missing(view: EligibilityView) -> tuple[list[str], list[str]]:
    """The default verdict: ``output_contract.requires`` evaluated on the measured facts. Returns ``(missing,
    reasons)``; ``missing`` empty means eligible."""
    missing: list[str] = []
    reasons: list[str] = []
    req, rp = view.requires, view.readout_params
    if view.dense and view.N == 1:
        missing.append("tokens>1")
        reasons.append(f"N == 1 (pooled features) cannot serve the dense scorer {view.scorer!r}")
    for axis in tuple(req.get("pos") or ()):
        if view.pos_mode == "none" or axis not in view.pos_axes:
            missing.append(f"pos.{axis}")
    required_streams = list(req.get("streams") or ())
    for key in ("visual_stream", "conditioning_stream"):
        if rp.get(key) and rp[key] not in required_streams:
            required_streams.append(rp[key])
    for sname in required_streams:
        if sname not in view.stream_tags:
            missing.append(f"stream:{sname}")
    g = req.get("grid")
    if isinstance(g, Mapping) and g.get("min_cell_coverage") is not None:
        cov = view.grid_cell_coverage
        need = float(g["min_cell_coverage"])
        if cov is None or cov["median"] < need:
            missing.append("grid_cell_coverage")
            frame = "the target region" if view.target_region is not None else "the full frame"
            got = "no positioned tokens" if cov is None else f"median {cov['median']:.2f}"
            reasons.append(f"grid coverage {got} < {need:g} over {frame}")
    tb = req.get("t_bins")
    if tb is not None:
        got_tb = view.t_bins
        if got_tb is None or got_tb["median"] < int(tb):
            missing.append("t_bins")
            reasons.append(f"scored seconds with a timed token: median {None if got_tb is None else got_tb['median']} < {int(tb)}")
    af = req.get("annotated_frame")
    if isinstance(af, Mapping) and view.annot_second is not None:
        dt = view.annotated_frame_dt
        tol = float(view.annot_tol or 0.0)
        if dt is None or dt.get("median") is None or float(dt["median"]) > tol:
            missing.append("annotated_frame")
            got = "no timed token" if dt is None or dt.get("median") is None else f"median {float(dt['median']):.2f} s"
            reasons.append(f"no {view.annot_stream!r} token within {tol:g} s of the annotated second {view.annot_second:g} s ({got} away): "
                           "the target is drawn on that frame")
    return missing, reasons


def normalize_verdict(verdict: Any) -> tuple[list[str], list[str]]:
    """A hook's return value as ``(missing, reasons)``: a list of missing items, or a ``(missing, reasons)`` pair."""
    if isinstance(verdict, tuple) and len(verdict) == 2:
        missing, reasons = verdict
        return [str(m) for m in missing], [str(r) for r in reasons]
    if isinstance(verdict, list | set | frozenset):
        return [str(m) for m in verdict], []
    raise TypeError(f"Scorer.eligibility must return a list of missing items or (missing, reasons), got {type(verdict).__name__}")
