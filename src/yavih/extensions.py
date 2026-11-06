"""What a benchmark folder extends: the parts of a protocol the folder owns and the core runs.

The core is read-only for a benchmark contribution: whatever a protocol needs that the core lacks is written in the
folder, through these extension points, and the core picks it up
(https://github.com/Bizilizi/yavih/blob/main/docs/extending-through-a-folder.md):

=================  =====================================================================================
``scorer``         ``eval.py`` ``SCORER``: the folder's own scorer (``probe``, ``stateless`` or ``direct``), with its
                   own head families and losses (``Scorer.build_head`` / ``Scorer.loss_fn``) or its own driver
``metrics``        ``eval.py`` ``METRICS``: metrics of its own (an authors' scorer, a transcription)
``hooks``          ``eval.py`` ``targets`` / ``score`` / ``join``: target construction, the authors' scorer, a
                   cross-reading join
``codecs``         ``eval.py`` ``CODECS``: output types of its own
``checks``         ``eval.py`` ``BRIDGE_CHECKS``: bridge-check gates (library gates by name, its own callables)
``eligible``       ``eval.py`` ``eligible``: an eligibility rule of its own on the validated features
``stand_in``       ``testing.py`` ``make_stand_in`` (and ``write_bridge``): its own synthetic stand-in for
                   ``yavih bench trial --synthetic``
``reader``         ``reader.py``: its own sample reader
``readings``       ``eval.py`` ``READINGS`` beyond the core a / v / av / t
``capability``     capability tags beyond ``yavih.suites.CAPABILITIES``
``label_kind``     a ``manifest.label.kind`` beyond the core kinds (its scorer reads the cells)
=================  =====================================================================================

``yavih bench info`` prints the one-line summary (:func:`extends_line`); ``yavih bench extensions [--all]`` the table.
Torch-free: the folder is loaded and its ``eval.py`` / ``testing.py`` imported as ``Benchmark.load`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["EXTENSION_POINTS", "DOC_URL", "folder_extensions", "extends_line"]

#: The extension points, in report order.
EXTENSION_POINTS: tuple[str, ...] = ("scorer", "metrics", "hooks", "codecs", "checks", "eligible", "stand_in", "reader", "readings",
                                     "capability", "label_kind")
DOC_URL = "https://github.com/Bizilizi/yavih/blob/main/docs/extending-through-a-folder.md"
_CORE_READINGS = frozenset({"a", "v", "av", "t"})


def folder_extensions(bench: Any) -> dict[str, Any]:
    """``{point: value}`` for every extension point the loaded ``bench`` uses (absent points are left out): ``scorer``
    ``{name, kind, family, head_families, losses, own_driver}`` (``own_driver``: its ``score`` drives scoring; it may
    still train with the core trainer); ``metrics`` names; ``hooks`` names; ``codecs`` names;
    ``checks`` gate names (``default`` when the folder declares none and its output contract implies the core's);
    ``eligible`` ``True``; ``stand_in`` ``["make_stand_in", "write_bridge"?]``; ``reader`` ``True``; ``readings``,
    ``capability``, ``label_kind`` the values beyond the core vocabularies."""
    from yavih.benchmark import LABEL_KINDS
    from yavih.lint import folder_readings, template_readings
    from yavih.readouts import registry
    from yavih.suites import CAPABILITIES
    from yavih.trial import folder_testing

    out: dict[str, Any] = {}
    folder = Path(bench.folder)
    spec = bench.spec
    mod = bench.eval_module()
    if spec.readout == bench.name:
        sc = registry.get(spec.readout)
        from yavih.probes.config import HEAD_FAMILIES, LOSSES

        out["scorer"] = {"name": sc.name, "kind": sc.kind, "family": sc.family,
                         "head_families": sorted(sc.head_families - HEAD_FAMILIES), "losses": sorted(sc.losses - LOSSES),
                         "own_driver": sc.score is not None}
    if mod is not None:
        metrics = getattr(mod, "METRICS", None) or {}
        if metrics:
            out["metrics"] = sorted(metrics)
        hooks = [h for h in ("targets", "score", "join") if callable(getattr(mod, h, None))]
        if hooks:
            out["hooks"] = hooks
        codecs = getattr(mod, "CODECS", None) or {}
        if codecs:
            out["codecs"] = sorted(codecs)
        checks = getattr(mod, "BRIDGE_CHECKS", None)
        if checks is not None:
            out["checks"] = [g if isinstance(g, str) else str(getattr(g, "__name__", "gate")) for g in checks]
        if callable(getattr(mod, "eligible", None)):
            out["eligible"] = True
    if "checks" not in out and getattr(spec.output_contract, "stream", None) is not None:
        out["checks"] = ["stream (default)"]
    tmod = folder_testing(bench)
    if tmod is not None and callable(getattr(tmod, "make_stand_in", None)):
        out["stand_in"] = ["make_stand_in"] + (["write_bridge"] if callable(getattr(tmod, "write_bridge", None)) else [])
    if (folder / "reader.py").is_file():
        out["reader"] = True
    tpl = folder / "bridge_template.py"
    readings = template_readings(tpl.read_text(encoding="utf-8")) if tpl.is_file() else set()
    own_readings = sorted((set(readings) | set(folder_readings(folder))) - _CORE_READINGS - {r for r in readings if r.endswith("-rt")})
    if own_readings:
        out["readings"] = own_readings
    own_caps = [c for c in spec.capability if c not in CAPABILITIES]
    if own_caps:
        out["capability"] = own_caps
    if spec.manifest.label.kind not in LABEL_KINDS:
        out["label_kind"] = spec.manifest.label.kind
    return out


def extends_line(ext: dict[str, Any], *, scorer: str | None = None) -> str:
    """One line for ``yavih bench info``: ``extends: scorer mybench (probe, own losses [...]); metrics [...]; ...`` or
    ``extends: nothing (core scorer <scorer>)``."""
    if not ext:
        return f"extends: nothing (core scorer {scorer})" if scorer else "extends: nothing"
    parts = []
    for point in EXTENSION_POINTS:
        if point not in ext:
            continue
        v = ext[point]
        if point == "scorer":
            bits = [v["kind"]] + ([f"family {v['family']}"] if v.get("family") else [])
            bits += [f"own head families {v['head_families']}"] if v.get("head_families") else []
            bits += [f"own losses {v['losses']}"] if v.get("losses") else []
            bits += ["own driver"] if v.get("own_driver") else []
            parts.append(f"scorer {v['name']} ({', '.join(bits)})")
        elif v is True:
            parts.append(point)
        elif isinstance(v, list):
            shown = ", ".join(map(str, v[:6])) + (f" (+{len(v) - 6} more)" if len(v) > 6 else "")
            parts.append(f"{point} {shown}")
        else:
            parts.append(f"{point} {v}")
    return "extends: " + "; ".join(parts)
