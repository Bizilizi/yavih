"""``nsynth``: the folder's own synthetic stand-in for ``yavih bench trial --synthetic``.

The core ``clf`` stand-in (``yavih.testing.make_synthetic_like``: this folder's name, readout and audio stream over
tiny generated wavs) scored with THIS folder's metrics: ``eval.py`` is copied in and the ``metrics`` block is the
real folder's (``pitch_acc`` headline, ``chroma_acc``, their floors), so HEAR's score functions run in every stage of
the trial, not only in the conformance stage. The core oracle and random bridges serve it (no ``write_bridge``).
Module scope: stdlib, yaml and yavih core.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def make_stand_in(work, real, *, n_eval, n_train):
    """The core clf stand-in with this folder's ``eval.py`` and ``metrics`` block."""
    from yavih.benchmark import Benchmark
    from yavih.doctables import replace_glance
    from yavih.testing import make_synthetic_like

    folder = Path(make_synthetic_like(work, real, n_eval=n_eval, n_train=n_train))
    shutil.copy2(Path(real.folder) / "eval.py", folder / "eval.py")
    path = folder / "benchmark.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    metrics = dict(real.spec.raw["metrics"])
    metrics["notes"] = ["Synthetic stand-in: the numbers carry over to nothing."]
    spec["metrics"] = metrics
    spec["title"] = f"Synthetic stand-in for {real.name}, scored with its eval.py (yavih bench trial --synthetic)"
    path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    readme = folder / "README.md"
    readme.write_text(replace_glance(readme.read_text(encoding="utf-8"), folder), encoding="utf-8")
    Benchmark.load(str(folder))  # the stand-in must load with the folder's metrics
    return folder
