"""``crema_d``'s own synthetic stand-in for ``yavih bench trial --synthetic``: the core ``clf`` fixture, re-cut into
five folds and scored with this folder's ``eval.py`` (HEAR's ``top1_acc`` metrics), so the folds driver and the
folder metrics run in every stage, not only in the conformance stage.

``make_stand_in`` builds the core fixture with :func:`yavih.testing.make_synthetic_like` (the real folder's name,
stream and info keys over tiny generated wavs), then deals each class's rows round-robin to ``fold00`` … ``fold04``,
copies ``eval.py`` and ``hear_score.py`` next to it and takes the real folder's ``metrics`` block. The oracle and
random test bridges are the core's. Module scope: stdlib and yavih core (pyyaml through the core).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import shutil
from pathlib import Path

import yaml

FOLDS = ["fold00", "fold01", "fold02", "fold03", "fold04"]
_COPIED = ("eval.py", "hear_score.py")


def make_stand_in(work, real, *, n_eval: int, n_train: int) -> Path:
    from yavih import testing
    from yavih.freeze import csv_bytes, gzip_bytes

    # one population of n_train + n_eval rows (at least two per fold), split into five folds below
    folder = Path(testing.make_synthetic_like(work, real, n_eval=max(n_eval, 10), n_train=max(n_train, 10)))
    text = gzip.decompress((folder / "manifest.csv.gz").read_bytes()).decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    k, lab = header.index("split"), header.index("label")
    seen: dict[str, int] = {}
    for r in body:  # each class dealt round-robin over the folds, so every fold holds every class
        n = seen.get(r[lab], 0)
        r[k] = FOLDS[n % len(FOLDS)]
        seen[r[lab]] = n + 1
    data = csv_bytes(header, body)
    (folder / "manifest.csv.gz").write_bytes(gzip_bytes(data))

    path = folder / "benchmark.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    counts = {f: sum(1 for r in body if r[k] == f) for f in FOLDS}
    spec["manifest"]["sha256"] = hashlib.sha256(data).hexdigest()
    spec["splits"].update(train=[], eval=[], floor=[], folds=list(FOLDS), unused=[])
    spec["budget"]["clips"] = dict(counts)
    up = spec["manifest"].get("upstream") or {}
    if up.get("curation"):
        up["curation"]["per_split"] = {f: {"listed": n, "staged": n, "missing": 0} for f, n in counts.items()}
        up["curation"]["total_upstream"] = sum(counts.values()) + sum((up["curation"].get("excluded") or {}).values())
    real_metrics = dict(real.spec.raw["metrics"])
    real_metrics.pop("notes", None)
    spec["metrics"] = real_metrics
    path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for name in _COPIED:
        shutil.copyfile(Path(real.folder) / name, folder / name)
    return folder
