"""``avrobustbench``: the folder's own synthetic stand-in for ``yavih bench trial --synthetic``.

The core clf fixture (``yavih.testing.make_synthetic_like``) writes the media and the folder; this module turns its
eval rows into this benchmark's shape, so that the folder's own metrics (``eval.py``: per-condition accuracy, ``macc``,
``alpha`` / ``rho``, the source-clip bootstrap) run in every trial stage: each eval row becomes one condition of a
source clip, ``clip_id = val_<source>#<condition>`` over the 16 ``protocol.conditions``, all conditions of a source
sharing its label, plus the real folder's ``protocol`` keys, ``condition`` column, metrics block and ``eval.py``.
The core oracle / random bridges serve it (no ``write_bridge``). Module scope: stdlib, yaml, yavih core.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import shutil
from pathlib import Path

import yaml

from yavih.freeze import csv_bytes, gzip_bytes


def make_stand_in(work, real, *, n_eval, n_train):
    """The stand-in folder: ``ceil(n_eval / 16) * 16`` eval rows (at least 4 source clips x 16 conditions)."""
    from yavih.doctables import replace_glance
    from yavih.testing import make_synthetic_like

    conditions = list(real.spec.protocol["conditions"])
    n_src = max(4, -(-int(n_eval) // len(conditions)))
    folder = Path(make_synthetic_like(work, real, n_eval=n_src * len(conditions), n_train=max(int(n_train), 8)))
    spec = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    classes = [r["name"] for r in csv.DictReader(io.StringIO((folder / "classes.csv").read_text(encoding="utf-8")))]

    with gzip.open(folder / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    header = [c for c in rows[0] if c != "label"]
    header = ["clip_id", "split", "label", "condition", *[c for c in header if c not in ("clip_id", "split")]]
    k = 0
    for r in rows:
        if r["split"] in spec["splits"]["eval"]:
            src, cond = divmod(k, len(conditions))
            r["clip_id"], r["condition"] = f"val_{src:04d}#{conditions[cond]}", conditions[cond]
            r["label"] = classes[src % len(classes)]
            k += 1
        else:
            r["clip_id"], r["condition"] = f"{r['clip_id']}#clean", "clean"
    data = csv_bytes(header, [[r[c] for c in header] for r in rows])
    (folder / "manifest.csv.gz").write_bytes(gzip_bytes(data))
    spec["manifest"]["sha256"] = hashlib.sha256(data).hexdigest()
    spec["manifest"]["upstream"]["sha256"] = spec["manifest"]["sha256"]
    spec["manifest"]["columns"] = {"condition": {"type": "str", "role": "extra"}, **spec["manifest"]["columns"]}
    for key in ("severity", "conditions", "corruption_groups"):
        spec["protocol"][key] = real.spec.protocol[key]
    spec["protocol"]["views"] = "rows"
    m = real.spec.raw["metrics"]
    spec["metrics"] = {key: m[key] for key in ("headline", "list", "table", "floors", "controls", "chance", "ci", "bootstrap")
                       if key in m}
    spec["metrics"]["chance"] = {key: round(1.0 / len(classes), 6) for key in m.get("chance", {})}
    shutil.copy2(Path(real.folder) / "eval.py", folder / "eval.py")
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    readme = folder / "README.md"
    readme.write_text(replace_glance(readme.read_text(encoding="utf-8"), folder), encoding="utf-8")
    return folder
