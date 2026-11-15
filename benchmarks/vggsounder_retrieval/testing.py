"""``vggsounder_retrieval``: the folder's own synthetic stand-in and test bridges for ``yavih bench trial --synthetic``.

The clips, the stand-in folder and the bridges are ``xmodal_synth.py`` (next to this file, byte-identical in every
shared-space folder): the stand-in keeps this folder's name, scorer, metrics, output contract and readout parameters;
the oracle bridge decodes each clip's code from each modality on its own, the random bridge embeds noise. This file
adds what the class-level metrics read: a ``labels_av`` column (``readout_params.classes_column``) whose classes are
contiguous bands of the clip code (``N_CLASSES`` bands; every third clip also carries its neighbour band), so the
oracle's nearest clips share its class.
Module scope: stdlib, numpy, yaml, yavih core; writing the media needs PyAV.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import sys
from pathlib import Path


def _sibling(name: str):
    """A file next to this one, imported by path (keyed on its content, like yavih loads eval.py)."""
    path = Path(__file__).resolve().with_name(name)
    data = path.read_bytes()
    mod_name = f"yavih_folder_{path.parent.name}_{path.stem}_{hashlib.sha256(data).hexdigest()[:12]}"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
    return sys.modules[mod_name]


synth = _sibling("xmodal_synth.py")
BRIDGES = {"oracle": "OracleRetrievalBridge", "random": "RandomRetrievalBridge", "leaky": "LeakyRetrievalBridge"}


N_CLASSES = 4


def _classes(i: int, n: int) -> str:
    band = min(N_CLASSES - 1, int((i + 0.5) / n * N_CLASSES))
    names = [f"band{band}"] + ([f"band{(band + 1) % N_CLASSES}"] if i % 3 == 0 else [])
    return ";".join(names)


def make_stand_in(work, real, *, n_eval, n_train):
    """The stand-in folder (``xmodal_synth.make_stand_in``) over ``n_eval`` synthetic clips, with a ``labels_av``
    column added to its manifest (and its sha256 and columns to its yaml)."""
    import yaml

    dst = synth.make_stand_in(work, real, n_eval=n_eval, n_train=n_train)
    column = str((yaml.safe_load((Path(real.folder) / "benchmark.yaml").read_text(encoding="utf-8"))["readout_params"])["classes_column"])
    lines = gzip.decompress((dst / "manifest.csv.gz").read_bytes()).decode("utf-8").splitlines()
    n = len(lines) - 1
    data = ("\n".join([lines[0] + "," + column] + [f"{line},{_classes(i, n)}" for i, line in enumerate(lines[1:])]) + "\n").encode("utf-8")
    (dst / "manifest.csv.gz").write_bytes(gzip.compress(data, mtime=0))
    y = yaml.safe_load((dst / "benchmark.yaml").read_text(encoding="utf-8"))
    sha = hashlib.sha256(data).hexdigest()
    y["manifest"]["sha256"] = y["manifest"]["upstream"]["sha256"] = sha
    y["manifest"]["columns"][column] = {"type": "str", "role": "extra"}
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, width=120, allow_unicode=True), encoding="utf-8")
    return dst


def write_bridge(path, kind, *, D, bench_folder):
    """The stand-in's ``oracle`` / ``random`` bridge file (``leaky``: the one the isolation gate refuses); ``D`` is
    ignored (the shared space has 9 dimensions)."""
    if kind not in BRIDGES:
        raise ValueError(f"write_bridge: kind must be one of {sorted(BRIDGES)}, got {kind!r}")
    return synth.bridge_source(path, BRIDGES[kind], kind=kind, synth=Path(bench_folder) / "xmodal_synth.py")
