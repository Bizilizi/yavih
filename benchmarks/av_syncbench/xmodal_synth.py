"""Synthetic cross-modal stand-ins and test bridges of the shared-space folders (``vggss_loc``, ``vggsound_retrieval``).

Byte-identical in every folder that uses it (a test pins the copies). Every clip carries one code ``z`` in ``(0, 1)``
(a lattice over the split, so codes are well separated): its video (32 x 32, 10 fps, 2 s, mpeg4) shows a square of
colour ``(255 z, 255 (1 - z), 0)`` on a gray background, at a position and size the clip id fixes (the localisation
box); its audio (16 kHz, 2 s) is a tone at ``300 + 1200 z`` Hz, a wav of its own or, with ``shared=True``, an AAC
track of the mp4. The shared space is :func:`code`, a Fourier embedding of ``z`` plus a saturation channel. The
oracle bridges decode ``z`` from each modality on its own (the colour of the square, the spectral peak); the leaky
ones derive a visual output from the audio (only ``bridge check``'s modality-isolation gate can see it); the random
ones embed noise. The localisation bridges emit one token per cell of a ``TOKEN_GRID`` grid over the frame.

:func:`make_stand_in` writes a stand-in of a real folder: the real folder's name, scorer, ``eval.py`` and sibling
modules, metrics, output contract and readout parameters over these clips. Module scope: stdlib, numpy, yaml, yavih
core; writing media needs PyAV.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from yavih.bridge import Bridge
from yavih.codecs import read_wav, wav_bytes
from yavih.hashing import rng_seed

__all__ = ["TOKEN_GRID", "code", "make_stand_in", "bridge_source", "OracleRetrievalBridge", "LeakyRetrievalBridge",
           "RandomRetrievalBridge", "OracleLocBridge", "LeakyLocBridge", "RandomLocBridge"]

FPS, SIZE, SECONDS, RATE = 10, 32, 2.0, 16000
GRAY = 128
TOKEN_GRID = 8
F_LO, F_SPAN = 300.0, 1200.0
#: protocol values a stand-in's 2 s, 32 x 32 clips need in place of the real folder's
PROTOCOL_OVERRIDES: dict[str, Any] = {"clip_seconds": SECONDS, "gt_size": 64, "annotated_second": 1.0}


def code(z: float, saturation: float = 1.0) -> np.ndarray:
    """The shared space: ``[cos(2πkz), sin(2πkz)]`` for ``k = 1..4`` scaled by ``saturation``, plus
    ``saturation * (0.5 + z / 2)`` (a channel that is zero on the gray background and varies across clips)."""
    k = np.arange(1, 5)
    return np.concatenate([saturation * np.cos(2 * np.pi * k * z), saturation * np.sin(2 * np.pi * k * z), [saturation * (0.5 + z / 2)]]).astype(np.float32)


def _z_of(index: int, n: int) -> float:
    return (index + 0.5) / n


def _box_of(clip_id: str) -> tuple[float, float, float, float]:
    """``(x1, y1, x2, y2)`` in fractions, on the ``TOKEN_GRID`` grid, 0.3 to 0.6 of the frame."""
    rng = np.random.default_rng(rng_seed(f"yavih-synth-xmodal-box:{clip_id}"))
    g = TOKEN_GRID
    while True:
        w, h = int(rng.integers(3, g)), int(rng.integers(3, g))
        if 0.3 <= w * h / (g * g) <= 0.6:
            break
    x, y = int(rng.integers(0, g - w + 1)), int(rng.integers(0, g - h + 1))
    return x / g, y / g, (x + w) / g, (y + h) / g


def _frame(z: float, box: tuple[float, float, float, float]) -> np.ndarray:
    img = np.full((SIZE, SIZE, 3), GRAY, dtype=np.uint8)
    x1, y1, x2, y2 = (int(round(v * SIZE)) for v in box)
    img[y1:y2, x1:x2] = (int(round(255 * z)), int(round(255 * (1 - z))), 0)
    return img


def _tone(z: float) -> np.ndarray:
    t = np.arange(int(SECONDS * RATE)) / RATE
    return (0.5 * np.sin(2 * np.pi * (F_LO + F_SPAN * z) * t)).astype(np.float32)


def _write_media(path: Path, z: float, box: tuple[float, float, float, float], *, audio_track: bool) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        v = c.add_stream("mpeg4", rate=FPS)
        v.width = v.height = SIZE
        v.pix_fmt = "yuv420p"
        v.options = {"qscale": "2"}
        a = None
        if audio_track:
            a = c.add_stream("aac", rate=RATE)
            a.layout = "mono"
        img = _frame(z, box)
        for _ in range(int(SECONDS * FPS)):
            for p in v.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in v.encode(None):
            c.mux(p)
        if a is not None:
            pcm = (_tone(z) * 32767).astype(np.int16)[None]
            af = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
            af.sample_rate = RATE
            for p in a.encode(af):
                c.mux(p)
            for p in a.encode(None):
                c.mux(p)


def _media_rows(media: Path, n: int, *, shared: bool) -> list[tuple[str, tuple[float, float, float, float], str, str]]:
    out = []
    for i in range(n):
        cid = f"test_{i:04d}"
        z, box = _z_of(i, n), _box_of(cid)
        rel_v, rel_a = f"test/{cid}.mp4", f"test/{cid}.wav"
        if not (media / rel_v).is_file():
            _write_media(media / rel_v, z, box, audio_track=shared)
        if not shared and not (media / rel_a).is_file():
            (media / rel_a).parent.mkdir(parents=True, exist_ok=True)
            (media / rel_a).write_bytes(wav_bytes(_tone(z), RATE))
        out.append((cid, box, rel_v, rel_a))
    return out


def _chance(names: Any, n: int, n_subsets: dict[str, int]) -> dict[str, float]:
    """``r@k...`` chance ``k / n`` over the stand-in's gallery (a subset metric ``..._<subset>``: its own gallery)."""
    out = {}
    for name in names:
        m = re.match(r"r@(\d+)", str(name))
        if m is None:
            continue
        sub = next((s for s in n_subsets if str(name).endswith(f"_{s}")), None)
        out[str(name)] = round(int(m.group(1)) / (n_subsets[sub] if sub else n), 6)
    return out


def make_stand_in(work: str | Path, real: Any, *, n_eval: int, n_train: int = 0, shared: bool = False, media_root: str | Path | None = None) -> Path:
    """Write ``<work>/<real.name>``: a stand-in of the real shared-space folder over ``n_eval`` synthetic test clips
    (``n_train`` is ignored: the scorers train nothing). It keeps the real folder's name, scorer, header, metrics,
    output contract (schema and isolation) and readout parameters, copies its ``eval.py`` and sibling modules, and
    adapts what the 2 s, 32 x 32 clips need: ``PROTOCOL_OVERRIDES``, the heatmap ``grid`` (``TOKEN_GRID``), the
    retrieval chance levels. A ``readout_params.boxes_column`` gets each clip's box; every ``readout_params.subsets``
    column flags every other row. ``shared``: the audio is a track of the mp4 (one file for both streams)."""
    real_dir = Path(real.folder)
    y_real = yaml.safe_load((real_dir / "benchmark.yaml").read_text(encoding="utf-8"))
    rp = dict(y_real.get("readout_params") or {})
    dst = Path(work) / str(y_real["name"])
    media = Path(media_root) if media_root is not None else Path(work) / "media" / str(y_real["name"])
    rows_media = _media_rows(media, int(n_eval), shared=shared)
    boxes_column = rp.get("boxes_column")
    subsets = dict(rp.get("subsets") or {})
    header = ["clip_id", "split", "rel_video"] + ([] if shared else ["rel_audio"]) + ([boxes_column] if boxes_column else []) + list(subsets.values())
    rows = []
    for i, (cid, box, rel_v, rel_a) in enumerate(rows_media):
        rows.append([cid, "test", rel_v] + ([] if shared else [rel_a]) + (["|".join(f"{v:.6f}" for v in box)] if boxes_column else [])
                    + [str(i % 2) for _ in subsets])
    data = ("\n".join(",".join(r) for r in [header] + rows) + "\n").encode("utf-8")
    columns: dict[str, Any] = {"rel_video": {"type": "str", "role": "media", "stream": ["video", "audio"] if shared else "video"}}
    if not shared:
        columns["rel_audio"] = {"type": "str", "role": "media", "stream": "audio"}
    if boxes_column:
        columns[boxes_column] = {"type": "str", "role": "label_aux"}
    for column in subsets.values():
        columns[column] = {"type": "int", "role": "extra"}
    n = len(rows)
    n_subsets = {name: sum(int(r[-len(subsets) + j]) for r in rows) for j, name in enumerate(subsets)} if subsets else {}
    if "grid" in rp:
        rp["grid"] = TOKEN_GRID
    protocol = {**dict(y_real.get("protocol") or {}), **{k: v for k, v in PROTOCOL_OVERRIDES.items() if k in (y_real.get("protocol") or {})}}
    metrics = dict(y_real["metrics"])
    metrics["chance"] = _chance((metrics.get("chance") or {}).keys(), n, n_subsets)
    metrics["bootstrap"] = {"samples": 200, "seed": 0}
    sha = hashlib.sha256(data).hexdigest()
    info_audio = ["duration_s", "sample_rate", "channels", "container", "bytes"] if shared else ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]
    y = {
        "schema_version": y_real["schema_version"], "name": y_real["name"], "aliases": [], "title": f"{y_real['title']} (synthetic stand-in)",
        "capability": list(y_real.get("capability") or []), "header": y_real.get("header"), "core": y_real.get("core"),
        "scorer": y_real["scorer"], "protocol_version": y_real["protocol_version"],
        "changelog": [{"version": y_real["protocol_version"], "date": "2026-11-06", "reason": "Synthetic stand-in (xmodal_synth.py).", "invalidates": False}],
        "source": {"kind": "derived", "urls": [], "licence": "synthetic", "bytes": 0, "sha256": {}, "notes": "xmodal_synth.py"},
        "budget": {"clips": {"test": n}, "media_gb": 0.0, "extract_min_per_1k_clips_ref": 0.1, "probe_minutes_ref": dict(y_real["budget"]["probe_minutes_ref"])},
        "manifest": {"file": "manifest.csv.gz", "sha256": sha, "classes": None, "classes_sha256": None, "key": "clip_id",
                     "label": {"column": None, "kind": "none"}, "columns": columns,
                     "upstream": {"source": "xmodal_synth.py", "sha256": sha, "rows": n, "note": "all rows synthetic", "upstream_files": {},
                                  "curation": {"total_upstream": n, "per_split": {"test": {"listed": n, "staged": n, "missing": 0}}, "excluded": {}}}},
        "splits": {"train": [], "eval": ["test"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95},
        "fields": {"video": {"storage": "file", "kind": "video", "media_column": "rel_video", "root": "media_root", "layout": "dir", "layout_params": {}, "required": True},
                   "audio": {"storage": "file", "kind": "audio", "media_column": "rel_video" if shared else "rel_audio", "root": "media_root", "layout": "dir",
                             "layout_params": {}, "required": True}},
        "targets": None,
        "data": {"roots": {"media_root": {"default": str(media), "description": "synthetic cross-modal media"}}, "tar_index": None, "derived": {}},
        "input_contract": {"handed_over": {"video": "the synthetic mp4 as stored (2 s, 10 fps, 32 x 32, mpeg4" + (", with its AAC tone track)" if shared else ")"),
                                           "audio": "the same mp4 (its AAC tone track)" if shared else "the synthetic wav as stored (16 kHz mono, 2 s)"},
                           "info": {"video": ["duration_s", "fps", "container", "bytes"], "audio": info_audio},
                           "facts": list((y_real.get("input_contract") or {}).get("facts") or []), "materialize": None},
        "output_contract": y_real["output_contract"], "external": [], "protocol": protocol, "metrics": metrics, "readout_params": rp,
    }
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "manifest.csv.gz").write_bytes(gzip.compress(data, mtime=0))
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, width=120, allow_unicode=True), encoding="utf-8")
    (dst / "README.md").write_text(f"# {y_real['name']} (synthetic stand-in)\n\nWritten by the folder's `testing.py` (`xmodal_synth.py`).\n", encoding="utf-8")
    for f in real_dir.glob("*.py"):
        if f.name not in ("make_manifest.py", "bridge_template.py", "testing.py"):
            shutil.copy2(f, dst / f.name)
    return dst


# ----------------------------------------------------------------------------- stand-in bridges


def _video_frame(sample: Any) -> np.ndarray:
    import av

    with av.open(io.BytesIO(sample.streams["video"].bytes())) as c:
        for f in c.decode(video=0):
            return f.to_ndarray(format="rgb24")
    raise ValueError("no video frame")


def _audio_samples(sample: Any) -> tuple[np.ndarray, int]:
    data = sample.streams["audio"].bytes()
    if data[:4] == b"RIFF":
        x, sr = read_wav(data)
        return np.asarray(x, dtype=np.float64).reshape(len(x), -1)[:, 0], int(sr)
    import av

    with av.open(io.BytesIO(data)) as c:
        parts = [f.to_ndarray().reshape(-1) for f in c.decode(audio=0)]
        sr = int(c.streams.audio[0].rate)
    return np.concatenate(parts).astype(np.float64), sr


def _z_from_audio(sample: Any) -> float:
    x, sr = _audio_samples(sample)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = float(np.argmax(spec[1:]) + 1) * sr / len(x)
    return float(np.clip((f - F_LO) / F_SPAN, 0.0, 1.0))


def _cell_colours(img: np.ndarray, grid: int) -> np.ndarray:
    """``float64 [grid * grid, 3]`` mean colour per cell, row-major."""
    h = img.shape[0] // grid
    return img.reshape(grid, h, grid, h, 3).mean(axis=(1, 3)).reshape(-1, 3)


def _code_of_colour(rgb: np.ndarray) -> np.ndarray:
    """A cell's code: the square's ``z`` from its red channel, saturation = how far from the gray background."""
    sat = float(np.clip((GRAY - rgb[2]) / GRAY, 0.0, 1.0))  # the square has B = 0, the background B = GRAY
    return code(float(rgb[0]) / 255.0, sat) if sat > 0.3 else code(0.0, 0.0)


def _z_from_video(sample: Any) -> float:
    cells = _cell_colours(_video_frame(sample), TOKEN_GRID)
    return float(cells[int(np.argmin(cells[:, 2])), 0] / 255.0)


class _XModalBase(Bridge):
    """Preprocessing source:
        none: a synthetic fixture; it runs no model, so there is no preprocessing to paste.
    """

    batch_size = 2
    outputs: tuple[str, ...] = ()

    def describe(self) -> dict:
        return {"model": f"synth-{type(self).__name__.lower()}", "reading": "av", "weights_sha256": None, "weights_ref": "-", "batch_size": self.batch_size,
                "views": 1, "outputs": list(self.outputs),
                "input": {"video": {"norm": {"source": "none: pixel values in [0, 1]"}, "decoder": {"lib": "pyav", "version": "stand-in"},
                                    "temporal": {"rule": "custom", "description": "the first frame"},
                                    "spatial": {"rule": "custom", "description": "the mean colour per cell of an 8 x 8 grid"}},
                          "audio": {"norm": {"source": "none: the spectral peak"}, "decoder": {"lib": "pyav", "version": "stand-in"},
                                    "temporal": {"rule": "custom", "description": "the whole track"},
                                    "front_end": {"sample_rate": RATE, "resample": "none", "features": "magnitude spectrum (spectral peak)"}}},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
                "args": dict(self.args)}

    def _used(self, s: Any) -> dict:
        info = s.streams["audio"].info
        n = int(info.get("n_samples") or round(float(info["duration_s"]) * float(info["sample_rate"])))
        return {"video": {"times": [0.0]}, "audio": {"samples": [[0, n]], "pad": [0, 0]}}

    def encode(self, samples: list[Any]) -> dict[str, Any]:
        outs = [self._one(s) for s in samples]
        res = {k: [o[k] for o in outs] for k in self.outputs}
        res["used"] = [self._used(s) for s in samples]
        return res

    def _one(self, s: Any) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError


class OracleRetrievalBridge(_XModalBase):
    """``audio_emb`` from the tone, ``video_emb`` from the square's colour: the same code, decoded independently."""

    outputs = ("audio_emb", "video_emb")

    def _one(self, s: Any) -> dict[str, Any]:
        return {"audio_emb": code(_z_from_audio(s)), "video_emb": code(_z_from_video(s))}


class LeakyRetrievalBridge(OracleRetrievalBridge):
    """``video_emb`` derived from the AUDIO: perfect retrieval, caught only by the isolation gate."""

    def _one(self, s: Any) -> dict[str, Any]:
        a = code(_z_from_audio(s))
        return {"audio_emb": a, "video_emb": a.copy()}


class RandomRetrievalBridge(_XModalBase):
    """Noise seeded by each stream's bytes."""

    outputs = ("audio_emb", "video_emb")

    @staticmethod
    def _noise(data: bytes, salt: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(salt.encode() + data).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal(9).astype(np.float32)

    def _one(self, s: Any) -> dict[str, Any]:
        return {"audio_emb": self._noise(s.streams["audio"].bytes(), "a"), "video_emb": self._noise(s.streams["video"].bytes(), "v")}


def _grid_pos(grid: int) -> np.ndarray:
    c = (np.arange(grid) + 0.5) / grid
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1).astype(np.float64)


class OracleLocBridge(_XModalBase):
    """One token per cell: the code of the cell's colour (gray cells: the zero code); ``audio_emb`` from the tone."""

    outputs = ("audio_emb", "video_tokens", "video_pos")

    def _one(self, s: Any) -> dict[str, Any]:
        cells = _cell_colours(_video_frame(s), TOKEN_GRID)
        return {"audio_emb": code(_z_from_audio(s)), "video_tokens": np.stack([_code_of_colour(c) for c in cells]), "video_pos": _grid_pos(TOKEN_GRID)}


class LeakyLocBridge(OracleLocBridge):
    """The tokens carry the audio code: the heatmap is honest, the tokens are not."""

    def _one(self, s: Any) -> dict[str, Any]:
        out = super()._one(s)
        out["video_tokens"] = out["video_tokens"] + 0.1 * out["audio_emb"][None]
        return out


class RandomLocBridge(_XModalBase):
    outputs = ("audio_emb", "video_tokens", "video_pos")

    def _one(self, s: Any) -> dict[str, Any]:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(s.streams["video"].bytes()).digest()[:8], "big"))
        return {"audio_emb": rng.standard_normal(9).astype(np.float32), "video_tokens": rng.standard_normal((TOKEN_GRID * TOKEN_GRID, 9)).astype(np.float32),
                "video_pos": _grid_pos(TOKEN_GRID)}


_BRIDGE_FILE = '''"""Synthetic {kind} bridge of a shared-space stand-in (written by the folder's testing.py)."""

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("xmodal_synth_{tag}", {synth!r})
_synth = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_synth)


class {cls}(_synth.{base}):
    """The {kind} test bridge.

    Preprocessing source:
        none: a synthetic fixture ({kind}); it runs no model, so there is no preprocessing to paste.
    """
'''


def bridge_source(path: str | Path, base: str, *, kind: str, synth: str | Path) -> Path:
    """Write a bridge file at ``path`` whose one class subclasses ``base`` (a bridge class of this module), loading
    this module from ``synth`` (the stand-in folder's copy)."""
    if base not in __all__ or not base.endswith("Bridge"):
        raise ValueError(f"bridge_source: {base!r} is not a bridge of xmodal_synth")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(str(synth).encode()).hexdigest()[:12]
    path.write_text(_BRIDGE_FILE.format(kind=kind, cls=f"{kind.capitalize()}Bridge", base=base, synth=str(Path(synth).resolve()), tag=tag),
                    encoding="utf-8")
    return path
