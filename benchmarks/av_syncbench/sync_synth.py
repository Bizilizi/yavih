"""Synthetic stand-in and test bridges of ``av_syncbench`` for ``yavih bench trial --synthetic``.

Every synthetic video carries an on/off pattern over 0.16 s blocks and a code ``z`` in ``(0, 1)``: while a block is
on, the whole 32 x 32 frame shows the colour ``(255 z, 255 (1 - z), 0)`` and the mp4's own AAC track plays a tone at
``300 + 1200 z`` Hz; while it is off, the frame is gray and the track is silent. Each row pairs a video with a
perturbed wav made from its tone (:func:`edited`): shifted (``offset``), one window shifted (``jitter``), resampled
(``speed``), or the same pattern at another pitch (the two timbre tasks). The rows cycle over every (task, level)
group of the real folder, so every metric is computed.

The oracle bridge reads each stream on its own (frames: the colour of the block centres; audio: the energy and the
spectral peak) and embeds every complete 0.64 s segment as its four block levels (in ``[-1, 1]``: the frame colour, the tone's energy
around the block centre) followed by a Fourier code of ``z``: the original pair matches exactly, a perturbed one does not. The random bridge embeds noise seeded by each
stream's bytes. Module scope: stdlib, numpy, yaml, yavih core; writing and decoding media needs PyAV.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from yavih.bridge import Bridge
from yavih.codecs import read_wav, wav_bytes
from yavih.freeze import csv_bytes
from yavih.hashing import rng_seed

__all__ = ["GROUPS", "make_stand_in", "bridge_source", "OracleSyncBridge", "RandomSyncBridge"]

FPS, SIZE, RATE = 25, 32, 16000
BLOCK = 0.16          # seconds per on/off block; four blocks per 0.64 s segment
N_BLOCKS = 20         # 3.2 s clips: five complete segments
SEGMENT = 0.64
F_LO, F_SPAN = 300.0, 1200.0
GRAY = 128
ON_RMS = 0.5 / np.sqrt(2.0)   # the RMS of an on block's tone
#: every (task, level, direction) group of the real folder's metrics, in the order the rows cycle over them
GROUPS: list[tuple[str, str, str]] = (
    [("offset", lv, d) for lv in ("50ms", "100ms", "200ms", "300ms", "500ms") for d in ("advance", "delay")]
    + [("jitter", lv, "") for lv in ("L1", "L2", "L3")]
    + [("speed", lv, "") for lv in ("0.80x", "0.83x", "0.87x", "0.91x", "0.95x", "1.05x", "1.10x", "1.15x", "1.20x", "1.25x")]
    + [("voice_timbre", "gender=male", ""), ("instrument_timbre", "solo_flute", "")]
)
MIN_ROWS = 2 * len(GROUPS)


def code(z: float) -> np.ndarray:
    """``[cos(2πkz), sin(2πkz)]`` for ``k = 1..3``: the pitch / colour code of a clip."""
    k = np.arange(1, 4)
    return np.concatenate([np.cos(2 * np.pi * k * z), np.sin(2 * np.pi * k * z)]).astype(np.float64)


def _bits(video_id: str) -> np.ndarray:
    rng = np.random.default_rng(rng_seed(f"yavih-synth-av_syncbench:{video_id}"))
    return rng.integers(0, 2, N_BLOCKS).astype(np.int64)


def _tone(bits: np.ndarray, z: float) -> np.ndarray:
    n = int(round(len(bits) * BLOCK * RATE))
    t = np.arange(n) / RATE
    env = bits[np.minimum((t / BLOCK).astype(np.int64), len(bits) - 1)].astype(np.float64)
    return (0.5 * env * np.sin(2 * np.pi * (F_LO + F_SPAN * z) * t)).astype(np.float32)


def edited(bits: np.ndarray, z: float, task: str, level: str, direction: str) -> np.ndarray:
    """The perturbed waveform of one row, built from the original tone: ``offset`` shifts it by twice the level
    (``delay`` zero-pads the start, ``advance`` drops it), ``jitter`` shifts the 0.64-1.92 s window by 0.12 / 0.24 /
    0.40 s, ``speed`` resamples it by the level's factor (pitch included), the timbre tasks re-synthesise the pattern
    at another pitch."""
    x = _tone(bits, z)
    if task == "offset":
        n = int(round(2 * float(level.rstrip("ms")) / 1000.0 * RATE))
        return np.concatenate([np.zeros(n, np.float32), x[:-n]]) if direction == "delay" else x[n:].copy()
    if task == "jitter":
        n = int(round({"L1": 0.12, "L2": 0.24, "L3": 0.40}[level] * RATE))
        lo, hi = int(0.64 * RATE), int(1.92 * RATE)
        out = x.copy()
        out[lo:hi] = x[lo + n:hi + n]
        return out
    if task == "speed":
        f = float(level.rstrip("x"))
        t = np.arange(int(np.floor(len(x) / f))) * f
        return np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return _tone(bits, (z + (0.5 if task == "voice_timbre" else 0.3)) % 1.0)


def _write_video(path: Path, bits: np.ndarray, z: float) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    on = np.full((SIZE, SIZE, 3), 0, dtype=np.uint8)
    on[..., 0], on[..., 1] = int(round(255 * z)), int(round(255 * (1 - z)))
    off = np.full((SIZE, SIZE, 3), GRAY, dtype=np.uint8)
    with av.open(str(path), "w") as c:
        v = c.add_stream("mpeg4", rate=FPS)
        v.width = v.height = SIZE
        v.pix_fmt = "yuv420p"
        v.options = {"qscale": "2"}
        a = c.add_stream("aac", rate=RATE)
        a.layout = "mono"
        for f in range(int(round(N_BLOCKS * BLOCK * FPS))):
            b = bits[min(int((f + 0.5) / FPS / BLOCK), N_BLOCKS - 1)]
            for p in v.encode(av.VideoFrame.from_ndarray(on if b else off, format="rgb24")):
                c.mux(p)
        for p in v.encode(None):
            c.mux(p)
        pcm = (_tone(bits, z) * 32767).astype(np.int16)[None]
        af = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        af.sample_rate = RATE
        for p in a.encode(af):
            c.mux(p)
        for p in a.encode(None):
            c.mux(p)


def make_stand_in(work: str | Path, real: Any, *, n_eval: int, n_train: int = 0) -> Path:
    """Write ``<work>/<real.name>``: a stand-in of the real folder over ``max(n_eval, MIN_ROWS)`` synthetic pairs (two
    rows per video; ``n_train`` is ignored: the scorer trains nothing). It keeps the real folder's name, scorer, header,
    metrics, protocol, input facts, output contract and fields, copies its ``eval.py`` and sibling modules, and
    adapts the chance-free bits (bootstrap samples, budgets)."""
    real_dir = Path(real.folder)
    y_real = yaml.safe_load((real_dir / "benchmark.yaml").read_text(encoding="utf-8"))
    dst = Path(work) / str(y_real["name"])
    media = Path(work) / "media" / str(y_real["name"])
    n = max(int(n_eval), MIN_ROWS)
    n_videos = (n + 1) // 2
    header = ["clip_id", "split", "task", "level", "direction", "category", "video_id", "rel_video", "rel_edited", "duration",
              "target_attribute", "source_attribute", "upstream_line"]
    rows = []
    for i in range(n):
        vid = f"v{i // 2:04d}"
        z = (i // 2 + 0.5) / n_videos
        bits = _bits(vid)
        rel_v, rel_e = f"AV_SyncBench/synth/original_video/{vid}.mp4", f"AV_SyncBench/synth/edited_audio/{vid}_{i:04d}.wav"
        if not (media / rel_v).is_file():
            _write_video(media / rel_v, bits, z)
        task, level, direction = GROUPS[i % len(GROUPS)]
        wav = edited(bits, z, task, level, direction)
        (media / rel_e).parent.mkdir(parents=True, exist_ok=True)
        (media / rel_e).write_bytes(wav_bytes(wav, RATE))
        rows.append([f"{vid}_{i:04d}", "test", task, level, direction, "synthetic", vid, rel_v, rel_e, json.dumps(len(wav) / RATE),
                     json.dumps(f"synthetic {task} {level} {direction}".strip()), "null", f"t{i + 1}"])
    data = csv_bytes(header, rows)
    sha = hashlib.sha256(data).hexdigest()
    metrics = dict(y_real["metrics"])
    metrics["bootstrap"] = {"samples": 200, "seed": 0}
    metrics.pop("notes", None)
    y = {
        "schema_version": y_real["schema_version"], "name": y_real["name"], "aliases": [], "title": f"{y_real['title']} (synthetic stand-in)",
        "capability": list(y_real.get("capability") or []), "header": y_real.get("header"), "core": y_real.get("core"),
        "scorer": y_real["scorer"], "protocol_version": y_real["protocol_version"],
        "changelog": [{"version": y_real["protocol_version"], "date": "2026-11-15", "reason": "Synthetic stand-in (sync_synth.py).", "invalidates": False}],
        "source": {"kind": "derived", "urls": [], "licence": "synthetic", "bytes": 0, "sha256": {}, "notes": "sync_synth.py"},
        "budget": {"clips": {"test": n}, "media_gb": 0.0, "extract_min_per_1k_clips_ref": 0.1, "probe_minutes_ref": dict(y_real["budget"]["probe_minutes_ref"])},
        "manifest": {"file": "manifest.csv.gz", "sha256": sha, "classes": None, "classes_sha256": None, "key": "clip_id",
                     "label": {"column": None, "kind": "none"}, "columns": y_real["manifest"]["columns"],
                     "upstream": {"source": "sync_synth.py", "sha256": sha, "rows": n, "note": "all rows synthetic", "upstream_files": {},
                                  "curation": {"total_upstream": n, "per_split": {"test": {"listed": n, "staged": n, "missing": 0}}, "excluded": {}}}},
        "splits": {"train": [], "eval": ["test"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95},
        "fields": y_real["fields"],
        "targets": None,
        "data": {"roots": {"media_root": {"default": str(media), "description": "synthetic pairs"}}, "tar_index": None, "derived": {}},
        "input_contract": {"handed_over": {"video": "the synthetic mp4 as stored (3.2 s, 25 fps, 32 x 32, mpeg4, with its AAC tone track)",
                                           "audio": "the same mp4, read for its AAC tone track",
                                           "edited_audio": "the synthetic perturbed wav as stored (16 kHz mono)"},
                           "info": {"video": ["duration_s", "fps", "container", "bytes"], "audio": ["duration_s", "sample_rate", "channels", "container", "bytes"],
                                    "edited_audio": ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]},
                           "facts": list((y_real.get("input_contract") or {}).get("facts") or []), "materialize": None},
        "output_contract": y_real["output_contract"], "external": [], "protocol": dict(y_real["protocol"]), "metrics": metrics,
        "readout_params": dict(y_real.get("readout_params") or {}),
    }
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "manifest.csv.gz").write_bytes(gzip.compress(data, mtime=0))
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, width=120, allow_unicode=True), encoding="utf-8")
    (dst / "README.md").write_text(f"# {y_real['name']} (synthetic stand-in)\n\nWritten by the folder's `testing.py` (`sync_synth.py`).\n", encoding="utf-8")
    for f in real_dir.glob("*.py"):
        if f.name not in ("make_manifest.py", "bridge_template.py", "testing.py"):
            shutil.copy2(f, dst / f.name)
    return dst


# ----------------------------------------------------------------------------- stand-in bridges


def _frames(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """``(uint8 [n, H, W, 3], float64 [n] seconds)`` of every frame of the mp4's video track."""
    import av

    frames, times = [], []
    with av.open(io.BytesIO(data)) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            times.append(float(f.pts * s.time_base) if f.pts is not None else len(times) / FPS)
    return np.stack(frames), np.asarray(times)


def _track(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float64 samples of a wav, or of an mp4's audio track (its frames are never decoded). The track is first
    converted to 16-bit PCM at its own rate and layout (the stored precision of the wavs): ``bridge check``'s
    isolation gate hands over a lossless s16 copy of the track, and this keeps the two decodes bit-identical."""
    if data[:4] == b"RIFF":
        x, sr = read_wav(data)
        return np.asarray(x, dtype=np.float64).reshape(len(x), -1).mean(axis=1), int(sr)
    import av

    with av.open(io.BytesIO(data)) as c:
        s = c.streams.audio[0]
        sr = int(s.codec_context.sample_rate or s.rate)
        ch = s.codec_context.channels or 1
        res = av.AudioResampler(format="s16", layout=s.codec_context.layout.name, rate=sr)
        parts = [f.to_ndarray().reshape(-1) for frame in c.decode(s) for f in res.resample(frame)]
        parts += [f.to_ndarray().reshape(-1) for f in res.resample(None)]
    pcm = np.concatenate(parts).astype(np.float64) / 32768.0
    return pcm.reshape(-1, ch).mean(axis=1), sr


def _segments(duration: float) -> int:
    return int(np.floor((duration + 1e-9) / SEGMENT))


def _video_embed(data: bytes) -> tuple[np.ndarray, list[float]]:
    frames, times = _frames(data)
    n_seg = _segments(len(frames) / FPS)
    out, used = [], []
    for k in range(n_seg):
        bits, z = [], None
        for j in range(4):
            t = (4 * k + j + 0.5) * BLOCK
            i = int(np.argmin(np.abs(times - t)))
            used.append(float(times[i]))
            rgb = frames[i].reshape(-1, 3).mean(axis=0)
            on = rgb[2] < GRAY / 2
            bits.append(float(np.clip(1.0 - 2.0 * rgb[2] / GRAY, -1.0, 1.0)))
            if on:
                z = float(rgb[0]) / 255.0
        out.append((np.asarray(bits), z))
    zs = [z for _, z in out if z is not None]
    zc = code(float(np.median(zs))) if zs else np.zeros(6)
    return np.stack([np.concatenate([b, 0.5 * zc]) for b, _ in out]).astype(np.float32), used


def _audio_embed(data: bytes) -> tuple[np.ndarray, list[list[int]]]:
    x, sr = _track(data)
    n_seg = _segments(len(x) / sr)
    half = int(0.03 * sr)
    bits = []
    for b in range(4 * n_seg):
        c = int(round((b + 0.5) * BLOCK * sr))
        w = x[max(0, c - half):c + half]
        rms = float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0
        bits.append(float(np.clip(2.0 * rms / ON_RMS - 1.0, -1.0, 1.0)))
    on = np.concatenate([x[int(b * BLOCK * sr):int((b + 1) * BLOCK * sr)] for b in range(len(bits)) if bits[b] > 0.5] or [np.zeros(1)])
    if len(on) > 16:
        spec = np.abs(np.fft.rfft(on * np.hanning(len(on))))
        f = float(np.argmax(spec[1:]) + 1) * sr / len(on)
        zc = code(float(np.clip((f - F_LO) / F_SPAN, 0.0, 1.0)))
    else:
        zc = np.zeros(6)
    emb = np.stack([np.concatenate([np.asarray(bits[4 * k:4 * k + 4]), 0.5 * zc]) for k in range(n_seg)]).astype(np.float32)
    return emb, [[0, int(min(len(x), round(n_seg * SEGMENT * sr)))]]


class _SyncBase(Bridge):
    """Preprocessing source:
        none: a synthetic fixture; it runs no model, so there is no preprocessing to paste.
    """

    batch_size = 2
    outputs = ("video_seg", "audio_seg", "edited_seg")

    def describe(self) -> dict:
        audio = {"norm": {"source": "none: block energy and the spectral peak"}, "decoder": {"lib": "pyav", "version": "stand-in"},
                 "temporal": {"rule": "custom", "description": "every complete 0.64 s segment, 30 ms around each 0.16 s block centre"},
                 "front_end": {"sample_rate": RATE, "resample": "none", "features": "block RMS and magnitude spectrum"}}
        return {"model": f"synth-{type(self).__name__.lower()}", "reading": "av", "weights_sha256": None, "weights_ref": "-", "batch_size": self.batch_size,
                "views": 1, "outputs": list(self.outputs),
                "input": {"video": {"norm": {"source": "none: pixel values"}, "decoder": {"lib": "pyav", "version": "stand-in"},
                                    "temporal": {"rule": "custom", "description": "the frames nearest the 0.16 s block centres of every complete segment"},
                                    "spatial": {"rule": "custom", "description": "the mean colour of the frame"}},
                          "audio": audio, "edited_audio": dict(audio)},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
                "args": dict(self.args)}

    def encode(self, samples: list[Any]) -> dict[str, Any]:
        res: dict[str, list[Any]] = {k: [] for k in self.outputs}
        res["used"] = []
        for s in samples:
            out, used = self._one(s)
            for k in self.outputs:
                res[k].append(out[k])
            res["used"].append(used)
        return res

    def _one(self, s: Any) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError


class OracleSyncBridge(_SyncBase):
    """Block bits and the pitch code per segment, each stream decoded on its own."""

    def _one(self, s: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        v, vt = _video_embed(s.streams["video"].bytes())
        a, ar = _audio_embed(s.streams["audio"].bytes())
        e, er = _audio_embed(s.streams["edited_audio"].bytes())
        return ({"video_seg": v, "audio_seg": a, "edited_seg": e},
                {"video": {"times": vt}, "audio": {"samples": ar, "pad": [0, 0]}, "edited_audio": {"samples": er, "pad": [0, 0]}})


class RandomSyncBridge(_SyncBase):
    """Noise seeded by each stream's bytes, one row per complete segment."""

    @staticmethod
    def _noise(data: bytes, salt: str, n: int) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(salt.encode() + data).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal((max(n, 1), 10)).astype(np.float32)

    def _one(self, s: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        info = {k: s.streams[k].info for k in ("video", "audio", "edited_audio")}
        nseg = {k: _segments(float(info[k]["duration_s"])) for k in info}
        out = {"video_seg": self._noise(s.streams["video"].bytes(), "v", nseg["video"]),
               "audio_seg": self._noise(s.streams["audio"].bytes(), "a", nseg["audio"]),
               "edited_seg": self._noise(s.streams["edited_audio"].bytes(), "e", nseg["edited_audio"])}
        ns = int(info["edited_audio"].get("n_samples") or 0)
        return out, {"video": {"times": [0.0]}, "audio": {"samples": [[0, 1]], "pad": [0, 0]}, "edited_audio": {"samples": [[0, max(ns, 1)]], "pad": [0, 0]}}


_BRIDGE_FILE = '''"""Synthetic {kind} bridge of the av_syncbench stand-in (written by the folder's testing.py)."""

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("sync_synth_{tag}", {synth!r})
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
        raise ValueError(f"bridge_source: {base!r} is not a bridge of sync_synth")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(str(synth).encode()).hexdigest()[:12]
    path.write_text(_BRIDGE_FILE.format(kind=kind, cls=f"{kind.capitalize()}Bridge", base=base, synth=str(Path(synth).resolve()), tag=tag),
                    encoding="utf-8")
    return path
