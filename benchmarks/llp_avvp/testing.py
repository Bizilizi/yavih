"""``llp_avvp``: the folder's own synthetic stand-in and test bridges for ``yavih bench trial --synthetic``.

The stand-in keeps this folder's name, scorer (``eval.py`` and its sibling modules), metrics, protocol and output
contract over tiny generated clips: one 10 s mp4 per row (16 x 16 video at 2 fps and a mono AAC track, both streams
read from it, as in the real folder), split into ``train`` (weak labels only), ``val`` and ``test`` (dense audio and
visual events). Every clip's events are drawn from its clip id (:func:`clip_events`): an audio-visual event of one
class (the same class heard and seen over overlapping seconds) and an audio-only event of another, among the first
``K_CLASSES`` classes.

The oracle bridge reads the stand-in's events from the clip id it finds by the media path (a guardrail violation
allowed only under ``YAVIH_TESTING=1``) and emits one token per second and stream: the audio token of second ``k``
is the sum of the class codes heard in it, the video token the sum of those seen, plus a little noise. The random
bridge emits noise seeded by the media bytes in the same layout. Module scope: stdlib, numpy, yaml, yavih core;
writing the media needs PyAV.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from yavih.bridge import ArgSpec, Bridge
from yavih.hashing import rng_seed

__all__ = ["K_CLASSES", "clip_events", "make_stand_in", "write_bridge", "OracleAVVPBridge", "RandomAVVPBridge"]

SECONDS, FPS, SIZE, RATE = 10.0, 2, 16, 8000
K_CLASSES = 6  # classes the synthetic events use (of the real 25)
T = 10
C = 25
_NOISE = 0.05
#: the stand-in's train / val split sizes beyond the trial's n_train (val: dense labels for epoch selection)
N_VAL = 16


def clip_events(clip_id: str) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """``(audio, visual)`` events ``[(onset, offset, class)]`` of a stand-in clip, drawn from its id: one audio-visual
    event (class ``c``, heard over ``[a0, a1)`` and seen over an overlapping ``[v0, v1)``) and one audio-only event of
    another class."""
    rng = np.random.default_rng(rng_seed(f"llp_avvp-synth-events:{clip_id}"))
    c, d = (int(x) for x in rng.choice(K_CLASSES, size=2, replace=False))
    a0 = int(rng.integers(0, 4))
    a1 = int(rng.integers(a0 + 4, T + 1))
    v0 = int(rng.integers(max(0, a0 - 1), a0 + 2))
    v1 = int(rng.integers(max(v0 + 3, a1 - 1), T + 1))
    b0 = int(rng.integers(0, 6))
    b1 = int(rng.integers(b0 + 3, T + 1))
    return [(a0, a1, c), (b0, b1, d)], [(v0, v1, c)]


def _cell(events: list[tuple[int, int, int]]) -> str:
    return ";".join(f"{a}:{b}:{c}" for a, b, c in events)


def _write_clip(path: Path, index: int) -> None:
    """A 10 s mp4: a constant 16 x 16 frame of a per-clip colour (mpeg4, 2 fps) and a per-clip tone (AAC, 8 kHz)."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    colour = ((37 * index) % 256, (91 * index + 40) % 256, (17 * index + 80) % 256)
    with av.open(str(path), "w") as c:
        v = c.add_stream("mpeg4", rate=FPS)
        v.width = v.height = SIZE
        v.pix_fmt = "yuv420p"
        a = c.add_stream("aac", rate=RATE)
        a.layout = "mono"
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        img[:] = colour
        for _ in range(int(SECONDS * FPS)):
            for p in v.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in v.encode(None):
            c.mux(p)
        t = np.arange(int(SECONDS * RATE)) / RATE
        pcm = (0.3 * np.sin(2 * np.pi * (200 + 7 * index) * t) * 32767).astype(np.int16)[None]
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = RATE
        for p in a.encode(frame):
            c.mux(p)
        for p in a.encode(None):
            c.mux(p)


def make_stand_in(work: str | Path, real: Any, *, n_eval: int, n_train: int, media_root: str | Path | None = None) -> Path:
    """Write ``<work>/llp_avvp``: the stand-in over ``n_train`` train, ``N_VAL`` val and ``n_eval`` test clips (the
    real folder's yaml with the manifest, splits, budget, media root and info replaced), its ``eval.py`` and sibling
    modules copied."""
    real_dir = Path(real.folder)
    y_real = yaml.safe_load((real_dir / "benchmark.yaml").read_text(encoding="utf-8"))
    name = str(y_real["name"])
    dst = Path(work) / name
    media = Path(media_root) if media_root is not None else Path(work) / "media" / name
    classes_csv = (real_dir / "classes.csv").read_bytes()
    names = [line.split(",", 1)[1] for line in classes_csv.decode("utf-8").splitlines()[1:]]
    header = ["clip_id", "split", "labels", "rel_media", "audio_events", "visual_events"]
    rows, index = [], 0
    for split, n in (("train", int(n_train)), ("val", N_VAL), ("test", int(n_eval))):
        for i in range(n):
            cid = f"{split}_{i:04d}"
            rel = f"clips/{cid}.mp4"
            if not (media / rel).is_file():
                _write_clip(media / rel, index)
            index += 1
            aud, vis = clip_events(cid)
            weak = ";".join(names[c] for c in sorted({c for _, _, c in aud + vis}))
            dense = split != "train"
            rows.append([cid, split, weak, rel, _cell(aud) if dense else "", _cell(vis) if dense else ""])
    data = ("\n".join(",".join(r) for r in [header] + rows) + "\n").encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    counts = {s: sum(1 for r in rows if r[1] == s) for s in ("train", "val", "test")}
    y = {k: v for k, v in y_real.items()}
    y.update({
        "title": f"{y_real['title']} (synthetic stand-in)", "aliases": [],
        "changelog": [{"version": y_real["protocol_version"], "date": "2026-11-15", "reason": "Synthetic stand-in (testing.py).", "invalidates": False}],
        "source": {"kind": "derived", "urls": [], "licence": "synthetic", "bytes": 0, "sha256": {}, "notes": "testing.py"},
        "budget": {"clips": counts, "media_gb": 0.0, "extract_min_per_1k_clips_ref": 0.1, "probe_minutes_ref": {name: 1}},
        "splits": {**y_real["splits"]},
        "data": {"roots": {"media_root": {"default": str(media), "description": "synthetic 10 s clips"}}, "tar_index": None, "derived": {}},
        "input_contract": {"handed_over": {"video": "the synthetic mp4 as stored (10 s, 2 fps, 16 x 16, mpeg4, with an AAC tone track)",
                                           "audio": "the same mp4 (its AAC tone track)"},
                           "info": {"video": ["duration_s", "fps", "container", "bytes"], "audio": ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]},
                           "facts": list(y_real["input_contract"]["facts"]), "materialize": None},
    })
    y.pop("obtain", None)
    y["manifest"] = {**y_real["manifest"], "sha256": sha, "classes_sha256": hashlib.sha256(classes_csv).hexdigest(),
                     "columns": {"rel_media": {"type": "str", "role": "media", "stream": ["video", "audio"]},
                                 "audio_events": {"type": "str", "role": "label_aux"}, "visual_events": {"type": "str", "role": "label_aux"}},
                     "upstream": {"source": "testing.py", "sha256": sha, "rows": len(rows), "note": "all rows synthetic", "upstream_files": {},
                                  "curation": {"total_upstream": len(rows), "per_split": {s: {"listed": n, "staged": n, "missing": 0} for s, n in counts.items()},
                                               "excluded": {}}}}
    y["metrics"] = {**y_real["metrics"], "bootstrap": {"samples": 200, "seed": 0}}
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "manifest.csv.gz").write_bytes(gzip.compress(data, mtime=0))
    (dst / "classes.csv").write_bytes(classes_csv)
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, width=120, allow_unicode=True), encoding="utf-8")
    (dst / "README.md").write_text(f"# {name} (synthetic stand-in)\n\nWritten by the folder's `testing.py`.\n", encoding="utf-8")
    for f in real_dir.glob("*.py"):
        if f.name not in ("make_manifest.py", "bridge_template.py", "testing.py"):
            shutil.copy2(f, dst / f.name)
    return dst


# ----------------------------------------------------------------------------- the test bridges


def _codebook(D: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed(f"llp_avvp-synth-codebook:{C}:{D}"))
    return rng.choice(np.array([-1.0, 1.0]), size=(C, D))


class _AVVPBridgeBase(Bridge):
    """One token per second and stream: 10 video tokens at the decoded pts of 10 uniform frames, 10 audio tokens at
    the centres of the 10 one-second windows of the track.

    Preprocessing source:
        none: a synthetic fixture; it runs no model, so there is no preprocessing to paste.
    """

    batch_size = 4
    dtype = "float32"
    pos_mode = "per_sample"
    streams = ("video", "audio")
    D = 32
    model_name = "llp-avvp-synth"
    bench_folder: str | None = None
    ARGS = {"reading": ArgSpec(type=str, default="av", choices=("av",), help="the stand-in serves the av reading")}

    def _input(self) -> dict:
        from yavih.media.decode import decoder_info

        return {"video": {"temporal": {"rule": "uniform", "n": T, "span": "full"}, "spatial": {"resize": None, "crop": None, "interp": "none", "antialias": False},
                          "norm": {"scale": 1.0, "source": "none: a synthetic test bridge"}, "decoder": decoder_info("video")},
                "audio": {"temporal": {"rule": "full", "max_s": SECONDS, "pad_to_s": SECONDS},
                          "front_end": {"sample_rate": RATE, "resample": "none", "features": "none (test bridge)", "channels": "mean"},
                          "norm": {"source": "none: a synthetic test bridge"}, "decoder": decoder_info("audio")}}

    def describe(self) -> dict:
        return {"weights_sha256": hashlib.sha256(f"{self.model_name}:{self.D}".encode()).hexdigest(), "weights_ref": "llp_avvp testing.py",
                "dtype": self.dtype, "batch_size": self.batch_size, "streams": list(self.streams), "pos_mode": self.pos_mode,
                "model": self.model_name, "reading": "av", "layer": 0, "pooling": "none", "views": 1, "input": self._input(),
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
                "args": dict(self.args), "N": 2 * T, "D": self.D}

    def prepare(self, sample: Any) -> dict:
        from yavih.media import decode, rules

        rin = self._input()
        v, a = sample.streams["video"], sample.streams["audio"]
        _, pts = decode.frames_at(v, rules.rule_times(rin["video"]["temporal"], v.info))
        ranges, (pl, pr) = rules.audio_ranges(rin["audio"]["temporal"], a.info)
        decode.audio(a, ranges)
        return {"pts": np.asarray(pts, dtype=np.float64), "v_dur": float(v.info["duration_s"]), "a_dur": float(a.info["duration_s"]),
                "used": {"video": {"times": [float(t) for t in pts]},
                         "audio": {"samples": [[int(lo), int(hi)] for lo, hi in ranges], "pad": [int(pl), int(pr)]}},
                "digest": hashlib.sha256(v.bytes() if hasattr(v, "bytes") else b"").hexdigest()}

    def _pos(self, p: dict) -> np.ndarray:
        pos = np.full((2 * T, 4), -1.0, dtype=np.float32)
        pos[:T, 0] = 0
        pos[:T, 1] = np.clip(p["pts"] / p["v_dur"], 0.0, 1.0)
        centres = (np.arange(T) + 0.5)
        pos[T:, 0] = 1
        pos[T:, 1] = np.where(centres < p["a_dur"], centres / p["a_dur"], -1.0)
        return pos

    def _feat(self, sample: Any, p: dict) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def encode(self, samples: list) -> tuple[Any, Any, list[dict]]:
        feats, pos, used = [], [], []
        for s in samples:
            p = s.prepared
            feats.append(self._feat(s, p))
            pos.append(self._pos(p))
            used.append(p["used"])
        return np.stack(feats).astype(self.dtype), np.stack(pos), used


class OracleAVVPBridge(_AVVPBridgeBase):
    """The token of second ``k`` carries the codes of the classes heard (audio) or seen (video) in it."""

    model_name = "llp-avvp-oracle"

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        folder = Path(self.bench_folder) if self.bench_folder else Path(bench.folder)
        with gzip.open(folder / "manifest.csv.gz", "rt", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        head = lines[0].split(",")
        self._by_media = {}
        for line in lines[1:]:
            r = dict(zip(head, line.split(","), strict=True))
            self._by_media[r["rel_media"]] = r["clip_id"]
        self.codebook = _codebook(self.D)

    def _clip_id(self, sample: Any) -> str:
        for st in sample.streams.values():
            path = getattr(st, "_real_path", None) or getattr(st, "path", None)
            if path:
                parts = Path(path).parts
                hit = self._by_media.get("/".join(parts[-2:]))
                if hit is not None:
                    return hit
        raise KeyError("oracle: the clip's media is not in the stand-in manifest")

    def _feat(self, sample: Any, p: dict) -> np.ndarray:
        cid = self._clip_id(sample)
        aud, vis = clip_events(cid)
        out = np.zeros((2 * T, self.D))
        for k in range(T):
            sec = int(np.floor(p["pts"][k] + 1e-9))
            for a, b, c in vis:
                if a <= sec < b:
                    out[k] += self.codebook[c]
            for a, b, c in aud:
                if a <= k < b:
                    out[T + k] += self.codebook[c]
        noise = _NOISE * np.random.default_rng(rng_seed(f"llp_avvp-synth-noise:{cid}")).standard_normal(out.shape)
        return out + noise


class RandomAVVPBridge(_AVVPBridgeBase):
    """Noise seeded by the clip's bytes, in the oracle's layout."""

    model_name = "llp-avvp-random"

    def _feat(self, sample: Any, p: dict) -> np.ndarray:
        return np.random.default_rng(rng_seed(f"llp_avvp-synth-random:{p['digest']}")).standard_normal((2 * T, self.D))


_BRIDGE_FILE = '''"""Synthetic {kind} bridge of the llp_avvp stand-in (written by the folder's testing.py)."""

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("llp_avvp_testing_{tag}", {testing!r})
_t = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_t)


class {cls}(_t.{base}):
    """The {kind} test bridge.

    Preprocessing source:
        none: a synthetic fixture ({kind}); it runs no model, so there is no preprocessing to paste.
    """

    D = {D}
    bench_folder = {folder!r}
'''


def write_bridge(path: str | Path, kind: str, *, D: int, bench_folder: str | Path) -> Path:
    """The stand-in's ``oracle`` / ``random`` bridge file at ``path``."""
    bases = {"oracle": "OracleAVVPBridge", "random": "RandomAVVPBridge"}
    if kind not in bases:
        raise ValueError(f"write_bridge: kind must be one of {sorted(bases)}, got {kind!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    me = str(Path(__file__).resolve())
    path.write_text(_BRIDGE_FILE.format(kind=kind, cls=f"{kind.capitalize()}Bridge", base=bases[kind], testing=me,
                                        tag=hashlib.sha256(me.encode()).hexdigest()[:12], D=int(D), folder=str(Path(bench_folder).resolve())),
                    encoding="utf-8")
    return path


