"""A synthetic streaming benchmark and stand-in streaming bridges (tests, ``bench trial``).

Every recording is a tiny mp4 (32 x 32, 10 fps, 6 to 9 s) whose second ``k`` shows one flat colour: class ``c`` of
``classes`` (``PALETTE[c]``) or the background gray. The folder (``scorer: stream``, ``output_contract.shape:
sequence``) scores the class of every 1-s bin at the bin's end. :class:`OracleStreamBridge` decodes the frames up
to each cutoff and embeds the colour of the latest one (causal, honest support); :class:`LeakyStreamBridge` peeks
one second past the cutoff while declaring an honest support (only the future-invariance gate can see it);
:class:`RealtimeStreamBridge` runs the oracle through :func:`yavih.stream.replay`; :class:`RandomStreamBridge`
embeds noise. Needs PyAV (the ``media`` extra).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from yavih import stream as _stream
from yavih.bridge import Bridge
from yavih.codecs import STREAM_SCHEMA
from yavih.hashing import rng_seed

__all__ = ["PALETTE", "make_stream_benchmark", "OracleStreamBridge", "LeakyStreamBridge", "RealtimeStreamBridge", "RandomStreamBridge"]

#: class colours (RGB); index 0 is the background gray
PALETTE: tuple[tuple[int, int, int], ...] = ((128, 128, 128), (220, 30, 30), (30, 200, 40), (40, 60, 220), (230, 220, 40))
FPS, SIZE = 10, 32
UPDATE_INTERVAL_S = 0.5


def _seconds_of(clip_id: str, classes: int) -> list[int]:
    rng = np.random.default_rng(rng_seed(f"yavih-synth-stream:{clip_id}"))
    n = int(rng.integers(6, 10))
    out, cur = [], int(rng.integers(0, classes + 1))
    for _ in range(n):
        if rng.random() < 0.35:
            cur = int(rng.integers(0, classes + 1))
        out.append(cur)
    return out


def _write_mp4(path: Path, seconds: list[int]) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        v = c.add_stream("mpeg4", rate=FPS)
        v.width = v.height = SIZE
        v.pix_fmt = "yuv420p"
        v.options = {"qscale": "2"}
        for cls in seconds:
            for _ in range(FPS):
                img = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
                img[...] = PALETTE[cls]
                for p in v.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    c.mux(p)
        for p in v.encode(None):
            c.mux(p)


_EVAL_PY = '''\
"""Synthetic streaming benchmark: targets and metrics (yavih.testing.stream)."""

import numpy as np

from yavih.metrics.registry import MetricSpec
from yavih.metrics.stream import recording_interval, segment_accuracy


def targets(bench, split, rows):
    man = bench.manifest(split)
    out = []
    for r in rows:
        secs = [int(x) for x in str(man.extra[int(r)]["seconds"]).split(";")]
        out.append({"times": np.arange(1, len(secs) + 1, dtype=np.float64), "labels": {"event": np.asarray(secs, dtype=np.int64)}})
    return out


def _acc(outputs, targets, protocol, split_meta):
    return segment_accuracy(outputs["event"], targets["event"])


def _ci(kind, outputs, targets, protocol, split_meta):
    b = split_meta.get("bootstrap") or {}
    return recording_interval(lambda w: segment_accuracy(outputs["event"], targets["event"], w=w), split_meta["recording"],
                              samples=int(b.get("samples", 200)), seed=int(b.get("seed", 0)))


METRICS = {
    "bin_acc_linear": MetricSpec("bin_acc_linear", _acc, frozenset({"stream"}), ci=frozenset({"bootstrap_clip"}), interval=_ci),
    "bin_acc_gru": MetricSpec("bin_acc_gru", _acc, frozenset({"stream"}), ci=frozenset({"bootstrap_clip"}), interval=_ci),
    "bin_acc_floor": MetricSpec("bin_acc_floor", _acc, frozenset({"stream"}), kind="floor", floor_of=("bin_acc_linear", "bin_acc_gru")),
}
'''


def make_stream_benchmark(root: str | os.PathLike, *, name: str = "synth_stream", n_train: int = 24, n_eval: int = 12, classes: int = 3,
                          media_root: str | os.PathLike | None = None) -> Path:
    """Write the synthetic streaming folder ``<root>/<name>`` and its media (under ``media_root``, default
    ``<root>/../media/<name>``); returns the folder."""
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    media = Path(media_root) if media_root is not None else Path(root).parent / "media" / name
    rows = []
    for split, n in (("train", n_train), ("test", n_eval)):
        for i in range(n):
            cid = f"{split}_{i:04d}"
            secs = _seconds_of(cid, classes)
            rel = f"{split}/{cid}.mp4"
            if not (media / rel).is_file():
                _write_mp4(media / rel, secs)
            rows.append([cid, split, rel, ";".join(str(s) for s in secs)])
    header = ["clip_id", "split", "rel_media", "seconds"]
    text = "\n".join(",".join(r) for r in [header] + rows) + "\n"
    data = text.encode("utf-8")
    (folder / "manifest.csv.gz").write_bytes(gzip.compress(data, mtime=0))
    (folder / "eval.py").write_text(_EVAL_PY, encoding="utf-8")
    y = {
        "schema_version": 3, "name": name, "aliases": [], "title": "Synthetic causal recognition (stream)", "scorer": "stream",
        "protocol_version": 1,
        "changelog": [{"version": 1, "date": "2026-11-03", "reason": "Synthetic test fixture (yavih.testing.stream).", "invalidates": False}],
        "header": {"name": name, "title": "Synthetic causal recognition", "venue": "yavih.testing · synthetic fixture",
                   "abstract": "Flat-colour recordings whose every second shows one class."},
        "source": {"kind": "derived", "urls": [], "licence": "synthetic", "bytes": 0, "sha256": {}, "notes": "yavih.testing.stream"},
        "budget": {"clips": {"train": n_train, "test": n_eval}, "media_gb": 0.0, "extract_min_per_1k_clips_ref": 0.1,
                   "probe_minutes_ref": {"stream": 1}},
        "manifest": {"file": "manifest.csv.gz", "sha256": hashlib.sha256(data).hexdigest(), "classes": None, "classes_sha256": None,
                     "key": "clip_id",
                     "label": {"column": None, "kind": "none", "sep": ";", "class_key": "name", "sets": None, "train_set": None,
                               "allow_unknown": False, "empty_ok": False},
                     "columns": {"rel_media": {"type": "str", "role": "media", "stream": "video"},
                                 "seconds": {"type": "str", "role": "label_aux"}},
                     "upstream": {"source": "yavih.testing.stream", "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows),
                                  "note": "all rows synthetic", "upstream_files": {},
                                  "curation": {"total_upstream": len(rows), "per_split": {
                                      "train": {"listed": n_train, "staged": n_train, "missing": 0},
                                      "test": {"listed": n_eval, "staged": n_eval, "missing": 0}},
                                      "excluded": {}, "short_clips": 0, "leakage_audit": None}}},
        "splits": {"train": ["train"], "eval": ["test"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95},
        "fields": {"video": {"storage": "file", "kind": "video", "media_column": "rel_media", "root": "media_root", "layout": "dir",
                             "layout_params": {}, "required": True}},
        "targets": None,
        "data": {"roots": {"media_root": {"default": str(media), "description": "synthetic streaming media"}}, "tar_index": None, "derived": {}},
        "input_contract": {"handed_over": {"video": "the synthetic mp4 as stored (6-9 s, 10 fps, 32 x 32, mpeg4)"},
                           "info": {"video": ["duration_s", "fps", "container", "bytes"]}, "facts": [], "materialize": None},
        "output_contract": {"shape": "sequence", "coordinates": "media", "target_frame": None,
                            "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None},
                            "schema": {k: dict(v) for k, v in STREAM_SCHEMA.items()},
                            "stream": {"update_interval_s": UPDATE_INTERVAL_S}},
        "external": [],
        "protocol": {"clip_seconds": None, "train_draw": None, "views": "none"},
        "metrics": {"headline": "bin_acc_gru", "list": ["bin_acc_linear", "bin_acc_gru"], "floors": ["bin_acc_floor"], "controls": [],
                    "chance": {}, "ci": {"bin_acc_linear": "bootstrap_clip", "bin_acc_gru": "bootstrap_clip"},
                    "bootstrap": {"samples": 200, "seed": 0}, "aggregation": None},
        "readout_params": {"groups": {"event": {"loss": "ce", "classes": classes + 1}}},
    }
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, width=120), encoding="utf-8")
    (folder / "README.md").write_text(textwrap.dedent(f"""\
        # {name}

        Synthetic streaming fixture written by `yavih.testing.stream.make_stream_benchmark`.
        """), encoding="utf-8")
    return folder


# ----------------------------------------------------------------------------- stand-in bridges


def _frames(sample: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(times [F], colours [F, 3])`` of the recording's video frames (the mean colour of each frame)."""
    import av

    st = sample.streams["video"]
    with av.open(io.BytesIO(st.bytes())) as c:
        out = [(float(f.time), f.to_ndarray(format="rgb24").reshape(-1, 3).mean(axis=0)) for f in c.decode(video=0)]
    return np.asarray([t for t, _ in out]), np.asarray([x for _, x in out], dtype=np.float32)


def _class_of(colour: np.ndarray) -> int:
    d = ((np.asarray(PALETTE, dtype=np.float32) - colour[None]) ** 2).sum(axis=1)
    return int(np.argmin(d))


class _StreamBase(Bridge):
    batch_size = 1
    outputs = ("emb", "support", "cutoff")
    mode = "causal"
    lookahead_s = 0.0
    dim = 8

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        self.dt = float((bench.output_contract or {}).get("stream", {}).get("update_interval_s", UPDATE_INTERVAL_S))

    def describe(self) -> dict:
        return {"model": f"synth-{type(self).__name__.lower()}", "reading": "v-rt" if self.mode == "realtime" else "v", "weights_sha256": None, "weights_ref": "-",
                "batch_size": self.batch_size, "views": 1, "outputs": list(self.outputs),
                "input": {"video": {"norm": {"source": "none: the frame's mean colour in [0, 1]"},
                                    "decoder": {"lib": "pyav", "version": "stand-in"},
                                    "temporal": {"rule": "custom", "description": "every frame at or before each update's cutoff"},
                                    "spatial": {"rule": "custom", "description": "the frame's mean colour"}}},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
                "args": dict(self.args),
                "stream": {"mode": self.mode, "native": False, "video_window_s": 1.0, "audio_window_s": None, "num_frames": 1,
                           "frame_sampling": "the latest frame at or before the cutoff", "startup_policy": "none needed (one frame)",
                           "gap_policy": "none (stateless)", "device": "cpu"}}

    def _emb(self, colour: np.ndarray, cls: int) -> np.ndarray:
        e = np.zeros(self.dim, dtype=np.float32)
        e[cls] = 1.0
        e[5:8] = colour / 255.0
        return e

    def _step_fn(self, sample: Any):
        times, cols = _frames(sample)

        def step(cutoff: float, since: float):  # the latest frame strictly before the cutoff (the bin being scored)
            k = int(np.searchsorted(times, cutoff + self.lookahead_s - 1e-9, side="left")) - 1
            k = max(k, 0)
            honest = max(int(np.searchsorted(times, cutoff - 1e-9, side="left")) - 1, 0)
            t = float(times[honest])
            return self._emb(cols[k], _class_of(cols[k])), np.array([max(0.0, t), t, np.nan, np.nan])

        return step

    def _run(self, sample: Any) -> dict[str, Any]:
        dur = _stream.recording_duration({n: st.info for n, st in sample.streams.items()})
        return _stream.causal(self._step_fn(sample), duration_s=dur, update_interval_s=self.dt, dim=self.dim)

    def encode(self, samples: list[Any]) -> dict[str, Any]:
        outs = [self._run(s) for s in samples]
        res = {k: [o[k] for o in outs] for k in self.outputs}
        res["used"] = [{"video": {"times": [float(x) for x in o["support"][:, 1]]}} for o in outs]
        return res


class OracleStreamBridge(_StreamBase):
    """Embeds the colour (and its class) of the latest frame at or before each cutoff."""


class LeakyStreamBridge(_StreamBase):
    """Reads the frame one second AFTER each cutoff while declaring the honest support."""

    lookahead_s = 1.0


class RandomStreamBridge(_StreamBase):
    """Noise seeded by the recording's bytes and the cutoff: no information about the label."""

    def _step_fn(self, sample: Any):
        seed = int.from_bytes(hashlib.sha256(sample.streams["video"].bytes()).digest()[:8], "big")
        times, _ = _frames(sample)

        def step(cutoff: float, since: float):
            rng = np.random.default_rng([seed, int(round(cutoff * 1000))])
            k = max(int(np.searchsorted(times, cutoff + 1e-9, side="right")) - 1, 0)
            t = float(times[k])
            return rng.standard_normal(self.dim).astype(np.float32), np.array([t, t, np.nan, np.nan])

        return step


class _VirtualClock:
    """A clock that advances only when the replay sleeps or a step runs: the scheduler's behaviour without waiting."""

    def __init__(self) -> None:
        self.t = 0.0

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(0.0, s)


class RealtimeStreamBridge(_StreamBase):
    """The oracle under the real-time replay (``yavih.stream.replay``); ``cost_s`` seconds per update. With
    ``virtual_clock`` the replay runs on a virtual clock that each step advances by ``cost_s`` (tests: the
    scheduler's skips and delays without the wall-clock wait)."""

    outputs = ("emb", "support", "cutoff", "timing")
    mode = "realtime"
    cost_s = 0.01
    virtual_clock = False

    def _run(self, sample: Any) -> dict[str, Any]:
        import time

        inner = self._step_fn(sample)
        vc = _VirtualClock() if self.virtual_clock else None

        def step(cutoff: float, since: float):
            if vc is not None:
                vc.t += self.cost_s
            else:
                time.sleep(self.cost_s)
            return inner(cutoff, since)

        dur = _stream.recording_duration({n: st.info for n, st in sample.streams.items()})
        kw = {"clock": vc.clock, "sleep": vc.sleep} if vc is not None else {}
        return _stream.replay(step, duration_s=dur, update_interval_s=self.dt, dim=self.dim, **kw)

    def encode(self, samples: list[Any]) -> dict[str, Any]:
        outs = [self._run(s) for s in samples]
        res = {k: [o[k] for o in outs] for k in self.outputs}
        res["used"] = [{"video": {"times": [float(x) for x in o["support"][:, 1]]}} for o in outs]
        return res
