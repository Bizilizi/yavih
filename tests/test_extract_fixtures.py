"""Shared fixtures for the bridge + extract tests: an in-memory ``Benchmark`` stand-in that
hands over RAW stored bytes (no decoder packages, no media on disk) and a family of test bridges.

Not a test module (no ``test_*`` functions); imported by ``test_bridge*.py`` / ``test_extract*.py``.
The reader contract it imitates (docs/bridge.md): ``RawStream`` = ``bytes()`` + factual ``info``,
contiguous ``plan_shards`` ranges, ascending rows, canaries copied from rows ≥ ``2 * batch_size`` before the
shard end carrying the private ``_canary_of``, ``MediaMissing`` raised by ``bytes()`` of an absent file.
It does not import ``yavih.sample``: ``run_extract`` and ``bridge check`` duck-type Samples.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from yavih.bridge import ArgSpec, Bridge, NotApplicable, grid_pos
from yavih.io.shards import plan_shards
from yavih.io.validate import crop_region, rule_times
from yavih.media import MediaMissing

MANIFEST_HASH = "a" * 64
PROTOCOL_HASH = "b" * 64
INPUT_HASH = "c" * 64


def _seed(text: str | bytes) -> int:
    b = text.encode() if isinstance(text, str) else text
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "big")


class FakeSource:
    """The byte-source surface the extractor touches (``yavih.sample._Source.release``), with a release counter."""

    def __init__(self) -> None:
        self.releases = 0

    def release(self) -> None:
        self.releases += 1


class FakeRawStream:
    """``RawStream``: the stored bytes (here: synthetic, a function of the clip), ``info`` and a ``FakeSource``."""

    def __init__(self, name: str, kind: str, payload: bytes, info: dict, *, fail: BaseException | None = None, clip_id: str = "") -> None:
        self.name = name
        self.kind = kind
        self.path = None
        self.member = None
        self.info = dict(info)
        self._payload = payload
        self._fail = fail
        self.reads = 0
        self._src = FakeSource()

    def bytes(self) -> bytes:
        if self._fail is not None:
            raise self._fail
        self.reads += 1
        return self._payload

    def open(self) -> io.BytesIO:
        return io.BytesIO(self.bytes())


@dataclass(frozen=True)
class Sample:
    """``Sample`` (local copy so these tests do not depend on ``yavih.sample``)."""

    row: int
    clip_id: str
    split: str
    streams: dict
    meta: dict
    _canary_of: int | None = field(default=None, repr=False, compare=False)
    prepared: object = field(default=None, init=False, repr=False, compare=False)


VIDEO_INFO = {"duration_s": 2.0, "fps": 25.0, "n_frames": 50, "width": 320, "height": 240, "container": "mp4", "bytes": 64,
              "sample_rate": None, "n_samples": None, "channels": None}
AUDIO_INFO = {"duration_s": 2.0, "sample_rate": 16000, "n_samples": 32000, "channels": 1, "container": "wav", "bytes": 64,
              "fps": None, "n_frames": None, "width": None, "height": None}


class FakeBenchmark:
    """Duck-typed ``Benchmark`` (the surface used by extract / bridge check): ``name``, ``spec``, hashes,
    ``n_rows``, ``roles``, ``reader``, ``manifest(split).media_key``, ``data_status``."""

    def __init__(
        self,
        *,
        name: str = "synth_fake",
        n: dict[str, int] | None = None,
        streams: dict[str, str] | None = None,
        missing_rows: set[int] | None = None,
        fail_rows: dict[int, BaseException] | None = None,
        infos: dict[int, dict[str, dict]] | None = None,
        shared_media: dict[int, int] | None = None,
        payload_fn=None,
    ) -> None:
        self.name = name
        self.n = dict(n or {"train": 20, "val": 8})
        self.stream_kinds = dict(streams or {"video": "video"})
        self.missing_rows = set(missing_rows or ())
        self.fail_rows = dict(fail_rows or {})
        self.infos = dict(infos or {})
        self.shared_media = dict(shared_media or {})  # row -> row whose media it shares (AVE's two events of one video)
        self.payload_fn = payload_fn  # (split, row, stream) -> stored bytes (e.g. a real WAV); default: synthetic bytes
        self.manifest_hash = MANIFEST_HASH
        self.protocol_hash = PROTOCOL_HASH
        self.input_hash = INPUT_HASH
        self.spec = SimpleNamespace(
            name=name,
            aliases=(),
            protocol_version=2,
            readout="clf",
            streams={k: SimpleNamespace(kind=v, layout_params={}) for k, v in self.stream_kinds.items()},
            splits=SimpleNamespace(train=("train",), eval=("val",), floor=(), folds=None, unused=()),
            protocol={"clip_seconds": 2.0},
            input_contract={"handed_over": {k: "synthetic bytes" for k in self.stream_kinds}, "info": {}, "facts": [], "materialize": None},
            output_contract={"shape": "fixed", "coordinates": "media", "target_frame": None, "requires": {"pos": [], "streams": []}},
        )
        self.reader_calls: list[dict] = []

    def n_rows(self, split: str) -> int:
        return self.n[split]

    def data_status(self):
        return SimpleNamespace(media_fingerprint="f" * 64, media_info_sha256=None, roots={"media_root": "/nonexistent/media"})

    def roles(self) -> dict[str, list[str]]:
        return {"train": ["train"], "eval": ["val"], "floor": [], "folds": []}

    def media_key(self, split: str, row: int) -> str:
        return f"{split}/{self.shared_media.get(row, row):04d}.mp4"

    def manifest(self, split: str):
        return SimpleNamespace(media_key=[self.media_key(split, r) for r in range(self.n[split])])

    def _sample(self, split: str, row: int, canary_of: int | None = None) -> Sample:
        clip_id = f"{split}_{row:04d}"
        streams: dict[str, FakeRawStream] = {}
        for sname, kind in self.stream_kinds.items():
            fail = MediaMissing(clip_id, sname, f"/nonexistent/media/{clip_id}.mp4") if row in self.missing_rows else self.fail_rows.get(row)
            info = dict(VIDEO_INFO if kind == "video" else AUDIO_INFO)
            info.update((self.infos.get(row) or {}).get(sname) or {})
            if self.payload_fn is not None:
                payload = self.payload_fn(split, self.shared_media.get(row, row), sname)
            else:
                payload = hashlib.sha256(f"{self.media_key(split, row)}:{sname}".encode()).digest() * 2
            streams[sname] = FakeRawStream(sname, kind, payload, info, fail=fail, clip_id=clip_id)
        return Sample(row=row, clip_id=clip_id, split=split, streams=streams, meta={"facts": {}}, _canary_of=canary_of)

    def reader(self, split: str, *, shard=None, batch_size: int = 1, rows: range | None = None, canaries: int = 0, canary_seed: int = 0) -> Iterator[Sample]:
        n = self.n_rows(split)
        if shard is not None:
            lo, hi = plan_shards(n, batch_size, shard[1])[shard[0]]
        elif rows is not None:
            lo, hi = rows.start, min(rows.stop, n)
        else:
            lo, hi = 0, n
        self.reader_calls.append({"split": split, "shard": shard, "rows": (lo, hi), "batch_size": batch_size, "canaries": canaries})
        regular = [self._sample(split, r) for r in range(lo, hi)]
        yield from regular
        k = min(int(canaries), len(regular) // 4)
        eligible = [s.row for s in regular if s.row < hi - 2 * batch_size]
        k = min(k, len(eligible))
        if k > 0:
            rng = np.random.default_rng(_seed(f"{self.manifest_hash}:{split}:{shard[0] if shard else 0}:{canary_seed}"))
            for r in sorted(int(x) for x in rng.choice(eligible, size=k, replace=False)):
                yield self._sample(split, r, canary_of=r)


# ----------------------------------------------------------------------------- bridges

PRECISION_OK = {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True}
DECODER = {"lib": "fake-decoder", "version": "1.0"}
VIDEO_INPUT = {
    "temporal": {"rule": "uniform", "n": 4, "span": "full"},
    "spatial": {"resize": {"short": 224}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True},
    "norm": {"scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "source": "example/model@abc123 transforms.py"},
    "decoder": dict(DECODER),
}
AUDIO_INPUT = {
    "temporal": {"rule": "span", "start_s": 0.0, "length_s": 2.0, "pad": "zeros"},
    "front_end": {"sample_rate": 16000, "resample": "none", "features": "kaldi_fbank", "n_mels": 8},
    "norm": {"mean": 0.0, "std": 1.0, "source": "example/model@abc123 dataset.py"},
    "decoder": {"lib": "fake-audio", "version": "1.0"},
}


def frame_times(info: dict, n: int) -> list[float]:
    """The decoded frame times a bridge reports for the uniform rule: the pts of the nearest frames."""
    t = rule_times({"rule": "uniform", "n": n, "span": "full"}, info)
    fps = float(info["fps"])
    return [float(round(x * fps) / fps) for x in t]


def clip_features(payload: bytes, n: int, d: int, scale: float = 1.0) -> np.ndarray:
    """``[n, d]``: a pure function of the stored bytes (what a deterministic, instance-wise model computes)."""
    return (np.random.default_rng(_seed(payload)).standard_normal((n, d)) * scale).astype(np.float32)


class NumpyBridge(Bridge):
    """A torch-free bridge: 4 frame tokens of 6 dims, per-sample pos in media coordinates.

    Preprocessing source: example/model@abc123 transforms.py::eval_transform
        Resize(224) -> CenterCrop(224) -> x / 255 -> (x - 0.5) / 0.5
    """

    name = "np-test"
    batch_size = 4
    dtype = "float32"
    pos_mode = "per_sample"
    streams = ("video",)
    ARGS = {
        "layer": ArgSpec(type=int, default=12),
        "scale": ArgSpec(type=float, default=1.0),
        "flag": ArgSpec(type=bool, default=False),
        "mode": ArgSpec(type=str, default="a", choices=("a", "b")),
    }
    T = 4
    D = 6
    weights = {"w": np.arange(6, dtype=np.float32)}
    #: describe()["model"] / ["reading"]: the declared names results are filed under
    MODEL_NAME = "np-test"
    READING_NAME = "v"

    def __init__(self, bench, args=None):
        super().__init__(bench, args)
        self.seen_rows: list[int] = []
        self.seen_clip_ids: list[str] = []
        self.encode_calls = 0

    def input_block(self) -> dict:
        return {"video": {k: (dict(v) if isinstance(v, dict) else v) for k, v in VIDEO_INPUT.items()}}

    def describe(self) -> dict:
        from yavih.bridge import fingerprint_state_dict

        return {
            "model": self.MODEL_NAME,
            "reading": self.READING_NAME,
            "weights_sha256": fingerprint_state_dict(self.weights),
            "weights_ref": "memory://np-test",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "layer": self.args["layer"],
            "pooling": "none",
            "views": 1,
            "input": self.input_block(),
            "precision": dict(PRECISION_OK),
            "args": dict(self.args),
        }

    def one(self, s) -> tuple[np.ndarray, np.ndarray, dict]:
        st = s.streams["video"]
        info = st.info
        times = frame_times(info, self.T)
        feat = clip_features(st.bytes(), self.T, self.D, self.args["scale"])
        region = crop_region(VIDEO_INPUT["spatial"], info["width"], info["height"])
        pos = grid_pos(self.T, 1, 1, 0, t_centres=[t / info["duration_s"] for t in times], region=region)
        return feat, pos, {"video": {"times": times}}

    def encode_one(self, s) -> tuple[np.ndarray, np.ndarray, dict]:
        return self.one(s)

    def encode(self, samples):
        self.encode_calls += 1
        feats, poss, used = [], [], []
        for s in samples:
            self.seen_rows.append(s.row)
            self.seen_clip_ids.append(s.clip_id)
            f, p, u = self.encode_one(s)
            feats.append(f)
            poss.append(p)
            used.append(u)
        pos = np.stack(poss) if self.pos_mode == "per_sample" else (poss[0] if self.pos_mode == "constant" else None)
        return np.stack(feats), pos, used


class PrepareBridge(NumpyBridge):
    """``NumpyBridge`` with the per-sample work moved into ``prepare``: ``encode`` reads only ``s.prepared``.
    Records the prepare threads, whether ``torch.inference_mode`` was on there, and samples ``encode`` got unprepared."""

    name = "np-prepare"

    def __init__(self, bench, args=None):
        super().__init__(bench, args)
        self.prepare_threads: list[str] = []
        self.prepare_inference: list[bool] = []
        self.unprepared: list[str] = []

    def prepare(self, s):
        import sys
        import threading

        self.prepare_threads.append(threading.current_thread().name)
        torch = sys.modules.get("torch")
        if torch is not None:
            self.prepare_inference.append(bool(torch.is_inference_mode_enabled()))
        return self.one(s)

    def encode_one(self, s):
        if s.prepared is None:
            self.unprepared.append(s.clip_id)
            return self.one(s)
        return s.prepared


class ConstantPosBridge(NumpyBridge):
    name = "np-constpos"
    pos_mode = "constant"


class NoPosBridge(NumpyBridge):
    name = "np-nopos"
    pos_mode = "none"
    dtype = "float16"


class AVBridge(NumpyBridge):
    """Reads video and audio; tags both (pos per sample)."""

    name = "np-av"
    streams = ("video", "audio")
    READING_NAME = "av"

    def input_block(self) -> dict:
        return {**super().input_block(), "audio": {k: (dict(v) if isinstance(v, dict) else v) for k, v in AUDIO_INPUT.items()}}

    def one(self, s):
        f, p, u = super().one(s)
        a = s.streams["audio"]
        n = int(a.info["n_samples"])
        L = 32000
        u = {**u, "audio": {"samples": [[0, min(n, L)]], "pad": [0, max(0, L - n)]}}
        fa = clip_features(a.bytes(), 1, self.D)
        pa = np.array([[1.0, 0.5, -1.0, -1.0]], dtype=np.float32)
        return np.concatenate([f, fa]), np.concatenate([p, pa]), u


class BadPrecisionBridge(NumpyBridge):
    name = "np-badprec"

    def describe(self) -> dict:
        d = super().describe()
        d["precision"] = {"autocast": None, "matmul_precision": None, "allow_tf32": False, "deterministic": True}
        return d


class NotApplicableInitBridge(NumpyBridge):
    name = "np-na-init"

    def __init__(self, bench, args=None):
        super().__init__(bench, args)
        raise NotApplicable("audio-only bridge: benchmark has no audio stream")


class NotApplicableEncodeBridge(NumpyBridge):
    name = "np-na-encode"

    def encode(self, samples):
        raise NotApplicable("cannot serve this media")


class CrashingBridge(NumpyBridge):
    name = "np-crash"

    def encode(self, samples):
        raise RuntimeError("boom inside encode")


class SigtermBridge(NumpyBridge):
    """Raises SIGTERM at itself during the first encode (the scheduler's ``--signal=B:TERM``)."""

    name = "np-sigterm"
    signal_name = "SIGTERM"

    def encode(self, samples):
        import signal

        out = super().encode(samples)
        if self.encode_calls == 1:
            signal.raise_signal(getattr(signal, self.signal_name))
        return out


class WrongShapeBridge(NumpyBridge):
    name = "np-wrongshape"

    def encode(self, samples):
        feats, pos, used = super().encode(samples)
        return feats[0], pos, used  # [N, D] instead of [B, N, D]


class TwoTupleBridge(NumpyBridge):
    """A malformed bridge: ``(feats, pos)`` without ``used``."""

    name = "np-twotuple"

    def encode(self, samples):
        feats, pos, _ = super().encode(samples)
        return feats, pos


class BadUsedBridge(NumpyBridge):
    name = "np-badused"
    variant = "missing"

    def encode(self, samples):
        feats, pos, used = super().encode(samples)
        if self.variant == "missing":
            used = [{} for _ in used]
        elif self.variant == "short":
            used = used[:-1]
        elif self.variant == "illformed":
            used = [{"video": {"times": "soon"}} for _ in used]
        return feats, pos, used


class BatchStatsBridge(NumpyBridge):
    """Normalizes features with batch statistics: fails the alone pass."""

    name = "np-batchstats"

    def encode(self, samples):
        feats, pos, used = super().encode(samples)
        return feats - feats.mean(axis=0, keepdims=True) * 0.5, pos, used


class ClipIdCacheBridge(NumpyBridge):
    """Caches features by clip_id and serves a different value on a cache miss pattern: the scrambled pass
    changes clip ids, so features change (alone pass)."""

    name = "np-clipcache"

    def encode(self, samples):
        feats, pos, used = super().encode(samples)
        for k, s in enumerate(samples):
            feats[k] += _seed(s.clip_id) % 7  # identity leaks into the features
        return feats, pos, used


class OrderBridge(NumpyBridge):
    """Returns the batch's features sorted by clip_id instead of in input order (a dict-keyed bug): correct for
    row-ordered batches and for batches of one, wrong once the batch is shuffled (order pass)."""

    name = "np-order"

    def encode(self, samples):
        feats, pos, used = super().encode(samples)
        order = sorted(range(len(samples)), key=lambda k: samples[k].clip_id)
        return feats[order], pos, used


class DropoutBridge(NumpyBridge):
    """Fresh noise every call (dropout left on): fails the rerun pass."""

    name = "np-dropout"

    def __init__(self, bench, args=None):
        super().__init__(bench, args)
        self.rng = np.random.default_rng(0)

    def encode(self, samples):
        feats, pos, used = super().encode(samples)
        return feats + self.rng.standard_normal(feats.shape).astype(np.float32) * 0.1, pos, used


class NetworkBridge(NumpyBridge):
    """Opens a TCP connection inside encode (a hub download): ``E_NETWORK``."""

    name = "np-network"

    def encode(self, samples):
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.01)
            s.connect(("127.0.0.1", 9))
        except OSError:
            pass  # swallowed on purpose: the audit hook still records it
        finally:
            s.close()
        return super().encode(samples)


class WritingBridge(NumpyBridge):
    """Writes a file outside the run directory during encode: ``W_SIDE_EFFECT``."""

    name = "np-writer"
    target: str = ""

    def encode(self, samples):
        with open(self.target, "w") as fh:
            fh.write("cache")
        return super().encode(samples)


class UndecodableBridge(NumpyBridge):
    """Cannot decode some clips: raises ``MediaUndecodable`` naming them."""

    name = "np-undecodable"
    bad_rows: frozenset[int] = frozenset({3})

    def encode(self, samples):
        from yavih.bridge import MediaUndecodable

        bad = [s.clip_id for s in samples if s.row in self.bad_rows]
        if bad:
            raise MediaUndecodable(bad, "moov atom not found")
        return super().encode(samples)


class WrongTimesBridge(NumpyBridge):
    """Declares uniform n=4 but reports the first 4 frames: ``E_RULE_MISMATCH`` in bridge check."""

    name = "np-wrongtimes"

    def encode(self, samples):
        feats, pos, used = super().encode(samples)
        return feats, pos, [{"video": {"times": [0.0, 0.04, 0.08, 0.12]}} for _ in used]


class NoPreprocBridge(NumpyBridge):
    name = "np-nopreproc"
    __doc__ = """A bridge whose template slot was never filled.

    Preprocessing source: TODO
    """


def describe_of(bridge_cls: type[Bridge], bench=None, args=None) -> dict:
    from yavih.bridge import BenchmarkView

    return bridge_cls(BenchmarkView.from_benchmark(bench or FakeBenchmark()), args or {}).describe()


def view_of(bench=None):
    from yavih.bridge import BenchmarkView

    return BenchmarkView.from_benchmark(bench or FakeBenchmark())
