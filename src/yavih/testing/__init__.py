"""Test fixtures shipped in the wheel: synthetic benchmark folders and example bridges.

- ``make_synthetic_benchmark(tmp, readout, n_train, n_eval, media)`` writes a complete schema-2 benchmark
  folder (``benchmark.yaml`` with ``input_contract`` / ``output_contract`` / ``external``,
  ``README.md`` and ``setup.md`` in the forms ``bench lint`` checks, a ``reader.py`` and ``bridge_template.py``)
  for each readout: ``clf`` K400-like (one mp4), ``seq`` AVE-like (mp4 + wav, per-second labels, facts
  ``num_segments``/``segment_seconds``), ``grid`` NYUv2-like (one png, depth targets over a
  ``target_region`` of the full frame, fact ``target_region``), ``box`` VGG-SS-like (mp4 + wav, boxes in
  full-frame fractions, fact ``annotated_second``). ``media="tiny"`` (the default) writes tiny raw media
  (PyAV mp4, soundfile wav, pillow png): what the reader hands over is the stored file, so synthetic
  benchmarks always carry media. ``census=True`` also writes the site's media-info table.
- ``random_bridge``: ``feat = default_rng(sha256(f"{seed}:{bench}:" + sha256 of the bytes of every
  stream it reads)).standard_normal((N, D))``, a pure function of the MEDIA (so it passes ``bridge check``'s hard checks and the
  scrambled-identity alone pass); it declares ``custom`` rules (the legitimate opt-out, validate notes
  ``N_RULE_CUSTOM``) and reports empty ``used``. ``describe()["model"]`` is ``yavih-random``.
- ``oracle_bridge``: features that encode the label so every readout clears its floor. It breaks
  the bridge guardrails on purpose — it reads the folder's manifest and finds a row from the media file it is handed — which
  is why ``bridge check`` refuses it unless ``YAVIH_TESTING=1``. Unlike the random bridge it declares
  CHECKABLE rules (``uniform`` / ``explicit`` frame times, an audio ``span``, a full-frame spatial
  rule), decodes with ``yavih.media.decode`` and reports the decoded pts as ``used``, so the end-to-end test
  exercises validate's ``E_RULE_MISMATCH`` / ``E_TIMES_OUT_OF_MEDIA`` checks on real members.
  ``describe()["model"]`` is ``yavih-oracle``.
- Both declare ``describe()["reading"] = "synthetic"`` (a declared tag), so their run directories are
  ``runs/<bench>/yavih-random/synthetic/`` and ``runs/<bench>/yavih-oracle/synthetic/``. ``model=`` /
  ``reading=`` (``random_bridge``, ``oracle_bridge``, ``write_bridge_file``) name them otherwise; ``name`` is the
  bridge's optional label, recorded only.
- ``fast(cfg)`` shrinks a pinned ``ProbeConfig`` for tests (gated on ``YAVIH_TESTING=1``).
- ``yavih.testing.folder``: the shared fixtures, loaders and assertions of the benchmark folders' own
  ``tests/test_folder.py``.

Positions are media coordinates: ``t`` = fraction of the stream's duration, ``(y, x)`` = fractions
of the ORIGINAL frame; grid/box cells tile the target region.

The module is core-only at import time (numpy + pyyaml + stdlib); media backends are imported inside
functions.
"""

from __future__ import annotations

import csv
import dataclasses
import gzip
import hashlib
import io
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml

from yavih.bridge import ArgSpec, BenchmarkView, Bridge
from yavih.hashing import rng_seed, sha256_hex
from yavih.versions import __version__

__all__ = [
    "make_synthetic_benchmark",
    "make_synthetic_like",
    "random_bridge",
    "oracle_bridge",
    "fast",
    "testing_enabled",
    "write_bridge_file",
    "read_manifest_rows",
    "synthetic_depth",
    "token_layout",
    "RandomBridgeBase",
    "OracleBridgeBase",
    "READOUTS",
    "MEDIA_MODES",
    "TINY_VIDEO",
    "TINY_AUDIO",
    "TINY_IMAGE",
    "GRID_TARGET_REGION",
]

#: The readouts this module builds synthetic fixtures for (the built-in ones; the readout registry may hold more).
READOUTS: tuple[str, ...] = ("clf", "seq", "grid", "box")
MEDIA_MODES: frozenset[str] = frozenset({"none", "tiny"})

#: Tiny media geometry (``media="tiny"``): 10 s mp4 at 2 fps 16×16, 10 s wav at 8 kHz, a 48×32 png
#: (not square, so full-frame fractions and the grid's target region are exercised for real).
TINY_VIDEO: dict[str, Any] = {"seconds": 10.0, "fps": 2, "size": 16, "codec": "mpeg4"}
TINY_AUDIO: dict[str, Any] = {"seconds": 10.0, "rate": 8000, "subtype": "PCM_16"}
TINY_IMAGE: dict[str, Any] = {"width": 48, "height": 32}

_CLASSES_DEFAULT = 5
_GRID_HW = 4  # synthetic grid/box cell grid (output_contract.requires.grid h = w)
_TARGET_SIZE = 32  # synthetic grid protocol.target_size (divisible by _GRID_HW)
_MAX_DEPTH = 10.0
#: the synthetic grid's scored region of the full frame (NYUv2-like: the centre of a wider frame)
GRID_TARGET_REGION: tuple[float, float, float, float] = (0.0, 0.125, 1.0, 0.875)
#: AVE v3: all ten one-second segments scored; 8 of 10 must hold a timed token
_SEQ = {"num_segments": 10, "segment_seconds": 1.0, "segments_scored": 10, "segment_offset": 0}
_SEQ_T_BINS = 8
_ANNOTATED_SECOND = 5.0  # box: the frame the box marks (VGG-SS-like fact)
_ORACLE_VIDEO_FRAMES = 5  # uniform n over span full on a 10 s, 2 fps clip: times 1, 3, 5, 7, 9 = stored pts
_NOISE = 0.05  # per-clip noise on oracle features (keeps rows distinct: E_DUPLICATE_CLIPS / E_DEAD_ROWS)
_INFO_KEYS = {
    "video": ["duration_s", "fps", "width", "height", "container", "bytes"],
    "audio": ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"],
    "image": ["width", "height", "container", "bytes"],
}
_HANDED_OVER = {
    "video": "the synthetic mp4 exactly as stored (10 s, 2 fps, 16 x 16, mpeg4); nothing decoded, cut or resized",
    "audio": "the synthetic wav exactly as stored (10 s, 8 kHz mono PCM_16); nothing decoded, resampled or padded",
    "image": "the synthetic png exactly as stored (48 x 32 RGB); nothing decoded, cropped or resized",
}


def testing_enabled(env: Mapping[str, str] | None = None) -> bool:
    """``YAVIH_TESTING=1`` gates non-pinned configs."""
    env = os.environ if env is None else env
    return env.get("YAVIH_TESTING", "") == "1"


# ============================================================================= helpers


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(header))
    for r in rows:
        w.writerow([str(v) for v in r])
    return buf.getvalue().encode("utf-8")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def read_manifest_rows(folder: str | os.PathLike, manifest_file: str = "manifest.csv.gz") -> list[dict[str, str]]:
    """Rows of ``<folder>/manifest.csv.gz`` as dicts in file order (stdlib only)."""
    with gzip.open(Path(folder) / manifest_file, "rb") as fh:
        raw = fh.read()
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))


def _read_classes(folder: Path, classes_file: str | None) -> list[dict[str, str]]:
    if not classes_file:
        return []
    text = (folder / classes_file).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def synthetic_depth(clip_id: str, size: int = _TARGET_SIZE, grid: int = _GRID_HW, max_depth: float = _MAX_DEPTH) -> np.ndarray:
    """The synthetic dense-depth target of ``clip_id``: float32 ``[size, size]`` = the raster of the synthetic
    ``target_region``, piecewise constant over a ``grid × grid`` cell lattice of that region, the
    top two pixel rows invalid (``0``). Depth = a per-clip level, log-uniform in ``[0.15, 0.85]·max_depth``,
    times a per-cell factor in ``[0.85, 1.15]``: the level carries most of the label and, being log-uniform,
    no constant prediction beats the floor (the per-pixel mean train log-depth map) — a feature set without
    depth information lands on the floor while one that encodes the level clears it. A pure function of
    ``clip_id`` so ``make_synthetic_benchmark`` (writer) and ``oracle_bridge`` (reader) agree."""
    rng = np.random.default_rng(rng_seed(f"yavih-synth-depth:{clip_id}"))
    level = math.exp(rng.uniform(math.log(0.15), math.log(0.85))) * max_depth
    cells = (level * rng.uniform(0.85, 1.15, size=(grid, grid))).astype(np.float32)
    rep = size // grid
    depth = np.kron(cells, np.ones((rep, rep), dtype=np.float32))
    depth[:2, :] = 0.0
    return depth


def _synthetic_box(clip_id: str) -> tuple[float, float, float, float]:
    """``(x1, y1, x2, y2)`` in full-frame fractions (the manifest's box form), a pure function of ``clip_id``."""
    rng = np.random.default_rng(rng_seed(f"yavih-synth-box:{clip_id}"))
    cx, cy = rng.uniform(0.3, 0.7, size=2)
    w, h = rng.uniform(0.2, 0.5, size=2)
    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    return (round(float(x1), 4), round(float(y1), 4), round(float(x2), 4), round(float(y2), 4))


# ============================================================================= tiny media writers


def _write_tiny_video(path: Path, clip_id: str) -> None:
    import av  # lazy: [media] extra

    size, fps, seconds = TINY_VIDEO["size"], TINY_VIDEO["fps"], TINY_VIDEO["seconds"]
    rng = np.random.default_rng(rng_seed(f"yavih-synth-video:{clip_id}"))
    base = rng.integers(0, 256, size=3)
    ramp = (np.arange(size, dtype=np.int64) * (256 // size)) % 256
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(TINY_VIDEO["codec"], rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(int(round(seconds * fps))):
            arr = np.empty((size, size, 3), dtype=np.uint8)
            arr[..., 0] = (base[0] + 11 * i) % 256
            arr[..., 1] = ramp[None, :]
            arr[..., 2] = (base[2] + ramp[:, None]) % 256
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)


def _write_tiny_audio(path: Path, clip_id: str) -> None:
    import soundfile as sf  # lazy: [media] extra

    rate, seconds = TINY_AUDIO["rate"], TINY_AUDIO["seconds"]
    rng = np.random.default_rng(rng_seed(f"yavih-synth-audio:{clip_id}"))
    freq = float(rng.uniform(200.0, 800.0))
    t = np.arange(int(round(seconds * rate)), dtype=np.float64) / rate
    wave = 0.3 * np.sin(2 * math.pi * freq * t) + 0.02 * rng.standard_normal(t.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.astype(np.float32), rate, subtype=TINY_AUDIO["subtype"])


def _write_tiny_image(path: Path, clip_id: str) -> None:
    from PIL import Image  # lazy: [media] extra

    w, h = TINY_IMAGE["width"], TINY_IMAGE["height"]
    rng = np.random.default_rng(rng_seed(f"yavih-synth-image:{clip_id}"))
    base = rng.integers(0, 256, size=3)
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([(base[0] + 4 * yy) % 256, (base[1] + 4 * xx) % 256, (base[2] + 2 * (yy + xx)) % 256], axis=-1).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(str(path), format="PNG")


# ============================================================================= benchmark folder


#: file extension of a synthetic stream file per kind
_EXT = {"video": "mp4", "audio": "wav", "image": "png"}
#: the default streams of each synthetic readout ``{name: kind}``; only ``clf`` takes others (``streams=``)
_DEFAULT_STREAMS: dict[str, dict[str, str]] = {
    "clf": {"video": "video"}, "seq": {"video": "video", "audio": "audio"}, "grid": {"image": "image"}, "box": {"video": "video", "audio": "audio"},
}


def _clf_columns(streams: Mapping[str, str] | None) -> dict[str, str]:
    """``{manifest column: stream}`` of a clf fixture: ``rel_media`` for the default single video stream, else
    ``rel_<stream>`` per stream."""
    if streams is None:
        return {"rel_media": "video"}
    return {f"rel_{s}": s for s in streams}


def _rows_for(readout: str, n_train: int, n_eval: int, n_classes: int, streams: Mapping[str, str] | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """Manifest header and rows per readout. Labels are balanced (``k % C``) so floors are low and
    deterministic; event spans / boxes / depth maps are pure functions of ``clip_id``."""
    rows: list[dict[str, Any]] = []
    n_bg = n_classes - 1  # seq: the last class is `background`
    for split, n in (("train", n_train), ("val", n_eval)):
        for k in range(n):
            cid = f"{split}_{k:04d}"
            if readout == "clf":
                label = f"class_{k % n_classes:02d}"
                row = {"clip_id": cid, "split": split, "label": label}
                for col, s in _clf_columns(streams).items():
                    row[col] = f"{split}/{label}/{cid}.{_EXT[(streams or {'video': 'video'})[s]]}"
                rows.append(row)
            elif readout == "seq":
                label = f"class_{k % n_bg:02d}"
                if k % 3 == 0:  # a full-clip event (not a boundary row)
                    start, end = 0, _SEQ["num_segments"]
                else:
                    rng = np.random.default_rng(rng_seed(f"yavih-synth-seq:{cid}"))
                    start = int(rng.integers(0, _SEQ["num_segments"] - 2))
                    end = int(rng.integers(start + 1, _SEQ["num_segments"]))
                rows.append(
                    {"clip_id": cid, "split": split, "label": label, "event_start": start, "event_end": end,
                     "rel_video": f"{split}/{cid}.mp4", "rel_audio": f"{split}/{cid}.wav"}
                )
            elif readout == "grid":
                rows.append({"clip_id": cid, "split": split, "rel_media": f"{split}/{cid}.png"})
            elif readout == "box":
                x1, y1, x2, y2 = _synthetic_box(cid)
                rows.append(
                    {"clip_id": cid, "split": split, "box": f"{x1}|{y1}|{x2}|{y2}", "rel_video": f"{split}/{cid}.mp4", "rel_audio": f"{split}/{cid}.wav"}
                )
            else:
                raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    header = list(rows[0].keys())
    return header, rows


def _classes_for(readout: str, n_classes: int) -> list[dict[str, str]] | None:
    if readout == "clf":
        return [{"index": str(i), "name": f"class_{i:02d}"} for i in range(n_classes)]
    if readout == "seq":
        names = [f"class_{i:02d}" for i in range(n_classes - 1)] + ["background"]
        return [{"index": str(i), "name": n} for i, n in enumerate(names)]
    return None


def _stream(kind: str, column: str) -> dict[str, Any]:
    return {"kind": kind, "media_column": column, "root": "media_root", "layout": "dir", "layout_params": {}, "required": True}


def _spec_for(
    readout: str,
    name: str,
    *,
    n_train: int,
    n_eval: int,
    n_classes: int,
    manifest_sha: str,
    classes_sha: str | None,
    media_dir: Path,
    derived_dir: Path,
    target_shas: dict[str, str],
    clf_streams: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The ``benchmark.yaml`` (schema 2) document for one synthetic readout."""
    background_index = n_classes - 1
    roots = {"media_root": {"default": str(media_dir), "description": "synthetic tiny media written by yavih.testing"}}
    base_protocol = {"train_draw": None, "views": "none"}
    facts: list[str] = []
    target_frame: str | None = None

    if readout == "clf":
        label = {"column": "label", "kind": "single"}
        kinds = dict(clf_streams or {"video": "video"})
        columns = {col: {"type": "str", "role": "media", "stream": s} for col, s in _clf_columns(clf_streams).items()}
        streams = {s: _stream(kinds[s], col) for col, s in _clf_columns(clf_streams).items()}
        timed = any(k != "image" for k in kinds.values())
        protocol = {"clip_seconds": TINY_VIDEO["seconds"] if timed else None, **base_protocol}
        metrics = {
            "headline": "acc", "list": ["acc", "n_eval"], "floors": ["acc_floor"], "controls": [],
            "chance": {"acc": round(1.0 / n_classes, 6)}, "ci": {"acc": "binomial"}, "bootstrap": {"samples": 2000, "seed": 0}, "aggregation": None,
        }
        requires: dict[str, Any] = {"pos": [], "streams": [], "grid": None, "t_bins": None}
        readout_params: dict[str, Any] = {}
        targets = None
        title = "Synthetic single-label clip classification (clf)"
    elif readout == "seq":
        label = {"column": "label", "kind": "segments"}
        columns = {
            "rel_video": {"type": "str", "role": "media", "stream": "video"},
            "rel_audio": {"type": "str", "role": "media", "stream": "audio"},
            "event_start": {"type": "int", "role": "label_aux"},
            "event_end": {"type": "int", "role": "label_aux"},
        }
        streams = {"video": _stream("video", "rel_video"), "audio": _stream("audio", "rel_audio")}
        protocol = {"clip_seconds": TINY_VIDEO["seconds"], **base_protocol, **_SEQ, "background_index": background_index}
        facts = ["num_segments", "segment_seconds"]
        metrics = {
            "headline": "seg_acc", "list": ["seg_acc", "clip_acc", "seg_acc_boundary", "n_boundary", "n_eval"],
            "floors": ["seg_acc_floor", "seg_acc_boundary_floor"], "controls": [], "chance": {"seg_acc": round(1.0 / n_classes, 6)},
            "ci": {"seg_acc": "bootstrap_clip", "seg_acc_boundary": "bootstrap_clip"}, "bootstrap": {"samples": 2000, "seed": 0}, "aggregation": None,
        }
        requires = {"pos": ["t"], "streams": ["video"], "grid": None, "t_bins": _SEQ_T_BINS}
        readout_params = {"T": _SEQ["segments_scored"]}
        targets = None
        title = "Synthetic audio-visual event segmentation (seq)"
    elif readout == "grid":
        label = {"column": None, "kind": "none"}
        columns = {"rel_media": {"type": "str", "role": "media", "stream": "image"}}
        streams = {"image": _stream("image", "rel_media")}
        protocol = {
            "clip_seconds": None, **base_protocol, "max_depth": _MAX_DEPTH, "target_size": _TARGET_SIZE,
            "target_region": list(GRID_TARGET_REGION),
        }
        facts = ["target_region"]
        target_frame = "full_frame"
        metrics = {
            "headline": "d1", "list": ["d1", "d2", "d3", "absrel", "rmse", "n_eval"], "floors": ["d1_floor"], "controls": [],
            "chance": {}, "ci": {"d1": "bootstrap_clip"}, "bootstrap": {"samples": 2000, "seed": 0}, "aggregation": None,
        }
        requires = {"pos": ["y", "x"], "streams": ["image"], "grid": {"h": _GRID_HW, "w": _GRID_HW, "min_cell_coverage": 0.5}, "t_bins": None}
        readout_params = {"h": _GRID_HW, "w": _GRID_HW, "out": 1, "visual_stream": "image"}
        roots["derived_root"] = {"default": str(derived_dir), "description": "synthetic derived depth targets"}
        targets = {
            split: {"kind": "dense_depth", "source": "derived_file", "root": "derived_root", "file": f"{split}_depth_{_TARGET_SIZE}.npy",
                    "sha256": target_shas[split], "index_column": None}
            for split in ("train", "val")
        }
        title = "Synthetic monocular depth on a cell lattice (grid)"
    elif readout == "box":
        label = {"column": "box", "kind": "box"}
        columns = {
            "rel_video": {"type": "str", "role": "media", "stream": "video"},
            "rel_audio": {"type": "str", "role": "media", "stream": "audio"},
        }
        streams = {"video": _stream("video", "rel_video"), "audio": _stream("audio", "rel_audio")}
        protocol = {"clip_seconds": TINY_VIDEO["seconds"], **base_protocol, "annotated_second": _ANNOTATED_SECOND}
        facts = ["annotated_second"]
        target_frame = "full_frame"
        metrics = {
            "headline": "iou_mean", "list": ["iou_mean", "acc@0.5", "acc@0.3", "center_in_gt", "n_eval", "n_excluded_degenerate"],
            "floors": ["iou_floor"], "controls": ["iou_shuffled", "audio_blind"], "chance": {}, "ci": {"iou_mean": "bootstrap_clip"},
            "bootstrap": {"samples": 2000, "seed": 0}, "aggregation": None,
        }
        requires = {"pos": ["y", "x"], "streams": ["video", "audio"], "grid": {"h": _GRID_HW, "w": _GRID_HW, "min_cell_coverage": 0.25}, "t_bins": None,
                    "annotated_frame": {"fact": "annotated_second", "stream": "video", "tol_s": 0.5}}
        readout_params = {"grid": _GRID_HW, "visual_stream": "video", "conditioning_stream": "audio", "shuffled_control": True, "shuffle_seed": 20260913}
        targets = None
        title = "Synthetic sound-source localization (box)"
    else:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")

    label_block = {
        "column": label["column"], "kind": label["kind"], "sep": ";", "class_key": "name", "sets": None, "train_set": None,
        "allow_unknown": False, "empty_ok": False,
    }
    derived = {t["file"]: t["sha256"] for t in targets.values()} if targets else {}
    n_rows = n_train + n_eval
    return {
        "schema_version": 3,
        "name": name,
        "aliases": [f"{name}_alias"],
        "title": title,
        "scorer": readout,
        "protocol_version": 1,
        "changelog": [
            {"version": 1, "date": "2026-10-12", "reason": "Synthetic test fixture (yavih.testing.make_synthetic_benchmark).", "invalidates": False},
        ],
        "header": {"name": name, "title": title, "venue": "yavih.testing · synthetic fixture",
                   "abstract": "A synthetic benchmark whose every label is a pure function of the row's clip_id."},
        "source": {"kind": "derived", "urls": [], "licence": "synthetic", "bytes": 0, "sha256": {},
                   "notes": "Generated by yavih.testing; every row is a pure function of its clip_id."},
        "budget": {"clips": {"train": n_train, "val": n_eval}, "media_gb": 0.0, "extract_min_per_1k_clips_ref": 0.1, "probe_minutes_ref": {readout: 1}},
        "manifest": {
            "file": "manifest.csv.gz",
            "sha256": manifest_sha,
            "classes": "classes.csv" if classes_sha else None,
            "classes_sha256": classes_sha,
            "key": "clip_id",
            "label": label_block,
            "columns": columns,
            "upstream": {
                "source": "yavih.testing.make_synthetic_benchmark",
                "sha256": manifest_sha,
                "rows": n_rows,
                "note": "all rows synthetic; labels balanced k % C",
                "upstream_files": {},
                "curation": {
                    "total_upstream": n_rows,
                    "per_split": {
                        "train": {"listed": n_train, "staged": n_train, "missing": 0},
                        "val": {"listed": n_eval, "staged": n_eval, "missing": 0},
                    },
                    "excluded": {},
                    "short_clips": 0,
                    "leakage_audit": None,
                },
            },
        },
        "splits": {"train": ["train"], "eval": ["val"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95},
        "fields": {n: {"storage": "file", **e} for n, e in streams.items()},
        "targets": targets,
        "data": {"roots": roots, "tar_index": None, "derived": derived},
        "input_contract": {
            "handed_over": {s: _HANDED_OVER[v["kind"]] for s, v in streams.items()},
            "info": {s: list(_INFO_KEYS[v["kind"]]) for s, v in streams.items()},
            "facts": facts,
            "materialize": None,
        },
        "output_contract": {"shape": "fixed", "coordinates": "media", "target_frame": target_frame, "requires": requires},
        "external": [],
        "protocol": protocol,
        "metrics": metrics,
        "readout_params": readout_params,
    }


# ----------------------------------------------------------------------------- folder documents


def _yaml_block(obj: Any) -> str:
    return "```yaml\n" + yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).rstrip("\n") + "\n```"


def _write_readme(folder: Path, spec: dict[str, Any], *, n_train: int, n_eval: int) -> None:
    """``README.md`` in the folder-contract form (header image, links, Description, Citation, In yavih)."""
    m = spec["metrics"]
    headline, floor = m["headline"], m["floors"][0]
    text = f"""![{spec['header']['name']}](assets/header.png)

Homepage: `yavih.testing.make_synthetic_benchmark` · [setup.md](setup.md)

## Description
A synthetic `{spec['scorer']}` benchmark: every label, box and depth map is a pure function of the row's
`clip_id`, so `oracle_bridge` can encode it and `random_bridge` cannot. Roles: train = `train` ({n_train} rows),
eval = `val` ({n_eval} rows).

## Citation
```bibtex
@misc{{yavih_synthetic, title = {{yavih synthetic benchmark {spec['name']}}}, note = {{test fixture}}}}
```

## In yavih
Scorer `{spec['scorer']}`; headline `metrics.headline` = `{headline}`, floor `{floor}`. `protocol_version`
{spec['protocol_version']}. What is handed over (the stored tiny media, raw) is in [setup.md](setup.md). Alias
`{spec['aliases'][0]}`; no other variant. `oracle_bridge` under `ProbeConfig.pinned(r).fast()` clears `{floor}`;
`random_bridge` does not.

Re-freezing: `python make_manifest.py` rewrites the folder byte-identically; `pytest tests/` checks it.
"""
    (folder / "README.md").write_text(text, encoding="utf-8")


def _write_setup(folder: Path, spec: dict[str, Any]) -> None:
    """``setup.md``, the agent's first read; each contract section ends with a yaml copy."""
    name = spec["name"]
    text = f"""# {name} — setup

## Obtain the data
Nothing to download: `bash "$FOLDER/prepare/01_generate.sh"` (`FOLDER=$(yavih bench path <folder>)`) regenerates the
folder, the tiny media under the `media_root` default and, for `grid`, the depth targets under `derived_root`.

## Dependencies
None. Writing the tiny media needs `yavih[media]` (PyAV, soundfile, pillow).

## Input contract
The reader hands over the stored files, raw; every frame, window, crop, resolution and audio front end is the
bridge's decision, declared in `describe()["input"]`.

{_yaml_block(spec["input_contract"])}

## Output contract
`[N, D]` tokens, fixed for every row; positions in media coordinates (`t` = fraction of the stream's duration,
`(y, x)` = fractions of the original frame). validate checks `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH` and
`E_BATCH_INVARIANCE` and decides eligibility.

{_yaml_block(spec["output_contract"])}

## Prepare & verify
`yavih bench verify {name} --census` reports full presence and writes the media-info table.

## Gotchas
The media are 16 x 16 videos at 2 fps, 8 kHz tones and 48 x 32 images: a bridge must not rely on pixel content.
"""
    (folder / "setup.md").write_text(text, encoding="utf-8")


_READER_PY = '''"""Reader of the synthetic benchmark (the folder's ``reader.py``): delegates to the generic
``yavih.sample.iter_samples``; imports only yavih core."""

from __future__ import annotations

from collections.abc import Iterator

from yavih.sample import iter_samples as _generic


def iter_samples(bench, split, rows, batch_size, canaries, canary_seed, *, shard=None, roots=None) -> Iterator:
    return _generic(bench, split, rows, batch_size, canaries, canary_seed, shard=shard, roots=roots)
'''

_BRIDGE_TEMPLATE_PY = '''"""Bridge template for the synthetic benchmark ``{name}`` (the folder's ``bridge_template.py``).

Preprocessing source:
    TODO: paste the model's own preprocessing code here, verbatim; describe()["input"] must match it line by line.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import Bridge


class TemplateBridge(Bridge):
    name = "TODO-model-{name}"  # optional label, recorded only; the model's name is describe()["model"]
    batch_size = 8
    dtype = "float16"
    pos_mode = "none"  # "per_sample" with positions in media coordinates for seq/grid/box
    streams = {streams!r}
    ARGS = {{}}

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)
        # TODO: build the model, load its weights here (never during encode), call .eval()

    def describe(self) -> dict:
        return {{
            "weights_sha256": None,  # TODO: yavih.bridge.fingerprint_state_dict(model)
            "weights_ref": "TODO",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "reading": "TODO",  # a | v | av: what the model sees
            "layer": "TODO",
            "pooling": "TODO",
            "views": 1,
            "input": {{}},  # TODO: per stream read, the declared temporal / spatial / front_end / norm / decoder rules
            "precision": {{"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True}},
            "args": dict(self.args),
        }}

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        # TODO: decode each RawStream (yavih.media.decode or the model's own stack), apply the declared rules,
        # run the model; report what was used per stream (``used``)
        feats = np.zeros((len(samples), 1, 1), dtype=np.float32)
        used = [{{}} for _ in samples]
        return feats, None, used
'''

_MAKE_MANIFEST_PY = '''"""Regenerate this synthetic benchmark folder byte-identically (the folder's ``make_manifest.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

from yavih.testing import make_synthetic_benchmark

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    make_synthetic_benchmark(here.parent, {readout!r}, n_train={n_train}, n_eval={n_eval}, media={media!r}, n_classes={n_classes}, name={name!r})
    sys.exit(0)
'''

_PREPARE_SH = """#!/usr/bin/env sh
# Synthetic benchmark: nothing to download. Regenerates the folder, media and derived targets.
set -eu
python "$(dirname "$0")/../make_manifest.py"
"""

_TEST_FOLDER_PY = '''"""Folder test of the synthetic benchmark (the folder's ``tests/test_folder.py``)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent


def test_folder():
    spec = yaml.safe_load((HERE / "benchmark.yaml").read_text())
    assert spec["name"] == HERE.name and spec["schema_version"] == 3
    with gzip.open(HERE / spec["manifest"]["file"], "rb") as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == spec["manifest"]["sha256"]
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    roles = spec["splits"]
    assigned = roles["train"] + roles["eval"] + roles["floor"] + roles["unused"] + (roles["folds"] or [])
    assert sorted(set(assigned)) == sorted({r["split"] for r in rows})
    cur = spec["manifest"]["upstream"]["curation"]
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"]
'''


def make_synthetic_benchmark(
    tmp: str | os.PathLike,
    readout: str,
    n_train: int = 64,
    n_eval: int = 32,
    media: str = "tiny",
    *,
    n_classes: int = _CLASSES_DEFAULT,
    name: str | None = None,
    census: bool = False,
    streams: Mapping[str, str] | None = None,
) -> Path:
    """Write ``<tmp>/<name>/{README.md, setup.md, benchmark.yaml, manifest.csv.gz, classes.csv?, reader.py,
    bridge_template.py, make_manifest.py, prepare/01_generate.sh, tests/test_folder.py}`` that pass ``bench lint``
    for ``readout`` (default name ``synth_<readout>``). ``media="tiny"`` writes the tiny raw media under
    ``<tmp>/media/<name>/`` (PyAV mp4, soundfile wav, pillow png); ``grid`` always writes its derived depth
    targets under ``<tmp>/derived/<name>/``. The data roots in the yaml default to those absolute
    directories, so no config is needed. ``census=True`` (with tiny media) also writes the site's media-info
    table under ``$YAVIH_HOME/media_info``, as ``yavih bench verify --census`` would. ``streams``
    (``{name: kind}``, kinds video / audio / image) replaces a ``clf`` fixture's single ``video`` stream; the other
    readouts accept only their own streams. Returns the folder. Re-running with the same arguments is
    byte-identical (a re-freeze changes nothing)."""
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    if media not in MEDIA_MODES:
        raise ValueError(f"media must be one of {sorted(MEDIA_MODES)}, got {media!r}")
    if n_train < 1 or n_eval < 1:
        raise ValueError("n_train and n_eval must be >= 1")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if streams is not None:
        streams = dict(streams)
        bad = sorted(k for k in streams.values() if k not in _EXT)
        if bad or not streams:
            raise ValueError(f"streams must map names to kinds in {sorted(_EXT)}, got {streams}")
        if readout != "clf" and streams != _DEFAULT_STREAMS[readout]:
            raise ValueError(f"a synthetic {readout} benchmark has the streams {_DEFAULT_STREAMS[readout]}, got {streams}")
        if readout != "clf":
            streams = None
    tmp = Path(tmp)
    name = name or f"synth_{readout}"
    folder = tmp / name
    media_dir = tmp / "media" / name
    derived_dir = tmp / "derived" / name
    folder.mkdir(parents=True, exist_ok=True)

    header, rows = _rows_for(readout, n_train, n_eval, n_classes, streams)
    manifest_bytes = _csv_bytes(header, [[r[h] for h in header] for r in rows])
    # mtime=0: a re-written fixture is byte-identical.
    with open(folder / "manifest.csv.gz", "wb") as raw_fh, gzip.GzipFile(fileobj=raw_fh, mode="wb", mtime=0) as fh:
        fh.write(manifest_bytes)
    classes = _classes_for(readout, n_classes)
    classes_sha = None
    if classes is not None:
        classes_bytes = _csv_bytes(["index", "name"], [[c["index"], c["name"]] for c in classes])
        (folder / "classes.csv").write_bytes(classes_bytes)
        classes_sha = _sha256(classes_bytes)

    target_shas: dict[str, str] = {}
    if readout == "grid":
        derived_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            cids = [r["clip_id"] for r in rows if r["split"] == split]
            depth = np.stack([synthetic_depth(c) for c in cids]).astype(np.float32)
            path = derived_dir / f"{split}_depth_{_TARGET_SIZE}.npy"
            buf = io.BytesIO()
            np.lib.format.write_array(buf, depth, version=(1, 0), allow_pickle=False)
            path.write_bytes(buf.getvalue())
            target_shas[split] = _sha256(buf.getvalue())

    spec = _spec_for(
        readout, name, n_train=n_train, n_eval=n_eval, n_classes=n_classes, manifest_sha=_sha256(manifest_bytes),
        classes_sha=classes_sha, media_dir=media_dir, derived_dir=derived_dir, target_shas=target_shas, clf_streams=streams,
    )
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if media == "tiny":
        for r in rows:
            for col, stream in ((c, s["stream"]) for c, s in spec["manifest"]["columns"].items() if s["role"] == "media"):
                kind = spec["fields"][stream]["kind"]
                path = media_dir / r[col]
                if path.exists():
                    continue
                if kind == "video":
                    _write_tiny_video(path, r["clip_id"])
                elif kind == "audio":
                    _write_tiny_audio(path, r["clip_id"])
                elif kind == "image":
                    _write_tiny_image(path, r["clip_id"])

    _write_readme(folder, spec, n_train=n_train, n_eval=n_eval)
    _write_setup(folder, spec)
    (folder / "reader.py").write_text(_READER_PY, encoding="utf-8")
    (folder / "bridge_template.py").write_text(_BRIDGE_TEMPLATE_PY.format(name=name, streams=tuple(spec["fields"])), encoding="utf-8")
    (folder / "make_manifest.py").write_text(
        _MAKE_MANIFEST_PY.format(readout=readout, n_train=n_train, n_eval=n_eval, media=media, n_classes=n_classes, name=name), encoding="utf-8"
    )
    (folder / "prepare").mkdir(exist_ok=True)
    sh = folder / "prepare" / "01_generate.sh"
    sh.write_text(_PREPARE_SH, encoding="utf-8")
    sh.chmod(0o755)
    (folder / "tests").mkdir(exist_ok=True)
    (folder / "tests" / "test_folder.py").write_text(_TEST_FOLDER_PY, encoding="utf-8")

    if census:
        if media != "tiny":
            raise ValueError("census=True needs media='tiny' (the census probes every present file)")
        from yavih.benchmark import Benchmark
        from yavih.media.info import write_census

        write_census(Benchmark.from_dir(folder))
    return folder


#: media facts a tiny synthetic file reliably carries, per kind (``make_synthetic_like`` guarantees no others)
_TINY_INFO = {
    "video": ("duration_s", "fps", "n_frames", "width", "height", "container", "bytes"),
    "audio": ("duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"),
    "image": ("width", "height", "container", "bytes"),
}


def make_synthetic_like(tmp: str | os.PathLike, bench: Any, *, n_eval: int = 32, n_train: int = 2) -> Path:
    """A tiny stand-in for a real benchmark (``yavih bridge check --synthetic``): written by
    ``make_synthetic_benchmark`` under the real benchmark's NAME and readout, with its stream names and kinds, the
    media facts its ``input_contract.info`` guarantees (as far as a tiny file carries them) and its label facts'
    values, over tiny generated media (10 s 16 × 16 mp4 at 2 fps, 10 s 8 kHz wav, 48 × 32 png). Labels, sizes and
    media content are synthetic: a bridge's shapes, rules, invariance and declarations can be smoke-tested before
    any data is staged; nothing measured on it (speed, numbers) carries over. ``bench`` is a ``Benchmark`` or a
    reference ``Benchmark.load`` accepts. Needs the media writers (PyAV, soundfile, pillow)."""
    from yavih.benchmark import Benchmark

    real = bench if hasattr(bench, "spec") else Benchmark.load(os.fspath(bench))
    kinds = {s: st.kind for s, st in real.spec.streams.items()}
    unsupported = sorted(s for s, k in kinds.items() if k not in _EXT)
    if unsupported:
        raise ValueError(f"{real.name}: stream(s) {unsupported} of kind(s) {[kinds[s] for s in unsupported]} have no synthetic stand-in")
    folder = make_synthetic_benchmark(tmp, real.spec.readout, n_train=n_train, n_eval=n_eval, media="tiny", name=real.name, streams=kinds)
    path = folder / "benchmark.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    real_info = dict(real.spec.raw["input_contract"].get("info") or {})
    ic = spec["input_contract"]
    for s, kind in kinds.items():
        want = [k for k in real_info.get(s, []) if k in _TINY_INFO[kind]]
        ic["info"][s] = want + [k for k in ic["info"][s] if k not in want]
    for key in ic["facts"]:
        if key in real.spec.protocol and type(real.spec.protocol[key]) is type(spec["protocol"].get(key)):
            spec["protocol"][key] = real.spec.protocol[key]
    spec["title"] = f"Synthetic stand-in for {real.name} (yavih bridge check --synthetic)"
    path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return folder


# ============================================================================= bridges


def _view(bench: Any) -> BenchmarkView:
    return BenchmarkView.from_benchmark(bench)


def token_layout(view: Any, N: int) -> tuple[list[str], list[dict[str, Any]], np.ndarray]:
    """``(stream names in yaml order, token descriptors, pos float32 [N_total, 4])`` in MEDIA coordinates, read
    from what a bridge may see (the ``BenchmarkView``: ``output_contract.requires``, ``streams``, ``facts``;
    the layout validate's ``N_ELIGIBILITY`` judges):

    - ``pos ⊇ {y, x}``: one token per cell of ``requires.grid``, at the cell centres of the target region
      (``facts.target_region``, else the full frame) in full-frame fractions, tagged with the first required
      visual stream (``t = -1`` for an image, the annotated second / the clip centre for a video); a second
      required stream (box: the conditioning audio) adds ``N`` tokens with ``t`` centres;
    - ``pos ⊇ {t}``: ``max(N, num_segments, t_bins)`` tokens at ``t = (k + 0.5)/n`` tagged with the first required
      stream, so every scored second holds one;
    - otherwise ``N`` tokens tagged with stream 0 (``t`` centres, or ``-1`` for an image), ``y = x = -1``.
    """
    v = _view(view)
    kinds = dict(v.streams)
    names = list(kinds)
    oc = dict(v.output_contract or {})
    req = dict(oc.get("requires") or {})
    axes = set(req.get("pos") or ())
    req_streams = [s for s in (req.get("streams") or ()) if s in kinds]
    facts = dict(v.facts or {})
    tokens: list[dict[str, Any]] = []
    if {"y", "x"} & axes:
        grid = dict(req.get("grid") or {})
        h, w = int(grid.get("h", _GRID_HW)), int(grid.get("w", _GRID_HW))
        visual = next((s for s in req_streams if kinds[s] in ("video", "image")), req_streams[0] if req_streams else names[0])
        y0, x0, y1, x1 = (float(c) for c in (facts.get("target_region") or (0.0, 0.0, 1.0, 1.0)))
        if kinds[visual] == "image":
            t_vis = -1.0
        else:
            # constant pos: every synthetic clip lasts TINY_VIDEO["seconds"] (the annotated frame, else the centre)
            t_vis = float(facts["annotated_second"]) / float(TINY_VIDEO["seconds"]) if "annotated_second" in facts else 0.5
        vis_tag = names.index(visual)
        for i in range(h):
            for j in range(w):
                tokens.append({"kind": "cell", "stream": vis_tag, "t": t_vis, "y": y0 + (i + 0.5) * (y1 - y0) / h,
                               "x": x0 + (j + 0.5) * (x1 - x0) / w, "cell": (i, j), "hw": (h, w)})
        cond = next((s for s in req_streams if s != visual), None)
        if cond is not None:
            cond_tag = names.index(cond)
            for k in range(N):
                tokens.append({"kind": "cond", "stream": cond_tag, "t": (k + 0.5) / N, "y": -1.0, "x": -1.0})
    elif "t" in axes:
        n_t = max(int(N), int(facts.get("num_segments") or 0), int(req.get("t_bins") or 0))
        tag = names.index(req_streams[0]) if req_streams else 0
        for k in range(n_t):
            tokens.append({"kind": "time", "stream": tag, "t": (k + 0.5) / n_t, "y": -1.0, "x": -1.0})
    else:
        image = kinds[names[0]] == "image"
        for k in range(N):
            tokens.append({"kind": "global", "stream": 0, "t": -1.0 if image else (k + 0.5) / N, "y": -1.0, "x": -1.0})
    pos = np.array([[tk["stream"], tk["t"], tk["y"], tk["x"]] for tk in tokens], dtype=np.float32)
    return names, tokens, pos


_PRECISION = {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True}


class _SyntheticBridge(Bridge):
    """Shared plumbing of the two test bridges: stream tags, pos in media coordinates, describe(), encode()."""

    name: ClassVar[str] = "yavih-synthetic"
    batch_size: ClassVar[int] = 4
    dtype: ClassVar[str] = "float32"
    pos_mode: ClassVar[str] = "constant"
    N: ClassVar[int] = 4
    D: ClassVar[int] = 16
    #: ``describe()["model"]``: the name results are filed under. Not ``model`` (bridges keep their
    #: torch module there).
    model_name: ClassVar[str] = "yavih-synthetic"
    #: ``describe()["reading"]``: a declared tag, ``[A-Za-z0-9_-]{1,32}``.
    reading: ClassVar[str] = "synthetic"
    _bench_name: ClassVar[str | None] = None
    _bench_folder: ClassVar[str | None] = None

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        view = _view(bench)
        self.view = view
        self.bench_name: str = self._bench_name or view.name
        streams, tokens, pos = token_layout(view, self.N)
        # Instance attributes shadow the ClassVars on purpose: the tags depend on the benchmark.
        self.streams = tuple(streams)
        self._kinds = dict(view.streams)
        self._tokens = tokens
        self._pos = pos
        self.n_tokens = int(pos.shape[0])

    def _weights_sha256(self) -> str:
        raise NotImplementedError

    def _extra_describe(self) -> dict:
        return {}

    def _input(self) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "weights_sha256": self._weights_sha256(),
            "weights_ref": "yavih.testing",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "model": self.model_name,
            "reading": self.reading,
            "layer": 0,
            "pooling": "none",
            "views": 1,
            "input": self._input(),
            "precision": dict(_PRECISION),
            "args": self._echo_args(),
            "N": self.n_tokens,
            "D": self.D,
            **self._extra_describe(),
        }

    def _echo_args(self) -> dict:
        return {}

    def _feat(self, sample: Any) -> np.ndarray:
        raise NotImplementedError

    def _used(self, sample: Any) -> dict:
        raise NotImplementedError

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        feats, used = [], []
        for s in samples:
            feats.append(self._feat(s).astype(self.dtype, copy=False))
            used.append(self._used(s))
        out = np.stack(feats).astype(self.dtype)
        if self.pos_mode == "none":
            pos = None
        elif self.pos_mode == "per_sample":
            pos = np.repeat(self._pos[None], len(samples), axis=0)
        else:
            pos = self._pos.copy()
        return out, pos, used

    def profile(self) -> dict | None:
        return {"peak_gpu_gb": 0.0}


class RandomBridgeBase(_SyntheticBridge):
    """``random_bridge``: ``feat = default_rng(sha256(f"{seed}:{bench}:" + sha256 of the bytes of every stream
    it reads)).standard_normal((N, D))``, a pure function of the media (bit-identical canaries, batch- and
    order-invariant, unaffected by the scrambled identity of bridge check's alone pass). Declares ``custom``
    rules (the declared opt-out: validate notes ``N_RULE_CUSTOM``, never a warning) and reports empty
    ``used``. ``--arg seed=K`` is the one consumed argument; a seed change changes ``describe_sha256``, not the
    model name."""

    name: ClassVar[str] = "yavih-random"
    model_name: ClassVar[str] = "yavih-random"
    seed: ClassVar[int] = 0
    ARGS: ClassVar[dict[str, ArgSpec | dict]] = {"seed": ArgSpec(type=int, default=0)}

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        self.seed = int(self.args["seed"]) if self.args.get("seed") is not None else int(type(self).seed)

    def _echo_args(self) -> dict:
        return {"seed": self.seed}

    def _weights_sha256(self) -> str:
        return sha256_hex(f"yavih-random:{self.seed}:{self.n_tokens}:{self.D}".encode())

    def _input(self) -> dict:
        out: dict[str, Any] = {}
        for s, kind in self._kinds.items():
            e: dict[str, Any] = {
                "norm": {"source": "none: yavih.testing.random_bridge hashes the stored bytes"},
                "decoder": {"lib": "none (sha256 of the stored bytes)", "version": __version__},
            }
            if kind != "image":
                e["temporal"] = {"rule": "custom", "description": "no decoding: the stored bytes are hashed"}
            if kind in ("video", "image"):
                e["spatial"] = {"rule": "custom", "description": "no decoding: the stored bytes are hashed"}
            if kind == "audio":
                e["front_end"] = {"sample_rate": 1, "resample": "none", "features": "sha256 of the stored bytes"}
            out[s] = e
        return out

    def media_digest(self, sample: Any) -> str:
        """``sha256`` hex of every stream's stored bytes, in stream-name order, concatenated."""
        parts = []
        for name in sorted(self._kinds):
            st = sample.streams.get(name)
            if st is None:
                continue
            parts.append(_sha256(st.bytes()))
        return "".join(parts)

    def _feat(self, sample: Any) -> np.ndarray:
        rng = np.random.default_rng(rng_seed(f"{self.seed}:{self.bench_name}:{self.media_digest(sample)}"))
        return rng.standard_normal((self.n_tokens, self.D))

    def _used(self, sample: Any) -> dict:
        out: dict[str, Any] = {}
        for s, kind in self._kinds.items():
            if kind == "video":
                out[s] = {"times": []}
            elif kind == "audio":
                out[s] = {"samples": [], "pad": [0, 0]}
            else:
                out[s] = {}
        return out


def _codebook(n_classes: int, D: int) -> np.ndarray:
    """Fixed ±1 class codes ``[C, D]`` (seeded); the oracle's "weights"."""
    rng = np.random.default_rng(rng_seed(f"yavih-oracle-codebook:{n_classes}:{D}"))
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_classes, D))


class OracleBridgeBase(_SyntheticBridge):
    """``oracle_bridge``: features that encode the row's label so every readout clears its floor.

    It breaks the bridge guardrails on purpose: it reads the benchmark folder's manifest and finds the row from the media FILE
    it is handed (so it survives bridge check's scrambled ``clip_id``), which is why ``bridge check`` refuses it
    unless ``YAVIH_TESTING=1``. Everything else is what a real bridge does: it decodes each stream with
    ``yavih.media.decode`` under declared, checkable rules (video ``uniform`` n=5 over the whole clip — seq: one
    frame per timed token, at its time —, or the ``explicit`` annotated second for box; audio ``span`` [0, 10 s) padded with zeros; full-frame spatial rule)
    and reports the decoded pts / sample ranges as ``used``.

    - ``clf``: every token = the class code (±1 codebook row) of the label;
    - ``seq``: the token at ``t`` carries the code of the class of absolute second ``floor(t · duration_s)``
      (``event_start ≤ s < event_end`` → label, else ``background_index``);
    - ``grid``: cell token ``(i, j)`` of the target region carries ``log(depth)`` and ``depth / max_depth`` of
      that cell of the synthetic target (``synthetic_depth``);
    - ``box``: conditioning (audio) tokens carry ``(cx, cy, w, h) · 2 − 1``; visual cell tokens an inside-box
      indicator (full-frame fractions) and the cell centre.

    Small media-seeded noise keeps rows distinct (``E_DUPLICATE_CLIPS``, ``E_DEAD_ROWS``).
    """

    name: ClassVar[str] = "yavih-oracle"
    model_name: ClassVar[str] = "yavih-oracle"
    ORACLE: ClassVar[bool] = True

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        folder = self._resolve_folder(bench)
        spec = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
        manifest = spec["manifest"]
        label = manifest["label"]
        self.readout = str(spec["scorer"])
        self.label_column = label.get("column")
        self.label_sep = str(label.get("sep") or ";")
        class_key = str(label.get("class_key") or "name")
        self.protocol = dict(spec.get("protocol") or {})
        self.readout_params = dict(spec.get("readout_params") or {})
        rows = read_manifest_rows(folder, str(manifest.get("file", "manifest.csv.gz")))
        self.rows = {r[manifest.get("key", "clip_id")]: r for r in rows}
        media_cols = [c for c, v in (manifest.get("columns") or {}).items() if v.get("role") == "media"]
        # the guardrail violation: media cell (a relative path) → row
        self._by_media: dict[str, dict[str, str]] = {}
        for r in rows:
            for c in media_cols:
                if r.get(c):
                    self._by_media[r[c]] = r
        classes = _read_classes(folder, manifest.get("classes"))
        self.class_index = {c[class_key]: int(c["index"]) for c in classes}
        self.n_classes = max(len(classes), 1)
        self.codebook = _codebook(self.n_classes, self.D)

    def _resolve_folder(self, bench: Any) -> Path:
        if self._bench_folder:
            return Path(self._bench_folder)
        folder = getattr(bench, "folder", None)
        if folder is not None:
            return Path(folder)
        from yavih.benchmark import Benchmark

        return Path(Benchmark.load(self._bench_name or _view(bench).name).folder)

    def _weights_sha256(self) -> str:
        return sha256_hex(np.ascontiguousarray(self.codebook).tobytes())

    def _extra_describe(self) -> dict:
        return {"oracle_readout": self.readout}

    # -- declared rules -------------------------------------------------
    def _temporal(self, kind: str) -> dict:
        if kind == "audio":
            return {"rule": "span", "start_s": 0.0, "length_s": float(TINY_AUDIO["seconds"]), "pad": "zeros"}
        if self.readout == "box":
            return {"rule": "explicit", "times": [_ANNOTATED_SECOND]}
        if self.readout == "seq":
            # one frame per timed token, at the token's own time: pos t = used time / duration_s
            n_t = sum(1 for tk in self._tokens if tk.get("kind") == "time")
            return {"rule": "uniform", "n": max(1, n_t), "span": "full"}
        return {"rule": "uniform", "n": _ORACLE_VIDEO_FRAMES, "span": "full"}

    def _input(self) -> dict:
        from yavih.media.decode import decoder_info

        out: dict[str, Any] = {}
        for s, kind in self._kinds.items():
            e: dict[str, Any] = {
                "norm": {"scale": 1.0, "source": "none: yavih.testing.oracle_bridge reads the label, not the pixels"},
                "decoder": decoder_info("audio" if kind == "audio" else ("image" if kind == "image" else "video")),
            }
            if kind != "image":
                e["temporal"] = self._temporal(kind)
            if kind in ("video", "image"):
                e["spatial"] = {"resize": None, "crop": None, "interp": "none", "antialias": False}
            if kind == "audio":
                e["front_end"] = {"sample_rate": int(TINY_AUDIO["rate"]), "resample": "none", "features": "raw waveform", "channels": "mean"}
            out[s] = e
        return out

    def _used(self, sample: Any) -> dict:
        """Decode what the declared rules say and report it (``used``): decoded pts, sample ranges."""
        from yavih.media import decode
        from yavih.media.rules import audio_ranges, rule_times

        out: dict[str, Any] = {}
        for s, kind in self._kinds.items():
            st = sample.streams[s]
            info = st.info
            if kind in ("video", "audio") and info.get("has_track") is False:  # no such track in the file
                out[s] = {"times": []} if kind == "video" else {"samples": [], "pad": [0, 0]}
            elif kind == "video":
                times = rule_times(self._temporal(kind), info)
                _, pts = decode.frames_at(st, times)
                out[s] = {"times": [float(t) for t in pts]}
            elif kind == "audio":
                ranges, (pl, pr) = audio_ranges(self._temporal(kind), info)
                decode.audio(st, ranges)
                out[s] = {"samples": [[int(a), int(b)] for a, b in ranges], "pad": [int(pl), int(pr)]}
            else:
                decode.image(st)
                out[s] = {}
        return out

    # -- labels ---------------------------------------------------------------------
    def _row(self, sample: Any) -> dict[str, str]:
        for st in sample.streams.values():
            # the stored path behind the opaque handle: reading it is exactly the guardrail violation the
            # oracle exists to commit
            path = getattr(st, "_real_path", None) or getattr(st, "path", None)
            if not path:
                continue
            parts = Path(path).parts
            for k in range(1, min(len(parts), 6) + 1):
                hit = self._by_media.get("/".join(parts[-k:]))
                if hit is not None:
                    return hit
        raise KeyError(f"oracle_bridge: none of the media of {getattr(sample, 'clip_id', '?')!r} is in {self.bench_name}'s manifest")

    def _label_ids(self, row: Mapping[str, str]) -> list[int]:
        cell = row.get(self.label_column or "", "") if self.label_column else ""
        if cell == "":
            return []
        return [self.class_index[v] for v in cell.split(self.label_sep) if v != ""]

    def _feat(self, sample: Any) -> np.ndarray:
        row = self._row(sample)
        cid = row["clip_id"]
        noise = _NOISE * np.random.default_rng(rng_seed(f"yavih-oracle-noise:{cid}")).standard_normal((self.n_tokens, self.D))
        feat = np.zeros((self.n_tokens, self.D), dtype=np.float64)
        if self.readout == "clf":
            ids = self._label_ids(row)
            feat[:] = self.codebook[ids].sum(axis=0) if ids else np.zeros(self.D)
        elif self.readout == "seq":
            ids = self._label_ids(row)
            bg = int(self.protocol.get("background_index", self.n_classes - 1))
            seg_s = float(self.protocol.get("segment_seconds", 1.0))
            start, end = int(row.get("event_start", 0)), int(row.get("event_end", 0))
            names = list(self._kinds)
            for n, tk in enumerate(self._tokens):
                dur = float(sample.streams[names[int(tk["stream"])]].info["duration_s"])
                s = int(math.floor(tk["t"] * dur / seg_s))  # absolute second
                cls = (ids[0] if ids else bg) if start <= s < end else bg
                feat[n] = self.codebook[cls]
        elif self.readout == "grid":
            size = int(self.protocol.get("target_size", _TARGET_SIZE))
            max_depth = float(self.protocol.get("max_depth", _MAX_DEPTH))
            h, w = self._tokens[0]["hw"]
            depth = synthetic_depth(cid, size=size, grid=h, max_depth=max_depth)
            valid = (depth > 0) & (depth <= max_depth)
            for n, tk in enumerate(self._tokens):
                i, j = tk["cell"]
                ys, xs = slice(i * size // h, (i + 1) * size // h), slice(j * size // w, (j + 1) * size // w)
                cell, mask = depth[ys, xs], valid[ys, xs]
                d = float(cell[mask].mean()) if mask.any() else max_depth / 2
                feat[n, 0] = math.log(d)
                feat[n, 1] = d / max_depth
                feat[n, 2] = (i + 0.5) / h * 2 - 1
                feat[n, 3] = (j + 0.5) / w * 2 - 1
        elif self.readout == "box":
            x1, y1, x2, y2 = (float(v) for v in str(row.get(self.label_column or "box", "")).split("|"))
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            for n, tk in enumerate(self._tokens):
                if tk["kind"] == "cond":
                    feat[n, :4] = np.array([cx, cy, w, h]) * 2 - 1
                    feat[n, 4] = 1.0
                else:
                    inside = x1 <= tk["x"] < x2 and y1 <= tk["y"] < y2
                    feat[n, 0] = 1.0 if inside else -1.0
                    feat[n, 1] = tk["y"] * 2 - 1
                    feat[n, 2] = tk["x"] * 2 - 1
                    feat[n, 5] = 1.0
        else:
            raise ValueError(f"oracle_bridge: unknown readout {self.readout!r}")
        return feat + noise


def _subclass(base: type, cls_name: str, bench: Any, **attrs: Any) -> type:
    bench_name, bench_folder = None, None
    if bench is not None:
        if isinstance(bench, str):
            bench_name = bench
        else:
            bench_name = str(_get(bench, "name"))
            folder = getattr(bench, "folder", None)
            bench_folder = None if folder is None else os.fspath(folder)
    attrs["_bench_name"] = bench_name
    attrs["_bench_folder"] = bench_folder
    attrs["__module__"] = __name__
    if "dtype" in attrs and attrs["dtype"] not in ("float16", "float32"):
        raise ValueError(f"dtype must be float16 or float32, got {attrs['dtype']!r}")
    if "pos_mode" in attrs and attrs["pos_mode"] not in ("none", "constant", "per_sample"):
        raise ValueError(f"pos_mode must be none|constant|per_sample, got {attrs['pos_mode']!r}")
    if "seed" in attrs:
        attrs["ARGS"] = {"seed": ArgSpec(type=int, default=int(attrs["seed"]))}
    for key in ("model_name", "reading"):
        if attrs.get(key) is None:
            attrs.pop(key, None)
    _check_names(attrs.get("model_name"), attrs.get("reading"))
    return type(cls_name, (base,), attrs)


_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_READING_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


def _check_names(model: str | None, reading: str | None) -> None:
    """The ``model`` / ``reading`` charsets, checked when the fixture is made (``validate_describe`` checks them again)."""
    if model is not None and not _MODEL_RE.fullmatch(model):
        raise ValueError(f"model must match [A-Za-z0-9_.-]{{1,64}} and start with a letter or digit, got {model!r}")
    if reading is not None and not _READING_RE.fullmatch(reading):
        raise ValueError(f"reading must match [A-Za-z0-9_-]{{1,32}} and start with a letter or digit, got {reading!r}")


def random_bridge(bench: Any, *, N: int = 4, D: int = 16, seed: int = 0, pos_mode: str = "constant", dtype: str = "float32", batch_size: int = 4,
                  model: str | None = None, reading: str | None = None) -> type[RandomBridgeBase]:
    """``Bridge`` subclass with ``feat = default_rng(sha256(f"{seed}:{bench}:" + media digest)).standard_normal((N, D))``. ``bench`` is a ``Benchmark`` or its name (``None`` = the benchmark the bridge is constructed with);
    the token count is ``N`` unless the output contract dictates a layout (see ``token_layout``). ``model`` /
    ``reading`` override ``describe()["model"]`` (``yavih-random``) and ``describe()["reading"]`` (``synthetic``)."""
    return _subclass(RandomBridgeBase, "RandomBridge", bench, N=int(N), D=int(D), seed=int(seed), pos_mode=pos_mode, dtype=dtype, batch_size=int(batch_size),
                     model_name=model, reading=reading)


def oracle_bridge(bench: Any, *, N: int = 4, D: int = 16, pos_mode: str = "constant", dtype: str = "float32", batch_size: int = 4,
                  model: str | None = None, reading: str | None = None) -> type[OracleBridgeBase]:
    """``Bridge`` subclass whose features encode the label; see ``OracleBridgeBase``. ``model`` /
    ``reading`` override ``describe()["model"]`` (``yavih-oracle``) and ``describe()["reading"]`` (``synthetic``)."""
    if D < 6:
        raise ValueError("oracle_bridge needs D >= 6")
    return _subclass(OracleBridgeBase, "OracleBridge", bench, N=int(N), D=int(D), pos_mode=pos_mode, dtype=dtype, batch_size=int(batch_size),
                     model_name=model, reading=reading)


_BRIDGE_FILE = '''"""Synthetic {kind} bridge written by ``yavih.testing.write_bridge_file``."""

import yavih.testing as _testing


class {cls}(_testing.{base}):
    """The {kind} test bridge.

    Preprocessing source:
        none: a yavih.testing fixture ({kind}); it runs no model, so there is no preprocessing to paste.
    """

    name = {name!r}
    model_name = {model!r}
    reading = {reading!r}
    N = {N}
    D = {D}
    pos_mode = {pos_mode!r}
    dtype = {dtype!r}
    batch_size = {batch_size}
{seed_lines}{folder_lines}'''


def write_bridge_file(
    path: str | os.PathLike,
    kind: str,
    *,
    N: int = 4,
    D: int = 16,
    seed: int = 0,
    pos_mode: str = "constant",
    dtype: str = "float32",
    batch_size: int = 4,
    name: str | None = None,
    model: str | None = None,
    reading: str = "synthetic",
    bench_folder: str | os.PathLike | None = None,
) -> Path:
    """Write a bridge FILE (what ``yavih extract --bridge P`` / ``bridge check`` load) defining the single
    ``Bridge`` subclass ``random_bridge``/``oracle_bridge`` would return; ``kind`` is ``"random"`` or ``"oracle"``.
    The benchmark is the one the bridge is constructed with (the oracle resolves its folder by name, or reads
    ``bench_folder`` when given: a stand-in that shares a packaged folder's name, ``yavih bench trial``).
    ``model`` is ``describe()["model"]`` (default ``yavih-<kind>``), ``reading`` ``describe()["reading"]``,
    ``name`` the optional label (default ``yavih-<kind>``, recorded only)."""
    if kind not in ("random", "oracle"):
        raise ValueError("kind must be 'random' or 'oracle'")
    model = model or f"yavih-{kind}"
    _check_names(model, reading)
    base = "RandomBridgeBase" if kind == "random" else "OracleBridgeBase"
    seed_lines = f"    seed = {int(seed)}\n    ARGS = {{'seed': _testing.ArgSpec(type=int, default={int(seed)})}}\n" if kind == "random" else ""
    text = _BRIDGE_FILE.format(
        kind=kind, cls="RandomBridge" if kind == "random" else "OracleBridge", base=base, name=name or f"yavih-{kind}",
        N=int(N), D=int(D), pos_mode=pos_mode, dtype=dtype, batch_size=int(batch_size), seed_lines=seed_lines,
        model=model, reading=reading,
        folder_lines=f"    _bench_folder = {os.fspath(Path(bench_folder).absolute())!r}\n" if bench_folder is not None else "",
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ============================================================================= fast probe config


def fast(cfg):
    """``ProbeConfig.pinned(r).fast()`` as a function: a non-pinned variant for tests
    (``result.standard = false``). Requires ``YAVIH_TESTING=1``; raises ``RuntimeError`` otherwise. The ONE
    definition is ``ProbeConfig.fast()`` (``yavih.probes.config``, torch-free); this delegates so both agree."""
    if not testing_enabled():
        raise RuntimeError("ProbeConfig.fast() is a test-only config: set YAVIH_TESTING=1")
    if not dataclasses.is_dataclass(cfg) or not callable(getattr(cfg, "fast", None)):
        raise TypeError(f"fast() expects a ProbeConfig dataclass, got {type(cfg).__name__}")
    return cfg.fast()
