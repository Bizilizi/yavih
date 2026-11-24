"""``nyuv2_depth`` probe hooks (the folder's setup.md): Probe3D's own depth metrics.

``d1``, ``d2``, ``d3``, ``rmse`` and the floor ``d1_floor`` are computed by the benchmark authors' code,
``evaluate_depth`` of Probe3D (El Banani et al., "Probing the 3D Awareness of Visual Foundation Models",
CVPR 2024; https://github.com/mbanani/probe3d at commit ``a1f14640076e38b8bc07b66d0fe2d01d15691e9d``,
``evals/utils/metrics.py``), COPIED into this folder as ``probe3d_metrics.py``: Probe3D is not on PyPI, and the
only install route (an editable install from a git URL) is one many agents refuse, so the folder carries the code.
The copy keeps ``evaluate_depth`` and ``match_scale_and_shift`` byte for byte (its header lists what was dropped);
``load_probe3d`` imports it by path and refuses a copy whose functions differ from ``PROBE3D_BLOCK_SHA256``.

What Probe3D computes (``evaluate_depth(pred, gt, image_average=False)``): per image, ``valid = gt > 0``,
``pred = pred * valid`` first, ``thresh = max(gt / pred.clamp(1e-9), pred / gt.clamp(1e-9))``, ``dk`` = valid
fraction with ``thresh < 1.25**k``, ``rmse = sqrt(mean over valid of (gt - pred)**2)``; this file averages the
per-image values over images (Probe3D's ``validate`` concatenates the per-image tensors and takes the mean).
The targets store invalid pixels as 0 and were filtered to ``0 < d <= max_depth`` when derived, so ``gt > 0``
is exactly yavih's validity predicate; values above ``max_depth`` are zeroed again here defensively.

``absrel`` (not in Probe3D's file) and ``n_eval`` stay yavih's registry metrics (``yavih.metrics.depth``).
The floor map (per-pixel mean TRAIN log-depth, ``yavih.metrics.depth.mean_log_depth_map``) is yavih's
construction; it is SCORED by Probe3D's function like any prediction (floors go through the same
metric function).

Module scope imports stdlib, numpy and yavih core only; torch and the copied Probe3D file load at call time.
Its normalized AST (comments and docstrings excluded) enters the scoring hash and so ``protocol_hash`` =
H(input_hash, scoring_hash); an edit here is scoring-side and never invalidates features.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

from yavih.metrics.registry import MetricSpec

#: Where probe3d_metrics.py was copied from: the commit and the sha256 of the upstream ``evals/utils/metrics.py``.
PROBE3D_COMMIT = "a1f14640076e38b8bc07b66d0fe2d01d15691e9d"
PROBE3D_METRICS_SHA256 = "39c0c919f0474b1a7b6d09eedba9dccda19abff921381343250c5058a44a34a8"
#: sha256 of the copied functions (probe3d_metrics.py after its ``import torch`` line), equal to the same two
#: functions of the upstream file at that commit.
PROBE3D_BLOCK_SHA256 = "d5085d79d0d6fa36cee057a002ce3e7f849357a7d0c99d7ea3548262070e5c1b"
PROBE3D_FILE = "probe3d_metrics.py"
PROBE3D_KEYS = ("d1", "d2", "d3", "rmse")
CHUNK = 64  # images per evaluate_depth call (memory only; per-image results are independent of it)

PROTOCOL_KEYS = frozenset({"max_depth", "target_size", "depth_param", "source_size", "target_region"})


class Probe3DMismatch(RuntimeError):
    """The copied ``probe3d_metrics.py`` no longer holds Probe3D's functions as copied."""


def copied_block(text: str) -> str:
    """The copied functions of ``probe3d_metrics.py``: everything after its ``import torch`` line."""
    head, sep, block = text.partition("import torch\n\n\n")
    if not sep:
        raise Probe3DMismatch(f"{PROBE3D_FILE}: no 'import torch' line before the copied functions")
    return block


def load_probe3d():
    """The copied Probe3D module (``evaluate_depth``), imported by path after the block-digest check."""
    path = Path(__file__).resolve().with_name(PROBE3D_FILE)
    data = path.read_bytes()
    got = hashlib.sha256(copied_block(data.decode("utf-8")).encode("utf-8")).hexdigest()
    if got != PROBE3D_BLOCK_SHA256:
        raise Probe3DMismatch(f"{path}: copied functions sha256 {got} != {PROBE3D_BLOCK_SHA256} "
                              f"(Probe3D commit {PROBE3D_COMMIT[:8]}, evals/utils/metrics.py)")
    name = f"yavih_folder_{path.parent.name}_probe3d_{hashlib.sha256(data).hexdigest()[:12]}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _check_protocol(protocol, targets: np.ndarray) -> None:
    s = int(protocol["target_size"])
    if targets.ndim != 3 or targets.shape[1:] != (s, s):
        raise ValueError(f"depth targets {targets.shape} are not [n, {s}, {s}] (protocol.target_size)")
    if protocol["depth_param"] != "log":
        raise ValueError(f"depth_param {protocol['depth_param']!r}: the pinned grid head predicts log-depth")
    h, w = (int(v) for v in protocol["source_size"])
    y0, x0, y1, x1 = (float(v) for v in protocol["target_region"])
    if (round((y1 - y0) * h), round((x1 - x0) * w)) != (480, 480):  # Probe3D's 480x480 centre region
        raise ValueError(f"target_region {protocol['target_region']} of {protocol['source_size']} is not 480x480")


def per_image(outputs, targets, protocol, metrics_module=None) -> dict[str, np.ndarray]:
    """Probe3D's per-image ``d1, d2, d3, rmse`` (``float64 [n]``). ``outputs``: predicted depth in metres,
    ``[n, S, S]``, ``[n, 1, S, S]`` or one ``[S, S]`` map broadcast to every image (the floor)."""
    import torch

    m = metrics_module if metrics_module is not None else load_probe3d()
    t = np.asarray(targets, dtype=np.float32)
    _check_protocol(protocol, t)
    gt = np.where((t > 0) & (t <= float(protocol["max_depth"])), t, np.float32(0.0))
    p = np.asarray(outputs, dtype=np.float32)
    if p.ndim == 4 and p.shape[1] == 1:
        p = p[:, 0]
    if p.ndim == 2:
        p = np.broadcast_to(p, gt.shape)
    if p.shape != gt.shape:
        raise ValueError(f"depth outputs {p.shape} do not match targets {gt.shape}")
    out: dict[str, list[np.ndarray]] = {k: [] for k in PROBE3D_KEYS}
    with torch.inference_mode():
        for lo in range(0, gt.shape[0], CHUNK):
            r = m.evaluate_depth(torch.from_numpy(np.ascontiguousarray(p[lo : lo + CHUNK])),
                                 torch.from_numpy(np.ascontiguousarray(gt[lo : lo + CHUNK])), image_average=False)
            for k in PROBE3D_KEYS:
                out[k].append(np.asarray(r[k].double().cpu().numpy(), dtype=np.float64))
    return {k: np.concatenate(v) for k, v in out.items()}


def _metric(key: str):
    def fn(outputs, targets, protocol, split_meta) -> float:
        return float(np.mean(per_image(outputs, targets, protocol)[key]))

    fn.__name__ = f"probe3d_{key}"
    return fn


def _d1_floor(outputs, targets, protocol, split_meta) -> float:
    """Probe3D ``d1`` of the per-pixel mean train log-depth map (yavih's floor map)."""
    if outputs is None:
        from yavih.metrics.depth import mean_log_depth_map

        outputs = mean_log_depth_map(split_meta["train_targets"], protocol)
    return float(np.mean(per_image(outputs, targets, protocol)["d1"]))


_KEYS = PROTOCOL_KEYS
METRICS: dict[str, MetricSpec] = {
    **{k: MetricSpec(name=k, fn=_metric(k), readouts=frozenset({"grid"}), protocol_keys=_KEYS, shape="scalar",
                     kind="metric", higher_is_better=(k != "rmse")) for k in PROBE3D_KEYS},
    "d1_floor": MetricSpec(name="d1_floor", fn=_d1_floor, readouts=frozenset({"grid"}), protocol_keys=_KEYS,
                           shape="scalar", kind="floor", higher_is_better=True, floor_of=("d1",)),
}


def score(bench, split, rows, outputs, targets) -> dict[str, float]:
    """The scorer for ``METRICS`` minus the floor: Probe3D's per-image values (the copied code), averaged."""
    per = per_image(outputs, targets, bench.spec.protocol)
    return {k: float(np.mean(v)) for k, v in per.items()}
