"""vggsounder_v2a scorer: video-to-audio generation scored by av-benchmark (the folder's setup.md).

Every number is a value ``av_bench`` (the MMAudio authors' evaluation code, MIT, a git checkout pinned in
``benchmark.yaml`` ``external``) returns; yavih computes none of them. The package is NOT a yavih dependency: it is
installed independently and reached only through the assert-import ``yavih.external.require``. Module scope imports
stdlib, numpy and yavih core only.

The scorer is folder-owned (``SCORER``, kind ``direct``): the generated audio is read from the feature set
(``FeatureReader.values("audio", ...)``), written under the clip ids, and av_bench extracts its features and compares
them with the reference audio's (the clip's VGGSound wav) and with the muted video's (ImageBind, Synchformer):

- ``fd_passt`` / ``fd_panns`` / ``fd_vgg``: Fréchet distance between generated and reference audio in PaSST, PANNs
  and VGGish embeddings (distributional quality; three embeddings, as the review asks);
- ``kl_passt`` / ``kl_panns``: mean paired KL between the reference's and the generation's class posteriors
  (semantic agreement with the clip's actual sound);
- ``is_passt`` / ``is_panns``: inception scores of the generations (reference-free; disclosed, not a target);
- ``ib_score``: ImageBind cosine between the video and the generated audio (audio-visual semantic alignment);
- ``desync``: Synchformer's absolute offset in seconds over the first and last 4.8 s (temporal synchrony);
- floors ``*_floor``: the same paired metrics on the REFERENCE audio re-paired by a seeded derangement
  (``protocol.floor_seed``): real audio of the right distribution on the wrong video. A generator must align better
  than that to be read as aligned; FD of a re-pairing is 0 by construction, so FD has no floor here;
- ``ib_score_ref`` / ``desync_ref``: the same metrics on the correctly paired reference audio (what real audio scores).

The reference features (audio and video) are extracted once per site into ``derived_root/av_bench/gt-...`` and
reused; the generations' features go to a temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yavih.metrics.registry import MetricSpec
from yavih.readouts.registry import Scorer

NAME = "vggsounder_v2a"
#: av_bench's result keys -> this folder's metric ids
AV_BENCH_KEYS: dict[str, str] = {
    "FD-PASST": "fd_passt", "FD-PANN": "fd_panns", "FD-VGG": "fd_vgg",
    "KL-PASST-softmax": "kl_passt", "KL-PANNS-softmax": "kl_panns",
    "ISC-PASST-mean": "is_passt", "ISC-PANNS-mean": "is_panns",
    "IB-Score": "ib_score", "DeSync": "desync",
}
LOWER_IS_BETTER = frozenset({"fd_passt", "fd_panns", "fd_vgg", "kad_passt", "kad_panns", "kl_passt", "kl_panns", "desync"})
PROTOCOL_KEYS: frozenset[str] = frozenset({"audio_length", "floor_seed"})
#: av_bench extraction batch (its default, 128 ten-second clips, exceeds a 24 GB GPU in PaSST); not part of the protocol
EXTRACT_BATCH = 16
#: the paired metrics recomputed per clip from av_bench's caches (intervals, samples): their means are av_bench's values
PER_CLIP_METRICS: tuple[str, ...] = ("kl_passt", "kl_panns", "ib_score", "desync")
#: cosine metrics: (reference-side feature file, generation-side feature file) in the caches
PAIRED_FEATURES: dict[str, tuple[str, str]] = {"ib_score": ("imagebind_video.pth", "imagebind_audio.pth")}
#: the checkpoints av_bench loads from ``<checkout>/weights`` (its README; it builds every encoder, Synchformer included),
#: with the sha256 of the files its README links (downloaded 2026-10-30)
WEIGHTS: dict[str, str] = {
    "music_speech_audioset_epoch_15_esc_89.98.pt": "51c68f12f9d7ea25fdaaccf741ec7f81e93ee594455410f3bca4f47f88d8e006",
    "synchformer_state_dict.pth": "8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c",
}


class _ComputedByScore:
    """``MetricSpec.fn`` placeholder: these metrics come from ``score()`` (av_bench)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{self.name} is computed by the {NAME} folder's eval.py score() with av_bench")


# ----------------------------------------------------------------------------- per-clip values, KAD, intervals

#: KAD as kadtk defines it (github.com/YoonjinXD/kadtk @ 7373161, ``calc_kernel_audio_distance``; See et al. 2025):
#: Gaussian RBF kernel, bandwidth = the median pairwise distance of the reference set, unbiased within-set terms,
#: scaled by 100. Here on av_bench's cached clip-level embeddings (PaSST, PANNs 2048-d), in float64.
KAD_SCALE = 100.0
KAD_EMBEDDINGS: dict[str, tuple[str, str | None]] = {"kad_passt": ("passt_features_embed.pth", None), "kad_panns": ("pann_features.pth", "2048")}
#: The details of the last scored outputs (per-clip values, KAD kernels), keyed by ``id(outputs)``: the interval and
#: samples steps read them back for the same outputs object the score hook saw.
_DETAILS: dict[int, tuple[Any, dict[str, Any]]] = {}


def _load_feats(cache: Path, name: str) -> dict:
    import torch

    return torch.load(cache / name, map_location="cpu", weights_only=True)


def _rows(d: Mapping[str, Any], ids: Sequence[str], key: str | None = None) -> np.ndarray:
    """``float64 [n, ...]``: the features of ``ids`` in that order."""
    return np.stack([np.asarray((d[i][key] if key else d[i]).float().numpy(), dtype=np.float64) for i in ids])


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.reshape(x.shape[0], -1)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def _kl_rows(gt_logits: np.ndarray, pred_logits: np.ndarray) -> np.ndarray:
    """Per clip KL(reference || generation) of the softmax posteriors: av_bench's ``kl_softmax`` is their mean."""
    lg, lp = _log_softmax(gt_logits), _log_softmax(pred_logits)
    return (np.exp(lg) * (lg - lp)).sum(axis=1)


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per clip cosine along the last axis (averaged over any leading per-clip axes), as av_bench's
    ``torch.cosine_similarity(..., dim=-1).mean()`` averages them."""
    num = (a * b).sum(axis=-1)
    den = np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8)
    c = num / den
    return c.reshape(c.shape[0], -1).mean(axis=1)


def _desync_rows(root: Path, gt: Path, pred: Path, ids: Sequence[str]) -> np.ndarray:
    """Per clip DeSync: the mean |offset| Synchformer predicts on the first and the last 14 segments, as av_bench's
    ``evaluate`` computes them (its DeSync is the mean over both windows of every clip)."""
    import torch
    from av_bench.synchformer.synchformer import Synchformer, make_class_grid

    v_all, a_all = _load_feats(gt, "synchformer_video.pth"), _load_feats(pred, "synchformer_audio.pth")
    model = Synchformer().to("cuda").eval()
    model.load_state_dict(torch.load(root / "weights" / "synchformer_state_dict.pth", weights_only=True))
    grid = make_class_grid(-2, 2, 21)
    out: list[float] = []
    with torch.inference_mode():
        for s in range(0, len(ids), 16):
            batch = ids[s:s + 16]
            v = torch.stack([v_all[i] for i in batch]).to("cuda")
            a = torch.stack([a_all[i] for i in batch]).to("cuda")
            first = torch.argmax(model.compare_v_a(v[:, :14], a[:, :14]), dim=-1).cpu().numpy()
            last = torch.argmax(model.compare_v_a(v[:, -14:], a[:, -14:]), dim=-1).cpu().numpy()
            out += [(abs(float(grid[f])) + abs(float(grid[la]))) / 2.0 for f, la in zip(first, last, strict=True)]
    return np.asarray(out, dtype=np.float64)


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d2, 0.0)


def kad_kernels(ref: np.ndarray, gen: np.ndarray) -> dict[str, np.ndarray]:
    """The three Gaussian kernel matrices of KAD (kadtk's kernel and bandwidth rule) for paired clip-level embeddings."""
    ref, gen = ref.reshape(ref.shape[0], -1), gen.reshape(gen.shape[0], -1)
    d_rr = _sq_dists(ref, ref)
    np.fill_diagonal(d_rr, 0.0)
    iu = np.triu_indices(ref.shape[0], k=1)
    bandwidth = float(np.median(np.sqrt(d_rr[iu])))
    gamma = 1.0 / (2.0 * bandwidth**2 + 1e-8)
    d_gg = _sq_dists(gen, gen)
    np.fill_diagonal(d_gg, 0.0)
    return {"rr": np.exp(-gamma * d_rr), "gg": np.exp(-gamma * d_gg), "rg": np.exp(-gamma * _sq_dists(ref, gen)), "bandwidth": np.asarray(bandwidth)}


def kad_from_kernels(k: Mapping[str, np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    """KAD for the full set, or for each resample given as clip counts ``weights [s, n]`` (a resampled clip enters both
    sets, the pairs stay paired). Within-set terms drop only the self-pairs, as kadtk does on a materialized resample:
    ``sum_{p != q} K[i_p, i_q] = w K w - n`` (the kernel's diagonal is 1)."""
    n = k["rr"].shape[0]
    w = np.ones((1, n)) if weights is None else np.asarray(weights, dtype=np.float64)
    def quad(m: np.ndarray) -> np.ndarray:
        return ((w @ m) * w).sum(axis=1)
    rr = (quad(k["rr"]) - n) / (n * (n - 1))
    gg = (quad(k["gg"]) - n) / (n * (n - 1))
    rg = quad(k["rg"]) / (n * n)
    return KAD_SCALE * (rr + gg - 2.0 * rg)


def _bootstrap_cfg(split_meta: Any) -> dict[str, int]:
    from yavih.metrics.intervals import DEFAULT_BOOTSTRAP

    cfg = dict(DEFAULT_BOOTSTRAP)
    if isinstance(split_meta, Mapping) and isinstance(split_meta.get("bootstrap"), Mapping):
        cfg.update(split_meta["bootstrap"])
    return {"samples": int(cfg["samples"]), "seed": int(cfg["seed"])}


def _details_of(outputs: Any) -> dict[str, Any]:
    hit = _DETAILS.get(id(outputs))
    if hit is None or hit[0] is not outputs:
        raise RuntimeError(f"{NAME}: no per-clip details for these outputs (score() runs before the intervals)")
    return hit[1]


def _per_clip_of(name: str):
    def per_clip(outputs: Any, targets: Any, protocol: Any, split_meta: Any) -> np.ndarray:
        return _details_of(outputs)["per_clip"][name]
    return per_clip


def _kad_interval_of(name: str):
    def interval(kind: str, outputs: Any, targets: Any, protocol: Any, split_meta: Any) -> tuple[float, float]:
        from yavih.metrics.intervals import bootstrap_draws

        k = _details_of(outputs)["kad_kernels"][name]
        n = k["rr"].shape[0]
        cfg = _bootstrap_cfg(split_meta)
        draws = bootstrap_draws(n, samples=cfg["samples"], seed=cfg["seed"])
        weights = np.stack([np.bincount(d, minlength=n) for d in draws]).astype(np.float64)
        stats = kad_from_kernels(k, weights)
        lo, hi = np.percentile(stats[np.isfinite(stats)], [2.5, 97.5])  # the percentile rule of every bootstrap_clip
        return float(lo), float(hi)
    return interval


def details(root: Path, gt: Path, pred: Path, ids: Sequence[str], scores: Mapping[str, Any]) -> dict[str, Any]:
    """Per-clip values of the paired metrics and KAD with its kernels, recomputed from av_bench's caches; every
    per-clip mean must equal av_bench's own number (else av_bench and this folder disagree on the definition)."""
    per: dict[str, np.ndarray] = {}
    pann_gt, pann_pred = _load_feats(gt, "pann_features.pth"), _load_feats(pred, "pann_features.pth")
    per["kl_panns"] = _kl_rows(_rows(pann_gt, ids, "logits"), _rows(pann_pred, ids, "logits"))
    per["kl_passt"] = _kl_rows(_rows(_load_feats(gt, "passt_logits.pth"), ids), _rows(_load_feats(pred, "passt_logits.pth"), ids))
    for metric, (gt_file, pred_file) in PAIRED_FEATURES.items():
        per[metric] = _cos_rows(_rows(_load_feats(gt, gt_file), ids), _rows(_load_feats(pred, pred_file), ids))
    if "desync" in PER_CLIP_METRICS:
        per["desync"] = _desync_rows(root, gt, pred, ids)
    off = {m: (float(np.mean(v)), float(scores[m])) for m, v in per.items() if m in scores and scores[m] is not None
           and abs(float(np.mean(v)) - float(scores[m])) > 1e-4 * max(1.0, abs(float(scores[m])))}
    if off:
        raise RuntimeError(f"{NAME}: per-clip means disagree with av_bench's values {off} (mean, av_bench)")
    kernels: dict[str, dict[str, np.ndarray]] = {}
    kad: dict[str, float] = {}
    for metric, (file, key) in KAD_EMBEDDINGS.items():
        g, p = _load_feats(gt, file), _load_feats(pred, file)
        kernels[metric] = kad_kernels(_rows(g, ids, key), _rows(p, ids, key))
        kad[metric] = float(kad_from_kernels(kernels[metric])[0])
    return {"per_clip": per, "kad": kad, "kad_kernels": kernels}


def _spec(name: str, description: str, *, kind: str = "metric", floor_of: tuple[str, ...] = ()) -> MetricSpec:
    hib: bool | None = None if kind == "disclosure" else (name.removesuffix("_floor") not in LOWER_IS_BETTER)
    extra: dict[str, Any] = {}
    if name in PER_CLIP_METRICS:  # a clip-bootstrap interval over the per-clip values av_bench averages
        extra = {"per_clip": _per_clip_of(name), "ci": frozenset({"bootstrap_clip"})}
    elif name in KAD_EMBEDDINGS:  # a clip-bootstrap interval recomputing KAD on each resample
        extra = {"interval": _kad_interval_of(name), "ci": frozenset({"bootstrap_clip"})}
    return MetricSpec(name=name, fn=_ComputedByScore(name), readouts=frozenset({NAME}), kind=kind, higher_is_better=hib,
                      floor_of=floor_of, external="av_bench", description=description, **extra)


METRICS: dict[str, MetricSpec] = {
    "fd_passt": _spec("fd_passt", "av_bench FD-PASST: Fréchet distance, PaSST embeddings, generated vs reference audio"),
    "fd_panns": _spec("fd_panns", "av_bench FD-PANN: Fréchet distance, PANNs 2048-d embeddings"),
    "fd_vgg": _spec("fd_vgg", "av_bench FD-VGG: Fréchet distance, VGGish embeddings"),
    "kad_passt": _spec("kad_passt", "KAD (kadtk's estimator: Gaussian kernel, median-distance bandwidth, x100) on av_bench's PaSST clip embeddings"),
    "kad_panns": _spec("kad_panns", "KAD (kadtk's estimator: Gaussian kernel, median-distance bandwidth, x100) on av_bench's PANNs 2048-d clip embeddings"),
    "kl_passt": _spec("kl_passt", "av_bench KL-PASST-softmax: mean paired KL of the PaSST class posteriors (reference || generation)"),
    "kl_panns": _spec("kl_panns", "av_bench KL-PANNS-softmax: mean paired KL of the PANNs class posteriors"),
    "is_passt": _spec("is_passt", "av_bench ISC-PASST-mean: inception score of the generations (PaSST logits, 10 splits, seed 2020)"),
    "is_panns": _spec("is_panns", "av_bench ISC-PANNS-mean: inception score of the generations (PANNs logits)"),
    "ib_score": _spec("ib_score", "av_bench IB-Score: ImageBind cosine between the muted video and the generated audio"),
    "desync": _spec("desync", "av_bench DeSync: mean |offset| in seconds Synchformer predicts, first and last 4.8 s"),
    "ib_score_ref": _spec("ib_score_ref", "av_bench IB-Score of the reference audio (correctly paired)", kind="disclosure"),
    "desync_ref": _spec("desync_ref", "av_bench DeSync of the reference audio (correctly paired)", kind="disclosure"),
    "ib_score_floor": _spec("ib_score_floor", "av_bench IB-Score of the reference audio re-paired by a seeded derangement",
                            kind="floor", floor_of=("ib_score",)),
    "desync_floor": _spec("desync_floor", "av_bench DeSync of the reference audio re-paired by a seeded derangement",
                          kind="floor", floor_of=("desync",)),
    "kl_passt_floor": _spec("kl_passt_floor", "av_bench KL-PASST-softmax of the reference audio re-paired by a seeded derangement",
                            kind="floor", floor_of=("kl_passt",)),
}


# ----------------------------------------------------------------------------- the assert-import


def _external(bench: Any) -> dict[str, str]:
    for e in getattr(bench.spec, "external", ()):
        if e.name == "av_bench":
            return {"name": e.name, "pip": e.pip, "version_pin": e.version_pin}
    raise KeyError("benchmark.yaml external has no 'av_bench' entry")


def require_av_bench(bench: Any) -> Path:
    """Assert-import av_bench at its pinned version; return the checkout root (its ``weights/`` live there)."""
    from yavih.external import record_version, require

    e = _external(bench)
    mod = require(e["name"], e["pip"], e["version_pin"])
    root = Path(mod.__file__).resolve().parent.parent
    missing = _check_weights(root)
    if missing:
        from yavih.external import ExternalMissing

        raise ExternalMissing("av_bench", f"E_EXTERNAL_MISSING: av_bench weights {missing} in {root / 'weights'} (setup.md \"## Dependencies\")")
    head = root / ".git" / "HEAD"
    if head.is_file():
        record_version("av_bench", f"{e['version_pin']}+{_git_head(root)[:8]}")
    return root


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

def _check_weights(root: Path) -> list[str]:
    """The weight files that are missing or whose sha256 differs from ``WEIGHTS``. A verified file is stamped in
    ``weights/.yavih_sha256.json`` by size and mtime, so later runs do not rehash gigabytes."""
    stamp_path = root / "weights" / ".yavih_sha256.json"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamp = {}
    bad, changed = [], False
    for name, want in WEIGHTS.items():
        p = root / "weights" / name
        if not p.is_file():
            bad.append(f"{name} (missing)")
            continue
        st = p.stat()
        key = [st.st_size, st.st_mtime_ns]
        rec = stamp.get(name)
        if rec and rec[:2] == key and rec[2] == want:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != want:
            bad.append(f"{name} (sha256 {h.hexdigest()[:12]} != {want[:12]})")
            continue
        stamp[name], changed = [*key, want], True
    if changed:
        try:
            stamp_path.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
        except OSError:
            pass
    return bad


def _free_gpu() -> None:
    """Release what av_bench's in-process extraction left on the GPU (its encoders), before a script subprocess or the
    next extraction needs the memory."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _in_dir:
    """Run av_bench's extraction from its checkout, as the commands in its README do (a relative path then resolves there)."""

    def __init__(self, root: Path) -> None:
        self.root, self.prev = root, None

    def __enter__(self) -> None:
        self.prev = os.getcwd()
        os.chdir(self.root)

    def __exit__(self, *exc: Any) -> None:
        os.chdir(self.prev)


# ----------------------------------------------------------------------------- targets, floors


def targets_of(bench: Any, split: str, rows: Sequence[int]) -> list[dict[str, str]]:
    """``{clip_id, rel_audio, rel_media}`` per row (manifest order)."""
    m = bench.manifest(split)
    out = []
    for r in np.asarray(rows, dtype=np.int64).ravel():
        extra = m.extra[int(r)]
        out.append({"clip_id": str(m.clip_id[int(r)]), "rel_audio": str(extra["rel_audio"]), "rel_media": str(m.media[int(r)]["video"])})
    return out


def derangement(n: int, seed: int) -> list[int]:
    """A seeded permutation with no fixed point (``yavih.conformance`` stream, never numpy's RNG methods)."""
    if n < 2:
        return list(range(n))
    from yavih.conformance import stream

    keys = stream(f"{NAME}:floor:{seed}:{n}", n)
    perm = [int(i) for i in np.argsort(keys, kind="stable")]
    # rotate each cycle element that maps to itself onto its successor: a derangement, deterministic
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def floor_outputs(targets: Sequence[Mapping[str, str]], n_eval: int, *, protocol: Mapping[str, Any], **_kw: Any) -> list[dict[str, str]]:
    """The floor's "outputs": each row gets ANOTHER row's reference audio (a seeded derangement)."""
    perm = derangement(len(targets), int(protocol.get("floor_seed", 0)))
    return [{"reference_of": targets[perm[i]]["clip_id"], "rel_audio": targets[perm[i]]["rel_audio"]} for i in range(len(targets))]


# ----------------------------------------------------------------------------- av_bench calls


def _roots(bench: Any) -> tuple[Path, Path]:
    roots = bench.resolved_roots()
    return Path(roots["media_root"]), Path(roots["derived_root"])


def _link_dir(d: Path, items: Mapping[str, Path]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, src in items.items():
        link = d / name
        if not link.exists():
            if not src.is_file():
                raise FileNotFoundError(f"{src}: missing (setup.md \"## Prepare & verify\")")
            os.symlink(src, link)


def check_names(ids: Sequence[str]) -> None:
    """Every clip id must pass av_bench's pairing unchanged: ``av_bench.utils.clean_sample_name`` rewrites names of
    some lengths, and an unpaired row is only printed, never raised."""
    from av_bench.utils import clean_sample_name

    bad = [i for i in ids if clean_sample_name(i) != i]
    if bad:
        raise ValueError(f"{NAME}: {len(bad)} clip ids are rewritten by av_bench's clean_sample_name (first {bad[:3]}); they would be dropped unpaired")


def check_cache(cache: Path, ids: Sequence[str]) -> None:
    """Every feature file av_bench wrote in ``cache`` holds exactly ``ids``. Its loaders skip an unreadable or too-short
    file by scoring the NEXT file under that file's name, which collapses to a missing key here."""
    import torch

    want = set(ids)
    for f in sorted(cache.glob("*.pth")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(d, dict) and set(map(str, d)) != want:
            got = set(map(str, d))
            raise RuntimeError(f"{NAME}: {f.name} holds {len(got)} clips for {len(want)} scored; missing {sorted(want - got)[:5]}, "
                               f"extra {sorted(got - want)[:5]} (av_bench skipped a file: check its log)")


def gt_cache(bench: Any, targets: Sequence[Mapping[str, str]], *, audio_length: float, root: Path) -> Path:
    """The reference cache of these rows: av_bench audio features of the reference wavs and video features of the muted
    videos, extracted once into ``derived_root/av_bench/gt-<n>-<ids sha12>-<audio_length>`` (keyed by the row set)."""
    media_root, derived_root = _roots(bench)
    ids = [t["clip_id"] for t in targets]
    check_names(ids)
    key = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:12]
    cache = derived_root / "av_bench" / f"gt-{len(ids)}-{key}-{audio_length:g}"
    done = cache / "DONE"
    if done.is_file():
        return cache
    audio_dir, video_dir = cache / "_audio", cache / "_video"
    _link_dir(audio_dir, {f"{t['clip_id']}.wav": media_root / t["rel_audio"] for t in targets})
    _link_dir(video_dir, {f"{t['clip_id']}.mp4": derived_root / t["rel_media"] for t in targets})
    from av_bench.extract import extract

    audio_files = ("pann_features.pth", "vggish_features.pth", "passt_features_embed.pth", "passt_logits.pth", "imagebind_audio.pth",
                   "synchformer_audio.pth")
    if not all((cache / f).is_file() for f in audio_files):  # resume: an interrupted build keeps its audio features
        with _in_dir(root):
            extract(audio_path=audio_dir, output_path=cache, audio_length=audio_length, device="cuda", skip_clap=True, batch_size=EXTRACT_BATCH, num_workers=8)
        _free_gpu()
    subprocess.run([sys.executable, str(root / "extract_video.py"), "--video_path", str(video_dir), "--gt_cache", str(cache),
                    "--audio_length", str(audio_length), "--gt_batch_size", "2", "--num_workers", "8"], check=True, cwd=str(root))
    check_cache(cache, ids)
    done.write_text(json.dumps({"n": len(ids), "audio_length": audio_length}), encoding="utf-8")
    return cache


def _evaluate(gt: Path, pred: Path) -> dict[str, float]:
    from av_bench.evaluate import evaluate

    raw = evaluate(gt, pred, is_paired=True, skip_clap=True)
    return {AV_BENCH_KEYS[k]: float(v) for k, v in raw.items() if k in AV_BENCH_KEYS}


def reference_scores(bench: Any, targets: Sequence[Mapping[str, str]], *, audio_length: float, root: Path) -> dict[str, float]:
    """av_bench on the correctly paired reference audio (cached next to the reference features)."""
    cache = gt_cache(bench, targets, audio_length=audio_length, root=root)
    path = cache / "reference_scores.json"
    if not path.is_file():
        path.write_text(json.dumps(_evaluate(cache, cache)), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def score(bench: Any, split: str, rows: Any, outputs: Any, targets: Any = None) -> dict[str, Any]:
    """av_bench on ``outputs`` (one generated wav's bytes per row, or the floor's re-paired reference audio) against the
    rows' reference audio and muted video. Returns every ``METRICS`` id this call computes."""
    root = require_av_bench(bench)
    protocol = bench.spec.protocol
    audio_length = float(protocol["audio_length"])
    tg = list(targets) if targets is not None else targets_of(bench, split, rows)
    outs = list(outputs)
    if len(outs) != len(tg):
        raise ValueError(f"{NAME} score: {len(outs)} outputs for {len(tg)} rows")
    gt = gt_cache(bench, tg, audio_length=audio_length, root=root)
    media_root, _ = _roots(bench)
    with tempfile.TemporaryDirectory(prefix=f"yavih-{NAME}-") as tmp:
        pred_audio, pred_cache = Path(tmp) / "audio", Path(tmp) / "cache"
        pred_audio.mkdir()
        for t, o in zip(tg, outs, strict=True):
            dst = pred_audio / f"{t['clip_id']}.wav"
            if isinstance(o, (bytes, bytearray, memoryview)):
                dst.write_bytes(bytes(o))
            else:  # the floor: another row's reference audio under this row's name
                os.symlink(media_root / o["rel_audio"], dst)
        from av_bench.extract import extract

        with _in_dir(root):
            extract(audio_path=pred_audio, output_path=pred_cache, audio_length=audio_length, device="cuda", skip_clap=True, batch_size=EXTRACT_BATCH, num_workers=8)
        _free_gpu()
        check_cache(pred_cache, [t["clip_id"] for t in tg])
        scores = _evaluate(gt, pred_cache)
        det = details(root, gt, pred_cache, [t["clip_id"] for t in tg], scores)
    scores.update(det["kad"])
    if len(_DETAILS) > 8:
        _DETAILS.clear()
    _DETAILS[id(outputs)] = (outputs, det)
    ref = reference_scores(bench, tg, audio_length=audio_length, root=root)
    scores["ib_score_ref"] = ref.get("ib_score")
    scores["desync_ref"] = ref.get("desync")
    return scores


# ----------------------------------------------------------------------------- the scorer


def _run(ctx: Any) -> dict[str, Any]:
    """Score the eval rows: read the generated audio, then every metric and floor through ``evaluate_metrics`` with
    this folder's ``score`` hook (the same path the conformance fixture runs)."""
    from yavih.readouts.base import evaluate_metrics

    rows_all, outputs, targets = [], [], []
    splits = list(ctx.roles["eval"])
    for split in splits:
        reader = ctx.reader(split)
        rows = np.asarray(reader.present_rows, dtype=np.int64)
        outputs += reader.values("audio", rows, codecs=ctx.bench.codecs)
        targets += targets_of(ctx.bench, split, rows)
        rows_all.append(rows)
    rows = np.concatenate(rows_all) if rows_all else np.zeros(0, np.int64)
    blocks = evaluate_metrics(ctx.metric_specs, outputs=outputs, targets=targets, protocol=ctx.protocol,
                              split_meta={"n_eval": len(targets), "bootstrap": dict(ctx.bootstrap or {})},
                              floor_outputs=floor_outputs(targets, len(targets), protocol=ctx.protocol), ci=ctx.ci,
                              scorer=ctx.scorer(splits, rows, targets))
    if getattr(ctx, "log_samples", False):  # per-clip values: paired differences between systems on the same clips
        blocks["samples"] = [{"role": "eval", "rows": rows, "clip_ids": [t["clip_id"] for t in targets],
                              "values": dict(_details_of(outputs)["per_clip"])}]
    return {**blocks, "probe": {}, "fit": {"scorer": NAME}, "timing": {}, "eval_rows": rows}


#: conformance population size: small enough for a CI GPU, large enough for a covariance on 3 embeddings
POPULATION = 32


def _population(ctx: Any, tag: str) -> dict[str, Any]:
    """The conformance population: ``POPULATION`` real eval rows (``conformance.select_rows``) with fabricated
    generations (seeded noise through the ``wav`` codec's synthetic sample); needs the staged reference audio and
    muted videos (else the fixture is skipped here)."""
    from yavih import codecs as C
    from yavih.conformance import W_SKIPPED, ConformanceError, select_rows

    bench = ctx.bench
    split = list(ctx.roles["eval"])[0]
    rows = select_rows(bench.manifest_hash, split, bench.n_rows(split), POPULATION, salt="eval")
    targets = targets_of(bench, split, rows)
    try:
        media_root, derived_root = _roots(bench)
    except Exception as e:  # noqa: BLE001 - no roots here: the fixture cannot run
        raise ConformanceError(W_SKIPPED, f"{NAME}: data roots do not resolve here ({e})") from e
    lacking = [t["clip_id"] for t in targets if not (media_root / t["rel_audio"]).is_file() or not (derived_root / t["rel_media"]).is_file()]
    if lacking:
        raise ConformanceError(W_SKIPPED, f"{NAME}: the reference audio / muted video of {len(lacking)} fixture rows is not staged here")
    spec = dict(bench.spec.output_contract.schema["audio"])
    rng = np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))
    outputs = [C.WAV.synthetic(spec, {}, rng) for _ in targets]
    return {"outputs": outputs, "targets": targets, "split_meta": {"n_eval": len(targets)},
            "floor_outputs": floor_outputs(targets, len(targets), protocol=ctx.protocol), "eval_splits": [split], "eval_rows": rows}


SCORER = Scorer(name=NAME, version=1, module=__name__, kind="direct", label_kinds=frozenset({"none"}), requires_t_bins=False,
                requires_grid=False, score=_run, floor_outputs=floor_outputs, conformance_population=_population, needs_gpu=True)
