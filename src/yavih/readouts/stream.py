"""``stream`` readout: causal heads over a streaming benchmark's update sequences (``output_contract.shape: sequence``).

Each row is a recording; its features are the bridge's update sequence (``emb [n, D]``, ``cutoff [n]`` and, for a
real-time replay, the ``timing`` log; :mod:`yavih.stream`). The folder's ``eval.py`` ``targets(bench, split, rows)``
hook returns, per row, the scored times and their labels::

    {"times": float [m], "labels": {group: int [m] (ce; -1 = not scored) | float [m, C] (bce; NaN = not scored)},
     "meta": {name: [m]}}   # optional per-target values the metrics read (e.g. EPIC's unseen / tail flags)

``readout_params.groups`` declares each label group ``{name: {loss: ce | bce, classes: C}}``.

**Alignment** (the held prediction): a target at time ``t`` reads the latest update AVAILABLE by ``t``: its cutoff in
the causal mode, its ``ready_at`` plus the head's measured per-update latency in the real-time mode. Before the first
available update there is no evidence and the prediction is the train prior; that interval is scored like any other.

**Heads** (both trained per seed on the train split, selected on a held-out ``val_fraction`` of train RECORDINGS by
validation loss over the pinned lr grid with patience):

* ``linear`` — the held embedding (standardized with train statistics) through one linear layer per group.
* ``gru`` — ``causal_gru``: ``[emb, elapsed / update_interval]`` projected to 512, one unidirectional GRU layer of 512,
  a linear layer per group; one GRU step per completed update, in availability order, the state carried through the
  recording (truncated back-propagation through chunks of ``TBPTT`` updates, state detached between chunks).

Outputs are probabilities (softmax for ``ce`` groups, sigmoid for ``bce``). Metrics are the folder's, named
``<metric>_linear`` / ``<metric>_gru``: each sees ``{group: scores [M, C]}``, ``{group: labels}`` and ``split_meta``
with the per-target ``recording`` (for intervals resampled by recording), ``time``, ``meta`` and ``groups``, and under
``val`` the same for the held-out train recordings (``outputs``, ``targets``, ``recording``, ``time``, ``meta``): a
metric that needs an operating point (a detection threshold) chooses it there, never on eval. Floors see the prior on
every target. torch is imported on first use.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from yavih.readouts.base import ProbeRefused, ReadoutContext, evaluate_metrics
from yavih.readouts.registry import Scorer

__all__ = ["SCORER", "VERSION", "HEADS", "TBPTT", "run_stream", "held_index"]

#: Enters ``protocol_hash`` (``READOUT_VERSION``): a change to the heads, alignment, losses or selection bumps it.
VERSION: int = 1
HEADS: tuple[str, ...] = ("linear", "gru")
PROJ, HIDDEN = 512, 512
#: updates per truncated back-propagation chunk of the GRU head
TBPTT: int = 256
#: targets per optimizer step of the linear head
LINEAR_BATCH: int = 256
#: the synthetic end-to-end band (``tests/test_stream_e2e.py``)
CONFORMANCE: dict[str, float] = {"oracle_min": 0.8, "random_max": 0.6}
_STREAM_TRAIN = dict(lr=1e-3, lr_grid=(1e-3, 1e-4), weight_decay=1e-4, max_epochs=40, batch_size=16, patience=5, warmup_fraction=0.0)


def held_index(avail: np.ndarray, times: np.ndarray) -> np.ndarray:
    """For each target time, the index of the latest update available by it (``-1``: none yet)."""
    return np.searchsorted(np.asarray(avail, dtype=np.float64), np.asarray(times, dtype=np.float64) + 1e-9, side="right") - 1


@dataclass
class _Rec:
    rid: int
    clip_id: str
    emb: np.ndarray            # float32 [n, D]
    avail: np.ndarray          # float64 [n]: availability time of each update
    times: np.ndarray          # float64 [m]
    labels: dict[str, np.ndarray]
    meta: dict[str, np.ndarray] = field(default_factory=dict)
    gap: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # elapsed since the previous update / interval

    @property
    def held(self) -> np.ndarray:
        return held_index(self.avail, self.times)


def _groups(ctx: ReadoutContext) -> dict[str, dict[str, Any]]:
    g = dict((ctx.spec.readout_params or {}).get("groups") or {})
    if not g:
        raise ProbeRefused("E_SPEC", "readout_params.groups must declare the label groups {name: {loss: ce|bce, classes: C}}")
    return {str(k): {"loss": str(v["loss"]), "classes": int(v["classes"])} for k, v in g.items()}


def _mode(ctx: ReadoutContext) -> str:
    return str((dict(ctx.describe or {}).get("stream") or {}).get("mode") or "causal")


def _interval(ctx: ReadoutContext) -> float:
    return float(getattr(getattr(ctx.spec.output_contract, "stream", None), "update_interval_s", None) or 1.0)


def _load(ctx: ReadoutContext, splits: list[str], groups: dict[str, dict[str, Any]], latency: float, rid0: int = 0) -> list[_Rec]:
    mode = _mode(ctx)
    dt = _interval(ctx)
    out: list[_Rec] = []
    for s in splits:
        rows = ctx.present_rows(s)
        if len(rows) == 0:
            continue
        reader = ctx.reader(s)
        codecs = getattr(ctx.bench, "codecs", None)
        embs = reader.values("emb", rows, codecs=codecs)
        cuts = reader.values("cutoff", rows, codecs=codecs)
        timings = reader.values("timing", rows, codecs=codecs) if mode == "realtime" else [None] * len(rows)
        tg = ctx.hook_targets(s, rows)
        if tg is None or len(tg) != len(rows):
            raise ProbeRefused("E_TARGETS", "a streaming benchmark's eval.py defines targets(bench, split, rows) -> one "
                                             "{times, labels, meta?} per row")
        ids = ctx.clip_ids(s, rows)
        for k in range(len(rows)):
            cut = np.asarray(cuts[k], dtype=np.float64)
            if timings[k] is not None:
                avail = np.asarray(timings[k]["updates"], dtype=np.float64).reshape(-1, 3)[:, 2] + latency
            else:
                avail = cut
            t = tg[k]
            labels = {}
            for name, spec in groups.items():
                lab = np.asarray(t["labels"][name])
                labels[name] = lab.astype(np.int64) if spec["loss"] == "ce" else lab.astype(np.float32)
            out.append(_Rec(rid=rid0 + len(out), clip_id=str(ids[k]), emb=np.asarray(embs[k], dtype=np.float32), avail=avail,
                            times=np.asarray(t["times"], dtype=np.float64), labels=labels,
                            meta={str(n): np.asarray(v) for n, v in dict(t.get("meta") or {}).items()},
                            gap=(np.diff(np.r_[0.0, avail]) / dt).astype(np.float32)))
    return out


def _prior(train: list[_Rec], groups: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """The no-evidence prediction per group: train class frequencies (ce, softmax-normalised) or prevalences (bce)."""
    out = {}
    for name, spec in groups.items():
        c = spec["classes"]
        if spec["loss"] == "ce":
            y = np.concatenate([r.labels[name] for r in train]) if train else np.zeros(0, np.int64)
            y = y[y >= 0]
            cnt = np.bincount(y, minlength=c).astype(np.float64) + 1e-3
            out[name] = (cnt / cnt.sum()).astype(np.float32)
        else:
            y = np.concatenate([r.labels[name] for r in train], axis=0) if train else np.zeros((0, c), np.float32)
            p = np.nanmean(np.where(np.isnan(y), np.nan, y), axis=0) if y.size else np.full(c, 0.5)
            out[name] = np.clip(np.nan_to_num(p, nan=0.0), 1e-4, 1 - 1e-4).astype(np.float32)
    return out


def _split_val(recs: list[_Rec], frac: float, seed: int) -> tuple[list[_Rec], list[_Rec]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7319]))
    perm = rng.permutation(len(recs))
    n_val = int(round(frac * len(recs))) if len(recs) > 1 else 0
    n_val = min(max(n_val, 1 if len(recs) >= 4 else 0), len(recs) - 1)
    val = {int(i) for i in perm[:n_val]}
    return [r for i, r in enumerate(recs) if i not in val], [r for i, r in enumerate(recs) if i in val]


# ----------------------------------------------------------------------------- torch heads


def _torch():
    import torch

    return torch


def _build(kind: str, d: int, groups: dict[str, dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> Any:
    torch = _torch()
    nn = torch.nn
    out_dim = sum(g["classes"] for g in groups.values())

    class Linear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(mean))
            self.register_buffer("std", torch.as_tensor(std))
            self.fc = nn.Linear(d, out_dim)

        def forward(self, x):  # x [..., D]
            return self.fc((x - self.mean) / self.std)

    class GRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(mean))
            self.register_buffer("std", torch.as_tensor(std))
            self.proj = nn.Linear(d + 1, PROJ)
            self.gru = nn.GRU(PROJ, HIDDEN, num_layers=1, batch_first=True)
            self.fc = nn.Linear(HIDDEN, out_dim)

        def forward(self, x, gap, h=None):  # x [B, L, D], gap [B, L]
            z = torch.relu(self.proj(torch.cat([(x - self.mean) / self.std, gap.unsqueeze(-1)], dim=-1)))
            y, h = self.gru(z, h)
            return self.fc(y), h

    return Linear() if kind == "linear" else GRU()


def _loss(logits: Any, labels: dict[str, Any], groups: dict[str, dict[str, Any]]) -> Any:
    torch = _torch()
    total, at = 0.0, 0
    for name, spec in groups.items():
        c = spec["classes"]
        z = logits[..., at:at + c]
        at += c
        y = labels[name]
        if spec["loss"] == "ce":
            ok = y >= 0
            if ok.any():
                total = total + torch.nn.functional.cross_entropy(z[ok], y[ok])
        else:
            ok = ~torch.isnan(y)
            if ok.any():
                total = total + torch.nn.functional.binary_cross_entropy_with_logits(z[ok], torch.nan_to_num(y)[ok])
    return total


def _probs(logits: np.ndarray, groups: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    out, at = {}, 0
    for name, spec in groups.items():
        z = logits[:, at:at + spec["classes"]].astype(np.float64)
        at += spec["classes"]
        if spec["loss"] == "ce":
            z = np.exp(z - z.max(axis=1, keepdims=True))
            out[name] = (z / z.sum(axis=1, keepdims=True)).astype(np.float32)
        else:
            out[name] = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    return out


def _targets_at(recs: list[_Rec], groups: dict[str, dict[str, Any]], held: bool) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Flattened targets: (record index, held update index, {group: labels}); ``held`` keeps only targets with evidence."""
    ri, hi, lab = [], [], {n: [] for n in groups}
    for k, r in enumerate(recs):
        h = r.held
        m = h >= 0 if held else np.ones(h.shape, bool)
        ri.append(np.full(int(m.sum()), k))
        hi.append(h[m])
        for n in groups:
            lab[n].append(r.labels[n][m])
    cat = (lambda xs: np.concatenate(xs, axis=0) if xs else np.zeros(0))  # noqa: E731
    return cat(ri).astype(np.int64), cat(hi).astype(np.int64), {n: cat(v) for n, v in lab.items()}


class _Fit:
    """One head's trained module plus its training record."""

    def __init__(self, module: Any, record: dict[str, Any]) -> None:
        self.module, self.record = module, record


def _fit(kind: str, train: list[_Rec], val: list[_Rec], groups: dict[str, dict[str, Any]], cfg: Any, seed: int, device: str,
         ctx: ReadoutContext) -> _Fit:
    torch = _torch()
    allx = np.concatenate([r.emb for r in train], axis=0)
    mean, std = allx.mean(axis=0).astype(np.float32), (allx.std(axis=0) + 1e-6).astype(np.float32)
    lrs = list(cfg.lr_grid) or [float(cfg.lr)]
    best: tuple[float, Any, dict[str, Any]] | None = None
    for lr in lrs:
        torch.manual_seed(seed)
        np_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        model = _build(kind, allx.shape[1], groups, mean, std).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(cfg.weight_decay))
        best_arm, best_state, bad, epochs, best_epoch = math.inf, None, 0, 0, 0
        for epoch in range(int(cfg.max_epochs)):
            ctx.check_interrupt()
            model.train()
            _epoch(kind, model, opt, train, groups, int(cfg.batch_size), device, np_rng)
            v = _val_loss(kind, model, val or train, groups, device)
            epochs = epoch + 1
            if v < best_arm - 1e-6:
                best_arm, best_state, bad, best_epoch = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0, epochs
            else:
                bad += 1
                if bad >= int(cfg.patience):
                    break
        model.load_state_dict(best_state)
        rec = {"selected_lr": float(lr), "epochs_run": epochs, "best_epoch": best_epoch, "val_loss": best_arm,
               "stop": "patience" if bad >= int(cfg.patience) else "max_epochs"}
        if best is None or best_arm < best[0]:
            best = (best_arm, model, rec)
    assert best is not None
    model = best[1]
    model.eval()
    return _Fit(model, best[2])


def _epoch(kind: str, model: Any, opt: Any, recs: list[_Rec], groups: dict[str, dict[str, Any]], bs: int, device: str, rng: Any) -> None:
    torch = _torch()
    if kind == "linear":
        ri, hi, lab = _targets_at(recs, groups, held=True)
        if ri.size == 0:
            return
        x = np.stack([recs[r].emb[h] for r, h in zip(ri, hi, strict=True)])
        order = rng.permutation(len(ri))
        step = int(min(LINEAR_BATCH, max(16, len(ri) // 8)))  # at least ~8 steps per epoch on a small train set
        for s in range(0, len(order), step):
            b = order[s:s + step]
            logits = model(torch.as_tensor(x[b], device=device))
            loss = _loss(logits, {n: torch.as_tensor(v[b], device=device) for n, v in lab.items()}, groups)
            if torch.is_tensor(loss):
                opt.zero_grad()
                loss.backward()
                opt.step()
        return
    order = rng.permutation(len(recs))
    for s in range(0, len(order), bs):
        batch = [recs[int(i)] for i in order[s:s + bs]]
        x, gap, n = _pad(batch)
        h = None
        for c0 in range(0, x.shape[1], TBPTT):
            c1 = min(c0 + TBPTT, x.shape[1])
            out, h = model(torch.as_tensor(x[:, c0:c1], device=device), torch.as_tensor(gap[:, c0:c1], device=device), h)
            h = h.detach()
            sel_b, sel_t, lab = [], [], {g: [] for g in groups}
            for b, r in enumerate(batch):
                hh = r.held
                m = (hh >= c0) & (hh < c1)
                sel_b.append(np.full(int(m.sum()), b))
                sel_t.append(hh[m] - c0)
                for g in groups:
                    lab[g].append(r.labels[g][m])
            bi, ti = np.concatenate(sel_b).astype(np.int64), np.concatenate(sel_t).astype(np.int64)
            if bi.size == 0:
                continue
            logits = out[torch.as_tensor(bi, device=device), torch.as_tensor(ti, device=device)]
            loss = _loss(logits, {g: torch.as_tensor(np.concatenate(v, axis=0), device=device) for g, v in lab.items()}, groups)
            if torch.is_tensor(loss):
                opt.zero_grad()
                loss.backward()
                opt.step()


def _pad(batch: list[_Rec]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray([r.emb.shape[0] for r in batch])
    d = batch[0].emb.shape[1]
    length = max(1, int(n.max()))
    x = np.zeros((len(batch), length, d), np.float32)
    gap = np.zeros((len(batch), length), np.float32)
    for b, r in enumerate(batch):
        x[b, : n[b]] = r.emb
        gap[b, : n[b]] = r.gap
    return x, gap, n


def _outputs(kind: str, model: Any, recs: list[_Rec], groups: dict[str, dict[str, Any]], device: str) -> list[np.ndarray]:
    """Per recording, the head's logits at each of its updates ``[n, sum C]`` (no grad)."""
    torch = _torch()
    out = []
    with torch.no_grad():
        for r in recs:
            if r.emb.shape[0] == 0:
                out.append(np.zeros((0, sum(g["classes"] for g in groups.values())), np.float32))
                continue
            if kind == "linear":
                out.append(model(torch.as_tensor(np.array(r.emb), device=device)).float().cpu().numpy())
            else:
                x, gap, _ = _pad([r])
                y, _ = model(torch.as_tensor(x, device=device), torch.as_tensor(gap, device=device))
                out.append(y[0].float().cpu().numpy())
    return out


def _val_loss(kind: str, model: Any, recs: list[_Rec], groups: dict[str, dict[str, Any]], device: str) -> float:
    torch = _torch()
    model.eval()
    per = _outputs(kind, model, recs, groups, device)
    ri, hi, lab = _targets_at(recs, groups, held=True)
    if ri.size == 0:
        return 0.0
    logits = torch.as_tensor(np.stack([per[r][h] for r, h in zip(ri, hi, strict=True)]))
    loss = _loss(logits, {n: torch.as_tensor(v) for n, v in lab.items()}, groups)
    return float(loss) if torch.is_tensor(loss) else 0.0


def _scores(kind: str, fit: _Fit, recs: list[_Rec], groups: dict[str, dict[str, Any]], prior: dict[str, np.ndarray],
            device: str) -> dict[str, np.ndarray]:
    """``{group: probabilities [M, C]}`` at every target of ``recs`` (held prediction; the prior before evidence)."""
    per = _outputs(kind, fit.module, recs, groups, device)
    ri, hi, _ = _targets_at(recs, groups, held=False)
    total = sum(g["classes"] for g in groups.values())
    logits = np.zeros((ri.size, total), np.float32)
    ok = hi >= 0
    if ok.any():
        logits[ok] = np.stack([per[r][h] for r, h in zip(ri[ok], hi[ok], strict=True)])
    probs = _probs(logits, groups)
    for name in groups:
        probs[name][~ok] = prior[name]
    return probs


def _head_latency(fits: dict[str, _Fit], d: int, device: str) -> dict[str, float]:
    """Median seconds of one head update (batch 1, one step) on ``device``: added to ``ready_at`` in real time."""
    torch = _torch()
    out = {}
    for kind, fit in fits.items():
        x = torch.zeros((1, 1, d), device=device)
        gap = torch.zeros((1, 1), device=device)
        ts = []
        with torch.no_grad():
            for _ in range(30):
                t0 = time.perf_counter()
                _ = fit.module(x[0]) if kind == "linear" else fit.module(x, gap)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        out[kind] = float(np.median(ts[5:]))
    return out


def _flatten(recs: list[_Rec], groups: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _, _, lab = _targets_at(recs, groups, held=False)
    rec = np.concatenate([np.full(r.times.size, r.rid) for r in recs]) if recs else np.zeros(0, np.int64)
    times = np.concatenate([r.times for r in recs]) if recs else np.zeros(0)
    names = sorted({k for r in recs for k in r.meta})
    meta = {k: np.concatenate([r.meta[k] for r in recs]) for k in names}
    held = np.concatenate([r.held for r in recs]) if recs else np.zeros(0, np.int64)
    return lab, {"recording": rec, "time": times, "meta": meta, "has_evidence": held >= 0,
                 "recording_ids": [r.clip_id for r in recs]}


def run_stream(ctx: ReadoutContext) -> dict:
    """Train both heads on the train split (one seed), hold their predictions on the eval targets and score them."""
    groups = _groups(ctx)
    cfg = ctx.cfg
    device = str(ctx.device or "cpu")
    t0 = time.time()
    mode = _mode(ctx)
    train_all = _load(ctx, ctx.roles["train"], groups, 0.0)
    if not train_all:
        raise ProbeRefused("E_SPLIT_MISSING", f"no present train recordings in {ctx.roles['train']}")
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)
    tr, va = _split_val(train_all, float(cfg.val_fraction), int(ctx.seed))
    t1 = time.time()
    fits = {kind: _fit(kind, tr, va, groups, cfg, int(ctx.seed), device, ctx) for kind in HEADS}
    ctx.timing["fit_s"] = ctx.timing.get("fit_s", 0.0) + (time.time() - t1)
    latency = _head_latency(fits, train_all[0].emb.shape[1], device) if mode == "realtime" else {k: 0.0 for k in HEADS}
    prior = _prior(train_all, groups)
    t2 = time.time()
    blocks: dict[str, Any] = {"metrics": {}, "floors": {}, "controls": {}, "ci95": {}}
    eval_rec: list[_Rec] = []
    labels: dict[str, np.ndarray] = {}
    smeta: dict[str, Any] = {}
    val_lat = {kind: [_Rec(**{**r.__dict__}) for r in va] for kind in HEADS}
    for kind in HEADS:  # the validation recordings at the head's real-time availability (as the eval ones)
        for r in val_lat[kind]:
            r.avail = r.avail + latency[kind]
    for i, kind in enumerate(HEADS):
        recs = _load(ctx, ctx.roles["eval"], groups, latency[kind])  # the head's latency shifts the real-time availability
        if not recs:
            raise ProbeRefused("E_SPLIT_MISSING", f"no present eval recordings in {ctx.roles['eval']}")
        scores = _scores(kind, fits[kind], recs, groups, prior, device)
        labels, smeta = _flatten(recs, groups)
        eval_rec = recs
        floor = {n: np.broadcast_to(prior[n], scores[n].shape).copy() for n in groups}
        specs = {b: {n: s for n, s in dict(ctx.metric_specs.get(b, {})).items() if n.endswith(f"_{kind}")} for b in ("metrics", "controls")}
        specs["floors"] = dict(ctx.metric_specs.get("floors", {})) if i == 0 else {}
        split_meta = {**smeta, "groups": groups, "head": kind, "mode": mode, "bootstrap": dict(ctx.bootstrap or {}),
                      "n_train": len(train_all), "n_eval": len(recs)}
        if va:  # the held-out train recordings: thresholds and operating points are chosen here, never on eval
            vlab, vmeta = _flatten(val_lat[kind], groups)
            split_meta["val"] = {"outputs": _scores(kind, fits[kind], val_lat[kind], groups, prior, device), "targets": vlab, **vmeta}
        ci = {n: k for n, k in dict(ctx.ci or {}).items() if n in specs["metrics"] or n in specs["controls"]}
        got = evaluate_metrics(specs, outputs=scores, targets=labels, protocol=ctx.protocol, split_meta=split_meta, floor_outputs=floor, ci=ci)
        for b in blocks:
            blocks[b].update(got.get(b, {}))
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t2)
    gru = fits["gru"].record
    rows = np.asarray([r.rid for r in eval_rec], dtype=np.int64)
    fit = {"heads": {k: f.record for k, f in fits.items()}, "groups": groups, "mode": mode, "head_latency_s": latency,
           "update_interval_s": _interval(ctx),
           "n_train_recordings": len(tr), "n_val_recordings": len(va), "n_eval_recordings": len(eval_rec),
           "n_eval_targets": int(smeta["recording"].size), "evidence_fraction": float(np.mean(smeta["has_evidence"])) if smeta["recording"].size else None,
           "tbptt": TBPTT, "proj": PROJ, "hidden": HIDDEN}
    return {**blocks,
            "probe": {"selected_lr": gru["selected_lr"], "epochs_run": gru["epochs_run"], "best_epoch": gru["best_epoch"], "stop": gru["stop"],
                      "n_train": len(tr), "n_val": len(va)},
            "fit": fit, "eval_outputs": None, "eval_rows": rows, "extra_arrays": {}, "backend": "ram",
            "timing": {k: ctx.timing[k] for k in ("materialize_s", "fit_s", "score_s") if k in ctx.timing}, "samples": None}


def _floor_outputs(train_targets: Any, n_eval: int, *, protocol: dict, label_kind: str, num_classes: int | None = None) -> dict[str, np.ndarray]:
    """The label-only floor: per group, the train prior on every one of ``n_eval`` targets (``run_stream`` scores the
    same prior at every target with ``_prior``). ``train_targets``: the ``targets`` hook's per-row dicts."""
    out: dict[str, np.ndarray] = {}
    names = sorted({g for t in train_targets for g in t["labels"]})
    for g in names:
        ys = [np.asarray(t["labels"][g]) for t in train_targets]
        if ys and ys[0].ndim == 1:
            y = np.concatenate(ys).astype(np.int64)
            y = y[y >= 0]
            cnt = np.bincount(y, minlength=int(num_classes or (y.max() + 1 if y.size else 1))).astype(np.float64) + 1e-3
            p = cnt / cnt.sum()
        else:
            p = np.clip(np.nan_to_num(np.nanmean(np.concatenate(ys, axis=0), axis=0)), 1e-4, 1 - 1e-4)
        out[g] = np.broadcast_to(p.astype(np.float32), (int(n_eval), p.shape[0])).copy()
    return out


#: recordings per role in the conformance population
POPULATION: dict[str, int] = {"train": 40, "val": 20, "eval": 40}


def _population(ctx: Any, tag: str) -> dict[str, Any]:
    """The conformance population: the folder's own targets (``eval.py`` ``targets``) on a seeded selection of train
    and eval recordings, and deterministic scores that are partly informative (the label plus stable noise), so every
    metric, floor and interval of the folder runs on real target structure without features."""
    from yavih.conformance import _values, select_rows

    groups = {str(k): {"loss": str(v["loss"]), "classes": int(v["classes"])} for k, v in dict(ctx.spec.readout_params["groups"]).items()}

    def role(splits: list[str], n: int, salt: str) -> list[dict[str, Any]]:
        out = []
        for s in splits:
            rows = select_rows(ctx.bench.manifest_hash, s, ctx.bench.n_rows(s), n, salt=salt)
            tg = ctx.hook_targets(s, rows)
            if tg is None:
                from yavih.conformance import E_UNSUPPORTED, ConformanceError

                raise ConformanceError(E_UNSUPPORTED, f"{ctx.bench.name}: a streaming folder's eval.py defines targets(bench, split, rows)")
            out.extend(tg)
        return out

    train = role(list(ctx.roles["train"]), POPULATION["train"], "train")
    val = role(list(ctx.roles["train"]), POPULATION["val"], "val")
    ev = role(list(ctx.roles["eval"]), POPULATION["eval"], "eval")
    def part(tg: list[dict[str, Any]], salt: str) -> dict[str, Any]:
        labels = {g: np.concatenate([np.asarray(t["labels"][g]) for t in tg], axis=0) for g in groups}
        rec = np.concatenate([np.full(len(np.asarray(t["times"])), k) for k, t in enumerate(tg)])
        times = np.concatenate([np.asarray(t["times"], dtype=np.float64) for t in tg])
        names = sorted({n for t in tg for n in dict(t.get("meta") or {})})
        meta = {n: np.concatenate([np.asarray(t["meta"][n]) for t in tg]) for n in names}
        outputs = {}
        for g, spec in groups.items():
            c, y = spec["classes"], labels[g]
            noise = (_values(f"{tag}:stream:{g}" if salt == "eval" else f"{tag}:stream:{salt}:{g}", (len(rec), c)) - 0.5) * 6.0
            if spec["loss"] == "ce":
                onehot = np.zeros((len(rec), c), np.float64)
                ok = y >= 0
                onehot[np.flatnonzero(ok), y[ok].astype(np.int64)] = 2.0
                z = onehot + noise
                z = np.exp(z - z.max(axis=1, keepdims=True))
                outputs[g] = (z / z.sum(axis=1, keepdims=True)).astype(np.float32)
            else:
                outputs[g] = (1.0 / (1.0 + np.exp(-(3.0 * (np.nan_to_num(y) - 0.5) + noise)))).astype(np.float32)
        return {"outputs": outputs, "targets": labels, "recording": rec, "time": times, "meta": meta,
                "has_evidence": np.ones(len(rec), bool)}

    e = part(ev, "eval")
    outputs, labels, rec, times, meta = e["outputs"], e["targets"], e["recording"], e["time"], e["meta"]
    floor = _floor_outputs(train, len(rec), protocol=ctx.protocol, label_kind="none")
    for g, spec in groups.items():
        if floor[g].shape[1] != spec["classes"]:  # a class absent from the drawn train targets
            p = np.zeros(spec["classes"], np.float32)
            p[: floor[g].shape[1]] = floor[g][0]
            floor[g] = np.broadcast_to(p, (len(rec), spec["classes"])).copy()
    split_meta = {"recording": rec, "time": times, "meta": meta, "has_evidence": np.ones(len(rec), bool), "groups": groups, "head": "gru",
                  "mode": "causal", "bootstrap": dict(ctx.bootstrap or {}), "n_train": len(train), "n_eval": len(ev),
                  "val": part(val, "val") if val else None}
    return {"outputs": outputs, "targets": labels, "split_meta": split_meta, "floor_outputs": floor,
            "eval_splits": list(ctx.roles["eval"]), "eval_rows": np.arange(len(ev), dtype=np.int64)}


def _head(*, num_classes: int | None = None, T: int | None = None, grid: Any = None, target_size: int | None = None) -> dict:
    return {"family": "stream_heads_v1", "heads": list(HEADS), "proj": PROJ, "hidden": HIDDEN, "gru_layers": 1, "tbptt": TBPTT,
            "standardize": "train mean/std", "layout": "point", "out": num_classes}


def _loss_name(label_kind: str | None) -> str:
    return "stream_groups"


def _geometry(cfg: Any) -> dict:
    return {"num_classes": cfg.head.get("out")}


def _validate_spec(readout_params: dict, requires: Any, protocol: dict) -> None:
    """``readout_params.groups`` is ``{name: {loss: ce | bce, classes: int >= 2}}`` (``ValueError`` names the key)."""
    groups = readout_params.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(".groups: {name: {loss: ce|bce, classes: C}}, at least one label group")
    for name, g in groups.items():
        if not isinstance(g, dict) or set(g) != {"loss", "classes"} or g["loss"] not in ("ce", "bce") \
                or isinstance(g["classes"], bool) or not isinstance(g["classes"], int) or g["classes"] < 2:
            raise ValueError(f".groups.{name}: {{loss: ce|bce, classes: int >= 2}}")


SCORER = Scorer(
    name="stream", version=VERSION, module=__name__, kind="probe", label_kinds=frozenset({"none"}), param_keys=frozenset({"groups"}),
    requires_t_bins=False, requires_grid=False, dense=False, train=_STREAM_TRAIN, head=_head, loss=_loss_name, geometry=_geometry,
    head_families=frozenset({"stream_heads_v1"}), losses=frozenset({"stream_groups"}), score=run_stream, validate_spec=_validate_spec,
    output_keys=frozenset({"emb", "support", "cutoff"}), floor_outputs=_floor_outputs, conformance_population=_population,
)
