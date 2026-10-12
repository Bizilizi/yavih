"""Readouts (docs/readouts.md, docs/results.md) with small in-test fakes for the
FeatureReader and the trainer, so these tests do not depend on those modules. The
metric functions (numpy) and the label-only floors (``probes.floors``, numpy) are the real ones.
The fakes follow the ``benchmark.yaml`` shape (``output_contract.requires``, ``input_contract``,
``external``) and the ``_meta.json`` (``rules_checked``, ``canary.tolerance``)."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from yavih import hashing
from yavih.probes.config import ProbeConfig
from yavih.readouts import _fit, base
from yavih.readouts.base import (
    ProbeInterrupted,
    ProbeRefused,
    ResultSchemaError,
    check_result,
    probe,
    result_dir,
    write_result,
)
from yavih.versions import FEATURE_FORMAT_VERSION, READOUT_VERSION, RESULT_SCHEMA_VERSION

N_TOK, D_FEAT = 4, 8
#: The user's model name and the reading tag; they key run dirs, results dirs and report rows.
MODEL, READING = "fakemodel", "av"
DESCRIBE = {"name": "fakebridge", "model": MODEL, "reading": READING, "streams": ["video", "audio"], "weights_sha256": "ab" * 32}


# ============================================================================ fakes: benchmark


def make_spec(d: dict) -> NS:
    """A ``BenchmarkSpec``-shaped namespace from a benchmark.yaml dict (only what the driver reads)."""
    m, lab, sp, me, rq = d["manifest"], d["manifest"]["label"], d["splits"], d["metrics"], d["output_contract"]["requires"]
    requires = NS(pos=tuple(rq.get("pos") or ()), streams=tuple(rq.get("streams") or ()), grid=rq.get("grid"), t_bins=rq.get("t_bins"))
    oc = d["output_contract"]
    return NS(
        schema_version=2, name=d["name"], readout=d["readout"], protocol_version=d["protocol_version"],
        manifest=NS(
            classes=m.get("classes"), key=m.get("key", "clip_id"),
            label=NS(column=lab.get("column"), kind=lab["kind"], sep=lab.get("sep", ";"), class_key=lab.get("class_key", "name"),
                     sets=lab.get("sets"), train_set=lab.get("train_set"), allow_unknown=False, empty_ok=lab.get("empty_ok", False)),
        ),
        splits=NS(train=tuple(sp["train"]), eval=tuple(sp["eval"]), floor=tuple(sp.get("floor") or ()),
                  folds=(tuple(sp["folds"]) if sp.get("folds") else None), unused=tuple(sp.get("unused") or ()),
                  min_eval_coverage=float(sp.get("min_eval_coverage", 0.95))),
        streams=dict(d["streams"]), targets=d.get("targets"), protocol=dict(d["protocol"]),
        metrics=NS(headline=me["headline"], list=tuple(me["list"]), floors=tuple(me["floors"]), controls=tuple(me["controls"]),
                   chance=dict(me["chance"]), ci=dict(me["ci"]), bootstrap=(dict(me["bootstrap"]) if me.get("bootstrap") else None),
                   aggregation=me.get("aggregation")),
        output_contract=NS(shape=oc["shape"], coordinates=oc["coordinates"], target_frame=oc.get("target_frame"), requires=requires),
        input_contract=dict(d["input_contract"]), external=list(d.get("external") or []),
        readout_params=dict(d["readout_params"]), raw=d,
    )


class FakeManifest:
    def __init__(self, rows: list[dict]) -> None:
        self.row = np.arange(len(rows), dtype=np.int64)
        self.clip_id = [r["clip_id"] for r in rows]
        self.splits = [r["split"] for r in rows]
        self.label_ids = [r.get("label_ids") for r in rows]
        self.label_sets = [r["label_sets"] for r in rows] if rows and "label_sets" in rows[0] else None
        self.media = [{} for _ in rows]
        self.media_key = [None for _ in rows]
        self.extra = [dict(r.get("extra") or {}) for r in rows]
        self.boxes = np.asarray([r["box"] for r in rows], dtype=np.float32) if rows and "box" in rows[0] else None
        self.segments = np.asarray([r["segment"] for r in rows], dtype=np.int64) if rows and "segment" in rows[0] else None

    def __len__(self) -> int:
        return len(self.clip_id)


class FakeTargetReader:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.kind = "dense_depth"

    def rows(self, idx):
        return self.arr[np.asarray(idx, dtype=np.int64)]


def avsync_probe():
    """avsync's probe.py module: its METRICS overlay holds the offset metrics and floor-arm controls."""
    from yavih.benchmark import Benchmark

    return base.folder_probe(Benchmark.load("avsync"))


class FakeBench:
    def __init__(self, d: dict, rows: list[dict], classes: list[dict] | None, *, manifest_hash: str = "m" * 64, targets: dict | None = None,
                 probe_module=None) -> None:
        self.spec = make_spec(d)
        self.probe_module = probe_module  # the folder probe.py hooks (base.folder_probe reads this attribute)
        self.name = d["name"]
        self.folder = Path("/nonexistent")
        self.manifest_hash = manifest_hash
        # any digest of the whole yaml stands in for the protocol hash here (the driver only compares it with _meta.json)
        self.protocol_hash = hashing.sha256_canon({"yaml": d, "manifest_hash": manifest_hash})
        self.input_hash = hashing.sha256_canon({"input": d, "manifest_hash": manifest_hash})
        self._rows = rows
        self._classes = classes
        self._targets = targets or {}

    def manifest(self, split=None):
        return FakeManifest([r for r in self._rows if split is None or r["split"] == split])

    def classes(self):
        if self._classes is None:
            raise NotImplementedError("no classes.csv")
        return self._classes

    def targets(self, split):
        return self._targets[split]

    def roles(self):
        s = self.spec.splits
        return {"train": list(s.train), "eval": list(s.eval), "floor": list(s.floor), "folds": list(s.folds or ())}


def clf_yaml(name: str, *, n_classes: int, train=("train",), eval=("val",), floor=(), folds=None, metrics=None, floors=None,
             controls=None, chance=None, ci=None, protocol_extra=None, label=None, aggregation=None, min_eval_coverage=0.95) -> dict:
    return {
        "schema_version": 2, "name": name, "aliases": [], "title": name, "readout": "clf", "protocol_version": 2,
        "manifest": {"file": "manifest.csv.gz", "sha256": "0" * 64, "classes": "classes.csv", "classes_sha256": "0" * 64, "key": "clip_id",
                     "label": label or {"column": "label", "kind": "single", "sep": ";", "class_key": "name", "sets": None, "train_set": None,
                                        "allow_unknown": False, "empty_ok": False},
                     "columns": {}},
        "splits": {"train": list(train), "eval": list(eval), "floor": list(floor), "folds": (list(folds) if folds else None), "unused": [],
                   "min_eval_coverage": min_eval_coverage},
        "streams": {"video": {"kind": "video", "media_column": "rel_media", "root": "media_root", "layout": "dir", "layout_params": {},
                              "required": True}},
        "targets": None,
        "input_contract": {"handed_over": {"video": "the stored mp4"}, "info": {"video": ["duration_s"]}, "facts": [], "materialize": None},
        "external": [],
        "protocol": {"clip_seconds": 10.0, **(protocol_extra or {})},
        "metrics": {"headline": (metrics or ["acc", "n_eval"])[0], "list": list(metrics or ["acc", "n_eval"]), "floors": list(floors or ["acc_floor"]),
                    "controls": list(controls or []), "chance": dict(chance or {"acc": round(1 / n_classes, 6)}),
                    "ci": dict(ci if ci is not None else {"acc": "binomial"}), "bootstrap": {"samples": 200, "seed": 0}, "aggregation": aggregation},
        "output_contract": {"shape": "fixed", "coordinates": "media", "target_frame": None,
                            "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None}},
        "readout_params": {},
    }


def clf_rows(split: str, n: int, n_classes: int, *, prefix: str | None = None) -> list[dict]:
    prefix = prefix or split
    return [{"clip_id": f"{prefix}_{k:04d}", "split": split, "label_ids": k % n_classes} for k in range(n)]


def classes_list(n: int) -> list[dict]:
    return [{"index": i, "name": f"class_{i:02d}"} for i in range(n)]


# ============================================================================ fakes: features on disk + reader


def clf_features(labels: np.ndarray, n_classes: int, *, seed: int = 0, N: int = N_TOK, D: int = D_FEAT) -> np.ndarray:
    """Features whose token mean encodes the label (an oracle), so a centroid head is exact."""
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((len(labels), N, D)).astype(np.float32) * 0.1
    for i, c in enumerate(labels):
        out[i, :, int(c) % D] += 3.0
    return out


FAKE_SHARD = "shard-000000.tar"  # the fake world's single "shard" (an npz; see write_fake_features)


def write_fake_features(run_dir: Path, split: str, *, bench: FakeBench, feats: np.ndarray, present: np.ndarray | None = None,
                        pos: np.ndarray | None = None, eligible: bool = True, missing=(), reason=None, validation_ok: bool = True,
                        protocol_hash: str | None = None, manifest_hash: str | None = None, model: str = MODEL, reading: str = READING,
                        describe: dict | None = None, fingerprint: str | None = None, nbytes: int | None = None,
                        format_version: int = FEATURE_FORMAT_VERSION, durations: dict[str, np.ndarray] | None = None,
                        input_hash: str | None = None) -> Path:
    d = run_dir / "features" / split
    d.mkdir(parents=True, exist_ok=True)
    n_rows = feats.shape[0]
    present = np.arange(n_rows, dtype=np.int64) if present is None else np.asarray(present, dtype=np.int64)
    # The fake "shard": one npz under the real tar name so `features stage` (which copies only
    # shard-*.tar / shard-*.json / _meta.json) carries it along like a real shard.
    arrays = {"feats": feats, "present": present}
    if pos is not None:
        arrays["pos"] = np.asarray(pos, dtype=np.float32)  # [N, 4] constant or [n_rows, N, 4] per sample
    for stream, dur in (durations or {}).items():
        arrays[f"dur__{stream}"] = np.asarray(dur, dtype=np.float64)  # media.duration_s per manifest row
    with open(d / FAKE_SHARD, "wb") as fh:
        np.savez(fh, **arrays)
    describe = dict(describe or DESCRIBE)
    man = bench.manifest(split)
    ids = [man.clip_id[int(r)] for r in present]
    meta = {
        "format_version": format_version,
        "yavih": {"writer_versions": ["0.1.0"], "finalizer_version": "0.1.0"},
        "finalized": "2026-10-07T00:00:00Z",
        "benchmark": {"name": bench.name, "protocol_version": bench.spec.protocol_version, "manifest_hash": manifest_hash or bench.manifest_hash,
                      "protocol_hash": protocol_hash or bench.protocol_hash, "media_fingerprint": None, "data_roots": {},
                      "input_hash": input_hash or bench.input_hash},
        "split": split,
        "model": model, "reading": reading,
        "bridge": {"name": describe["name"], "describe": describe, "describe_sha256": hashing.describe_sha256(describe),
                   "source_sha256": "s" * 64},
        "features": {"dtype": str(feats.dtype), "N": int(feats.shape[1]), "D": int(feats.shape[2]),
                     "pos_mode": ("none" if pos is None else "per_sample" if np.asarray(pos).ndim == 3 else "constant"), "pos_sha256": None,
                     "bytes": int(nbytes if nbytes is not None else feats.nbytes)},
        "shards": [{"file": "shard-000000.tar", "rows": [0, n_rows], "n_written": int(len(present)), "bytes": int(feats.nbytes), "sha256": "t" * 64}],
        "coverage": {"nominal": int(n_rows), "present": int(len(present)), "missing": int(n_rows - len(present)), "ids_sha256": base.ids_sha256(ids)},
        "canary": {"max_rel_l2": 0.0, "n": 0, "tolerance": 1e-05},
        "rules_checked": {"video": {"temporal": "uniform", "spatial": True}},
        "dead_rows": [],
        "fingerprint": fingerprint or f"fp-{split}-" + "0" * 56,
        "eligibility": {"readout": bench.spec.readout, "eligible": bool(eligible), "missing": list(missing), "reason": reason,
                        "grid_cell_coverage": None, "t_bins": None, "stream_tags": []},
        "validation": {"ok": bool(validation_ok), "errors": [], "warnings": [], "notes": [], "validated": "2026-10-07T00:00:00Z", "yavih_version": "0.1.0"},
    }
    (d / "_meta.json").write_text(json.dumps(meta))
    return d


class FakeRowView:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr
        self.shape = arr.shape
        self.backend = "ram"

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        return self.arr[idx]


class FakeReader:
    """Stand-in for ``FeatureReader`` over the npy files ``write_fake_features`` leaves in a split dir."""

    calls: list[str] = []

    def __init__(self, split_dir, *, require_meta: bool = True) -> None:
        d = Path(split_dir)
        FakeReader.calls.append(str(d))
        self.meta = json.loads((d / "_meta.json").read_text())
        with np.load(d / FAKE_SHARD) as z:
            self._feats = z["feats"]
            self.present_rows = z["present"]
            self._pos = z["pos"] if "pos" in z.files else None
            self._dur = {k[len("dur__"):]: z[k] for k in z.files if k.startswith("dur__")}
        self.pos_mode = "none" if self._pos is None else "per_sample" if self._pos.ndim == 3 else "constant"
        self.n_rows, self.N, self.D = self._feats.shape
        self.dtype = self._feats.dtype

    def rows(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if not np.isin(idx, self.present_rows).all():
            raise KeyError("row not present")
        return self._feats[idx]

    def matrix(self, *, ram_budget_bytes=None):
        return FakeRowView(self._feats[self.present_rows])

    def pos(self, row=None):
        """As the real reader: per_sample ``pos()`` is ``[n_present, N, 4]`` over the PRESENT
        rows in row order, never ``None``."""
        if self._pos is not None and self._pos.ndim == 3:
            return self._pos[self.present_rows] if row is None else self._pos[int(row)]
        return self._pos

    def durations(self, idx, stream):
        """``float64 [len(idx)]`` media.duration_s from the sidecars; NaN when null."""
        idx = np.asarray(idx, dtype=np.int64)
        if stream not in self._dur:
            return np.full(idx.shape[0], np.nan)
        return self._dur[stream][idx]

    def clip_ids(self):
        man_ids = self.meta.get("clip_ids")
        return man_ids or [str(r) for r in self.present_rows]


# ============================================================================ fakes: trainer


class CentroidHead:
    """Nearest-centroid logits over the token mean: exact on the oracle features."""

    def __init__(self, centroids: np.ndarray) -> None:
        self.c = torch.as_tensor(centroids, dtype=torch.float32)

    def __call__(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        xm = x.float().mean(1)
        return -torch.cdist(xm, self.c.to(xm.device))


class ConstHead:
    def __init__(self, value: np.ndarray) -> None:
        self.v = torch.as_tensor(np.asarray(value, dtype=np.float32))

    def __call__(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        return self.v.to(x.device).unsqueeze(0).expand(x.shape[0], *self.v.shape).clone()


def fake_fit(head, n: int, **over) -> NS:
    rec = dict(head=head, selected_lr=1e-4, epochs_run=3, best_epoch=1, stop="patience", n_train=max(1, n - max(1, round(0.1 * n))),
               n_val=max(1, round(0.1 * n)), s_per_epoch=0.01, history=[], best_val=0.5, arms=[], device="cpu")
    rec.update(over)
    return NS(**rec)


def make_fake_train_head(record: list[dict] | None = None, *, behaviour: str = "fit"):
    """A ``train_head`` stand-in honouring the ``yavih.probes.trainer.train_head`` call contract; ``behaviour`` ∈ fit | stop | deadline | boom."""

    def train_head(features, targets, *, build_head, cfg, seed, loss=None, deadline=None, ckpt_path=None, device=None,
                   loader_workers=None, stop_flag=None, pos=None, **kw):
        if record is not None:
            record.append({"n": int(features.shape[0]), "loss": loss or cfg.loss, "seed": seed, "pos": None if pos is None else np.asarray(pos).shape,
                           "features": features, "targets": targets, "ckpt_path": ckpt_path, "pos_obj": pos})
        if behaviour == "stop":
            stop_flag.set(signal.SIGTERM)
            return fake_fit(ConstHead(np.zeros(1)), int(features.shape[0]))
        if behaviour == "deadline":
            from yavih.probes import trainer

            raise trainer.ProbeDeadline("deadline between epochs")
        if behaviour == "boom":
            raise RuntimeError("boom in the fit")
        n = int(features.shape[0])
        X = np.asarray(features[np.arange(n)], dtype=np.float32).mean(1)  # [n, D]
        t = targets[np.arange(len(targets))] if not isinstance(targets, np.ndarray) else targets
        t = np.asarray(t)
        if t.ndim == 1 and t.dtype.kind in "iu":  # clf single
            c = int(cfg.head["out"])
            cent = np.stack([X[t == k].mean(0) if (t == k).any() else np.zeros(X.shape[1]) for k in range(c)])
            return fake_fit(CentroidHead(cent), n)
        if t.ndim == 2 and t.dtype.kind == "f" and cfg.readout == "clf":  # clf multi
            cent = np.stack([(X * t[:, k:k + 1]).sum(0) / max(1.0, t[:, k].sum()) for k in range(t.shape[1])])
            return fake_fit(CentroidHead(cent), n)
        if cfg.readout == "seq":  # [n, T] → per-segment majority one-hot logits
            c = int(cfg.head["out"])
            T = t.shape[1]
            out = np.zeros((T, c), dtype=np.float32)
            for k in range(T):
                col = t[:, k][t[:, k] >= 0]
                out[k, np.bincount(col.astype(np.int64), minlength=c).argmax()] = 1.0
            return fake_fit(ConstHead(out), n)
        if cfg.readout == "grid":  # [n, S, S] with NaN → nanmean depth map
            return fake_fit(ConstHead(np.nan_to_num(np.nanmean(t, axis=0), nan=1.0)), n)
        if cfg.readout == "box":
            return fake_fit(ConstHead(t.mean(0)), n)
        raise AssertionError(f"fake trainer: unexpected targets {t.shape} {t.dtype}")

    return train_head


@pytest.fixture(autouse=True)
def _results_store_in_tmp(monkeypatch, tmp_path):
    """Several worlds here use ``tmp_path / "run"`` (not the ``runs/<bench>/<model>/<reading>`` layout), so the
    default results root (``<runs root>/../results``) would land outside ``tmp_path``; pin it inside."""
    monkeypatch.setenv("YAVIH_RESULTS", str(tmp_path / "_results"))


@pytest.fixture
def fakes(monkeypatch):
    """Route the driver to the fake reader + trainer; returns the call record."""
    import yavih.probes.trainer as trainer

    FakeReader.calls = []
    record: list[dict] = []
    monkeypatch.setattr(base, "open_reader", lambda d: FakeReader(d))
    monkeypatch.setattr(trainer, "train_head", make_fake_train_head(record))
    return record


def clf_cfg(n_classes: int, *, loss: str = "ce") -> ProbeConfig:
    return ProbeConfig(readout="clf", head={"family": "attentive_probe_v1", "layout": "point", "out": n_classes, "memory_pos": "none"},
                       lr=1e-3, lr_grid=(), weight_decay=0.0, max_epochs=2, batch_size=8, patience=1, warmup_fraction=0.0, loss=loss)


def simple_clf_world(tmp_path: Path, *, n_train=12, n_eval=6, n_classes=3, **meta_kw) -> tuple[FakeBench, Path]:
    d = clf_yaml("synth_clf", n_classes=n_classes)
    rows = clf_rows("train", n_train, n_classes) + clf_rows("val", n_eval, n_classes)
    bench = FakeBench(d, rows, classes_list(n_classes))
    run = tmp_path / "runs" / bench.name / MODEL / READING
    for split in ("train", "val"):
        labels = np.asarray([r["label_ids"] for r in rows if r["split"] == split])
        write_fake_features(run, split, bench=bench, feats=clf_features(labels, n_classes, seed=hash(split) % 1000), **meta_kw.get(split, {}))
    return bench, run


# ============================================================================ outputs / result schema / staleness


def test_result_dir_layout(tmp_path):
    """``probe/<eval_hash12>/seed<k>/`` — the comparability key names the directory."""
    p = result_dir(tmp_path, "a" * 64, "b" * 64, 2)
    assert p == tmp_path / "probe" / hashing.eval_hash("a" * 64, "b" * 64)[:12] / "seed2"


def _ok_record(**over) -> dict:
    rec = {
        "schema_version": RESULT_SCHEMA_VERSION, "status": "ok", "reason": None, "standard": True,
        "eval_hash": hashing.eval_hash("p" * 64, "q" * 64),
        "benchmark": {"name": "x", "protocol_version": 1, "readout": "clf", "readout_version": READOUT_VERSION["clf"], "manifest_hash": "m" * 64, "protocol_hash": "p" * 64},
        "model": MODEL, "reading": READING,
        "bridge": {"name": "b", "describe": {}, "describe_sha256": "d" * 64, "source_sha256": None},
        "features": {"fingerprint": "f" * 64, "dtype": "float32", "N": 4, "D": 8, "pos_mode": "none", "bytes": 10, "backend": "ram", "canary_max_rel_l2": 0.0,
                     "batch_tolerance": 1e-05, "rules_checked": None},
        "probe": {"hash": "q" * 64, "config": {}, "seed": 0, "selected_lr": 1e-4, "epochs_run": 3, "best_epoch": 1, "stop": "patience", "n_train": 10, "n_val": 2},
        "metrics": {"acc": 1.0, "n_eval": 6}, "ci95": {"acc": [0.6, 1.0]}, "floors": {"acc_floor": 0.33}, "controls": {}, "chance": {"acc": 0.33},
        "coverage": {"train": {"present": 12, "nominal": 12, "ids_sha256": "i" * 64}, "eval": {"present": 6, "nominal": 6, "ids_sha256": "j" * 64}, "floor": None},
        "not_applicable": None,
        "timing": {"validate_s": 0.0, "materialize_s": 0.0, "fit_s": 0.1, "score_s": 0.0, "wall_s": 0.2, "s_per_epoch": 0.01},
        "env": {"host": "h", "gpu": None, "torch": None, "python": "3.12", "yavih": "0.1.0", "slurm_job_id": None},
        "created": "2026-10-07T00:00:00Z", "warnings": [], "external": {},
    }
    rec.update(over)
    return rec


def test_check_result_accepts_ok_and_pins_metric_key_sets():
    rec = _ok_record()
    check_result(rec)
    check_result(rec, expect={"metrics": ["acc", "n_eval"], "floors": ["acc_floor"], "controls": [], "ci": {"acc": "binomial"}})
    with pytest.raises(ResultSchemaError, match="metrics keys"):
        check_result(rec, expect={"metrics": ["acc"], "floors": ["acc_floor"], "controls": [], "ci": {"acc": "binomial"}})
    with pytest.raises(ResultSchemaError, match="ci95 keys"):
        check_result(rec, expect={"metrics": ["acc", "n_eval"], "floors": ["acc_floor"], "controls": [], "ci": {"acc": None}})


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda r: r.update(status="weird"), "status"),
        (lambda r: r.update(extra_key=1), "unknown result keys"),
        (lambda r: r.pop("timing"), "missing result keys"),
        (lambda r: r["benchmark"].update(readout_version=99), "readout_version"),
        (lambda r: r.update(status="failed", reason="x"), "must be ABSENT"),
        (lambda r: r.update(not_applicable={"missing": [], "reason": "r", "decided_by": "plan"}), "must be null"),
        (lambda r: r["ci95"].update(acc=[1.0]), r"\[lo, hi\]"),
        # the comparability key must be the one over the two hashes; names keep their charsets
        (lambda r: r.update(eval_hash="e" * 64), "eval_hash"),
        (lambda r: r["probe"].update(hash="r" * 64), "eval_hash"),
        (lambda r: r.update(model="vjepa@0123456789ab"), "model"),
        (lambda r: r.update(model=".."), "model"),
        (lambda r: r.update(model="_stale"), "model"),  # a leading _ is invisible to the report index
        (lambda r: r.update(reading="a+v"), "reading"),
        (lambda r: r.update(reading="_stale"), "reading"),
        (lambda r: r.update(bogus="m@1"), "unknown result keys"),
        (lambda r: r["bridge"].update(bogus=False), "bridge"),
    ],
)
def test_check_result_rejects(mutate, msg):
    rec = _ok_record()
    mutate(rec)
    with pytest.raises(ResultSchemaError, match=msg):
        check_result(rec)


def test_check_result_not_applicable_rules():
    rec = _ok_record(status="not_applicable", reason="nope", features=None, coverage=None,
                     not_applicable={"missing": ["pos.y"], "reason": "nope", "decided_by": "plan"})
    for k in ("metrics", "floors", "controls", "ci95"):
        rec.pop(k)
    check_result(rec)
    rec["not_applicable"]["decided_by"] = "me"
    with pytest.raises(ResultSchemaError, match="decided_by"):
        check_result(rec)
    # features/coverage null is reserved for not_applicable
    bad = _ok_record(features=None)
    with pytest.raises(ResultSchemaError, match="features/coverage"):
        check_result(bad)


def test_write_result_writes_done_marker_only_for_ok_or_na(tmp_path):
    rec = _ok_record()
    p = write_result(tmp_path / "seed0", rec, features_fingerprint="f" * 64)
    assert p == tmp_path / "seed0" / "result.json"
    done = json.loads((tmp_path / "seed0" / "DONE").read_text())
    assert set(done) == {"result_sha256", "protocol_hash", "probe_hash", "eval_hash", "model", "reading", "describe_sha256",
                         "features_fingerprint", "yavih_version", "created"}
    assert done["eval_hash"] == rec["eval_hash"] and (done["model"], done["reading"]) == (MODEL, READING) and done["describe_sha256"] == "d" * 64
    assert done["result_sha256"] == hashing.sha256_hex(p.read_bytes())
    assert done["protocol_hash"] == "p" * 64 and done["probe_hash"] == "q" * 64 and done["features_fingerprint"] == "f" * 64
    failed = _ok_record(status="failed", reason="boom")
    for k in ("metrics", "floors", "controls", "ci95"):
        failed.pop(k)
    write_result(tmp_path / "seed1", failed, features_fingerprint=None)
    assert not (tmp_path / "seed1" / "DONE").exists()
    # a failed rewrite of a seed dir removes a stale DONE
    write_result(tmp_path / "seed0", failed, features_fingerprint=None)
    assert not (tmp_path / "seed0" / "DONE").exists()
    with pytest.raises(ResultSchemaError):
        write_result(tmp_path / "seed2", {"status": "ok"}, features_fingerprint=None)


# ============================================================================ clf: the happy path and its refusals


def test_probe_clf_ok_end_to_end(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    res = probe(run, bench=bench, cfg=clf_cfg(3), seeds=(0, 1))
    assert [r.status for r in res] == ["ok", "ok"] and [r.seed for r in res] == [0, 1]
    rec = res[0].result
    check_result(rec, expect={"metrics": ["acc", "n_eval"], "floors": ["acc_floor"], "controls": [], "ci": {"acc": "binomial"}})
    assert rec["metrics"]["acc"] == 1.0 and rec["metrics"]["n_eval"] == 6
    assert rec["floors"]["acc_floor"] == pytest.approx(1 / 3)
    assert rec["ci95"]["acc"][0] < 1.0 <= rec["ci95"]["acc"][1]
    assert rec["standard"] is False  # a non-pinned cfg under YAVIH_TESTING=1
    assert rec["probe"]["hash"] == hashing.probe_hash(clf_cfg(3)) and rec["probe"]["seed"] == 0
    assert rec["probe"]["n_train"] == 11 and rec["probe"]["n_val"] == 1 and rec["probe"]["stop"] == "patience"
    assert rec["benchmark"] == {"name": "synth_clf", "protocol_version": 2, "readout": "clf", "readout_version": READOUT_VERSION["clf"],
                                "manifest_hash": bench.manifest_hash, "protocol_hash": bench.protocol_hash}
    assert (rec["model"], rec["reading"]) == (MODEL, READING) and rec["bridge"]["describe"] == DESCRIBE and rec["features"]["backend"] == "ram"
    assert rec["eval_hash"] == hashing.eval_hash(bench.protocol_hash, rec["probe"]["hash"])
    assert res[0].path.parent.parent.name == rec["eval_hash"][:12]
    # no decoders block; the batch tolerance and the rules validate checked; no external code
    assert rec["features"]["batch_tolerance"] == 1e-05 and rec["features"]["rules_checked"] == {"video": {"temporal": "uniform", "spatial": True}}
    assert "decoders" not in rec["features"] and rec["external"] == {}
    assert rec["coverage"]["train"]["present"] == 12 and rec["coverage"]["eval"]["present"] == 6 and rec["coverage"]["floor"] is None
    assert rec["coverage"]["eval"]["ids_sha256"] == base.ids_sha256(f"val_{k:04d}" for k in range(6))
    assert rec["chance"] == {"acc": 0.333333} and rec["not_applicable"] is None and rec["warnings"] == []
    assert set(rec["timing"]) == {"validate_s", "materialize_s", "fit_s", "score_s", "wall_s", "s_per_epoch"}
    out = res[0].path.parent
    assert out == result_dir(run, bench.protocol_hash, hashing.probe_hash(clf_cfg(3)), 0)
    logits = np.load(out / "eval_logits.npy")
    rows = np.load(out / "eval_rows.npy")
    assert logits.shape == (6, 3) and logits.dtype == np.float32 and rows.tolist() == list(range(6)) and rows.dtype == np.int64
    fit = json.loads((out / "probe_fit.json").read_text())
    assert fit["epochs_run"] == 3 and fit["s_per_epoch"] == 0.01 and fit["backend"] == "ram" and fit["selected_lr"] == 1e-4
    assert (out / "DONE").exists()
    # the trainer saw the train population only, with the train targets, and never the eval rows
    assert [c["n"] for c in fakes] == [12, 12] and [c["seed"] for c in fakes] == [0, 1]
    assert fakes[0]["loss"] == "ce" and fakes[0]["ckpt_path"].endswith("seed0/probe_ckpt.pt")


def test_probe_pinned_config_is_standard_and_defaults_from_bench(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    res = probe(run, bench=bench)
    rec = res[0].result
    assert rec["standard"] is True
    assert rec["probe"]["config"]["head"]["out"] == 3 and rec["probe"]["config"]["loss"] == "ce"
    assert rec["probe"]["hash"] == hashing.probe_hash(ProbeConfig.pinned("clf", num_classes=3, label_kind="single"))


def test_nonstandard_config_requires_testing_env(tmp_path, fakes, monkeypatch):
    bench, run = simple_clf_world(tmp_path)
    monkeypatch.delenv("YAVIH_TESTING")
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=clf_cfg(3))
    assert e.value.code == "E_NONSTANDARD_CONFIG" and e.value.exit_code == 1


def test_readout_must_match_benchmark(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, readout="seq", cfg=clf_cfg(3))
    assert e.value.code == "E_READOUT" and e.value.exit_code == 1


def test_probe_skips_matching_done_and_moves_stale(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    first = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    again = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    assert again.skipped is True and again.status == "ok" and again.path == first.path
    assert len(fakes) == 1  # no second fit
    # re-extracted features (new fingerprint) make the DONE stale: outputs move to _stale/<utc>/
    labels = np.asarray([r["label_ids"] for r in bench._rows if r["split"] == "train"])
    write_fake_features(run, "train", bench=bench, feats=clf_features(labels, 3, seed=9), fingerprint="fp-train-new" + "1" * 52)
    third = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    assert third.skipped is False and len(fakes) == 2
    stale = list((run / "probe" / "_stale").glob("*/*/seed0/result.json"))
    assert len(stale) == 1
    assert json.loads((third.path.parent / "DONE").read_text())["features_fingerprint"] != json.loads(stale[0].parent.joinpath("DONE").read_text())["features_fingerprint"]


def test_reconfigured_bridge_is_reprobed_in_the_same_directory(tmp_path, fakes):
    """A new bridge configuration (another ``describe_sha256``) under the same model name
    and reading makes DONE stale even with an identical features fingerprint; the probe directory, the result key
    and the eval_hash do not move — the configuration is recorded, never a key."""
    bench, run = simple_clf_world(tmp_path, train={"fingerprint": "fp-same-train" + "0" * 51}, val={"fingerprint": "fp-same-val" + "0" * 53})
    first = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    other = dict(DESCRIBE, layer=18, dtype="float32")
    for split in ("train", "val"):
        labels = np.asarray([r["label_ids"] for r in bench._rows if r["split"] == split])
        write_fake_features(run, split, bench=bench, feats=clf_features(labels, 3, seed=hash(split) % 1000), describe=other,
                            fingerprint=f"fp-same-{split}" + "0" * (64 - len(f"fp-same-{split}")))
    second = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    assert second.skipped is False and len(fakes) == 2 and second.path == first.path
    assert second.result["eval_hash"] == first.result["eval_hash"] and (second.result["model"], second.result["reading"]) == (MODEL, READING)
    assert second.result["bridge"]["describe_sha256"] == hashing.describe_sha256(other) != first.result["bridge"]["describe_sha256"]
    assert json.loads((second.path.parent / "DONE").read_text())["describe_sha256"] == hashing.describe_sha256(other)
    assert len(list((run / "probe" / "_stale").glob(f"*/{first.result['eval_hash'][:12]}/seed0/result.json"))) == 1


def test_null_weights_fingerprint_is_a_recorded_warning_only(tmp_path, fakes):
    """``W_WEIGHTS_UNFINGERPRINTED`` in ``warnings``; no ``~``, no other effect on the record or its path."""
    unfp = dict(DESCRIBE, weights_sha256=None)
    bench, run = simple_clf_world(tmp_path, train={"describe": unfp}, val={"describe": unfp})
    res = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    assert res.status == "ok" and [w["code"] for w in res.result["warnings"]] == ["W_WEIGHTS_UNFINGERPRINTED"]
    assert res.result["model"] == MODEL and res.path == result_dir(run, bench.protocol_hash, hashing.probe_hash(clf_cfg(3)), 0) / "result.json"
    check_result(res.result)


def test_probe_accepts_features_of_another_protocol_hash_with_the_same_input_hash(tmp_path, fakes):
    """A scoring edit (another protocol_hash) never refuses extracted features."""
    bench, run = simple_clf_world(tmp_path, train={"protocol_hash": "z" * 64}, val={"protocol_hash": "z" * 64})
    res = probe(run, bench=bench, cfg=clf_cfg(3))[0]
    assert res.status == "ok" and res.result["benchmark"]["protocol_hash"] == bench.protocol_hash


@pytest.mark.parametrize(
    "case, code, exit_code",
    [
        ("mixed_input", "E_HASH_MISMATCH", 2),
        ("stale_input", "E_HASH_INPUT", 2),
        ("stale_manifest", "E_HASH_MANIFEST", 2),
        ("mixed_model", "E_DESCRIBE_MISMATCH", 2),
        ("mixed_reading", "E_DESCRIBE_MISMATCH", 2),
        ("old_format", "E_FORMAT", 2),
        ("validation_failed", "E_VALIDATION", 2),
        ("format_version", "E_FORMAT", 2),
        ("meta_missing", "E_META_MISSING", 2),
        ("eval_missing", "E_SPLIT_MISSING", 2),
        ("no_features", "E_NO_FEATURES", 3),
        ("eligibility_disagree", "E_ELIGIBILITY_MISMATCH", 2),
    ],
)
def test_probe_refusals(tmp_path, fakes, case, code, exit_code):
    kw: dict[str, dict] = {}
    if case == "mixed_input":
        kw["val"] = {"input_hash": "z" * 64}
    elif case == "stale_input":
        kw["train"] = kw["val"] = {"input_hash": "z" * 64}
    elif case == "stale_manifest":
        kw["train"] = kw["val"] = {"manifest_hash": "z" * 64}
    elif case == "mixed_model":
        kw["val"] = {"model": "othermodel"}
    elif case == "mixed_reading":
        kw["val"] = {"reading": "v"}
    elif case == "old_format":  # an older feature format must be re-extracted
        kw["train"] = kw["val"] = {"format_version": FEATURE_FORMAT_VERSION - 1}
    elif case == "validation_failed":
        kw["val"] = {"validation_ok": False}
    elif case == "format_version":
        kw["train"] = {"format_version": 99}
    elif case == "eligibility_disagree":
        kw["val"] = {"eligible": False, "missing": ["pos.t"]}
    bench, run = simple_clf_world(tmp_path, **kw)
    if case == "meta_missing":
        (run / "features" / "val" / "_meta.json").unlink()
    elif case == "eval_missing":
        import shutil

        shutil.rmtree(run / "features" / "val")
    elif case == "no_features":
        import shutil

        shutil.rmtree(run / "features")
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=clf_cfg(3))
    assert e.value.code == code and e.value.exit_code == exit_code
    assert not (run / "probe").exists()


def test_probe_refuses_train_eval_overlap(tmp_path, fakes):
    d = clf_yaml("synth_clf", n_classes=3)
    rows = clf_rows("train", 6, 3) + clf_rows("val", 6, 3, prefix="train")  # same clip_ids in both roles
    bench = FakeBench(d, rows, classes_list(3))
    run = tmp_path / "run"
    for split in ("train", "val"):
        write_fake_features(run, split, bench=bench, feats=clf_features(np.arange(6) % 3, 3))
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=clf_cfg(3))
    assert e.value.code == "E_LEAKAGE"


def test_probe_coverage_refusal_and_allow_low_coverage(tmp_path, fakes):
    d = clf_yaml("synth_clf", n_classes=3)
    rows = clf_rows("train", 12, 3) + clf_rows("val", 10, 3)
    bench = FakeBench(d, rows, classes_list(3))
    run = tmp_path / "run"
    write_fake_features(run, "train", bench=bench, feats=clf_features(np.arange(12) % 3, 3))
    write_fake_features(run, "val", bench=bench, feats=clf_features(np.arange(10) % 3, 3), present=np.arange(8))  # 0.8 < 0.95
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=clf_cfg(3))
    assert e.value.code == "E_COVERAGE"
    res = probe(run, bench=bench, cfg=clf_cfg(3), allow_low_coverage=True)[0]
    rec = res.result
    assert rec["status"] == "ok" and rec["metrics"]["n_eval"] == 8 and rec["coverage"]["eval"] == {"present": 8, "nominal": 10, "ids_sha256": base.ids_sha256(f"val_{k:04d}" for k in range(8))}
    assert [w["code"] for w in rec["warnings"]] == ["W_COVERAGE"]
    assert np.load(res.path.parent / "eval_rows.npy").tolist() == list(range(8))


def test_probe_metrics_incomplete_is_a_refusal(tmp_path, fakes, monkeypatch):
    bench, run = simple_clf_world(tmp_path)
    real = base.resolve_metric_specs

    def broken(spec, probe_mod=None):
        specs = real(spec, probe_mod)
        specs["metrics"] = dict(specs["metrics"])
        specs["metrics"]["acc"] = NS(name="acc", fn=lambda *a: (_ for _ in ()).throw(ValueError("no")), shape="scalar", kind="metric")
        return specs

    monkeypatch.setattr(base, "resolve_metric_specs", broken)
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=clf_cfg(3))
    assert e.value.code == "E_METRICS_INCOMPLETE" and "acc" in str(e.value)


def test_probe_failure_writes_failed_result_without_done(tmp_path, fakes, monkeypatch):
    import yavih.probes.trainer as trainer

    bench, run = simple_clf_world(tmp_path)
    monkeypatch.setattr(trainer, "train_head", make_fake_train_head(behaviour="boom"))
    with pytest.raises(RuntimeError, match="boom"):
        probe(run, bench=bench, cfg=clf_cfg(3))
    seed_dir = result_dir(run, bench.protocol_hash, hashing.probe_hash(clf_cfg(3)), 0)
    rec = json.loads((seed_dir / "result.json").read_text())
    assert rec["status"] == "failed" and "boom" in rec["reason"] and "metrics" not in rec
    assert not (seed_dir / "DONE").exists()


@pytest.mark.parametrize("behaviour, code", [("stop", "E_PREEMPTED"), ("deadline", "E_DEADLINE")])
def test_probe_interrupted_exits_4_without_partial_result(tmp_path, fakes, monkeypatch, behaviour, code):
    import yavih.probes.trainer as trainer

    bench, run = simple_clf_world(tmp_path)
    monkeypatch.setattr(trainer, "train_head", make_fake_train_head(behaviour=behaviour))
    with pytest.raises(ProbeInterrupted) as e:
        probe(run, bench=bench, cfg=clf_cfg(3))
    assert e.value.exit_code == 4 and e.value.code == code
    assert isinstance(e.value, TimeoutError)
    seed_dir = result_dir(run, bench.protocol_hash, hashing.probe_hash(clf_cfg(3)), 0)
    assert not (seed_dir / "result.json").exists() and not (seed_dir / "DONE").exists()


def test_probe_past_deadline_before_fit(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    with pytest.raises(ProbeInterrupted) as e:
        probe(run, bench=bench, cfg=clf_cfg(3), deadline=time.time() - 1)
    assert e.value.code == "E_DEADLINE" and fakes == []


# ============================================================================ not_applicable: validate / bridge / plan


def test_not_applicable_from_meta_eligibility(tmp_path, fakes):
    kw = {"train": {"eligible": False, "missing": ["pos.y", "pos.x"], "reason": "N == 1"}, "val": {"eligible": False, "missing": ["pos.y", "pos.x"], "reason": "N == 1"}}
    bench, run = simple_clf_world(tmp_path, **kw)
    res = probe(run, bench=bench, cfg=clf_cfg(3))
    assert len(res) == 1 and res[0].status == "not_applicable"
    rec = res[0].result
    check_result(rec)
    assert rec["not_applicable"] == {"missing": ["pos.y", "pos.x"], "reason": "N == 1", "decided_by": "validate"}
    assert rec["reason"] == "N == 1" and "metrics" not in rec and "floors" not in rec
    assert rec["features"]["N"] == N_TOK and rec["features"]["D"] == D_FEAT and rec["features"]["dtype"] == "float32"
    assert rec["coverage"]["train"]["present"] == 12 and rec["coverage"]["eval"]["present"] == 6
    assert rec["probe"]["epochs_run"] is None and rec["probe"]["hash"] == hashing.probe_hash(clf_cfg(3))
    assert (res[0].path.parent / "DONE").exists() and fakes == []  # exit 0 path, no fit


def test_not_applicable_from_bridge_file(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    import shutil

    shutil.rmtree(run / "features" / "val")
    (run / "features" / "val").mkdir()
    (run / "features" / "val" / "NOT_APPLICABLE.json").write_text(json.dumps({
        "status": "not_applicable", "reason": "audio-only bridge: benchmark has no audio stream", "benchmark": bench.name, "split": "val",
        "model": MODEL, "reading": READING, "describe_sha256": hashing.describe_sha256(DESCRIBE), "bridge": DESCRIBE,
        "created": "2026-10-07T00:00:00Z"}))
    res = probe(run, bench=bench, cfg=clf_cfg(3))
    rec = res[0].result
    check_result(rec)
    assert rec["status"] == "not_applicable" and rec["not_applicable"]["decided_by"] == "bridge"
    assert rec["reason"].startswith("audio-only") and (res[0].path.parent / "DONE").exists()
    # null features/coverage are reserved for the plan-time case; a bridge-declared NA carries
    # the manifest's nominal rows (nothing present) and the train split's real coverage
    n_val, n_train = len(bench.manifest("val").row), len(bench.manifest("train").row)
    assert rec["coverage"]["eval"] == {"present": 0, "nominal": n_val, "ids_sha256": None}
    assert rec["coverage"]["train"]["present"] == rec["coverage"]["train"]["nominal"] == n_train
    assert rec["features"]["fingerprint"] and rec["features"]["N"] is not None


def test_not_applicable_plan_time_unit(tmp_path, fakes):
    bench = FakeBench(clf_yaml("synth_clf", n_classes=3), clf_rows("train", 4, 3) + clf_rows("val", 2, 3), classes_list(3))
    run = tmp_path / "runs" / bench.name / MODEL / READING
    cfg = clf_cfg(3)
    unit = {"schema_version": 2, "id": "abc", "kind": "probe", "benchmark": bench.name, "bridge": "fakebridge", "model": MODEL, "reading": READING,
            "split": None, "shard": None, "seed": 0, "protocol_hash": bench.protocol_hash, "probe_hash": hashing.probe_hash(cfg),
            "expected_status": "not_applicable"}
    profile = {"bench": bench.name, "describe": DESCRIBE, "describe_sha256": hashing.describe_sha256(DESCRIBE), "model": MODEL, "reading": READING,
               "N": 1, "D": 8, "pos_mode": "none", "streams": ["video", "audio"]}
    res = probe(run, bench=bench, cfg=cfg, unit=unit, profile=profile)
    assert len(res) == 1 and res[0].status == "not_applicable" and res[0].seed == 0
    rec = res[0].result
    check_result(rec)
    assert rec["features"] is None and rec["coverage"] is None and rec["not_applicable"]["decided_by"] == "plan"
    assert rec["bridge"]["describe"] == DESCRIBE and (rec["model"], rec["reading"]) == (MODEL, READING)
    assert res[0].path == result_dir(run, bench.protocol_hash, hashing.probe_hash(cfg), 0) / "result.json"
    done = json.loads((res[0].path.parent / "DONE").read_text())
    assert done["features_fingerprint"] is None and (done["model"], done["reading"]) == (MODEL, READING)
    assert done["describe_sha256"] == hashing.describe_sha256(DESCRIBE)
    # idempotent: a matching DONE is skipped
    assert probe(run, bench=bench, cfg=cfg, unit=unit, profile=profile)[0].skipped is True
    # a unit planned under another protocol is refused
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=cfg, unit={**unit, "protocol_hash": "z" * 64}, profile=profile)
    assert e.value.code == "E_HASH_PROTOCOL"


def test_plan_time_missing_rules():
    spec_clf = make_spec(clf_yaml("x", n_classes=2))
    assert base.plan_time_missing(spec_clf, {"N": 1, "pos_mode": "none", "streams": ["video"]}) == []
    d = clf_yaml("x", n_classes=2)
    d["readout"] = "grid"
    d["output_contract"]["requires"] = {"pos": ["y", "x"], "streams": ["image"], "grid": {"h": 14, "w": 14, "min_cell_coverage": 0.5}, "t_bins": None}
    spec = make_spec(d)
    assert base.plan_time_missing(spec, {"N": 1, "pos_mode": "none", "streams": ["image"]}) == ["pos.y", "pos.x"]
    assert base.plan_time_missing(spec, {"N": 196, "pos_mode": "constant", "streams": ["video"]}) == ["stream.image"]
    assert base.plan_time_missing(spec, {"N": 196, "pos_mode": "constant", "streams": ["image"]}) == []


# ============================================================================ clf variants: floor role, folds, multi, multi_sets


def test_clf_floor_role_controls(tmp_path, fakes):
    offsets = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    d = clf_yaml("avsync_like", n_classes=9, floor=("floor",),
                 metrics=["acc", "acc_tol1", "mae_seconds", "acc_coarse5", "n_coarse5", "n_eval"], floors=["acc_floor"],
                 controls=["acc_floor_arm", "floor_violated"], chance={"acc": 0.111111},
                 protocol_extra={"offsets_seconds": offsets, "nominal_window_start": 1.0, "floor_sigmas": 3.0, "coarse_grid_seconds": 1.0})
    rows = clf_rows("train", 27, 9) + clf_rows("val", 9, 9) + clf_rows("floor", 18, 9)
    bench = FakeBench(d, rows, classes_list(9), probe_module=avsync_probe())
    run = tmp_path / "run"
    for split, n in (("train", 27), ("val", 9)):
        write_fake_features(run, split, bench=bench, feats=clf_features(np.arange(n) % 9, 9, seed=1))
    # floor rows carry NO label information in their features: the trained probe must land near chance there
    write_fake_features(run, "floor", bench=bench, feats=np.random.default_rng(5).standard_normal((18, N_TOK, D_FEAT)).astype(np.float32))
    rec = probe(run, bench=bench, cfg=clf_cfg(9))[0].result
    check_result(rec, expect={"metrics": d["metrics"]["list"], "floors": ["acc_floor"], "controls": ["acc_floor_arm", "floor_violated"], "ci": {"acc": "binomial"}})
    assert rec["metrics"]["acc"] == 1.0 and rec["metrics"]["acc_tol1"] == 1.0 and rec["metrics"]["mae_seconds"] == 0.0
    assert rec["metrics"]["n_coarse5"] == 5 and rec["metrics"]["acc_coarse5"] == 1.0
    assert 0.0 <= rec["controls"]["acc_floor_arm"] <= 1.0 and isinstance(rec["controls"]["floor_violated"], bool)
    assert rec["coverage"]["floor"] == {"present": 18, "nominal": 18, "ids_sha256": base.ids_sha256(f"floor_{k:04d}" for k in range(18))}
    out = Path(probe(run, bench=bench, cfg=clf_cfg(9))[0].path).parent
    assert np.load(out / "floor_logits.npy").shape == (18, 9) and np.load(out / "floor_rows.npy").tolist() == list(range(18))


def test_clf_folds(tmp_path, fakes):
    folds = ["1", "2", "3"]
    d = clf_yaml("esc_like", n_classes=4, train=(), eval=(), folds=folds,
                 metrics=["acc_mean", "acc_pstd", "acc_per_fold", "n_folds", "n_eval"], floors=["acc_floor"], ci={"acc_mean": "pooled_binomial"},
                 chance={"acc_mean": 0.25}, aggregation={"over": "folds", "report": ["mean", "pstd", "per_fold"], "ci": "pooled_binomial"})
    rows = [r for f in folds for r in clf_rows(f, 8, 4, prefix=f"f{f}")]
    bench = FakeBench(d, rows, classes_list(4))
    run = tmp_path / "run"
    for f in folds:
        write_fake_features(run, f, bench=bench, feats=clf_features(np.arange(8) % 4, 4, seed=int(f)))
    rec = probe(run, bench=bench, cfg=clf_cfg(4))[0].result
    check_result(rec, expect={"metrics": d["metrics"]["list"], "floors": ["acc_floor"], "controls": [], "ci": {"acc_mean": "pooled_binomial"}})
    assert rec["metrics"]["acc_mean"] == 1.0 and rec["metrics"]["acc_pstd"] == 0.0 and rec["metrics"]["acc_per_fold"] == [1.0, 1.0, 1.0]
    assert rec["metrics"]["n_folds"] == 3 and rec["metrics"]["n_eval"] == 24 and rec["floors"]["acc_floor"] == pytest.approx(0.25)
    assert rec["ci95"]["acc_mean"][1] == 1.0
    assert [c["n"] for c in fakes] == [16, 16, 16]  # train = the two other folds, each time
    # one checkpoint identity per fold: fold k can never resume fold 1's finished head
    assert [Path(c["ckpt_path"]).name for c in fakes] == ["probe_ckpt.fold0.pt", "probe_ckpt.fold1.pt", "probe_ckpt.fold2.pt"]
    assert rec["coverage"]["folds"].keys() == set(folds) and rec["coverage"]["train"] is None
    out = Path(result_dir(run, bench.protocol_hash, hashing.probe_hash(clf_cfg(4)), 0))
    assert np.load(out / "eval_logits.npy").shape == (24, 4) and np.load(out / "eval_fold_index.npy").tolist() == [0] * 8 + [1] * 8 + [2] * 8
    assert len(json.loads((out / "probe_fit.json").read_text())["folds"]) == 3


def test_clf_folds_count_mismatch_is_refused(tmp_path, fakes):
    folds = ["1", "2", "3"]
    d = clf_yaml("esc_like", n_classes=4, train=(), eval=(), folds=folds, metrics=["acc_mean", "n_eval"], ci={}, chance={"acc_mean": 0.25})
    bench = FakeBench(d, [r for f in folds for r in clf_rows(f, 4, 4, prefix=f"f{f}")], classes_list(4))
    run = tmp_path / "run"
    for f in folds[:2]:
        write_fake_features(run, f, bench=bench, feats=clf_features(np.arange(4) % 4, 4))
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=clf_cfg(4))
    assert e.value.code == "E_SPLIT_MISSING"


def test_clf_multi_label(tmp_path, fakes):
    d = clf_yaml("fsd_like", n_classes=4, metrics=["map", "effective_pos_classes", "n_eval"], floors=["map_floor"], ci={}, chance={"map": 0.1},
                 label={"column": "labels", "kind": "multi", "sep": ";", "class_key": "name", "sets": None, "train_set": None, "allow_unknown": False, "empty_ok": False})
    rows = [{"clip_id": f"train_{k}", "split": "train", "label_ids": [k % 4, (k + 1) % 4]} for k in range(16)] + \
           [{"clip_id": f"val_{k}", "split": "val", "label_ids": [k % 4]} for k in range(8)]
    bench = FakeBench(d, rows, classes_list(4))
    run = tmp_path / "run"
    write_fake_features(run, "train", bench=bench, feats=clf_features(np.arange(16) % 4, 4))
    write_fake_features(run, "val", bench=bench, feats=clf_features(np.arange(8) % 4, 4))
    rec = probe(run, bench=bench, cfg=clf_cfg(4, loss="bce"))[0].result
    check_result(rec, expect={"metrics": d["metrics"]["list"], "floors": ["map_floor"], "controls": [], "ci": {}})
    assert 0.0 < rec["metrics"]["map"] <= 1.0 and rec["metrics"]["effective_pos_classes"] == 4 and 0.0 < rec["floors"]["map_floor"] <= 1.0
    t = fakes[0]["targets"]
    assert t.shape == (16, 4) and t.dtype == np.float32 and t.sum(1).tolist() == [2.0] * 16


def test_clf_multi_sets(tmp_path, fakes):
    sets = {"audible": ["labels_a", "labels_av"], "all": ["labels_a", "labels_v", "labels_av"]}
    d = clf_yaml("vgg_like", n_classes=3, metrics=["auc_pr_audible", "auc_roc_audible", "auc_pr_all", "n_eval"],
                 floors=["auc_pr_audible_floor", "auc_pr_all_floor"], ci={}, chance={"auc_pr_audible": 0.1},
                 label={"column": "label", "kind": "multi_sets", "sep": ";", "class_key": "name", "sets": sets, "train_set": None, "allow_unknown": False, "empty_ok": True})
    rows = [{"clip_id": f"train_{k}", "split": "train", "label_ids": [k % 3], "label_sets": {}} for k in range(12)] + \
           [{"clip_id": f"val_{k}", "split": "val", "label_ids": [], "label_sets": {"audible": [k % 3], "all": [k % 3, (k + 1) % 3]}} for k in range(9)]
    bench = FakeBench(d, rows, classes_list(3))
    run = tmp_path / "run"
    write_fake_features(run, "train", bench=bench, feats=clf_features(np.arange(12) % 3, 3))
    write_fake_features(run, "val", bench=bench, feats=clf_features(np.arange(9) % 3, 3))
    rec = probe(run, bench=bench, cfg=clf_cfg(3, loss="bce"))[0].result
    check_result(rec, expect={"metrics": d["metrics"]["list"], "floors": d["metrics"]["floors"], "controls": [], "ci": {}})
    assert rec["metrics"]["auc_pr_audible"] == 1.0 and rec["metrics"]["auc_roc_audible"] == 1.0 and rec["metrics"]["auc_pr_all"] < 1.0
    assert rec["metrics"]["n_eval"] == 9 and rec["floors"]["auc_pr_audible_floor"] < rec["metrics"]["auc_pr_audible"]
    assert fakes[0]["targets"].shape == (12, 3) and fakes[0]["targets"].sum(1).tolist() == [1.0] * 12  # one-hot train targets


# ============================================================================ seq / grid / box


def dense_yaml(name: str, readout: str, *, protocol: dict, requires: dict, readout_params: dict, metrics: list, floors: list, controls: list,
               ci: dict, chance: dict, label: dict, classes: str | None, targets=None, streams=("video", "audio")) -> dict:
    d = clf_yaml(name, n_classes=2, metrics=metrics, floors=floors, controls=controls, ci=ci, chance=chance, label=label)
    d["readout"] = readout
    d["protocol"].update(protocol)
    d["output_contract"]["requires"] = requires
    d["output_contract"]["target_frame"] = "full_frame" if readout in ("grid", "box") else None
    d["readout_params"] = readout_params
    d["manifest"]["classes"] = classes
    d["manifest"]["classes_sha256"] = "0" * 64 if classes else None
    d["targets"] = targets
    d["streams"] = {s: {"kind": "video" if s != "audio" else "audio", "media_column": f"rel_{s}", "root": "media_root", "layout": "dir", "layout_params": {},
                        "required": True} for s in streams}
    d["input_contract"] = {"handed_over": {s: f"the stored {s} file" for s in streams}, "info": {s: ["duration_s"] for s in streams}, "facts": [], "materialize": None}
    return d


def test_seq_readout(tmp_path, fakes):
    T, C, N = 8, 5, 8  # background index 4
    protocol = {"num_segments": 10, "segment_seconds": 1.0, "segments_scored": T, "segment_offset": 1, "background_index": 4}
    d = dense_yaml("ave_like", "seq", protocol=protocol, requires={"pos": ["t"], "streams": ["video"], "grid": None, "t_bins": T}, readout_params={"T": T},
                   metrics=["seg_acc", "clip_acc", "seg_acc_boundary", "n_boundary", "n_eval"], floors=["seg_acc_floor", "seg_acc_boundary_floor"], controls=[],
                   ci={"seg_acc": "bootstrap_clip", "seg_acc_boundary": "bootstrap_clip"}, chance={"seg_acc": 0.2},
                   label={"column": "label", "kind": "segments", "sep": ";", "class_key": "name", "sets": None, "train_set": None, "allow_unknown": False, "empty_ok": False},
                   classes="classes.csv")
    rows = []
    for split, n in (("train", 12), ("val", 6)):
        for k in range(n):
            s, e = (k % 3), (k % 3) + 3 + (k % 2) * 7  # some events span the whole clip (e - s >= 10), some do not
            rows.append({"clip_id": f"{split}_{k}", "split": split, "label_ids": k % 4, "extra": {"event_start": s, "event_end": min(e, 10)}, "segment": [s, min(e, 10)]})
    bench = FakeBench(d, rows, classes_list(C))
    run = tmp_path / "run"
    pos = np.stack([np.zeros(N), (np.arange(N) + 0.5) / N, -np.ones(N), -np.ones(N)], 1).astype(np.float32)
    for split, n in (("train", 12), ("val", 6)):
        write_fake_features(run, split, bench=bench, feats=np.random.default_rng(1).standard_normal((n, N, D_FEAT)).astype(np.float32), pos=pos,
                            durations={"video": np.full(n, 10.0)})
    cfg = ProbeConfig.pinned("seq", num_classes=C, T=T).fast()
    rec = probe(run, bench=bench, cfg=cfg)[0].result
    check_result(rec, expect={"metrics": d["metrics"]["list"], "floors": d["metrics"]["floors"], "controls": [], "ci": d["metrics"]["ci"]})
    assert rec["metrics"]["n_eval"] == 6 and 0 < rec["metrics"]["n_boundary"] <= 6
    assert 0.0 <= rec["metrics"]["seg_acc"] <= 1.0 and 0.0 <= rec["floors"]["seg_acc_floor"] <= 1.0
    assert len(rec["ci95"]) == 2 and rec["ci95"]["seg_acc"][0] <= rec["ci95"]["seg_acc"][1]
    # pos reaches the head per row in the task frame (u from absolute seconds)
    assert fakes[0]["targets"].shape == (12, T) and fakes[0]["pos"] == (12, N, 4)
    # the seq target rule on one row: clip train_0 (class 0) has event [0, 3) → media seconds 1, 2 of the
    # scored window [1, 9) carry the class (k = 0, 1), the rest is background_index 4
    t = fakes[0]["targets"][0]
    assert t.tolist() == [0, 0, 4, 4, 4, 4, 4, 4]
    out = Path(result_dir(run, bench.protocol_hash, hashing.probe_hash(cfg), 0))
    assert np.load(out / "eval_logits.npy").shape == (6, T, C)


def test_grid_readout(tmp_path, fakes):
    S, H = 8, 2
    protocol = {"max_depth": 10.0, "target_size": S, "depth_param": "log", "source_size": [480, 640]}
    targets_yaml = {"train": {"kind": "dense_depth", "source": "derived_file", "root": "derived_root", "file": "train.npy", "sha256": "1" * 64, "index_column": None},
                    "val": {"kind": "dense_depth", "source": "derived_file", "root": "derived_root", "file": "eval.npy", "sha256": "2" * 64, "index_column": None}}
    d = dense_yaml("nyu_like", "grid", protocol=protocol, requires={"pos": ["y", "x"], "streams": ["image"], "grid": {"h": H, "w": H, "min_cell_coverage": 0.5}, "t_bins": None},
                   readout_params={"h": H, "w": H, "out": 1, "visual_stream": "image"}, metrics=["d1", "d2", "d3", "absrel", "rmse", "n_eval"], floors=["d1_floor"],
                   controls=[], ci={}, chance={"d1": 0.0},
                   label={"column": None, "kind": "none", "sep": ";", "class_key": "name", "sets": None, "train_set": None, "allow_unknown": False, "empty_ok": True},
                   classes=None, targets=targets_yaml, streams=("image",))
    rng = np.random.default_rng(0)
    depth = {"train": rng.uniform(1.0, 5.0, (10, S, S)).astype(np.float32), "val": rng.uniform(1.0, 5.0, (4, S, S)).astype(np.float32)}
    depth["train"][:, 0, 0] = 0.0  # invalid pixels
    depth["val"][:, 0, :] = 0.0
    rows = [{"clip_id": f"{s}_{k}", "split": s, "label_ids": None} for s, n in (("train", 10), ("val", 4)) for k in range(n)]
    bench = FakeBench(d, rows, None, targets={s: FakeTargetReader(a) for s, a in depth.items()})
    run = tmp_path / "run"
    N = H * H
    yy, xx = np.meshgrid((np.arange(H) + 0.5) / H, (np.arange(H) + 0.5) / H, indexing="ij")
    pos = np.stack([np.zeros(N), -np.ones(N), yy.ravel(), xx.ravel()], 1).astype(np.float32)
    describe = {**DESCRIBE, "streams": ["image"]}
    for split, n in (("train", 10), ("val", 4)):
        write_fake_features(run, split, bench=bench, feats=rng.standard_normal((n, N, D_FEAT)).astype(np.float32), pos=pos, describe=describe)
    cfg = ProbeConfig.pinned("grid", grid=(H, H), target_size=S).fast()
    rec = probe(run, bench=bench, cfg=cfg)[0].result
    check_result(rec, expect={"metrics": d["metrics"]["list"], "floors": ["d1_floor"], "controls": [], "ci": {}})
    assert rec["metrics"]["n_eval"] == 4 and 0.0 <= rec["metrics"]["d1"] <= 1.0 and rec["metrics"]["rmse"] >= 0.0
    assert 0.0 <= rec["floors"]["d1_floor"] <= 1.0
    t = fakes[0]["targets"]
    got = t[np.arange(len(t))]
    assert got.shape == (10, S, S) and np.isnan(got[:, 0, 0]).all() and np.isfinite(got[:, 1, 1]).all()  # invalid train pixels are NaN for the loss
    out = Path(result_dir(run, bench.protocol_hash, hashing.probe_hash(cfg), 0))
    assert np.load(out / "eval_logits.npy").shape == (4, S, S)


def test_box_readout_with_shuffled_control(tmp_path, fakes):
    G = 2
    d = dense_yaml("vggss_like", "box", protocol={"box_population": "single", "gt_geometry": "crop", "heatmap_grid": G, "annotated_second": 5.0},
                   requires={"pos": ["y", "x"], "streams": ["video", "audio"], "grid": {"h": G, "w": G, "min_cell_coverage": 0.5}, "t_bins": None},
                   readout_params={"grid": G, "visual_stream": "video", "conditioning_stream": "audio", "shuffled_control": True, "shuffle_seed": 7},
                   metrics=["iou_mean", "acc@0.5", "acc@0.3", "center_in_gt", "n_eval", "n_excluded_degenerate"], floors=["iou_floor"],
                   controls=["iou_shuffled", "audio_blind"], ci={}, chance={"iou_mean": 0.0},
                   label={"column": "box", "kind": "box", "sep": ";", "class_key": "name", "sets": None, "train_set": None, "allow_unknown": False, "empty_ok": False},
                   classes=None)
    rng = np.random.default_rng(3)

    def box(k):
        x1, y1 = 0.1 + 0.05 * (k % 5), 0.2
        return [x1, y1, x1 + 0.4, y1 + 0.3]

    rows = [{"clip_id": f"train_{k}", "split": "train", "label_ids": None, "box": box(k)} for k in range(10)] + \
           [{"clip_id": f"val_{k}", "split": "val", "label_ids": None, "box": box(k)} for k in range(5)] + \
           [{"clip_id": "val_degenerate", "split": "val", "label_ids": None, "box": [0.5, 0.5, 0.4, 0.6]}]
    bench = FakeBench(d, rows, None)
    run = tmp_path / "run"
    nv, na = G * G, 3
    yy, xx = np.meshgrid((np.arange(G) + 0.5) / G, (np.arange(G) + 0.5) / G, indexing="ij")
    pos = np.concatenate([
        np.stack([np.zeros(nv), -np.ones(nv), yy.ravel(), xx.ravel()], 1),
        np.stack([np.ones(na), (np.arange(na) + 0.5) / na, -np.ones(na), -np.ones(na)], 1),
    ]).astype(np.float32)
    for split, n in (("train", 10), ("val", 6)):
        write_fake_features(run, split, bench=bench, feats=rng.standard_normal((n, nv + na, D_FEAT)).astype(np.float32), pos=pos)
    cfg = ProbeConfig.pinned("box", grid=G).fast()
    rec = probe(run, bench=bench, cfg=cfg)[0].result
    check_result(rec, expect={"metrics": d["metrics"]["list"], "floors": ["iou_floor"], "controls": ["iou_shuffled", "audio_blind"], "ci": {}})
    assert rec["metrics"]["n_eval"] == 5 and rec["metrics"]["n_excluded_degenerate"] == 1
    assert 0.0 < rec["metrics"]["iou_mean"] <= 1.0 and rec["floors"]["iou_floor"] == pytest.approx(rec["metrics"]["iou_mean"])  # the fake head IS the mean box
    assert isinstance(rec["controls"]["audio_blind"], bool) and 0.0 <= rec["controls"]["iou_shuffled"] <= 1.0
    # two arms were trained: the main one and the deranged-conditioning one; the latter sees swapped audio tokens only
    assert len(fakes) == 2 and fakes[0]["targets"].shape == (10, 4) and fakes[1]["ckpt_path"].endswith("probe_ckpt.shuffled.pt")
    main, shuf = fakes[0]["features"], fakes[1]["features"]
    a, b = np.asarray(main[np.arange(10)]), np.asarray(shuf[np.arange(10)])
    assert np.array_equal(a[:, :nv], b[:, :nv]) and not np.array_equal(a[:, nv:], b[:, nv:])
    perm = shuf.perm
    assert (perm != np.arange(10)).all() and sorted(perm.tolist()) == list(range(10))
    assert np.array_equal(b[:, nv:], a[perm][:, nv:])
    out = Path(result_dir(run, bench.protocol_hash, hashing.probe_hash(cfg), 0))
    assert np.load(out / "eval_outputs_shuffled.npy").shape == (5, 4) and np.load(out / "eval_rows.npy").tolist() == [0, 1, 2, 3, 4]


# ============================================================================ helpers: deadline, derangement, scratch


def test_resolve_deadline():
    assert base.resolve_deadline(None, env={}) is None
    assert base.resolve_deadline(1234.5, env={}) == 1234.5
    assert base.resolve_deadline("1700000000", env={}) == 1700000000.0
    assert base.resolve_deadline(None, env={"YAVIH_DEADLINE": "99"}) == 99.0  # exec's epoch, taken as is
    assert base.resolve_deadline("slurm", env={"SLURM_JOB_END_TIME": "1000", "SLURM_JOB_ID": "1"}, margin=300) == 700.0
    assert base.resolve_deadline("slurm", env={}) is None  # outside SLURM: no deadline
    assert abs(base.resolve_deadline("+60", env={}) - (time.time() + 60)) < 5  # the shared parser (yavih.deadline)
    from yavih.deadline import DeadlineError

    with pytest.raises(DeadlineError):  # unset end time inside a job: a configuration error (exit 1), never "now"
        base.resolve_deadline("slurm", env={"SLURM_JOB_ID": "1"})


@pytest.mark.parametrize("n", [2, 3, 7, 50])
def test_derangement(n):
    perm = _fit.derangement(n, 20260913)
    assert sorted(perm.tolist()) == list(range(n)) and (perm != np.arange(n)).all()
    assert np.array_equal(perm, _fit.derangement(n, 20260913))


def test_sweep_scratch_removes_dead_or_old(tmp_path, monkeypatch):
    import socket

    host = socket.gethostname()
    alive = tmp_path / f"unit1.{host}.{os.getpid()}"
    dead = tmp_path / f"unit2.{host}.999999"
    foreign_new = tmp_path / "unit3.otherhost.1"
    foreign_old = tmp_path / "unit4.otherhost.2"
    for d in (alive, dead, foreign_new, foreign_old):
        d.mkdir()
    old = time.time() - base.STALE_SCRATCH_SECONDS - 10
    os.utime(foreign_old, (old, old))
    (tmp_path / "unrelated").mkdir()
    removed = base.sweep_scratch(tmp_path)
    assert {p.name for p in removed} == {dead.name, foreign_old.name}
    assert alive.exists() and foreign_new.exists() and (tmp_path / "unrelated").exists()


def test_stage_features_when_train_exceeds_ram_budget(tmp_path, fakes, monkeypatch):
    bench, run = simple_clf_world(tmp_path, train={"nbytes": 10**9}, val={"nbytes": 10**9})
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("YAVIH_RAM_BUDGET_GB", "0.0001")  # 100 KB budget → bytes_train (1 GB claimed) exceeds it
    monkeypatch.setenv("YAVIH_UNIT_ID", "unitxyz")
    rec = probe(run, bench=bench, cfg=clf_cfg(3), scratch=str(scratch))[0].result
    assert rec["status"] == "ok"
    staged = [c for c in FakeReader.calls if str(scratch) in c]
    assert staged and all(".unitxyz." not in c and "unitxyz." in c for c in staged)  # <scratch>/<unit_id>.<host>.<pid>/features/<split>
    assert not any(scratch.iterdir())  # removed on completion (hygiene)


def test_stage_features_preflights_free_space(tmp_path, monkeypatch):
    import shutil

    bench, run = simple_clf_world(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: NS(total=10, used=10, free=10))
    with pytest.raises(ProbeRefused) as e:
        base.stage_features(run / "features", ["train"], scratch_root=tmp_path / "scratch", unit_id="u", needs_bytes=10**6)
    assert e.value.code == "E_SCRATCH_SPACE" and e.value.exit_code == 3


def test_prestaged_copy_is_used_and_never_deleted(tmp_path, fakes):
    import shutil

    bench, run = simple_clf_world(tmp_path)
    staged = tmp_path / "staged_copy"
    shutil.copytree(run / "features", staged / "features")
    rec = probe(run, bench=bench, cfg=clf_cfg(3), scratch=str(staged))[0].result
    assert rec["status"] == "ok" and all(str(staged) in c for c in FakeReader.calls)
    assert (staged / "features" / "train" / "_meta.json").exists()


def test_feature_rows_gathers_across_segments(tmp_path, fakes):
    bench, run = simple_clf_world(tmp_path)
    r_train, r_val = FakeReader(run / "features" / "train"), FakeReader(run / "features" / "val")
    rows = _fit.FeatureRows([(r_train, np.array([0, 5, 11])), (r_val, np.array([1, 2]))])
    assert rows.shape == (5, N_TOK, D_FEAT) and len(rows) == 5
    got = rows[np.array([4, 0, 2])]
    assert np.array_equal(got[0], r_val.rows([2])[0]) and np.array_equal(got[1], r_train.rows([0])[0]) and np.array_equal(got[2], r_train.rows([11])[0])
    assert np.array_equal(rows[3], r_val.rows([1])[0]) and rows.backend == "ram"
    assert np.array_equal(rows[1:3], rows[np.array([1, 2])])  # slices stream like the trainer/floors do
    with pytest.raises(IndexError):
        rows[np.array([5])]


def test_evaluate_metrics_calls_every_registered_fn_and_nothing_else():
    specs = {
        "metrics": {"a": NS(name="a", fn=lambda o, t, p, m: float(np.mean(o == t)), shape="scalar", kind="metric"),
                    "n": NS(name="n", fn=lambda o, t, p, m: len(t), shape="scalar", kind="disclosure"),
                    "l": NS(name="l", fn=lambda o, t, p, m: [1, 2], shape="list", kind="metric")},
        "floors": {"a_floor": NS(name="a_floor", fn=lambda o, t, p, m: float(np.mean(o == t)), shape="scalar", kind="floor")},
        "controls": {"flag": NS(name="flag", fn=lambda o, t, p, m: np.bool_(True), shape="scalar", kind="control")},
    }
    o, t = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])
    out = base.evaluate_metrics(specs, outputs=o, targets=t, protocol={}, split_meta={}, floor_outputs=np.zeros(4, dtype=int), ci=None)
    assert out["metrics"] == {"a": 0.75, "n": 4, "l": [1.0, 2.0]} and out["floors"] == {"a_floor": 0.75} and out["controls"] == {"flag": True}
    assert out["ci95"] == {} and isinstance(out["metrics"]["n"], int) and isinstance(out["controls"]["flag"], bool)
