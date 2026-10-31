"""A generative bridge on a declared output schema: describe(), bridge check, extraction and validation (format 5)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import yaml

from yavih import codecs as C
from yavih import testing as T
from yavih.benchmark import Benchmark
from yavih.bridge import Bridge
from yavih.extract import ExtractError, bridge_check, run_extract
from yavih.io import validate_features
from yavih.io.reader import FeatureReader

SCHEMA = {"audio": {"type": "wav", "channels": 1, "sample_rate": 16000, "duration_s": [0.05, 0.2]}}
SAMPLING = {"seed": 0, "steps": 4, "guidance": 1.0, "n_candidates": 1, "selection": "first"}


class ToyGenerator(Bridge):
    """A 'video-to-audio generator': a seeded noise clip whose seed is the input video's bytes, so the output is a
    pure function of the input and the declared sampling seed."""

    batch_size = 2
    outputs = ("audio",)
    batch_dependent = False

    def describe(self) -> dict:
        return {"model": "toy-v2a", "reading": "v", "weights_sha256": None, "weights_ref": "-", "batch_size": self.batch_size, "views": 1,
                "input": {"video": {"temporal": {"rule": "custom", "description": "the stored bytes seed the generator"},
                                    "spatial": {"rule": "custom", "description": "no decoding"},
                                    "norm": {"source": "none"}, "decoder": {"lib": "none", "version": "0"}}},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
                "args": {}, "outputs": list(self.outputs), "sampling": dict(SAMPLING)}

    def encode(self, samples):
        out = []
        for s in samples:
            seed = int.from_bytes(hashlib.sha256(s.streams["video"].bytes()).digest()[:8], "big") ^ SAMPLING["seed"]
            x = np.random.default_rng(seed).standard_normal(1600).astype(np.float32) * 0.1
            if self.batch_dependent:
                x = x * (1.0 + 0.2 * len(samples))
            out.append(C.wav_bytes(x, 16000))
        return {"audio": out, "used": [{"video": {"times": []}} for _ in samples]}


class BatchDependent(ToyGenerator):
    batch_dependent = True


class NoSampling(ToyGenerator):
    def describe(self) -> dict:
        d = super().describe()
        d.pop("sampling")
        return d


class FeatureTuple(ToyGenerator):
    def encode(self, samples):
        return np.zeros((len(samples), 2, 2), np.float32), None, [{"video": {"times": []}} for _ in samples]


@pytest.fixture
def bench(tmp_path, yavih_home):
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=8, n_eval=4, media="tiny")
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["output_contract"].update(schema=SCHEMA, sampling="required")
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    return Benchmark.from_dir(folder)


def test_bridge_check_extract_validate_a_generator(bench, tmp_path):
    prof = bridge_check(bridge=ToyGenerator, bench=bench, n=6, out=tmp_path / "toy.profile.json")
    assert prof["status"] == "ok" and prof["N"] is None and prof["dtype"] is None
    inv = prof["invariance"]
    assert inv["rerun_bit_identical"] is True and inv["batch_rel_l2"] == 0.0 and inv["order_rel_l2"] == 0.0
    assert prof["outputs"]["keys"] == {"audio": {"codec": "wav", "version": 1}} and prof["outputs"]["bytes_per_row"] > 3000

    run = tmp_path / "run"
    for split in ("train", "val"):
        for i in range(2):
            res = run_extract(bench=bench, bridge=ToyGenerator, split=split, shard=(i, 2), out=run, canaries=1)
            assert res["status"] == "ok", res
        rep = validate_features(run / "features" / split, bench=bench)
        assert rep.ok, [f"{f.code}: {f.message}" for f in rep.errors]
        assert rep.eligibility["eligible"] is True
    r = FeatureReader(run / "features" / "val")
    samples, sr = C.read_wav(r.values("audio", [0])[0])
    assert sr == 16000 and samples.shape == (1600, 1)
    # resumption: a matching sidecar is skipped
    assert run_extract(bench=bench, bridge=ToyGenerator, split="val", shard=(0, 2), out=run, canaries=1)["status"] == "skipped"


def test_a_batch_dependent_generator_is_refused_by_bridge_check(bench):
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=BatchDependent, bench=bench, n=6)
    assert ei.value.code == "E_BATCH_INVARIANCE"


def test_sampling_is_required_and_the_tuple_contract_is_refused(bench):
    with pytest.raises(Exception, match="sampling"):
        bridge_check(bridge=NoSampling, bench=bench, n=4)
    with pytest.raises(ExtractError) as ei:
        bridge_check(bridge=FeatureTuple, bench=bench, n=4)
    assert ei.value.code == "E_FEATURES"


EVAL_PY = '''
"""A folder-owned direct scorer: the mean RMS of the generated audio (a stand-in for FAD / alignment metrics)."""
import numpy as np

from yavih.metrics.registry import MetricSpec
from yavih.readouts.registry import Scorer

NAME = "synth_clf"


def _by_score(*_a, **_k):
    raise RuntimeError("computed by SCORER.score")


METRICS = {"mean_rms": MetricSpec(name="mean_rms", fn=_by_score, readouts=frozenset({NAME}), higher_is_better=None)}


def _score(ctx):
    from yavih import codecs as C

    vals, rows = [], []
    for split in ctx.roles["eval"]:
        reader = ctx.reader(split)
        idx = reader.present_rows
        for row, data in zip(idx, reader.values("audio", idx, codecs=ctx.bench.codecs)):
            x, _sr = C.read_wav(data)
            vals.append(float(np.sqrt(np.mean(x ** 2))))
            rows.append(int(row))
    return {"metrics": {"mean_rms": float(np.mean(vals)), "n_eval": len(vals)}, "floors": {}, "controls": {}, "ci95": {},
            "probe": {}, "fit": {"scorer": NAME}, "timing": {}, "eval_rows": np.asarray(rows)}


SCORER = Scorer(name=NAME, version=1, module=__name__, kind="direct", label_kinds=frozenset({"single"}),
                requires_t_bins=False, requires_grid=False, score=_score, needs_gpu=False)
'''


@pytest.mark.parametrize("train_split", [True, False], ids=["with-train", "eval-only"])
def test_a_folder_scorer_scores_a_generated_set(tmp_path, yavih_home, train_split):
    from yavih.readouts import probe

    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=8, n_eval=4, media="tiny")
    (folder / "eval.py").write_text(EVAL_PY, encoding="utf-8")
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["output_contract"].update(schema=SCHEMA, sampling="required")
    y["scorer"] = "synth_clf"
    y["budget"]["probe_minutes_ref"] = {"synth_clf": 1}
    y["metrics"].update(headline="mean_rms", list=["mean_rms", "n_eval"], floors=[], controls=[], chance={}, ci={})
    y["metrics"].pop("table", None)
    if not train_split:  # a scorer that trains nothing needs no train split
        y["splits"].update(train=[], unused=["train"])
        y["budget"]["clips"].pop("train", None)
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    bench = Benchmark.from_dir(folder)
    from yavih import hashing
    from yavih.readouts import registry

    assert registry.get("synth_clf").kind == "direct"
    run = tmp_path / "run"
    for split in (("train", "val") if train_split else ("val",)):
        assert run_extract(bench=bench, bridge=ToyGenerator, split=split, shard=(0, 1), out=run, canaries=1)["status"] == "ok"
        assert validate_features(run / "features" / split, bench=bench).ok
    res = probe(run, seeds=(0,), write_results=False, bench=str(folder))
    assert len(res) == 1 and res[0].status == "ok", res[0].reason
    rec = res[0].result
    assert rec["standard"] is True and set(rec["metrics"]) == {"mean_rms", "n_eval"} and rec["metrics"]["n_eval"] == 4
    assert 0.05 < rec["metrics"]["mean_rms"] < 0.2
    assert rec["probe"]["hash"] == hashing.scorer_config_hash("synth_clf", 1)
    assert rec["probe"]["config"] == {"scorer": "synth_clf", "version": 1, "kind": "direct"}
    assert rec["eval_hash"] == hashing.eval_hash(bench.protocol_hash, rec["probe"]["hash"])
