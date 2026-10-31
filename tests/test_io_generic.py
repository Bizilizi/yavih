"""Format 5: a declared ``output_contract.schema`` written, validated and read per key (docs/features.md)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from yavih import codecs as C
from yavih import testing as T
from yavih.benchmark import Benchmark
from yavih.io import validate_features
from yavih.io.format import FeatureShapeError, FeatureValueError
from yavih.io.reader import FeatureReader
from yavih.io.writer import FeatureWriter

SCHEMA = {
    "audio": {"type": "wav", "channels": 1, "duration_s": [0.05, 0.2]},
    "score": {"type": "array", "dtype": "float32", "shape": ["K"], "stats": True},
    "text": {"type": "json", "kind": "str"},
}


class GenBridge:
    name = "gen-bridge"
    batch_size = 2
    streams: tuple[str, ...] = ()

    def __init__(self):
        self.args = {}


def _describe() -> dict:
    return {"model": "toy-gen", "reading": "v", "weights_sha256": None, "weights_ref": "-", "batch_size": 2, "views": 1,
            "input": {"video": {"temporal": {"rule": "custom", "description": "first second"}, "norm": {"source": "none"}}},
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "outputs": ["audio", "score", "text"], "sampling": {"seed": 0, "steps": 4, "guidance": 1.0, "n_candidates": 1, "selection": "first"},
            "args": {}}


@pytest.fixture
def bench(tmp_path):
    folder = T.make_synthetic_benchmark(tmp_path, "clf", n_train=8, n_eval=4, media="none")
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["output_contract"].update(schema=SCHEMA, sampling="required")
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    return Benchmark.from_dir(folder)


def _value(row: int, *, scale: float = 1.0) -> dict:
    rng = np.random.default_rng(row)
    return {"audio": C.wav_bytes(rng.standard_normal(1600).astype(np.float32) * 0.1 * scale, 16000),
            "score": rng.standard_normal(3).astype(np.float32), "text": f"clip {row}"}


def _meta(rows):
    return [{"media_key": f"mk{r}", "media": {"video": {"duration_s": 10.0}}, "used": {"video": {"times": [0.5]}}} for r in rows]


def _write(bench, run, rows, shard, n, *, canary_scale=1.0, canary=True):
    w = FeatureWriter(run, "train", shard=(shard, n), rows=(rows[0], rows[-1] + 1), bench=bench, bridge=GenBridge(),
                      describe=_describe(), source_sha256="s" * 64)
    vals = [_value(r) for r in rows]
    w.write_outputs(rows, [f"train_{r:04d}" for r in rows], {k: [v[k] for v in vals] for k in SCHEMA}, meta=_meta(rows))
    if canary:
        c = [_value(rows[0], scale=canary_scale)]
        w.write_outputs([rows[0]], [f"train_{rows[0]:04d}"], {k: [v[k] for v in c] for k in SCHEMA}, meta=_meta([rows[0]]), canary=True)
    w.close(timing={}, env={}, canary_stats={})
    return w


def test_write_validate_and_read_a_generic_set(bench, tmp_path):
    run = tmp_path / "run"
    _write(bench, run, [0, 1, 2, 3], 0, 2)
    _write(bench, run, [4, 5, 6, 7], 1, 2)
    sc = json.loads((run / "features" / "train" / "shard-000000.json").read_text())
    assert sc["format_version"] == 5 and sc["features"]["dtype"] is None
    assert sc["outputs"]["schema"] == SCHEMA and sc["outputs"]["bound"] == {"K": 3}
    assert sc["outputs"]["keys"]["audio"] == {"codec": "wav", "version": 1, "ext": "wav", "dtype": None}
    assert set(sc["members"][0]["parts"]) == {"audio", "score", "text"}

    rep = validate_features(run / "features" / "train", bench=bench)
    assert rep.ok, [f"{f.code}: {f.message}" for f in rep.errors]
    meta = json.loads((run / "features" / "train" / "_meta.json").read_text())
    assert meta["outputs"]["bound"] == {"K": 3} and meta["features"]["N"] is None and meta["coverage"]["present"] == 8
    assert meta["eligibility"]["eligible"] is True

    r = FeatureReader(run / "features" / "train")
    assert r.generic and r.keys() == ["audio", "score", "text"] and list(r.present_rows) == list(range(8))
    assert r.values("text", [5, 1]) == ["clip 5", "clip 1"]
    assert np.array_equal(r.values("score", [2])[0], _value(2)["score"])
    samples, sr = C.read_wav(r.values("audio", [3])[0])
    assert sr == 16000 and samples.shape == (1600, 1)
    with pytest.raises(Exception, match="no feature matrix"):
        r.rows([0])
    assert len(r.fingerprint()) == 64 and r.fingerprint() == meta["fingerprint"]


def test_a_batch_dependent_generator_fails_the_canary(bench, tmp_path):
    run = tmp_path / "run"
    _write(bench, run, [0, 1, 2, 3], 0, 2, canary_scale=1.5)
    _write(bench, run, [4, 5, 6, 7], 1, 2)
    rep = validate_features(run / "features" / "train", bench=bench, write_meta=False)
    assert "E_BATCH_INVARIANCE" in {f.code for f in rep.errors}


def test_writer_refuses_bad_values_before_writing(bench, tmp_path):
    w = FeatureWriter(tmp_path / "f", "train", shard=(0, 1), rows=(0, 8), bench=bench, bridge=GenBridge(), describe=_describe(),
                      source_sha256="s" * 64)
    v = _value(0)
    with pytest.raises(FeatureShapeError, match="duration"):
        w.write_outputs([0], ["train_0000"], {"audio": [C.wav_bytes(np.zeros(16000, np.float32), 16000)], "score": [v["score"]], "text": [v["text"]]}, meta=_meta([0]))
    with pytest.raises(FeatureValueError, match="E_FINITE"):
        w.write_outputs([0], ["train_0000"], {"audio": [v["audio"]], "score": [np.array([np.nan, 0, 0], np.float32)], "text": [v["text"]]}, meta=_meta([0]))
    with pytest.raises(Exception, match="lack the schema key"):
        w.write_outputs([0], ["train_0000"], {"audio": [v["audio"]], "score": [v["score"]]}, meta=_meta([0]))
    w.write_outputs([0], ["train_0000"], {k: [v[k]] for k in SCHEMA}, meta=_meta([0]))
    v1 = _value(1)
    with pytest.raises(FeatureShapeError, match="bound to 3"):
        w.write_outputs([1], ["train_0001"], {"audio": [v1["audio"]], "score": [np.zeros(4, np.float32)], "text": [v1["text"]]}, meta=_meta([1]))
    assert w.members and len(w.members) == 1
    w.abort()


def test_the_default_schema_still_writes_format_4_members(bench, tmp_path):
    folder = T.make_synthetic_benchmark(tmp_path / "d", "clf", n_train=4, n_eval=2, media="none")
    b = Benchmark.from_dir(folder)
    w = FeatureWriter(tmp_path / "d" / "f", "train", shard=(0, 1), rows=(0, 4), bench=b, bridge=type("B", (), {"dtype": "float16", "pos_mode": "none", "batch_size": 2, "args": {}})(),
                      describe={**_describe(), "dtype": "float16", "pos_mode": "none"}, source_sha256="s" * 64)
    assert not w.generic
    w.write([0, 1, 2, 3], [f"train_{r:04d}" for r in range(4)], np.ones((4, 2, 3), np.float32), meta=_meta(range(4)))
    side = json.loads(w.close(timing={}, env={}, canary_stats={}).read_text())
    assert side["outputs"]["schema"] is None and side["features"]["N"] == 2 and "parts" not in side["members"][0]
    import tarfile

    names = tarfile.open(tmp_path / "d" / "f" / "features" / "train" / "shard-000000.tar").getnames()
    assert names[:2] == ["000000000.json", "000000000.feat.npy"]


def test_parallel_validation_of_a_generic_set_matches_the_serial_one(bench, tmp_path, monkeypatch):
    run = tmp_path / "run"
    _write(bench, run, [0, 1, 2, 3], 0, 2)
    _write(bench, run, [4, 5, 6, 7], 1, 2)
    monkeypatch.setenv("YAVIH_VALIDATE_WORKERS", "1")
    serial = validate_features(run / "features" / "train", bench=bench, write_meta=False)
    monkeypatch.setenv("YAVIH_VALIDATE_WORKERS", "2")
    parallel = validate_features(run / "features" / "train", bench=bench, write_meta=False)
    assert serial.ok and parallel.ok
    assert serial.to_json()["coverage"] == parallel.to_json()["coverage"] and serial.bytes == parallel.bytes
    assert [f.code for f in serial.warnings] == [f.code for f in parallel.warnings]
