"""The decode-ahead pipeline of ``run_extract`` on a larger shard: ``Bridge.prepare`` runs in the ``yavih-prepare``
pool for every row and canary, ``encode`` only ever sees prepared samples, and the features are those of the
encode-only bridge. No wall-time assertion (machines differ); the sidecar's timing block is only checked for shape."""

from __future__ import annotations

import tarfile
from pathlib import Path

from test_extract_fixtures import FakeBenchmark, NumpyBridge, PrepareBridge
from yavih.extract import run_extract
from yavih.io.sidecar import read_sidecar

N_TRAIN = 512


class _Recording(PrepareBridge):
    name = "np-prepare-throughput"
    instances: list[_Recording] = []

    def __init__(self, bench, args=None):
        super().__init__(bench, args)
        self.encoded: list[tuple[str, bool]] = []
        type(self).instances.append(self)

    def encode(self, samples):
        self.encoded.extend((s.clip_id, s.prepared is not None) for s in samples)
        return super().encode(samples)


def _members(run: Path) -> dict[str, bytes]:
    with tarfile.open(run / "features" / "train" / "shard-000000.tar") as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers()}


def test_prepare_pipeline_over_a_large_shard(tmp_path: Path):
    _Recording.instances.clear()
    summary = run_extract(bench=FakeBenchmark(n={"train": N_TRAIN, "val": 8}), bridge=_Recording, split="train", shard=(0, 1),
                          out=tmp_path / "prep", canaries=8, decode_workers=4, read_ahead=3)
    (br,) = _Recording.instances
    assert summary["status"] == "ok" and summary["rows"]["n_written"] == N_TRAIN and summary["canary"]["n"] == 8
    assert len(br.encoded) == N_TRAIN + 8 and all(ok for _, ok in br.encoded) and br.unprepared == []
    assert [c for c, _ in br.encoded[:N_TRAIN]] == [f"train_{r:04d}" for r in range(N_TRAIN)]  # row order kept
    assert len(br.prepare_threads) == N_TRAIN + 8 and {t.rsplit("_", 1)[0] for t in br.prepare_threads} == {"yavih-prepare"}
    assert 1 < len(set(br.prepare_threads)) <= 4  # the decode_workers budget
    timing = read_sidecar(tmp_path / "prep" / "features" / "train" / "shard-000000.json")["timing"]
    assert {"read_s", "prepare_s", "encode_s", "wall_s", "clips_per_s"} <= set(timing) and timing["prepare_s"] > 0
    run_extract(bench=FakeBenchmark(n={"train": N_TRAIN, "val": 8}), bridge=NumpyBridge, split="train", shard=(0, 1),
                out=tmp_path / "plain", canaries=8, decode_workers=4, read_ahead=3)
    assert _members(tmp_path / "prep") == _members(tmp_path / "plain")
