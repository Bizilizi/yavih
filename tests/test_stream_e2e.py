"""A streaming benchmark end to end on synthetic recordings: bridge check (determinism under forced causal, the
future-invariance gate, the real-time smoke), extract, validate and the ``stream`` scorer in both modes."""

from __future__ import annotations

import pytest

pytest.importorskip("av")
pytest.importorskip("torch")

from yavih.benchmark import Benchmark  # noqa: E402
from yavih.extract import ExtractError, bridge_check, run_extract  # noqa: E402
from yavih.io import validate_features  # noqa: E402
from yavih.probes.config import ProbeConfig  # noqa: E402
from yavih.readouts import probe  # noqa: E402
from yavih.readouts import stream as RS  # noqa: E402
from yavih import testing as T  # noqa: E402
from yavih.testing import stream as TS  # noqa: E402


class SlowRealtimeBridge(TS.RealtimeStreamBridge):
    """Slower than the update interval: requests are skipped and predictions arrive late."""

    cost_s = 0.7
    virtual_clock = True


@pytest.fixture
def bench(yavih_home, monkeypatch):
    monkeypatch.setenv("YAVIH_TESTING", "1")
    folder = TS.make_stream_benchmark(yavih_home / "benchmarks", n_train=24, n_eval=10)
    return Benchmark.from_dir(folder)


def _score(bench, bridge, run):
    for split in ("train", "test"):
        assert run_extract(bench=bench, bridge=bridge, split=split, shard=(0, 1), out=run, canaries=1)["status"] == "ok"
        assert validate_features(run / "features" / split, bench=bench).ok
    (res,) = probe(run, seeds=(0,), cfg=T.fast(ProbeConfig.pinned("stream", bench)), write_results=False)
    assert res.status == "ok", res.reason
    return res.result


def test_bridge_check_gates(bench):
    prof = bridge_check(bridge=TS.OracleStreamBridge, bench=bench, n=4)
    fi = prof["stream"]["future_invariance"]
    assert prof["invariance"]["rerun_bit_identical"] and len(fi["recordings"]) == 2 and fi["max_rel_l2"] == 0.0
    with pytest.raises(ExtractError) as e:
        bridge_check(bridge=TS.LeakyStreamBridge, bench=bench, n=4)
    assert e.value.code == "E_FUTURE_LEAK"
    rt = bridge_check(bridge=TS.RealtimeStreamBridge, bench=bench, n=4)
    assert rt["stream"]["realtime_smoke"]["updates"] > 0 and rt["stream"]["realtime_smoke"]["max_rel_l2_vs_causal"] == 0.0


def test_oracle_random_and_late_predictions(bench, tmp_path):
    oracle = _score(bench, TS.OracleStreamBridge, tmp_path / "oracle")
    random = _score(bench, TS.RandomStreamBridge, tmp_path / "random")
    slow = _score(bench, SlowRealtimeBridge, tmp_path / "slow")
    band = RS.CONFORMANCE
    for head in ("linear", "gru"):
        assert oracle["metrics"][f"bin_acc_{head}"] >= band["oracle_min"]
        assert random["metrics"][f"bin_acc_{head}"] <= band["random_max"]
        assert slow["metrics"][f"bin_acc_{head}"] < oracle["metrics"][f"bin_acc_{head}"]  # staleness costs accuracy
    assert oracle["floors"]["bin_acc_floor"] < band["random_max"]
    lo, hi = oracle["ci95"]["bin_acc_gru"]
    assert lo <= oracle["metrics"]["bin_acc_gru"] <= hi
