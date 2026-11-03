"""`yavih report paired`: per-clip paired differences between two systems' samples files."""

from __future__ import annotations

import json

import pytest

from yavih.report.paired import PairedError, paired, paired_text


def _write(path, values, seed=0, extra=()):
    with open(path, "w", encoding="utf-8") as f:
        for i, v in enumerate(values):
            f.write(json.dumps({"clip_id": f"c{i}", "row": i, "seed": seed, "role": "eval", "prediction": {"clap": v, "kl": 1.0}, "target": None}) + "\n")
        for line in extra:
            f.write(json.dumps(line) + "\n")
    return path


def test_paired_difference_and_interval(tmp_path):
    a = _write(tmp_path / "a.jsonl", [0.10 + 0.01 * (i % 7) for i in range(200)])
    b = _write(tmp_path / "b.jsonl", [0.20 + 0.01 * (i % 7) for i in range(200)])
    res = paired(a, b, samples=500, seed=0)
    c = res["metrics"]["clap"]
    assert res["n_common"] == 200 and c["diff"] == pytest.approx(0.10) and c["separated"]
    assert c["ci95"][0] == pytest.approx(0.10) and c["ci95"][1] == pytest.approx(0.10)  # a constant difference
    k = res["metrics"]["kl"]
    assert k["diff"] == 0.0 and not k["separated"]
    assert "| clap |" in paired_text(res)


def test_paired_uses_common_clips_only_and_refuses_bad_inputs(tmp_path):
    a = _write(tmp_path / "a.jsonl", [0.1, 0.2, 0.3])
    b = _write(tmp_path / "b.jsonl", [0.1, 0.2])
    res = paired(a, b, samples=100)
    assert (res["n_common"], res["only_first"], res["only_second"]) == (2, 1, 0)
    probe = tmp_path / "probe.jsonl"
    probe.write_text(json.dumps({"clip_id": "x", "row": 0, "seed": 0, "role": "eval", "prediction": 3, "target": 3}) + "\n")
    with pytest.raises(PairedError, match="no eval line"):
        paired(probe, b)
    two = _write(tmp_path / "two.jsonl", [0.1], extra=[{"clip_id": "c0", "row": 0, "seed": 1, "role": "eval", "prediction": {"clap": 0.2}}])
    with pytest.raises(PairedError, match="seeds"):
        paired(two, b)
    from yavih.report.paired import read_samples

    assert read_samples(two, seed=1) == {"c0": {"clap": 0.2}}
