"""Resource formulas and the shard ranges the plan relies on."""

from __future__ import annotations

import math

import pytest

from yavih.plan import estimate as E


def test_bytes_per_row():
    assert E.bytes_per_row(1568, 1024, "float16", "constant") == 1568 * 1024 * 2
    assert E.bytes_per_row(16, 768, "float32", "none") == 16 * 768 * 4
    assert E.bytes_per_row(16, 768, "float16", "per_sample") == 16 * 768 * 2 + 16 * 16
    with pytest.raises(ValueError):
        E.bytes_per_row(16, 768, "bfloat16", "none")


def test_n_shards_for_time_and_bytes_cap():
    bpr = E.bytes_per_row(1568, 1024, "float16", "constant")
    # K400 reference: 20 000 rows, ~1.1 clips/s, 30 min → 20000 / 1980 ≈ 10.1 → 11 by time, but 64 GB / 2 GiB → 30 by bytes.
    n = E.n_shards_for(20000, clips_per_s=1.1, shard_minutes=30, bytes_per_row=bpr)
    assert n == max(math.ceil(20000 / (1.1 * 60 * 30)), math.ceil(20000 * bpr / E.SHARD_BYTES_CAP)) == 30
    assert E.n_shards_for(100, clips_per_s=10.0, shard_minutes=30, bytes_per_row=100) == 1
    assert E.n_shards_for(0, clips_per_s=10.0, shard_minutes=30, bytes_per_row=100) == 1
    assert E.n_shards_for(3600, clips_per_s=1.0, shard_minutes=30, bytes_per_row=1) == 2
    with pytest.raises(ValueError):
        E.n_shards_for(10, clips_per_s=0, shard_minutes=30, bytes_per_row=1)


def test_shard_ranges_tile_and_align():
    for n_rows, B, k in [(20, 4, 3), (20, 4, 10), (0, 4, 2), (33, 32, 1), (100, 7, 5)]:
        rs = E.shard_ranges(n_rows, B, k)
        assert len(rs) == k
        lo = 0
        for a, b in rs:
            assert a == lo and a <= b and a % B == 0
            lo = b
        assert lo == n_rows


def test_extract_needs_formula():
    needs = E.extract_needs(640, clips_per_s=1.1, cpus=8)
    assert needs == {"gpu": 1, "cpus": 8, "mem_gb": 48, "scratch_gb": 0, "minutes": math.ceil(1.5 * 640 / 1.1 / 60)}
    assert E.extract_needs(0, clips_per_s=1.0, cpus=2)["minutes"] == E.EXTRACT_MINUTES_FLOOR
    assert E.extract_needs(640, clips_per_s=1.1, cpus=8, multipliers={"extract_minutes": 2.0})["minutes"] == math.ceil(2 * 1.5 * 640 / 1.1 / 60)
    with pytest.raises(ValueError):
        E.extract_needs(1, clips_per_s=1.0, cpus=8, multipliers={"extract_minutes": -1})


def test_probe_needs_formula_and_floor():
    ram = 16 * 2**30
    small = E.probe_needs(bytes_train=10 * 2**20, N=1, D=768, probe_minutes_ref=20, seeds=1, ram_budget_bytes=ram)
    assert small == {"gpu": 1, "cpus": 8, "mem_gb": 16, "scratch_gb": 0, "minutes": 30}  # floor 30, pooled scale 1
    big_bytes = 64 * 2**30
    big = E.probe_needs(bytes_train=big_bytes, N=1568, D=1024, probe_minutes_ref=20, seeds=1, ram_budget_bytes=ram)
    assert big["mem_gb"] == math.ceil(1.3 * 64 + 8)
    assert big["scratch_gb"] == math.ceil(1.1 * 64)  # above the RAM budget → staged to scratch
    assert big["minutes"] == 20 * 24  # token scale 1568*1024/(16*768) = 130.7 → capped at 24 → 8 h, not weeks
    two = E.probe_needs(bytes_train=0, N=32, D=768, probe_minutes_ref=20, seeds=2, ram_budget_bytes=ram, cpus=4)
    assert two["minutes"] == 20 * 2 * 2 and two["cpus"] == 4
    mult = E.probe_needs(bytes_train=0, N=16, D=768, probe_minutes_ref=40, seeds=1, ram_budget_bytes=ram, multipliers={"probe_minutes": 1.5})
    assert mult["minutes"] == 60


def test_minutes_bucket():
    assert [E.minutes_bucket(m) for m in (1, 15, 16, 30, 31, 60, 100, 1440, 5000)] == [15, 15, 30, 30, 60, 60, 120, 1440, 5000]


def test_n_shards_for_rows_cap():
    # fast and small: time and bytes say 1 shard, the rows cap says ceil(25 000 / 10 000) = 3
    assert E.MAX_ROWS_PER_SHARD == 10_000
    assert E.n_shards_for(25_000, clips_per_s=1000.0, shard_minutes=30, bytes_per_row=10) == 3
    assert E.n_shards_for(25_000, clips_per_s=1000.0, shard_minutes=30, bytes_per_row=10, max_rows_per_shard=5_000) == 5
    assert E.n_shards_for(25_000, clips_per_s=1000.0, shard_minutes=30, bytes_per_row=10, max_rows_per_shard=None) == 1
    assert E.n_shards_for(10_000, clips_per_s=1000.0, shard_minutes=30, bytes_per_row=10) == 1  # exactly the cap
    # the cap never lowers what time or bytes demand
    assert E.n_shards_for(3600, clips_per_s=1.0, shard_minutes=30, bytes_per_row=1, max_rows_per_shard=10_000) == 2
    with pytest.raises(ValueError):
        E.n_shards_for(10, clips_per_s=1.0, shard_minutes=30, bytes_per_row=1, max_rows_per_shard=0)
