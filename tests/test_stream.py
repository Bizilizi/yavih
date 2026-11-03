"""``yavih.stream``: the update grid, the causal and real-time modes, and the per-recording checks."""

from __future__ import annotations

import numpy as np
import pytest

from yavih import stream as S


class FakeClock:
    """A clock that advances only when the replay sleeps or a step runs (``cost`` seconds per step)."""

    def __init__(self, cost):
        self.t = 0.0
        self.cost = cost

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def step(self, cutoff, since):
        self.t += self.cost(cutoff) if callable(self.cost) else self.cost
        return np.full(3, cutoff, dtype=np.float32), np.array([max(0.0, cutoff - 2.0), cutoff, np.nan, np.nan])


def _replay(cost, duration=5.0, dt=0.5, **kw):
    fc = FakeClock(cost)
    return S.replay(fc.step, duration_s=duration, update_interval_s=dt, clock=fc.clock, sleep=fc.sleep, **kw)


def test_grid():
    assert S.grid(2.0, 0.5).tolist() == [0.5, 1.0, 1.5, 2.0]
    assert S.grid(1.99, 0.5).tolist() == [0.5, 1.0, 1.5]
    assert S.grid(0.3, 0.5).size == 0


def test_causal_answers_every_cutoff_and_passes_its_checks():
    fc = FakeClock(0.0)
    out = S.causal(fc.step, duration_s=3.0, update_interval_s=0.5)
    assert out["cutoff"].tolist() == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] and out["emb"].shape == (6, 3) and "timing" not in out
    assert S.check_row(out, duration_s=3.0, update_interval_s=0.5, mode="causal", windows={"video_window_s": 2.0}) == []


def test_a_fast_model_answers_every_request_in_real_time():
    out = _replay(0.1)
    t = out["timing"]
    assert out["cutoff"].tolist() == S.grid(5.0, 0.5).tolist() and t["skipped"] == []
    assert all(abs(s - r) < 1e-9 and abs(e - s - 0.1) < 1e-9 for r, s, e in t["updates"])
    assert S.check_row(out, duration_s=5.0, update_interval_s=0.5, mode="realtime") == []


def test_a_slow_model_skips_requests_and_is_never_ready_early():
    out = _replay(1.2)  # slower than the 0.5 s interval: the latest pending request wins
    t = out["timing"]
    assert t["skipped"] and len(t["updates"]) + len(t["skipped"]) == 10
    for r, s, e in t["updates"]:
        assert s >= r and e - s == pytest.approx(1.2)
    starts = [u[1] for u in t["updates"]]
    readies = [u[2] for u in t["updates"]]
    assert all(b >= a for a, b in zip(readies[:-1], starts[1:], strict=True))  # serial
    assert S.check_row(out, duration_s=5.0, update_interval_s=0.5, mode="realtime") == []


def test_a_finite_buffer_records_the_gap():
    out = _replay(2.5, buffer_s=1.0)
    assert out["timing"]["gaps"] and all(b - a > 0 for a, b in out["timing"]["gaps"])


def test_forced_causal_turns_replay_into_the_causal_mode():
    with S.forced_causal():
        out = _replay(9.0)
    assert "timing" not in out and out["cutoff"].size == 10


def test_check_row_finds_future_support_and_a_broken_log():
    fc = FakeClock(0.0)
    out = S.causal(fc.step, duration_s=2.0, update_interval_s=0.5)
    leak = dict(out, support=out["support"].copy())
    leak["support"][1, 1] = 1.7  # video support ends after the 1.0 s cutoff
    assert [c for c, _ in S.check_row(leak, duration_s=2.0, update_interval_s=0.5, mode="causal")] == ["E_FUTURE_SUPPORT"]
    short = {k: v[:-1] for k, v in out.items()}
    assert "E_MEMBERS" in [c for c, _ in S.check_row(short, duration_s=2.0, update_interval_s=0.5, mode="causal")]
    wide = S.check_row(out, duration_s=2.0, update_interval_s=0.5, mode="causal", windows={"video_window_s": 1.0})
    assert [c for c, _ in wide] == ["E_FUTURE_SUPPORT"]
    rt = _replay(0.1, duration=2.0)
    rt["timing"]["updates"][2][1] = 0.0  # started before its request
    assert "E_REPLAY_LOG" in [c for c, _ in S.check_row(rt, duration_s=2.0, update_interval_s=0.5, mode="realtime")]
    assert "E_REPLAY_LOG" in [c for c, _ in S.check_row(out, duration_s=2.0, update_interval_s=0.5, mode="realtime")]


def test_describe_stream_block():
    ok = {"mode": "causal", "native": False, "video_window_s": 4.0, "audio_window_s": None, "num_frames": 16,
          "frame_sampling": "uniform 4 fps", "startup_policy": "pad with the first frame", "gap_policy": "none", "device": "cpu"}
    assert S.check_describe_stream(ok) == []
    assert S.check_describe_stream({**ok, "mode": "fast"}) and S.check_describe_stream({k: v for k, v in ok.items() if k != "device"})


def _av_mp4(path, seconds=4.0, fps=10, sr=16000):
    """A small mp4: frame i is a flat gray level 30 + (7 * i) % 200 (h264), audio a 440 Hz sine (aac)."""
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as c:
        v = c.add_stream("libx264", rate=fps)
        v.width = v.height = 32
        v.pix_fmt = "yuv420p"
        v.options = {"qp": "0"}
        a = c.add_stream("aac", rate=sr)
        a.layout = "mono"
        for i in range(int(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 30 + (7 * i) % 200, dtype=np.uint8), format="rgb24")
            for p in v.encode(frame):
                c.mux(p)
        t = np.arange(int(seconds * sr)) / sr
        x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for k in range(0, len(x), 1024):
            af = av.AudioFrame.from_ndarray(x[None, k:k + 1024], format="fltp", layout="mono")
            af.sample_rate = sr
            af.pts = k
            for p in a.encode(af):
                c.mux(p)
        for s in (v, a):
            for p in s.encode(None):
                c.mux(p)
    return path.read_bytes()


def _decode(path):
    import av

    with av.open(str(path)) as c:
        frames = [(float(f.time), f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]
    with av.open(str(path)) as c:
        audio = np.concatenate([f.to_ndarray().reshape(-1) for f in c.decode(audio=0)])
    return frames, audio


def test_future_pair_blanks_only_the_future(tmp_path):
    from yavih.media.perturb import future_pair

    data = _av_mp4(tmp_path / "rec.mp4")
    paths = future_pair(data, tmp_path / "out", [1.0, 2.0], until_s=3.0)
    fa, aa = _decode(paths["A"])
    for c in (1.0, 2.0):
        fb, ab = _decode(paths[f"{c:.6f}"])
        assert [t for t, _ in fa] == [t for t, _ in fb] and max(t for t, _ in fa) < 3.0
        for (t, x), (_, y) in zip(fa, fb, strict=True):
            if t <= c:
                assert np.array_equal(x, y)
            else:
                assert not y.any() and x.any()
        n = int(c * 16000)
        assert np.array_equal(aa[: n - 2048], ab[: n - 2048]) and not ab[n + 2048:].any() and aa[n + 2048:].any()
