"""Folder tests for avsync (the yaml, the manifest, make_manifest.py, the materialization recipe).

Run only this file: ``python -m pytest "$(yavih bench path avsync)/tests" -q -p no:cacheprovider``.

- yaml / manifest / classes / curation / docs tests read the folder files directly and run anywhere;
- the make_manifest tests re-derive the committed rows from their clip selection and run the scoring and selection
  code on synthetic media (scoring needs torch, PyAV, soundfile; skips otherwise);
- the materialization test runs the pinned recipe of ``prepare/03_materialize.py`` on a synthetic source clip
  (needs ffmpeg with libx264, PyAV, soundfile; skips otherwise);
- the reader smoke and the output-contract checks run on the core (they need ffmpeg too);
- the site tests need the configured VGGSound tree, score directory or ``derived_root``; they skip otherwise.
Every Benchmark here is built with ``YAVIH_CONFIG`` pointed at a nonexistent file and ``XDG_CONFIG_HOME`` at an
empty dir (no shared site config); the site-data tests resolve their roots through ``yavih.config``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_rows, mm = F.spec, F.rows, F.mm

OFFSETS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
ROWS = {"train": 24000, "test": 3000, "stationary": 5997}
NO_CONFIG = str(HERE / "tests" / "_no_such_config.toml")
REMOVED_PROTOCOL_KEYS = {"windows", "window_shift", "frame_rule", "frame_rule_params", "frame_grid", "frame_select", "pad_policy", "short_clip_policy", "crop", "audio_norm_context"}
PX = 0.8  # synthetic video: pixel value per source frame index (306 frames at 30 fps stay < 256)


def _media_root() -> Path | None:
    return F.site_root(HERE, "media_root", needs=("video", "audio"))


def _ffmpeg_ok() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libx264" in out and " alac " in out


@pytest.fixture(scope="module")
def mat_mod():
    return F.load_module(HERE / "prepare" / "03_materialize.py", "avsync_materialize")


# ----------------------------------------------------------------------------- yaml (schema 2) and manifest


def test_yaml_schema2_and_changelog(spec):
    assert spec["schema_version"] == 2 and spec["name"] == HERE.name == "avsync" and spec["readout"] == "clf"
    assert spec["protocol_version"] == 1 == spec["changelog"][-1]["version"]
    assert [c["version"] for c in spec["changelog"]] == [1]
    assert spec["changelog"][-1]["supersedes"] == []
    # the reader does not cut windows: no window / frame-rule key in the protocol
    assert "requires" not in spec and not (REMOVED_PROTOCOL_KEYS & set(spec["protocol"]))
    for s in spec["streams"].values():
        assert "bridge_decode_allowed" not in s
    assert all(c.get("role") != "window" for c in spec["manifest"]["columns"].values())
    assert spec["external"] == []


def test_streams_and_input_contract(spec):
    st, ic = spec["streams"], spec["input_contract"]
    assert set(st) == {"video", "audio"} == set(ic["handed_over"]) == set(ic["info"])
    for name, s in st.items():
        assert s == {"kind": name, "media_column": "rel_media", "root": "derived_root", "layout": "dir", "layout_params": {}, "required": True}
    assert ic["info"]["video"] == ["duration_s", "fps", "width", "height", "container", "bytes"]
    assert ic["info"]["audio"] == ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]
    assert ic["facts"] == []  # the offset is the label: never a fact
    cols = spec["manifest"]["columns"]
    assert cols["rel_media"]["role"] == "media" and cols["rel_media"]["stream"] == ["video", "audio"]
    m = ic["materialize"]
    assert m["kind"] == "av_offset_mux" and m["offset_column"] == "offset_seconds"
    assert m["source_columns"] == {"video": "rel_video", "audio": "rel_audio"}
    assert all(cols[c]["role"] == "extra" for c in (m["offset_column"], *m["source_columns"].values()))
    assert (m["pivot"], m["length"], m["video"], m["audio"]) == (1.0, 8.0, 0.5, -0.5)
    assert m["pivot"] == spec["protocol"]["nominal_window_start"]
    assert m["tool"]["name"] == "ffmpeg" and m["tool"]["version_pin"].startswith("4.4.2")
    assert "outputs" not in m  # the recipe is the pin: re-encoded bytes differ between hosts, no output sha256 ships
    assert {"media_root", "derived_root", "curation_root"} == set(spec["data"]["roots"])
    assert spec["protocol"]["clip_seconds"] == m["length"]


def test_output_contract(spec):
    assert spec["output_contract"] == {
        "shape": "fixed",
        "coordinates": "media",
        "target_frame": None,
        "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None},
    }


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_sha256_match(spec, manifest_rows):
    F.assert_manifest_hashes(HERE, spec)
    assert len(manifest_rows) == sum(ROWS.values()) == spec["manifest"]["upstream"]["rows"]
    declared = {"clip_id", "label", "split", *spec["manifest"]["columns"]}
    assert set(manifest_rows[0]) == declared


def test_splits_assigned_once_and_roles(spec, manifest_rows):
    roles = spec["splits"]
    assert F.assert_splits_assigned_once(spec, manifest_rows) == ROWS == spec["budget"]["clips"]
    assert roles["train"] == ["train"] and roles["eval"] == ["test"] and roles["floor"] == ["stationary"]


def test_classes_contiguous_offset_grid(spec):
    rows = list(csv.DictReader((HERE / "classes.csv").open(newline="", encoding="utf-8")))
    assert list(rows[0].keys()) == ["index", "name"]
    assert [int(r["index"]) for r in rows] == list(range(9))
    assert [float(r["name"]) for r in rows] == OFFSETS == spec["protocol"]["offsets_seconds"]


def test_rows_are_clip_offset_pairs(manifest_rows, mm):
    for r in manifest_rows:
        clip, _, b = r["clip_id"].partition("#")
        assert b == r["label"] == r["offset_bin"], r["clip_id"]
        assert float(r["offset_seconds"]) == OFFSETS[int(b)]
        assert r["rel_video"] == f"video/{clip}.mp4" and r["rel_audio"] == f"audio/{clip}.wav"
        assert r["arm"] == ("stationary" if r["split"] == "stationary" else "transient")
        assert r["rel_media"] == mm.materialized_relpath(r["clip_id"])
    clips = defaultdict(set)
    for r in manifest_rows:
        clips[r["split"]].add(r["clip_id"].split("#", 1)[0])
    assert {s: len(c) for s, c in clips.items()} == {"train": 12000, "test": 3000, "stationary": 5997}
    assert not (clips["train"] & clips["test"]) and not (clips["train"] & clips["stationary"]) and not (clips["test"] & clips["stationary"])


def test_materialized_names_carry_no_label(manifest_rows):
    """The reader hands the bridge the file path, which bridge check cannot scramble: it must not hold the label
    (``#<bin>``), the clip stem or the split."""
    names = [r["rel_media"] for r in manifest_rows]
    assert len(set(names)) == len(names)
    pat = re.compile(r"^([0-9a-f]{2})/\1[0-9a-f]{22}\.mp4$")
    for r in manifest_rows:
        assert pat.match(r["rel_media"]), r["rel_media"]
        assert "#" not in r["rel_media"] and r["split"] not in r["rel_media"].split("/")[0]
    assert len({n.split("/")[0] for n in names}) == 256


def test_train_draws_distinct_and_bins_balanced(manifest_rows):
    per_clip = defaultdict(list)
    for r in manifest_rows:
        if r["split"] == "train":
            per_clip[r["clip_id"].split("#", 1)[0]].append(int(r["offset_bin"]))
    assert all(len(v) == 2 and v[0] != v[1] for v in per_clip.values())
    for split, n in ROWS.items():
        counts = Counter(int(r["offset_bin"]) for r in manifest_rows if r["split"] == split)
        assert sorted(counts) == list(range(9))
        assert max(counts.values()) - min(counts.values()) <= 1 and sum(counts.values()) == n
        surplus = n % 9  # surplus rows take bins 0..surplus-1 (clip-id order, never the RNG)
        assert [b for b in range(9) if counts[b] == n // 9 + 1] == list(range(surplus))


def test_curation_sums(spec, manifest_rows):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"]
    for split, v in cur["per_split"].items():
        assert v["listed"] == sum(r["split"] == split for r in manifest_rows)
        assert v["staged"] + v["missing"] == v["listed"]
    assert cur["total_upstream"] == 199176 + 12000  # 199 176 VGGSound clips + the 12 000 second train draws
    assert cur["leakage_audit"]["excluded"] == 0 and cur["short_clips"] == 0


def test_protocol_constants(spec, mm, mat_mod):
    p, m = spec["protocol"], spec["input_contract"]["materialize"]
    assert p["views"] == "rows" and p["floor_sigmas"] == 3.0 and p["coarse_grid_seconds"] == 1.0
    assert p["curation"] == mm.PROTOCOL_CURATION
    assert p["train_draw"]["seed"] == p["curation"]["offset_seed"] == 0
    # at δ = ±2 both 8 s source windows stay inside the 10 s clip (split shift, never a one-sided shift)
    for d in OFFSETS:
        w = mat_mod.windows(m, d)
        assert w["v0"] >= 0 and w["v1"] <= p["curation"]["required_seconds"]
        assert w["a0"] >= 0 and w["a1"] <= p["curation"]["required_seconds"] * 16000 and w["a1"] - w["a0"] == 128000
        assert math.isclose(w["v0"] - w["a0"] / 16000, d)  # video leads audio by δ
        assert mm.window_starts(d) == (w["a0"] / 16000, w["v0"])


def test_metric_names_resolve(spec):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert m["headline"] == "acc" and m["controls"] == ["acc_floor_arm", "floor_violated"]
    assert m["ci"] == {"acc": "binomial"}
    assert math.isclose(m["chance"]["acc"], 1 / 9, abs_tol=1e-4)
    assert math.isclose(m["chance"]["acc_tol1"], 25 / 81, abs_tol=1e-4)
    assert math.isclose(m["chance"]["mae_seconds"], 0.5 * 80 / 27, abs_tol=1e-4)


def test_floor_threshold_matches_readme(spec):
    floor_threshold = F.load_module(HERE / "probe.py", "avsync_probe").floor_threshold  # the folder's metrics

    sig = spec["protocol"]["floor_sigmas"]
    expected = 1 / 9 + sig * math.sqrt((1 / 9) * (8 / 9) / ROWS["stationary"])
    assert round(expected, 4) == 0.1233 and "0.1233" in (HERE / "README.md").read_text(encoding="utf-8")
    assert math.isclose(float(floor_threshold(1 / 9, ROWS["stationary"], sig)), expected, rel_tol=1e-9)


def test_benchmark_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from yavih import Benchmark

    monkeypatch.setenv("YAVIH_CONFIG", NO_CONFIG)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    b = Benchmark.from_dir(HERE)
    assert b.name == "avsync" and len(b.protocol_hash) == 64 and len(b.manifest_hash) == 64
    assert {s: b.n_rows(s) for s in ROWS} == ROWS
    v = b.view()
    assert v.facts == {} and v.streams == {"video": "video", "audio": "audio"}


# ----------------------------------------------------------------------------- documentation


def test_setup_dependencies_pin_ffmpeg(spec):
    """Folder fact in setup.md (the setup.md structure and contract copies are ``yavih bench lint``'s)."""
    deps = F.section((HERE / "setup.md").read_text(encoding="utf-8"), 2, "Dependencies")
    assert spec["input_contract"]["materialize"]["tool"]["version_pin"] in deps


def test_recipe_sha256_is_documented(spec, mat_mod):
    """The reproducibility pin is the recipe, not the output bytes: setup.md and the yaml comment carry its sha256."""
    rsha = mat_mod.recipe_sha256(spec["input_contract"]["materialize"])
    assert rsha in (HERE / "setup.md").read_text(encoding="utf-8")
    assert f"sha256 {rsha[:8]}…" in (HERE / "benchmark.yaml").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_imports_and_declares_rules():
    mod = F.load_module(HERE / "bridge_template.py", "avsync_bridge_template")
    B = mod.AVSyncBridge
    assert B.streams == ("video", "audio") and B.pos_mode == "per_sample"
    assert mod.VIDEO_TEMPORAL == {"rule": "uniform", "n": mod.N_FRAMES, "span": [0.0, 8.0], "edge": "error"}
    assert mod.N_FRAMES <= 24 and mod.N_FRAMES % mod.TUBELET == 0
    assert mod.AUDIO_TEMPORAL == {"rule": "span", "start_s": 0.0, "length_s": 8.0, "pad": "none"}
    src = (HERE / "bridge_template.py").read_text(encoding="utf-8")
    for key in ("weights_sha256", "weights_ref", "dtype", "batch_size", "streams", "pos_mode", "reading", "model", "layer", "pooling", "views", "input", "precision", "args"):
        assert f'"{key}"' in src, key
    assert '"decode"' not in src and "decode_plan" not in src  # keys the bridge API refuses
    assert "Preprocessing source".upper() in src and "TODO" in src
    # the nominal rule times are the 2 fps bucket centres, identical for every row (setup.md Gotcha 1)
    t = mod._rule_times(mod.VIDEO_TEMPORAL, {"duration_s": 7.9, "fps": 30.0, "n_frames": 237})
    assert t.tolist() == [(k + 0.5) * 0.5 for k in range(16)]


def test_bridge_template_describe_validates():
    from yavih.bridge import validate_input

    mod = F.load_module(HERE / "bridge_template.py", "avsync_bridge_template_d")
    b = mod.AVSyncBridge.__new__(mod.AVSyncBridge)
    b.args = {"reading": "av", "layer": -1}
    b.reading, b.streams, b._weights_sha256 = "av", ("video", "audio"), None
    d = b.describe()
    assert d["views"] == 1 and set(d["input"]) == {"video", "audio"}
    validate_input(d["input"], {"video": "video", "audio": "audio"}, ["video", "audio"])


# ----------------------------------------------------------------------------- make_manifest.py


def test_draw_semantics(mm):
    clips = [f"c{i:03d}" for i in range(23)]
    out = mm.assign_offset_bins(clips, 2, 0)
    assert len(out) == 46 and [c for c, _, _ in out] == [c for c in sorted(clips) for _ in range(2)]
    counts = Counter(b for _, _, b in out)
    assert max(counts.values()) - min(counts.values()) <= 1
    by_clip = defaultdict(list)
    for c, _, b in out:
        by_clip[c].append(b)
    assert all(v[0] != v[1] for v in by_clip.values())
    assert mm.assign_offset_bins(clips, 2, 0) == out
    assert mm.required_seconds() == 10.0 == mm.required_seconds(8, 1.0)
    assert mm.window_starts(2.0) == (0.0, 2.0) and mm.window_starts(-2.0) == (2.0, 0.0)
    assert mm._quantile([3.0, 1.0, 2.0, 4.0], 0.25) == 1.75
    assert mm.materialized_relpath("abc_000001#4") == mm.materialized_relpath("abc_000001#4") != mm.materialized_relpath("abc_000001#5")
    assert mm.bucket_indices(30.0, 1.0) == [round((1.0 + (k + 0.5) * 0.5) * 30) for k in range(16)]


def test_committed_selection_rederives_bytes(spec, mm):
    """Without scores, the rows re-derive byte-identically from the committed clip selection (offset draw,
    columns, order, rel_media), and benchmark.yaml carries the derived values."""
    result = mm.freeze(None, None, 0)
    rep = result["report"]
    assert rep["source"] == "committed selection" and rep["selection"] is None
    assert mm.gzip_bytes(result["manifest_bytes"]) == (HERE / "manifest.csv.gz").read_bytes()
    assert result["classes_bytes"] == (HERE / "classes.csv").read_bytes()
    assert rep["upstream"]["sha256"] == spec["manifest"]["upstream"]["sha256"] == mm.VGGSOUND_CSV_SHA
    assert rep["protocol_curation"] == spec["protocol"]["curation"]
    assert mm.yaml_drift(HERE, rep) == []
    assert rep["n_coarse5_eval"] == 1667 and rep["floor_threshold_3sigma"] == 0.1233


def _synthetic_scores(n_train: int = 60, n_test: int = 40) -> tuple[dict, dict, dict]:
    """Audio / video score records and splits of a small synthetic corpus (deterministic)."""
    import numpy as np

    rng = np.random.default_rng(1)
    audio, video, splits = {}, {}, {}
    for i in range(n_train + n_test):
        c = f"clip{i:04d}_{i:06d}"
        splits[c] = "train" if i < n_train else "test"
        audio[c] = {"clip": c, "duration_seconds": 9.5 if i % 17 == 0 else 10.0, "flux_peakiness": round(float(rng.uniform(1, 8)), 6),
                    "onset_rate": round(float(rng.uniform(0, 8)), 6), "env_stationarity": 0.5, "flux_bins": [1.0] * 40}
        video[c] = {"clip": c, "video_frames": 300, "video_fps": 30.0, "grid_ok": i % 23 != 5,
                    "motion": round(float(rng.uniform(0, 0.4)), 6), "av_corr": round(float(rng.uniform(-1, 1)), 6)}
    return audio, video, splits


def test_selection_gates_on_synthetic_scores(mm):
    """The gates and rankings of the freeze, on a synthetic corpus: duration, onset rate, peakiness quantile,
    stationary pool from the test split's bottom peakiness, grid_ok, motion quantile, av_corr ranks; every clip
    has one exclusion reason or is selected."""
    audio, video, splits = _synthetic_scores()
    cur = {**mm.PROTOCOL_CURATION, "stationary_pool_size": 8}
    cand = mm.select_candidates(audio, splits, cur)
    usable = {c for c, r in audio.items() if r["duration_seconds"] >= 10.0}
    assert set(cand["transient"]) <= usable and cand["too_short"] == len(audio) - len(usable)
    assert cand["peakiness_cut"] == mm._quantile([audio[c]["flux_peakiness"] for c in usable], 0.6)
    assert all(audio[c]["onset_rate"] >= 0.5 and audio[c]["flux_peakiness"] >= cand["peakiness_cut"] for c in cand["transient"])
    assert len(cand["stationary_pool"]) == 8 and all(splits[c] == "test" for c in cand["stationary_pool"])
    ev, tr, st, acc = mm.select(audio, video, splits, curation=cur, sizes=(3, 5), pinned=False)
    assert len(ev) <= 3 and len(tr) <= 5 and not set(ev) & set(st)
    assert all(splits[c] == "test" for c in ev + st) and all(splits[c] == "train" for c in tr)
    assert all(video[c]["grid_ok"] and video[c]["motion"] >= acc["motion_cut"] for c in ev + tr)
    assert [video[c]["av_corr"] for c in ev] == sorted((video[c]["av_corr"] for c in ev), reverse=True)
    assert [video[c]["av_corr"] for c in st] == sorted(video[c]["av_corr"] for c in st)
    assert len(ev) + len(tr) + len(st) + sum(acc["excluded"].values()) == len(audio)
    with pytest.raises(SystemExit, match="peakiness_cut"):
        mm.select(audio, video, splits, curation=cur, sizes=(3, 5))  # the recorded cuts are pinned


def _write_click_wav(path: Path, seconds: float = 10.2) -> None:
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, int(seconds * 16000))
    for t in np.arange(0.3, seconds - 0.1, 0.5):  # a click every 0.5 s
        k = int(t * 16000)
        x[k : k + 160] += rng.normal(0, 0.5, 160)
    sf.write(path, (np.clip(x, -1, 1) * 32767).astype(np.int16), 16000, subtype="PCM_16")


def test_audio_scores_on_synthetic_clicks(tmp_path: Path, mm):
    pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    _write_click_wav(tmp_path / "a.wav")
    r = mm.audio_scores(str(tmp_path / "a.wav"))
    assert r["duration_seconds"] == 10.2 and len(r["flux_bins"]) == int(10.2 / 0.25)
    assert r["onset_rate"] >= 1.5 and r["flux_peakiness"] > 3.0  # 2 clicks per second, sharp onsets
    assert mm.audio_scores(str(tmp_path / "a.wav")) == r


def test_log_mel_matches_kaldi_fbank(tmp_path: Path, mm):
    """The ported filterbank equals torchaudio's Kaldi-compatible fbank with the same options."""
    torch = pytest.importorskip("torch")
    kaldi = pytest.importorskip("torchaudio.compliance.kaldi")
    wav = torch.randn(16000) * 0.1
    ref = kaldi.fbank(wav.unsqueeze(0), sample_frequency=16000.0, num_mel_bins=128, frame_length=25.0, frame_shift=10.0,
                      window_type="hamming", use_energy=False, htk_compat=True, dither=0.0)
    assert torch.equal(mm.log_mel(wav), ref)


def test_video_scores_on_synthetic_clip(tmp_path: Path, mm):
    """Frame indexing (the pixel value encodes the source frame index), fps, grid_ok and the motion statistic."""
    pytest.importorskip("torch")
    av = pytest.importorskip("av")
    import numpy as np

    clip = "synthclip_000004"
    _write_source_clip(tmp_path, clip)
    with av.open(str(tmp_path / "video" / f"{clip}.mp4")) as f:
        st = f.streams.video[0]
        pts, keys, fps = mm._scan(f, st)
        idx = mm.bucket_indices(fps, 1.0)
        got = mm.grab_frames(f, st, pts, keys, idx)
    assert len(pts) == 306 and abs(fps - 30.0) < 1e-9 and keys
    assert all(abs(i - (1.0 + (k + 0.5) * 0.5) * 30) <= 0.5 for k, i in enumerate(idx))
    assert np.allclose([got[i].astype(np.float64).mean() / PX for i in idx], idx, atol=2.0)
    r = mm.video_scores(str(tmp_path / "video" / f"{clip}.mp4"), [1.0] * 40)
    assert r["video_frames"] == 306 and r["video_fps"] == 30.0 and r["grid_ok"] is True
    assert r["motion"] > 0 and r["av_corr"] == 0.0  # constant flux: no correlation
    _write_source_clip(tmp_path / "short", clip, seconds_video=9.5)
    assert mm.video_scores(str(tmp_path / "short" / "video" / f"{clip}.mp4"), [1.0] * 40)["grid_ok"] is False


def test_score_cli_writes_resumable_shards(tmp_path: Path, mm):
    """``make_manifest.py score --stage audio`` on a two-clip tree: one record per clip, a re-run scores nothing."""
    pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    (tmp_path / "metadata").mkdir()
    (tmp_path / "audio").mkdir()
    (tmp_path / "metadata" / "vggsound.csv").write_text("aaa,1,dog,train\nbbb,2,cat,test\n", encoding="utf-8")
    _write_click_wav(tmp_path / "audio" / "aaa_000001.wav")
    argv = ["score", "--stage", "audio", "--workers", "1", "--vggsound-root", str(tmp_path), "--work", str(tmp_path / "w")]
    assert mm.main(argv) == 0 and mm.main(argv) == 0
    recs = mm.read_scores(tmp_path / "w", "audio")
    assert sorted(recs) == ["aaa_000001", "bbb_000002"] and "error" in recs["bbb_000002"] and "flux_bins" in recs["aaa_000001"]
    lines = (tmp_path / "w" / "audio" / "shard-00000-of-00001.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def _site_scores() -> Path | None:
    from yavih import config

    try:
        work = config.folder_root(HERE, "curation_root") / "scores"
    except config.ConfigError:
        return None
    return work if (work / "video").is_dir() else None


@pytest.mark.skipif(_site_scores() is None or _media_root() is None, reason="no score directory (make_manifest.py score) or VGGSound tree at this site")
def test_rebuild_from_site_scores(spec, mm):
    """The freeze from this site's scores and the official listing: the committed files and yaml values."""
    result = mm.freeze(_media_root(), _site_scores(), 0)
    assert result["report"]["source"] == "scores"
    assert mm.gzip_bytes(result["manifest_bytes"]) == (HERE / "manifest.csv.gz").read_bytes()
    assert mm.yaml_drift(HERE, result["report"]) == []
    assert result["report"]["frame_grid_audit"]["uniform_n"]["16"]["rows_past_last_frame"] == 0


# ----------------------------------------------------------------------------- materialization


def _write_source_clip(root: Path, clip: str, seconds_audio: float = 10.2, seconds_video: float = 10.2) -> None:
    """16 kHz PCM_16 noise wav + a 30 fps 32×32 H.264 mp4 whose pixel value encodes the frame index (``i · PX``)."""
    import av
    import numpy as np
    import soundfile as sf

    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "video").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pcm = (rng.uniform(-0.5, 0.5, int(round(seconds_audio * 16000))) * 32767).astype(np.int16)
    sf.write(root / "audio" / f"{clip}.wav", pcm, 16000, subtype="PCM_16")
    with av.open(str(root / "video" / f"{clip}.mp4"), "w") as out:
        st = out.add_stream("libx264", rate=30)
        st.width, st.height, st.pix_fmt = 32, 32, "yuv420p"
        for i in range(int(round(seconds_video * 30))):
            img = np.full((32, 32, 3), int(round(i * PX)), dtype=np.uint8)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


def _materialize(mat_mod, spec, row: dict, src: Path, out: Path) -> dict:
    return mat_mod.materialize_row(
        {"row": row, "mat": spec["input_contract"]["materialize"], "src_root": str(src), "out_root": str(out),
         "recipe_sha256": mat_mod.recipe_sha256(spec["input_contract"]["materialize"]), "pts_tol": 1e-3, "check_x264": False}
    )


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg with libx264 + alac not on PATH")
@pytest.mark.parametrize("delta", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_materialization_recipe_on_synthetic_clip(tmp_path: Path, spec, mat_mod, delta: float):
    """The pinned recipe's output: audio bit-identical to the source window [1 − δ/2, 9 − δ/2], video frame times =
    source pts − (1 + δ/2), the first frame the source frame at or after 1 + δ/2 (decoded independently here)."""
    av = pytest.importorskip("av")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf

    clip = "synthclip_000001"
    _write_source_clip(tmp_path / "src", clip)
    b = OFFSETS.index(delta)
    row = {"clip_id": f"{clip}#{b}", "split": "test", "rel_video": f"video/{clip}.mp4", "rel_audio": f"audio/{clip}.wav",
           "offset_seconds": repr(delta), "rel_media": f"xx/{clip}_{b}.mp4"}
    rec = _materialize(mat_mod, spec, row, tmp_path / "src", tmp_path / "out")
    assert rec["ok"], rec
    out = tmp_path / "out" / row["rel_media"]
    assert hashlib.sha256(out.read_bytes()).hexdigest() == rec["sha256"]
    v0, a0 = 1.0 + delta / 2, round((1.0 - delta / 2) * 16000)
    with av.open(str(out)) as f:
        assert [s.type for s in f.streams] == ["video", "audio"]
        assert f.streams.audio[0].codec_context.name == "alac" and f.streams.video[0].codec_context.name == "h264"
        frames = [(float(fr.time), fr.to_ndarray(format="rgb24")) for fr in f.decode(f.streams.video[0])]
    with av.open(str(out)) as f:
        pcm = np.concatenate([fr.to_ndarray() for fr in f.decode(f.streams.audio[0])], axis=1)
    src_pcm, _ = sf.read(tmp_path / "src" / "audio" / f"{clip}.wav", dtype="int16")
    assert pcm.shape == (1, 128000) and np.array_equal(pcm[0], src_pcm[a0 : a0 + 128000])
    k0 = math.ceil(v0 * 30 - 1e-9)  # the source frame at or just after v0 (source pts = i / 30)
    assert len(frames) == 240
    times = np.array([t for t, _ in frames])
    assert np.allclose(times, (np.arange(k0, k0 + 240) / 30) - v0, atol=1e-3)
    idx = np.array([img.astype(np.float64).mean() / PX for _, img in frames])
    assert np.allclose(idx, np.arange(k0, k0 + 240), atol=2.0), idx[:5]  # the content is the source frame (re-encode noise)


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg with libx264 + alac not on PATH")
def test_materialization_refuses_padding_and_is_deterministic(tmp_path: Path, spec, mat_mod):
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    clip = "synthclip_000002"
    _write_source_clip(tmp_path / "src", clip, seconds_audio=9.5)
    row = {"clip_id": f"{clip}#0", "split": "test", "rel_video": f"video/{clip}.mp4", "rel_audio": f"audio/{clip}.wav",
           "offset_seconds": "-2.0", "rel_media": f"yy/{clip}.mp4"}
    rec = _materialize(mat_mod, spec, row, tmp_path / "src", tmp_path / "out")
    assert not rec["ok"] and "no padding" in rec["problems"][0]  # audio window [2, 10] needs 10 s of audio
    assert rec["refused"] is True  # by design: the run still exits 0
    assert not (tmp_path / "out" / row["rel_media"]).exists()
    row = {**row, "clip_id": f"{clip}#4", "offset_seconds": "0.0"}
    a = _materialize(mat_mod, spec, row, tmp_path / "src", tmp_path / "o1")
    b = _materialize(mat_mod, spec, row, tmp_path / "src", tmp_path / "o2")
    assert a["ok"] and b["ok"] and a["sha256"] == b["sha256"]  # the encoder is deterministic


# ----------------------------------------------------------------------------- materialization driver: logs, shards, exit codes
#
# Concurrent shards on different NFS clients must not share a log (appends are not atomic there), a torn log line must
# not be fatal, and by-design refusals must exit 0. The driver tests below replace materialize_row by a fake (no
# ffmpeg) and run main() itself.

_DRIVER = r'''
import hashlib, importlib.util, os, shutil, sys, time
spec = importlib.util.spec_from_file_location("avsync_materialize_driver", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PIN = mod.load_spec()["input_contract"]["materialize"]["tool"]["version_pin"]
mod.ffmpeg_version = lambda: PIN
shutil.which = lambda name, *a, **k: "/usr/bin/" + name
MODE = sys.argv[2]


def fake(task):
    r = task["row"]
    rec = {"clip_id": r["clip_id"], "split": r["split"], "rel_media": r["rel_media"], "recipe_sha256": task["recipe_sha256"],
           "ok": False, "pad": "x" * 8192, "t_done": time.time()}
    if r["offset_bin"] == "8":  # every delta = +2 row: refused by design
        rec.update(problems=["refused: video window [2.0, 10.0) extends past the source video end 9.900 s (no padding)"], refused=True)
        return rec
    if MODE == "fail" and r["offset_bin"] == "0":
        rec["problems"] = ["RuntimeError: ffmpeg exit 1: boom"]
        return rec
    out = os.path.join(task["out_root"], r["rel_media"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = ("mp4 of " + r["clip_id"]).encode()
    with open(out, "wb") as fh:
        fh.write(data)
    rec.update(ok=True, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return rec


mod.materialize_row = fake
sys.exit(mod.main(sys.argv[3:]))
'''


def _driver(tmp_path: Path, mode: str, *args: str) -> subprocess.Popen:
    drv = tmp_path / "driver.py"
    if not drv.is_file():
        drv.write_text(_DRIVER, encoding="utf-8")
    argv = [sys.executable, "-I", str(drv), str(HERE / "prepare" / "03_materialize.py"), mode,
            "--source-root", str(tmp_path / "src"), "--out-root", str(tmp_path / "out"), "--workers", "1", *args]
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _finish(p: subprocess.Popen) -> tuple[int, dict]:
    out, err = p.communicate(timeout=300)
    assert out.strip().startswith("{"), err
    return p.returncode, json.loads(out)


def test_read_log_merges_writer_logs_and_skips_torn_lines(tmp_path: Path, mat_mod):
    d = tmp_path / "_materialize"
    d.mkdir()
    rec = lambda c, sha, **kw: json.dumps({"clip_id": c, "ok": True, "sha256": sha, "recipe_sha256": "r", **kw})  # noqa: E731
    # a shared log with records without t_done, NUL runs and an interleaved record
    (d / "log.jsonl").write_bytes(
        (rec("a#0", "old-a") + "\n" + rec("b#1", "b") + "\n").encode()
        + b"\x00" * 300 + b"\n"
        + b'{"clip_id": "c#2", "ok": tr{"clip_id": "d#3", "ok": true}\n'
        + (json.dumps({"clip_id": "e#4", "ok": False, "problems": ["x"]}) + "\n").encode()
    )
    (d / "log.0-of-2.jsonl").write_text(rec("a#0", "new-a", t_done=2.0) + "\n" + rec("f#5", "f-late", t_done=9.0) + "\n", encoding="utf-8")
    (d / "log.1-of-2.jsonl").write_text(rec("f#5", "f-early", t_done=1.0) + "\n" + '{"clip_id": "g#6", "ok": tru', encoding="utf-8")
    (d / "notes.txt").write_text("not a log", encoding="utf-8")
    stats: dict = {}
    for arg in (d, d / "log.jsonl", d / "log.0-of-2.jsonl"):  # the directory or any log file in it: all logs merge
        done = mat_mod.read_log(arg, stats)
        assert sorted(done) == ["a#0", "b#1", "f#5"]
        assert done["a#0"]["sha256"] == "new-a" and done["f#5"]["sha256"] == "f-late"  # larger t_done wins over file order
        assert stats == {"files": 3, "lines": 9, "unparseable": 3}
    assert mat_mod.read_log(tmp_path / "nowhere" / "log.jsonl") == {}


def test_log_path_is_per_shard_and_torn_tail_is_terminated(tmp_path: Path, mat_mod):
    assert mat_mod.log_path(tmp_path, None) == tmp_path / "_materialize" / "log.jsonl"
    assert mat_mod.log_path(tmp_path, "3/64") == tmp_path / "_materialize" / "log.3-of-64.jsonl"
    for bad in ("64/64", "x/2", "1", "-1/4"):
        with pytest.raises(SystemExit):
            mat_mod.log_path(tmp_path, bad)
    log = mat_mod.log_path(tmp_path, "0/2")
    with mat_mod.open_log_for_append(log) as fh:  # a fresh file: nothing prepended
        fh.write(b'{"clip_id": "a#0", "ok": true}\n')
    with open(log, "ab") as fh:  # the shard was killed mid-line
        fh.write(b'{"clip_id": "b#1", "o')
    with mat_mod.open_log_for_append(log) as fh:
        fh.write(b'{"clip_id": "c#2", "ok": true}\n')
    stats: dict = {}
    assert sorted(mat_mod.read_log(log, stats)) == ["a#0", "c#2"] and stats["unparseable"] == 1


def test_concurrent_shards_write_own_logs_refusals_exit_0(tmp_path: Path, spec, mat_mod, manifest_rows):
    """Two --shard processes at once on one out_root: each writes only log.<i>-of-2.jsonl, every line parses, both
    exit 0 although their δ = +2 rows are refused by design; a rerun resumes from all logs; a torn line in a log
    is skipped and counted, not fatal; an unexpected failure exits 1."""
    rsha = mat_mod.recipe_sha256(spec["input_contract"]["materialize"])
    picked = [manifest_rows[k] for k in mat_mod.stratified(manifest_rows, 36)]
    shard_rows = {i: picked[i::2] for i in (0, 1)}
    procs = [_driver(tmp_path, "ok", "--rows", "36", "--shard", f"{i}/2") for i in (0, 1)]
    res = [_finish(p) for p in procs]
    logs = tmp_path / "out" / "_materialize"
    assert sorted(p.name for p in logs.iterdir()) == ["log.0-of-2.jsonl", "log.1-of-2.jsonl"]
    for i, (rc, summary) in enumerate(res):
        want = shard_rows[i]
        n_ref = sum(1 for r in want if r["offset_bin"] == "8")
        assert n_ref > 0 and rc == 0, summary
        assert (summary["ok"], summary["refused"], summary["failed"], summary["failures"]) == (len(want) - n_ref, n_ref, 0, [])
        assert summary["log"] == str(logs / f"log.{i}-of-2.jsonl") and summary["log_unparseable_lines"] == 0
        lines = (logs / f"log.{i}-of-2.jsonl").read_text(encoding="utf-8").splitlines()
        assert sorted(json.loads(x)["clip_id"] for x in lines) == sorted(r["clip_id"] for r in want)  # one writer per file
    done = mat_mod.read_log(logs)
    assert sorted(done) == sorted(r["clip_id"] for r in picked if r["offset_bin"] != "8")
    assert all(d["recipe_sha256"] == rsha and (tmp_path / "out" / d["rel_media"]).is_file() for d in done.values())

    # resume: verified rows are skipped (only the refused ones are tried again); a torn tail is skipped and counted
    with open(logs / "log.1-of-2.jsonl", "ab") as fh:
        fh.write(b"\x00" * 64 + b'{"clip_id": "zz#0", "ok": tr')
    rc, summary = _finish(_driver(tmp_path, "ok", "--rows", "36", "--shard", "1/2"))
    n_ref = sum(1 for r in shard_rows[1] if r["offset_bin"] == "8")
    assert rc == 0 and summary["rows_skipped_already_done"] == len(shard_rows[1]) - n_ref and summary["rows_run"] == n_ref
    assert summary["log_unparseable_lines"] == 1 and summary["log_files_read"] == 2
    tail = (logs / "log.1-of-2.jsonl").read_bytes().split(b"\n")
    assert json.loads(tail[-2])["refused"] is True  # the next record starts on its own line

    # an unexpected failure (not a refusal) still fails the shard
    s = next(i for i in (0, 1) if any(r["offset_bin"] == "0" for r in shard_rows[i]))
    rc, summary = _finish(_driver(tmp_path, "fail", "--rows", "36", "--shard", f"{s}/2", "--force"))
    assert rc == 1 and summary["failed"] == sum(1 for r in shard_rows[s] if r["offset_bin"] == "0") > 0 and all("RuntimeError" in f["problems"][0] for f in summary["failures"])

    # --write-record over the merged logs: rows not materialized yet are listed, nothing is written (the record is
    # per-site, under <out-root>/_materialize/, never in the folder)
    rc, summary = _finish(_driver(tmp_path, "ok", "--write-record"))
    assert rc == 1 and summary["write_record"] is False and summary["log_files"] == 2 and summary["log_unparseable_lines"] == 1
    assert not (logs / mat_mod.RECORD).exists() and not (HERE / mat_mod.RECORD).exists()


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg with libx264 + alac not on PATH")
def test_materialization_marks_short_source_as_refused(tmp_path: Path, spec, mat_mod):
    """A source video that ends before the δ = +2 window's end is refused BY DESIGN (``refused: true``), not failed."""
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    clip = "synthclip_000003"
    _write_source_clip(tmp_path / "src", clip, seconds_video=9.6)
    row = {"clip_id": f"{clip}#8", "split": "test", "rel_video": f"video/{clip}.mp4", "rel_audio": f"audio/{clip}.wav",
           "offset_seconds": "2.0", "rel_media": f"zz/{clip}.mp4"}
    rec = _materialize(mat_mod, spec, row, tmp_path / "src", tmp_path / "out")
    assert not rec["ok"] and rec["refused"] is True and "past the source video end" in rec["problems"][0], rec
    assert not (tmp_path / "out" / row["rel_media"]).exists()
    gone = {**row, "rel_video": "video/absent.mp4", "rel_audio": "audio/absent.wav"}  # missing source: a failure
    rec = _materialize(mat_mod, spec, gone, tmp_path / "src", tmp_path / "out")
    assert not rec["ok"] and not rec.get("refused")


# ----------------------------------------------------------------------------- reader smoke + output contract


@pytest.fixture()
def synthetic_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec, mat_mod):
    """Train rows 0 and 1 (two draws of one clip) materialized from a synthetic source under tmp_path."""
    if not _ffmpeg_ok():
        pytest.skip("ffmpeg with libx264 + alac not on PATH")
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    from yavih import Benchmark

    monkeypatch.setenv("YAVIH_CONFIG", NO_CONFIG)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for k in ("MEDIA_ROOT", "DERIVED_ROOT", "CURATION_ROOT"):
        monkeypatch.delenv(f"YAVIH_DATA__AVSYNC__{k}", raising=False)
    with gzip.open(HERE / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "train"][:2]
    clip = rows[0]["clip_id"].split("#")[0]
    assert rows[1]["clip_id"].split("#")[0] == clip, "train rows 0 and 1 are the two draws of one clip"
    _write_source_clip(tmp_path / "src", clip)
    for r in rows:
        assert _materialize(mat_mod, spec, r, tmp_path / "src", tmp_path / "derived")["ok"]
    return Benchmark.from_dir(HERE, data={"media_root": str(tmp_path / "src"), "derived_root": str(tmp_path / "derived")}), rows, tmp_path


def test_reader_smoke_synthetic_raw_files(synthetic_bench):
    """The raw hand-off: the stored file, byte for byte, as two streams of one mp4; facts only in meta; info keys of
    input_contract.info non-null; nothing about the offset reaches the Sample."""
    bench, rows, tmp = synthetic_bench
    samples = list(bench.reader("train", rows=range(0, 2)))
    assert [s.row for s in samples] == [0, 1]
    for s, r in zip(samples, rows, strict=True):
        assert set(s.streams) == {"video", "audio"} and s.meta == {"facts": {}}
        f = tmp / "derived" / r["rel_media"]
        v, a = s.streams["video"], s.streams["audio"]
        assert v.path == a.path and v._real_path == str(f) and v.bytes() == a.bytes() == f.read_bytes()  # one opaque handle per file
        assert v.kind == "video" and a.kind == "audio"
        for key in ("duration_s", "fps", "width", "height", "container", "bytes"):
            assert v.info[key] is not None, key
        for key in ("duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"):
            assert a.info[key] is not None, key
        assert a.info["sample_rate"] == 16000 and a.info["n_samples"] == 128000 and a.info["channels"] == 1
        assert (v.info["width"], v.info["height"]) == (32, 32) and abs(v.info["duration_s"] - 8.0) < 0.05
        assert "#" not in v.path and r["offset_seconds"] not in json.dumps(s.meta)
    assert samples[0].streams["video"].bytes() != samples[1].streams["video"].bytes()


def test_output_contract_with_template_bridge(synthetic_bench):
    """The template bridge on the synthetic raw files satisfies the output contract: fixed [B, N, D], pos in media
    coordinates ([0, 1] ∪ {-1}, video token centres inside the declared crop region), `used` per stream with frame
    times inside the media and matching the declared rule within half a frame."""
    import numpy as np

    from yavih.media.decode import frames_at  # noqa: F401
    from yavih.media.rules import crop_region, rule_times

    bench, _, _ = synthetic_bench
    mod = F.load_module(HERE / "bridge_template.py", "avsync_bridge_template_o")
    view = bench.view()
    bridge = mod.AVSyncBridge(view, {"reading": "av", "layer": -1})
    samples = list(bench.reader("train", rows=range(0, 2)))
    feats, pos, used = F.encode_prepared(bridge, samples)
    n_video = mod.N_FRAMES // mod.TUBELET * mod.VIDEO_H * mod.VIDEO_W
    assert feats.shape == (2, n_video + mod.AUDIO_T * mod.AUDIO_F, 1) and pos.shape == (2, feats.shape[1], 4)
    ok = ((pos >= 0) & (pos <= 1)) | (pos == -1)
    assert ok[..., 1:].all() and set(np.unique(pos[..., 0])) == {0.0, 1.0}
    for s, p, u in zip(samples, pos, used, strict=True):
        v = s.streams["video"]
        t = np.asarray(u["video"]["times"])
        assert len(t) == mod.N_FRAMES and (t >= 0).all() and (t <= v.info["duration_s"] + 1e-3).all()
        want = rule_times(mod.VIDEO_TEMPORAL, v.info)
        assert np.abs(t - want).max() <= 0.5 / v.info["fps"] + 1e-3
        y0, x0, y1, x1 = crop_region(mod.VIDEO_SPATIAL, v.info["width"], v.info["height"])
        vy, vx = p[:n_video, 2], p[:n_video, 3]
        assert (vy >= y0 - 1e-6).all() and (vy <= y1 + 1e-6).all() and (vx >= x0 - 1e-6).all() and (vx <= x1 + 1e-6).all()
        assert u["audio"] == {"samples": [[0, 128000]], "pad": [0, 0]}
    feats2, pos2, used2 = F.encode_prepared(bridge, samples[::-1])  # order invariance on the placeholder pipeline
    assert np.array_equal(feats2[::-1], feats) and np.array_equal(pos2[::-1], pos) and used2[::-1] == used


# ----------------------------------------------------------------------------- the site's materialized media


def _derived_root() -> Path | None:
    root = F.site_root(HERE, "derived_root")
    return root if root is not None and any((root / "_materialize").glob("log*.jsonl")) else None


@pytest.mark.skipif(_derived_root() is None, reason="no materialized avsync media (derived_root/_materialize/log*.jsonl) at this site")
def test_materialized_media_present_match_log(spec, mat_mod, manifest_rows):
    """Up to 12 verified rows of the site's materialization log: the file exists, its sha256 equals the log's, the
    log's recipe is the yaml's, and (when this site wrote one with --write-record) the site's per-row record."""
    root = _derived_root()
    done = mat_mod.read_log(root / "_materialize")  # every writer's log, merged
    rsha = mat_mod.recipe_sha256(spec["input_contract"]["materialize"])
    current = [d for d in done.values() if d.get("recipe_sha256") == rsha]
    if not current:
        pytest.skip("the site's materialization log was written under another recipe")
    by_id = {r["clip_id"]: r for r in manifest_rows}
    record = {}
    rec_file = root / "_materialize" / mat_mod.RECORD
    if rec_file.is_file():
        record = {r["clip_id"]: r["sha256"] for r in csv.DictReader(io.StringIO(gzip.decompress(rec_file.read_bytes()).decode()))}
    for d in sorted(current, key=lambda d: d["clip_id"])[:12]:
        assert by_id[d["clip_id"]]["rel_media"] == d["rel_media"]
        f = root / d["rel_media"]
        assert f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() == d["sha256"]
        assert d["audio_samples"] == 128000 and d["pts_max_err_s"] <= 1e-3
        if record and d["clip_id"] in record:
            assert record[d["clip_id"]] == d["sha256"]
