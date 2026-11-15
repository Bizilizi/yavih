"""Folder tests of avrobustbench (run: ``python -m pytest "$(yavih bench path avrobustbench)/tests" -q -p no:cacheprovider``).

- yaml / manifest / classes / curation / pins: read the folder files, run anywhere;
- metrics (eval.py) on hand-checked populations, and the testing.py stand-in;
- the materialization recipe (prepare/04_materialize.py with the copied corruption code) on a synthetic source clip:
  needs ffmpeg with libx264, PyAV, soundfile, opencv, scikit-image and Wand (skips otherwise);
- the reader smoke and the template bridge on synthetic materialized files (ffmpeg, PyAV, soundfile);
- make_manifest.py --check: needs the site's vggsound.csv and the VGGSOUND-2C member list (skips otherwise).
``yavih bench lint`` (run once per folder by the repository) checks the layout, hashes, conformance fixture and the
README.md / setup.md structure, so none of that is repeated here.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows = F.spec, F.rows

READOUT = "clf"
HEADLINE = "macc"
STREAMS = {"video": "video", "audio": "audio"}
CLIPS = {"train": 20000, "test": 224736}
EVAL_CLIPS = 14046
READINGS = ["a", "av", "v"]
EXTERNAL: list[str] = []
CONDITIONS = ["clean", "gaussian", "impulse", "shot", "speckle", "compression", "snow", "frost", "spatter", "wind",
              "rain", "underwater", "concert", "smoke", "crowd", "interference"]
NO_CONFIG = str(HERE / "tests" / "_no_such_config.toml")


@pytest.fixture(scope="module")
def ev():
    return F.load_module(HERE / "eval.py", "avrobustbench_eval_t")


@pytest.fixture(scope="module")
def mat():
    return F.load_module(HERE / "prepare" / "04_materialize.py", "avrobustbench_materialize_t")


def _ffmpeg_ok() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    return "libx264" in subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE
    assert spec["metrics"]["floors"] == ["macc_floor"]
    assert [m for m in spec["metrics"]["list"] if m.startswith("acc_")] == [f"acc_{c}" for c in CONDITIONS]


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)
    assert [e["name"] for e in spec["external"] or []] == EXTERNAL
    assert spec["protocol"]["conditions"] == CONDITIONS and spec["protocol"]["severity"] == 5
    groups = spec["protocol"]["corruption_groups"]
    assert sorted(c for g in groups.values() for c in g) == sorted(CONDITIONS[1:])
    assert [len(groups[g]) for g in ("digital", "environmental", "human")] == [5, 6, 4]
    m = ic["materialize"]
    assert m["kind"] == "avrobustbench_corrupt" and m["severity"] == spec["protocol"]["severity"]
    assert m["frames"] == 10 and m["size"] == 224 and m["fps"] == 1


def test_copied_code_and_pins(spec: dict):
    """The recipe pins the copied code files and the asset list; the three copies of each upstream pin agree."""
    m = spec["input_contract"]["materialize"]
    for key, name in (("corruptions", "corruptions.py"), ("cavmae_frames", "cavmae_frames.py")):
        assert hashlib.sha256((HERE / "prepare" / name).read_bytes()).hexdigest() == m["code_sha256"][key]
    sh = (HERE / "prepare" / "03_corruption_assets.sh").read_text(encoding="utf-8")
    listing = sh.split("<<'SHA'\n", 1)[1].split("SHA\n)", 1)[0]
    assert len(listing.splitlines()) == m["assets"]["files"] == 146
    assert hashlib.sha256(listing.encode()).hexdigest() == m["assets"]["list_sha256"]
    assert f"COMMIT={m['assets']['commit']}" in sh
    up = spec["manifest"]["upstream"]["upstream_files"]
    lst = F.load_module(HERE / "prepare" / "02_vggsound2c_list.py", "avrobustbench_list_t")
    mm = F.load_module(HERE / "make_manifest.py", "avrobustbench_mm_t")
    assert lst.MEMBERS_SHA256 == mm.MEMBERS_SHA == up["av_robust_data/clean_vgg.members.txt"] == spec["source"]["sha256"]["av_robust_data/clean_vgg.members.txt"]
    assert lst.ZIP_SHA256 == mm.CLEAN_ZIP_SHA == up["av_robust_data/vgg_data/clean_vgg.zip"]
    assert lst.REVISION == mm.HF_REVISION and lst.REVISION in spec["source"]["revision"]
    assert mm.CONDITIONS == CONDITIONS
    corr = (HERE / "prepare" / "corruptions.py").read_text(encoding="utf-8")
    for n in range(1, 6):
        assert f"yavih change {n}" in corr, n
    assert "f6c3f2e12de918373a36575f8fec14d84adc4707" in corr and "Permission is hereby granted" in corr


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file() and re.match(r"^\d", p.name))
    assert names == ["01_vggsound_media.sh", "02_vggsound2c_list.py", "03_corruption_assets.sh", "04_materialize.py"]


# ----------------------------------------------------------------------------- manifest facts


def test_classes(spec: dict):
    classes = F.read_classes(HERE, spec)
    names = [c["name"] for c in classes]
    assert len(names) == 309 and names == sorted(names) and [int(c["index"]) for c in classes] == list(range(309))


def test_eval_rows_are_clip_condition_pairs(rows: list[dict[str, str]]):
    """Every eval clip appears under all 16 conditions with one label; clip_id = <clip>#<condition>; paths match."""
    by_clip: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        clip, _, cond = r["clip_id"].rpartition("#")
        assert cond == r["condition"]
        assert r["src_video"] == f"video/{clip}.mp4" and r["src_audio"] == f"audio/{clip}.wav"
        assert r["rel_video"] == f"media/{r['split']}/{cond}/{clip}.mp4"
        assert r["rel_audio"] == f"media/{r['split']}/{cond}/{clip}.wav"
        if r["split"] == "test":
            assert r["label"] == by_clip[clip].setdefault("label", r["label"])
            by_clip[clip][cond] = r["label"]
        else:
            assert cond == "clean"
    assert len(by_clip) == EVAL_CLIPS
    assert all(sorted(k for k in v if k != "label") == sorted(CONDITIONS) for v in by_clip.values())
    order = [r["condition"] for r in rows if r["split"] == "test"]
    assert order == CONDITIONS * EVAL_CLIPS  # clip-major: the first 16 k rows are k whole clips (plan build --sample)


def test_no_ytid_shared_between_train_and_eval(rows: list[dict[str, str]]):
    yt = {s: {r["clip_id"].rpartition("#")[0].rsplit("_", 1)[0] for r in rows if r["split"] == s} for s in CLIPS}
    assert not yt["train"] & yt["test"]


def test_train_draw_is_class_balanced(rows: list[dict[str, str]]):
    counts = Counter(r["label"] for r in rows if r["split"] == "train")
    assert len(counts) == 309 and min(counts.values()) == 64 and max(counts.values()) == 65


def test_curation_sums(spec: dict):
    up = spec["manifest"]["upstream"]
    cur = up["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"] == 183730 + 16 * 15446
    assert cur["excluded"]["not_in_vggsound2c"] == 16 * (15446 - EVAL_CLIPS)
    assert cur["leakage_audit"]["excluded"] == cur["excluded"]["leakage_train_ytid"]
    assert up["rows"] == EVAL_CLIPS + 183730


# ----------------------------------------------------------------------------- metrics on hand-checked inputs


def _population(correct_by_cond: dict[str, list[bool]], n_classes: int = 5):
    """Rows of ``len(list)`` source clips per condition; each row's correctness given; returns outputs, targets, meta."""
    ids, outs, tgts = [], [], []
    for cond, oks in correct_by_cond.items():
        for k, ok in enumerate(oks):
            t = k % n_classes
            o = np.full(n_classes, -1.0)
            o[t if ok else (t + 1) % n_classes] = 1.0
            ids.append(f"clip{k:03d}_000001#{cond}")
            outs.append(o)
            tgts.append(t)
    return np.asarray(outs), np.asarray(tgts, np.int64), {"eval_clip_ids": ids}


def test_metrics_hand_checked(ev, spec: dict):
    proto = spec["protocol"]
    # clean 4/4 correct; corruption i: the first (i % 5) of 4 clips correct, capped at 4
    pat = {"clean": [True] * 4}
    for i, c in enumerate(CONDITIONS[1:]):
        pat[c] = [k < (i % 5) for k in range(4)]
    out, tgt, meta = _population(pat)
    acc = ev.per_condition(out, tgt, proto, meta)
    want = {c: (1.0 if c == "clean" else min(i % 5, 4) / 4) for i, c in enumerate(["clean", *CONDITIONS[1:]], -1)}
    want["clean"] = 1.0
    assert acc == pytest.approx(want)
    m = float(np.mean([want[c] for c in CONDITIONS[1:]]))
    M = ev.METRICS
    assert M["macc"].fn(out, tgt, proto, meta) == pytest.approx(m)
    assert M["alpha"].fn(out, tgt, proto, meta) == pytest.approx(1 - (1.0 - m))
    assert M["rho"].fn(out, tgt, proto, meta) == pytest.approx(1 - (1.0 - m) / 1.0)
    for g, members in proto["corruption_groups"].items():
        assert M[f"macc_{g}"].fn(out, tgt, proto, meta) == pytest.approx(np.mean([want[c] for c in members]))
    assert M["n_clips"].fn(out, tgt, proto, meta) == 4
    pc = M["macc"].per_clip(out, tgt, proto, meta)
    assert pc.shape == (4, 15) and np.nanmean(pc) == pytest.approx(m)
    v = M["acc_clean"].per_clip(out, tgt, proto, meta)
    assert np.isfinite(v).sum() == 4 and np.nansum(v) == 4
    # the floor: a constant prediction through the same function
    floor = np.zeros(len(tgt), np.int64)
    assert M["macc_floor"].fn(floor, tgt, proto, meta) == pytest.approx(
        np.mean([np.mean([k % 5 == 0 for k in range(4)])] * 15))


def test_rho_zero_clean_and_missing_condition(ev, spec: dict):
    proto = spec["protocol"]
    pat = {c: [False, False] for c in CONDITIONS}
    pat["gaussian"] = [True, False]
    out, tgt, meta = _population(pat)
    assert ev.METRICS["rho"].fn(out, tgt, proto, meta) == 0.0
    assert ev.METRICS["alpha"].fn(out, tgt, proto, meta) == pytest.approx(1 + 0.5 / 15)
    del pat["frost"]
    out, tgt, meta = _population(pat)
    with pytest.raises(ValueError, match="frost"):
        ev.METRICS["macc"].fn(out, tgt, proto, meta)
    with pytest.raises(ValueError, match="#<condition>"):
        ev.conditions_of({"eval_clip_ids": ["clip_000001"]})


def test_paper_table6_mean(ev, spec: dict):
    """macc and rho reproduce the paper's arithmetic: CAV-MAE source on VGGSOUND-2C (Table 6), mean 35.54 of 65.50."""
    t6 = [20.39, 23.73, 20.72, 25.34, 17.26, 25.07, 46.82, 48.46, 50.17, 29.89, 42.19, 47.61, 32.93, 47.71, 54.88]
    n = 10000
    pat = {"clean": [k < 6550 for k in range(n)]}
    for c, a in zip(CONDITIONS[1:], t6, strict=True):
        pat[c] = [k < round(a * 100) for k in range(n)]
    out, tgt, meta = _population(pat)
    proto = spec["protocol"]
    assert ev.METRICS["macc"].fn(out, tgt, proto, meta) * 100 == pytest.approx(35.54, abs=0.01)
    assert ev.METRICS["rho"].fn(out, tgt, proto, meta) == pytest.approx(0.54, abs=0.005)
    assert ev.METRICS["alpha"].fn(out, tgt, proto, meta) == pytest.approx(0.70, abs=0.005)


def test_stand_in_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    from yavih import Benchmark

    monkeypatch.setenv("YAVIH_CONFIG", NO_CONFIG)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    testing = F.load_module(HERE / "testing.py", "avrobustbench_testing_t")
    folder = testing.make_stand_in(tmp_path / "w", Benchmark.from_dir(HERE), n_eval=20, n_train=4)
    b = Benchmark.load(str(folder))
    assert b.spec.readout == "clf" and b.spec.metrics.headline == "macc"
    ids = [r["clip_id"] for r in F.read_rows(Path(folder)) if r["split"] == "val"]
    assert len(ids) == 64 and Counter(i.rpartition("#")[2] for i in ids) == {c: 4 for c in CONDITIONS}


# ----------------------------------------------------------------------------- the materialization recipe


def _write_source_clip(root: Path, clip: str, *, size=(240, 320), seconds=10.2, fps=30) -> None:
    """A 30 fps H.264 mp4 whose frames are smooth gradients that move with the frame index, and a 16 kHz noise wav."""
    import av
    import soundfile as sf

    (root / "video").mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    with av.open(str(root / "video" / f"{clip}.mp4"), "w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        for i in range(int(seconds * fps)):
            img = np.stack([(xx + 3 * i) % 256, (yy + 2 * i) % 256, np.full_like(xx, (5 * i) % 256)], -1).astype(np.uint8)
            for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    rng = np.random.default_rng(0)
    sf.write(str(root / "audio" / f"{clip}.wav"), (0.1 * rng.standard_normal(int(seconds * 16000))).astype(np.float32),
             16000, subtype="PCM_16")


@pytest.fixture()
def materialized(tmp_path: Path, mat, spec: dict):
    """Clean + four asset-free corruptions of one synthetic source clip, materialized twice (determinism)."""
    if not _ffmpeg_ok():
        pytest.skip("ffmpeg with libx264 not on PATH")
    for mod in ("av", "soundfile", "cv2", "skimage", "wand.image", "torchaudio", "torchvision"):
        pytest.importorskip(mod)
    clip = "SYNTHclip01_000030"
    _write_source_clip(tmp_path / "src", clip)
    rows_ = [{"clip_id": f"{clip}#{c}", "condition": c, "split": "test",
              "rel_video": f"media/test/{c}/{clip}.mp4", "rel_audio": f"media/test/{c}/{clip}.wav"}
             for c in ("clean", "gaussian", "impulse", "compression", "interference")]
    m = spec["input_contract"]["materialize"]
    mat._init_worker(str(tmp_path / "no_assets"))  # these five conditions read no noise file or image
    outs = []
    for k in range(2):
        job = {"src_video": f"video/{clip}.mp4", "src_audio": f"audio/{clip}.wav", "rows": rows_, "recipe": m,
               "recipe_sha256": mat.recipe_sha256(m), "out_root": str(tmp_path / f"out{k}"),
               "source_root": str(tmp_path / "src"), "versions": {}, "drift": False}
        outs.append(mat.process_clip(job))
    return tmp_path, rows_, outs


def test_materialize_recipe(materialized):
    import av
    import soundfile as sf

    tmp, rows_, (a, b) = materialized
    assert all(not r["refused"] and "error" not in r for r in a), a
    # deterministic: the same seeds give the same bytes on one host
    assert [r["outputs"] for r in a] == [r["outputs"] for r in b]
    frames = {}
    for r in rows_:
        with av.open(str(tmp / "out0" / r["rel_video"])) as c:
            assert len(c.streams.audio) == 0
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            assert c.streams.video[0].average_rate == 1
        assert len(fr) == 10 and all(f.shape == (224, 224, 3) for f in fr)
        frames[r["condition"]] = np.stack(fr).astype(np.float64)
        info = sf.info(str(tmp / "out0" / r["rel_audio"]))
        assert (info.samplerate, info.channels, info.subtype, info.frames) == (16000, 1, "PCM_16", int(10.2 * 16000))
    # every corruption changes the frames; the clean frames follow the moving gradient (CAV-MAE indices)
    for c in ("gaussian", "impulse", "compression", "interference"):
        assert np.abs(frames[c] - frames["clean"]).mean() > 1.0, c
    w_clean, _ = sf.read(str(tmp / "out0" / rows_[0]["rel_audio"]))
    w_int, _ = sf.read(str(tmp / "out0" / rows_[-1]["rel_audio"]))
    zero = np.mean(w_int == 0)
    assert 0.4 <= zero <= 0.6 and np.mean(w_clean == 0) < 0.01  # interference at severity 5 silences ~50 %


def test_row_seed_is_per_row(mat):
    a = mat.row_seed(0, "x_000001#gaussian", "video")
    assert a != mat.row_seed(0, "x_000001#impulse", "video") != mat.row_seed(0, "x_000001#gaussian", "audio")
    assert 0 <= a < 2**32


# ----------------------------------------------------------------------------- reader smoke + template bridge


@pytest.fixture()
def synthetic_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, str]]):
    """Two eval rows (clean and gaussian of one clip) written as 10-frame 224 px mp4s + wavs under tmp_path."""
    if not _ffmpeg_ok():
        pytest.skip("ffmpeg with libx264 not on PATH")
    av = pytest.importorskip("av")
    sf = pytest.importorskip("soundfile")
    from yavih import Benchmark

    monkeypatch.setenv("YAVIH_CONFIG", NO_CONFIG)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for k in ("MEDIA_ROOT", "DERIVED_ROOT"):
        monkeypatch.delenv(f"YAVIH_DATA__AVROBUSTBENCH__{k}", raising=False)
    test_rows = [r for r in rows if r["split"] == "test"]
    picked = {0: test_rows[0], 1: test_rows[1]}  # clean and gaussian of the first clip
    assert picked[0]["clip_id"].rpartition("#")[0] == picked[1]["clip_id"].rpartition("#")[0]
    assert (picked[0]["condition"], picked[1]["condition"]) == ("clean", "gaussian")
    for k, r in picked.items():
        mp4, wav = tmp_path / r["rel_video"], tmp_path / r["rel_audio"]
        mp4.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(k)
        with av.open(str(mp4), "w") as c:
            st = c.add_stream("libx264", rate=1)
            st.width, st.height, st.pix_fmt = 224, 224, "yuv420p"
            for _ in range(10):
                img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    c.mux(p)
            for p in st.encode():
                c.mux(p)
        sf.write(str(wav), (0.1 * rng.standard_normal(150000)).astype(np.float32), 16000, subtype="PCM_16")
    bench = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path / "src"), "derived_root": str(tmp_path)})
    return bench, picked, tmp_path


def test_reader_smoke_synthetic_raw(synthetic_bench):
    bench, picked, tmp = synthetic_bench
    samples = [bench.sample("test", k) for k in sorted(picked)]
    assert [s.row for s in samples] == sorted(picked)
    for s, (_, r) in zip(samples, sorted(picked.items()), strict=True):
        assert set(s.streams) == {"video", "audio"} and s.meta == {"facts": {}}
        v, a = s.streams["video"], s.streams["audio"]
        assert v.bytes() == (tmp / r["rel_video"]).read_bytes() and a.bytes() == (tmp / r["rel_audio"]).read_bytes()
        for key in bench.spec.raw["input_contract"]["info"]["video"]:
            assert v.info[key] is not None, key
        for key in bench.spec.raw["input_contract"]["info"]["audio"]:
            assert a.info[key] is not None, key
        assert (v.info["width"], v.info["height"], v.info["n_frames"]) == (224, 224, 10)
        assert a.info["sample_rate"] == 16000 and a.info["n_samples"] == 150000
        assert r["condition"] not in json.dumps(s.meta)


def test_template_bridge_output_contract(synthetic_bench):
    from yavih.media.rules import rule_times

    bench, picked, _ = synthetic_bench
    mod = F.load_module(HERE / "bridge_template.py", "avrobustbench_bridge_template_t")
    bridge = mod.AVRobustBenchBridge(bench.view(), {"reading": "av", "layer": -1})
    samples = [bench.sample("test", k) for k in sorted(picked)]
    feats, pos, used = F.encode_prepared(bridge, samples)
    n_video = mod.N_FRAMES // mod.TUBELET * (mod.IMG // mod.PATCH) ** 2
    assert feats.shape == (2, n_video + mod.AUDIO_T * mod.AUDIO_F, mod.D) and pos.shape == (2, feats.shape[1], 4)
    ok = ((pos >= 0) & (pos <= 1)) | (pos == -1)
    assert ok[..., 1:].all() and set(np.unique(pos[..., 0])) == {0.0, 1.0}
    want = rule_times(mod.VIDEO_RULES["temporal"], samples[0].streams["video"].info)
    assert np.allclose(want, np.arange(10)) and all(np.allclose(u["video"]["times"], want) for u in used)
    d = bridge.describe()
    assert d["model"] == "TODO-model" and d["reading"] in READINGS


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values()
              if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    d = cls(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] in READINGS and set(d["input"]) <= set(STREAMS)


# ----------------------------------------------------------------------------- needs this benchmark's metadata


def _metadata_dir(tmp_path: Path) -> Path | None:
    media = F.site_root(HERE, "media_root", needs=("metadata/vggsound.csv",))
    derived = F.site_root(HERE, "derived_root", needs=("meta/clean_vgg.members.txt",))
    if media is None or derived is None:
        return None
    d = tmp_path / "meta"
    d.mkdir()
    shutil.copy2(media / "metadata" / "vggsound.csv", d / "vggsound.csv")
    shutil.copy2(derived / "meta" / "clean_vgg.members.txt", d / "clean_vgg.members.txt")
    return d


def test_make_manifest_check(tmp_path: Path, spec: dict, capsys: pytest.CaptureFixture):
    pytest.importorskip("torch")
    meta = _metadata_dir(tmp_path)
    if meta is None:
        pytest.skip("needs the site's VGGSound metadata/vggsound.csv (media_root) and meta/clean_vgg.members.txt "
                    "(derived_root, prepare step 02)")
    mm = F.load_module(HERE / "make_manifest.py", "avrobustbench_mm_check")
    assert mm.main(["--no-media", "--metadata", str(meta), "--check"]) == 0
    rep = json.loads(capsys.readouterr().out)
    up = spec["manifest"]["upstream"]
    assert rep["check"] == "ok" and rep["budget_clips"] == spec["budget"]["clips"]
    for key in ("source", "sha256", "rows", "upstream_files"):
        assert rep["upstream"][key] == up[key], key
    cur, got = up["curation"], rep["upstream"]["curation"]
    assert {k: v for k, v in got.items() if k != "per_split"} == {k: v for k, v in cur.items() if k != "per_split"}
    assert {s: v["listed"] for s, v in got["per_split"].items()} == {s: v["listed"] for s, v in cur["per_split"].items()}


def test_members_list_parses():
    """The member-list parser on a hand-written list (the real one needs prepare step 02)."""
    mm = F.load_module(HERE / "make_manifest.py", "avrobustbench_mm_parse")
    raw = b"clean_vgg/\nclean_vgg/DKN30Bwf_HQ_30.mp4\nclean_vgg/---g-f_I2yQ_1.mp4\n"
    assert mm.read_members(raw) == ["DKN30Bwf_HQ_000030", "---g-f_I2yQ_000001"]
    with pytest.raises(SystemExit):
        mm.read_members(b"clean_vgg/x.wav\n")
    buf = io.StringIO("aaa,1,dog,train\n")
    assert mm.read_vggsound_csv(buf.getvalue().encode()) == [("aaa_000001", "dog", "train")]
    assert gzip.decompress(mm.gzip_bytes(b"x")) == b"x" and csv is not None
