"""Folder tests for ``audioset20k`` (the folder's README.md and setup.md).

Run only this file: ``python -m pytest "$(yavih bench path audioset20k)/tests" -q -p no:cacheprovider``.
Offline except two tests: the real-media reader smoke and the byte-identical re-freeze need the downloaded
dataset (``index.json``, ``clips.csv`` and the tars under ``media_root``) and skip otherwise (site facts only).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows, bench, mm = F.spec, F.rows, F.bench, F.mm

REPO_ID = "11hu83/audioset20k"
HF_URL = "https://huggingface.co/datasets/11hu83/audioset20k"
REVISION = "4a0055b21c183257fd9b65ee8d1cd11f13362f0f"
ROWS = {"train": 17726, "eval": 16223}  # every published clip
# A recorded fact, never an exclusion: the published mp4s without an audio track (make_manifest.py NO_AUDIO_TRACK)
NO_AUDIO = {("train", "NlBQ4kLpdzk_0260000"), ("train", "d8WgfWSf1VM_0280000"), ("eval", "chc33hy6iIU_0510000")}


def _view(b):
    from yavih.bridge import BenchmarkView

    return b.view() if callable(getattr(b, "view", None)) and not _takes_split(b) else BenchmarkView.from_benchmark(b)


def _takes_split(b) -> bool:
    import inspect

    try:
        return "split" in inspect.signature(b.view).parameters
    except (TypeError, ValueError):
        return False


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


# ----------------------------------------------------------------------------- benchmark.yaml


def test_yaml_shape(spec):
    assert spec["schema_version"] == 2 and spec["name"] == HERE.name == "audioset20k"
    assert spec["aliases"] == [] and spec["readout"] == "clf"
    assert list(spec) == ["schema_version", "name", "aliases", "title", "header", "readout", "protocol_version", "changelog", "source",
                          "obtain", "budget", "manifest", "splits", "streams", "targets", "data", "input_contract", "output_contract",
                          "external", "protocol", "metrics", "readout_params"]
    assert spec["header"]["name"] == "AudioSet-20K" and "Gemmeke" in spec["header"]["venue"]
    assert sorted(p.name for p in (HERE / "prepare").iterdir() if p.name != "__pycache__") == ["01_download.sh", "02_verify.sh"]


def test_changelog(spec):
    log = spec["changelog"]
    assert [e["version"] for e in log] == [1] and spec["protocol_version"] == 1
    assert log[0]["invalidates"] is False and log[0]["supersedes"] == []


def test_source_pins_the_published_dataset(spec, mm):
    src = spec["source"]
    assert src["kind"] == "public_url" and src["urls"] == [HF_URL] and src["revision"] == REVISION == mm.REVISION
    assert src["sha256"] == {"SHA256SUMS": mm.SHA256SUMS_SHA256, "index.json": mm.INDEX_SHA256, "clips.csv": mm.CLIPS_SHA256}
    assert src["bytes"] == 66081637551
    assert "CC BY 4.0" in src["licence"] and "research" in src["licence"]
    up = spec["manifest"]["upstream"]
    assert up["sha256"] == mm.CLIPS_SHA256 and REVISION in up["source"] and up["rows"] == 33949
    assert up["upstream_files"] == {"clips.csv": mm.CLIPS_SHA256, "MISSING.csv": mm.MISSING_SHA256, "index.json": mm.INDEX_SHA256,
                                    "SHA256SUMS": mm.SHA256SUMS_SHA256}
    assert spec["manifest"]["classes_sha256"] == mm.CLASSES_SHA256
    assert spec["data"]["roots"]["media_root"]["default"] == "${data.root}/audioset20k-wds-v1"


def test_streams_one_member_two_streams(spec):
    """setup.md "## Input contract": one media column, one tar member read once, handed over as ``video`` and ``audio``."""
    s = spec["streams"]
    assert list(s) == ["video", "audio"]
    for name in ("video", "audio"):
        assert s[name] == {"kind": name, "media_column": "rel_media", "root": "media_root", "layout": "tar_index", "layout_params": {}, "required": True}
    assert spec["manifest"]["columns"]["rel_media"] == {"type": "str", "role": "media", "stream": ["video", "audio"]}
    assert spec["data"]["tar_index"] == "index.json" and spec["data"]["derived"] == {} and spec["targets"] is None


def test_input_output_contract(spec):
    ic, oc = spec["input_contract"], spec["output_contract"]
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(ic["info"]) == set(spec["streams"])
    assert ic["info"] == {"video": ["duration_s", "fps", "n_frames", "width", "height", "container", "bytes"],
                          "audio": ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]}
    assert REPO_ID in ic["handed_over"]["video"] and REVISION in ic["handed_over"]["video"]  # the release is hashed with the protocol
    assert "same mp4" in ic["handed_over"]["audio"]
    assert "3 clips' mp4 holds no audio track" in ic["handed_over"]["audio"] and "has_track false" in ic["handed_over"]["audio"]
    assert ic["facts"] == [] and ic["materialize"] is None
    assert oc == {"shape": "fixed", "coordinates": "media", "target_frame": None,
                  "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None}}
    assert spec["external"] == []
    assert spec["protocol"] == {"clip_seconds": 10.0, "train_draw": None, "views": "none"}


def test_info_keys_in_vocabulary(spec):
    from yavih.sample import INFO_VOCAB

    for name, ks in spec["input_contract"]["info"].items():
        assert set(ks) <= set(INFO_VOCAB[spec["streams"][name]["kind"]]), name


# ----------------------------------------------------------------------------- manifest / classes


def test_sha256_match(spec, rows):
    F.assert_manifest_hashes(HERE, spec)
    assert len(rows) == sum(ROWS.values()) == 33949 == spec["manifest"]["upstream"]["rows"]
    assert list(rows[0]) == ["clip_id", "split", "labels", "rel_media", "ytid", "start_s", "duration_s"]


def test_splits_assigned_once(spec, rows):
    assert F.assert_splits_assigned_once(spec, rows) == ROWS == spec["budget"]["clips"]
    assert [r["split"] for r in rows] == ["train"] * ROWS["train"] + ["eval"] * ROWS["eval"]
    for s in ("train", "eval"):  # clips.csv order: each split sorted by key
        keys = [r["clip_id"] for r in rows if r["split"] == s]
        assert keys == sorted(keys)
    assert len({r["clip_id"] for r in rows}) == len(rows) == len({r["ytid"] for r in rows})
    assert not ({r["ytid"] for r in rows if r["split"] == "train"} & {r["ytid"] for r in rows if r["split"] == "eval"})


def test_classes_contiguous_ontology_order():
    cls = list(csv.DictReader((HERE / "classes.csv").open(newline="", encoding="utf-8")))
    assert list(cls[0]) == ["index", "name", "mid"]
    assert [int(r["index"]) for r in cls] == list(range(527))
    assert len({r["name"] for r in cls}) == 527 and len({r["mid"] for r in cls}) == 527
    assert (cls[0]["name"], cls[0]["mid"]) == ("Speech", "/m/09x0r")


def test_label_and_media_cells(rows):
    cls = list(csv.DictReader((HERE / "classes.csv").open(newline="", encoding="utf-8")))
    idx = {r["mid"]: int(r["index"]) for r in cls}
    pos = {"train": set(), "eval": set()}
    for r in rows:
        mids = r["labels"].split(";")
        assert mids and all(m in idx for m in mids)
        assert [idx[m] for m in mids] == sorted({idx[m] for m in mids}), r["clip_id"]
        pos[r["split"]].update(idx[m] for m in mids)
        assert re.fullmatch(r"[A-Za-z0-9_-]{11}_[0-9]{7}", r["clip_id"])
        assert r["clip_id"] == f"{r['ytid']}_{round(float(r['start_s']) * 1000):07d}"
        assert r["rel_media"] == f"{r['split']}/{r['clip_id']}.mp4"
        assert 1.0 <= float(r["duration_s"]) <= 63.731
    assert pos["train"] == pos["eval"] == set(range(527))  # every class has train and eval positives


def test_trackless_clips_are_rows_and_a_recorded_fact(spec, rows, mm):
    """No clip is excluded: the 3 published mp4s without an audio track are manifest rows
    like every other; their keys are a recorded fact (make_manifest.py, the yaml's upstream note, setup.md), never an
    exclusion; no published clip lacks a video track."""
    keys = {(r["split"], r["clip_id"]) for r in rows}
    assert NO_AUDIO <= keys
    assert {(s, k) for s, ks in mm.NO_AUDIO_TRACK.items() for k in ks} == NO_AUDIO
    assert not any(mm.NO_VIDEO_TRACK.values())
    note = spec["manifest"]["upstream"]["note"]
    assert "no clip is excluded" in note and "except" not in note
    text = (HERE / "setup.md").read_text(encoding="utf-8")
    setup = {h: F.section(text, 2, h) for h in ("Input contract", "Gotchas")}
    for split, key in NO_AUDIO:
        assert f"{split}/{key}" in note and f"{split}/{key}" in setup["Input contract"] and f"{split}/{key}" in setup["Gotchas"]
    assert "has_track" in setup["Input contract"] and "describe()" in setup["Input contract"]
    text = (HERE / "make_manifest.py").read_text(encoding="utf-8")
    assert "excluded_keys" not in text and "NO_AUDIO_TRACK" in text


def test_curation_sums(spec, rows, mm):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"] == 22160 + 20371
    for split, v in cur["per_split"].items():
        assert v["listed"] == v["staged"] == sum(r["split"] == split for r in rows)
    assert all(v["missing"] == 0 for v in cur["per_split"].values())
    assert cur["excluded"] == {"train_unavailable": 4434, "eval_unavailable": 4148}  # MISSING.csv only; no clip is excluded
    assert {s: v["listed"] for s, v in cur["per_split"].items()} == ROWS == mm.PUBLISHED
    # published = frozen; published + unavailable = official per split
    assert ROWS["train"] + 4434 == 22160 and ROWS["eval"] + 4148 == 20371
    assert cur["short_clips"] == sum(float(r["duration_s"]) < 9.9 for r in rows) == 2463
    assert cur["leakage_audit"]["excluded"] == 0


def test_metric_names_resolve(spec):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["metrics"]["headline"] == "map"


def test_chance_is_mean_eval_prevalence(spec, rows):
    n_eval = sum(r["split"] == "eval" for r in rows)
    inst = sum(len(r["labels"].split(";")) for r in rows if r["split"] == "eval")
    assert inst == 41508
    assert round(inst / (527 * n_eval), 6) == spec["metrics"]["chance"]["map"] == 0.004855


def test_budget_media_gb_is_the_frozen_members(spec):
    assert spec["budget"]["media_gb"] == round(66013790920 / 1e9, 1)


# ----------------------------------------------------------------------------- docs


def test_readme_facts():
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    proto = F.section(text, 3, "Protocol")
    for fact in ("0.004855", "16 223", "17 726", "`map`", "protocol_version: 1"):
        assert fact in proto, fact
    refreeze = F.section(text, 3, "Re-freezing the manifest")
    for name in ("make_manifest.py", "clips.csv", "MISSING.csv", "index.json", "SHA256SUMS", "--check"):
        assert name in refreeze, name


def test_setup_obtain_is_the_pinned_download(spec):
    """"## Obtain the data" is the pinned download (``source.revision`` in the three commands, lint checks it is there)."""
    text = (HERE / "setup.md").read_text(encoding="utf-8")
    obtain = F.section(text, 2, "Obtain the data")
    assert re.search(rf"hf download {re.escape(REPO_ID)} --repo-type dataset --revision {REVISION} --local-dir \S*audioset20k-wds-v1", obtain)
    assert "sha256sum -c SHA256SUMS" in obtain and "yavih bench verify audioset20k" in obtain
    assert HF_URL in obtain
    script = (HERE / "prepare" / "01_download.sh").read_text(encoding="utf-8")
    assert f"REVISION={REVISION}" in script and "hf download 11hu83/audioset20k --repo-type dataset" in script
    verify = (HERE / "prepare" / "02_verify.sh").read_text(encoding="utf-8")
    assert "sha256sum -c" in verify and "bench verify audioset20k --census" in verify
    for name, sha in spec["source"]["sha256"].items():
        assert f"pin {name}" in verify and sha in verify, name
        assert sha in obtain, name


def test_benchmarks_index_row(spec):
    idx = HERE.parent / "README.md"
    rows_ = [ln for ln in idx.read_text(encoding="utf-8").splitlines() if ln.startswith("| [audioset20k]")]
    assert rows_, "benchmarks/README.md index has no audioset20k row"
    cells = [c.strip() for c in rows_[0].strip("|").split("|")]
    assert cells[0] == "[audioset20k](audioset20k/README.md)" and cells[2] == spec["readout"]
    assert "public" in cells[4].lower() and cells[3] == "video + audio"  # generated from benchmark.yaml streams


# ----------------------------------------------------------------------------- bridge template + output contract


def _template():
    return F.load_module(HERE / "bridge_template.py")


def test_bridge_template_declares_model_and_geometry():
    mod = _template()
    assert mod.VIDEO_TEMPORAL == {"rule": "uniform", "n": 8, "span": [0.0, 8.0], "edge": "clamp"}
    assert mod.AUDIO_TEMPORAL == {"rule": "span", "start_s": 0.0, "length_s": 8.0, "pad": "zeros"}
    assert mod.AudioSet20kBridge.pos_mode == "per_sample" and mod.AudioSet20kBridge.streams == ("video", "audio")
    assert "decode_plan" not in vars(mod.AudioSet20kBridge)
    src = (HERE / "bridge_template.py").read_text(encoding="utf-8")
    assert '"model": "TODO-model",' in src


def _check_output_contract(bridge, samples, feats, pos, used):
    """The member-level checks validate applies: fixed N/D, pos range, used vs the declared rule,
    token centres inside the declared crop region, audio ranges inside the media."""
    from yavih.media.rules import audio_ranges, crop_region, rule_times

    feats = np.asarray(feats)
    assert feats.ndim == 3 and feats.shape[0] == len(samples) and np.isfinite(feats.astype(np.float32)).all()
    assert pos.shape == feats.shape[:2] + (4,) and pos.dtype == np.float32
    ok = ((pos[..., 1:] >= 0) & (pos[..., 1:] <= 1)) | (pos[..., 1:] == -1)
    assert ok.all()
    d = bridge.describe()
    for s, p, u in zip(samples, pos, used, strict=True):
        assert set(u) == set(d["input"])
        for name, entry in d["input"].items():
            info = s.streams[name].info
            tag = d["streams"].index(name)
            if name == "video":
                got = np.asarray(u[name]["times"])
                assert ((got >= -1e-3) & (got <= info["duration_s"] + 1e-3)).all()  # E_TIMES_OUT_OF_MEDIA
                # E_RULE_MISMATCH as validate evaluates it: the rule's times, a clamped time capped at the last frame
                # that exists (duration_s − 1/fps), half a frame + 1 ms tolerance
                want = np.clip(rule_times(entry["temporal"], info), 0.0, info["duration_s"] - 1.0 / info["fps"])
                assert np.abs(got - want).max() <= 0.5 / info["fps"] + 1e-3
                y0, x0, y1, x1 = crop_region(entry["spatial"], info["width"], info["height"])
                yx = p[p[:, 0] == tag][:, 2:]
                assert ((yx[:, 0] >= y0 - 1e-6) & (yx[:, 0] <= y1 + 1e-6) & (yx[:, 1] >= x0 - 1e-6) & (yx[:, 1] <= x1 + 1e-6)).all()
            else:
                ranges, pad = audio_ranges(entry["temporal"], info)
                assert u[name]["samples"] == ranges.tolist() and u[name]["pad"] == list(pad)
                assert all(0 <= lo <= hi <= info["n_samples"] for lo, hi in u[name]["samples"])
    return feats.shape[1:]


# ----------------------------------------------------------------------------- reader smoke (synthetic tar)

SYN = {"train": {"video_s": 12.0, "audio_s": 10.0, "size": (96, 128)}, "eval": {"video_s": 5.0, "audio_s": 3.0, "size": (128, 96)}}


def _mp4(video_s: float, audio_s: float | None, hw: tuple[int, int], seed: int) -> bytes:
    """A small mp4 with one h264 video track (10 fps, each frame a solid grey = its index) and one AAC track
    (``audio_s=None``: no audio track at all, like the 3 published track-less clips)."""
    av = pytest.importorskip("av")
    h, w = hw
    buf = io.BytesIO()
    rng = np.random.default_rng(seed)
    with av.open(buf, mode="w", format="mp4") as c:
        vs = c.add_stream("libx264", rate=10)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        if audio_s is not None:
            ast = c.add_stream("aac", rate=16000)
            ast.layout = "mono"
        for i in range(int(video_s * 10)):
            for pkt in vs.encode(av.VideoFrame.from_ndarray(np.full((h, w, 3), (2 * i) % 256, np.uint8), format="rgb24")):
                c.mux(pkt)
        for pkt in vs.encode():
            c.mux(pkt)
        n = 0
        while audio_s is not None and n < int(audio_s * 16000):
            af = av.AudioFrame.from_ndarray(rng.uniform(-0.3, 0.3, (1, 1024)).astype(np.float32), format="fltp", layout="mono")
            af.sample_rate, af.pts = 16000, n
            for pkt in ast.encode(af):
                c.mux(pkt)
            n += 1024
        if audio_s is not None:
            for pkt in ast.encode():
                c.mux(pkt)
    return buf.getvalue()


def _write_synthetic_dataset(tmp_path: Path, rows) -> dict:
    """One WebDataset-style tar per split holding row 0 of that split (``<key>.mp4`` + ``<key>.json``) and a v1
    ``index.json`` with relative shard names, as the published dataset is laid out. Returns the stored bytes."""
    import tarfile

    first = {s: next(r for r in rows if r["split"] == s) for s in ("train", "eval")}
    stored: dict[str, bytes] = {}
    members: dict[str, list[int]] = {}
    shards = []
    for k, (split, r) in enumerate(first.items()):
        mp4 = _mp4(SYN[split]["video_s"], SYN[split]["audio_s"], SYN[split]["size"], k)
        stored[r["rel_media"]] = mp4
        shard = f"audioset20k-{split}-000000.tar"
        shards.append(shard)
        with tarfile.open(tmp_path / shard, "w", format=tarfile.USTAR_FORMAT) as tf:
            for name, data in ((f"{r['clip_id']}.mp4", mp4), (f"{r['clip_id']}.json", json.dumps({"key": r["clip_id"]}).encode())):
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        with tarfile.open(tmp_path / shard) as tf:
            m = tf.getmember(f"{r['clip_id']}.mp4")
            members[r["rel_media"]] = [k, m.offset_data, m.size]
    (tmp_path / "index.json").write_text(json.dumps({"version": 1, "shards": shards, "members": members}), encoding="utf-8")
    return {"first": first, "stored": stored}


def test_reader_smoke_on_synthetic_tar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows, spec):
    """Reader smoke on synthetic raw files: one mp4 member handed over as ``video`` and ``audio``
    (the same bytes, read in place), each stream's info describes its own track, meta holds only facts, a row
    whose media is absent raises MediaMissing; then the template encodes the samples in every reading and
    satisfies the output contract (fixed N/D, used vs the declared rules, batch invariance)."""
    pytest.importorskip("av")
    syn = _write_synthetic_dataset(tmp_path, rows)
    monkeypatch.setenv("YAVIH_DATA__AUDIOSET20K__MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "home"))  # no site census: info comes from a live header probe
    from yavih import Benchmark
    from yavih.media import MediaMissing

    b = Benchmark.from_dir(HERE)
    assert b.resolved_roots()["media_root"] == str(tmp_path)
    samples = {}
    for split in ("train", "eval"):
        r = syn["first"][split]
        s = next(b.reader(split, rows=range(0, 1)))
        samples[split] = s
        assert (s.row, s.clip_id, s.split) == (0, r["clip_id"], split)
        assert set(s.streams) == {"video", "audio"} and s.meta == {"facts": {}}
        vi, au = s.streams["video"], s.streams["audio"]
        assert vi.kind == "video" and au.kind == "audio"
        assert vi.path is None and au.path is None and vi.member is not None and vi.member == au.member
        assert vi.member[0].endswith(f"audioset20k-{split}-000000.tar")
        assert vi.bytes() == au.bytes() == syn["stored"][r["rel_media"]] and vi.open().read() == vi.bytes()
        for name in ("video", "audio"):
            for k in spec["input_contract"]["info"][name]:
                assert s.streams[name].info[k] is not None, (name, k)
        h, w = SYN[split]["size"]
        assert (vi.info["width"], vi.info["height"], vi.info["fps"], vi.info["container"]) == (w, h, 10.0, "mp4")
        assert vi.info["duration_s"] == pytest.approx(SYN[split]["video_s"], abs=0.11)
        assert vi.info["n_frames"] == int(SYN[split]["video_s"] * 10)
        assert au.info["sample_rate"] == 16000 and au.info["channels"] == 1
        assert au.info["duration_s"] == pytest.approx(SYN[split]["audio_s"], abs=0.1)  # its own track, not the video's
    with pytest.raises(MediaMissing):  # row 1 of train is not in the synthetic tar → before any bridge sees it
        s1 = next(b.reader("train", rows=range(1, 2)))
        s1.streams["video"].bytes()

    pytest.importorskip("yavih.media.decode")
    mod = _template()
    view = _view(b)
    from yavih.bridge import validate_describe

    for reading in ("av", "a", "v"):
        br = mod.AudioSet20kBridge(view, {"reading": reading})
        d = br.describe()
        validate_describe(d, br, view)
        assert d["model"] == "TODO-model" and d["reading"] == reading
        ND = None
        for split in ("train", "eval"):
            s = samples[split]
            feats, pos, used = br.encode([s])
            nd = _check_output_contract(br, [s], feats, pos, used)
            ND = ND or nd
            assert nd == ND  # fixed (N, D) across splits
            if "video" in br.streams:
                assert len(used[0]["video"]["times"]) == mod.N_FRAMES
                if split == "eval":  # 5 s of video: 5.5, 6.5, 7.5 s clamp to the clip's end (edge: clamp)
                    assert max(used[0]["video"]["times"]) <= s.streams["video"].info["duration_s"]
            if "audio" in br.streams:
                n = s.streams["audio"].info["n_samples"]
                assert used[0]["audio"] == {"samples": [[0, min(n, 128000)]], "pad": [0, max(0, 128000 - n)]}
                t = pos[0][pos[0][:, 0] == br.streams.index("audio")][:, 1]
                if split == "eval":  # 3 s of audio fed as 8 s: buckets centred beyond 3 s lie in the padding
                    assert (t == -1).sum() == 5 and (t[t >= 0] <= 1.0).all()
                else:
                    assert (t >= 0).all()
        # batch invariance (bridge check's alone pass): a batch of both equals each alone
        f2, p2, u2 = br.encode([samples["train"], samples["eval"]])
        fa, pa, ua = br.encode([samples["eval"]])
        assert np.array_equal(f2[1], fa[0]) and np.array_equal(p2[1], pa[0]) and u2[1] == ua[0]


# ----------------------------------------------------------------------------- a track-less clip end to end (synthetic tar)


class _Findings:
    """What ``yavih.io.validate.check_member`` reports through ``add(code, message, where)``."""

    def __init__(self) -> None:
        self.found: list[tuple] = []

    def add(self, code: str, message: str, where: str | None = None) -> None:
        self.found.append((code, message, where))


def _toy_audio_bridge():
    """A toy ``a`` bridge: the first 2 s of audio (span rule, zero-padded), downmixed, as
    ``[log power, mean |x|, fed seconds, track flag]``; for a clip whose mp4 holds no audio track
    (``info["has_track"] is False``) it feeds its own declared choice — 2 s of silence — and reports an empty
    consumption. One fixed-shape row per clip either way."""
    from yavih.bridge import Bridge
    from yavih.media import decode, rules

    temporal = {"rule": "span", "start_s": 0.0, "length_s": 2.0, "pad": "zeros"}

    class ToyAudio(Bridge):
        name = "toy-audio"
        batch_size = 2
        dtype = "float32"
        pos_mode = "none"
        streams = ()

        def describe(self) -> dict:
            return {
                "model": "toy-audio", "reading": "a", "weights_sha256": "0" * 64, "weights_ref": "none",
                "dtype": self.dtype, "batch_size": self.batch_size, "streams": [], "pos_mode": self.pos_mode,
                "layer": 0, "pooling": "mean", "views": 1,
                "input": {"audio": {"temporal": temporal,
                                    "front_end": {"sample_rate": 16000, "resample": "none", "features": "log power, mean |x|", "channels": "mean"},
                                    "norm": {"mean": 0.0, "std": 1.0, "source": "toy: none"},
                                    "decoder": decode.decoder_info("audio")}},
                "no_track": {"audio": "2 s of zeros through the same features; used samples [] pad [0, 0]"},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
                "args": {},
            }

        def encode(self, samples: list):
            feats, used = [], []
            for smp in samples:
                a = smp.streams["audio"]
                if a.info["has_track"] is False:  # the bridge's declared choice for a clip without an audio track
                    x, u = np.zeros(32000, np.float32), {"samples": [], "pad": [0, 0]}
                else:
                    ranges, (pl, pr) = rules.audio_ranges(temporal, a.info)
                    wav, _ = decode.audio(a, ranges)
                    want = int(sum(int(hi) - int(lo) for lo, hi in ranges))
                    x = wav.mean(axis=1)[:want]
                    x = np.concatenate([np.zeros(int(pl), np.float32), x, np.zeros(want - len(x) + int(pr), np.float32)])
                    u = {"samples": [[int(lo), int(hi)] for lo, hi in ranges], "pad": [int(pl), int(pr)]}
                track = 0.0 if a.info["has_track"] is False else 1.0
                f = [np.log(np.mean(x.astype(np.float64) ** 2) + 1e-10), float(np.abs(x).mean()), len(x) / 16000.0, track]
                feats.append(np.asarray([f], np.float32))
                used.append({"audio": u})
            return np.stack(feats), None, used

    return ToyAudio


def _small_copy(tmp_path: Path, spec: dict, picked: list[dict]) -> Path:
    """This folder with a manifest of the ``picked`` rows only (in manifest order), so a run touches a few synthetic
    members; everything else (reader, contracts, streams) is the folder's own."""
    import shutil

    folder = tmp_path / "audioset20k"
    shutil.copytree(HERE, folder, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    header = ["clip_id", "split", "labels", "rel_media", "ytid", "start_s", "duration_s"]
    w.writerow(header)
    for r in picked:
        w.writerow([r[h] for h in header])
    raw = out.getvalue().encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(raw)
    (folder / "manifest.csv.gz").write_bytes(buf.getvalue())
    y = json.loads(json.dumps(spec))
    y["manifest"]["sha256"] = hashlib.sha256(raw).hexdigest()
    y["manifest"]["upstream"] = None
    y["budget"]["clips"] = {s: sum(r["split"] == s for r in picked) for s in ("train", "eval")}
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return folder


def test_trackless_clip_reader_bridge_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows, spec):
    """No clip is excluded, end to end on a synthetic tar: a published-style mp4
    WITHOUT an audio track is a row; the reader hands over the same mp4 as ``video`` and ``audio`` (nothing
    missing, nothing raised), the audio stream's info says ``has_track: false`` with the track facts null; a toy
    audio bridge emits its declared silence row and reports an empty used range; ``run_extract`` writes one
    member per row (coverage 100 %) and ``validate`` accepts it (no error, no unchecked-rule warning). The
    folder's template also serves the row in every reading."""
    pytest.importorskip("av")
    pytest.importorskip("yavih.media.decode")
    import tarfile

    from yavih import Benchmark
    from yavih.extract import run_extract
    from yavih.io.validate import check_member, validate_features
    from yavih.sample import TRACK_FACTS

    by_key = {(r["split"], r["clip_id"]): r for r in rows}
    picked = [next(r for r in rows if r["split"] == "train"), by_key[("train", "NlBQ4kLpdzk_0260000")],
              next(r for r in rows if r["split"] == "eval"), by_key[("eval", "chc33hy6iIU_0510000")]]
    folder = _small_copy(tmp_path, spec, picked)
    media = tmp_path / "media"
    media.mkdir()
    members, shards, stored = {}, [], {}
    for k, split in enumerate(("train", "eval")):
        shard = f"audioset20k-{split}-000000.tar"
        shards.append(shard)
        with tarfile.open(media / shard, "w", format=tarfile.USTAR_FORMAT) as tf:
            for j, r in enumerate(x for x in picked if x["split"] == split):
                trackless = (split, r["clip_id"]) in NO_AUDIO
                data = _mp4(3.0, None if trackless else 2.5, (48, 64), 10 * k + j)
                stored[r["rel_media"]] = data
                ti = tarfile.TarInfo(f"{r['clip_id']}.mp4")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        with tarfile.open(media / shard) as tf:
            for r in (x for x in picked if x["split"] == split):
                m = tf.getmember(f"{r['clip_id']}.mp4")
                members[r["rel_media"]] = [k, m.offset_data, m.size]
    (media / "index.json").write_text(json.dumps({"version": 1, "shards": shards, "members": members}), encoding="utf-8")
    monkeypatch.setenv("YAVIH_DATA__AUDIOSET20K__MEDIA_ROOT", str(media))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "home"))  # no census: info from a live header probe

    b = Benchmark.from_dir(folder)
    # ---- reader: the track-less row is handed over whole, with the fact said explicitly
    for split in ("train", "eval"):
        samples = list(b.reader(split))
        assert [s.clip_id for s in samples] == [r["clip_id"] for r in picked if r["split"] == split]
        for s in samples:
            vi, au = s.streams["video"], s.streams["audio"]
            assert set(s.streams) == {"video", "audio"} and vi.member == au.member
            assert vi.bytes() == au.bytes() == stored[f"{split}/{s.clip_id}.mp4"]
            assert vi.info["has_track"] is True and vi.info["n_frames"] == 30
            if (split, s.clip_id) in NO_AUDIO:
                assert au.info["has_track"] is False and au.info["container"] == "mp4" and au.info["bytes"] == len(au.bytes())
                assert all(au.info[k] is None for k in TRACK_FACTS["audio"])
            else:
                assert au.info["has_track"] is True and au.info["sample_rate"] == 16000 and au.info["n_samples"] > 0

    # ---- toy audio bridge → run_extract → validate
    bridge = _toy_audio_bridge()
    for split in ("train", "eval"):
        res = run_extract(bench=b, bridge=bridge, split=split, shard=(0, 1), out=tmp_path / "run", canaries=1)
        assert res["status"] == "ok", res
        sd = tmp_path / "run" / "features" / split
        metas = {}
        for tar in sd.glob("shard-*.tar"):
            with tarfile.open(tar) as tf:
                for m in tf.getmembers():
                    if m.name.endswith(".json"):
                        d = json.loads(tf.extractfile(m).read())
                        metas.setdefault(d.get("row", m.name), d)
        rep = validate_features(sd, bench=b)
        assert rep.ok, [(f.code, f.message) for f in rep.errors]
        assert not [f for f in rep.warnings if f.code in ("W_RULE_UNCHECKED", "W_COVERAGE")], [(f.code, f.message) for f in rep.warnings]
        assert rep.coverage["present"] == rep.coverage["nominal"] == 2 and rep.N == 1
        trackless = [d for d in metas.values() if d["media"]["audio"]["has_track"] is False]
        assert len(trackless) >= 1 and all(d["used"]["audio"] == {"samples": [], "pad": [0, 0]} for d in trackless)

    # ---- the folder template serves the track-less row in every reading (one fixed-shape row, empty audio report)
    mod = _template()
    view = _view(b)
    s_eval = list(b.reader("eval"))
    kinds = {"video": "video", "audio": "audio"}
    for reading in ("a", "av", "v"):
        br = mod.AudioSet20kBridge(view, {"reading": reading})
        d = br.describe()
        assert set(d["no_track"]) == set(br.streams)
        feats, pos, used = br.encode(s_eval)
        assert feats.shape[0] == 2 and len(used) == 2
        for s, p, u in zip(s_eval, pos, used, strict=True):
            fx = _Findings()
            media_block = {n: st.info for n, st in s.streams.items()}
            check_member({"media": media_block, "used": u}, p, describe=d, kinds=kinds, where=s.clip_id, fx=fx)
            assert fx.found == [], fx.found
            if ("eval", s.clip_id) in NO_AUDIO and "audio" in br.streams:
                assert u["audio"] == {"samples": [], "pad": [0, 0]}
                assert (p[p[:, 0] == br.streams.index("audio")][:, 1] == -1).all()


# ----------------------------------------------------------------------------- real media (skips off-site)


def _media_root(bench) -> Path:
    root = F.site_root(HERE, "media_root")  # None: no data root configured or nothing staged here
    if root is None:
        pytest.skip("audioset20k media_root not configured or not present on this host")
    return root


def test_reader_smoke_if_media_present(bench, spec):
    root = _media_root(bench)
    index = root / spec["data"]["tar_index"]
    if not index.is_file():
        pytest.skip(f"dataset index not present at {index}")
    shards = json.loads(index.read_text(encoding="utf-8"))["shards"]
    if not all(os.path.isfile(root / p) for p in shards):
        pytest.skip("dataset tars not reachable from this host")
    pytest.importorskip("av")
    mod = _template()
    br = mod.AudioSet20kBridge(_view(bench), {"reading": "av"})
    ND = None
    for split in ("train", "eval"):
        samples = list(bench.reader(split, rows=range(0, 2)))
        assert [s.row for s in samples] == [0, 1]
        for s in samples:
            vi, au = s.streams["video"], s.streams["audio"]
            assert vi.member == au.member and vi.bytes()[4:8] == b"ftyp"
            assert vi.info["container"] == au.info["container"] == "mp4"
            assert vi.info["n_frames"] and vi.info["fps"] and au.info["sample_rate"] in (32000, 44100, 48000)
            assert s.meta == {"facts": {}}
        feats, pos, used = br.encode(samples)
        nd = _check_output_contract(br, samples, feats, pos, used)
        ND = ND or nd
        assert nd == ND


def test_make_manifest_check_if_media_present(bench):
    """Byte-identical re-freeze from the published clips.csv; skips without the downloaded dataset."""
    root = _media_root(bench)
    if not (root / "clips.csv").is_file():
        pytest.skip(f"dataset clips.csv not present at {root}")
    mod = F.load_module(HERE / "make_manifest.py", "audioset20k_make_manifest_check")
    assert mod.main(["--media-root", str(root), "--check"]) == 0


# ----------------------------------------------------------------------------- core load / hash


def test_core_load_and_hash(bench, spec):
    from yavih import Benchmark

    assert bench.name == "audioset20k" and list(bench.spec.aliases) == []
    assert bench.spec.protocol_version == 1 and bench.unknown_labels == 0
    assert bench.spec.source.revision == REVISION
    assert bench.manifest_hash == "8ada78f8fa94f980582500498d5011542bf7b45bfb0646ffb9d8b9187d8b8b2d"
    assert len(bench.protocol_hash) == 64
    assert Benchmark.load("audioset20k").protocol_hash == bench.protocol_hash == spec["changelog"][-1]["protocol_hash"]
    assert bench.input_hash == spec["changelog"][-1]["input_hash"]
    view = _view(bench)
    assert view.streams == {"video": "video", "audio": "audio"} and view.facts == {}
