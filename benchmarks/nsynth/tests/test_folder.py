"""Folder tests of nsynth (run: ``python -m pytest "$(yavih bench path nsynth)/tests" -q -p no:cacheprovider``).

Three groups:

* the frozen manifest's facts (HEAR's 5h draw, the vocabulary, the instrument split, the curation sums) and the
  copied HEAR code (``make_manifest.py``'s subsample against pandas, ``eval.py``'s metrics on hand-checked inputs),
  which run anywhere;
* ``make_manifest.py`` on a tiny synthetic upstream and the reader on synthetic wavs, which run anywhere;
* ``make_manifest.py --check`` on the staged ``examples.json`` files and the reader on the staged wavs, which skip
  when no NSynth root is configured.

``yavih bench lint`` (run once per folder by the repository) checks the layout, the yaml, the hashes, the
conformance fixture and the docs' structure; this file does not repeat it.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows = F.spec, F.rows

READOUT = "clf"
HEADLINE = "pitch_acc"
STREAMS = {"audio": "audio"}
CLIPS = {"train": 4000, "test": 500}
MANIFEST_SPLITS = {"train": 4000, "valid": 500, "test": 500}
READINGS = ["a"]
EXTERNAL: list[str] = []
HAS_EVAL = True
PITCHES = list(range(21, 109))
CLIP_SAMPLES, NATIVE_SR, WAV_BYTES = 64000, 16000, 128044
INFO_KEYS = ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No site config, no site census, no media-root override."""
    monkeypatch.setenv("YAVIH_CONFIG", str(tmp_path / "_no_such_config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "yavih_home"))
    monkeypatch.delenv("YAVIH_DATA__NSYNTH__MEDIA_ROOT", raising=False)
    return tmp_path


def _mm():
    return F.load_module(HERE / "make_manifest.py")


def _ev():
    return F.load_module(HERE / "eval.py")


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert dict(counts) == MANIFEST_SPLITS
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]
    assert spec["splits"]["train"] == ["train"] and spec["splits"]["eval"] == ["test"] and spec["splits"]["unused"] == ["valid"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)
    assert [e["name"] for e in spec["external"] or []] == EXTERNAL
    assert (HERE / "eval.py").is_file() is HAS_EVAL
    mm = _mm()
    assert spec["protocol"]["train_draw"] == {"cap": 4000, "seed": mm.str2int("SubsampleSplit(train)"), "balance": None}
    assert spec["protocol"]["clip_seconds"] == mm.generic_task_config["sample_duration"] == 4.0


# ----------------------------------------------------------------------------- the frozen manifest's facts


def test_classes_are_the_88_piano_pitches():
    """HEAR's labelvocabulary: the sorted distinct pitches, all 88 present, so class index = pitch - 21 and
    heareval's ``index % 12`` is the pitch class."""
    lines = (HERE / "classes.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "index,name"
    assert [tuple(x.split(",")) for x in lines[1:]] == [(str(i), str(p)) for i, p in enumerate(PITCHES)]


def test_rows_match_their_note_names(rows: list[dict[str, str]]):
    """``<instrument>-<pitch>-<velocity>``: the note name agrees with the label and the extras."""
    for r in rows:
        m = re.fullmatch(r"([a-z_]+_\d{3})-(\d{3})-(\d{3})", r["clip_id"])
        assert m, r["clip_id"]
        assert m.group(1) == r["instrument_str"] and int(m.group(2)) == int(r["labels"]) and int(m.group(3)) == int(r["velocity"])
        assert r["instrument_str"].startswith(r["instrument_family_str"] + "_")
        assert r["instrument_source_str"] in {"acoustic", "electronic", "synthetic"}
        assert r["rel_path"] == f"nsynth-{r['split']}/audio/{r['clip_id']}.wav"
        assert 21 <= int(r["labels"]) <= 108


def test_every_split_has_every_pitch(rows: list[dict[str, str]]):
    for s in MANIFEST_SPLITS:
        assert sorted({int(r["labels"]) for r in rows if r["split"] == s}) == PITCHES, s


def test_instrument_split(rows: list[dict[str, str]]):
    """Train shares no instrument with valid or test; valid and test draw from the same instruments (why valid is
    unused)."""
    inst = {s: {r["instrument_str"] for r in rows if r["split"] == s} for s in MANIFEST_SPLITS}
    assert not inst["train"] & inst["test"] and not inst["train"] & inst["valid"]
    assert inst["test"] & inst["valid"]


def test_row_order(rows: list[dict[str, str]]):
    """Split by split (train, valid, test), note-name order within a split."""
    assert [r["split"] for r in rows] == ["train"] * 4000 + ["valid"] * 500 + ["test"] * 500
    for s in MANIFEST_SPLITS:
        ids = [r["clip_id"] for r in rows if r["split"] == s]
        assert ids == sorted(ids)


def test_curation_sums(spec: dict):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"] == 305979
    assert {s: v["listed"] for s, v in cur["per_split"].items()} == MANIFEST_SPLITS
    mm = _mm()
    assert set(cur["excluded"]) == {mm.EXCLUDED_PITCH, mm.EXCLUDED_DRAW}
    assert spec["manifest"]["upstream"]["upstream_files"] == {f"nsynth-{s}/examples.json": d for s, d in mm.EXAMPLES_SHA256.items()}


# ----------------------------------------------------------------------------- the copied HEAR selection


def test_hear_constants():
    mm = _mm()
    assert {s: mm.max_files_of(s) for s in mm.SPLITS} == {"train": 4000, "valid": 500, "test": 500}
    assert mm.str2int("SubsampleSplit(train)") == 3488948419
    assert mm.str2int("SubsampleSplit(valid)") == 3511819133
    assert mm.str2int("SubsampleSplit(test)") == 2920447107
    recs = [{"pitch": p} for p in (20, 21, 60, 108, 109)]
    assert [r["pitch"] for r in mm.pitch_filter(recs)] == [21, 60, 108]


def test_subsample_split_equals_hears_pandas_call():
    """``subsample_split`` is ``SubsampleSplit.run``'s pandas selection on a synthetic population."""
    pd = pytest.importorskip("pandas")
    mm = _mm()
    rng = np.random.default_rng(0)
    names = [f"inst_{rng.integers(0, 50):03d}-{rng.integers(21, 109):03d}-{v:03d}" for v in rng.integers(0, 128, 3000)]
    names = list(dict.fromkeys(names))
    for split, k in (("train", 400), ("valid", 50), ("test", 10_000)):
        df = pd.DataFrame({"unique_filestem": names, "relpath": [f"/x/{n}.wav" for n in names]})
        # hearpreprocess pipeline.py SubsampleSplit.run, verbatim but for the frame built above
        split_filestem_relpaths = (
            df[["unique_filestem", "relpath"]].drop_duplicates().sort_values("unique_filestem")
        )
        split_filestem_relpaths = split_filestem_relpaths.sample(
            frac=1, random_state=mm.str2int(f"SubsampleSplit({split})")
        ).values
        hear = [stem for stem, _ in split_filestem_relpaths[:k]]
        assert mm.subsample_split(names[::-1], split, k) == hear


# ----------------------------------------------------------------------------- eval.py: HEAR's metrics


def _onehot(idx, c=88):
    return np.eye(c)[np.asarray(idx)]


def test_metrics_hand_checked():
    ev = _ev()
    targets = np.array([39, 39, 39, 0, 87])               # pitches 60, 60, 60, 21, 108
    outputs = _onehot([39, 51, 40, 12, 75])               # exact, +1 octave, +1 semitone, +1 octave, -1 octave
    assert ev.pitch_acc(outputs, targets, {}, {}) == pytest.approx(1 / 5)
    assert ev.chroma_acc(outputs, targets, {}, {}) == pytest.approx(4 / 5)
    # the copied heareval classes on one-hot targets give the same numbers
    assert ev.Top1Accuracy({}, name="pitch_acc")(outputs, _onehot(targets)) == pytest.approx(1 / 5)
    assert ev.ChromaAccuracy({})(outputs, _onehot(targets)) == pytest.approx(4 / 5)
    # per-clip quantities (the binomial interval's unit) average to the metrics
    assert ev.METRICS["pitch_acc"].per_clip(outputs, targets, {}, {}).mean() == pytest.approx(1 / 5)
    assert ev.METRICS["chroma_acc"].per_clip(outputs, targets, {}, {}).mean() == pytest.approx(4 / 5)


def test_metrics_on_logits_and_floor():
    ev = _ev()
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 88)).astype(np.float32)
    targets = rng.integers(0, 88, 200)
    pred = logits.argmax(1)
    assert ev.pitch_acc(logits, targets, {}, {}) == pytest.approx(np.mean(pred == targets))
    assert ev.chroma_acc(logits, targets, {}, {}) == pytest.approx(np.mean(pred % 12 == targets % 12))
    floor = np.zeros((200, 88), np.float32)
    floor[:, 35] = 1.0                                    # the clf readout's floor: one-hot of the train majority
    assert ev.pitch_acc_floor(floor, targets, {}, {}) == pytest.approx(np.mean(targets == 35))
    assert ev.chroma_acc_floor(floor, targets, {}, {}) == pytest.approx(np.mean(targets % 12 == 35 % 12))
    with pytest.raises(ValueError):
        ev.pitch_acc(logits[:3], targets, {}, {})


def test_floor_values_in_notes(rows: list[dict[str, str]]):
    """The floor values metrics.notes quotes, from the frozen manifest."""
    ev = _ev()
    train = Counter(int(r["labels"]) - 21 for r in rows if r["split"] == "train")
    major = max(sorted(train), key=lambda c: train[c])
    test = np.array([int(r["labels"]) - 21 for r in rows if r["split"] == "test"])
    floor = _onehot(np.full(test.size, major))
    assert major + 21 == 56
    assert ev.pitch_acc_floor(floor, test, {}, {}) == pytest.approx(0.008)
    assert ev.chroma_acc_floor(floor, test, {}, {}) == pytest.approx(0.092)


# ----------------------------------------------------------------------------- prepare/


def test_prepare_scripts_present(spec: dict):
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_download.sh", "02_extract.sh", "03_verify.sh", "04_make_manifest.sh"]
    assert all(re.match(r"^\d+_[A-Za-z0-9_.-]+\.(?:sh|py)$", n) for n in names), names
    src = spec["source"]["sha256"]
    dl = (HERE / "prepare" / "01_download.sh").read_text(encoding="utf-8")
    for s in ("train", "valid", "test"):
        assert f"[{s}]={src[f'nsynth-{s}.jsonwav.tar.gz']}" in dl
    ver = (HERE / "prepare" / "03_verify.sh").read_text(encoding="utf-8")
    assert f"want={src['audio.sha256sum']}" in ver
    for s in ("train", "valid", "test"):
        assert f"{src[f'nsynth-{s}/examples.json']}  nsynth-{s}/examples.json" in ver
    mk = (HERE / "prepare" / "04_make_manifest.sh").read_text(encoding="utf-8")
    assert spec["manifest"]["sha256"] in mk and spec["manifest"]["classes_sha256"] in mk


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values()
              if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    d = cls(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] in READINGS and set(d["input"]) <= set(STREAMS)


# ----------------------------------------------------------------------------- make_manifest.py on a synthetic upstream


def _write_examples(root: Path, split: str, notes: list[tuple[str, int, int]]) -> str:
    d = root / f"nsynth-{split}"
    d.mkdir(parents=True, exist_ok=True)
    data = {}
    for inst, pitch, vel in notes:
        name = f"{inst}-{pitch:03d}-{vel:03d}"
        family, source, _ = inst.rsplit("_", 2)
        data[name] = {"note_str": name, "pitch": pitch, "velocity": vel, "instrument_str": inst,
                      "instrument_family_str": family, "instrument_source_str": source, "sample_rate": 16000}
    p = d / "examples.json"
    p.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_make_manifest_on_synthetic_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """The freeze on a tiny upstream: pitch filter, the draw, the vocabulary, the audit, the curation sums, --check."""
    mm = _mm()
    up = tmp_path / "up"
    notes = {
        "train": [("bass_synthetic_001", p, 100) for p in PITCHES] + [("keyboard_acoustic_002", p, 50) for p in PITCHES]
        + [("bass_synthetic_001", 9, 100), ("bass_synthetic_001", 120, 100)],
        "valid": [("flute_acoustic_003", p, 25) for p in PITCHES] + [("flute_acoustic_003", 15, 25)],
        "test": [("flute_acoustic_003", p, 75) for p in PITCHES],
    }
    digests = {s: _write_examples(up, s, n) for s, n in notes.items()}
    monkeypatch.setattr(mm, "EXAMPLES_SHA256", digests)
    monkeypatch.setattr(mm, "UPSTREAM_ROWS", {s: len(n) for s, n in notes.items()})
    monkeypatch.setattr(mm, "TOTAL_UPSTREAM", sum(len(n) for n in notes.values()))
    monkeypatch.setattr(mm, "VALID_TEST_INSTRUMENTS", 1)
    caps = {"train": 176, "valid": 88, "test": 88}
    monkeypatch.setattr(mm, "max_files_of", lambda s, *a, **k: caps[s])
    out = tmp_path / "out"
    args = ["--no-media", "--metadata", str(up), "--out", str(out)]
    assert mm.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    with gzip.open(out / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        frozen = list(csv.DictReader(fh))
    assert Counter(r["split"] for r in frozen) == {"train": 176, "valid": 88, "test": 88}
    assert frozen[0] == {"clip_id": "bass_synthetic_001-021-100", "labels": "21", "split": "train",
                         "rel_path": "nsynth-train/audio/bass_synthetic_001-021-100.wav", "instrument_str": "bass_synthetic_001",
                         "instrument_family_str": "bass", "instrument_source_str": "synthetic", "velocity": "100"}
    cur = report["upstream"]["curation"]
    assert cur["excluded"] == {mm.EXCLUDED_PITCH: 3, mm.EXCLUDED_DRAW: 0}
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"]
    assert (out / "classes.csv").read_text(encoding="utf-8").splitlines()[1:3] == ["0,21", "1,22"]
    assert mm.main([*args, "--check"]) == 0
    # a draw that misses a pitch in a split is refused (HEAR's _diagnose_split_labels raises on it too)
    caps["train"] = 100
    with pytest.raises(SystemExit, match="misses pitches|not exactly 21..108"):
        mm.main(args)
    caps["train"] = 176
    (out / "classes.csv").write_text("index,name\n0,x\n", encoding="utf-8")
    assert mm.main([*args, "--check"]) == 1
    # a changed upstream file is refused (pinned sha256)
    (up / "nsynth-test" / "examples.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="sha256"):
        mm.main(args)


# ----------------------------------------------------------------------------- make_manifest.py --check on the staged files


META_ROOT = F.site_root(HERE, "media_root", needs=("nsynth-train/examples.json", "nsynth-valid/examples.json", "nsynth-test/examples.json"))


@pytest.mark.skipif(META_ROOT is None, reason="NSynth examples.json files not staged (yavih bench prepare nsynth 02)")
def test_make_manifest_check(capsys: pytest.CaptureFixture):
    """``--check`` re-derives manifest.csv.gz and classes.csv byte-identically from the public examples.json files
    and prints the committed ``manifest.upstream`` block."""
    mm = _mm()
    assert mm.main(["--no-media", "--metadata", str(META_ROOT), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    y = F.load_yaml(HERE)
    assert report["manifest"]["sha256"] == y["manifest"]["sha256"]
    assert report["manifest"]["classes_sha256"] == y["manifest"]["classes_sha256"]
    assert report["upstream"] == {k: v for k, v in y["manifest"]["upstream"].items() if k not in ("note", "draw")}
    assert report["derived"]["per_split"] == {"train": {"upstream": 289205, "in_range": 287073, "drawn": 4000},
                                              "valid": {"upstream": 12678, "in_range": 12566, "drawn": 500},
                                              "test": {"upstream": 4096, "in_range": 4060, "drawn": 500}}


# ----------------------------------------------------------------------------- reader smoke on synthetic raw files


def _write_wav(path: Path, n_frames: int, seed: int) -> None:
    sf = pytest.importorskip("soundfile")
    pcm = (np.random.default_rng(seed).uniform(-0.5, 0.5, n_frames) * 32767).astype(np.int16)
    sf.write(path, pcm, NATIVE_SR, subtype="PCM_16")


def test_reader_smoke_synthetic_raw(isolated: Path):
    """The reader hands over each stored wav byte for byte with its header facts and ``meta == {"facts": {}}``; a
    short file is handed over as stored (nothing cut, padded or refused); canaries equal their originals."""
    from yavih import Benchmark

    probe = Benchmark.from_dir(HERE)
    m = probe.manifest("test")
    media = isolated / "media"
    (media / "nsynth-test" / "audio").mkdir(parents=True)
    lengths = [CLIP_SAMPLES] * 8
    lengths[2] = 60000
    for i, n in enumerate(lengths):
        _write_wav(media / "nsynth-test" / "audio" / f"{m.clip_id[i]}.wav", n, i)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    samples = list(b.reader("test", rows=range(0, 8), batch_size=1, canaries=1))
    assert [s.row for s in samples[:8]] == list(range(8)) and len(samples) == 9
    for s, n in zip(samples[:8], lengths, strict=True):
        assert set(s.streams) == {"audio"} and s.meta == {"facts": {}} and s.split == "test"
        st = s.streams["audio"]
        f = media / "nsynth-test" / "audio" / f"{s.clip_id}.wav"
        assert st.kind == "audio" and Path(st.path).resolve() == f.resolve()
        assert st.bytes() == f.read_bytes()
        assert {k: st.info[k] for k in INFO_KEYS} == {
            "duration_s": pytest.approx(n / NATIVE_SR), "sample_rate": NATIVE_SR, "n_samples": n, "channels": 1,
            "container": "wav", "bytes": f.stat().st_size,
        }
    c = samples[8]
    assert c._canary_of == c.row and c.meta == samples[c.row].meta
    assert c.streams["audio"].bytes() == samples[c.row].streams["audio"].bytes()


def test_reader_to_template_end_to_end(isolated: Path):
    """The template consumes what the reader hands over and reports ``used`` inside the media."""
    from yavih import Benchmark
    from yavih.bridge import validate_describe

    probe = Benchmark.from_dir(HERE)
    m = probe.manifest("train")
    media = isolated / "media"
    (media / "nsynth-train" / "audio").mkdir(parents=True)
    for i in range(2):
        _write_wav(media / "nsynth-train" / "audio" / f"{m.clip_id[i]}.wav", CLIP_SAMPLES, 10 + i)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    mod = F.load_module(HERE / "bridge_template.py")
    bridge = mod.NsynthBridge(b.view(), {})
    validate_describe(bridge.describe(), bridge, b)
    samples = list(b.reader("train", rows=range(0, 2)))
    feats, pos, used = F.encode_prepared(bridge, samples)
    assert np.asarray(feats).shape[0] == 2 and pos is None
    assert [u["audio"] for u in used] == [{"samples": [[0, CLIP_SAMPLES]], "pad": [0, 0]}] * 2


# ----------------------------------------------------------------------------- reader smoke on the staged media


MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("nsynth-test/audio",))


@pytest.mark.skipif(MEDIA_ROOT is None, reason="NSynth media not staged on this machine")
def test_reader_smoke_real_media(isolated: Path):
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": str(MEDIA_ROOT)})
    samples = list(b.reader("test", rows=range(0, 4)))
    assert [s.row for s in samples] == [0, 1, 2, 3]
    for s in samples:
        st = s.streams["audio"]
        assert st.info["n_samples"] == CLIP_SAMPLES and st.info["sample_rate"] == NATIVE_SR
        assert st.info["channels"] == 1 and st.info["bytes"] == WAV_BYTES and st.info["duration_s"] == pytest.approx(4.0)
        raw = st.bytes()
        assert len(raw) == WAV_BYTES and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
        assert s.meta == {"facts": {}}
