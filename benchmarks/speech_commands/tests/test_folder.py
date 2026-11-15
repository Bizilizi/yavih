"""Folder tests of speech_commands (run: ``python -m pytest "$(yavih bench path speech_commands)/tests" -q -p no:cacheprovider``).

What nothing else checks: the HEAR task the manifest freezes (rows, labels, the 5 h draw, the silence slices), the
copied HEAR scorer and its glue on hand-checked inputs, the re-derivation of the manifest from a listing rebuilt from
the committed manifest, the bridge template and the generic reader on synthetic wavs. ``yavih bench lint`` (run once
per folder by the repository) checks the layout, the yaml, the hashes, the conformance fixture and the docs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows, classes = F.spec, F.rows, F.classes

READOUT = "clf"
HEADLINE = "top1_acc"
STREAMS = {"audio": "audio"}
CLIPS = {"train": 16000, "valid": 2000, "test": 4890}
UNUSED = ("train_rest", "valid_rest", "testing_list")
HEAR_CLASSES = ["_silence_", "_unknown_", "down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
INFO_KEYS = ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]
SR = 16000
#: sha256 of heareval/score.py lines 57-155 @ 855964977238e89dfc76394aa11c37010edb6f20 (hear-eval-kit v2021.1.3).
HEAR_SCORE_BLOCK_SHA256 = "35e49e951719208524b152adb4f1aed59cb376454d8b860b5d7f07d24508f9f3"


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No site config, no site census, no media-root override."""
    monkeypatch.setenv("YAVIH_CONFIG", str(tmp_path / "_no_such_config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "yavih_home"))
    monkeypatch.delenv("YAVIH_DATA__SPEECH_COMMANDS__MEDIA_ROOT", raising=False)
    return tmp_path


def _write_wav(path: Path, n_frames: int, seed: int) -> None:
    sf = pytest.importorskip("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.random.default_rng(seed).uniform(-0.5, 0.5, n_frames) * 32767).astype(np.int16)
    sf.write(path, pcm, SR, subtype="PCM_16")


def _eval_module():
    return F.load_module(HERE / "eval.py")


def _mm():
    return F.load_module(HERE / "make_manifest.py")


# ----------------------------------------------------------------------------- yaml + manifest


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]
    assert spec["splits"]["train"] == ["train", "valid"] and spec["splits"]["eval"] == ["test"]
    assert tuple(spec["splits"]["unused"]) == UNUSED and set(counts) == set(CLIPS) | set(UNUSED)


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE
    assert spec["metrics"]["floors"] == ["top1_acc_floor"] and spec["metrics"]["ci"] == {"top1_acc": "binomial"}


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS) and ic["info"]["audio"] == INFO_KEYS
    assert ic["facts"] == [] and ic["materialize"] is None
    assert spec["external"] == [] and (HERE / "eval.py").is_file() and (HERE / "hear_score.py").is_file()
    mm = _mm()
    assert spec["protocol"]["train_draw"] == {"cap": 18000, "seed": mm.str2int("SubsampleSplit(train)"), "balance": None}
    assert spec["protocol"]["clip_seconds"] == mm.SAMPLE_DURATION == 1.0
    sil = F.load_module(HERE / "prepare" / "02_hear_silence.py")
    assert spec["data"]["derived"] == {"hear_silence/SHA256SUMS": sil.EXPECTED_SUMS_SHA256}


def test_manifest_columns_and_labels(rows: list[dict[str, str]], classes: list[dict[str, str]]):
    assert list(rows[0]) == ["clip_id", "split", "label", "rel_path", "word", "speaker"]
    assert [c["name"] for c in classes] == HEAR_CLASSES and [int(c["index"]) for c in classes] == list(range(12))
    for r in rows:
        if r["rel_path"].startswith("hear_silence/"):
            folder, filename = "_silence_", r["rel_path"].split("/")[1]
        else:
            folder, filename = r["rel_path"].split("/")[-2:]
        prefix = "testing_list:" if r["split"] == "testing_list" else ""
        assert r["clip_id"] == f"{prefix}{folder}_{filename[:-4]}"   # HEAR's unique_filestem
        want = folder if folder in HEAR_CLASSES[2:] or folder == "_silence_" else "_unknown_"
        assert r["label"] == want, r
        assert r["word"] == folder
    for s in CLIPS:   # every class in every read split (HEAR asserts it)
        assert {r["label"] for r in rows if r["split"] == s} == set(HEAR_CLASSES)


def test_silence_rows(rows: list[dict[str, str]]):
    """789 HEAR silence slices: the running_tap ones are HEAR valid, the rest HEAR train; names from the pinned
    background sample counts (range(0, n - 16000, 8000))."""
    mm = _mm()
    sil = [r for r in rows if r["rel_path"].startswith("hear_silence/")]
    assert len(sil) == len(mm.silence_files()) == 789
    assert sorted(r["rel_path"].split("/")[1] for r in sil) == mm.silence_files()
    assert all(r["split"] in ("valid", "valid_rest") for r in sil if "running_tap" in r["clip_id"])
    assert all(r["split"] in ("train", "train_rest") for r in sil if "running_tap" not in r["clip_id"])
    assert all(r["label"] == "_silence_" and r["speaker"] == "" for r in sil)
    assert mm.silence_starts(16000 * 2) == [0, 8000] and mm.silence_starts(16001) == [0]


def test_curation_sums(spec: dict, rows: list[dict[str, str]]):
    cur = spec["manifest"]["upstream"]["curation"]
    counts = Counter(r["split"] for r in rows)
    assert {s: v["listed"] for s, v in cur["per_split"].items()} == dict(counts)
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"] == len(rows)
    assert cur["excluded"] == {}


def test_hear_draw_matches_numpy_permutation():
    """``subsample`` keeps HEAR's first int(max_duration / 1.0) of a RandomState(str2int(...)) permutation of the
    sorted filestems; pandas' ``sample(frac=1, random_state=seed)`` draws exactly that permutation."""
    mm = _mm()
    assert mm.str2int("SubsampleSplit(train)") == 3488948419 and mm.str2int("SubsampleSplit(valid)") == 3511819133
    assert int(mm.MAX_TASK_DURATION_BY_SPLIT["train"]) == 16000 and int(mm.MAX_TASK_DURATION_BY_SPLIT["valid"]) == 2000
    stems = [f"w_{i:05d}" for i in range(20000)]
    kept = mm.subsample(list(reversed(stems)), "train")
    perm = np.random.RandomState(3488948419).permutation(20000)
    assert kept == [stems[i] for i in perm[:16000]]
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"unique_filestem": stems}).sort_values("unique_filestem")
    assert list(df.sample(frac=1, random_state=mm.str2int("SubsampleSplit(train)")).values[:16000, 0]) == kept


# ----------------------------------------------------------------------------- the HEAR scorer


def test_hear_score_block_is_verbatim():
    """hear_score.py carries heareval/score.py lines 57-155 unchanged (the pinned sha256 of the upstream lines)."""
    text = (HERE / "hear_score.py").read_text()
    begin = text.index("\n", text.index("# ----- begin copied")) + 1
    block = text[begin:text.index("# ----- end copied -----")]
    assert hashlib.sha256(block.encode()).hexdigest() == HEAR_SCORE_BLOCK_SHA256


def test_top1_acc_hand_checked():
    ev = _eval_module()
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [5.0, 5.0, 0.0], [0.0, 0.0, 9.0]])
    targets = np.array([0, 2, 0, 2])
    # argmax 0, 1, 0 (first of a tie), 2 -> 3 of 4 right
    assert ev.top1_acc(logits, targets, {}, {}) == 0.75
    assert list(ev.top1_correct(logits, targets)) == [1.0, 0.0, 1.0, 1.0]
    p, onehot = ev.hear_inputs(logits, targets)
    assert np.allclose(p.sum(axis=1), 1.0) and (onehot.argmax(axis=1) == targets).all()
    assert ev.Top1Accuracy(label_to_idx={})(p, onehot) == 0.75
    from yavih.metrics.classification import acc

    rng = np.random.default_rng(0)
    o, t = rng.normal(size=(500, 12)), rng.integers(0, 12, 500)
    assert ev.top1_acc(o, t, {}, {}) == pytest.approx(acc(o, t, {}, {}))   # equals the core accuracy
    fl = np.zeros((4, 3))
    fl[:, 2] = 1.0   # the floor: one-hot logits of the train-majority class
    assert ev.top1_acc_floor(fl, targets, {}, {}) == 0.5
    with pytest.raises(ValueError):
        ev.top1_acc(np.zeros((0, 3)), np.zeros(0, np.int64), {}, {})


def test_metric_specs():
    ev = _eval_module()
    assert set(ev.METRICS) == {"top1_acc", "top1_acc_floor"}
    assert ev.METRICS["top1_acc_floor"].kind == "floor" and ev.METRICS["top1_acc_floor"].floor_of == ("top1_acc",)
    assert "binomial" in ev.METRICS["top1_acc"].ci and ev.METRICS["top1_acc"].per_clip is not None


# ----------------------------------------------------------------------------- prepare/


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_download.sh", "02_hear_silence.py", "03_verify.sh"]
    mm = _mm()
    dl = (HERE / "prepare" / "01_download.sh").read_text()
    for a in (mm.TRAIN_ARCHIVE, mm.TEST_ARCHIVE):
        assert a["url"] in dl and a["sha256"] in dl and a["md5"] in dl
    sil = F.load_module(HERE / "prepare" / "02_hear_silence.py")
    assert sil.N_SLICES == len(mm.silence_files())
    assert sorted(sil.BACKGROUND_NOISE_SHA256) == sorted(mm.BACKGROUND_NOISE)


# ----------------------------------------------------------------------------- make_manifest.py re-derivation


def _listing_from_manifest(rows: list[dict[str, str]]) -> bytes:
    """The listing CSV ``make_manifest.py --write-metadata`` writes from the staged archives, rebuilt from the
    committed manifest (every upstream wav is one row): the archives' word wavs and the two list files' lines."""
    out: dict[str, list[str]] = {"train_archive": [], "test_archive": [], "validation_list": [], "testing_list": []}
    for r in rows:
        top, rest = r["rel_path"].split("/", 1)
        if top == "test":
            out["test_archive"].append(rest)
        elif top == "train":
            out["train_archive"].append(rest)
            if r["split"] in ("valid", "valid_rest"):
                out["validation_list"].append(rest)
            elif r["split"] == "testing_list":
                out["testing_list"].append(rest)
    lines = ["source,path"] + [f"{s},{p}" for s in out for p in sorted(out[s])]
    return ("\n".join(lines) + "\n").encode()


def test_make_manifest_check(tmp_path: Path, rows: list[dict[str, str]], capsys: pytest.CaptureFixture):
    """``make_manifest.py --no-media --metadata <listing> --check`` re-derives manifest.csv.gz and classes.csv
    byte-identically and prints the committed ``manifest.upstream`` block; a changed listing is refused."""
    mm = _mm()
    listing = tmp_path / "listing.csv"
    listing.write_bytes(_listing_from_manifest(rows))
    assert mm.main(["--no-media", "--metadata", str(listing), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    y = F.load_yaml(HERE)
    assert report["manifest"]["sha256"] == y["manifest"]["sha256"] and report["manifest"]["classes_sha256"] == y["manifest"]["classes_sha256"]
    assert report["upstream"] == {k: v for k, v in y["manifest"]["upstream"].items() if k not in ("note", "draw")}
    assert hashlib.sha256(listing.read_bytes()).hexdigest() == y["manifest"]["upstream"]["sha256"]
    gz = tmp_path / "listing.csv.gz"
    gz.write_bytes(gzip.compress(listing.read_bytes()))
    assert mm.main(["--no-media", "--metadata", str(gz), "--check"]) == 0
    capsys.readouterr()
    text = listing.read_text().splitlines()
    i = next(k for k, line in enumerate(text) if line.startswith("validation_list,"))
    bad = tmp_path / "bad.csv"
    bad.write_text("\n".join(text[:i] + text[i + 1:]) + "\n")   # one validation file would move to HEAR train
    with pytest.raises(SystemExit):
        mm.main(["--no-media", "--metadata", str(bad), "--check"])


MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("train/validation_list.txt", "test", "hear_silence"))


@pytest.mark.skipif(MEDIA_ROOT is None, reason="Speech Commands not staged on this machine")
def test_staged_listing_is_the_manifest_one(tmp_path: Path, rows: list[dict[str, str]]):
    mm = _mm()
    out = tmp_path / "staged.csv"
    assert mm.main(["--media-root", str(MEDIA_ROOT), "--write-metadata", str(out)]) == 0
    assert out.read_bytes() == _listing_from_manifest(rows)


# ----------------------------------------------------------------------------- bridge template


class _FakeRawStream:
    def __init__(self, path: Path):
        import soundfile as sf

        i = sf.info(str(path))
        self.name, self.kind, self.path, self.member, self.frames = "audio", "audio", str(path), None, None
        self.info = {"duration_s": i.frames / i.samplerate, "sample_rate": i.samplerate, "n_samples": i.frames,
                     "channels": i.channels, "container": "wav", "bytes": path.stat().st_size}

    def bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def open(self):
        return open(self.path, "rb")


def test_bridge_template_describe_keys():
    pytest.importorskip("soundfile")
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED

    mod = F.load_module(HERE / "bridge_template.py")
    src = (HERE / "bridge_template.py").read_text()
    assert "Preprocessing source of the model" in src and "TODO" in src
    d = mod.SpeechCommandsBridge(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] == "a" and d["streams"] == ["audio"] and d["views"] == 1 and set(d["input"]) == {"audio"}
    assert d["input"]["audio"]["temporal"] == {"rule": "full", "max_s": 1.0, "pad_to_s": 1.0}   # HEAR's trim/pad


def test_bridge_template_pads_like_hear(tmp_path: Path):
    """Every clip reaches 16 000 samples: the stored samples, then zeros at the end (HEAR's trim_pad_wav)."""
    pytest.importorskip("soundfile")
    from yavih.media.rules import audio_ranges

    lengths = [16000, 12345, 8000]
    streams = []
    for k, n in enumerate(lengths):
        p = tmp_path / f"c{k}.wav"
        _write_wav(p, n, k)
        streams.append(_FakeRawStream(p))
    mod = F.load_module(HERE / "bridge_template.py")
    bridge = mod.SpeechCommandsBridge(SimpleNamespace(name="speech_commands"), {})
    samples = [SimpleNamespace(row=k, clip_id=f"c{k}", split="test", streams={"audio": s}, meta={"facts": {}})
               for k, s in enumerate(streams)]
    feats, pos, used = F.encode_prepared(bridge, samples)
    assert pos is None and np.asarray(feats).shape[0] == 3
    for s, n, u in zip(streams, lengths, used, strict=True):
        ranges, (pl, pr) = audio_ranges(bridge.TEMPORAL, s.info)
        assert u == {"audio": {"samples": [[0, n]], "pad": [0, SR - n]}} and (pl, pr) == (0, SR - n)
        wav, sr, _ = bridge._read(s)
        assert sr == SR and wav.shape == (SR,) and np.all(wav[n:] == 0)


# ----------------------------------------------------------------------------- reader smoke


def test_reader_smoke_synthetic_raw(isolated: Path):
    """The generic reader hands over each stored wav byte for byte with its header facts and ``meta == {"facts":
    {}}``; a short clip is handed over as stored; canaries equal their originals."""
    from yavih import Benchmark

    m = Benchmark.from_dir(HERE).manifest("test")
    media = isolated / "media"
    lengths = [SR, SR, 11000, SR, SR, 9000]
    for i, n in enumerate(lengths):
        _write_wav(media / m.media[i]["audio"], n, i)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    samples = list(b.reader("test", rows=range(0, 6), batch_size=1, canaries=1))
    assert [s.row for s in samples[:6]] == list(range(6)) and len(samples) == 7
    for s, n in zip(samples[:6], lengths, strict=True):
        assert set(s.streams) == {"audio"} and s.meta == {"facts": {}} and s.split == "test"
        st = s.streams["audio"]
        f = media / m.media[s.row]["audio"]
        assert st.bytes() == f.read_bytes()
        assert {k: st.info[k] for k in INFO_KEYS} == {
            "duration_s": pytest.approx(n / SR), "sample_rate": SR, "n_samples": n, "channels": 1,
            "container": "wav", "bytes": f.stat().st_size,
        }
    c = samples[6]
    assert c._canary_of == c.row and c.streams["audio"].bytes() == samples[c.row].streams["audio"].bytes()


@pytest.mark.skipif(MEDIA_ROOT is None, reason="Speech Commands not staged on this machine")
def test_reader_smoke_real_media(isolated: Path):
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": str(MEDIA_ROOT)})
    for split in ("train", "valid", "test"):
        for s in b.reader(split, rows=range(0, 3)):
            st = s.streams["audio"]
            assert st.info["sample_rate"] == SR and st.info["channels"] == 1 and 0 < st.info["n_samples"] <= SR
            raw = st.bytes()
            assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
