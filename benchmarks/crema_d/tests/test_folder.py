"""Folder tests of the ``crema_d`` benchmark (run: ``python -m pytest "$(yavih bench path crema_d)/tests" -q -p no:cacheprovider``).

Three groups:

* yaml / manifest / classes facts, the copied HEAR scorer against the core's fold accuracy, and the copied
  TFDS / HEAR fold derivation (``hear_split.py``) on hand-checked inputs: run anywhere;
* ``make_manifest.py --check`` from a stand-in ``summaryTable.csv`` rebuilt from the committed manifest, the reader
  and the bridge template on synthetic wavs: run anywhere (soundfile for the wavs);
* checks against the staged CREMA-D clone (the pinned summaryTable.csv, the real re-derivation with TFDS's byte
  counts, the reader on real wavs): skip when it is absent.
"""

from __future__ import annotations

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
spec, rows, classes = F.spec, F.rows, F.classes

FOLDS = ["fold00", "fold01", "fold02", "fold03", "fold04"]
CLIPS = {"fold00": 1488, "fold01": 1488, "fold02": 1488, "fold03": 1487, "fold04": 1487}
LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
PER_CLASS = {"ANG": 1271, "DIS": 1270, "FEA": 1270, "HAP": 1271, "NEU": 1086, "SAD": 1270}
TFDS_SKIPS = ["1040_ITH_SAD_XX", "1006_TIE_NEU_XX", "1013_WSI_DIS_XX", "1017_IWW_FEA_XX"]
NATIVE_SR = 16000
INFO_KEYS = ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No site config, no site census, no media-root override."""
    monkeypatch.setenv("YAVIH_CONFIG", str(tmp_path / "_no_such_config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "yavih_home"))
    monkeypatch.delenv("YAVIH_DATA__CREMA_D__MEDIA_ROOT", raising=False)
    return tmp_path


def _write_wav(path: Path, n_frames: int, seed: int) -> None:
    sf = pytest.importorskip("soundfile")
    pcm = (np.random.default_rng(seed).uniform(-0.5, 0.5, n_frames) * 32767).astype(np.int16)
    sf.write(path, pcm, NATIVE_SR, subtype="PCM_16")


def _hear_split():
    return F.load_module(HERE / "hear_split.py")


def _eval():
    return F.load_module(HERE / "eval.py")


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_manifest_schema_and_names(rows: list[dict[str, str]]):
    assert list(rows[0]) == ["clip_id", "labels", "split", "rel_path", "actor", "sentence", "intensity", "tfds_split", "hear_file"]
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids) == 7438 and not set(ids) & set(TFDS_SKIPS)
    name_re = re.compile(r"^(1\d{3})_([A-Z]{3})_(ANG|DIS|FEA|HAP|NEU|SAD)_(LO|MD|HI|XX)$")
    stem_re = re.compile(r"^tfds_id_crema-d-(train|validation|test)-tfrecord-(\d{5})-of-(\d{5})-(\d+)$")
    shards = {"train": 16, "validation": 2, "test": 4}
    for r in rows:
        actor, sentence, emotion, intensity = name_re.match(r["clip_id"]).groups()
        assert (actor, sentence, emotion, intensity) == (r["actor"], r["sentence"], r["labels"], r["intensity"])
        assert r["rel_path"] == f"AudioWAV/{r['clip_id']}.wav"
        split, shard, n, _ = stem_re.match(r["hear_file"]).groups()
        assert split == r["tfds_split"] and int(n) == shards[split] and int(shard) < int(n)
    assert len({r["hear_file"] for r in rows}) == 7438
    assert len({r["actor"] for r in rows}) == 91 and len({r["sentence"] for r in rows}) == 12


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    s = spec["splits"]
    assert F.assert_splits_assigned_once(spec, rows) == CLIPS == spec["budget"]["clips"]
    assert s["folds"] == FOLDS and s["train"] == [] and s["eval"] == [] and s["unused"] == []


def test_classes_and_balance(classes: list[dict[str, str]], rows: list[dict[str, str]]):
    assert [c["name"] for c in classes] == LABELS == sorted({r["labels"] for r in rows})  # HEAR's labelvocabulary order
    assert [int(c["index"]) for c in classes] == list(range(6))
    assert dict(Counter(r["labels"] for r in rows)) == PER_CLASS
    for f in FOLDS:  # HEAR's split_k_folds keeps every label in every fold
        assert {r["labels"] for r in rows if r["split"] == f} == set(LABELS)


def test_tfds_split_is_speaker_grouped_and_folds_are_not(rows: list[dict[str, str]]):
    tfds = Counter(r["tfds_split"] for r in rows)
    assert tfds == {"train": 5144, "validation": 738, "test": 1556}  # TFDS crema_d 1.0.0 split sizes
    by_actor: dict[str, set[str]] = {}
    folds_of: dict[str, set[str]] = {}
    for r in rows:
        by_actor.setdefault(r["actor"], set()).add(r["tfds_split"])
        folds_of.setdefault(r["actor"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_actor.values())
    assert sum(len(v) > 1 for v in folds_of.values()) == 91  # HEAR draws its folds over files


def test_curation_sums(spec: dict, rows: list[dict[str, str]]):
    up = spec["manifest"]["upstream"]
    cur = up["curation"]
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"] == up["rows"] == 7442
    counts = Counter(r["split"] for r in rows)
    assert sorted(cur["per_split"]) == FOLDS
    for s, v in cur["per_split"].items():
        assert v["listed"] == counts[s] and v["staged"] == v["listed"] - v["missing"]
    assert list(cur["excluded"].values()) == [4] and cur["leakage_audit"]["excluded"] == 0
    assert spec["protocol"]["train_draw"] is None and spec["protocol"]["clip_seconds"] is None
    assert up["sha256"] == spec["source"]["sha256"]["processedResults/summaryTable.csv"]
    assert up["upstream_files"]["CREMA-D/processedResults/summaryTable.csv"] == up["sha256"]
    mm = F.load_module(HERE / "make_manifest.py")
    assert up["upstream_files"]["tfds-data/dataset_info/crema_d/1.0.0/dataset_info.json"] == mm.TFDS_INFO_SHA256
    assert spec["source"]["sha256"]["tfds/crema_d-1.0.0-dataset_info.json"] == mm.TFDS_INFO_SHA256


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert spec["scorer"] == "clf" and m["headline"] == "top1_acc_mean"
    assert m["list"] == ["top1_acc_mean", "top1_acc_std", "top1_acc_per_fold", "n_folds", "n_eval"]
    assert m["floors"] == ["top1_acc_floor"] and m["controls"] == []
    assert m["ci"] == {"top1_acc_mean": "pooled_binomial"}
    assert m["aggregation"] == {"over": "folds", "report": ["mean", "pstd", "per_fold"]}


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items()} == {"audio": "audio"}
    assert set(ic["handed_over"]) == set(ic["info"]) == {"audio"} and ic["info"]["audio"] == INFO_KEYS
    assert ic["facts"] == [] and ic["materialize"] is None and spec["external"] == []
    assert spec["protocol_version"] == 1 == spec["changelog"][-1]["version"] and len(spec["changelog"]) == 1


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_download.sh", "02_verify.sh", "03_make_manifest.sh"]


# ----------------------------------------------------------------------------- the copied HEAR scorer


def _fold_population(seed: int, sizes=(40, 41, 39, 40, 42), c: int = 6):
    rng = np.random.default_rng(seed)
    folds, outs, tgts = {}, [], []
    for k, n in enumerate(sizes):
        t = rng.integers(0, c, n)
        o = rng.normal(size=(n, c)).astype(np.float32)
        o[np.arange(n), t] += rng.uniform(0, 2)  # some signal
        folds[f"fold{k:02d}"] = {"outputs": o, "targets": t}
        outs.append(o)
        tgts.append(t)
    return np.concatenate(outs), np.concatenate(tgts), {"folds": folds, "num_classes": c}


def test_hear_top1_equals_core_fold_accuracy():
    """HEAR's Top1Accuracy + aggregate mean is the core's acc_mean / acc_per_fold; the std is the sample std."""
    from yavih.metrics import classification as core

    ev = _eval()
    for seed in range(5):
        o, t, meta = _fold_population(seed)
        per_fold = ev.top1_acc_per_fold(o, t, {}, meta)
        assert per_fold == pytest.approx(core.acc_per_fold(o, t, {}, meta), abs=1e-12)
        assert ev.top1_acc_mean(o, t, {}, meta) == pytest.approx(core.acc_mean(o, t, {}, meta), abs=1e-12)
        assert ev.top1_acc_std(o, t, {}, meta) == pytest.approx(float(np.std(per_fold, ddof=1)), abs=1e-12)
        assert ev.top1_correct(o, t, {}, meta).mean() == pytest.approx(core.acc(o, t, {}, meta), abs=1e-12)


def test_hear_floor_takes_class_indices():
    ev = _eval()
    o, t, meta = _fold_population(7)
    floor = np.concatenate([np.full(len(f["targets"]), 2, dtype=np.int64) for f in meta["folds"].values()])
    expected = np.mean([np.mean(np.asarray(f["targets"]) == 2) for f in meta["folds"].values()])
    assert ev.top1_acc_floor(floor, t, {}, meta) == pytest.approx(expected)
    with pytest.raises(ValueError):
        ev.top1_acc_mean(o[:-1], t[:-1], {}, meta)  # pooled rows must equal the folds' rows


def test_hear_top1_hand_checked():
    hs = F.load_module(HERE / "hear_score.py")
    pred = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    onehot = np.array([[0, 1], [0, 1], [1, 0]])
    assert hs.Top1Accuracy(label_to_idx={})(pred, onehot) == pytest.approx(2 / 3)  # a tie is argmax's first index
    agg = hs.aggregate_test_results({"fold00": {"top1_acc": 0.5}, "fold01": {"top1_acc": 0.7}})
    assert agg == {"top1_acc_mean": pytest.approx(0.6), "top1_acc_std": pytest.approx(0.1414213562)}
    assert hs.data_splits_from_folds(FOLDS)[0] == {"train": ["fold02", "fold03", "fold04"], "valid": ["fold01"], "test": ["fold00"]}


# ----------------------------------------------------------------------------- the copied TFDS / HEAR fold derivation


def test_tfds_writer_pieces_hand_checked():
    hs = _hear_split()
    # TFDS's md5 Hasher: md5(salt + key) as a 128-bit int
    assert hs.Hasher("train").hash_key("1001_DFA_ANG_XX") == int(hashlib.md5(b"train1001_DFA_ANG_XX").hexdigest(), 16)
    # TFDS's shard boundaries round half to even: the train split's shard lengths of dataset_info.json
    b = hs._get_shard_boundaries(5144, 16)
    assert [y - x for x, y in zip([0] + b[:-1], b, strict=True)] == [322, 321, 321, 322] * 4
    assert hs._get_number_shards(1223332734, 5144) == 16 and hs._get_number_shards(181972381, 738) == 2
    assert hs._get_number_shards(365486261, 1556) == 4
    assert hs.slugify("crema_d-train.tfrecord-00003-of-00016__17") == "crema-d-train-tfrecord-00003-of-00016-17"
    assert hs.packed_int64_payload(np.array([0, 127, 128, 16383, 16384, -1, -32768])) == 1 + 1 + 2 + 2 + 3 + 10 + 10


def test_example_size_matches_protobuf():
    """``example_size`` equals ``len(tf.train.Example.SerializeToString())`` (proto rebuilt from its descriptor)."""
    pytest.importorskip("google.protobuf")
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    L = descriptor_pb2.FieldDescriptorProto
    fdp = descriptor_pb2.FileDescriptorProto(name="crema_d_example.proto", package="tensorflow", syntax="proto3")
    fdp.message_type.add(name="BytesList").field.add(name="value", number=1, type=L.TYPE_BYTES, label=L.LABEL_REPEATED)
    fdp.message_type.add(name="Int64List").field.add(name="value", number=1, type=L.TYPE_INT64, label=L.LABEL_REPEATED,
                                                     options=descriptor_pb2.FieldOptions(packed=True))
    feat = fdp.message_type.add(name="Feature")
    feat.oneof_decl.add(name="kind")
    feat.field.add(name="bytes_list", number=1, type=L.TYPE_MESSAGE, type_name=".tensorflow.BytesList", label=L.LABEL_OPTIONAL, oneof_index=0)
    feat.field.add(name="int64_list", number=3, type=L.TYPE_MESSAGE, type_name=".tensorflow.Int64List", label=L.LABEL_OPTIONAL, oneof_index=0)
    fs = fdp.message_type.add(name="Features")
    entry = fs.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=L.TYPE_STRING, label=L.LABEL_OPTIONAL)
    entry.field.add(name="value", number=2, type=L.TYPE_MESSAGE, type_name=".tensorflow.Feature", label=L.LABEL_OPTIONAL)
    fs.field.add(name="feature", number=1, type=L.TYPE_MESSAGE, type_name=".tensorflow.Features.FeatureEntry", label=L.LABEL_REPEATED)
    fdp.message_type.add(name="Example").field.add(name="features", number=1, type=L.TYPE_MESSAGE, type_name=".tensorflow.Features",
                                                   label=L.LABEL_OPTIONAL)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    example = message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.Example"))
    hs = _hear_split()
    rng = np.random.default_rng(0)
    for _ in range(40):
        n, amp = int(rng.integers(1, 40000)), int(rng.choice([10, 200, 20000, 32767]))
        a = rng.integers(-amp - 1, amp + 1, size=n).astype(np.int64)
        label, speaker = int(rng.integers(0, 6)), str(rng.integers(1001, 1092))
        ex = example()
        ex.features.feature["audio"].int64_list.value.extend(a.tolist())
        ex.features.feature["label"].int64_list.value.append(label)
        ex.features.feature["speaker_id"].bytes_list.value.append(speaker.encode())
        assert hs.example_size(n, hs.packed_int64_payload(a), label, speaker) == len(ex.SerializeToString())


def test_split_k_folds_hand_checked():
    import random

    hs = _hear_split()
    keys = [(f"k{i:02d}", "AB"[i % 2]) for i in range(11)]
    folds = hs.split_k_folds(keys, k_folds=5)
    shuffled = sorted(k for k, _ in keys)
    random.Random("split_k_folds").shuffle(shuffled)
    expected = {k: f"fold{j:02d}" for j, part in enumerate(np.array_split(shuffled, 5)) for k in part.tolist()}
    if all({lab for k, lab in keys if expected[k] == f} == {"A", "B"} for f in set(expected.values())):
        assert folds == expected  # the first seed already puts both labels in every fold
    assert sorted(Counter(folds.values()).values()) == [2, 2, 2, 2, 3]


# ----------------------------------------------------------------------------- make_manifest.py --check


def _summary_table(rows: list[dict[str, str]]) -> bytes:
    """A stand-in summaryTable.csv: the manifest's names in order, TFDS's four skips, the upstream header."""
    lines = ['"","FileName","VoiceVote","VoiceLevel","FaceVote","FaceLevel","MultiModalVote","MultiModalLevel"']
    names = [r["clip_id"] for r in rows]
    for i, n in enumerate(names[:100] + TFDS_SKIPS + names[100:], start=1):
        lines.append(f'"{i}","{n}","N","50","N","50","N","50"')
    return ("\n".join(lines) + "\n").encode()


def test_make_manifest_check(tmp_path: Path, rows: list[dict[str, str]], capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """``make_manifest.py --no-media --check`` re-derives manifest.csv.gz and classes.csv byte-identically (TFDS
    split, writer order, shards, HEAR stems and folds) and prints the committed ``manifest.upstream`` block. The
    stand-in table differs from the pinned file in its vote columns only, so its digest is substituted."""
    mm = F.load_module(HERE / "make_manifest.py")
    meta = tmp_path / "summaryTable.csv"
    meta.write_bytes(_summary_table(rows))
    with pytest.raises(SystemExit):  # not the pinned upstream file
        mm.main(["--no-media", "--metadata", str(meta), "--check"])
    pinned = mm.SUMMARY_SHA256
    monkeypatch.setattr(mm, "SUMMARY_SHA256", hashlib.sha256(meta.read_bytes()).hexdigest())
    capsys.readouterr()
    assert mm.main(["--no-media", "--metadata", str(meta), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    y = F.load_yaml(HERE)
    assert report["manifest"]["sha256"] == y["manifest"]["sha256"] and report["manifest"]["classes_sha256"] == y["manifest"]["classes_sha256"]
    up = {k: v for k, v in y["manifest"]["upstream"].items() if k != "note"}
    up["sha256"] = report["upstream"]["sha256"]
    up["upstream_files"] = {**up["upstream_files"], "CREMA-D/processedResults/summaryTable.csv": report["upstream"]["sha256"]}
    assert report["upstream"] == up
    assert y["manifest"]["upstream"]["sha256"] == pinned
    # a swapped pair of names moves rows between TFDS shards and HEAR folds: --check fails
    swapped = [*rows]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    meta.write_bytes(_summary_table(swapped))
    monkeypatch.setattr(mm, "SUMMARY_SHA256", hashlib.sha256(meta.read_bytes()).hexdigest())
    assert mm.main(["--no-media", "--metadata", str(meta), "--check"]) == 1


MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("processedResults/summaryTable.csv", "AudioWAV"))


@pytest.mark.skipif(MEDIA_ROOT is None, reason="CREMA-D clone not staged on this machine")
def test_make_manifest_check_staged(capsys: pytest.CaptureFixture):
    """On the staged clone: the pinned summaryTable.csv, and (wavs read in full) TFDS's byte counts per split."""
    mm = F.load_module(HERE / "make_manifest.py")
    assert mm.main(["--media-root", str(MEDIA_ROOT), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["derived"]["num_shards"] == {"train": 16, "validation": 2, "test": 4}
    assert {k: v["serialized_bytes"] for k, v in report["derived"]["tfds_shard_margins"].items()} == mm.TFDS_NUM_BYTES


# ----------------------------------------------------------------------------- reader + template on synthetic wavs


def test_reader_smoke_synthetic_raw(isolated: Path):
    """The reader hands over each stored wav byte for byte with its header facts and ``meta == {"facts": {}}``,
    whatever its length; canaries equal their originals."""
    from yavih import Benchmark

    probe = Benchmark.from_dir(HERE)
    m = probe.manifest("fold02")
    media = isolated / "media"
    (media / "AudioWAV").mkdir(parents=True)
    lengths = [16000 * 2 + 37 * i for i in range(6)]
    lengths[1] = 80080  # 5.005 s
    for i, n in enumerate(lengths):
        _write_wav(media / "AudioWAV" / f"{m.clip_id[i]}.wav", n, i)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    samples = list(b.reader("fold02", rows=range(0, 6), batch_size=1, canaries=1))
    assert [s.row for s in samples[:6]] == list(range(6)) and len(samples) == 7
    for s, n in zip(samples[:6], lengths, strict=True):
        assert set(s.streams) == {"audio"} and s.meta == {"facts": {}} and s.split == "fold02"
        st = s.streams["audio"]
        f = media / "AudioWAV" / f"{s.clip_id}.wav"
        assert st.bytes() == f.read_bytes()
        assert {k: st.info[k] for k in INFO_KEYS} == {
            "duration_s": pytest.approx(n / NATIVE_SR), "sample_rate": NATIVE_SR, "n_samples": n, "channels": 1,
            "container": "wav", "bytes": f.stat().st_size,
        }
    c = samples[6]
    assert c._canary_of == c.row and c.streams["audio"].bytes() == samples[c.row].streams["audio"].bytes()


def test_reader_to_template_end_to_end(isolated: Path):
    """The template trims or pads every clip to HEAR's 5.0 s and reports ``used`` inside the media."""
    from yavih import Benchmark
    from yavih.bridge import validate_describe

    probe = Benchmark.from_dir(HERE)
    m = probe.manifest("fold00")
    media = isolated / "media"
    (media / "AudioWAV").mkdir(parents=True)
    lengths = [30000, 80080]
    for i, n in enumerate(lengths):
        _write_wav(media / "AudioWAV" / f"{m.clip_id[i]}.wav", n, 10 + i)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    mod = F.load_module(HERE / "bridge_template.py")
    bridge = mod.CremaDBridge(b.view(), {})
    d = bridge.describe()
    validate_describe(d, bridge, b)
    assert d["reading"] == "a" and d["input"]["audio"]["temporal"] == {"rule": "full", "max_s": 5.0, "pad_to_s": 5.0}
    samples = list(b.reader("fold00", rows=range(0, 2)))
    feats, pos, used = F.encode_prepared(bridge, samples)
    assert np.asarray(feats).shape[0] == 2 and pos is None
    assert [u["audio"] for u in used] == [{"samples": [[0, 30000]], "pad": [0, 50000]}, {"samples": [[0, 80000]], "pad": [0, 0]}]


# ----------------------------------------------------------------------------- the staged clone


@pytest.mark.skipif(MEDIA_ROOT is None, reason="CREMA-D media not staged on this machine")
def test_reader_smoke_real_media(isolated: Path):
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": str(MEDIA_ROOT)})
    samples = list(b.reader("fold01", rows=range(0, 4)))
    for s in samples:
        st = s.streams["audio"]
        assert st.info["sample_rate"] == NATIVE_SR and st.info["channels"] == 1 and 1.2 < st.info["duration_s"] <= 5.01
        raw = st.bytes()
        assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and s.meta == {"facts": {}}
