"""Folder tests of the packaged ``fsd50k`` benchmark.

Run with ``python -m pytest "$(yavih bench path fsd50k)/tests" -q -p no:cacheprovider``.
Everything is offline: the synthetic-media tests write their own wavs (stdlib ``wave``) and the re-freeze test
writes a synthetic ground truth. The real-media smoke and the official-draw check skip when the media / ground
truth are not staged (the configured ``[data.fsd50k] media_root``, via yavih's config resolver).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_bytes, rows, classes = F.spec, F.manifest_bytes, F.rows, F.classes

# Keys the schema does not have (window, padding and decode rules belong to the bridge).
REMOVED_KEYS = {"windows", "window_shift", "frame_rule", "frame_rule_params", "frame_grid", "frame_select", "pad_policy",
           "short_clip_policy", "crop", "audio_norm_context", "bridge_decode_allowed"}
AUDIO_INFO_VOCAB = {"duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"}  # RawStream.info keys, kind audio

# resolved at collection, before a test hides the site config
MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("eval_audio",))
GROUND_TRUTH = F.site_root(HERE, "media_root", needs=("ground_truth/dev.csv", "ground_truth/eval.csv"))


def _load(path: Path, name: str):
    return F.load_module(path, name)


def _write_wav(path: Path, seconds: float, sr: int = 44100, seed: int = 0) -> bytes:
    """PCM16 mono wav with strictly non-zero samples; returns the file bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(seconds * sr))
    pcm = (np.random.default_rng(seed).uniform(0.1, 0.5, n) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path.read_bytes()


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


# ----------------------------------------------------------------------------- yaml schema 2


def test_schema2_top_level_and_no_removed_keys(spec: dict):
    assert spec["schema_version"] == 2 and spec["name"] == "fsd50k" == HERE.name
    assert list(spec) == ["schema_version", "name", "aliases", "title", "capability", "header", "readout", "protocol_version", "changelog", "source", "obtain",
                          "budget", "manifest", "splits", "streams", "targets", "data", "input_contract", "output_contract",
                          "external", "protocol", "metrics", "readout_params"]
    assert "requires" not in spec  # moved into output_contract.requires

    def walk(node, where="<root>"):
        if isinstance(node, dict):
            for k, v in node.items():
                assert k not in REMOVED_KEYS, f"removed key {k!r} at {where}"
                walk(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")

    walk(spec)


def test_changelog(spec: dict):
    cl = spec["changelog"]
    assert [e["version"] for e in cl] == [1] and spec["protocol_version"] == 1
    assert cl[0]["invalidates"] is False and cl[0]["supersedes"] == []


def test_input_contract(spec: dict):
    ic = spec["input_contract"]
    assert list(spec["streams"]) == ["audio"]
    st = spec["streams"]["audio"]
    assert st == {"kind": "audio", "media_column": "rel_path", "root": "media_root", "layout": "dir", "layout_params": {}, "required": True}
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(ic["info"]) == set(spec["streams"])
    assert "exactly as stored" in ic["handed_over"]["audio"] and "nothing decoded" in ic["handed_over"]["audio"]
    assert ic["info"]["audio"] == ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]
    assert set(ic["info"]["audio"]) <= AUDIO_INFO_VOCAB
    assert ic["facts"] == [] and ic["materialize"] is None
    assert spec["targets"] is None and spec["data"]["derived"] == {}


def test_output_contract_and_external(spec: dict):
    assert spec["output_contract"] == {
        "shape": "fixed", "coordinates": "media", "target_frame": None,
        "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None},
    }
    assert spec["external"] == [] and not (HERE / "probe.py").exists()  # non-empty external would require probe.py
    assert spec["readout"] == "clf" and spec["readout_params"] == {}


def test_protocol_holds_label_facts_only(spec: dict):
    p = spec["protocol"]
    assert set(p) == {"clip_seconds", "train_draw", "views"}
    assert p["clip_seconds"] == 10.0 and p["views"] == "none"
    assert p["train_draw"] == {"cap": 20000, "seed": 0, "balance": "first_label", "drawn_from": p["train_draw"]["drawn_from"]}


# ----------------------------------------------------------------------------- manifest + classes


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_manifest_schema_and_charsets(rows: list[dict[str, str]]):
    assert list(rows[0]) == ["clip_id", "labels", "split", "rel_path", "dev_split", "duration_s"]
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids) == 30231
    key_re, media_re = re.compile(r"^[A-Za-z0-9_.#:-]+$"), re.compile(r"^[A-Za-z0-9_.#:/-]+$")
    for r in rows:
        assert key_re.match(r["clip_id"]) and media_re.match(r["rel_path"])
        assert r["rel_path"] == (("dev_audio/" if r["split"] == "train" else "eval_audio/") + r["clip_id"] + ".wav")
        # `;`-joined class names (label.sep): no leading/trailing/doubled separator, no padding, no repeats
        toks = r["labels"].split(";")
        assert toks and all(t and t == t.strip() for t in toks), r["clip_id"]
        assert len(set(toks)) == len(toks), r["clip_id"]
        assert (r["dev_split"] in ("train", "val")) == (r["split"] == "train")
        assert 0.3 <= float(r["duration_s"]) <= 65.0
    durs = [float(r["duration_s"]) for r in rows]
    assert (min(durs), max(durs)) == (0.3, 64.133)  # the range input_contract.handed_over states


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    assert F.assert_splits_assigned_once(spec, rows) == {"train": 20000, "eval": 10231}
    assert spec["budget"]["clips"] == {"train": 20000, "eval": 10231}


def test_classes_contiguous_official_order(classes: list[dict[str, str]]):
    assert list(classes[0]) == ["index", "name", "mid"]
    assert [int(c["index"]) for c in classes] == list(range(200))
    names = [c["name"] for c in classes]
    assert len(set(names)) == 200 and len({c["mid"] for c in classes}) == 200
    # The official vocabulary.csv order, NOT alphabetical.
    assert names[0] == "Accelerating_and_revving_and_vroom" and names[1] == "Accordion" and names[199] == "Zipper_(clothing)"
    # classes.csv == the upstream headerless file + header; upstream sha recorded.
    body = (HERE / "classes.csv").read_bytes().split(b"\n", 1)[1]
    assert hashlib.sha256(body).hexdigest() == "992b6e11422101ee575ab3022d37b79b98e19407b211331c72f5d9fdd16e7174"


def test_labels_in_vocabulary_and_first_label_is_leaf(rows: list[dict[str, str]], classes: list[dict[str, str]]):
    names = {c["name"] for c in classes}
    assert not any(";" in n or "," in n for n in names)  # a class name can never collide with the separator
    for r in rows:
        toks = r["labels"].split(";")
        assert toks and all(t in names for t in toks)
    # README "Paper": 1–22 labels per clip over the frozen rows, mean 3.1 (train 2.7, eval 3.8)
    per_clip = [len(r["labels"].split(";")) for r in rows]
    assert (min(per_clip), max(per_clip), round(sum(per_clip) / len(per_clip), 1)) == (1, 22, 3.1)
    # every class has eval positives (effective_pos_classes == 200 at freeze time)
    pos = Counter(t for r in rows if r["split"] == "eval" for t in set(r["labels"].split(";")))
    assert len(pos) == 200 and min(pos.values()) == 32


def test_curation_sums(spec: dict, rows: list[dict[str, str]]):
    up = spec["manifest"]["upstream"]
    cur = up["curation"]
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"] == up["rows"] == 51197
    counts = Counter(r["split"] for r in rows)
    for s, v in cur["per_split"].items():
        assert v["listed"] == counts[s] and v["staged"] == v["listed"] - v["missing"]
    assert cur["excluded"] == {"draw_remainder": 20966}
    assert cur["short_clips"] == sum(1 for r in rows if float(r["duration_s"]) < spec["protocol"]["clip_seconds"]) == 20315
    mm = _load(HERE / "make_manifest.py", "fsd50k_make_manifest")
    assert up["sha256"] == mm.GROUND_TRUTH_SHA256["dev.csv"] and up["sha256"] in spec["protocol"]["train_draw"]["drawn_from"]
    assert up["upstream_files"] == {f"FSD50K.ground_truth/{k}": v for k, v in mm.GROUND_TRUTH_SHA256.items()}


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert m["headline"] == "map" and m["floors"] == ["map_floor"] and m["controls"] == []
    assert m["list"] == ["map", "effective_pos_classes", "n_eval"] and m["ci"] == {"map": "bootstrap_clip"}
    assert set(m["chance"]) <= set(m["list"]) and set(m["ci"]) <= set(m["list"])


# ----------------------------------------------------------------------------- README.md / setup.md


def test_readme_facts(spec: dict):
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    assert "setup.md" in F.section(text, 2, "In yavih")
    assert "@article" in F.section(text, 2, "Citation") and "2010.00475" in text.split("\n\n")[1]
    for needle in ("20 000", "10 231", "map_floor", "0.0189", spec["manifest"]["sha256"]):
        assert needle in text, needle


def test_setup_facts():
    """Folder facts in setup.md (the setup.md structure and contract copies are ``yavih bench lint``'s)."""
    text = (HERE / "setup.md").read_text(encoding="utf-8")
    # setup facts the bridge author needs
    for needle in ("c480d119b8f7a7e32fdb58f3ea4d6c5a", "zip -s 0", "18 412 790 687", "51 210", "992b6e11",
                   "7ad288bbfb6b957ba83579f33feb44bce2fbad932bdd1fa5dc51c36776369c55", "2 515"):
        assert needle in text, needle


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir())
    assert names == ["01_download.sh", "02_extract.sh", "03_make_manifest.sh"]
    dl = (HERE / "prepare" / "01_download.sh").read_text()
    assert "zenodo.org/records/4060432" in dl and "c480d119b8f7a7e32fdb58f3ea4d6c5a" in dl
    assert "zip -q -s 0" in (HERE / "prepare" / "02_extract.sh").read_text()
    step3 = (HERE / "prepare" / "03_make_manifest.sh").read_text()
    assert "yavih bench verify fsd50k --census" in step3 and "--check" in step3


# ----------------------------------------------------------------------------- the train draw and the re-freeze


def test_class_balanced_subsample_draw():
    torch = pytest.importorskip("torch")
    mm = _load(HERE / "make_manifest.py", "fsd50k_make_manifest")
    labels = ["b", "a", "a", "c", "b", "a", "c", "b", "a", "c"]
    keep = mm.class_balanced_subsample(labels, 6, 0)
    assert keep == sorted(keep) and len(keep) == 6
    assert Counter(labels[i] for i in keep) == {"a": 2, "b": 2, "c": 2}  # round robin: 2 per class
    assert mm.class_balanced_subsample(labels, 6, 0) == keep  # deterministic
    assert mm.class_balanced_subsample(labels, 0, 0) == list(range(10)) == mm.class_balanced_subsample(labels, 10, 0)
    # The exact draw: ONE torch.Generator seeded once, buckets shuffled in sorted class
    # order ("a" first), so the two kept "a" rows are the first two of that permutation.
    g = torch.Generator()
    g.manual_seed(0)
    bucket_a = [1, 2, 5, 8]
    first_two_a = [bucket_a[p] for p in torch.randperm(4, generator=g).tolist()][:2]
    assert sorted(i for i in keep if labels[i] == "a") == sorted(first_two_a)
    # a different seed changes the draw (the seed is a protocol constant)
    assert mm.class_balanced_subsample(labels * 50, 100, 1) != mm.class_balanced_subsample(labels * 50, 100, 0)


@pytest.mark.skipif(GROUND_TRUTH is None, reason="FSD50K ground truth not staged on this machine")
def test_frozen_rows_equal_official_draw(rows: list[dict[str, str]]):
    """The frozen train rows are the seeded draw over the official dev.csv; eval is eval.csv in file order."""
    pytest.importorskip("torch")
    mm = _load(HERE / "make_manifest.py", "fsd50k_make_manifest")
    gt = GROUND_TRUTH / "ground_truth"
    for name, sha in mm.GROUND_TRUTH_SHA256.items():
        assert mm.sha256_file(gt / name) == sha, name
    labels = {name for _, name, _ in mm.read_vocabulary(gt / "vocabulary.csv")}
    up = mm.rows_from_ground_truth(gt, labels)
    dev = [r for r in up if r["split"] in ("train", "val")]
    keep = mm.class_balanced_subsample([r["labels"].split(";")[0] for r in dev], 20000, 0)
    frozen_train = [r for r in rows if r["split"] == "train"]
    assert [(dev[i]["clip_id"], dev[i]["labels"], dev[i]["split"]) for i in keep] == [(r["clip_id"], r["labels"], r["dev_split"]) for r in frozen_train]
    assert [(r["clip_id"], r["labels"]) for r in up if r["split"] == "eval"] == [(r["clip_id"], r["labels"]) for r in rows if r["split"] == "eval"]


def _write_synthetic_release(root: Path) -> None:
    """A 4-class FSD50K-shaped release: headerless vocabulary.csv, dev.csv/eval.csv with ,-joined labels, wavs."""
    gt = root / "ground_truth"
    gt.mkdir(parents=True)
    (gt / "vocabulary.csv").write_text("0,Bark,/m/a\n1,Dog,/m/b\n2,Meow,/m/c\n3,Animal,/m/d\n", encoding="utf-8")
    dev = ["fname,labels,mids,split"]
    for i in range(12):
        lab = ["Bark,Dog,Animal", "Meow,Animal", "Dog,Animal"][i % 3]
        dev.append(f'{100 + i},"{lab}",x,{"val" if i % 4 == 0 else "train"}')
    (gt / "dev.csv").write_text("\n".join(dev) + "\n", encoding="utf-8")
    (gt / "eval.csv").write_text('fname,labels,mids\n900,"Meow,Animal",x\n901,"Bark,Dog,Animal",x\n', encoding="utf-8")
    for i in range(12):
        _write_wav(root / "dev_audio" / f"{100 + i}.wav", 0.5 + 0.25 * i, sr=8000, seed=i)
    for cid in (900, 901):
        _write_wav(root / "eval_audio" / f"{cid}.wav", 12.0, sr=8000, seed=cid)


def test_refreeze_from_synthetic_ground_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """The public route end to end on a synthetic release: rows from dev.csv/eval.csv, the first-label draw,
    classes.csv = vocabulary.csv + header, header durations, the upstream block; a re-run is byte-identical."""
    pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    mm = _load(HERE / "make_manifest.py", "fsd50k_make_manifest_synthetic")
    media, out = tmp_path / "media", tmp_path / "out"
    _write_synthetic_release(media)
    gt = media / "ground_truth"
    monkeypatch.setattr(mm, "GROUND_TRUTH_SHA256", {n: mm.sha256_file(gt / n) for n in ("vocabulary.csv", "dev.csv", "eval.csv")})
    monkeypatch.setattr(mm, "DEV_UPSTREAM", 12)
    monkeypatch.setattr(mm, "EVAL_UPSTREAM", 2)
    monkeypatch.setattr(mm, "TOTAL_UPSTREAM", 14)
    monkeypatch.setattr(mm, "NUM_CLASSES", 4)
    assert mm.main(["--media-root", str(media), "--out", str(out), "--cap", "6", "--workers", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    first = (out / "manifest.csv.gz").read_bytes()
    got = list(csv.DictReader(io.StringIO(gzip.decompress(first).decode("utf-8"))))
    assert [r["split"] for r in got] == ["train"] * 6 + ["eval"] * 2
    assert Counter(r["labels"].split(";")[0] for r in got if r["split"] == "train") == {"Bark": 2, "Meow": 2, "Dog": 2}
    assert [r["clip_id"] for r in got[6:]] == ["900", "901"] and got[7]["labels"] == "Bark;Dog;Animal" and got[7]["dev_split"] == ""
    assert all(r["dev_split"] in ("train", "val") for r in got[:6])
    assert [r["duration_s"] for r in got[6:]] == ["12.000", "12.000"]
    assert (out / "classes.csv").read_text(encoding="utf-8").splitlines() == ["index,name,mid", "0,Bark,/m/a", "1,Dog,/m/b", "2,Meow,/m/c", "3,Animal,/m/d"]
    up = report["upstream"]
    assert up["sha256"] == mm.GROUND_TRUTH_SHA256["dev.csv"] and up["rows"] == 14
    assert up["curation"]["excluded"] == {"draw_remainder": 6} and up["curation"]["short_clips"] == 6
    assert mm.main(["--media-root", str(media), "--out", str(out), "--cap", "6", "--workers", "2"]) == 0
    assert (out / "manifest.csv.gz").read_bytes() == first
    # a ground-truth file that is not the pinned one is refused
    monkeypatch.setattr(mm, "GROUND_TRUTH_SHA256", {**mm.GROUND_TRUTH_SHA256, "dev.csv": "0" * 64})
    with pytest.raises(SystemExit, match="expected"):
        mm.main(["--media-root", str(media), "--out", str(out)])


# ----------------------------------------------------------------------------- no reader.py (generic reader)


def test_folder_ships_no_reader_py():
    """The folder ships no reader.py: ``Benchmark.reader()`` falls back to the generic
    ``yavih.sample.iter_samples``, which hands over what benchmark.yaml declares; the reader smokes in this
    file go through ``bench.reader``."""
    assert not (HERE / "reader.py").exists()


def test_benchmark_loads_schema2(monkeypatch: pytest.MonkeyPatch):
    from yavih import Benchmark

    monkeypatch.setenv("YAVIH_CONFIG", str(HERE / "tests" / "_no_such_config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(HERE / "tests" / "_no_such_xdg"))
    b = Benchmark.from_dir(HERE)
    assert b.name == "fsd50k" and b.spec.readout == "clf" and b.spec.manifest.label.kind == "multi"
    assert b.n_rows("train") == 20000 and b.n_rows("eval") == 10231
    assert b.roles() == {"train": ["train"], "eval": ["eval"], "floor": [], "folds": []}
    assert b.manifest_hash == "7ad288bbfb6b957ba83579f33feb44bce2fbad932bdd1fa5dc51c36776369c55"
    assert len(b.protocol_hash) == 64
    m = b.manifest("eval")
    assert m.clip_id[0] == "37199" and m.label_ids[0] == [69, 90, 127, 121, 120]
    v = b.view()
    assert v.streams == {"audio": "audio"} and v.facts == {} and v.protocol_version == 1


def test_reader_smoke_synthetic_raw_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Raw hand-off on synthetic wavs under the manifest's first ``rel_path``s: ``bytes()``/``open()`` are
    the stored file byte for byte (a 0.3 s and a 12 s clip are neither padded nor cut), ``info`` holds the
    guaranteed header facts, ``meta == {"facts": {}}``, canaries are copies flagged only privately."""
    from yavih import Benchmark

    monkeypatch.setenv("YAVIH_CONFIG", str(HERE / "tests" / "_no_such_config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(HERE / "tests" / "_no_such_xdg"))
    monkeypatch.delenv("YAVIH_DATA__FSD50K__MEDIA_ROOT", raising=False)
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    m = b.manifest("eval")
    seconds = [0.3, 12.0, 3.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    stored = [_write_wav(tmp_path / m.media[i]["audio"], sec, seed=i) for i, sec in enumerate(seconds)]
    samples = list(b.reader("eval", rows=range(0, 8), canaries=1))
    assert [s.row for s in samples[:8]] == list(range(8)) and len(samples) == 9
    for s, sec, data in zip(samples[:8], seconds, stored, strict=True):
        assert set(s.streams) == {"audio"} and s.meta == {"facts": {}}
        assert not hasattr(s, "media_key")
        st = s.streams["audio"]
        assert st.name == "audio" and st.kind == "audio" and getattr(st, "frames", None) is None
        assert st.bytes() == data and st.open().read() == data
        assert Path(st._real_path) == tmp_path / m.media[s.row]["audio"]
        info = st.info
        assert {k for k in spec_info_keys() if info.get(k) is None} == set()
        assert info["sample_rate"] == 44100 and info["channels"] == 1 and info["n_samples"] == int(round(sec * 44100))
        assert abs(info["duration_s"] - sec) < 1e-6 and info["bytes"] == len(data) and info["container"] == "wav"
    can = samples[8]
    assert can._canary_of == can.row and can.meta == {"facts": {}} and can.streams["audio"].bytes() == stored[can.row]


def spec_info_keys() -> list[str]:
    return yaml.safe_load((HERE / "benchmark.yaml").read_text(encoding="utf-8"))["input_contract"]["info"]["audio"]


@pytest.mark.skipif(MEDIA_ROOT is None, reason="FSD50K media not staged on this machine")
def test_reader_smoke_real_media(monkeypatch: pytest.MonkeyPatch):
    from yavih import Benchmark

    monkeypatch.setenv("YAVIH_CONFIG", str(HERE / "tests" / "_no_such_config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(HERE / "tests" / "_no_such_xdg"))
    b = Benchmark.from_dir(HERE, data={"media_root": str(MEDIA_ROOT)})
    man = b.manifest("eval")
    for s in b.reader("eval", rows=range(0, 4)):
        st = s.streams["audio"]
        path = MEDIA_ROOT / man.media[s.row]["audio"]
        assert st.bytes() == path.read_bytes()
        assert st.info["sample_rate"] == 44100 and st.info["channels"] == 1
        assert abs(st.info["duration_s"] - man.extra[s.row]["duration_s"]) < 1e-3


# ----------------------------------------------------------------------------- bridge_template.py vs the output contract


class _FakeStream:
    """Minimal RawStream stand-in over an in-memory wav."""

    def __init__(self, data: bytes, sr: int, n: int):
        self.name, self.kind, self.path, self.member, self.frames = "audio", "audio", None, None, None
        self._data = data
        self.info = {"duration_s": n / sr, "sample_rate": sr, "n_samples": n, "channels": 1, "container": "wav", "bytes": len(data)}

    def bytes(self) -> bytes:
        return self._data

    def open(self):
        return io.BytesIO(self._data)


class _FakeSample:
    def __init__(self, row: int, data: bytes, sr: int, n: int):
        self.row, self.clip_id, self.split = row, f"c{row}", "eval"
        self.streams = {"audio": _FakeStream(data, sr, n)}
        self.meta = {"facts": {}}


def _fake_samples(tmp_path: Path, seconds: list[float]) -> list[_FakeSample]:
    out = []
    for i, sec in enumerate(seconds):
        data = _write_wav(tmp_path / f"{i}.wav", sec, seed=100 + i)
        out.append(_FakeSample(i, data, 44100, int(round(sec * 44100))))
    return out


@pytest.fixture(scope="module")
def template():
    pytest.importorskip("soundfile")
    return _load(HERE / "bridge_template.py", "fsd50k_bridge_template_under_test")


def test_template_describe_declares_rules(template):
    br = template.Fsd50kBridge(None, {})
    d = br.describe()
    assert "decode" not in d and d["views"] == 1 and d["streams"] == ["audio"] and d["pos_mode"] == "none"
    assert set(d["input"]) == {"audio"}
    a = d["input"]["audio"]
    assert set(a) == {"temporal", "front_end", "norm", "decoder"}
    assert a["temporal"] == {"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"}
    assert "sample_rate" in a["front_end"] and "resample" in a["front_end"] and "features" in a["front_end"]
    assert "source" in a["norm"] and a["decoder"]["lib"] == "soundfile"
    assert all(v is not None for k, v in d["precision"].items() if k != "autocast")
    assert d["args"] == {"layer": -1}
    assert not hasattr(template.Fsd50kBridge, "decode_plan") or not template.Fsd50kBridge.decode_plan
    json.dumps(d)  # JSON primitives only
    try:
        from yavih.media.rules import check_temporal
    except ImportError:  # pragma: no cover - core without the rules module
        return
    check_temporal(a["temporal"], "audio")


def test_template_encode_meets_output_contract(template, tmp_path: Path):
    """Fixed (N, D) over 0.3–64 s clips, ``used`` equal to the declared ``span`` rule on each row's facts
    (what validate's E_RULE_MISMATCH re-evaluates), within the media (E_TIMES_OUT_OF_MEDIA), and
    batch-, order- and rerun-invariant (bridge check's hard checks)."""
    seconds = [0.3, 3.0, 9.9995, 10.0, 12.0, 64.133]
    samples = _fake_samples(tmp_path, seconds)
    br = template.Fsd50kBridge(None, {})
    feats, pos, used = F.encode_prepared(br, samples)
    assert pos is None and len(used) == len(samples)
    feats = np.asarray(feats)
    assert feats.ndim == 3 and feats.shape[:2] == (len(samples), template.N_TOKENS) and np.isfinite(feats).all()
    rule = br.describe()["input"]["audio"]["temporal"]
    try:
        from yavih.media.rules import audio_ranges
    except ImportError:  # pragma: no cover
        audio_ranges = None
    for s, u in zip(samples, used, strict=True):
        info = s.streams["audio"].info
        (lo, hi), = u["audio"]["samples"]
        L = int(round(rule["length_s"] * info["sample_rate"]))
        assert lo == 0 and hi == min(L, info["n_samples"]) and 0 <= lo <= hi <= info["n_samples"]
        assert u["audio"]["pad"] == [0, L - hi]
        if audio_ranges is not None:
            ranges, pad = audio_ranges(rule, info)
            assert ranges.tolist() == u["audio"]["samples"] and list(pad) == u["audio"]["pad"]
    # rerun bit-identical; alone and shuffled batches equal (instance-wise extraction)
    again, _, used2 = F.encode_prepared(br, samples)
    assert np.array_equal(np.asarray(again), feats) and used2 == used
    for i, s in enumerate(samples):
        f1, _, u1 = F.encode_prepared(br, [s])
        assert np.array_equal(np.asarray(f1)[0], feats[i]) and u1[0] == used[i]
    perm = [4, 0, 5, 2, 1, 3]
    fp, _, up = F.encode_prepared(br, [samples[i] for i in perm])
    assert np.array_equal(np.asarray(fp), feats[perm]) and up == [used[i] for i in perm]
    br.decode_workers = 4  # the CPU budget (the prepare threads in run_extract) changes nothing
    fw, _, uw = F.encode_prepared(br, samples)
    assert np.array_equal(np.asarray(fw), feats) and uw == used


def test_template_per_sample_pos_in_media_coordinates(template, tmp_path: Path):
    samples = _fake_samples(tmp_path, [3.0, 20.0])
    br = template.Fsd50kBridge(None, {})
    br.pos_mode = "per_sample"
    _, pos, _ = F.encode_prepared(br, samples)
    pos = np.asarray(pos)
    assert pos.shape == (2, template.N_TOKENS, 4) and pos.dtype == np.float32
    assert (pos[..., 0] == 0).all() and (pos[..., 2:] == -1).all()  # audio: stream 0, no space
    t = pos[..., 1]
    assert ((t == -1) | ((t >= 0) & (t <= 1))).all()
    # 3 s clip: centres 0.5, 1.5, 2.5 s lie in the media (t = c / 3), the rest is added padding (t = -1)
    np.testing.assert_allclose(t[0, :3], np.array([0.5, 1.5, 2.5]) / 3.0, rtol=0, atol=1e-6)
    assert (t[0, 3:] == -1).all()
    np.testing.assert_allclose(t[1], (np.arange(10) + 0.5) / 20.0, rtol=0, atol=1e-6)


def test_template_corrupt_media_is_undecodable(template, tmp_path: Path):
    import yavih

    s = _fake_samples(tmp_path, [1.0])[0]
    s.streams["audio"]._data = b"RIFF\x00\x00\x00\x00not a wav at all"
    with pytest.raises(yavih.MediaUndecodable):
        F.encode_prepared(template.Fsd50kBridge(None, {}), [s])


def test_template_passes_bridge_check(tmp_path: Path):
    """End to end on 32 synthetic eval wavs: ``yavih bridge check`` on the template (TODO identity fields
    filled by a subclass) passes bridge check's hard checks and validate's per-member checks."""
    pytest.importorskip("soundfile")
    from yavih import Benchmark

    env = {**os.environ, "YAVIH_CONFIG": str(HERE / "tests" / "_no_such_config.toml"), "YAVIH_DATA__FSD50K__MEDIA_ROOT": str(tmp_path / "media"),
           "YAVIH_HOME": str(tmp_path / "home"), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    m = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path / "media")}).manifest("eval")
    for i in range(32):
        _write_wav(tmp_path / "media" / m.media[i]["audio"], [0.3, 4.0, 10.0, 15.0][i % 4], seed=i)
    bridge = tmp_path / "smoke_bridge.py"
    bridge.write_text(
        "import importlib.util\n"
        f"_s = importlib.util.spec_from_file_location('fsd50k_tpl', {str(HERE / 'bridge_template.py')!r})\n"
        "_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n"
        "class SmokeBridge(_m.Fsd50kBridge):\n"
        "    name = 'fsd50k-template-smoke'\n"
        "    dtype = 'float32'\n"
        "    def describe(self):\n"
        "        d = super().describe()\n"
        "        d['weights_sha256'] = '0' * 64; d['weights_ref'] = 'none'; d['model'] = 'template-smoke'\n"
        "        d['input']['audio']['front_end'].update(resample='none', features='log-rms+peak per 1 s chunk', n_mels=0, win_ms=1000, hop_ms=1000)\n"
        "        d['input']['audio']['norm'].update(mean=0.0, std=1.0, source='placeholder')\n"
        "        return d\n",
        encoding="utf-8",
    )
    out = subprocess.run([sys.executable, "-m", "yavih.cli", "bridge", "check", str(bridge), "--bench", str(HERE), "--n", "32", "--json"],
                         env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    data = json.loads(out.stdout)["data"]
    assert data["invariance"]["rerun_bit_identical"] is True and data["N"] == 10
