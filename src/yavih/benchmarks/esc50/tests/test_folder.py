"""Folder tests of the packaged ``esc50`` benchmark.

Run with ``python -m pytest "$(yavih bench path esc50)/tests" -q -p no:cacheprovider``.

Three groups:

* yaml / manifest / docs tests read the folder files with pyyaml only and run anywhere;
* tests that use the core (``Benchmark.from_dir`` on a schema-2 yaml, the raw ``RawStream``
  reader, ``yavih.media.rules``) and the ``make_manifest.py --check`` re-derivation from a
  reconstructed ``meta/esc50.csv`` always run;
* the checks against the staged clone (metadata, real-media reader smoke) skip when it is absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_bytes, rows, classes = F.spec, F.manifest_bytes, F.rows, F.classes

FOLDS = ["1", "2", "3", "4", "5"]
CROSS_FOLD_SOURCES = {"131943", "134049", "209698", "234879"}
CLIP_SAMPLES, NATIVE_SR = 220500, 44100
REMOVED_READER_KEYS = {
    "windows", "window_shift", "frame_rule", "frame_rule_params", "frame_grid", "frame_select", "pad_policy",
    "short_clip_policy", "crop", "audio_norm_context", "bridge_decode_allowed",
}
INFO_KEYS = ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No site config, no site census, no media-root override."""
    monkeypatch.setenv("YAVIH_CONFIG", str(tmp_path / "_no_such_config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "yavih_home"))
    monkeypatch.delenv("YAVIH_DATA__ESC50__MEDIA_ROOT", raising=False)
    return tmp_path


def _write_wav(path: Path, n_frames: int, seed: int) -> None:
    sf = pytest.importorskip("soundfile")
    pcm = (np.random.default_rng(seed).uniform(-0.5, 0.5, n_frames) * 32767).astype(np.int16)
    sf.write(path, pcm, NATIVE_SR, subtype="PCM_16")


def _load_template():
    return F.load_module(HERE / "bridge_template.py")


# ----------------------------------------------------------------------------- yaml + files


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_manifest_schema_and_charsets(rows: list[dict[str, str]]):
    assert list(rows[0]) == ["clip_id", "labels", "split", "rel_path", "src_file", "take", "esc10"]
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids) == 2000
    key_re, media_re = re.compile(r"^[A-Za-z0-9_.#:-]+$"), re.compile(r"^[A-Za-z0-9_.#:/-]+$")
    name_re = re.compile(r"^(\d)-(\d+)-([A-Z])-(\d+)$")
    for r in rows:
        assert key_re.match(r["clip_id"]) and media_re.match(r["rel_path"])
        assert r["rel_path"] == f"audio/{r['clip_id']}.wav"
        fold, src, take, _target = name_re.match(r["clip_id"]).groups()
        assert (fold, src, take) == (r["split"], r["src_file"], r["take"])  # filename encodes fold/source/take
        assert r["esc10"] in ("True", "False")
    assert sum(r["esc10"] == "True" for r in rows) == 400
    # fold-major, filename order within a fold (meta/esc50.csv order)
    assert [r["split"] for r in rows] == sorted(r["split"] for r in rows)
    for f in FOLDS:
        fold_ids = [r["clip_id"] for r in rows if r["split"] == f]
        assert fold_ids == sorted(fold_ids)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    s = spec["splits"]
    assert F.assert_splits_assigned_once(spec, rows) == {f: 400 for f in FOLDS}
    assert s["folds"] == FOLDS and s["train"] == [] and s["eval"] == []
    assert spec["budget"]["clips"] == {f: 400 for f in FOLDS}


def test_classes_contiguous_sorted(classes: list[dict[str, str]], rows: list[dict[str, str]]):
    assert list(classes[0]) == ["index", "name"]
    assert [int(c["index"]) for c in classes] == list(range(50))
    names = [c["name"] for c in classes]
    # sorted distinct labels of the full manifest
    assert names == sorted({r["labels"] for r in rows}) and len(set(names)) == 50
    assert names[0] == "airplane" and names[18] == "dog" and names[49] == "wind"
    # balanced: 40 per class, 8 per class per fold
    assert set(Counter(r["labels"] for r in rows).values()) == {40}
    assert set(Counter((r["split"], r["labels"]) for r in rows).values()) == {8}


def test_curation_sums(spec: dict, rows: list[dict[str, str]]):
    up = spec["manifest"]["upstream"]
    cur = up["curation"]
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"] == up["rows"] == 2000
    counts = Counter(r["split"] for r in rows)
    assert sorted(cur["per_split"]) == FOLDS
    for s, v in cur["per_split"].items():
        assert v["listed"] == counts[s] and v["staged"] == v["listed"] - v["missing"]
    assert cur["excluded"] == {} and cur["short_clips"] == 0
    assert cur["leakage_audit"]["excluded"] == 0 and "4 groups / 9 clips" in cur["leakage_audit"]["against"]
    assert spec["protocol"]["train_draw"] is None
    assert up["upstream_files"] == {"ESC-50/meta/esc50.csv": up["sha256"]}
    assert up["sha256"] == spec["source"]["sha256"]["meta/esc50.csv"] and "33c8ce9" in up["source"]


def test_cross_fold_sources_are_the_known_four(rows: list[dict[str, str]]):
    folds_by_src: dict[str, set[str]] = {}
    for r in rows:
        folds_by_src.setdefault(r["src_file"], set()).add(r["split"])
    assert {s for s, f in folds_by_src.items() if len(f) > 1} == CROSS_FOLD_SOURCES
    for s in CROSS_FOLD_SOURCES:  # always under two different classes, one per fold
        labs = {(r["split"], r["labels"]) for r in rows if r["src_file"] == s}
        assert len({lab for _, lab in labs}) == 2


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert m["headline"] == "acc_mean"
    assert m["list"] == ["acc_mean", "acc_pstd", "acc_per_fold", "n_folds", "n_eval"]
    assert m["floors"] == ["acc_floor"] and m["controls"] == []
    assert m["chance"] == {"acc_mean": 0.02} and m["ci"] == {"acc_mean": "pooled_binomial"}
    assert m["aggregation"] == {"over": "folds", "report": ["mean", "pstd", "per_fold"]}


def test_schema_and_protocol(spec: dict):
    """Schema 2, protocol_version 1 with one changelog entry, no removed reader key anywhere."""
    assert spec["schema_version"] == 3
    assert spec["protocol_version"] == 1 == spec["changelog"][-1]["version"]
    assert [c["version"] for c in spec["changelog"]] == [1]
    assert spec["changelog"][0]["invalidates"] is False and spec["changelog"][0]["supersedes"] == []
    assert not (F.walk_keys(spec) & REMOVED_READER_KEYS), F.walk_keys(spec) & REMOVED_READER_KEYS
    assert "requires" not in spec  # moved into output_contract.requires
    assert spec["protocol"] == {"clip_seconds": 5.0, "train_draw": None, "views": "none"}
    assert F.file_fields(spec) == {
        "audio": {"kind": "audio", "media_column": "rel_path", "root": "media_root", "layout": "dir", "layout_params": {}, "required": True}
    }
    assert spec["scorer"] == "clf" and spec["readout_params"] == {} and spec["targets"] is None
    assert spec["data"]["derived"] == {} and spec["data"]["tar_index"] is None


def test_input_contract(spec: dict):
    ic = spec["input_contract"]
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(ic["info"]) == set(F.file_fields(spec)) == {"audio"}
    assert "as stored" in ic["handed_over"]["audio"] and "44.1 kHz" in ic["handed_over"]["audio"]
    assert ic["info"]["audio"] == INFO_KEYS
    assert ic["facts"] == [] and ic["materialize"] is None


def test_output_contract_and_external(spec: dict):
    assert spec["output_contract"] == {
        "shape": "fixed",
        "coordinates": "media",
        "target_frame": None,
        "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None},
    }
    assert spec["external"] == []
    assert not (HERE / "eval.py").exists()  # no authors' scorer: accuracy is yavih's


def test_yaml_parses_via_benchmark_from_dir(isolated: Path):
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE)
    assert b.name == "esc50" == HERE.name
    assert b.spec.readout == "clf" and b.spec.manifest.label.kind == "single"
    assert [b.n_rows(f) for f in FOLDS] == [400] * 5
    assert b.unknown_labels == 0
    assert b.roles() == {"train": [], "eval": [], "floor": [], "folds": FOLDS}
    assert len(b.protocol_hash) == 64 and len(b.manifest_hash) == 64
    # single-label cells parse to the sorted-name index: dog = 18, not the upstream target 0
    m = b.manifest("1")
    assert m.clip_id[0] == "1-100032-A-0" and m.label_ids[0] == 18
    assert m.media[0] == {"audio": "audio/1-100032-A-0.wav"}
    assert m.extra[0]["src_file"] == "100032" and m.extra[0]["take"] == "A" and m.extra[0]["esc10"] == "True"
    view = b.view()
    assert view.streams == {"audio": "audio"} and view.facts == {} and view.protocol_version == 1
    assert not hasattr(view, "manifest")


# ----------------------------------------------------------------------------- README.md + setup.md


def test_readme_facts(spec: dict):
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    cite = F.section(text, 2, "Citation")
    assert "```bibtex" in cite and "piczak2015dataset" in cite and "10.1145/2733373.2806390" in cite
    for needle in ("2 000", "400", "acc_floor", "0.02", "not comparable", "ESC-10", "make_manifest.py", "--check",
                   spec["manifest"]["sha256"][:12]):
        assert needle.lower() in text.lower(), needle


def test_setup_facts():
    """Folder facts in setup.md (the setup.md structure and contract copies are ``yavih bench lint``'s)."""
    text = (HERE / "setup.md").read_text(encoding="utf-8")
    assert "--census" in F.section(text, 2, "Prepare & verify")
    for needle in (
        "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6", "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246",
        "f6aec3a3283a2efdcbb1feff334adad7ab17c097bdec0cbb4eebdb9e16886074",
        "52754b9099c1c48a9fec379bd7dd75dbc6d78387faecc2a9318fc0c43332d398", "645 520 453", "CC BY-NC 3.0",
        "131943", "234879", "e413d9039c94", "pad_to_s", "44.1 kHz",
    ):
        assert needle in text, needle


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir())
    assert names == ["01_download.sh", "02_verify.sh", "03_make_manifest.sh"]
    dl = (HERE / "prepare" / "01_download.sh").read_text()
    assert "git clone --depth 1" in dl and "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6" in dl
    assert "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246" in dl
    verify = (HERE / "prepare" / "02_verify.sh").read_text()
    assert "52754b9099c1c48a9fec379bd7dd75dbc6d78387faecc2a9318fc0c43332d398" in verify
    assert "(220500, 44100, 1, \"PCM_16\"): 2000" in verify


def test_folder_ships_no_reader_py():
    """The folder ships no reader.py: ``Benchmark.reader()`` falls back to the generic
    ``yavih.sample.iter_samples``, which hands over what benchmark.yaml declares; the reader smokes in this
    file go through ``bench.reader``."""
    assert not (HERE / "reader.py").exists()


# ----------------------------------------------------------------------------- bridge template


class _FakeRawStream:
    """Stand-in for ``yavih.sample.RawStream`` over a local wav."""

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
    mod = _load_template()
    src = (HERE / "bridge_template.py").read_text()
    assert "Preprocessing source of the model" in src and "TODO" in src
    assert "decode_plan" not in src and "waveform(" not in src
    cls = mod.Esc50Bridge
    bridge = cls(SimpleNamespace(name="esc50"), {})
    d = bridge.describe()
    for key in ("weights_sha256", "weights_ref", "dtype", "batch_size", "streams", "pos_mode", "reading", "model",
                "layer", "pooling", "views", "input", "precision", "args"):
        assert key in d, key
    assert d["reading"] == "a"
    assert "decode" not in d and d["views"] == 1 and d["streams"] == ["audio"]
    a = d["input"]["audio"]
    assert set(a) == {"temporal", "front_end", "norm", "decoder"}
    assert a["temporal"] == {"rule": "full", "max_s": None, "pad_to_s": None}
    assert "sample_rate" in a["front_end"] and "resample" in a["front_end"] and "features" in a["front_end"]
    assert "source" in a["norm"] and a["decoder"]["lib"] == "soundfile"
    assert all(v is not None for k, v in d["precision"].items() if k != "autocast")
    assert d["args"] == {"layer": -1}


@pytest.mark.parametrize("pad_to_s", [None, 10.0])
def test_bridge_template_output_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pad_to_s):
    """encode returns (feats [B, N, D], pos, used) with ``used`` = the declared rule evaluated on each clip's
    facts (what validate's E_RULE_MISMATCH recomputes), N/D fixed even for a short clip, pos None. Uses
    core's ``yavih.media.rules``."""
    pytest.importorskip("soundfile")
    from yavih.media.rules import audio_ranges

    lengths = [CLIP_SAMPLES, CLIP_SAMPLES, 220000]
    streams = []
    for k, n in enumerate(lengths):
        p = tmp_path / f"c{k}.wav"
        _write_wav(p, n, k)
        streams.append(_FakeRawStream(p))
    mod = _load_template()
    cls = type("B", (mod.Esc50Bridge,), {"TEMPORAL": {"rule": "full", "max_s": pad_to_s, "pad_to_s": pad_to_s}})
    bridge = cls(SimpleNamespace(name="esc50"), {})
    samples = [SimpleNamespace(row=k, clip_id=f"c{k}", split="1", streams={"audio": s}, meta={"facts": {}}) for k, s in enumerate(streams)]
    feats, pos, used = F.encode_prepared(bridge, samples)
    assert pos is None and len(used) == 3 and np.asarray(feats).ndim == 3 and np.asarray(feats).shape[0] == 3
    for s, n, u in zip(streams, lengths, used, strict=True):
        ranges, (pl, pr) = audio_ranges(bridge.TEMPORAL, s.info)
        assert u == {"audio": {"samples": [[int(lo), int(hi)] for lo, hi in np.asarray(ranges).tolist()], "pad": [int(pl), int(pr)]}}
        want_pad = 0 if pad_to_s is None else int(round(pad_to_s * NATIVE_SR)) - n
        assert u["audio"]["samples"] == [[0, n]] and u["audio"]["pad"] == [0, want_pad]
        wav, sr, _ = bridge._read(s)
        assert sr == NATIVE_SR and wav.shape == (n + want_pad,) and wav.dtype == np.float32
        assert np.all(wav[n:] == 0) and float(np.abs(wav).max()) <= 1.0
    # media-coordinate time pos (optional constant pos): padding-only tokens carry t = -1, audio y = x = -1
    bridge.n_t, bridge.n_f = 5, 2
    p = bridge._time_pos(5.0)
    assert p.shape == (10, 4) and np.all(p[:, 0] == 0) and np.all(p[:, 2:] == -1)
    assert np.allclose(p[::2, 1], [0.1, 0.3, 0.5, 0.7, 0.9])


# ----------------------------------------------------------------------------- make_manifest.py re-derivation


def _metadata_bytes(rows: list[dict[str, str]]) -> bytes:
    """``meta/esc50.csv`` rebuilt from the committed manifest (the upstream ``target`` is the filename's last
    field); byte-identical to the pinned upstream file."""
    lines = ["filename,fold,target,category,esc10,src_file,take"]
    lines += [f"{r['clip_id']}.wav,{r['split']},{r['clip_id'].rsplit('-', 1)[1]},{r['labels']},{r['esc10']},{r['src_file']},{r['take']}" for r in rows]
    return ("\n".join(lines) + "\n").encode()


def test_make_manifest_check_from_metadata(tmp_path: Path, rows: list[dict[str, str]], capsys: pytest.CaptureFixture):
    """``make_manifest.py --check`` re-derives manifest.csv.gz and classes.csv byte-identically from the public
    ``meta/esc50.csv`` alone, and prints the committed ``manifest.upstream`` block."""
    mm = F.load_module(HERE / "make_manifest.py")
    meta = tmp_path / "esc50.csv"
    meta.write_bytes(_metadata_bytes(rows))
    assert hashlib.sha256(meta.read_bytes()).hexdigest() == mm.METADATA_SHA256
    assert mm.main(["--no-media", "--metadata", str(meta), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    y = F.load_yaml(HERE)
    assert report["manifest"]["sha256"] == y["manifest"]["sha256"] and report["manifest"]["classes_sha256"] == y["manifest"]["classes_sha256"]
    up = {k: v for k, v in y["manifest"]["upstream"].items() if k != "note"}
    assert report["upstream"] == up
    # a changed metadata file is refused (pinned sha256)
    bad = tmp_path / "bad.csv"
    bad.write_bytes(meta.read_bytes().replace(b"1-100032-A-0.wav,1,0,dog", b"1-100032-A-0.wav,1,0,cat", 1))
    with pytest.raises(SystemExit):
        mm.main(["--no-media", "--metadata", str(bad), "--check"])


META_ROOT = F.site_root(HERE, "media_root", needs=("meta/esc50.csv",))


@pytest.mark.skipif(META_ROOT is None, reason="ESC-50 clone not staged on this machine")
def test_staged_metadata_is_the_pinned_file(rows: list[dict[str, str]]):
    assert (META_ROOT / "meta" / "esc50.csv").read_bytes() == _metadata_bytes(rows)


# ----------------------------------------------------------------------------- reader smoke on synthetic raw files


def test_reader_smoke_synthetic_raw(isolated: Path):
    """Runs anywhere: the reader hands over each stored wav byte for byte with its header facts and
    ``meta == {"facts": {}}``; a short file is handed over as stored (nothing cut, padded or refused);
    canaries equal their originals."""
    from yavih import Benchmark

    probe = Benchmark.from_dir(HERE)
    m = probe.manifest("1")
    media = isolated / "media"
    (media / "audio").mkdir(parents=True)
    lengths = [CLIP_SAMPLES] * 8
    lengths[2] = 220000
    for i, n in enumerate(lengths):
        _write_wav(media / "audio" / f"{m.clip_id[i]}.wav", n, i)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    samples = list(b.reader("1", rows=range(0, 8), batch_size=1, canaries=1))
    assert [s.row for s in samples[:8]] == list(range(8)) and len(samples) == 9
    for s, n in zip(samples[:8], lengths, strict=True):
        assert set(s.streams) == {"audio"} and s.meta == {"facts": {}} and s.split == "1"
        st = s.streams["audio"]
        f = media / "audio" / f"{s.clip_id}.wav"
        assert st.kind == "audio" and st.name == "audio" and Path(st._real_path).resolve() == f.resolve() and Path(st.path).resolve() == f.resolve()
        assert st.bytes() == f.read_bytes()
        with st.open() as fh:
            assert fh.read() == f.read_bytes()
        assert {k: st.info[k] for k in INFO_KEYS} == {
            "duration_s": pytest.approx(n / NATIVE_SR), "sample_rate": NATIVE_SR, "n_samples": n, "channels": 1,
            "container": "wav", "bytes": f.stat().st_size,
        }
        assert not hasattr(st, "waveform")
    c = samples[8]
    assert c._canary_of == c.row and c.row < 6 and c.meta == samples[c.row].meta
    assert c.streams["audio"].bytes() == samples[c.row].streams["audio"].bytes()


def test_reader_to_template_end_to_end(isolated: Path):
    """The template consumes what the reader hands over and reports ``used`` inside the media."""
    from yavih import Benchmark
    from yavih.bridge import validate_describe

    probe = Benchmark.from_dir(HERE)
    m = probe.manifest("2")
    media = isolated / "media"
    (media / "audio").mkdir(parents=True)
    for i in range(2):
        _write_wav(media / "audio" / f"{m.clip_id[i]}.wav", CLIP_SAMPLES, 10 + i)
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    mod = _load_template()
    bridge = mod.Esc50Bridge(b.view(), {})
    validate_describe(bridge.describe(), bridge, b)
    samples = list(b.reader("2", rows=range(0, 2)))
    feats, pos, used = F.encode_prepared(bridge, samples)
    assert np.asarray(feats).shape[0] == 2 and pos is None
    assert [u["audio"] for u in used] == [{"samples": [[0, CLIP_SAMPLES]], "pad": [0, 0]}] * 2


# ----------------------------------------------------------------------------- reader smoke on local media


MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("audio",))  # resolved at collection, before `isolated` hides the site config


@pytest.mark.skipif(MEDIA_ROOT is None, reason="ESC-50 media not staged on this machine")
def test_reader_smoke_real_media(isolated: Path):
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": str(MEDIA_ROOT)})
    samples = list(b.reader("3", rows=range(0, 4)))
    assert [s.row for s in samples] == [0, 1, 2, 3]
    for s in samples:
        st = s.streams["audio"]
        assert st.info["n_samples"] == CLIP_SAMPLES and st.info["sample_rate"] == NATIVE_SR
        assert st.info["channels"] == 1 and st.info["bytes"] == 441044 and st.info["duration_s"] == pytest.approx(5.0)
        raw = st.bytes()
        assert len(raw) == 441044 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
        assert s.meta == {"facts": {}}
