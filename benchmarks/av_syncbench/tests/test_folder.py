"""Folder tests of av_syncbench (run: ``python -m pytest "$(yavih bench path av_syncbench)/tests" -q -p no:cacheprovider``).

Pure checks of the frozen artefacts (always run): hashes, rows, columns, curation sums, the re-derivation of the
manifest from the two upstream metadata files (rebuilt byte for byte from the manifest itself); the scorer on
hand-checked inputs (the segment selection against the authors' torch arithmetic, the strict decision, the
mean-of-levels aggregation, the copied upstream summary); the bridge template; a reader smoke on synthetic raw media.
Checks that need the staged release skip elsewhere. ``yavih bench lint`` (run once per folder by the repository)
checks the layout, the yaml, the hashes, the conformance fixture and the docs' structure.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows, mm, bench, template = F.spec, F.rows, F.mm, F.bench, F.template

READOUT = "av_syncbench"
HEADLINE = "acc_task_mean"
STREAMS = {"video": "video", "audio": "audio", "edited_audio": "audio"}
CLIPS = {"test": 37743}
READINGS = ["av"]
COLUMNS = ["clip_id", "split", "task", "level", "direction", "category", "video_id", "rel_video", "rel_edited", "duration",
           "target_attribute", "source_attribute", "upstream_line"]
TASKS = {"offset": 14600, "speed": 14570, "jitter": 7755, "voice_timbre": 589, "instrument_timbre": 229}
#: torch.linspace(0, n - 1, steps=18).round().long() (torch 2.x), the authors' paper_legacy segment selection
TORCH_SELECTION = {
    19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    20: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19],
    21: [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20],
    22: [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21],
    23: [0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 21, 22],
    24: [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23],
    25: [0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 20, 21, 23, 24],
}


def _pairwise():
    """The folder's scorer module (``pairwise.py``, loaded by ``eval.py``)."""
    return F.load_module(HERE / "eval.py").pairwise


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE
    assert spec["metrics"]["list"][:-1] == _pairwise().metric_names() and spec["metrics"]["list"][-1] == "n_eval"
    assert set(spec["metrics"]["chance"]) == {m for m in spec["metrics"]["list"] if m.startswith("acc_")}
    assert set(spec["metrics"]["chance"].values()) == {0.5} and spec["metrics"]["floors"] == []


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert spec["fields"]["video"]["media_column"] == spec["fields"]["audio"]["media_column"] == "rel_video"
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS) and ic["facts"] == ["segment_seconds"]
    assert spec["external"] == [] and (HERE / "eval.py").is_file()
    assert spec["protocol"] == {"clip_seconds": None, "train_draw": None, "views": "rows", "segment_seconds": 0.64, "max_segments": 18}
    oc = spec["output_contract"]
    assert set(oc["schema"]) == {"video_seg", "audio_seg", "edited_seg"}
    assert oc["isolation"] == {"video_seg": ["video"], "audio_seg": ["audio"], "edited_seg": ["edited_audio"]}
    assert spec["splits"]["eval"] == ["test"] and spec["splits"]["train"] == [] and spec["readout_params"] == {}


def test_rows_columns_and_counts(rows: list[dict[str, str]], mm):
    assert list(rows[0]) == COLUMNS and len(rows) == 37743
    assert Counter(r["task"] for r in rows) == TASKS
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert len({r["rel_edited"] for r in rows}) == len(rows) and len({r["rel_video"] for r in rows}) == 3494
    groups = [mm.group_of(r) for r in rows]
    assert set(groups) == set(mm.GROUP_ORDER) and len(mm.GROUP_ORDER) == 20
    assert groups[:20] == list(mm.GROUP_ORDER)  # --sample 20 (or more) covers every scored group
    assert all(Counter(groups[:20 * k]) == {g: k for g in mm.GROUP_ORDER} for k in (1, 2, 50, 229))
    for g in mm.GROUP_ORDER:  # inside a group: rel_video, then upstream_line
        keys = [(r["rel_video"], r["upstream_line"][0], int(r["upstream_line"][1:])) for r, gr in zip(rows, groups, strict=True) if gr == g]
        assert keys == sorted(keys)
    lv = Counter((r["task"], r["level"]) for r in rows if r["task"] in ("offset", "speed", "jitter"))
    assert set(lv.values()) == {2920, 1457, 2585} and len(lv) == 18
    for r in rows:
        d = "hf_temporal_challenge" if r["upstream_line"].startswith("t") else "hf_timbre_edit"
        assert r["rel_video"].startswith(f"AV_SyncBench/{d}/original_video/") and r["rel_edited"].startswith(f"AV_SyncBench/{d}/edited_audio/")
        assert (r["direction"] in ("advance", "delay")) == (r["task"] == "offset")


def test_curation_sums(spec: dict):
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["total_upstream"] == 36925 + 821 == spec["manifest"]["upstream"]["rows"]
    assert cur["excluded"] == {"duplicate_line": 3}
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"]


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_download.sh", "02_extract.sh", "03_check_manifest.py", "04_verify.sh"]
    text = (HERE / "prepare" / "01_download.sh").read_text(encoding="utf-8")
    y = F.load_yaml(HERE)
    assert y["source"]["revision"] in text and all(sha in text for sha in y["source"]["sha256"].values())


# ----------------------------------------------------------------------------- make_manifest: the metadata rebuilt from the manifest


def _metadata_files(rows: list[dict[str, str]], mm) -> dict[str, bytes]:
    """Both upstream metadata.jsonl files, byte for byte, from the manifest's raw columns and the pinned duplicates."""
    lines: dict[str, dict[int, str]] = {d: {} for d in mm.FILES}
    for r in rows:
        d = "hf_temporal_challenge" if r["upstream_line"][0] == "t" else "hf_timbre_edit"
        rec = {"sample_id": r["clip_id"], "video_id": r["video_id"],
               "original_video": r["rel_video"].split("/", 2)[2], "edited_audio": r["rel_edited"].split("/", 2)[2],
               "edit_type": mm.EDIT_TYPE.get(r["task"], ""), "target_attribute": json.loads(r["target_attribute"])}
        if d == "hf_timbre_edit":
            rec["source_attribute"] = json.loads(r["source_attribute"])
        rec.update({"category": r["category"], "duration": json.loads(r["duration"]), "task_type": mm.TASK_TYPE[d]})
        lines[d][int(r["upstream_line"][1:])] = json.dumps(rec)
    for d, dups in mm.DUPLICATE_LINES.items():
        for line, of in dups.items():
            lines[d][line] = lines[d][of]
    out = {}
    for d, ls in lines.items():
        assert sorted(ls) == list(range(1, mm.FILES[d][1] + 1))
        out[d] = ("\n".join(ls[i] for i in sorted(ls)) + "\n").encode("utf-8")
    return out


def test_make_manifest_check(tmp_path: Path, rows, mm, capsys):
    """``make_manifest.py --no-media --metadata DIR --check`` re-derives manifest.csv.gz byte for byte from the two
    upstream files (rebuilt here from the manifest, matching their pinned sha256) and prints the yaml's upstream block."""
    for d, data in _metadata_files(rows, mm).items():
        assert hashlib.sha256(data).hexdigest() == mm.FILES[d][0]
        (tmp_path / d).mkdir()
        (tmp_path / d / "metadata.jsonl").write_bytes(data)
    assert mm.main(["--no-media", "--metadata", str(tmp_path), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    y = F.load_yaml(HERE)
    assert report["manifest"]["sha256"] == y["manifest"]["sha256"]
    assert report["upstream"] == {k: v for k, v in y["manifest"]["upstream"].items() if k != "note"}
    bad = tmp_path / "hf_timbre_edit" / "metadata.jsonl"
    bad.write_bytes(bad.read_bytes().replace(b"solo_flute", b"solo_oboe", 1))
    with pytest.raises(SystemExit):
        mm.main(["--no-media", "--metadata", str(tmp_path), "--check"])


def test_parse_task(mm):
    t = "hf_temporal_challenge"
    assert mm.parse_task(t, {"target_attribute": "audio_shift: delay 300ms"}) == ("offset", "300ms", "delay")
    assert mm.parse_task(t, {"target_attribute": "audio_speed: 1.05x"}) == ("speed", "1.05x", "")
    jit = "audio_local_jitter: seg=10s level=L2 range=[150,250]ms windows=2s picked=[{'start_sec': 2, 'dur_sec': 2, 'shift_ms': 184}]"
    assert mm.parse_task(t, {"target_attribute": jit}) == ("jitter", "L2", "")
    assert mm.parse_task("hf_timbre_edit", {"edit_type": "instrument_timbre_edit", "target_attribute": "solo_violin"}) == ("instrument_timbre", "solo_violin", "")
    with pytest.raises(SystemExit):
        mm.parse_task(t, {"sample_id": "x", "target_attribute": "audio_pitch: +1"})


MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("AV_SyncBench/hf_temporal_challenge/metadata.jsonl", "AV_SyncBench/hf_timbre_edit/metadata.jsonl"))


@pytest.mark.skipif(MEDIA_ROOT is None, reason="the AV-SyncBench release is not extracted on this machine")
def test_staged_metadata_is_the_pinned_release(rows, mm):
    for d, data in _metadata_files(rows, mm).items():
        assert (MEDIA_ROOT / "AV_SyncBench" / d / "metadata.jsonl").read_bytes() == data


# ----------------------------------------------------------------------------- the scorer


def test_segment_selection_matches_the_authors_torch_arithmetic():
    P = _pairwise()
    for n, expected in TORCH_SELECTION.items():
        assert P.segment_indices(n, 18).tolist() == expected
    assert P.segment_indices(18, 18).tolist() == list(range(18)) and P.segment_indices(5, None).tolist() == list(range(5))
    assert P.segment_indices(0, 18).size == 0
    assert P.non_overlapping_starts(1.28, 0.64) == [0.0, 0.64] and len(P.non_overlapping_starts(1.2799, 0.64)) == 1


def test_pair_scores_against_a_naive_loop():
    P = _pairwise()
    rng = np.random.default_rng(0)
    v, a, e = rng.standard_normal((21, 8)), rng.standard_normal((23, 8)), rng.standard_normal((20, 8))
    pos, neg, m = P.pair_scores(v, a, e, max_segments=18)
    keep = TORCH_SELECTION[20]  # the common duration holds min(21, 23, 20) = 20 segments

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    assert m == 18
    assert pos == pytest.approx(np.mean([cos(v[k], a[k]) for k in keep]), abs=1e-12)
    assert neg == pytest.approx(np.mean([cos(v[k], e[k]) for k in keep]), abs=1e-12)
    pos_all, _, m_all = P.pair_scores(v, a, e, max_segments=None)
    assert m_all == 20 and pos_all == pytest.approx(np.mean([cos(v[k], a[k]) for k in range(20)]), abs=1e-12)
    assert np.isnan(P.pair_scores(v[:0], a, e, max_segments=18)[0])  # no common segment: invalid
    with pytest.raises(ValueError):
        P.pair_scores(v, a[:, :4], e, max_segments=18)


def _meta(tasks, levels):
    return {"task": list(tasks), "level": list(levels), "n_eval": len(tasks)}


def test_metrics_on_hand_checked_pairs():
    P = _pairwise()
    m = P.make_metrics(READOUT)
    # offset: 50ms 2/2 right, 100ms 1/2 (one tie); jitter L1 0/1; speed 0.80x 1/1 and one invalid row; timbre 1/2
    tasks = ["offset", "offset", "offset", "offset", "jitter", "speed", "speed", "voice_timbre", "instrument_timbre"]
    levels = ["50ms", "50ms", "100ms", "100ms", "L1", "0.80x", "0.80x", "gender=male", "solo_flute"]
    out = np.array([[0.9, 0.1], [0.5, 0.4], [0.3, 0.3], [0.6, 0.2], [0.1, 0.2], [0.7, 0.6], [np.nan, 0.1], [0.2, 0.1], [0.0, 0.5]])
    meta = _meta(tasks, levels)

    def val(name):
        return m[name].fn(out, None, {}, meta)

    assert val("acc_offset_50ms") == 1.0 and val("acc_offset_100ms") == 0.5 and val("acc_offset") == 0.75
    assert val("acc_jitter") == 0.0 and val("acc_speed") == 1.0 and val("acc_speed_080") == 1.0
    assert val("acc_timbre") == 0.5 and val("acc_voice_timbre") == 1.0 and val("acc_instrument_timbre") == 0.0
    assert val("acc_task_mean") == pytest.approx((0.75 + 0.0 + 1.0 + 0.5) / 4)
    assert val("acc_all") == pytest.approx(5 / 8)
    assert val("tie_rate") == pytest.approx(1 / 8) and val("invalid_rate") == pytest.approx(1 / 9)
    assert np.isnan(val("acc_offset_500ms"))  # an empty level: undefined, and left out of acc_offset
    pc = m["acc_task_mean"].per_clip(out, None, {}, meta)
    assert np.nanmean(pc) == pytest.approx(val("acc_task_mean")) and np.isnan(pc[6])
    lo, hi = m["acc_task_mean"].interval("bootstrap_clip", out, None, {}, {**meta, "bootstrap": {"samples": 200, "seed": 0}})
    assert 0.0 <= lo <= hi <= 1.0


def test_acc_all_is_the_copied_upstream_summary():
    P = _pairwise()
    rng = np.random.default_rng(3)
    out = np.round(rng.uniform(-1, 1, (200, 2)), 1)  # coarse values: ties happen
    out[7] = np.nan
    tasks = ["offset"] * 200
    res = [P.PairwiseResult(sample_id=str(i), task="offset", positive_score=float(o[0]), negative_score=float(o[1])) for i, o in enumerate(out)]
    summary = P.summarize_results(res)["overall"]
    m = P.make_metrics(READOUT)
    meta = _meta(tasks, ["50ms"] * 200)
    assert m["acc_all"].fn(out, None, {}, meta) == pytest.approx(summary["accuracy"])
    assert summary["valid"] == 199 and summary["ties"] > 0
    assert m["tie_rate"].fn(out, None, {}, meta) == pytest.approx(summary["ties"] / summary["valid"])


def test_benchmark_loads_with_the_folder_scorer(bench):
    assert bench.spec.readout == HERE.name
    ev = bench.eval_module()
    assert ev.SCORER.output_keys == frozenset({"video_seg", "audio_seg", "edited_seg"}) and ev.SCORER.kind == "stateless"
    assert ev.PROTOCOL_KEYS == frozenset({"segment_seconds", "max_segments"})


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys(template):
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED_CORE, Bridge

    (cls,) = [v for v in vars(template).values() if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == template.__name__]
    d = cls(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED_CORE) | {"outputs"} <= set(d), set(DESCRIBE_REQUIRED_CORE) - set(d)
    assert d["reading"] in READINGS and set(d["input"]) == set(STREAMS) and tuple(cls.outputs) == ("video_seg", "audio_seg", "edited_seg")
    assert template.n_segments(1.28, 0.64) == 2 and template.n_segments(1.2799, 0.64) == 1


# ----------------------------------------------------------------------------- reader smoke on synthetic raw files


def test_reader_smoke_synthetic_raw(tmp_path: Path, rows, spec):
    """The first two rows (two synthetic originals, each an mp4 with its own AAC track) and their wavs under a temporary
    root: the reader hands each mp4 over as both ``video`` and ``audio`` and the wav as ``edited_audio``, byte for
    byte, with the guaranteed media facts and ``meta == {"facts": {"segment_seconds": 0.64}}``."""
    pytest.importorskip("av")
    from yavih import Benchmark
    from yavih.codecs import wav_bytes

    synth = F.load_module(HERE / "sync_synth.py")
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    man = b.manifest("test")
    picked = [0, 1]
    mp4s = {}
    stored_wavs = {}
    for k in picked:
        mp4s[k] = tmp_path / man.media[k]["video"]
        if not mp4s[k].is_file():
            synth._write_video(mp4s[k], np.tile([1, 0], 10), 0.25)
        p = tmp_path / man.media[k]["edited_audio"]
        p.parent.mkdir(parents=True, exist_ok=True)
        data = wav_bytes(np.zeros(16000 * (k + 2), dtype=np.float32), 16000)
        p.write_bytes(data)
        stored_wavs[k] = data
    samples = list(b.reader("test", rows=range(0, 2)))
    assert [s.row for s in samples] == picked
    for s in samples:
        assert set(s.streams) == set(STREAMS) and s.meta == {"facts": {"segment_seconds": 0.64}}
        v, a, e = s.streams["video"], s.streams["audio"], s.streams["edited_audio"]
        assert v.bytes() == a.bytes() == mp4s[s.row].read_bytes() and e.bytes() == stored_wavs[s.row]
        for name, st in (("video", v), ("audio", a), ("edited_audio", e)):
            missing = [k for k in spec["input_contract"]["info"][name] if st.info.get(k) is None]
            assert not missing, (name, missing)
        assert e.info["sample_rate"] == 16000 and e.info["n_samples"] == 16000 * (s.row + 2) and a.info["sample_rate"] == 16000
