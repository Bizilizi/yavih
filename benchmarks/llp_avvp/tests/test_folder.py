"""Folder tests of llp_avvp (run: ``python -m pytest "$(yavih bench path llp_avvp)/tests" -q -p no:cacheprovider``).

What nothing else checks: the frozen rows and their label facts, the scorer on hand-checked inputs and against the
authors' evaluation loop, the per-second pooling, the HAN head and loss, the floor, the re-derivation of the
manifest, the reader on synthetic media and the stand-in. ``yavih bench lint`` (run once per folder by the
repository) checks the layout, the yaml, the hashes, the conformance fixture and the README.md / setup.md structure.
Tests that need the staged annotation files skip where they are not configured.
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
spec, rows, classes = F.spec, F.rows, F.classes

READOUT = "llp_avvp"
HEADLINE = "seg_type_av"
STREAMS = {"video": "video", "audio": "audio"}
CLIPS = {"train": 10000, "val": 649, "test": 1200}
READINGS = ["a", "av", "v"]
CATEGORIES = ["Speech", "Car", "Cheering", "Dog", "Cat", "Frying_(food)", "Basketball_bounce", "Fire_alarm", "Chainsaw", "Cello", "Banjo",
              "Singing", "Chicken_rooster", "Violin_fiddle", "Vacuum_cleaner", "Baby_laughter", "Accordion", "Lawn_mower", "Motorcycle",
              "Helicopter", "Acoustic_guitar", "Telephone_bell_ringing", "Baby_cry_infant_cry", "Blender", "Clapping"]
ANNOT = F.site_root(HERE, "annotations_root", needs=("AVVP_test_pd.csv", "AVVP_eval_audio.csv"))


def _avvp():
    return F.load_module(HERE / "eval.py", "llp_avvp_eval_under_test").avvp


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]
    assert spec["splits"]["train"] == ["train", "val"] and spec["splits"]["eval"] == ["test"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["scorer"] == READOUT and spec["metrics"]["headline"] == HEADLINE
    assert spec["metrics"]["floors"] == [f"{m}_floor" for m in ("seg_type_av", "seg_event_av", "evt_type_av", "evt_event_av")]


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)
    assert ic["facts"] == ["num_segments", "segment_seconds"] and spec["protocol"]["num_segments"] == 10
    assert spec["external"] == [] and (HERE / "eval.py").is_file()
    assert spec["output_contract"]["requires"] == {"pos": ["t"], "streams": [], "grid": None, "t_bins": 8}


def test_classes_are_the_authors_order(classes: list[dict[str, str]]):
    assert [c["name"] for c in classes] == CATEGORIES and [int(c["index"]) for c in classes] == list(range(25))
    mm = F.load_module(HERE / "make_manifest.py")
    assert list(mm.CATEGORIES) == CATEGORIES


def test_manifest_rows(rows: list[dict[str, str]]):
    assert list(rows[0]) == ["clip_id", "split", "labels", "rel_media", "ytid", "start_s", "audio_events", "visual_events"]
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids) == 11849
    assert [r["split"] for r in rows] == ["train"] * 10000 + ["val"] * 649 + ["test"] * 1200
    key_re, media_re = re.compile(r"^[A-Za-z0-9_.#:-]+$"), re.compile(r"^llp_videos/[A-Za-z0-9_-]{11}\.mp4$")
    names = set(CATEGORIES)
    for r in rows:
        assert key_re.match(r["clip_id"]) and media_re.match(r["rel_media"])
        assert r["ytid"] == r["clip_id"][:11] and r["rel_media"] == f"llp_videos/{r['ytid']}.mp4"
        start, end = r["clip_id"][12:].split("_")
        assert r["start_s"] == start and float(end) - float(start) == pytest.approx(10.0)
        labels = r["labels"].split(";")
        assert labels and set(labels) <= names and labels == sorted(set(labels), key=CATEGORIES.index)
        if r["split"] == "train":
            assert r["audio_events"] == r["visual_events"] == ""
    assert len({r["ytid"] for r in rows}) == 11849  # one clip per YouTube video: no overlap between splits


def test_dense_label_facts(rows: list[dict[str, str]]):
    """The dense rows as the authors' evaluation reads them: 4 131 - 14 audio and 2 495 - 11 visual rows land on a
    val/test video (the rest name none); 5 / 121 test videos have no audio / visual event."""
    avvp = _avvp()
    tok = Counter()
    empty = Counter()
    for r in rows:
        for m in ("audio_events", "visual_events"):
            ev = avvp.parse_events(r[m])
            tok[m] += len(ev)
            empty[(r["split"], m)] += int(not ev and r["split"] != "train")
    assert tok == {"audio_events": 4117, "visual_events": 2484}
    assert empty[("test", "audio_events")] == 5 and empty[("test", "visual_events")] == 121
    assert empty[("val", "audio_events")] == 4 and empty[("val", "visual_events")] == 70
    by_id = {r["clip_id"]: r for r in rows}
    assert by_id["19jWnmYnlD8_30_40"]["audio_events"] == by_id["19jWnmYnlD8_30_40"]["visual_events"] == "0:10:13"  # Clapping: filed under _30_400
    assert by_id["JfZEqeLZqWk_30_40"]["audio_events"] == "0:10:22;0:4:0;7:10:0"  # the 2024 fix of the audio file


def test_curation_sums(spec: dict):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"] == 11849
    assert {s: v["listed"] for s, v in cur["per_split"].items()} == CLIPS


# ----------------------------------------------------------------------------- the scorer on hand-checked inputs


def _outputs_from(so_a: np.ndarray, so_v: np.ndarray, glob: np.ndarray) -> np.ndarray:
    """Raw outputs ``[1, 23, 25]`` from 0/1 grids ``[25, 10]`` and a video-level 0/1 ``[25]``."""
    out = np.zeros((1, 23, 25), dtype=np.float32)
    out[0, 0:10] = so_a.T
    out[0, 10:20] = so_v.T
    out[0, 20] = glob
    return out


def test_gt_matrices_follow_the_authors():
    avvp = _avvp()
    gt = avvp.gt_matrices("0:4:0;7:10:0;9:9:3;8:0:2", "2:5:3")
    assert gt.shape == (2, 25, 10)
    assert gt[0, 0].tolist() == [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    assert gt[0, 3].sum() == 0 and gt[0, 2].sum() == 0  # offset <= onset marks nothing (GT[idx, x1:x2] = 1)
    assert gt[1, 3].tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        avvp.parse_events("0:4")


def test_scores_on_hand_checked_inputs():
    avvp = _avvp()
    gt = avvp.gt_matrices("0:4:0", "0:10:1")[None]  # speech heard 0-4 s, a car seen throughout
    so_a, so_v, glob = np.zeros((25, 10)), np.zeros((25, 10)), np.zeros(25)
    so_a[0, 0:4] = 1
    so_v[1, :] = 1
    glob[[0, 1]] = 1
    per = avvp.video_scores(_outputs_from(so_a, so_v, glob), gt)
    assert per[0].tolist() == [1.0] * 8  # perfect parse; audio-visual: nothing annotated, nothing predicted -> 1
    s = avvp.summary(per)
    assert s["seg_type_av"] == s["evt_event_av"] == 100.0
    # the video-level mask: frame predictions of a class whose video-level probability is < 0.5 are dropped
    glob[1] = 0.4
    per = avvp.video_scores(_outputs_from(so_a, so_v, glob), gt)
    assert per[0, 1] == 0.0 and per[0, 5] == 0.0 and per[0, 0] == 1.0
    # half the speech seconds: segment F = 2·2 / (2·2 + 2) = 2/3; the one predicted event [0, 2) has IoU 0.5 with [0, 4)
    glob[1] = 1
    so_a[0, 2:4] = 0
    per = avvp.video_scores(_outputs_from(so_a, so_v, glob), gt)
    assert per[0, 0] == pytest.approx(2 / 3) and per[0, 4] == 1.0
    so_a[0, 1] = 0  # [0, 1): IoU 0.25 < 0.5 -> one false positive and one miss
    per = avvp.video_scores(_outputs_from(so_a, so_v, glob), gt)
    assert per[0, 0] == pytest.approx(2 / 5) and per[0, 4] == 0.0
    # Type@AV and Event@AV from the per-video columns
    pv = np.array([[0.5, 0.25, 0.4, 1.0, 0, 0, 0, 0], [1.0, 0.75, 0.6, 0.0, 0, 0, 0, 0]])
    s = avvp.summary(pv)
    assert s["seg_a"] == 75.0 and s["seg_v"] == 50.0 and s["seg_av"] == 50.0 and s["seg_event_av"] == 50.0
    assert s["seg_type_av"] == pytest.approx((75 + 50 + 50) / 3)
    assert avvp.per_clip(pv, "seg_type_av").mean() == pytest.approx(s["seg_type_av"])


def test_floor_predicts_nothing():
    avvp = _avvp()
    weak = np.zeros((4, 25), dtype=np.float32)
    weak[[0, 1], 0] = 1
    weak[2, 3] = 1
    fl = avvp.floor_outputs(weak, 3)
    assert fl.shape == (3, 23, 25) and fl[:, :, 0].max() == pytest.approx(0.5) and fl[:, :, 3].max() == pytest.approx(0.25)
    gt = np.stack([avvp.gt_matrices("0:4:0", ""), avvp.gt_matrices("", ""), avvp.gt_matrices("0:2:1", "0:2:1")])
    per = avvp.video_scores(fl, gt)  # 0.5 is >= the threshold: class 0 is predicted everywhere, in both modalities
    assert per[0, 0] == pytest.approx(8 / 14) and per[1].tolist() == [0.0] * 8  # a false positive on a clip without events scores 0
    weak[0, 0] = 0
    per = avvp.video_scores(avvp.floor_outputs(weak, 3), gt)  # nothing reaches 0.5: an empty parse
    assert per[1].tolist() == [1.0] * 8 and per[0, 0] == 0.0 and per[0, 1] == 1.0


def test_token_slots_and_pooling():
    avvp = _avvp()
    names, kinds = ["video", "audio"], {"video": "video", "audio": "audio"}
    br = avvp.stream_branches(names, kinds)
    assert br.tolist() == [1, 0]
    assert avvp.stream_branches(["audio"], kinds).tolist() == [2]  # one stream feeds both branches
    pos = np.array([[[0, 0.05, -1, -1], [0, 0.55, -1, -1], [1, 0.95, -1, -1], [1, -1, -1, -1], [0, 0.999, -1, -1]]], dtype=np.float32)
    dur = np.array([[10.0, 8.0]])
    b, sec = avvp.token_slots(pos, dur, br, segment_seconds=1.0)
    assert b.tolist() == [[1, 1, 0, 0, 1]] and sec.tolist() == [[0, 5, 7, -1, 9]]
    x = np.arange(5 * 2, dtype=np.float32).reshape(1, 5, 2)
    pooled = avvp.pool_seconds(x, b, sec)
    assert pooled.shape == (1, 2, 10, 2)
    assert pooled[0, 1, 0].tolist() == [0, 1] and pooled[0, 1, 5].tolist() == [2, 3] and pooled[0, 0, 7].tolist() == [4, 5]
    assert pooled[0, 0, 0].tolist() == [0, 0]  # no audio token in second 0: zeros
    both = avvp.pool_seconds(x[:, :1], np.array([[2]]), np.array([[3]]))
    assert both[0, 0, 3].tolist() == both[0, 1, 3].tolist() == [0, 1]


def test_han_head_and_loss():
    torch = pytest.importorskip("torch")
    avvp = _avvp()
    torch.manual_seed(0)
    net = avvp.build_net(16).eval()
    out = net(torch.randn(3, 2, 10, 16))
    assert tuple(out.shape) == (3, 23, 25) and float(out.min()) >= 0 and float(out.max()) <= 1
    assert torch.allclose(out[:, 20], out[:, 21] * 0 + out[:, 20])  # global prob is a probability per class
    y = torch.zeros(3, 25)
    y[0, 0] = 1
    loss = avvp.han_loss(out, y)
    assert tuple(loss.shape) == (3,) and bool(torch.isfinite(loss).all())
    # per-sample = mean over classes of each BCE term, summed (main_avvp.py's nn.BCELoss mean over B x C, times 3 terms)
    bce = torch.nn.BCELoss()
    eps = 1e-7
    ref = (bce(out[:, 21].clamp(eps, 1 - eps), y) + bce(out[:, 22].clamp(eps, 1 - eps), 0.9 * y + 0.05) + bce(out[:, 20].clamp(eps, 1 - eps), y))
    assert float(loss.mean()) == pytest.approx(float(ref), rel=1e-5)


@pytest.mark.skipif(ANNOT is None, reason="llp_avvp annotations_root not staged on this machine")
def test_ground_truth_equals_the_authors_reading():
    """The val/test ground truth of the frozen manifest equals main_avvp.py::eval's reading of the staged dense
    files (rows matched by exact filename, GT[idx, onset:offset] = 1)."""
    from yavih import Benchmark

    avvp = _avvp()
    b = Benchmark.from_dir(HERE)
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    dense = {}
    for m, name in ((0, "AVVP_eval_audio.csv"), (1, "AVVP_eval_visual.csv")):
        with open(ANNOT / name, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                dense.setdefault(r["filename"], []).append((m, int(r["onset"]), int(r["offset"]), idx[r["event_labels"]]))
    for split, name in (("val", "AVVP_val_pd.csv"), ("test", "AVVP_test_pd.csv")):
        with open(ANNOT / name, newline="", encoding="utf-8") as fh:
            files = [r["filename"] for r in csv.DictReader(fh, delimiter="\t")]
        gt = avvp.gt_targets(b, split, np.arange(len(files)))
        assert b.manifest(split).clip_id == files
        for i, f in enumerate(files):
            want = np.zeros((2, 25, 10))
            for m, x1, x2, c in dense.get(f, []):
                want[m, c, x1:x2] = 1
            assert np.array_equal(gt[i], want), f


# ----------------------------------------------------------------------------- make_manifest


def _upstream_from_manifest(rows: list[dict[str, str]]) -> dict:
    """The parsed upstream files rebuilt from the committed manifest (event order per video kept; orphan rows, which
    the derivation ignores, are absent)."""
    splits = {"train": [], "val": [], "test": []}
    events = {"audio": [], "visual": []}
    for r in rows:
        splits[r["split"]].append((r["clip_id"], ",".join(r["labels"].split(";"))))
        for m in ("audio", "visual"):
            for tok in filter(None, r[f"{m}_events"].split(";")):
                on, off, c = tok.split(":")
                events[m].append((r["clip_id"], on, off, CATEGORIES[int(c)]))
    return {"splits": splits, "events": events, "full": None}


def test_make_manifest_check(rows: list[dict[str, str]]):
    """``derive`` re-creates manifest.csv.gz and classes.csv byte for byte from the upstream rows (rebuilt from the
    committed manifest); the staged files are checked by the next test where they exist."""
    from yavih.freeze import csv_bytes

    mm = F.load_module(HERE / "make_manifest.py")
    out, facts = mm.derive(_upstream_from_manifest(rows))
    assert csv_bytes(mm.MANIFEST_COLUMNS, out) == gzip.decompress((HERE / "manifest.csv.gz").read_bytes())
    assert csv_bytes(mm.CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(mm.CATEGORIES)]) == (HERE / "classes.csv").read_bytes()
    assert facts["ytid_overlap"] == {"train/val": 0, "train/test": 0, "val/test": 0}


@pytest.mark.skipif(ANNOT is None, reason="llp_avvp annotations_root not staged on this machine")
def test_make_manifest_check_on_staged_files(spec: dict, capsys: pytest.CaptureFixture):
    mm = F.load_module(HERE / "make_manifest.py")
    assert mm.main(["--no-media", "--metadata", str(ANNOT), "--check"]) == 0
    printed = json.loads(capsys.readouterr().out)
    up = spec["manifest"]["upstream"]
    for key in ("source", "sha256", "rows", "upstream_files"):
        assert printed["upstream"][key] == up[key], key
    assert printed["manifest"]["sha256"] == spec["manifest"]["sha256"]
    d = printed["derived"]
    assert d["orphan_event_rows"] == {"audio": 14, "visual": 11} and d["duplicate_event_rows"] == {"audio": 8, "visual": 0}
    assert d["empty_event_rows"] == {"audio": 5, "visual": 0} and d["weak_label_cells_with_duplicates"] == 1


def test_make_manifest_refuses_another_file(tmp_path: Path):
    mm = F.load_module(HERE / "make_manifest.py")
    for name in mm.FILES_SHA256:
        (tmp_path / name).write_text("filename\tevent_labels\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="sha256"):
        mm.main(["--no-media", "--metadata", str(tmp_path), "--check"])


# ----------------------------------------------------------------------------- prepare/


def test_prepare_scripts_present(spec: dict):
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_annotations.sh", "02_media.sh"]
    text = "".join((HERE / "prepare" / n).read_text(encoding="utf-8") for n in names)
    for name, sha in spec["source"]["sha256"].items():
        assert sha in text, name  # every pinned upstream file is checked by a prepare step
    assert spec["source"]["revision"] in text


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values()
              if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    for reading, streams in (("av", {"video", "audio"}), ("a", {"audio"}), ("v", {"video"})):
        d = cls(Benchmark.from_dir(HERE).view(), {"reading": reading}).describe()
        assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
        assert d["reading"] == reading and set(d["input"]) == streams == set(d["streams"])
    assert mod.N_FRAMES == 80 and mod.T_AUDIO == 10 and mod.AUDIO_RATE == 16000  # the authors' feature recipe


# ----------------------------------------------------------------------------- reader and stand-in


def test_reader_smoke_synthetic_raw(tmp_path: Path):
    """Synthetic mp4s (video + AAC) at two test rows' paths: the reader hands over the stored bytes as both streams
    with the guaranteed facts and ``meta == {"facts": {num_segments: 10, segment_seconds: 1.0}}``."""
    pytest.importorskip("av")
    from yavih import Benchmark

    testing = F.load_module(HERE / "testing.py", "llp_avvp_testing_under_test")
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    man = b.manifest("test")
    stored = {}
    for k in (0, 1):
        rel = man.media[k]["video"]
        testing._write_clip(tmp_path / rel, k)
        stored[k] = (tmp_path / rel).read_bytes()
    ic = F.load_yaml(HERE)["input_contract"]
    samples = list(b.reader("test", rows=range(0, 2)))
    assert [s.row for s in samples] == [0, 1]
    for k, s in enumerate(samples):
        assert s.meta == {"facts": {"num_segments": 10, "segment_seconds": 1.0}} and set(s.streams) == {"video", "audio"}
        v, a = s.streams["video"], s.streams["audio"]
        assert v.bytes() == a.bytes() == stored[k]
        for st, keys in ((v, ic["info"]["video"]), (a, ic["info"]["audio"])):
            assert not [key for key in keys if st.info.get(key) is None], st.name
        assert v.info["duration_s"] == pytest.approx(10.0, abs=0.6)


def test_stand_in_events_are_valid():
    testing = F.load_module(HERE / "testing.py", "llp_avvp_testing_under_test")
    avvp = _avvp()
    for i in range(50):
        aud, vis = testing.clip_events(f"test_{i:04d}")
        assert aud == testing.clip_events(f"test_{i:04d}")[0]
        gt = avvp.gt_matrices(";".join(f"{a}:{b}:{c}" for a, b, c in aud), ";".join(f"{a}:{b}:{c}" for a, b, c in vis))
        assert (gt[0] * gt[1]).sum() > 0  # every stand-in clip has an audio-visual event: its floor stays low
        assert all(c < testing.K_CLASSES and 0 <= a < b <= 10 for a, b, c in aud + vis)


def test_copied_upstream_files_are_marked():
    """Every copied upstream file names its project, commit, path and licence in its header."""
    for name, path in (("avvp_eval_metrics.py", "utils/eval_metrics.py"), ("han.py", "nets/net_audiovisual.py")):
        head = (HERE / name).read_text(encoding="utf-8").split("# -----", 1)[0]
        assert "https://github.com/YapengTian/AVVP-ECCV20" in head and "fde56118f2c2e7c19188dfefa2569a37b70054fe" in head
        assert path in head and "GNU General Public License v3.0" in head
    body = (HERE / "avvp_eval_metrics.py").read_bytes().split(b"# " + b"-" * 112 + b"\n", 1)[1]
    assert hashlib.sha256(body).hexdigest() == "fcfa1247ebf9d28e34c7e555f97d72ad01f59f82924c171a15d3b0f98d75cb88"
