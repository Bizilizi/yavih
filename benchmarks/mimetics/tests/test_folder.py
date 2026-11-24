"""Folder tests for mimetics.

Run only this file: ``python -m pytest "$(yavih bench path mimetics)/tests" -q -p no:cacheprovider``.

Groups:
- yaml / manifest / classes facts: pure yaml + csv, run everywhere;
- ``eval.py``: the copied IntegralAction top-1 equals the core ``acc``, and every folder metric on hand-checked inputs;
- the bridge template's declared rules and positions;
- ``make_manifest.py`` on a synthetic release (a 3-class release and a small K400 listing); needs torch for the draw;
- the reader smoke builds a tiny copy of this folder over generated mp4s under two roots and needs PyAV to write them;
- ``make_manifest.py --check`` against the real release and K400 listing runs only where ``mimetics.media_root``
  (with ``release/``) and ``mimetics.k400_root`` (with ``train/``) are configured and staged.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import shutil
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import yaml

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
spec, rows = F.spec, F.rows

READOUT = "clf"
HEADLINE = "acc"
STREAMS = {"video": "video"}
CLIPS = {"train": 10000, "test": 713}
READINGS = ["v"]
EXTERNAL: list[str] = []
HAS_EVAL = True

# The staged release and K400 tree from the site config, resolved at collection (before ``_isolated_site`` hides
# them); None when unconfigured or not staged here.
MEDIA_ROOT = F.site_root(HERE, "media_root", needs=("release/mimetics_v1.0.csv",))
K400_ROOT = F.site_root(HERE, "k400_root", needs=("train/archery",))


@pytest.fixture(autouse=True)
def _isolated_site(monkeypatch, tmp_path_factory):
    """No site config and a fresh YAVIH_HOME."""
    monkeypatch.setenv("YAVIH_CONFIG", "/nonexistent/yavih-config.toml")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path_factory.mktemp("yavih_home")))


def _mm(name: str):
    return F.load_module(HERE / "make_manifest.py", name)


def _eval():
    return F.load_module(HERE / "eval.py", "mimetics_eval")


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]
    assert spec["splits"]["train"] == ["train"] and spec["splits"]["eval"] == ["test"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert spec["scorer"] == READOUT and m["headline"] == HEADLINE
    assert m["list"] == ["acc", "top5", "mean_top1", "mean_top5", "mean_ap", "n_eval"] and m["floors"] == ["acc_floor"]
    assert m["chance"]["acc"] == pytest.approx(1 / 50) and m["chance"]["top5"] == pytest.approx(5 / 50)
    # chance of mean_ap: E[1 / rank] for a uniformly random rank among 50 classes = H_50 / 50
    assert m["chance"]["mean_ap"] == pytest.approx(sum(1 / k for k in range(1, 51)) / 50, abs=5e-4)


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)
    assert [e["name"] for e in spec["external"] or []] == EXTERNAL
    assert (HERE / "eval.py").is_file() is HAS_EVAL
    assert spec["fields"]["video"]["root"] == {"train": "k400_root", "test": "media_root"}
    assert set(spec["data"]["roots"]) == {"media_root", "k400_root"}
    assert spec["protocol"]["train_draw"] == {"cap": 10000, "seed": 0, "balance": "class"}


def test_classes_are_the_release_order():
    cls = F.read_classes(HERE)
    names = [c["name"] for c in cls]
    assert [int(c["index"]) for c in cls] == list(range(50)) and len(set(names)) == 50
    assert names[0] == "archery" and names[-1] == "writing" and "punching_person_-boxing-" in names
    assert "skiing_-not_slalom_or_crosscountry-" in names and "shooting_goal_-soccer-" in names
    assert all(re.match(r"^[a-z_-]+$", n) for n in names)


def test_rows_layout_labels_and_leakage(rows: list[dict[str, str]]):
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    assert set(Counter(r["label"] for r in train).values()) == {200}
    per = Counter(r["label"] for r in test)
    assert len(per) == 50 and min(per.values()) == per["tying_tie"] == 8 and max(per.values()) == per["drinking"] == 27
    assert all(r["rel_media"] == f"train/{r['label']}/{r['clip_id']}.mp4" for r in train)
    assert all(r["rel_media"] == f"clips/{r['clip_id']}.mp4" for r in test)
    assert all(re.match(r"^[A-Za-z0-9_-]{11}_\d{6}_\d{6}$", r["clip_id"]) for r in test)
    assert all(r["object"] == r["scene"] == r["mime"] == "" for r in train)
    assert all({r["object"], r["scene"], r["mime"]} <= {"yes", "no"} for r in test)
    assert Counter(r["mime"] for r in test)["yes"] > 0
    mm = _mm("mimetics_mm_rows")
    train_yt = {mm.norm_ytid(mm.k400_ytid(r["clip_id"])) for r in train}
    assert not {mm.norm_ytid(r["clip_id"][:11]) for r in test} & train_yt
    assert len({r["clip_id"] for r in rows}) == len(rows)
    # the frozen order: train by class then clip_id, then test by class then clip_id
    order = {c["name"]: int(c["index"]) for c in F.read_classes(HERE)}
    for part in (train, test):
        keys = [(order[r["label"]], r["clip_id"]) for r in part]
        assert keys == sorted(keys)
    assert rows[: len(train)] == train


def test_curation_sums(spec: dict):
    up = spec["manifest"]["upstream"]
    cur = up["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"] == up["rows"] == 713 + 34106
    assert cur["per_split"]["train"] == {"listed": 10000, "staged": 10000, "missing": 0}
    t = cur["per_split"]["test"]
    assert t["listed"] == 713 and t["staged"] + t["missing"] == 713
    assert cur["leakage_audit"]["excluded"] == 0
    assert spec["splits"]["min_eval_coverage"] <= t["staged"] / t["listed"]


def test_make_manifest_pins_agree_with_yaml(spec: dict):
    mm = _mm("mimetics_mm_pins")
    up = spec["manifest"]["upstream"]
    assert up["sha256"] == mm.RELEASE_SHA["mimetics_v1.0.csv"]
    assert up["upstream_files"] == {"Mimetics_release_v1.0.zip": mm.RELEASE_ZIP_SHA256,
                                    **{f"release/{k}": v for k, v in mm.RELEASE_SHA.items()},
                                    "k400_train_listing_50_classes": mm.K400_LISTING_SHA256}
    assert spec["source"]["sha256"] == {"Mimetics_release_v1.0.zip": mm.RELEASE_ZIP_SHA256}
    assert (mm.CAP, mm.SEED) == (spec["protocol"]["train_draw"]["cap"], spec["protocol"]["train_draw"]["seed"])
    assert up["curation"]["per_split"]["test"]["staged"] == mm.STAGED_AT_FREEZE["staged"]
    assert up["rows"] == mm.N_TEST + mm.K400_LISTING_ROWS
    assert mm.class_dir("punching person (boxing)") == "punching_person_-boxing-"
    assert mm.k400_ytid("035LtPeUFTE_000085_000095") == "035LtPeUFTE"
    assert mm.k400_ytid("youtube-NLf-rU1wlTY-1-11") == "NLf-rU1wlTY"


# ----------------------------------------------------------------------------- eval.py


def test_integralaction_top1_is_the_core_acc():
    from yavih.metrics.classification import acc

    ev = _eval()
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(300, 50)).astype(np.float32)
    t = rng.integers(0, 50, size=300)
    scores[np.arange(100), t[:100]] += 5.0  # a third clearly right
    hit1, hit5 = ev.integralaction_topk(scores, t)
    assert hit1.mean() == pytest.approx(acc(scores, t, {}, {}))
    assert hit5.mean() == pytest.approx(ev.top5(scores, t, {}, {})) and (hit5 >= hit1).all()
    assert np.array_equal(hit5, ev.label_rank(scores, t) <= 5)  # no ties among continuous scores


def test_folder_metrics_hand_checked():
    ev = _eval()
    s = np.array([[9, 1, 0, 0, 0, 0],    # label 0, rank 1
                  [5, 9, 0, 0, 0, 0],    # label 0, rank 2
                  [0, 1, 2, 3, 4, 5],    # label 0, rank 6
                  [0, 0, 9, 0, 0, 0]],   # label 2, rank 1
                 dtype=np.float64)
    t = np.array([0, 0, 0, 2])
    assert list(ev.label_rank(s, t)) == [1, 2, 6, 1]
    assert ev.top5(s, t, {}, {}) == pytest.approx(3 / 4)
    assert ev.mean_top1(s, t, {}, {}) == pytest.approx((1 / 3 + 1) / 2)
    assert ev.mean_top5(s, t, {}, {}) == pytest.approx((2 / 3 + 1) / 2)
    assert ev.mean_ap(s, t, {}, {}) == pytest.approx(((1 + 1 / 2 + 1 / 6) / 3 + 1) / 2)
    assert list(ev.label_rank(np.zeros((1, 6)), np.array([3]))) == [1]  # ties count in the label's favour
    with pytest.raises(ValueError):
        ev.top5(np.zeros(4), t, {}, {})


def test_top5_ties_resolve_the_same_on_every_cpu():
    """The copied top-5 uses a stable argsort: of classes tied across the fifth place, the lower-indexed are dropped.

    numpy's default sort (introsort, AVX-512 kernels where the CPU has them) leaves the order of ties to the platform;
    this row would then put class 1 inside or outside the top five depending on the machine.
    """
    ev = _eval()
    s = np.zeros((4, 8))
    s[:, 7] = 1.0  # one clear winner, then seven classes tied at 0 across the cut
    t = np.array([0, 1, 3, 4])
    hit1, hit5 = ev.integralaction_topk(s, t)
    # ascending stable order is 0, 1, ..., 6, 7; the last five are 3, 4, 5, 6, 7
    assert list(hit5) == [False, False, True, True]
    assert not hit1.any()
    assert ev.top5(s, t, {}, {}) == pytest.approx(2 / 4)
    # argmax keeps the FIRST of tied maxima, the stable top-5 the LAST: on an all-tied row the label 0 is a top-1
    # hit but not a top-5 hit (the reference's own behaviour, now the same everywhere)
    z1, z5 = ev.integralaction_topk(np.zeros((1, 6)), np.array([0]))
    assert z1[0] and not z5[0]


def test_chance_levels_hold_for_random_scores(spec: dict):
    ev = _eval()
    rng = np.random.default_rng(1)
    s = rng.normal(size=(20000, 50))
    t = rng.integers(0, 50, size=20000)
    ch = spec["metrics"]["chance"]
    for name in ("top5", "mean_top1", "mean_top5", "mean_ap"):
        assert getattr(ev, name)(s, t, {}, {}) == pytest.approx(ch[name], abs=0.01), name


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED

    mod = F.load_module(HERE / "bridge_template.py", "mimetics_bridge_template")
    d = mod.MimeticsBridge(Benchmark.from_dir(HERE).view(), {"layer": -1}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] in READINGS and set(d["input"]) <= set(STREAMS) and d["views"] == 1
    assert mod.TEMPORAL == {"rule": "fps", "fps": 3.75, "n": 8, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
    assert mod.SPATIAL["resize"] == {"short": 256} and mod.SPATIAL["crop"] == {"kind": "center", "size": [224, 224]}


@pytest.mark.parametrize(("width", "height", "duration"), [(454, 256, 10.0), (630, 360, 4.0), (640, 360, 1.0)])
def test_template_positions_meet_output_contract(width, height, duration):
    from yavih.media.rules import crop_region, rule_times

    mod = F.load_module(HERE / "bridge_template.py", "mimetics_bridge_template_pos")
    info = {"duration_s": duration, "fps": 30.0, "width": width, "height": height, "container": "mp4", "bytes": 1}
    times = rule_times(mod.TEMPORAL, info)
    assert times.shape == (mod.N_FRAMES,) and times.min() >= 0.0 and times.max() <= duration
    pos = mod.MimeticsBridge._pos(times, info)
    assert pos.shape == (mod.T_TOK * mod.H_TOK * mod.W_TOK, 4) and pos.dtype == np.float32
    assert (pos[:, 0] == 0).all() and (pos[:, 1] >= 0).all() and (pos[:, 1] <= 1).all()
    y0, x0, y1, x1 = crop_region(mod.SPATIAL, width, height)
    assert (pos[:, 2] > y0 - 1e-6).all() and (pos[:, 2] < y1 + 1e-6).all()
    assert (pos[:, 3] > x0 - 1e-6).all() and (pos[:, 3] < x1 + 1e-6).all()


# ----------------------------------------------------------------------------- make_manifest.py


def _synthetic_release(tmp_path: Path) -> tuple[Path, list[str]]:
    """A 3-class release (csv + the two annotation files) and a K400 listing of those classes."""
    rel = tmp_path / "release"
    rel.mkdir()
    (rel / "mimetics_v1.0.csv").write_text(
        "label,youtube_id,time_start,time_end,split,is_cc\n"
        "playing guitar,BBBBBBBBBBB,3,5,test,0\n"
        "archery,-AAAAAAAAAA,1,10,test,0\n"
        "punching person (boxing),CCCCCCCCCCC,12,13,test,0\n"
        "archery,AAAAAAAAAAA,0,4,test,0\n", encoding="utf-8")
    (rel / "mimetics_v1.0_clsannot.txt").write_text(
        "#label   object scene\narchery   large yes\nplaying_guitar large no\npunching_person_(boxing) small yes\n",
        encoding="utf-8")
    (rel / "mimetics_v1.0_objectsceneannot.txt").write_text(
        "#video\tobject\tscene\tmime\n"
        "-AAAAAAAAAA_000001_000010.mp4  \tno \tyes\tno \n"
        "AAAAAAAAAAA_000000_000004.mp4\tno\tno\tyes\n"
        "BBBBBBBBBBB_000003_000005.mp4\tyes\tno\tno\n"
        "CCCCCCCCCCC_000012_000013.mp4\tno\tno\tyes\n", encoding="utf-8")
    listing = [f"train/{c}/{i:011d}_000000_000010.mp4" for c in ("archery", "playing_guitar", "punching_person_-boxing-")
               for i in range(3 if c == "archery" else 5)]
    return rel / "mimetics_v1.0.csv", listing


def test_make_manifest_on_synthetic_release(tmp_path):
    pytest.importorskip("torch")
    mm = _mm("mimetics_mm_synth")
    csv_path, listing = _synthetic_release(tmp_path)
    with pytest.raises(SystemExit, match="sha256"):
        mm.read_release(csv_path)  # the real pins refuse another release
    release = mm.read_release(csv_path, pins=False)
    with pytest.raises(SystemExit, match="3 classes, expected 50"):
        mm.freeze(release, listing)  # the real pins refuse another class list ...
    real = {**release, "labels": [f"c{i:02d}" for i in range(50)]}
    with pytest.raises(SystemExit, match="different media set"):
        mm.freeze(real, listing)  # ... and another K400 listing
    res = mm.freeze(release, listing, cap=6, seed=0, pins=False)
    out = list(csv.DictReader(io.StringIO(res["manifest_bytes"].decode("utf-8"))))
    assert list(out[0]) == mm.MANIFEST_COLUMNS
    train = [r for r in out if r["split"] == "train"]
    test = [r for r in out if r["split"] == "test"]
    assert Counter(r["label"] for r in train) == {"archery": 2, "playing_guitar": 2, "punching_person_-boxing-": 2}
    assert [r["clip_id"] for r in test] == ["-AAAAAAAAAA_000001_000010", "AAAAAAAAAAA_000000_000004",
                                            "BBBBBBBBBBB_000003_000005", "CCCCCCCCCCC_000012_000013"]
    assert [(r["object"], r["scene"], r["mime"]) for r in test] == [("no", "yes", "no"), ("no", "no", "yes"),
                                                                    ("yes", "no", "no"), ("no", "no", "yes")]
    assert res["classes_bytes"] == b"index,name\n0,archery\n1,playing_guitar\n2,punching_person_-boxing-\n"
    cur = res["upstream"]["curation"]
    assert cur["total_upstream"] == 4 + 13 and cur["excluded"] == {"draw_remainder": 7}
    assert res["upstream"]["upstream_files"]["k400_train_listing_50_classes"] == mm.listing_sha256(listing)
    again = mm.freeze(release, list(reversed(listing)), cap=6, seed=0, pins=False)
    assert again["manifest_bytes"] == res["manifest_bytes"]  # the listing's order does not matter


def test_make_manifest_check(spec: dict, tmp_path):
    """``make_manifest.py --no-media --check`` on the public release and the K400 listing re-derives the committed
    files, and the printed ``upstream`` block is the yaml's."""
    if MEDIA_ROOT is None or K400_ROOT is None:
        pytest.skip("mimetics.media_root (release/) or mimetics.k400_root (train/) not configured or not staged here")
    pytest.importorskip("torch")
    mm = _mm("mimetics_mm_check")
    listing = tmp_path / "k400_50.txt"
    assert mm.main(["--k400-root", str(K400_ROOT), "--media-root", str(MEDIA_ROOT), "--write-listing", str(listing)]) == 0
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mm.main(["--no-media", "--metadata", str(MEDIA_ROOT / "release" / "mimetics_v1.0.csv"),
                      "--k400-listing", str(listing), "--check"])
    assert rc == 0
    printed = json.loads(buf.getvalue())
    up = spec["manifest"]["upstream"]
    for key in ("sha256", "rows", "upstream_files", "curation"):
        assert printed["upstream"][key] == up[key], key
    assert printed["manifest"]["sha256"] == spec["manifest"]["sha256"]
    assert printed["manifest"]["classes_sha256"] == spec["manifest"]["classes_sha256"]


# ----------------------------------------------------------------------------- reader smoke


def _write_mp4(path: Path, *, width: int, height: int, seconds: float, fps: int = 25) -> None:
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(path), "w", format="mp4") as c:
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = width, height, "yuv420p"
        for k in range(max(1, round(seconds * fps))):
            img = np.full((height, width, 3), (k * 7) % 256, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def test_reader_smoke_synthetic_raw(tmp_path, spec):
    """A 4-row copy of this folder over generated mp4s: train rows are read from k400_root, test rows from
    media_root; the generic reader hands over the stored bytes, the guaranteed info keys and no facts."""
    folder = tmp_path / "mimetics"
    folder.mkdir()
    roots = {"k400_root": tmp_path / "k400", "media_root": tmp_path / "mimetics_media"}
    geoms = {  # (split, label, clip_id) -> (width, height, seconds)
        ("train", "archery", "aaaaaaaaaaa_000001_000011"): (96, 64, 2.0),
        ("train", "writing", "bbbbbbbbbbb_000002_000012"): (64, 96, 0.5),
        ("test", "archery", "ccccccccccc_000006_000010"): (112, 64, 4.0),
        ("test", "writing", "-dddddddddd_000001_000002"): (112, 64, 1.0),
    }
    lines = ["clip_id,split,label,rel_media,object,scene,mime"]
    for (split, label, cid), (w, h, sec) in geoms.items():
        rel = f"train/{label}/{cid}.mp4" if split == "train" else f"clips/{cid}.mp4"
        _write_mp4(roots["k400_root" if split == "train" else "media_root"] / rel, width=w, height=h, seconds=sec)
        flags = ",," if split == "train" else "no,no,yes"
        lines.append(f"{cid},{split},{label},{rel},{flags}")
    man = ("\n".join(lines) + "\n").encode("utf-8")
    (folder / "manifest.csv.gz").write_bytes(gzip.compress(man, mtime=0))
    shutil.copy(HERE / "classes.csv", folder / "classes.csv")
    shutil.copy(HERE / "eval.py", folder / "eval.py")
    y = dict(spec)
    y["manifest"] = {**spec["manifest"], "sha256": hashlib.sha256(man).hexdigest()}
    y["manifest"]["upstream"] = {**spec["manifest"]["upstream"], "curation": {
        "total_upstream": 4, "per_split": {"train": {"listed": 2, "staged": 2, "missing": 0},
                                           "test": {"listed": 2, "staged": 2, "missing": 0}},
        "excluded": {}, "leakage_audit": None}}
    y["budget"] = {**spec["budget"], "clips": {"train": 2, "test": 2}}
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")

    from yavih import Benchmark
    from yavih.sample import RawStream

    b = Benchmark.from_dir(folder, data={k: str(v) for k, v in roots.items()})
    assert [e["mime"] for e in b.manifest("test").extra] == ["yes", "yes"]
    for split, root in (("train", roots["k400_root"]), ("test", roots["media_root"])):
        samples = list(b.reader(split, rows=range(0, 2)))
        assert [s.row for s in samples] == [0, 1]
        for s in samples:
            assert set(s.streams) == {"video"} and s.meta == {"facts": {}}
            v = s.streams["video"]
            assert isinstance(v, RawStream) and v.kind == "video"
            key = next(k for k in geoms if k[0] == split and k[2] == s.clip_id)
            f = root / (f"train/{key[1]}/{key[2]}.mp4" if split == "train" else f"clips/{key[2]}.mp4")
            assert v.bytes() == f.read_bytes()
            for k in spec["input_contract"]["info"]["video"]:
                assert v.info[k] is not None, k
            w, h, sec = geoms[key]
            assert (v.info["width"], v.info["height"]) == (w, h)
            assert v.info["duration_s"] == pytest.approx(sec, abs=0.1) and v.info["bytes"] == f.stat().st_size


# ----------------------------------------------------------------------------- prepare/


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_release.sh", "02_clips.py"]
    assert all(re.match(r"^\d+_[A-Za-z0-9_.-]+\.(?:sh|py)$", n) for n in names), names
    assert _mm("mimetics_mm_prep").RELEASE_ZIP_SHA256 in (HERE / "prepare" / "01_release.sh").read_text(encoding="utf-8")
    crawler = (HERE / "prepare" / "02_clips.py").read_text(encoding="utf-8")
    assert "82304fa9f4a7219421df13dae6347a445c4cf5bc" in crawler and '"-c:v", "libx264", "-c:a", "copy"' in crawler
