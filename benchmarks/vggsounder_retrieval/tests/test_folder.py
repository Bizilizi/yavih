"""Folder tests of vggsounder_retrieval (run: ``python -m pytest "$(yavih bench path vggsounder_retrieval)/tests" -q -p no:cacheprovider``).

Pure checks of the frozen artefacts and the copied scorer code (always run); checks through the core (always run);
the re-derivation of the manifest, which needs the vggsounder 1.0 label file and the VGGSound listing (skips
without them: ``VGGSOUNDER_CSV=/path/to/vggsounder.csv`` or an installed ``vggsounder`` package, and a configured
media root with ``metadata/vggsound.csv``). ``yavih bench lint`` checks the layout, hashes, fixture and docs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows, mm, bench = F.spec, F.rows, F.mm, F.bench

READOUT = "vggsounder_retrieval"
HEADLINE = "r@1"
STREAMS = {"video": "video", "audio": "audio"}
CLIPS = {"test": 5294}
READINGS = ["av"]
DIRS = ("a2v", "v2a")


def _eval():
    return F.load_module(HERE / "eval.py")


# ----------------------------------------------------------------------------- yaml + files


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert spec["scorer"] == READOUT and m["headline"] == HEADLINE and m["floors"] == [] and m["controls"] == []
    inst = ["r@1"] + [f"r@{k}_{d}" for d in DIRS for k in (1, 5, 10)] + [f"medr_{d}" for d in DIRS]
    cls = [f"cls_map_{d}" for d in DIRS] + [f"cls_r@{k}_{d}" for d in DIRS for k in (1, 5, 10)] + ["cls_n_no_relevant"]
    assert set(m["list"]) == set(inst) | set(cls) | {"n_eval"}
    for k in (1, 5, 10):
        for d in DIRS:
            assert m["chance"][f"r@{k}_{d}"] == pytest.approx(k / 5294, rel=0.01)


def test_yaml_identity(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["fields"].items() if v.get("storage", "file") == "file"} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS) and ic["facts"] == []
    assert spec["external"] == [] and spec["manifest"]["label"] == {"column": None, "kind": "none"}
    assert spec["readout_params"] == {"subsets": {}, "classes_column": "labels_av"}
    oc = spec["output_contract"]
    assert set(oc["schema"]) == {"audio_emb", "video_emb"} and oc["isolation"] == {"audio_emb": ["audio"], "video_emb": ["video"]}


def test_rows_columns_and_the_row_rule(spec: dict, rows: list[dict[str, str]]):
    assert list(rows[0]) == ["clip_id", "split", "labels_av", "rel_video", "rel_audio"]
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids) == 5294
    for r in rows:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,32}_\d{6}", r["clip_id"]) and r["split"] == "test" and r["labels_av"]
        assert r["rel_video"] == f"video/{r['clip_id']}.mp4" and r["rel_audio"] == f"audio/{r['clip_id']}.wav"
    classes = Counter(c for r in rows for c in r["labels_av"].split(";"))
    assert len(classes) == 308 and classes.most_common(1) == [("male speech, man speaking", 697)]
    assert sum(1 for v in classes.values() if v < 10) == 50
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["per_split"]["test"]["listed"] + sum(cur["excluded"].values()) == cur["total_upstream"] == 12014


def test_rows_are_the_av_only_rows_of_the_vggsounder_folder(rows: list[dict[str, str]]):
    """Cross-check against the sibling folder's manifest when it is next to this one: the rows are exactly its
    test rows with labels_av non-empty and labels_a, labels_v empty, in its order, with the same AV cells."""
    sib = HERE.parent / "vggsounder"
    if not (sib / "manifest.csv.gz").is_file():
        pytest.skip("the vggsounder folder is not next to this one")
    theirs = [r for r in F.read_rows(sib) if r["split"] == "test" and r["labels_av"] and not r["labels_a"] and not r["labels_v"]]
    assert [(r["clip_id"], r["labels_av"]) for r in theirs] == [(r["clip_id"], r["labels_av"]) for r in rows]


def test_shared_files_are_the_sibling_copies():
    sib = HERE.parent / "vggsound_retrieval"
    if not sib.is_dir():
        pytest.skip("vggsound_retrieval is not next to this one")
    for name in ("retrieval.py", "xmodal_synth.py"):
        assert (HERE / name).read_bytes() == (sib / name).read_bytes(), name


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_vggsound_media.sh", "02_make_manifest.sh", "03_verify.sh"]


# ----------------------------------------------------------------------------- the scorer


def test_instance_ranks_match_a_naive_loop():
    R = _eval().retrieval
    rng = np.random.default_rng(1)
    q, g = rng.standard_normal((40, 8)), rng.standard_normal((40, 8))
    g[:20] = q[:20] + 0.01 * rng.standard_normal((20, 8))
    ours = R.ranks(q, g)
    qn, gn = q / np.linalg.norm(q, axis=1, keepdims=True), g / np.linalg.norm(g, axis=1, keepdims=True)
    for i in range(40):
        sims = [float(qn[i] @ gn[j]) for j in range(40)]
        assert ours[i] == 1 + sum(1 for s in sims if s > sims[i]) + (sum(1 for s in sims if s == sims[i]) - 1) / 2


def _dcmh_upstream(qB, rB, query_L, retrieval_L):
    """DCMH's calc_map as published (Hamming ranking of +-1 codes), only xrange -> range and int(tsum)."""
    q = rB.shape[1]
    total = 0
    for it in range(query_L.shape[0]):
        gnd = (np.dot(query_L[it, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = 0.5 * (q - np.dot(qB[it, :], rB.transpose()))
        gnd = gnd[np.argsort(hamm)]
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        total = total + np.mean(count / tindex)
    return total / query_L.shape[0]


def test_calc_map_is_dcmh_on_its_own_example():
    C = _eval().class_level
    qB = np.array([[1, -1, 1, 1], [-1, -1, -1, 1], [1, 1, -1, 1], [1, 1, 1, -1]])
    rB = np.array([[1, -1, 1, -1], [-1, -1, 1, -1], [-1, -1, 1, -1], [1, 1, -1, -1], [-1, 1, -1, -1], [1, 1, -1, 1]])
    qL = np.array([[0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    rL = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    sim = -(0.5 * (4 - qB @ rB.T))  # the negative Hamming distance: the same order
    m, aps = C.calc_map(sim, qL, rL)
    assert m == pytest.approx(_dcmh_upstream(qB, rB, qL, rL)) and m == pytest.approx(aps.mean())


def test_class_level_against_a_naive_definition():
    C = _eval().class_level
    rng = np.random.default_rng(3)
    n = 30
    labels = [";".join(sorted({f"c{rng.integers(5)}" for _ in range(rng.integers(1, 3))})) for _ in range(n)]
    labels[0] = "lonely"
    a = rng.standard_normal((n, 6))
    v = a + 0.8 * rng.standard_normal((n, 6))
    out = C.class_level(a, v, labels)
    sets = [set(x.split(";")) for x in labels]
    an, vn = a / np.linalg.norm(a, axis=1, keepdims=True), v / np.linalg.norm(v, axis=1, keepdims=True)
    for j, (q, g) in enumerate(((an, vn), (vn, an))):
        for i in range(n):
            others = [t for t in range(n) if t != i]
            order = sorted(others, key=lambda t: -float(q[i] @ g[t]))
            rel = [1 if sets[i] & sets[t] else 0 for t in order]
            hits, prec = 0, []
            for pos, r in enumerate(rel, 1):
                if r:
                    hits += 1
                    prec.append(hits / pos)
            assert out["ap"][i, j] == pytest.approx(np.mean(prec) if prec else 0.0)
            for k in (1, 5, 10):
                assert out["hits"][k][i, j] == float(any(rel[:k]))
    assert out["n_relevant"][0] == 0 and out["ap"][0].tolist() == [0.0, 0.0]
    m = _eval().METRICS
    meta = {"class_level": out}
    assert m["cls_map_a2v"].fn(None, None, {}, meta) == pytest.approx(out["ap"][:, 0].mean())
    assert m["cls_r@5_v2a"].fn(None, None, {}, meta) == pytest.approx(out["hits"][5][:, 1].mean())
    assert m["cls_n_no_relevant"].fn(None, None, {}, meta) == 1.0


def test_class_chance_matches_a_random_ranking(mm, rows: list[dict[str, str]]):
    chance = mm.class_chance([set(r["labels_av"].split(";")) for r in rows[:400]])
    C = _eval().class_level
    rng = np.random.default_rng(0)
    labels = [r["labels_av"] for r in rows[:400]]
    hits = np.mean([C.class_level(rng.standard_normal((400, 32)), rng.standard_normal((400, 32)), labels)["hits"][10].mean() for _ in range(5)])
    assert hits == pytest.approx(chance["cls_r@10_a2v"], abs=0.03)


def test_benchmark_loads(bench):
    assert bench.spec.readout == READOUT and bench.spec.output_contract.isolation == {"audio_emb": ("audio",), "video_emb": ("video",)}
    ev = bench.eval_module()
    assert ev.SCORER.output_keys == frozenset({"audio_emb", "video_emb"}) and ev.SCORER.kind == "stateless"


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED_CORE, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    d = cls(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED_CORE) <= set(d), set(DESCRIBE_REQUIRED_CORE) - set(d)  # a declared-schema bridge
    assert d["reading"] in READINGS and set(d["input"]) <= set(STREAMS) and tuple(cls.outputs) == ("audio_emb", "video_emb")


# ----------------------------------------------------------------------------- needs the upstream files


def _package_csv() -> Path | None:
    env = os.environ.get("VGGSOUNDER_CSV")
    if env and Path(env).is_file():
        return Path(env)
    try:
        import vggsounder
    except ImportError:
        return None
    p = Path(vggsounder.__file__).resolve().parent / "data" / "vggsounder.csv"
    return p if p.is_file() else None


def test_make_manifest_check(mm, spec: dict, capsys: pytest.CaptureFixture):
    csv_path = _package_csv()
    root = F.site_root(HERE, "media_root", needs=("metadata/vggsound.csv",))
    if csv_path is None or root is None:
        pytest.skip("needs the vggsounder 1.0 data/vggsounder.csv (VGGSOUNDER_CSV or the package) and VGGSound's metadata/vggsound.csv")
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == mm.VGGSOUNDER_CSV_SHA
    assert mm.main(["--no-media", "--metadata", str(csv_path), "--vggsound-csv", str(root / "metadata" / "vggsound.csv"), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["upstream"] == {k: v for k, v in spec["manifest"]["upstream"].items() if k != "note"}
    assert report["stats"]["same_source"] == {"ytids_with_several_clips": 135, "clips": 270}


def test_reader_smoke_synthetic_raw(tmp_path: Path):
    """The reader hands over the stored mp4 and wav byte for byte, with ``meta == {"facts": {}}``."""
    from yavih import Benchmark

    synth = F.load_module(HERE / "xmodal_synth.py")
    from yavih.codecs import wav_bytes

    probe = Benchmark.from_dir(HERE)
    m = probe.manifest("test")
    media = tmp_path / "media"
    for i in range(3):
        cid = m.clip_id[i]
        synth._write_media(media / "video" / f"{cid}.mp4", 0.2 + 0.3 * i, (0.25, 0.25, 0.75, 0.75), audio_track=True)
        (media / "audio").mkdir(parents=True, exist_ok=True)
        (media / "audio" / f"{cid}.wav").write_bytes(wav_bytes(synth._tone(0.2 + 0.3 * i), 16000))
    b = Benchmark.from_dir(HERE, data={"media_root": str(media)})
    samples = list(b.reader("test", rows=range(0, 3), batch_size=1))
    assert [s.row for s in samples] == [0, 1, 2]
    for s in samples[:3]:
        assert set(s.streams) == {"video", "audio"} and s.meta == {"facts": {}}
        assert s.streams["video"].bytes() == (media / "video" / f"{s.clip_id}.mp4").read_bytes()
        assert s.streams["audio"].bytes() == (media / "audio" / f"{s.clip_id}.wav").read_bytes()
        assert s.streams["audio"].info["sample_rate"] == 16000 and s.streams["audio"].info["channels"] == 1
        assert {"duration_s", "fps", "width", "height"} <= set(s.streams["video"].info)
