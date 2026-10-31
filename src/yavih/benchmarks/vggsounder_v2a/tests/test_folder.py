"""Folder tests of the packaged ``vggsounder_v2a`` benchmark (protocol_version 1).

Run only this folder: ``python -m pytest "$(yavih bench path vggsounder_v2a)/tests" -q -p no:cacheprovider``.

Three groups:
- file-level checks (yaml, manifest, README.md / setup.md facts) that need only numpy + pyyaml;
- checks through the core (``Benchmark``, ``eval.py``'s ``SCORER`` and metrics, the derangement floor, the bridge
  template's describe and dict encode against the folder's output schema): always run, no av_bench needed;
- the manifest re-freeze (``make_manifest.py --check``): needs torch (the seeded draw), skipped without it.

Scoring itself needs av_bench, its weights and a GPU; the conformance fixture covers it where those are installed.
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import subprocess
import sys
import types
from collections import Counter

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_bytes, rows, bench = F.spec, F.manifest_bytes, F.rows, F.bench

N_ROWS = 3000
METRICS = ["fd_passt", "fd_panns", "fd_vgg", "kad_passt", "kad_panns", "kl_passt", "kl_panns", "is_passt", "is_panns", "ib_score", "desync",
           "ib_score_ref", "desync_ref", "n_eval"]
FLOORS = ["ib_score_floor", "desync_floor", "kl_passt_floor"]


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(scope="module")
def eval_mod():
    return F.load_module(HERE / "eval.py", "vggsounder_v2a_folder_eval")


# ----------------------------------------------------------------------------- files and yaml


def test_folder_lint_clean():
    # conformance/expected.json is recorded where av_bench, its weights and a GPU are installed
    F.assert_lint_clean(HERE, skip=("conformance",) if not (HERE / "conformance" / "expected.json").is_file() else ())


def test_yaml_identity(spec):
    assert spec["schema_version"] == 3 and spec["name"] == HERE.name == "vggsounder_v2a"
    assert spec["scorer"] == "vggsounder_v2a" and spec["protocol_version"] == 1
    assert "generation" in spec["capability"]
    (entry,) = spec["changelog"]
    assert entry["version"] == 1 and entry["invalidates"] is False and entry["supersedes"] == []


def test_splits_and_fields(spec):
    assert spec["splits"]["train"] == [] and spec["splits"]["eval"] == ["test"]
    assert spec["manifest"]["label"] == {"column": None, "kind": "none"}
    (name,) = spec["fields"]
    f = spec["fields"][name]
    assert name == "video" and f["storage"] == "file" and f["kind"] == "video" and f["root"] == "derived_root"
    ic = spec["input_contract"]
    assert set(ic["handed_over"]) == {"video"} and ic["facts"] == ["audio_length"]
    assert ic["materialize"]["kind"] == "strip_audio" and "-an" in ic["materialize"]["tool"]["args"]


def test_output_contract(spec):
    oc = spec["output_contract"]
    assert oc["sampling"] == "required"
    assert oc["schema"] == {"audio": {"type": "wav", "channels": 1, "duration_s": [8.0, 8.05]}}
    assert oc["requires"] == {"pos": [], "streams": [], "grid": None, "t_bins": None}
    # the scored window and the accepted duration agree: exactly audio_length, plus 50 ms of slack above
    lo, hi = oc["schema"]["audio"]["duration_s"]
    assert lo == spec["protocol"]["audio_length"] and hi - lo <= 0.05 + 1e-9


def test_external_pin(spec):
    (e,) = spec["external"]
    assert e["name"] == "av_bench" and e["version_pin"] == "1.0.0"
    assert "@f351b9a6fc6abde746d5f8e1d4c47c883319cb41#egg=av_bench" in e["pip"]


def test_metric_lists(spec, eval_mod):
    m = spec["metrics"]
    assert m["list"] == METRICS and m["floors"] == FLOORS and m["headline"] == "fd_passt"
    assert set(m["table"]) <= set(m["list"])
    F.assert_metric_names_resolve(HERE, spec)
    assert set(eval_mod.AV_BENCH_KEYS.values()) <= set(METRICS)
    assert eval_mod.LOWER_IS_BETTER <= set(METRICS)
    assert set(spec["protocol"]) - {"clip_seconds", "train_draw", "views"} == set(eval_mod.PROTOCOL_KEYS)


# ----------------------------------------------------------------------------- manifest


def test_manifest_hashes(spec, rows):
    F.assert_manifest_hashes(HERE, spec)
    assert len(rows) == N_ROWS == spec["budget"]["clips"]["test"]


def test_manifest_rows(spec, rows):
    assert F.assert_splits_assigned_once(spec, rows) == {"test": N_ROWS}
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == N_ROWS
    for r in rows:
        assert r["rel_media"] == f"muted/{r['clip_id']}.mp4"
        assert r["rel_video"].endswith(f"{r['clip_id']}.mp4") and r["rel_audio"].endswith(f"{r['clip_id']}.wav")
        labs = r["labels_audible"].split(";")
        assert labs and all(labs) and len(set(labs)) == len(labs)


def test_curation_sums(spec, rows):
    up = spec["manifest"]["upstream"]
    cur = up["curation"]
    assert cur["total_upstream"] == up["rows"] == 12008
    assert cur["per_split"]["test"] == {"listed": N_ROWS, "staged": N_ROWS, "missing": 0}
    assert cur["total_upstream"] == N_ROWS + sum(cur["excluded"].values())
    assert cur["excluded"]["shorter_than_audio_length"] == 113


def test_rows_come_from_the_vggsounder_eval_rows(spec, rows):
    src = HERE.parent / "vggsounder" / "manifest.csv.gz"
    up = list(csv.DictReader(io.StringIO(gzip.decompress(src.read_bytes()).decode())))
    test = {r["clip_id"]: r for r in up if r["split"] == "test"}
    for r in rows:
        u = test[r["clip_id"]]
        assert (u["rel_video"], u["rel_audio"]) == (r["rel_video"], r["rel_audio"])
        audible = [t for c in (u["labels_a"], u["labels_av"]) for t in c.split(";") if t]
        assert r["labels_audible"].split(";") == list(dict.fromkeys(audible))


def test_draw_is_class_balanced(rows):
    first = Counter(r["labels_audible"].split(";")[0] for r in rows)
    assert len(first) == 305
    # the cap is spread evenly: no class holds more than its share plus the classes too small to fill theirs
    assert max(first.values()) <= int(np.ceil(N_ROWS / len(first))) + 5


def test_short_clips_are_excluded(rows):
    mm = F.load_module(HERE / "make_manifest.py", "vggsounder_v2a_mm")
    short = mm._short_ids()  # sha256-pinned
    assert len(short) == 113 and not short & {r["clip_id"] for r in rows}
    with open(HERE / "short_clips.csv", newline="") as f:
        for r in csv.DictReader(f):  # the audio is too short, or av-benchmark's video loader refuses the window
            assert float(r["audio_s"]) < mm.AUDIO_LENGTH or "too short" in r["reason"], r


@pytest.mark.skipif(not _has("torch"), reason="the seeded draw needs torch")
def test_make_manifest_check():
    r = subprocess.run([sys.executable, str(HERE / "make_manifest.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ----------------------------------------------------------------------------- README / setup.md


def test_readme_follows_the_review():
    text = (HERE / "README.md").read_text()
    assert "Audio generative models in the multimodal era" in text
    assert "mitsufuji2026audio" in text


def test_setup_names_the_prepare_steps():
    obtain = F.section((HERE / "setup.md").read_text(), 2, "Obtain the data")
    for step in ("01", "02"):
        assert f"yavih bench prepare vggsounder_v2a {step}" in obtain
    assert sorted(p.name for p in (HERE / "prepare").iterdir() if p.name[:2].isdigit()) == ["01_mute.py", "02_verify.sh"]


# ----------------------------------------------------------------------------- through the core


def test_bench_loads(bench):
    assert bench.name == "vggsounder_v2a" and bench.spec.readout == "vggsounder_v2a"
    assert bench.protocol_hash == "cd82b94880351c03de1a1032d3d2cc12c29ca2a69ae8e9ba6e86d398d44347a0"
    assert bench.input_hash == "f9b97023716899aceea730227050888f584172a48aa6a8ed310c38013d2d807a"
    view = bench.view()
    assert view.fields == {"video": {"storage": "file", "kind": "video"}} and view.facts == {"audio_length": 8.0}
    assert not hasattr(view, "manifest")


def test_scorer(eval_mod):
    s = eval_mod.SCORER
    assert s.name == "vggsounder_v2a" and s.kind == "direct" and s.needs_gpu
    assert s.floor_outputs is eval_mod.floor_outputs and s.conformance_population is not None


def test_eval_imports_without_av_bench(monkeypatch):
    monkeypatch.setitem(sys.modules, "av_bench", None)
    mod = F.load_module(HERE / "eval.py", "vggsounder_v2a_no_av_bench")
    assert mod.SCORER.name == "vggsounder_v2a"


@pytest.mark.parametrize("n", [2, 3, 5, 32, 100, 3000])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_derangement(eval_mod, n, seed):
    perm = eval_mod.derangement(n, seed)
    assert sorted(perm) == list(range(n)) and all(p != i for i, p in enumerate(perm))
    assert perm == eval_mod.derangement(n, seed)


def test_floor_outputs_repair_reference_audio(bench, eval_mod, spec):
    rows_ = list(range(16))
    targets = eval_mod.targets_of(bench, "test", rows_)
    assert [t["clip_id"] for t in targets] == [bench.manifest("test").clip_id[i] for i in rows_]
    floors = eval_mod.floor_outputs(targets, len(targets), protocol=spec["protocol"])
    assert len(floors) == len(targets)
    for t, f in zip(targets, floors, strict=True):
        assert f["reference_of"] != t["clip_id"] and f["rel_audio"].endswith(f"{f['reference_of']}.wav")
    assert sorted(f["reference_of"] for f in floors) == sorted(t["clip_id"] for t in targets)


# ----------------------------------------------------------------------------- the bridge template


class _FakeRaw:
    def __init__(self, info: dict) -> None:
        self.name = self.kind = "video"
        self.path, self.member, self.info = "/nonexistent.mp4", None, info


def test_bridge_template(bench, spec):
    pytest.importorskip("soundfile")
    from yavih import codecs
    from yavih.bridge import _check_sampling, describe_outputs

    mod = F.load_module(HERE / "bridge_template.py", "vggsounder_v2a_template")
    b = mod.TemplateBridge(bench.view(), {})
    d = b.describe()
    _check_sampling(d["sampling"])
    schema = spec["output_contract"]["schema"]
    assert describe_outputs(d, schema) == ["audio"]
    samples = [types.SimpleNamespace(row=i, clip_id=f"c{i}", split="test", meta={"facts": {"audio_length": 8.0}}, cells={},
                                     streams={"video": _FakeRaw({"duration_s": secs, "fps": 25.0, "width": 64, "height": 48,
                                                                 "container": "mp4", "bytes": 1})})
               for i, secs in enumerate((10.0, 8.0))]
    out = F.encode_prepared(b, samples)
    assert set(out) == {"audio", "used"} and len(out["audio"]) == len(out["used"]) == 2
    for blob, secs in zip(out["audio"], (8.0, 8.0), strict=True):
        assert not [f for f in codecs.WAV.check(blob, schema["audio"], {}) if f.is_error]
        x, sr = codecs.read_wav(blob)
        assert abs(len(x) / sr - secs) < 1e-3


def test_clip_ids_survive_av_bench_pairing(rows):
    # av_bench.utils.clean_sample_name (pinned commit) rewrites names of 20, 21 and 30 characters; an 18-character
    # VGGSound id (11-character YouTube id, "_", 6-digit start) is returned unchanged
    assert {len(r["clip_id"]) for r in rows} == {18}


def test_check_cache_catches_a_skipped_file(tmp_path, eval_mod):
    torch = pytest.importorskip("torch")
    torch.save({"a": torch.zeros(2), "b": torch.zeros(2)}, tmp_path / "pann_features.pth")
    eval_mod.check_cache(tmp_path, ["a", "b"])
    torch.save({"b": torch.zeros(2)}, tmp_path / "passt_logits.pth")  # "a" failed to load: av_bench scored "b" twice
    with pytest.raises(RuntimeError, match="missing \\['a'\\]"):
        eval_mod.check_cache(tmp_path, ["a", "b"])


def _kadtk_reference(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """kadtk's calc_kernel_audio_distance (@7373161) transcribed: x = reference, y = generated, materialized sets."""
    gamma = 1 / (2 * bandwidth**2 + 1e-8)
    def k(a, b):
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return np.exp(-gamma * d2)
    m, n = x.shape[0], y.shape[0]
    kxx, kyy = k(x, x), k(y, y)
    kxx_mean = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    kyy_mean = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(100 * (kxx_mean + kyy_mean - 2 * k(x, y).mean()))


def test_kad_matches_kadtk_and_its_fast_resampling(eval_mod):
    rng = np.random.default_rng(0)
    ref, gen = rng.normal(size=(40, 8)), rng.normal(loc=0.3, size=(40, 8))
    k = eval_mod.kad_kernels(ref, gen)
    d = np.sqrt(((ref[:, None] - ref[None]) ** 2).sum(-1))[np.triu_indices(40, 1)]
    assert float(k["bandwidth"]) == pytest.approx(float(np.median(d)))
    assert eval_mod.kad_from_kernels(k)[0] == pytest.approx(_kadtk_reference(ref, gen, float(k["bandwidth"])), rel=1e-9)
    idx = rng.integers(0, 40, size=40)  # one bootstrap resample, the pairs kept paired
    w = np.bincount(idx, minlength=40)[None].astype(float)
    assert eval_mod.kad_from_kernels(k, w)[0] == pytest.approx(_kadtk_reference(ref[idx], gen[idx], float(k["bandwidth"])), rel=1e-9)
    # identical sets: kadtk's cross term keeps the self-pairs, so KAD is slightly negative, -100 * 2 (1 - mean K) / n
    same = eval_mod.kad_kernels(ref, ref.copy())
    assert -200 / 40 <= eval_mod.kad_from_kernels(same)[0] < 0


def test_per_clip_kl_matches_av_bench_formula(eval_mod):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as tf

    rng = np.random.default_rng(1)
    gt, pred = rng.normal(size=(6, 527)), rng.normal(size=(6, 527))
    ref = float(tf.kl_div(tf.log_softmax(torch.tensor(pred), 1), tf.log_softmax(torch.tensor(gt), 1), reduction="sum", log_target=True) / 6)
    assert eval_mod._kl_rows(gt, pred).mean() == pytest.approx(ref, rel=1e-9)
