"""Folder tests of the packaged ``audiocaps_t2a`` benchmark (protocol_version 1).

Run only this folder: ``python -m pytest "$(yavih bench path audiocaps_t2a)/tests" -q -p no:cacheprovider``.

Three groups:
- file-level checks (yaml, manifest, README.md / setup.md facts) that need only numpy + pyyaml;
- checks through the core (``Benchmark`` with a cell-only input, ``eval.py``'s ``SCORER`` and metrics, the
  derangement floor, the bridge template's describe and dict encode against the folder's output schema): always run,
  no av_bench needed;
- source checks (``make_manifest.py --check``, the staged recordings' sha256): run when ``source_root`` /
  ``derived_root`` resolve through yavih's config resolver and hold the data, skipped otherwise.

Scoring itself needs av_bench, its weights and a GPU; the conformance fixture covers it where those are installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
import types

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_bytes, rows, bench = F.spec, F.manifest_bytes, F.rows, F.bench

N_ROWS = 776
METRICS = ["fd_passt", "fd_panns", "fd_vgg", "kad_passt", "kad_panns", "kl_passt", "kl_panns", "is_passt", "is_panns", "clap_laion", "clap_ms",
           "clap_laion_ref", "clap_ms_ref", "n_eval"]
FLOORS = ["clap_laion_floor", "clap_ms_floor", "kl_passt_floor"]
_source = F.site_root(HERE, "source_root", needs=("upstream/test.csv",))
_derived = F.site_root(HERE, "derived_root", needs=("audio",))


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(scope="module")
def eval_mod():
    return F.load_module(HERE / "eval.py", "audiocaps_t2a_folder_eval")


# ----------------------------------------------------------------------------- files and yaml


def test_folder_lint_clean():
    # conformance/expected.json is recorded where av_bench, its weights and a GPU are installed
    F.assert_lint_clean(HERE, skip=("conformance",) if not (HERE / "conformance" / "expected.json").is_file() else ())


def test_yaml_identity(spec):
    assert spec["schema_version"] == 3 and spec["name"] == HERE.name == "audiocaps_t2a"
    assert spec["scorer"] == "audiocaps_t2a" and spec["protocol_version"] == 1
    assert "generation" in spec["capability"]
    assert spec["source"]["revision"] == "b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded"
    (entry,) = spec["changelog"]
    assert entry["version"] == 1 and entry["invalidates"] is False and entry["supersedes"] == []


def test_the_only_input_is_the_caption_cell(spec):
    assert spec["splits"]["train"] == [] and spec["splits"]["eval"] == ["test"]
    assert spec["manifest"]["label"] == {"column": None, "kind": "none"}
    assert spec["fields"] == {"caption": {"storage": "cell", "kind": "text", "column": "caption"}}
    assert spec["manifest"]["columns"]["caption"] == {"type": "str", "role": "cell", "field": "caption"}
    ic = spec["input_contract"]
    assert set(ic["handed_over"]) == {"caption"} and ic["info"] == {} and ic["facts"] == ["audio_length"]
    assert ic["materialize"] is None


def test_output_contract(spec):
    oc = spec["output_contract"]
    assert oc["sampling"] == "required"
    assert oc["schema"] == {"audio": {"type": "wav", "channels": 1, "duration_s": [10.0, 10.05]}}
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
    assert len(set(ids)) == N_ROWS and len({r["audiocap_id"] for r in rows}) == N_ROWS
    for r in rows:
        assert re.fullmatch(r"[A-Za-z0-9_-]{11}_\d{6}", r["clip_id"]), r["clip_id"]
        assert r["rel_audio"] == f"audio/{r['clip_id']}.wav"
        assert re.fullmatch(r"[0-9a-f]{64}", r["audio_sha256"])
        assert r["caption"] and r["caption"] == r["caption"].strip()


def test_curation_sums(spec):
    up = spec["manifest"]["upstream"]
    cur = up["curation"]
    assert cur["total_upstream"] == up["rows"] == 975
    assert cur["per_split"]["test"] == {"listed": N_ROWS, "staged": N_ROWS, "missing": 0}
    assert cur["total_upstream"] == N_ROWS + sum(cur["excluded"].values())
    assert cur["excluded"] == {"not_in_mirror": 92, "mirror_audio_ambiguous": 1, "shorter_than_audio_length": 106}


def test_clip_ids_survive_av_bench_pairing(rows):
    # av_bench.utils.clean_sample_name (pinned commit) rewrites names of 20, 21 and 30 characters
    assert {len(r["clip_id"]) for r in rows} == {18}


@pytest.mark.skipif(_source is None or not _has("pyarrow"), reason="needs source_root (prepare/01) and pyarrow")
def test_make_manifest_check():
    r = subprocess.run([sys.executable, str(HERE / "make_manifest.py"), "--root", str(_source), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(_derived is None, reason="needs derived_root (prepare/02)")
def test_staged_recordings_match_the_frozen_sha256(rows):
    bad = [r["clip_id"] for r in rows if hashlib.sha256((_derived / r["rel_audio"]).read_bytes()).hexdigest() != r["audio_sha256"]]
    assert not bad, bad[:5]


# ----------------------------------------------------------------------------- README / setup.md


def test_readme_follows_the_review():
    text = (HERE / "README.md").read_text()
    assert "Audio generative models in the multimodal era" in text and "mitsufuji2026audio" in text


def test_setup_names_the_prepare_steps():
    obtain = F.section((HERE / "setup.md").read_text(), 2, "Obtain the data")
    for step in ("01", "02", "03"):
        assert f"yavih bench prepare audiocaps_t2a {step}" in obtain
    assert sorted(p.name for p in (HERE / "prepare").iterdir() if p.name[:2].isdigit()) == ["01_download.sh", "02_stage.py", "03_verify.sh"]


# ----------------------------------------------------------------------------- through the core


def test_bench_loads(bench, rows, monkeypatch, tmp_path):
    monkeypatch.setenv("YAVIH_DATA__ROOT", str(tmp_path))  # reading a caption needs no data; the roots must still resolve
    assert bench.name == "audiocaps_t2a" and bench.spec.readout == "audiocaps_t2a"
    assert bench.protocol_hash == "22b53a9988f61ba8284c26a729ee580ac2a2884bea65bf2d8840c983faab2552"
    assert bench.input_hash == "6f4dd3c852142ebb113ca7eab8bd982a7c2d909b564619cddd817ccd5ea48ab2"
    assert bench.spec.streams == {}
    view = bench.view()
    assert view.fields == {"caption": {"storage": "cell", "kind": "text"}} and view.streams == {} and view.facts == {"audio_length": 10.0}
    s = next(iter(bench.reader("test", rows=range(1))))
    assert s.streams == {} and s.cells == {"caption": rows[0]["caption"]}


def test_scorer(eval_mod):
    s = eval_mod.SCORER
    assert s.name == "audiocaps_t2a" and s.kind == "direct" and s.needs_gpu
    assert s.floor_outputs is eval_mod.floor_outputs and s.conformance_population is not None
    assert set(eval_mod.WEIGHTS) == {"music_speech_audioset_epoch_15_esc_89.98.pt", "synchformer_state_dict.pth"}


def test_eval_imports_without_av_bench(monkeypatch):
    monkeypatch.setitem(sys.modules, "av_bench", None)
    assert F.load_module(HERE / "eval.py", "audiocaps_t2a_no_av_bench").SCORER.name == "audiocaps_t2a"


@pytest.mark.parametrize("n", [2, 3, 32, N_ROWS])
def test_derangement(eval_mod, n):
    perm = eval_mod.derangement(n, 0)
    assert sorted(perm) == list(range(n)) and all(p != i for i, p in enumerate(perm))


def test_targets_and_floor(bench, eval_mod, spec, rows):
    targets = eval_mod.targets_of(bench, "test", range(16))
    assert [(t["clip_id"], t["caption"], t["rel_audio"]) for t in targets] == [(r["clip_id"], r["caption"], r["rel_audio"]) for r in rows[:16]]
    floors = eval_mod.floor_outputs(targets, len(targets), protocol=spec["protocol"])
    for t, f in zip(targets, floors, strict=True):
        assert f["reference_of"] != t["clip_id"] and f["rel_audio"] == f"audio/{f['reference_of']}.wav"


def test_check_cache_catches_a_skipped_file(tmp_path, eval_mod):
    torch = pytest.importorskip("torch")
    torch.save({"a": torch.zeros(2), "b": torch.zeros(2)}, tmp_path / "clap_laion_text.pth")
    eval_mod.check_cache(tmp_path, ["a", "b"])
    torch.save({"b": torch.zeros(2)}, tmp_path / "clap_laion_audio.pth")
    with pytest.raises(RuntimeError, match="missing \\['a'\\]"):
        eval_mod.check_cache(tmp_path, ["a", "b"])


# ----------------------------------------------------------------------------- the bridge template


def test_bridge_template(bench, spec):
    pytest.importorskip("soundfile")
    from yavih import codecs
    from yavih.bridge import _check_sampling, describe_outputs, validate_input

    mod = F.load_module(HERE / "bridge_template.py", "audiocaps_t2a_template")
    b = mod.TemplateBridge(bench.view(), {})
    d = b.describe()
    _check_sampling(d["sampling"])
    validate_input(d["input"], {}, [], {"caption": "text"})
    schema = spec["output_contract"]["schema"]
    assert describe_outputs(d, schema) == ["audio"]
    samples = [types.SimpleNamespace(row=i, clip_id=f"c{i}", split="test", meta={"facts": {"audio_length": 10.0}}, streams={},
                                     cells={"caption": cap}) for i, cap in enumerate(("a dog barks", "rain on a roof"))]
    out = F.encode_prepared(b, samples)
    assert set(out) == {"audio", "used"} and out["used"] == [{}, {}]
    for blob in out["audio"]:
        assert not [f for f in codecs.WAV.check(blob, schema["audio"], {}) if f.is_error]
        x, sr = codecs.read_wav(blob)
        assert abs(len(x) / sr - 10.0) < 1e-3


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


def test_ms_clap_crop_is_seeded_per_clip(eval_mod, monkeypatch):
    """MS-CLAP draws its 7 s window from Python's global random; inside _seeded_ms_clap each file's offset depends only
    on the seed and the file's stem, so any file order gives the same windows, and the global random is restored."""
    import random as _random

    wrapper = types.ModuleType("msclap.CLAPWrapper")
    wrapper.random = _random

    class CLAPWrapper:
        def load_audio_into_tensor(self, audio_path, audio_duration, resample=False):
            return wrapper.random.randrange(1000)

    wrapper.CLAPWrapper = CLAPWrapper
    pkg = types.ModuleType("msclap")
    pkg.CLAPWrapper = wrapper
    monkeypatch.setitem(sys.modules, "msclap", pkg)
    monkeypatch.setitem(sys.modules, "msclap.CLAPWrapper", wrapper)
    files = [f"/tmp/x/{c}.wav" for c in ("a", "b", "c", "d")]

    def offsets(order, seed=0):
        with eval_mod._seeded_ms_clap(seed):
            return {f: CLAPWrapper().load_audio_into_tensor(f, 7) for f in order}

    assert offsets(files) == offsets(list(reversed(files))) and offsets(files) != offsets(files, seed=1)
    assert wrapper.random is _random and CLAPWrapper.load_audio_into_tensor.__name__ == "load_audio_into_tensor"
