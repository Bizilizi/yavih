"""The complete metric registry table (docs/readouts.md), patterns, lazy fns, resolution helpers."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from yavih.metrics import registry as R

# name -> (readouts, kind, shape, protocol_keys) exactly as in the documented registry table
TABLE = {
    "acc": ({"clf"}, "metric", "scalar", set()),
    "n_eval": ({"*"}, "disclosure", "scalar", set()),  # every readout, including a registered one
    "n_train": ({"clf"}, "disclosure", "scalar", set()),
    "acc_floor": ({"clf"}, "floor", "scalar", set()),
    "map": ({"clf"}, "metric", "scalar", set()),
    "effective_pos_classes": ({"clf"}, "disclosure", "scalar", set()),
    "map_floor": ({"clf"}, "floor", "scalar", set()),
    "acc_mean": ({"clf"}, "metric", "scalar", set()),
    "acc_pstd": ({"clf"}, "metric", "scalar", set()),
    "acc_per_fold": ({"clf"}, "metric", "list", set()),
    "n_folds": ({"clf"}, "disclosure", "scalar", set()),
    "seg_acc": ({"seq"}, "metric", "scalar", {"num_segments", "segments_scored", "segment_offset", "background_index"}),
    "clip_acc": ({"seq"}, "metric", "scalar", {"num_segments", "segments_scored", "segment_offset", "background_index"}),
    "seg_acc_boundary": ({"seq"}, "metric", "scalar", {"num_segments", "segments_scored", "segment_offset", "background_index"}),
    "n_boundary": ({"seq"}, "disclosure", "scalar", {"num_segments", "segments_scored", "segment_offset", "background_index"}),
    "seg_acc_floor": ({"seq"}, "floor", "scalar", {"num_segments", "segments_scored", "segment_offset", "background_index"}),
    "seg_acc_boundary_floor": ({"seq"}, "floor", "scalar", {"num_segments", "segments_scored", "segment_offset", "background_index"}),
    "d1": ({"grid"}, "metric", "scalar", {"max_depth", "target_size"}),
    "d2": ({"grid"}, "metric", "scalar", {"max_depth", "target_size"}),
    "d3": ({"grid"}, "metric", "scalar", {"max_depth", "target_size"}),
    "absrel": ({"grid"}, "metric", "scalar", {"max_depth", "target_size"}),
    "rmse": ({"grid"}, "metric", "scalar", {"max_depth", "target_size"}),
    "d1_floor": ({"grid"}, "floor", "scalar", {"max_depth", "target_size"}),
    "iou_mean": ({"box"}, "metric", "scalar", set()),
    "acc@0.5": ({"box"}, "metric", "scalar", set()),
    "acc@0.3": ({"box"}, "metric", "scalar", set()),
    "center_in_gt": ({"box"}, "metric", "scalar", set()),
    "n_excluded_degenerate": ({"box"}, "disclosure", "scalar", set()),
    "iou_floor": ({"box"}, "floor", "scalar", set()),
    "iou_shuffled": ({"box"}, "control", "scalar", set()),
    "audio_blind": ({"box"}, "control", "scalar", set()),
}
#: avsync's metrics are its probe.py METRICS overlay, with the same readouts/kind/shape/keys.
AVSYNC_TABLE = {
    "acc_tol1": ({"clf"}, "metric", "scalar", {"offsets_seconds"}),
    "mae_seconds": ({"clf"}, "metric", "scalar", {"offsets_seconds"}),
    "acc_coarse5": ({"clf"}, "metric", "scalar", {"offsets_seconds", "coarse_grid_seconds"}),
    "n_coarse5": ({"clf"}, "disclosure", "scalar", {"offsets_seconds", "coarse_grid_seconds"}),
    "acc_floor_arm": ({"clf"}, "control", "scalar", {"floor_sigmas"}),
    "floor_violated": ({"clf"}, "control", "scalar", {"floor_sigmas"}),
}
PATTERNS = {"auc_pr_<set>": "metric", "auc_roc_<set>": "metric", "auc_pr_<set>_floor": "floor"}


def test_registry_has_exactly_the_documented_metric_table():
    assert set(R.REGISTRY) >= set(TABLE), set(TABLE) - set(R.REGISTRY)
    extra = set(R.REGISTRY) - set(TABLE)
    assert not extra, f"registry names not in the documented table (a new metric updates this table): {extra}"
    for name, (readouts, kind, shape, keys) in TABLE.items():
        spec = R.REGISTRY[name]
        assert spec.name == name
        assert spec.readouts == frozenset(readouts), name
        assert spec.kind == kind, name
        assert spec.shape == shape, name
        assert spec.protocol_keys == frozenset(keys), name
        assert isinstance(spec.fn, R.LazyFn) and spec.fn.ref.startswith("yavih.metrics."), name


def test_patterns_registered():
    assert set(R.PATTERNS) == set(PATTERNS)
    for template, kind in PATTERNS.items():
        assert R.PATTERNS[template].kind == kind
        assert R.PATTERNS[template].readouts == frozenset({"clf"})


def test_expand_sets_and_match_pattern():
    specs = R.expand_sets(["audible", "visible", "all"])
    assert set(specs) == {
        "auc_pr_audible", "auc_roc_audible", "auc_pr_audible_floor",
        "auc_pr_visible", "auc_roc_visible", "auc_pr_visible_floor",
        "auc_pr_all", "auc_roc_all", "auc_pr_all_floor",
    }
    s = specs["auc_pr_audible_floor"]
    assert s.kind == "floor" and s.fn.bound == {"set_name": "audible"} and s.fn.ref == "yavih.metrics.classification:auc_pr_floor"
    pat, set_name = R.match_pattern("auc_pr_all_floor")
    assert pat.template == "auc_pr_<set>_floor" and set_name == "all"  # floor pattern wins over set="all_floor"
    pat, set_name = R.match_pattern("auc_roc_visible")
    assert pat.template == "auc_roc_<set>" and set_name == "visible"
    assert R.match_pattern("acc") is None


def test_resolve_and_resolve_names():
    assert R.resolve("acc") is R.REGISTRY["acc"]
    assert R.resolve("auc_pr_audible", sets=["audible"]).name == "auc_pr_audible"
    with pytest.raises(R.UnknownMetric):
        R.resolve("auc_pr_audible")  # no sets declared
    with pytest.raises(R.UnknownMetric):
        R.resolve("auc_pr_audible", sets=["visible"])
    with pytest.raises(R.UnknownMetric):
        R.resolve("nope")
    # the VGGSounder metrics.list resolves for clf
    names = ["auc_pr_audible", "auc_roc_audible", "auc_pr_visible", "auc_roc_visible", "auc_pr_all", "auc_roc_all", "n_eval"]
    out = R.resolve_names(names, "clf", sets=["audible", "visible", "all"])
    assert list(out) == names
    with pytest.raises(R.MetricNotAllowed):
        R.resolve_names(["seg_acc"], "clf")
    with pytest.raises(R.MetricNotAllowed):
        R.resolve_names(["acc_floor"], "clf", kinds=["metric", "disclosure"])  # a floor named in metrics.list
    assert R.resolve_names(["acc_floor"], "clf", kinds=["floor"])


def _avsync_metrics():
    from yavih.benchmark import Benchmark
    from yavih.readouts.base import folder_probe

    return folder_probe(Benchmark.load("avsync")).METRICS


def test_avsync_folder_metrics_match_their_table():
    m = _avsync_metrics()
    assert set(m) == set(AVSYNC_TABLE)
    for name, (readouts, kind, shape, keys) in AVSYNC_TABLE.items():
        assert (set(m[name].readouts), m[name].kind, m[name].shape, set(m[name].protocol_keys)) == (readouts, kind, shape, keys), name


def test_protocol_keys_for_matches_the_packaged_benchmarks():
    m = _avsync_metrics()
    avsync = [*R.resolve_names(["acc", "n_eval"], "clf").values(), *m.values()]
    assert R.protocol_keys_for(avsync) == {"offsets_seconds", "coarse_grid_seconds", "floor_sigmas"}
    ave = R.resolve_names(["seg_acc", "clip_acc", "seg_acc_boundary", "n_boundary", "n_eval"], "seq")
    assert R.protocol_keys_for(ave.values()) == {"num_segments", "segments_scored", "segment_offset", "background_index"}
    k400 = R.resolve_names(["acc", "n_eval"], "clf")
    assert R.protocol_keys_for(k400.values()) == frozenset()


def test_every_lazy_fn_resolves_to_a_defined_function():
    """Each registry fn names a real function in the module (stubs raise NotImplementedError when called)."""
    for name, spec in R.REGISTRY.items():
        fn = spec.fn.resolve()
        assert callable(fn), name
        module_name, func_name = spec.fn.ref.split(":")
        assert getattr(importlib.import_module(module_name), func_name) is fn
    for template, pat in R.PATTERNS.items():
        assert callable(pat.expand("x").fn.resolve()), template


def test_lazy_fn_call_passes_bound_kwargs_and_rejects_bad_ref():
    spec = R.expand_pattern("auc_pr_<set>", "audible")
    # the implemented fn is reached with set_name bound: without targets for that set it refuses with
    # ValueError naming the missing label set (a stub raising NotImplementedError must FAIL this test)
    with pytest.raises(ValueError, match=r"no targets for label set"):
        spec.fn(None, None, {}, {})
    with pytest.raises(ValueError):
        R.LazyFn("no_colon")
    with pytest.raises(AttributeError):
        R.LazyFn("yavih.metrics.classification:does_not_exist").resolve()


def test_register_refuses_duplicates_and_validates():
    spec = R.MetricSpec(name="acc", fn=R.LazyFn("yavih.metrics.classification:acc"), readouts=frozenset({"clf"}))
    with pytest.raises(ValueError):
        R.register(spec)
    with pytest.raises(ValueError):
        R.MetricSpec(name="x", fn=R.LazyFn("a:b"), readouts=frozenset({"clf"}), kind="nope")
    with pytest.raises(ValueError):
        R.MetricSpec(name="x", fn=R.LazyFn("a:b"), readouts=frozenset({"clf"}), shape="matrix")
    with pytest.raises(ValueError):
        R.MetricSpec(name="x", fn=R.LazyFn("a:b"), readouts=frozenset({"Not a readout"}))
    with pytest.raises(ValueError):
        R.MetricSpec(name="x", fn=R.LazyFn("a:b"), readouts=frozenset())
    # any readout name is accepted (a registered plugin readout); Benchmark.load refuses an unregistered readout
    assert R.MetricSpec(name="x", fn=R.LazyFn("a:b"), readouts=frozenset({"retrieval"})).readouts == {"retrieval"}


def test_ci_kinds():
    assert R.CI_KINDS == {"binomial", "bootstrap_clip", "pooled_binomial"}


def test_registry_importable_without_sklearn_or_torch():
    code = (
        "import sys; sys.modules['sklearn'] = None; sys.modules['torch'] = None\n"
        "import yavih.metrics, yavih.metrics.registry as r; print(len(r.REGISTRY), 'sklearn' in sys.modules and sys.modules['sklearn'] is not None)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    n, imported = out.stdout.split()
    assert int(n) == len(TABLE) and imported == "False"
