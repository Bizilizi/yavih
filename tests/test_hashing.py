"""Canonical JSON and every digest (docs/hashing.md); comparability covers only what yavih controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

from yavih import hashing
from yavih.versions import PLAN_SCHEMA_VERSION, PROBE_SCHEMA, PROTOCOL_SCHEMA, READOUT_VERSION

# ----------------------------------------------------------------------------- canon


def test_canon_is_sorted_compact_and_utf8():
    assert hashing.canon({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert hashing.canon({"é": "ü"}) == '{"é":"ü"}'  # ensure_ascii=False
    assert hashing.sha256_canon({"x": 1}) == hashlib.sha256(b'{"x":1}').hexdigest()


def test_canon_normalizes_tuples_and_numpy_scalars():
    assert hashing.normalize((1, 2)) == [1, 2]
    assert hashing.canon({"t": (1, (2, 3))}) == '{"t":[1,[2,3]]}'
    assert hashing.normalize(np.int64(3)) == 3 and type(hashing.normalize(np.int64(3))) is int
    assert hashing.normalize(np.float32(0.5)) == 0.5 and type(hashing.normalize(np.float32(0.5))) is float
    assert hashing.normalize(np.bool_(True)) is True
    assert hashing.canon({"f": np.float64(0.1)}) == '{"f":0.1}'


def test_canon_floats_shortest_roundtrip_and_bools_stay_bools():
    assert hashing.canon(0.1) == "0.1"
    assert hashing.canon(1e-7) == repr(1e-7)
    assert hashing.canon(1.0) == "1.0"  # a float never collapses into an int
    assert hashing.canon(True) == "true"
    assert hashing.canon({"a": True, "b": 1}) == '{"a":true,"b":1}'
    assert hashing.canon(1) != hashing.canon(1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "a"}, {"s": {1, 2}}, {"b": b"x"}, object(), {"d": np.zeros(2)}])
def test_canon_refuses_non_json(bad):
    with pytest.raises((TypeError, ValueError)):
        hashing.canon(bad)


def test_rng_seed_and_hash12():
    s = hashing.rng_seed("abc:train:0:0")
    assert isinstance(s, int) and s > 0
    assert s == int.from_bytes(hashlib.sha256(b"abc:train:0:0").digest(), "big")
    np.random.default_rng(s)  # accepted by numpy
    assert hashing.hash12("0123456789abcdefXYZ") == "0123456789ab"


# ----------------------------------------------------------------------------- manifest_hash


def test_manifest_hash_formula(synthetic_benchmark):
    sb = synthetic_benchmark
    got = hashing.manifest_hash(sb.manifest_bytes, sb.classes_bytes)
    assert got == sb.manifest_hash  # conftest computes the manifest_hash formula independently
    expected = hashlib.sha256(b"manifest:" + sb.manifest_sha256.encode() + b"\nclasses:" + sb.classes_sha256.encode()).hexdigest()
    assert got == expected


def test_manifest_hash_without_classes():
    m = b"clip_id,split\na,train\n"
    got = hashing.manifest_hash(m, None)
    expected = hashlib.sha256(b"manifest:" + hashlib.sha256(m).hexdigest().encode() + b"\nclasses:none").hexdigest()
    assert got == expected
    assert got != hashing.manifest_hash(m, b"index,name\n0,x\n")


# ----------------------------------------------------------------------------- protocol_hash


#: The hashed blocks of a schema-2 benchmark.yaml; not a folder, just the payload inputs.
BENCH_SPEC = {
    "name": "synth_clf",
    "protocol_version": 2,
    "readout": "clf",
    "protocol": {"clip_seconds": 10.0, "train_draw": None, "views": "none"},
    "splits": {"train": ["train"], "eval": ["val"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95},
    "streams": {"video": {"kind": "video", "media_column": "rel_media", "root": "media_root", "layout": "dir", "layout_params": {}, "required": True}},
    "input_contract": {"handed_over": {"video": "the stored mp4"}, "info": {"video": ["duration_s"]}, "facts": [], "materialize": None},
    "output_contract": {"shape": "fixed", "coordinates": "media", "target_frame": None, "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None}},
    "external": [],
    "metrics": {"headline": "acc", "list": ["acc", "n_eval"], "floors": ["acc_floor"], "controls": [], "chance": {"acc": 0.5}, "ci": {"acc": "binomial"}},
    "readout_params": {},
    "targets": None,
}
MANIFEST_HASH = "m" * 64
#: ``benchmark.yaml[manifest]`` as written: only ``key``, ``label`` and ``columns`` are hashed.
MANIFEST_BLOCK = {
    "file": "manifest.csv.gz",
    "sha256": "f" * 64,
    "classes": "classes.csv",
    "classes_sha256": "c" * 64,
    "key": "clip_id",
    "label": {"column": "label", "kind": "single", "sep": ";", "class_key": "name", "sets": None, "train_set": None,
              "allow_unknown": False, "empty_ok": False},
    "columns": {"rel_media": {"type": "str", "role": "media", "stream": "video"}},
    "upstream": {"url": "https://example.org/x", "note": "provenance"},
}


def _protocol_kwargs(**over):
    spec = BENCH_SPEC
    kw = dict(
        name=spec["name"],
        protocol_version=spec["protocol_version"],
        protocol=spec["protocol"],
        splits=spec["splits"],
        streams=spec["streams"],
        input_contract=spec["input_contract"],
        output_contract=spec["output_contract"],
        external=spec["external"],
        metrics=spec["metrics"],
        readout_params=spec["readout_params"],
        targets=spec["targets"],
        manifest=MANIFEST_BLOCK,
        manifest_hash=MANIFEST_HASH,
        readout=spec["readout"],
    )
    kw.update(over)
    return kw


def test_protocol_schema_tag():
    assert PROTOCOL_SCHEMA == "yavih-protocol-5"  # protocol_hash = H(input_hash, scoring_hash) over the declaration only


def _part(part, comps):
    digests = {k: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
               for k, v in comps.items()}
    payload = {"schema": PROTOCOL_SCHEMA, "part": part, "components": digests}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def test_protocol_hash_payload_matches_the_hand_written_payload():
    """yavih-protocol-5: input side, scoring side, combined — each written out here by hand, from the declaration
    only (typed views: no prose, no display keys, no source digest)."""
    kw = _protocol_kwargs()
    inp = {
        "name": kw["name"],
        "manifest_hash": MANIFEST_HASH,
        "manifest": {"key": "clip_id", "columns": MANIFEST_BLOCK["columns"]},
        "splits": kw["splits"],
        "streams": kw["streams"],
        "input_contract": {"info": kw["input_contract"]["info"], "facts": [], "materialize": None},
        "output_contract": kw["output_contract"],
        "readout": "clf",
        "readout_inputs": {"params": {k: v for k, v in sorted(kw["readout_params"].items()) if k in hashing.INPUT_READOUT_PARAMS},
                           "protocol": {k: v for k, v in sorted(kw["protocol"].items())
                                        if k in set(hashing.INPUT_PROTOCOL_KEYS) | set(kw["input_contract"].get("facts") or [])}},
    }
    sco = {
        "protocol_version": kw["protocol_version"],
        "protocol": kw["protocol"],
        "readout_params": kw["readout_params"],
        "metrics": {"headline": "acc", "list": ["acc", "n_eval"], "floors": ["acc_floor"], "controls": [], "chance": {"acc": 0.5},
                    "ci": {"acc": "binomial"}, "bootstrap": None, "aggregation": None},
        "external": [],
        "targets": None,
        "label": MANIFEST_BLOCK["label"],
        "readout": {"name": "clf", "version": READOUT_VERSION["clf"]},
    }
    ih, sh = _part("input", inp), _part("scoring", sco)
    combined = {"schema": PROTOCOL_SCHEMA, "input_hash": ih, "scoring_hash": sh}
    expected = hashlib.sha256(json.dumps(combined, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    hs = hashing.protocol_hashes(**kw)
    assert (hs.input_hash, hs.scoring_hash, hs.protocol_hash) == (ih, sh, expected)
    assert hashing.protocol_hash(**kw) == expected and hashing.input_hash(**kw) == ih
    assert hs.input_components == hashing.component_digests(inp) and set(hs.scoring_components) == set(sco)
    assert hs.input_values == inp and hs.scoring_values == sco
    # no source digest is an input at all
    for extra in ("reader_sha256", "probe_sha256"):
        with pytest.raises(TypeError):
            hashing.protocol_hashes(**kw, **{extra: "a" * 64})
    # explain(): the canonical text of every component reproduces every digest with stdlib only
    ex = hs.explain()
    assert ex["schema"] == PROTOCOL_SCHEMA and ex["protocol_hash"] == expected
    for side, h in (("input", ih), ("scoring", sh)):
        assert ex[side]["hash"] == h
        for c in ex[side]["components"].values():
            assert hashlib.sha256(c["canon"].encode("utf-8")).hexdigest() == c["digest"]
        digests = {k: c["digest"] for k, c in ex[side]["components"].items()}
        assert hashlib.sha256(json.dumps({"schema": PROTOCOL_SCHEMA, "part": side, "components": digests}, sort_keys=True,
                                         separators=(",", ":"), ensure_ascii=False).encode()).hexdigest() == h


def test_typed_views_exclude_prose_and_display_keys():
    """A wording, instruction or report-layout edit never moves a hash; a value edit does."""
    base = hashing.protocol_hashes(**_protocol_kwargs())
    same = [
        {"protocol": dict(BENCH_SPEC["protocol"], train_draw={"cap": 10, "seed": 0, "balance": "class"})},
        {"metrics": dict(BENCH_SPEC["metrics"], table=["acc"])},
        {"metrics": dict(BENCH_SPEC["metrics"], tolerance={"acc": 0.01})},
        {"input_contract": dict(BENCH_SPEC["input_contract"], handed_over={"video": "another sentence about the mp4"})},
    ]
    td = {"cap": 10, "seed": 0, "balance": "class"}
    a = hashing.protocol_hashes(**_protocol_kwargs(protocol=dict(BENCH_SPEC["protocol"], train_draw=td)))
    b = hashing.protocol_hashes(**_protocol_kwargs(protocol=dict(BENCH_SPEC["protocol"], train_draw=dict(td, drawn_from="a long story"))))
    assert a.protocol_hash == b.protocol_hash != base.protocol_hash
    assert hashing.protocol_hashes(**_protocol_kwargs(protocol=dict(BENCH_SPEC["protocol"], train_draw=dict(td, seed=1)))).protocol_hash != a.protocol_hash
    for over in same[1:]:
        assert hashing.protocol_hashes(**_protocol_kwargs(**over)).protocol_hash == base.protocol_hash, over
    ext = [{"name": "pkg", "pip": "pip install pkg==1.0", "version_pin": "1.0", "used_for": "everything"}]
    e1 = hashing.protocol_hashes(**_protocol_kwargs(external=ext))
    e2 = hashing.protocol_hashes(**_protocol_kwargs(external=[dict(ext[0], pip="pip install --no-deps pkg==1.0", used_for="x")]))
    assert e1.protocol_hash == e2.protocol_hash != base.protocol_hash
    assert hashing.external_view(ext + [{"name": "a", "version_pin": "2"}]) == [{"name": "a", "version_pin": "2"}, {"name": "pkg", "version_pin": "1.0"}]
    assert hashing.metrics_view({"headline": "acc"}) == {"headline": "acc", "list": None, "floors": None, "controls": None, "chance": None,
                                                        "ci": None, "bootstrap": None, "aggregation": None}
    assert hashing.input_contract_view(BENCH_SPEC["input_contract"]) == {"info": {"video": ["duration_s"]}, "facts": [], "materialize": None}
    assert hashing.protocol_view({"train_draw": None, "x": 1}) == {"train_draw": None, "x": 1}


def test_find_prose_flags_sentences_only():
    v = {"a": "clip_id", "b": ["-nostdin", "trim=start={v0}:end={v1},setpts=PTS-{v0}/TB"], "c": {"d": "the quick brown fox jumps over the lazy dog today"},
         "e": "x" * 80, "f": 1.5, "g": None}
    assert hashing.find_prose(v) == [("c.d", v["c"]["d"]), ("e", "x" * 80)]
    assert hashing.find_prose("one two three") == []
    assert hashing.find_prose("one two three four five six seven eight") == [("<root>", "one two three four five six seven eight")]
    assert hashing.find_prose(["a", "b"]) == []


def test_normalized_source_ignores_comments_docstrings_and_layout():
    a = '"""Module doc."""\n# a comment\nimport os\n\n\ndef f(x):\n    """Doc."""\n    #: attr doc\n    y = x + 1  # trailing\n    "a bare string statement"\n    return y\n'
    b = "import os\ndef f(x):\n    y = x+1\n    return y\n"
    assert hashing.source_sha256(a) == hashing.source_sha256(b)
    assert hashing.source_sha256(b) != hashing.source_sha256(b.replace("x+1", "x+2"))
    assert hashing.source_sha256(b) != hashing.source_sha256(b.replace("return y", "return x"))
    # a string that is a value (not a bare statement) still counts
    assert hashing.source_sha256('X = "a"\n') != hashing.source_sha256('X = "b"\n')
    # version stability: the serialization drops empty/absent fields, so this digest is the same on Python 3.10 through 3.15 (checked)
    assert hashing.source_sha256(b) == "84be28145e0add60b16b477bc3aee08e7b45e0b528892bd71176ec3d2dd8f0ba"
    assert hashing.source_sha256(b.encode()) == hashing.source_sha256(b)
    with pytest.raises(SyntaxError):
        hashing.source_sha256("def (:\n")


def test_protocol_hash_requires_the_manifest_block():
    """How rows and labels are read changes numbers, so it may not silently default."""
    kw = _protocol_kwargs()
    del kw["manifest"]
    with pytest.raises(TypeError):
        hashing.protocol_hash(**kw)
    with pytest.raises(TypeError):
        hashing.protocol_hash(**_protocol_kwargs(manifest=None))


def test_protocol_hash_sensitivity():
    base = hashing.protocol_hash(**_protocol_kwargs())
    assert hashing.protocol_hash(**_protocol_kwargs()) == base  # deterministic
    assert hashing.protocol_hash(**_protocol_kwargs(protocol_version=3)) != base
    assert hashing.protocol_hash(**_protocol_kwargs(manifest_hash="0" * 64)) != base
    assert hashing.protocol_hash(**_protocol_kwargs(protocol=dict(BENCH_SPEC["protocol"], clip_seconds=9.0))) != base
    # streams and both contracts are hashed
    assert hashing.protocol_hash(**_protocol_kwargs(streams={"video": dict(BENCH_SPEC["streams"]["video"], layout="tar_index")})) != base
    assert hashing.protocol_hash(**_protocol_kwargs(input_contract=dict(BENCH_SPEC["input_contract"], facts=["clip_seconds"]))) != base
    oc = dict(BENCH_SPEC["output_contract"], requires={"pos": ["t"], "streams": [], "grid": None, "t_bins": None})
    assert hashing.protocol_hash(**_protocol_kwargs(output_contract=oc)) != base
    ext = [{"name": "pkg", "pip": "pkg==1.0", "version_pin": "1.0", "used_for": "x"}]
    assert hashing.protocol_hash(**_protocol_kwargs(external=ext)) != base
    assert hashing.protocol_hash(**_protocol_kwargs(external=[dict(ext[0], version_pin="1.1")])) != hashing.protocol_hash(**_protocol_kwargs(external=ext))
    # aliases/title/budget/source/data/changelog (and a top-level `requires`) are not inputs at all
    for extra in ("aliases", "requires", "source", "data", "budget", "changelog", "title"):
        with pytest.raises(TypeError):
            hashing.protocol_hash(**_protocol_kwargs(), **{extra: {}})
    with pytest.raises(ValueError):
        hashing.protocol_hash(**_protocol_kwargs(readout="nope"))


def _sides(**over):
    hs = hashing.protocol_hashes(**_protocol_kwargs(**over))
    return hs.input_hash, hs.scoring_hash


@pytest.mark.parametrize("over", [
    {"protocol_version": 3},
    {"metrics": dict(BENCH_SPEC["metrics"], headline="top5")},
    {"external": [{"name": "pkg", "pip": "pkg==1.1", "version_pin": "1.1", "used_for": "x"}]},
    {"protocol": dict(BENCH_SPEC["protocol"], clip_seconds=9.0)},  # not a fact the bridge receives
    {"targets": {"eval": {"kind": "dense_depth", "file": "e.npy", "sha256": "e" * 64}}},
])
def test_scoring_edits_leave_the_input_hash_alone(over):
    """A metric pin, a readout version or a scoring-only protocol value never invalidates features."""
    i0, s0 = _sides()
    i1, s1 = _sides(**over)
    assert i1 == i0 and s1 != s0


@pytest.mark.parametrize("over", [
    {"manifest_hash": "0" * 64},
    {"streams": {"video": dict(BENCH_SPEC["streams"]["video"], layout="tar_index")}},
    {"input_contract": dict(BENCH_SPEC["input_contract"], facts=["clip_seconds"])},  # clip_seconds now reaches the bridge
    {"output_contract": dict(BENCH_SPEC["output_contract"], requires={"pos": ["t"], "streams": [], "grid": None, "t_bins": None})},
    {"protocol": dict(BENCH_SPEC["protocol"], views="multi")},
    {"readout_params": {"visual_stream": "video"}},
])
def test_input_edits_change_the_input_hash(over):
    i0, _ = _sides()
    i1, _ = _sides(**over)
    assert i1 != i0


@pytest.mark.parametrize(
    "edit",
    [
        {"key": "ytid"},
        {"label": dict(MANIFEST_BLOCK["label"], column="labels")},
        {"label": dict(MANIFEST_BLOCK["label"], kind="multi")},
        {"label": dict(MANIFEST_BLOCK["label"], sep=",")},
        {"label": dict(MANIFEST_BLOCK["label"], class_key="index")},
        {"label": dict(MANIFEST_BLOCK["label"], sets={"all": ["labels_a"]})},
        {"label": dict(MANIFEST_BLOCK["label"], train_set="all")},
        {"label": dict(MANIFEST_BLOCK["label"], allow_unknown=True)},
        {"label": dict(MANIFEST_BLOCK["label"], empty_ok=True)},
        {"columns": dict(MANIFEST_BLOCK["columns"], extra={"type": "int", "role": "label_aux"})},
        {"columns": {"rel_media": {"type": "str", "role": "media", "stream": ["video", "audio"]}}},
    ],
)
def test_protocol_hash_covers_manifest_read_rules(edit):
    """manifest.key / label / columns decide the targets, hence the numbers. Key and
    columns are input side (which rows and media a Sample carries), the label read rule is scoring side."""
    base = hashing.protocol_hashes(**_protocol_kwargs())
    hs = hashing.protocol_hashes(**_protocol_kwargs(manifest=dict(MANIFEST_BLOCK, **edit)))
    assert hs.protocol_hash != base.protocol_hash
    assert (hs.input_hash != base.input_hash) == ("label" not in edit)


@pytest.mark.parametrize(
    "edit",
    [
        {"file": "other.csv.gz"},
        {"sha256": "0" * 64},
        {"classes": "c2.csv"},
        {"classes_sha256": "1" * 64},
        {"upstream": {"url": "https://elsewhere.example/y"}},
    ],
)
def test_protocol_hash_ignores_manifest_provenance(edit):
    """``file``/``sha256``/``classes``/``classes_sha256`` are covered by ``manifest_hash``; ``upstream`` is provenance."""
    base = hashing.protocol_hash(**_protocol_kwargs())
    assert hashing.protocol_hash(**_protocol_kwargs(manifest=dict(MANIFEST_BLOCK, **edit))) == base


def test_protocol_manifest_view():
    assert hashing.PROTOCOL_MANIFEST_KEYS == ("key", "label", "columns")
    assert hashing.protocol_manifest(MANIFEST_BLOCK) == {"key": "clip_id", "label": MANIFEST_BLOCK["label"], "columns": MANIFEST_BLOCK["columns"]}
    # a key absent from the yaml is null: as written, not as defaulted (vggss_box's label block has no `sets`)
    assert hashing.protocol_manifest({"key": "clip_id"}) == {"key": "clip_id", "label": None, "columns": None}
    with pytest.raises(TypeError):
        hashing.protocol_manifest(None)  # type: ignore[arg-type]


def test_protocol_targets_normalization():
    targets = {
        "eval": {"kind": "dense_depth", "source": "derived_file", "root": "derived_root", "file": "eval_depth_224.npy", "sha256": "e" * 64, "index_column": "labeled_index"},
        "train": {"kind": "dense_depth", "source": "derived_file", "root": "derived_root", "file": "train_depth_224.npy", "sha256": "t" * 64, "index_column": None},
    }
    view = hashing.protocol_targets(targets)
    assert view == [
        {"split": "eval", "kind": "dense_depth", "file": "eval_depth_224.npy", "sha256": "e" * 64, "index_column": "labeled_index"},
        {"split": "train", "kind": "dense_depth", "file": "train_depth_224.npy", "sha256": "t" * 64, "index_column": None},
    ]
    assert hashing.protocol_targets(None) is None
    h_dict = hashing.protocol_hash(**_protocol_kwargs(targets=targets, readout="grid"))
    h_list = hashing.protocol_hash(**_protocol_kwargs(targets=view, readout="grid"))
    assert h_dict == h_list
    # source / root are site layout: never hashed
    t2 = {k: dict(v, root="elsewhere", source="other") for k, v in targets.items()}
    assert hashing.protocol_hash(**_protocol_kwargs(targets=t2, readout="grid")) == h_dict
    # index_column decides which memmap row a manifest row is scored against → hashed (absent == null)
    t3 = {"eval": dict(targets["eval"], index_column="eval_row"), "train": targets["train"]}
    assert hashing.protocol_hash(**_protocol_kwargs(targets=t3, readout="grid")) != h_dict
    t4 = {"eval": targets["eval"], "train": {k: v for k, v in targets["train"].items() if k != "index_column"}}
    assert hashing.protocol_hash(**_protocol_kwargs(targets=t4, readout="grid")) == h_dict


def _packaged():
    from yavih.benchmark import packaged_benchmarks

    return packaged_benchmarks()


@pytest.mark.parametrize("name", _packaged())
def test_packaged_protocol_hashes_are_recorded_in_the_folder(name):
    """The expected hashes live in each folder's benchmark.yaml (newest changelog entry), not
    here. A protocol edit without a changelog record fails with the exact lines to write (hash_history.check_recorded,
    also ``yavih bench lint``)."""
    from yavih.benchmark import Benchmark
    from yavih.hash_history import check_recorded

    b = Benchmark.load(name)
    problems = check_recorded(b)
    assert not problems, f"{name}: " + "\n".join(problems)
    rec = b.spec.changelog[-1]
    # recomputed from the raw yaml blocks with the documented payload, independently of Benchmark.load's call
    raw = b.spec.raw
    again = hashing.protocol_hashes(
        name=b.spec.name, protocol_version=b.spec.protocol_version, protocol=raw["protocol"], splits=raw["splits"], streams=raw["streams"],
        input_contract=raw["input_contract"], output_contract=raw["output_contract"], external=raw["external"], metrics=raw["metrics"],
        readout_params=raw["readout_params"], targets=raw["targets"], manifest=raw["manifest"], manifest_hash=b.manifest_hash,
        readout=b.spec.readout,
    )
    assert (again.protocol_hash, again.input_hash) == (rec.protocol_hash, rec.input_hash) == (b.protocol_hash, b.input_hash)
    assert b.folder_source == {"probe": b.probe_sha256, "reader": None}  # the packaged folders ship no reader.py
    assert not hashing.find_prose(b.hashes.input_values) and not hashing.find_prose(b.hashes.scoring_values)


def test_check_recorded_tells_the_author_what_to_write():
    from types import SimpleNamespace

    from yavih.hash_history import check_recorded

    entry = SimpleNamespace(version=2, protocol_hash=None, input_hash=None, supersedes=(), supersedes_input=())
    bench = SimpleNamespace(spec=SimpleNamespace(changelog=[entry]), protocol_hash="p" * 64, input_hash="i" * 64)
    (msg,) = check_recorded(bench)
    assert "protocol_hash: " + "p" * 64 in msg and "input_hash: " + "i" * 64 in msg
    entry = SimpleNamespace(version=2, protocol_hash="o" * 64, input_hash="i" * 64, supersedes=(), supersedes_input=())
    bench = SimpleNamespace(spec=SimpleNamespace(changelog=[entry]), protocol_hash="p" * 64, input_hash="i" * 64)
    (msg,) = check_recorded(bench)
    assert f"`protocol_hash: {'p' * 64}`" in msg and f"append `{'o' * 64}`" in msg and "supersedes" in msg


# ----------------------------------------------------------------------------- describe_sha256 (consistency, not identity)

DESCRIBE = {
    "model": "vjepa2-l",
    "reading": "v",
    "weights_sha256": "a" * 64,
    "weights_ref": "/path/to/weights.pt",
    "dtype": "float16",
    "batch_size": 32,
    "streams": ["video"],
    "pos_mode": "constant",
    "layer": 12,
    "pooling": "none",
    "input": {"frames": 16, "resize": 224},
    "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
    "args": {"layer": 12},
}


def test_describe_sha256_excludes_provenance():
    # yavih-describe-2: batch size and decoder builds are provenance
    assert hashing.DESCRIBE_SHA_EXCLUDED == frozenset({"weights_ref", "batch_tolerance", "batch_size"})
    assert hashing.DESCRIBE_PROVENANCE_INPUT_KEYS == frozenset({"decoder"})
    sub = hashing.describe_subset(DESCRIBE)
    assert set(sub) == set(DESCRIBE) - {"weights_ref", "batch_size"}
    assert hashing.describe_sha256(DESCRIBE) == hashing.sha256_canon(sub)
    assert hashing.describe_sha256(dict(DESCRIBE, batch_size=8)) == hashing.describe_sha256(DESCRIBE)
    dec = {"video": {"temporal": {"rule": "uniform"}, "decoder": {"lib": "pyav", "version": "12.0.0", "libav": "60.31.102"}}}
    dec2 = {"video": {"temporal": {"rule": "uniform"}, "decoder": {"lib": "pyav", "version": "12.3.0", "libav": "61.3.100"}}}
    assert hashing.describe_sha256(dict(DESCRIBE, input=dec)) == hashing.describe_sha256(dict(DESCRIBE, input=dec2))
    assert hashing.describe_sha256(dict(DESCRIBE, input=dec)) != hashing.describe_sha256(
        dict(DESCRIBE, input={"video": dict(dec["video"], temporal={"rule": "all"})}))
    # a declared check threshold or a moved checkpoint path is not a computation: same configuration
    assert hashing.describe_sha256(dict(DESCRIBE, batch_tolerance={"rel_l2": 1e-4, "reason": "bf16 kernels"})) == hashing.describe_sha256(DESCRIBE)
    assert hashing.describe_sha256(dict(DESCRIBE, weights_ref="/another/path.pt")) == hashing.describe_sha256(DESCRIBE)


_BRIDGE_EDITS = [
    {"args": {"layer": 11}, "layer": 11},
    {"input": {"frames": 8, "resize": 224}},
    {"precision": dict(DESCRIBE["precision"], allow_tf32=True)},
    {"dtype": "float32"},
    {"weights_sha256": "b" * 64},
    {"weights_sha256": None},
    {"pooling": "mean"},
]


@pytest.mark.parametrize("edit", _BRIDGE_EDITS)
def test_bridge_decisions_change_describe_sha256_and_nothing_comparable(edit):
    """Any describe() key changes describe_sha256 (consistency) and nothing of protocol_hash,
    probe_hash, eval_hash or the unit id — none of them takes a bridge input at all."""
    import inspect

    changed = dict(DESCRIBE, **edit)
    assert hashing.describe_sha256(changed) != hashing.describe_sha256(DESCRIBE)
    ph = hashing.protocol_hash(**_protocol_kwargs())
    qh = hashing.probe_hash(_cfg())
    uid = dict(kind="probe", benchmark="k400", model=DESCRIBE["model"], reading=DESCRIBE["reading"], split=None, shard=None,
               protocol_hash=ph, probe_hash=qh, seed=0)
    assert hashing.unit_id(**dict(uid, model=changed["model"], reading=changed["reading"])) == hashing.unit_id(**uid)
    for fn in (hashing.protocol_hashes, hashing.input_components, hashing.scoring_components, hashing.probe_hash, hashing.eval_hash,
               hashing.unit_id):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"describe", "describe_sha256", "weights_sha256", "dtype"}, fn.__name__


# ----------------------------------------------------------------------------- eval_hash


def test_eval_hash_formula():
    ph, qh = "a" * 64, "b" * 64
    payload = {"protocol_hash": ph, "probe_hash": qh}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert hashing.eval_hash(ph, qh) == expected
    assert hashing.eval_hash(ph, qh) == hashing.sha256_canon({"protocol_hash": ph, "probe_hash": qh})
    assert hashing.eval_hash(qh, ph) != expected  # the components are named: swapping them is another key
    assert hashing.eval_hash("0" * 64, qh) != expected and hashing.eval_hash(ph, "0" * 64) != expected
    with pytest.raises(TypeError):
        hashing.eval_hash(ph, None)  # type: ignore[arg-type]


def test_eval_hash_of_k400_follows_its_recorded_protocol_hash():
    """k400 under its pinned clf probe: the eval hash computed from the loaded folder equals the one formed from the
    protocol hash its newest changelog entry records, and is stable across loads."""
    pytest.importorskip("torch")
    from yavih.benchmark import Benchmark
    from yavih.probes.config import ProbeConfig

    b = Benchmark.load("k400")
    qh = hashing.probe_hash(ProbeConfig.pinned(b.spec.readout, bench=b))
    eh = hashing.eval_hash(b.protocol_hash, qh)
    assert eh == hashing.eval_hash(b.spec.changelog[-1].protocol_hash, qh)
    b2 = Benchmark.load("k400")
    assert hashing.eval_hash(b2.protocol_hash, hashing.probe_hash(ProbeConfig.pinned(b2.spec.readout, bench=b2))) == eh
    assert len(hashing.hash12(eh)) == 12


# ----------------------------------------------------------------------------- probe_hash


@dataclass(frozen=True, kw_only=True)
class _Cfg:
    readout: str = "clf"
    head: dict = None  # type: ignore[assignment]
    lr: float = 1e-4
    lr_grid: tuple = ()
    weight_decay: float = 1e-4
    max_epochs: int = 30
    batch_size: int = 256
    patience: int = 5
    warmup_fraction: float = 0.0
    schedule: str = "cosine"
    val_fraction: float = 0.1
    loss: str = "ce"
    selection: str = "val_loss"
    max_tokens_per_forward: int = 131072
    precision: str = "fp32"
    seed: int = 0  # NOT hashed


def _cfg(**kw):
    head = {"family": "attentive_probe_v1", "self_attn_depth": 3, "cross_attn_depth": 1, "num_heads": 16, "mlp_ratio": 4, "memory_pos": "none", "layout": "point", "out": 3}
    return _Cfg(head=head, **kw)


def test_probe_hash_payload_and_seed_independence():
    cfg = _cfg()
    payload = {
        "schema": PROBE_SCHEMA,
        "readout": "clf",
        "head": cfg.head,
        "train": {
            "lr": 1e-4, "lr_grid": [], "weight_decay": 1e-4, "max_epochs": 30, "batch_size": 256, "patience": 5,
            "warmup_fraction": 0.0, "schedule": "cosine", "val_fraction": 0.1, "loss": "ce", "selection": "val_loss",
            "max_tokens_per_forward": 131072, "precision": "fp32",
        },
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert hashing.probe_hash(cfg) == expected
    assert hashing.probe_hash(_cfg(seed=7)) == expected  # seed is not included
    assert hashing.probe_hash(_cfg(lr=3e-4)) != expected
    assert hashing.probe_hash(_cfg(lr_grid=(1e-4, 3e-4))) != expected


def test_probe_hash_accepts_dict_and_rejects_incomplete():
    cfg = _cfg()
    as_dict = {f: getattr(cfg, f) for f in ("readout", "head", *hashing.PROBE_TRAIN_FIELDS)}
    assert hashing.probe_hash(as_dict) == hashing.probe_hash(cfg)
    with pytest.raises(KeyError):
        hashing.probe_hash({"readout": "clf", "head": {}})

    @dataclass(frozen=True)
    class Partial:
        readout: str = "clf"
        head: dict = None  # type: ignore[assignment]

    with pytest.raises(KeyError):
        hashing.probe_hash(Partial())


def test_probe_hash_works_with_real_probeconfig():
    from yavih.probes.config import ProbeConfig

    cfg = ProbeConfig(readout="clf", head={"family": "attentive_probe_v1"}, lr=1e-4, lr_grid=(), weight_decay=1e-4,
                      max_epochs=30, batch_size=256, patience=5, warmup_fraction=0.0, loss="ce")
    assert hashing.probe_hash(cfg) == hashing.probe_hash(_Cfg(head={"family": "attentive_probe_v1"}))


# ----------------------------------------------------------------------------- features_fingerprint


def test_features_fingerprint():
    shards = [[0, 608, "a" * 64], [608, 1216, "b" * 64]]
    fp = hashing.features_fingerprint(format_version=1, dtype="float16", N=1568, D=1024, pos_mode="constant",
                                      pos_sha256="c" * 64, shards=shards, n_present=1200)
    payload = {"format_version": 1, "dtype": "float16", "N": 1568, "D": 1024, "pos_mode": "constant",
               "pos_sha256": "c" * 64, "shards": shards, "n_present": 1200}
    assert fp == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    reordered = hashing.features_fingerprint(format_version=1, dtype="float16", N=1568, D=1024, pos_mode="constant",
                                             pos_sha256="c" * 64, shards=shards[::-1], n_present=1200)
    assert reordered != fp  # row order is part of the identity
    with pytest.raises(ValueError):
        hashing.features_fingerprint(format_version=1, dtype="float16", N=1, D=1, pos_mode="none", pos_sha256=None,
                                     shards=[[0, 1]], n_present=1)


# ----------------------------------------------------------------------------- unit id / inputs_hash


def test_unit_id_is_12_hex_of_the_documented_payload():
    kw = dict(kind="extract", benchmark="k400", model="vjepa2-l", reading="v", split="train", shard=(3, 32),
              protocol_hash=None, input_hash="i" * 64, probe_hash=None, seed=None)
    uid = hashing.unit_id(**kw)
    payload = {"kind": "extract", "benchmark": "k400", "model": "vjepa2-l", "reading": "v", "split": "train",
               "shard": [3, 32], "protocol_hash": None, "input_hash": "i" * 64, "probe_hash": None, "seed": None,
               "plan_schema": PLAN_SCHEMA_VERSION}
    assert PLAN_SCHEMA_VERSION == 3  # extract units are keyed by input_hash
    assert uid == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    assert len(uid) == 12 and all(c in "0123456789abcdef" for c in uid)
    assert hashing.unit_id(**dict(kw, shard=(4, 32))) != uid
    assert hashing.unit_id(**dict(kw, shard=[3, 32])) == uid  # list or tuple alike
    # two readings of one model are two units (runs/<b>/<model>/<reading>/ must not collide)
    assert hashing.unit_id(**dict(kw, reading="a")) != uid
    assert hashing.unit_id(**dict(kw, model="vjepa2-g")) != uid


def test_inputs_hash_changes_with_cmd_not_with_where_it_runs():
    kw = dict(cmd=["python", "-m", "yavih.extract"], protocol_hash=None, input_hash="i" * 64, probe_hash=None, model="vjepa2-l",
              reading="v", describe_sha256="d" * 64)
    h = hashing.inputs_hash(**kw)
    assert len(h) == 64
    assert h == hashlib.sha256(json.dumps(kw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert hashing.inputs_hash(**dict(kw, cmd=kw["cmd"] + ["--canaries", "8"])) != h
    assert hashing.inputs_hash(**dict(kw, cmd=tuple(kw["cmd"]))) == h
    # the bridge's configuration is staleness (a changed describe_sha256 re-runs the unit), not comparability
    assert hashing.inputs_hash(**dict(kw, describe_sha256="e" * 64)) != h
    assert hashing.inputs_hash(**dict(kw, reading="a")) != h
    # launcher, env, cwd and needs are logged, never hashed
    for extra in ({"launcher": "/envs/x/bin/python"}, {"env": {"A": "1"}}, {"cwd": "."}, {"needs": {"minutes": 15}}):
        with pytest.raises(TypeError):
            hashing.inputs_hash(**kw, **extra)
