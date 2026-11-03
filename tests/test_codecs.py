"""``yavih.codecs``: the output codecs, the schema grammar, symbol binding and the self-test (docs/codecs.md)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from yavih import codecs as C

SPECS = [
    ("array", {"type": "array", "dtype": ["float16", "float32"], "shape": ["N", "D"]}),
    ("array", {"type": "array", "dtype": "float32", "shape": ["*"]}),
    ("array", {"type": "array", "dtype": "int64", "shape": [3]}),
    ("json", {"type": "json", "kind": "str"}),
    ("json", {"type": "json", "kind": "object"}),
    ("bytes", {"type": "bytes", "container": "ply"}),
    ("wav", {"type": "wav", "channels": 1, "duration_s": [0.5, 2.0]}),
]


@pytest.mark.parametrize("name, spec", SPECS)
def test_every_builtin_passes_its_self_test(name, spec):
    assert C.self_test(C.BUILTIN[name], spec) == []
    assert C.check_schema({"out": spec}, C.registry()) == []


def test_default_schema_is_the_feature_matrix():
    s = C.schema_of({"shape": "fixed"})
    assert s == C.DEFAULT_SCHEMA and C.is_default_schema(s) and C.has_default_feature_keys(s)
    assert C.check_schema(s, C.registry()) == []
    extended = {**C.DEFAULT_SCHEMA, "text": {"type": "json"}}
    assert not C.is_default_schema(extended) and C.has_default_feature_keys(extended)
    assert not C.has_default_feature_keys({"audio": {"type": "wav"}})
    assert C.schema_sha256(None) == C.schema_sha256(C.DEFAULT_SCHEMA) != C.schema_sha256(extended)


def test_bind_shape_rules():
    bound: dict[str, int] = {}
    assert C.bind_shape(["N", "D"], (4, 8), bound, where="feat") is None and bound == {"N": 4, "D": 8}
    assert C.bind_shape(["N", 4], (4, 4), bound, where="pos") is None  # N shared across keys
    assert "bound to 4" in C.bind_shape(["N", "D"], (5, 8), bound, where="feat")
    assert "declares 4" in C.bind_shape(["N", 4], (4, 3), bound, where="pos")
    assert C.bind_shape(["*"], (17,), bound, where="x") is None and C.bind_shape(["*"], (3,), bound, where="x") is None
    assert "dims" in C.bind_shape(["N", "D"], (4,), bound, where="feat")


def test_array_check_findings():
    spec = {"type": "array", "dtype": "float16", "shape": ["N"]}
    assert [f.code for f in C.ARRAY.check(np.array([np.nan], np.float16), spec, {})] == ["E_FINITE"]
    assert [f.code for f in C.ARRAY.check(np.array([2e4], np.float16), spec, {})] == ["E_FEATURES"]
    assert [f.code for f in C.ARRAY.check(np.array([2e3], np.float16), spec, {})] == ["W_FP16_RANGE"]
    assert [f.code for f in C.ARRAY.check(np.array([1.0], np.float64), spec, {})] == ["E_MEMBERS"]  # dtype
    assert C.ARRAY.check(np.array([1.0], np.float16), spec, {}) == []


def test_array_distance_ignores_one_fp16_ulp_but_not_real_drift():
    a = np.linspace(-1, 1, 64).astype(np.float16)
    b = a.copy()
    b[3] = np.nextafter(b[3], np.float16(2))
    assert C.ARRAY.distance(a, a, {}) == 0.0 and C.ARRAY.distance(b, a, {}) == 0.0
    assert C.ARRAY.distance((a.astype(np.float32) * 1.1).astype(np.float16), a, {}) > 1e-3
    assert C.ARRAY.distance(a[:3], a, {}) == float("inf")


def test_array_stats_find_dead_rows_duplicates_and_constant_clips():
    spec = {"type": "array", "dtype": "float32", "shape": ["N", "D"]}
    st = C.ArrayStats(spec, {})
    rng = np.random.default_rng(0)
    for row in range(8):
        x = rng.standard_normal((3, 4)).astype(np.float32)
        x[1] = 0.5  # token 1 never varies
        st.add(row, f"c{row}", None, x, sampled=True)
    same = np.ones((3, 4), np.float32)
    same[1] = 0.5
    st.add(8, "c8", "k8", same, sampled=True)
    st.add(9, "c9", "k9", same, sampled=True)
    st.add(10, "c10", "k10", np.zeros((3, 4), np.float32), sampled=False)  # an all-zero clip outside the sample
    codes = {f.code for f in st.findings(key="feat")}
    assert st.dead_rows() == [1] and st.duplicate_rows() == [8, 9]
    assert codes == {"E_DEAD_ROWS", "E_DUPLICATE_CLIPS", "W_CONSTANT_CLIP"}


def test_parallel_merge_reproduces_the_serial_pass():
    spec = {"type": "array", "dtype": "float32", "shape": ["N", "D"]}
    rng = np.random.default_rng(1)
    rows = [rng.standard_normal((2, 3)).astype(np.float32) for _ in range(6)]
    serial = C.ArrayStats(spec, {})
    for i, x in enumerate(rows):
        serial.add(i, f"c{i}", None, x, sampled=True)
    merged = C.ArrayStats(spec, {})
    for part_rows in (range(0, 3), range(3, 6)):
        part = C.ArrayStats(spec, {}, deferred=True)
        for i in part_rows:
            part.add(i, f"c{i}", None, rows[i], sampled=True)
        merged.merge(part)
    assert np.array_equal(serial.sum, merged.sum) and np.array_equal(serial.sumsq, merged.sumsq)
    assert serial.dup_hash == merged.dup_hash and serial.bytes == merged.bytes


def test_json_and_bytes_are_exact():
    assert C.JSON.distance({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1}, {}) == 0.0
    assert C.JSON.distance("x", "y", {}) == 1.0 and C.JSON.exact and C.BYTES.exact
    assert C.JSON.decode(C.JSON.encode({"t": np.float32(0.5)}, {}), {}) == {"t": 0.5}
    assert [f.code for f in C.JSON.check(3, {"kind": "str"}, {})] == ["E_MEMBERS"]
    assert C.BYTES.ext_for({"container": "ply"}) == "ply" and C.ARRAY.ext_for({}) == "npy"


def test_wav_constraints_and_decoded_distance():
    sr = 16000
    x = (np.sin(np.linspace(0, 200, sr)) * 0.3).astype(np.float32)
    a = C.wav_bytes(x, sr)
    spec = {"type": "wav", "channels": 1, "sample_rate": 16000, "duration_s": [0.9, 1.1]}
    assert C.WAV.check(a, spec, {}) == []
    assert [f.code for f in C.WAV.check(a, {**spec, "channels": 2}, {})] == ["E_MEMBERS"]
    assert "duration" in C.WAV.check(a, {**spec, "duration_s": [5, 6]}, {})[0].message
    assert C.WAV.check(b"not a wav", spec, {})[0].message == "not a wav file"
    louder = C.wav_bytes(x * 1.5, sr)
    assert C.WAV.distance(a, a, spec) == 0.0 and C.WAV.distance(louder, a, spec) > 0.1
    assert C.WAV.distance(C.wav_bytes(x, 8000), a, spec) == float("inf")


def test_schema_grammar_refusals():
    reg = C.registry()
    probs = C.check_schema({"json": {"type": "json"}, "Bad": {"type": "array"}, "x": {"type": "nope"},
                            "y": {"type": "array", "shape": ["n"]}, "z": {"type": "json", "stats": True},
                            "w": {"type": "bytes"}, "v": {"type": "array", "shape": [1], "constant": "always"}}, reg)
    text = " | ".join(probs)
    for needle in ("schema.json: key", "schema.Bad: key", "unknown codec 'nope'", "shape entry 'n'", "schema.z.stats",
                   "schema.w: container", "schema.v.constant"):
        assert needle in text, needle
    assert C.check_schema({}, reg) == ["schema: a non-empty mapping {key: {type: <codec>, ...}}"]
    with pytest.raises(ValueError, match="unknown codec"):
        C.resolve_schema({"x": {"type": "nope"}}, reg)


def _toy_codec(name="pointcloud"):
    return C.Codec(name=name, version=2, ext="ply", encode=C.BYTES.encode, decode=C.BYTES.decode, check=C.BYTES.check,
                   distance=C.BYTES.distance, exact=True, synthetic=lambda spec, bound, rng: rng.bytes(32))


def test_folder_and_entry_point_codecs(monkeypatch):
    mod = types.SimpleNamespace(CODECS={"pointcloud": _toy_codec()})
    folder = C.folder_codecs(mod)
    assert set(C.registry(folder)) == {"array", "json", "bytes", "wav", "pointcloud"}
    assert C.self_test(folder["pointcloud"], {"type": "pointcloud"}) == []
    with pytest.raises(ValueError, match="redefines a built-in"):
        C.folder_codecs(types.SimpleNamespace(CODECS={"array": _toy_codec("array")}))
    with pytest.raises(ValueError, match="named"):
        C.folder_codecs(types.SimpleNamespace(CODECS={"x": _toy_codec("y")}))

    plugin = types.ModuleType("toy_codec_mod")
    plugin.CODECS = {"mesh": _toy_codec("mesh")}
    monkeypatch.setitem(sys.modules, "toy_codec_mod", plugin)

    class EP:
        name, value = "mesh", "toy_codec_mod"

    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda group=None: [EP()] if group == C.ENTRY_POINT_GROUP else [])
    assert "mesh" in C.registry()


def test_a_codec_whose_distance_is_blind_fails_the_self_test():
    blind = C.Codec(name="blind", version=1, ext="bin", encode=C.BYTES.encode, decode=C.BYTES.decode, check=C.BYTES.check,
                    distance=lambda a, b, spec: 0.0, synthetic=lambda spec, bound, rng: rng.bytes(8))
    assert any("could not see a change" in p for p in C.self_test(blind, {"type": "blind"}))
    lossy = C.Codec(name="lossy", version=1, ext="bin", encode=lambda v, s: bytes(v)[:4], decode=C.BYTES.decode, check=C.BYTES.check,
                    distance=C.BYTES.distance, synthetic=lambda spec, bound, rng: rng.bytes(8))
    assert any("round trip" in p for p in C.self_test(lossy, {"type": "lossy"}))


def test_codec_descriptor_refusals():
    with pytest.raises(ValueError, match="name"):
        _toy_codec("Bad")
    with pytest.raises(ValueError, match="version"):
        C.Codec(name="x", version=0, ext="bin", encode=C.BYTES.encode, decode=C.BYTES.decode, check=C.BYTES.check, distance=C.BYTES.distance)


# ----------------------------------------------------------------------------- the schema in benchmark.yaml


def _folder(tmp_path, mutate=None, eval_py: str | None = None):
    import yaml

    from yavih import testing as T
    from yavih.benchmark import Benchmark

    folder = T.make_synthetic_benchmark(tmp_path, "clf", n_train=8, n_eval=4, media="none")
    if mutate is not None:
        y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
        mutate(y)
        (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    if eval_py is not None:
        (folder / "eval.py").write_text(eval_py, encoding="utf-8")
    return Benchmark.from_dir(folder)


def test_a_declared_default_schema_hashes_like_none(tmp_path):
    base = _folder(tmp_path / "a")
    same = _folder(tmp_path / "b", lambda y: y["output_contract"].update(schema=C.DEFAULT_SCHEMA))
    assert same.spec.output_contract.schema == C.DEFAULT_SCHEMA
    assert (same.input_hash, same.protocol_hash) == (base.input_hash, base.protocol_hash)
    assert set(base.output_schema()) == {"feat", "pos"} and base.spec.output_contract.sampling == "optional"


def _generative(y):
    y["output_contract"].update(schema={"audio": {"type": "wav", "channels": 1, "duration_s": [0.5, 2.0]}}, sampling="required")


def test_a_generation_schema_moves_the_input_hash_and_records_codec_versions(tmp_path):
    base = _folder(tmp_path / "a")
    gen = _folder(tmp_path / "b", _generative)
    assert gen.input_hash != base.input_hash and gen.scoring_hash == base.scoring_hash
    view = gen.hashes.input_values["output_contract"]
    assert view["codecs"] == {"wav": 1} and view["schema"]["audio"]["type"] == "wav" and view["sampling"] == "required"
    assert gen.spec.output_contract.sampling == "required"
    assert list(gen.output_schema()) == ["audio"] and gen.output_schema()["audio"][1] is C.WAV


def test_schema_errors_are_load_errors(tmp_path):
    from yavih.benchmark import SchemaError

    with pytest.raises(SchemaError, match="unknown codec 'nope'"):
        _folder(tmp_path / "a", lambda y: y["output_contract"].update(schema={"x": {"type": "nope"}}))
    with pytest.raises(SchemaError, match="sampling"):
        _folder(tmp_path / "b", lambda y: y["output_contract"].update(sampling="always"))

    def needs_pos(y):
        _generative(y)
        y["output_contract"]["requires"]["pos"] = ["t"]

    with pytest.raises(SchemaError, match="default feature keys"):
        _folder(tmp_path / "c", needs_pos)


TOY_EVAL = '''
from yavih import codecs as C

def _syn(spec, bound, rng):
    return rng.bytes(16)

CODECS = {"pointcloud": C.Codec(name="pointcloud", version=3, ext="ply", encode=C.BYTES.encode, decode=C.BYTES.decode,
                                check=C.BYTES.check, distance=C.BYTES.distance, exact=True, synthetic=_syn)}
'''

BLIND_EVAL = TOY_EVAL.replace("distance=C.BYTES.distance", "distance=lambda a, b, s: 0.0")


def test_a_folder_codec_is_hashed_by_version_and_self_tested_by_lint(tmp_path):
    from yavih.lint import lint_folder

    def cloud(y):
        y["output_contract"]["schema"] = {"cloud": {"type": "pointcloud"}}

    b = _folder(tmp_path / "a", cloud, TOY_EVAL)
    assert b.hashes.input_values["output_contract"]["codecs"] == {"pointcloud": 3}
    rep = lint_folder(b.folder, checks=["codecs"])
    assert not [f for f in rep.findings if f.code == "E_CODEC_SELFTEST"], rep.findings
    blind = _folder(tmp_path / "b", cloud, BLIND_EVAL)
    codes = [f.code for f in lint_folder(blind.folder, checks=["codecs"]).findings]
    assert "E_CODEC_SELFTEST" in codes
