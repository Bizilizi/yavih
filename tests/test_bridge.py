"""``Bridge`` base, ``BenchmarkView``, ``parse_args``, ``validate_describe`` (incl. the declared-rule
vocabulary), ``MediaUndecodable``, ``fingerprint_state_dict``, ``grid_pos``, the preprocessing-source slot."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util

import numpy as np
import pytest

from test_extract_fixtures import (
    AUDIO_INPUT,
    VIDEO_INPUT,
    AVBridge,
    BadPrecisionBridge,
    FakeBenchmark,
    NoPreprocBridge,
    NumpyBridge,
    describe_of,
    view_of,
)
from yavih import hashing
from yavih.bridge import (
    DESCRIBE_REQUIRED,
    DESCRIBE_SHA_EXCLUDED,
    E_ARGS_UNCONSUMED,
    E_DESCRIBE,
    E_INPUT,
    E_PRECISION_UNDECLARED,
    TEMPORAL_RULES,
    ArgError,
    ArgSpec,
    BenchmarkView,
    Bridge,
    DescribeError,
    MediaUndecodable,
    NotApplicable,
    batch_tolerance,
    bridge_label,
    check_model_name,
    check_reading,
    fingerprint_state_dict,
    grid_pos,
    parse_args,
    preprocessing_source,
    stream_kinds,
    validate_describe,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None

# ----------------------------------------------------------------------------- parse_args / ARGS


def test_parse_args_defaults_and_coercion():
    out = parse_args(NumpyBridge.ARGS, ["layer=7", "scale=2.5", "flag=true", "mode=b"])
    assert out == {"layer": 7, "scale": 2.5, "flag": True, "mode": "b"}
    assert parse_args(NumpyBridge.ARGS, None) == {"layer": 12, "scale": 1.0, "flag": False, "mode": "a"}
    # passing a default explicitly changes nothing
    assert parse_args(NumpyBridge.ARGS, ["layer=12"]) == parse_args(NumpyBridge.ARGS, [])
    assert parse_args(NumpyBridge.ARGS, {"layer": "3"})["layer"] == 3
    assert parse_args(NumpyBridge.ARGS, {"layer": 3, "flag": True})["flag"] is True


def test_parse_args_unknown_key_is_usage_error_exit_1():
    with pytest.raises(ArgError) as ei:
        parse_args(NumpyBridge.ARGS, ["nope=1"])
    assert ei.value.exit_code == 1 and "nope" in str(ei.value)


@pytest.mark.parametrize("bad", [["layer=abc"], ["flag=maybe"], ["mode=z"], ["layer"], ["=1"], ["layer=1", "layer=2"]])
def test_parse_args_malformed(bad):
    with pytest.raises(ArgError):
        parse_args(NumpyBridge.ARGS, bad)


def test_parse_args_accepts_dict_specs():
    spec = {"layer": {"type": int, "default": 12}, "name": {"type": str}}
    assert parse_args(spec, ["layer=1"]) == {"layer": 1, "name": None}
    with pytest.raises(ArgError):
        parse_args({"x": {"typo": 1}}, [])


def test_bridge_init_takes_a_view_applies_args_and_rejects_unknown():
    view = view_of()
    b = NumpyBridge(view, {"layer": "4"})
    assert b.bench is view and b.args == {"layer": 4, "scale": 1.0, "flag": False, "mode": "a"}
    assert b.decode_workers == 1  # run_extract sets the CPU budget before the first encode
    with pytest.raises(ArgError):
        NumpyBridge(view, {"bogus": 1})


def test_base_bridge_requires_overrides():
    class Bare(Bridge):
        name = "bare"

    b = Bare(view_of(), {})
    with pytest.raises(NotImplementedError):
        b.describe()
    with pytest.raises(NotImplementedError):
        b.encode([])
    assert b.profile() is None


def test_not_applicable_and_media_undecodable_carry_their_payload():
    e = NotApplicable("no audio")
    assert e.reason == "no audio" and str(e) == "no audio"
    m = MediaUndecodable(["a", "b"], "moov atom not found")
    assert m.clip_ids == ["a", "b"] and m.reason == "moov atom not found" and "moov" in str(m)
    assert MediaUndecodable("c", "x").clip_ids == ["c"]


# ----------------------------------------------------------------------------- BenchmarkView


def test_benchmark_view_holds_primitives_only_and_no_manifest():
    bench = FakeBenchmark(streams={"video": "video", "audio": "audio"})
    bench.spec.input_contract["facts"] = ["clip_seconds"]
    v = BenchmarkView.from_benchmark(bench)
    assert v.name == "synth_fake" and v.protocol_hash == bench.protocol_hash and v.protocol_version == 2
    assert v.streams == {"video": "video", "audio": "audio"}
    assert v.facts == {"clip_seconds": 2.0}
    hashing.canon(dataclasses.asdict(v))  # JSON primitives only
    for leak in ("manifest", "reader", "targets", "splits", "data_status"):
        assert not hasattr(v, leak)
    assert BenchmarkView.from_benchmark(v) is v
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.name = "x"  # type: ignore[misc]


def test_benchmark_view_prefers_benchmark_view_method():
    bench = FakeBenchmark()
    own = BenchmarkView(name="own", aliases=[], readout="clf", protocol_version=2, protocol_hash="p", streams={"video": "video"})
    bench.view = lambda: own  # type: ignore[attr-defined]
    assert BenchmarkView.from_benchmark(bench) is own


def test_stream_kinds_of_view_benchmark_and_mapping():
    bench = FakeBenchmark(streams={"video": "video", "audio": "audio"})
    assert stream_kinds(bench) == {"video": "video", "audio": "audio"}
    assert stream_kinds(view_of(bench)) == {"video": "video", "audio": "audio"}
    assert stream_kinds({"streams": {"img": {"kind": "image"}}}) == {"img": "image"}


# ----------------------------------------------------------------------------- validate_describe


def test_validate_describe_accepts_a_complete_describe_against_view_and_benchmark():
    bench = FakeBenchmark()
    b = NumpyBridge(view_of(bench), {})
    d = b.describe()
    assert set(DESCRIBE_REQUIRED) <= set(d)
    assert "views" in DESCRIBE_REQUIRED
    validate_describe(d, b, b.bench)
    validate_describe(d, b, bench)
    assert DESCRIBE_REQUIRED[:2] == ("model", "reading")
    assert d["model"] == "np-test" and d["reading"] == "v"


@pytest.mark.parametrize("key", DESCRIBE_REQUIRED)
def test_validate_describe_refuses_missing_key(key):
    b = NumpyBridge(view_of(), {})
    d = b.describe()
    del d[key]
    with pytest.raises(DescribeError) as ei:
        validate_describe(d, b, b.bench)
    assert ei.value.exit_code == 2
    assert key in str(ei.value) or ei.value.code == E_INPUT  # removing input/streams can surface as an input-block error


@pytest.mark.parametrize("model", ["vjepa2-l", "beats_iter3", "AudioMAE.ft-as2m", "a", "x" * 64, "m.1", "0_", "m_stale", "x..."])
def test_model_name_charset_accepts(model):
    assert check_model_name(model) is None
    b = NumpyBridge(view_of(), {})
    validate_describe(dict(b.describe(), model=model), b, b.bench)


@pytest.mark.parametrize("model", ["", ".", "..", "has space", "a/b", "a:b", "x" * 65, "é", None, 3, "a+b", "a@b", "a~",
                                   "_stale", "_report", "_m", ".hidden", "...x", "-m"])
def test_model_name_charset_refuses(model):
    """``model`` names runs/<benchmark>/<model>/ and <results_root>/<model>/: filesystem-safe by construction, and its
    first character is a letter or digit (``_stale`` / ``_report`` are yavih's directories and invisible to
    the report index and the join; ``.x`` is hidden from ``*`` globs; ``-x`` reads as an option)."""
    assert check_model_name(model) is not None
    b = NumpyBridge(view_of(), {})
    with pytest.raises(DescribeError) as ei:
        validate_describe(dict(b.describe(), model=model), b, b.bench)
    assert ei.value.code == E_DESCRIBE and ei.value.exit_code == 2 and "model" in str(ei.value)


@pytest.mark.parametrize("reading", ["a", "v", "av", "video-only", "tap_12", "x" * 32])
def test_reading_charset_accepts(reading):
    assert check_reading(reading) is None


@pytest.mark.parametrize("reading", ["", "a+v", "a:v", "a.v", "a v", "x" * 33, None, 1, "_stale", "_a", "-a"])
def test_reading_charset_refuses(reading):
    """``reading`` enters the run path and the results key ``<benchmark>:<reading>``, so ``:``/``+``/``.`` stay free; a
    leading ``_`` (``_stale``: invisible to the report index) or ``-`` is refused."""
    assert check_reading(reading) is not None
    b = NumpyBridge(view_of(), {})
    with pytest.raises(DescribeError) as ei:
        validate_describe(dict(b.describe(), reading=reading), b, b.bench)
    assert ei.value.code == E_DESCRIBE and "reading" in str(ei.value)


def test_bridge_name_is_an_optional_label():
    """``Bridge.name`` is a recorded label (default the class name), never a key; ``model`` is the name."""

    class Unnamed(NumpyBridge):
        name = ""

    b = Unnamed(view_of(), {})
    validate_describe(b.describe(), b, b.bench)  # no name needed
    assert bridge_label(b) == "Unnamed" and bridge_label(Unnamed) == "Unnamed"
    assert bridge_label(NumpyBridge(view_of(), {})) == "np-test"


@pytest.mark.parametrize(
    "patch",
    [
        {"dtype": "float64"},
        {"dtype": "float16"},  # != Bridge.dtype
        {"batch_size": 8},
        {"batch_size": 0},
        {"streams": ["audio"]},
        {"streams": ["video", "video"]},
        {"pos_mode": "none"},
        {"reading": ""},
        {"model": ""},
        {"pooling": ""},
        {"layer": 1.5},
        {"layer": True},
        {"views": 3},
        {"views": 0},
        {"views": True},
        {"input": [1]},
        {"weights_sha256": "abc"},
        {"weights_sha256": "A" * 64},
        {"weights_ref": None},
        {"precision": {"autocast": "fp8", "matmul_precision": "highest", "allow_tf32": False, "deterministic": True}},
        {"precision": {"autocast": None, "matmul_precision": "fast", "allow_tf32": False, "deterministic": True}},
        {"precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": 1, "deterministic": True}},
        {"args": {"layer": 12}},  # does not echo every consumed arg
        {"batch_tolerance": {"rel_l2": 2e-3, "reason": "too loose"}},
        {"batch_tolerance": {"rel_l2": 1e-4, "reason": ""}},
        {"batch_tolerance": {"rel_l2": 1e-4}},
        {"batch_tolerance": {"rel_l2": 0.0, "reason": "zero"}},
    ],
)
def test_validate_describe_refuses_mismatch(patch):
    b = NumpyBridge(view_of(), {})
    d = dict(b.describe(), **patch)
    with pytest.raises(DescribeError):
        validate_describe(d, b, b.bench)


def test_validate_describe_accepts_a_declared_tolerance_and_reports_it():
    b = NumpyBridge(view_of(), {})
    d = dict(b.describe(), batch_tolerance={"rel_l2": 1e-3, "reason": "bf16 attention kernels differ between batch sizes"})
    validate_describe(d, b, b.bench)
    assert batch_tolerance(d) == (1e-3, "bf16 attention kernels differ between batch sizes")
    assert batch_tolerance(b.describe()) == (1e-5, None)


def test_validate_describe_args_unconsumed_code():
    b = NumpyBridge(view_of(), {"layer": 3})
    d = b.describe()
    d["args"] = dict(d["args"], layer=12)
    with pytest.raises(DescribeError) as ei:
        validate_describe(d, b, b.bench)
    assert ei.value.code == E_ARGS_UNCONSUMED
    d = b.describe()
    d["args"] = dict(d["args"], extra=1)
    with pytest.raises(DescribeError) as ei:
        validate_describe(d, b, b.bench)
    assert ei.value.code == E_ARGS_UNCONSUMED


def test_validate_describe_refuses_non_json_primitives():
    b = NumpyBridge(view_of(), {})
    d = b.describe()
    d["input"]["video"]["norm"]["mean"] = np.zeros(3)
    with pytest.raises(DescribeError):
        validate_describe(d, b, b.bench)


def test_validate_describe_null_weights_is_allowed_here():
    b = NumpyBridge(view_of(), {})
    validate_describe(dict(b.describe(), weights_sha256=None), b, b.bench)


def test_validate_describe_precision_undeclared():
    b = BadPrecisionBridge(view_of(), {})
    with pytest.raises(DescribeError) as ei:
        validate_describe(b.describe(), b, b.bench)
    assert ei.value.code == E_PRECISION_UNDECLARED
    b = NumpyBridge(view_of(), {})
    d = b.describe()
    del d["precision"]["deterministic"]
    with pytest.raises(DescribeError) as ei:
        validate_describe(d, b, b.bench)
    assert ei.value.code == E_PRECISION_UNDECLARED


def test_validate_describe_bad_bridge_name():
    class Bad(NumpyBridge):
        name = "has space"

    b = Bad(view_of(), {})
    with pytest.raises(DescribeError):
        validate_describe(b.describe(), b, b.bench)


# ----------------------------------------------------------------------------- input vocabulary


def _video_desc(**entry) -> tuple[dict, NumpyBridge]:
    b = NumpyBridge(view_of(), {})
    d = b.describe()
    d["input"]["video"].update(entry)
    return d, b


@pytest.mark.parametrize(
    "temporal",
    [
        {"rule": "uniform", "n": 16, "span": "full"},
        {"rule": "uniform", "n": 8, "span": [1.0, 9.0], "edge": "clamp"},
        {"rule": "fps", "fps": 4.0, "n": 8, "start": "centre"},
        {"rule": "fps", "fps": 2, "n": 8, "start": 0.0, "edge": "clamp"},
        {"rule": "index_linspace", "n": 8},
        {"rule": "explicit", "times": [5.0]},
        {"rule": "all"},
        {"rule": "custom", "description": "TSN segments with a centre frame each"},
    ],
)
def test_input_accepts_every_video_rule(temporal):
    d, b = _video_desc(temporal=temporal)
    validate_describe(d, b, b.bench)


@pytest.mark.parametrize(
    "temporal",
    [
        {"rule": "linspace", "n": 4},  # not in the vocabulary
        {"rule": "uniform"},  # no n
        {"rule": "uniform", "n": 0},
        {"rule": "uniform", "n": 4, "span": [3.0, 1.0]},
        {"rule": "uniform", "n": 4, "edge": "wrap"},
        {"rule": "uniform", "n": 4, "jitter": True},  # unknown parameter
        {"rule": "fps", "fps": 0, "n": 4},
        {"rule": "explicit", "times": []},
        {"rule": "stored", "times": {"rule": "uniform", "n": 8}},  # not in the vocabulary
        {"rule": "span", "length_s": 10.0},  # audio only
        {"rule": "custom"},  # no description
        "uniform",
    ],
)
def test_input_refuses_bad_video_rules(temporal):
    d, b = _video_desc(temporal=temporal)
    with pytest.raises(DescribeError) as ei:
        validate_describe(d, b, b.bench)
    assert ei.value.code == E_INPUT


@pytest.mark.parametrize(
    "spatial",
    [
        {"resize": None, "crop": None, "interp": "bilinear", "antialias": False},
        {"resize": {"long": 256}, "crop": {"kind": "region", "region": [0.0, 0.125, 1.0, 0.875]}, "interp": "bicubic", "antialias": True},
        {"resize": {"size": [224, 224]}, "crop": None, "interp": "bilinear", "antialias": True},
        {"rule": "custom", "description": "letterbox to 4:3"},
    ],
)
def test_input_accepts_spatial_rules(spatial):
    d, b = _video_desc(spatial=spatial)
    validate_describe(d, b, b.bench)


@pytest.mark.parametrize(
    "spatial",
    [
        {"resize": {"short": 0}, "crop": None, "interp": "bilinear", "antialias": True},
        {"resize": {"short": 224, "long": 300}, "crop": None, "interp": "bilinear", "antialias": True},
        {"resize": None, "crop": {"kind": "random", "size": [224, 224]}, "interp": "bilinear", "antialias": True},
        {"resize": None, "crop": {"kind": "region", "region": [0.5, 0, 0.4, 1]}, "interp": "bilinear", "antialias": True},
        {"resize": None, "crop": None, "interp": "bilinear"},
        {"resize": None, "crop": None, "interp": "bilinear", "antialias": True, "flip": True},
        {"rule": "custom"},
    ],
)
def test_input_refuses_bad_spatial_rules(spatial):
    d, b = _video_desc(spatial=spatial)
    with pytest.raises(DescribeError) as ei:
        validate_describe(d, b, b.bench)
    assert ei.value.code == E_INPUT


@pytest.mark.parametrize(
    "mutate",
    [
        lambda inp: inp["video"].pop("norm"),
        lambda inp: inp["video"]["norm"].pop("source"),
        lambda inp: inp["video"]["norm"].update(source="  "),
        lambda inp: inp["video"].pop("decoder"),
        lambda inp: inp["video"].update(decoder={"version": "1"}),
        lambda inp: inp["video"].pop("spatial"),
        lambda inp: inp["video"].update(front_end={"sample_rate": 16000}),  # not a video key
        lambda inp: inp.update(depth={"norm": {"source": "x"}, "decoder": {"lib": "x", "version": "1"}}),  # not a benchmark stream
        lambda inp: inp.clear(),
    ],
)
def test_input_block_structure(mutate):
    b = NumpyBridge(view_of(), {})
    d = b.describe()
    mutate(d["input"])
    with pytest.raises(DescribeError) as ei:
        validate_describe(d, b, b.bench)
    assert ei.value.code == E_INPUT


def test_input_must_cover_every_tagged_stream_and_audio_needs_a_front_end():
    bench = FakeBenchmark(streams={"video": "video", "audio": "audio"})
    b = AVBridge(view_of(bench), {})
    d = b.describe()
    validate_describe(d, b, b.bench)
    d2 = copy.deepcopy(d)
    del d2["input"]["audio"]
    with pytest.raises(DescribeError, match="tagged"):
        validate_describe(d2, b, b.bench)
    d3 = copy.deepcopy(d)
    d3["input"]["audio"]["front_end"] = {"sample_rate": 16000}
    with pytest.raises(DescribeError, match="resample"):
        validate_describe(d3, b, b.bench)
    d4 = copy.deepcopy(d)
    d4["input"]["audio"]["temporal"] = {"rule": "uniform", "n": 4}
    with pytest.raises(DescribeError, match="not legal"):
        validate_describe(d4, b, b.bench)


@pytest.mark.parametrize(
    "temporal",
    [
        {"rule": "span", "start_s": "centre", "length_s": 10.0, "pad": "none"},
        {"rule": "full", "max_s": 10.0, "pad_to_s": 10.0},
        {"rule": "full", "max_s": None, "pad_to_s": None},
        {"rule": "segments", "n": 4, "length_s": 1.0, "span": "full", "pad": "zeros"},
        {"rule": "custom", "description": "whole file, model pads internally"},
    ],
)
def test_input_accepts_audio_rules(temporal):
    bench = FakeBenchmark(streams={"video": "video", "audio": "audio"})
    b = AVBridge(view_of(bench), {})
    d = b.describe()
    d["input"]["audio"]["temporal"] = temporal
    validate_describe(d, b, b.bench)


def test_image_stream_has_no_temporal_rule():
    bench = FakeBenchmark(streams={"image": "image"})
    view = view_of(bench)

    class ImageBridge(NumpyBridge):
        name = "np-image"
        streams = ("image",)

        def input_block(self):
            return {"image": {"spatial": dict(VIDEO_INPUT["spatial"]), "norm": dict(VIDEO_INPUT["norm"]), "decoder": dict(VIDEO_INPUT["decoder"])}}

    b = ImageBridge(view, {})
    validate_describe(b.describe(), b, view)
    d = b.describe()
    d["input"]["image"]["temporal"] = {"rule": "all"}
    with pytest.raises(DescribeError, match="unknown key"):
        validate_describe(d, b, view)


def test_temporal_vocabulary_is_the_documented_table():
    assert set(TEMPORAL_RULES) == {"uniform", "fps", "index_linspace", "explicit", "all", "span", "full", "segments", "custom"}
    assert TEMPORAL_RULES["uniform"] == {"video"} and TEMPORAL_RULES["span"] == {"audio"}


def test_audio_input_fixture_is_valid_vocabulary():
    assert AUDIO_INPUT["temporal"]["rule"] in TEMPORAL_RULES


# ----------------------------------------------------------------------------- identity


def test_describe_sha_excluded_set_and_every_bridge_decision_changes_the_digest():
    """describe_sha256 is the consistency digest; the excluded set is the hashing module's."""
    assert DESCRIBE_SHA_EXCLUDED == frozenset({"weights_ref", "batch_tolerance", "batch_size"}) and DESCRIBE_SHA_EXCLUDED is hashing.DESCRIBE_SHA_EXCLUDED
    d = describe_of(NumpyBridge)
    assert hashing.describe_sha256(d) == hashing.describe_sha256(dict(d, weights_ref="/other/path"))
    # yavih-describe-2: the batch size and the decoder build are provenance
    assert hashing.describe_sha256(d) == hashing.describe_sha256(dict(d, batch_size=d["batch_size"] + 1))
    d3 = copy.deepcopy(d)
    d3["input"]["video"]["decoder"].update(version="2.0", libav="61.0")
    assert hashing.describe_sha256(d) == hashing.describe_sha256(d3)
    for change in (
        lambda x: x["precision"].update(allow_tf32=True),
        lambda x: x["args"].update(layer=13),
        lambda x: x["input"]["video"]["temporal"].update(n=8),
        lambda x: x["input"]["video"]["spatial"].update(interp="bicubic"),
        lambda x: x["input"]["video"]["norm"].update(source="other/repo@def"),
    ):
        d2 = copy.deepcopy(d)
        change(d2)
        assert hashing.describe_sha256(d) != hashing.describe_sha256(d2)


def test_batch_tolerance_is_outside_describe_sha256():
    d = describe_of(NumpyBridge)
    assert hashing.describe_sha256(d) == hashing.describe_sha256(dict(d, batch_tolerance={"rel_l2": 1e-4, "reason": "bf16"}))


# ----------------------------------------------------------------------------- preprocessing source slot


def test_preprocessing_source_slot():
    assert "CenterCrop" in (preprocessing_source(NumpyBridge) or "")
    assert preprocessing_source(NoPreprocBridge) is None

    class Inline(Bridge):
        """Preprocessing source: facebookresearch/vjepa2@abc src/datasets/utils/video/transforms.py"""

    assert "vjepa2" in (preprocessing_source(Inline) or "")

    class Bare(Bridge):
        pass

    assert preprocessing_source(Bare) is None


# ----------------------------------------------------------------------------- fingerprint_state_dict


def test_fingerprint_numpy_state_dict_is_order_independent_and_sensitive():
    a = {"w": np.arange(6, dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    b = {"b": np.zeros(2, dtype=np.float32), "w": np.arange(6, dtype=np.float32)}
    h = fingerprint_state_dict(a)
    assert len(h) == 64 and h == h.lower() and h == fingerprint_state_dict(b)
    assert fingerprint_state_dict({"w": np.arange(6, dtype=np.float64), "b": np.zeros(2, dtype=np.float32)}) != h  # dtype
    assert fingerprint_state_dict({"w": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.zeros(2, dtype=np.float32)}) != h  # shape
    c = dict(a)
    c["w"] = c["w"].copy()
    c["w"][0] = 1.0
    assert fingerprint_state_dict(c) != h  # bytes
    assert fingerprint_state_dict({"w2": a["w"], "b": a["b"]}) != h  # name


def test_fingerprint_accepts_object_with_state_dict_and_json_values():
    class M:
        def state_dict(self):
            return {"w": np.ones(3, dtype=np.float32), "step": 7}

    h = fingerprint_state_dict(M())
    assert h == fingerprint_state_dict({"w": np.ones(3, dtype=np.float32), "step": 7})
    assert h != fingerprint_state_dict({"w": np.ones(3, dtype=np.float32), "step": 8})
    with pytest.raises(TypeError):
        fingerprint_state_dict(42)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not importable")
def test_fingerprint_torch_module_matches_numpy_bytes_and_handles_bf16():
    import torch

    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 2)
    sd = {k: v.detach().cpu().numpy() for k, v in lin.state_dict().items()}
    assert fingerprint_state_dict(lin) == fingerprint_state_dict(sd)
    bf = {"w": torch.arange(4, dtype=torch.float32).to(torch.bfloat16)}
    h = fingerprint_state_dict(bf)
    assert len(h) == 64
    assert h != fingerprint_state_dict({"w": torch.arange(4, dtype=torch.float32)})
    # the digest is reproducible from the raw little-endian bytes (a non-torch implementation agrees)
    raw = bf["w"].contiguous().view(torch.uint8).numpy().tobytes()
    hh = hashlib.sha256()
    hh.update(hashing.canon(["w", "bfloat16", [4], len(raw)]).encode())
    hh.update(b"\0")
    hh.update(raw)
    hh.update(b"\n")
    assert hh.hexdigest() == h


# ----------------------------------------------------------------------------- grid_pos


def test_grid_pos_pinned_values():
    g = grid_pos(2, 2, 2, 1)
    assert g.dtype == np.float32 and g.shape == (8, 4)
    expected = np.array(
        [
            [1, 0.25, 0.25, 0.25],
            [1, 0.25, 0.25, 0.75],
            [1, 0.25, 0.75, 0.25],
            [1, 0.25, 0.75, 0.75],
            [1, 0.75, 0.25, 0.25],
            [1, 0.75, 0.25, 0.75],
            [1, 0.75, 0.75, 0.25],
            [1, 0.75, 0.75, 0.75],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(g, expected)
    assert g.flags["C_CONTIGUOUS"]


def test_grid_pos_cls_row_first():
    g = grid_pos(1, 1, 1, 0, cls=True)
    np.testing.assert_array_equal(g, np.array([[0, -1, -1, -1], [0, 0.5, 0.5, 0.5]], dtype=np.float32))


def test_grid_pos_region_maps_patch_centres_into_the_original_frame():
    # resize short 224 + centre crop 224 of a 16:9 1280x720 frame keeps x in [0.21875, 0.78125]
    region = (0.0, 0.21875, 1.0, 0.78125)
    g = grid_pos(1, 2, 2, 0, region=region)
    np.testing.assert_allclose(g[:, 2], [0.25, 0.25, 0.75, 0.75])
    np.testing.assert_allclose(g[:, 3], [0.21875 + 0.25 * 0.5625, 0.21875 + 0.75 * 0.5625] * 2)
    assert (g[:, 3] >= region[1]).all() and (g[:, 3] <= region[3]).all()


def test_grid_pos_t_centres_are_media_fractions_and_padding_is_minus_one():
    times, duration = [0.24, 0.76, 1.24, 1.76], 2.0
    g = grid_pos(4, 1, 1, 0, t_centres=[t / duration for t in times])
    np.testing.assert_allclose(g[:, 1], [0.12, 0.38, 0.62, 0.88], rtol=1e-6)
    g = grid_pos(2, 1, 1, 0, t_centres=[0.9, -1.0])  # second token centred in bridge padding
    np.testing.assert_allclose(g[:, 1], [0.9, -1.0])


def test_grid_pos_token_count_and_order_is_row_major_t_y_x():
    g = grid_pos(3, 4, 5, 0)
    assert g.shape == (60, 4)
    assert np.all(np.diff(g[:5, 3]) > 0)  # x fastest, then y, then t
    assert g[5, 2] > g[0, 2] and g[20, 1] > g[0, 1]
    assert np.all((g[:, 1:] >= 0) & (g[:, 1:] <= 1))


@pytest.mark.parametrize(
    "kw",
    [
        dict(T=0, H=1, W=1, stream=0),
        dict(T=1, H=0, W=1, stream=0),
        dict(T=1, H=1, W=0, stream=0),
        dict(T=1, H=1, W=1, stream=-1),
        dict(T=2, H=1, W=1, stream=0, t_centres=[0.5]),
        dict(T=2, H=1, W=1, stream=0, t_centres=[0.5, 1.5]),
        dict(T=2, H=1, W=1, stream=0, t_centres=[0.5, -0.5]),
        dict(T=1, H=1, W=1, stream=0, region=(0.6, 0.0, 0.4, 1.0)),
        dict(T=1, H=1, W=1, stream=0, region=(0.0, 0.0, 1.0, 1.2)),
        dict(T=1, H=1, W=1, stream=True),
    ],
)
def test_grid_pos_refuses_bad_arguments(kw):
    with pytest.raises(ValueError):
        grid_pos(**kw)


def test_argspec_is_frozen_dataclass():
    s = ArgSpec(type=int, default=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.default = 2  # type: ignore[misc]


# ----------------------------------------------------------------------------- rule facts guaranteed


@pytest.mark.parametrize(
    "temporal,ok",
    [
        ({"rule": "uniform", "n": 8, "span": "full"}, True),
        ({"rule": "fps", "fps": 4.0, "n": 8, "start": "centre"}, True),
        ({"rule": "index_linspace", "n": 8}, False),  # needs n_frames, which k400-like folders do not guarantee
        ({"rule": "all"}, False),
        ({"rule": "custom", "description": "TSN"}, True),
    ],
)
def test_rule_needing_unguaranteed_facts_is_refused(temporal, ok):
    """A checkable rule must be evaluable on every row; a fact it needs that the folder's
    input_contract.info does not guarantee would leave it silently unchecked on rows where the fact is null."""
    from yavih.bridge import BenchmarkView

    b = NumpyBridge(view_of(), {})
    v = view_of()
    ic = dict(v.input_contract, info={"video": ["duration_s", "fps", "width", "height", "container", "bytes"]})
    k400_like = BenchmarkView(**{**dataclasses.asdict(v), "input_contract": ic})
    d = b.describe()
    d["input"]["video"]["temporal"] = temporal
    if ok:
        validate_describe(d, b, k400_like)
    else:
        with pytest.raises(DescribeError) as ei:
            validate_describe(d, b, k400_like)
        assert ei.value.code == E_INPUT and "n_frames" in str(ei.value)
        # the hint offers the legitimate opt-out, never calls it a lesser "unchecked" choice
        msg = str(ei.value)
        assert "'custom'" in msg and "N_RULE_CUSTOM" in msg and "W_RULE_UNCHECKED" not in msg
