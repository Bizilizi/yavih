"""``output_contract.isolation`` and ``bridge check``'s modality-isolation gate (``E_MODALITY_LEAK``, the library gate
``isolation``): parsing, refusals and hashing of the block, the stored-file helper the gate uses, the gate's default
selection and its verdicts on honest and leaky bridges. The fixtures are the shared-space folders' own stand-ins
(``vggsound_retrieval`` / ``vggss_loc`` ``testing.py``)."""

from __future__ import annotations

import pytest
import yaml

from yavih import testing as T
from yavih.benchmark import Benchmark, SchemaError, load_yaml
from yavih.benchmarks import packaged_folder
from yavih.extract import LIBRARY_GATES, ExtractError, bridge_check, folder_gates
from yavih.testing.folder import load_module
from yavih.waivers import WAIVABLE

pytest.importorskip("av")


def _synth(name: str):
    return load_module(packaged_folder(name) / "testing.py", f"{name}_testing_isolation").synth


def _stand_in(root, name: str, *, n_eval: int, shared: bool = False):
    return _synth(name).make_stand_in(root, Benchmark.load(name), n_eval=n_eval, n_train=0, shared=shared)


@pytest.fixture
def retrieval(yavih_home, monkeypatch):
    monkeypatch.setenv("YAVIH_TESTING", "1")
    return _stand_in(yavih_home / "benchmarks", "vggsound_retrieval", n_eval=8)


def _with(folder, edit):
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    edit(y)
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_isolation_is_parsed_and_hashed(retrieval):
    b = Benchmark.from_dir(retrieval)
    assert b.spec.output_contract.isolation == {"audio_emb": ("audio",), "video_emb": ("video",)}
    before = b.protocol_hash
    _with(retrieval, lambda y: y["output_contract"]["isolation"].pop("video_emb"))
    assert Benchmark.from_dir(retrieval).protocol_hash != before


@pytest.mark.parametrize("edit, message", [
    (lambda y: y["output_contract"].__setitem__("isolation", {}), "empty block"),
    (lambda y: y["output_contract"]["isolation"].__setitem__("nope", ["audio"]), "not a key of output_contract.schema"),
    (lambda y: y["output_contract"]["isolation"].__setitem__("audio_emb", []), "non-empty list"),
    (lambda y: y["output_contract"]["isolation"].__setitem__("audio_emb", ["audio", "audio"]), "listed twice"),
    (lambda y: y["output_contract"]["isolation"].__setitem__("audio_emb", ["text"]), "unknown stream"),
    (lambda y: y["output_contract"]["isolation"].__setitem__("audio_emb", ["audio", "video"]), "isolates nothing"),
])
def test_isolation_refusals(retrieval, edit, message):
    _with(retrieval, edit)
    with pytest.raises(SchemaError, match=message):
        load_yaml(retrieval / "benchmark.yaml")


def test_isolation_needs_a_schema(yavih_home):
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf")
    _with(folder, lambda y: y["output_contract"].__setitem__("isolation", {"feat": ["video"]}))
    with pytest.raises(SchemaError, match="not declared"):
        load_yaml(folder / "benchmark.yaml")


def test_stored_file_returns_the_file_or_writes_the_bytes(retrieval, tmp_path):
    from yavih import sample as S

    b = Benchmark.from_dir(retrieval)
    s = next(iter(S.iter_samples(b, "test", range(0, 1), 1, 0, 0)))
    st = s.streams["audio"]
    with open(S.stored_file(st, tmp_path), "rb") as f:
        assert f.read() == st.bytes()


def test_the_gate_runs_by_default_where_isolation_is_declared(retrieval, yavih_home):
    assert "isolation" in LIBRARY_GATES and "E_MODALITY_LEAK" in WAIVABLE
    assert [g for g, _ in folder_gates(Benchmark.from_dir(retrieval))] == ["isolation"]
    plain = Benchmark.from_dir(T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf"))
    assert folder_gates(plain) == []


def test_the_isolation_gate_passes_an_honest_bridge_and_refuses_a_leaky_one(retrieval):
    X = _synth("vggsound_retrieval")
    bench = Benchmark.from_dir(retrieval)
    iso = bridge_check(bridge=X.OracleRetrievalBridge, bench=bench, n=4)["gates"]["isolation"]
    assert iso["keys"] == {"audio_emb": ["audio"], "video_emb": ["video"]} and iso["max_rel_l2"] == 0.0
    assert {c["variant"] for c in iso["checks"]} == {"swap:video", "swap:audio"}
    with pytest.raises(ExtractError) as e:
        bridge_check(bridge=X.LeakyRetrievalBridge, bench=bench, n=4)
    assert e.value.code == "E_MODALITY_LEAK" and "'video_emb'" in str(e.value) and "audio" in str(e.value)


def test_a_shared_file_has_its_other_track_blanked(yavih_home, monkeypatch):
    """With the audio a track of the mp4, the gate blanks one track at a time (a swap would alter both streams)."""
    monkeypatch.setenv("YAVIH_TESTING", "1")
    X = _synth("vggsound_retrieval")
    bench = Benchmark.from_dir(_stand_in(yavih_home / "shared", "vggsound_retrieval", n_eval=8, shared=True))
    variants = {c["variant"] for c in bridge_check(bridge=X.OracleRetrievalBridge, bench=bench, n=4)["gates"]["isolation"]["checks"]}
    assert variants == {"blank:audio-track-of:video", "blank:video-track-of:audio"}, variants
    with pytest.raises(ExtractError) as e:
        bridge_check(bridge=X.LeakyRetrievalBridge, bench=bench, n=4)
    assert e.value.code == "E_MODALITY_LEAK" and "audio track" in str(e.value)


def test_the_gate_catches_tokens_that_read_the_audio(yavih_home, monkeypatch):
    monkeypatch.setenv("YAVIH_TESTING", "1")
    X = _synth("vggss_loc")
    bench = Benchmark.from_dir(_stand_in(yavih_home / "loc", "vggss_loc", n_eval=8))
    assert bridge_check(bridge=X.OracleLocBridge, bench=bench, n=4)["gates"]["isolation"]["max_rel_l2"] == 0.0
    with pytest.raises(ExtractError) as e:
        bridge_check(bridge=X.LeakyLocBridge, bench=bench, n=4)
    assert e.value.code == "E_MODALITY_LEAK" and "'video_tokens'" in str(e.value)
