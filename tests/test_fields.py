"""Schema 3 ``fields``: one vocabulary, file and cell storage, ``Sample.cells``, the hashed view, the migration
(docs/benchmark-folder.md)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from yavih import codecs as C
from yavih import fields as F
from yavih import testing as T
from yavih.benchmark import Benchmark, SchemaError, migrate_schema, migrate_text
from yavih.bridge import Bridge
from yavih.freeze import csv_bytes, gzip_bytes


def test_one_vocabulary():
    import yavih.benchmark as B
    import yavih.bridge as R
    import yavih.sample as S

    assert B.STREAM_KINDS is S.STREAM_KINDS is R.STREAM_KINDS is F.MEDIA_KINDS
    assert S.INFO_VOCAB is F.INFO_VOCAB and B.COLUMN_ROLES is F.COLUMN_ROLES and "cell" in F.COLUMN_ROLES and "json" in F.COLUMN_TYPES


def _with_cells(folder, *, caption_column=True):
    """Add a `caption` (str) and an `options` (json) cell field to a synthetic folder's manifest and yaml."""
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    raw = gzip.decompress((folder / "manifest.csv.gz").read_bytes()).decode("utf-8")
    rows = list(csv.reader(io.StringIO(raw)))
    header, body = rows[0], rows[1:]
    key = header.index(y["manifest"]["key"])
    header += ["caption", "options"]
    body = [r + [f"the sound of {r[key]}", json.dumps(["dog", "rain", r[key]])] for r in body]
    data = csv_bytes(header, body)
    (folder / "manifest.csv.gz").write_bytes(gzip_bytes(data))
    y["manifest"]["sha256"] = hashlib.sha256(data).hexdigest()
    y["manifest"]["columns"].update(caption={"type": "str", "role": "cell", "field": "caption"},
                                    options={"type": "json", "role": "cell", "field": "options"})
    y["fields"]["caption"] = {"storage": "cell", "kind": "text", "column": "caption"}
    y["fields"]["options"] = {"storage": "cell", "kind": "options", "column": "options"}
    y["input_contract"]["handed_over"].update(caption="the first caption of the clip", options="three candidate labels")
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    return Benchmark.from_dir(folder)


def test_cell_fields_reach_the_bridge_as_sample_cells(tmp_path):
    folder = T.make_synthetic_benchmark(tmp_path, "clf", n_train=4, n_eval=2, media="none")
    base = Benchmark.from_dir(folder)
    b = _with_cells(folder)
    assert set(b.spec.cells) == {"caption", "options"} and b.spec.cells["options"].kind == "options"
    assert set(b.spec.streams) == set(base.spec.streams)
    assert b.input_hash != base.input_hash  # what is handed over changed
    view = b.view()
    assert view.fields["caption"] == {"storage": "cell", "kind": "text"} and view.fields["video"]["storage"] == "file"
    s = b.sample("val", 0)
    assert s.cells == {"caption": f"the sound of {s.clip_id}", "options": ["dog", "rain", s.clip_id]}
    s.cells["options"].append("mutated")  # a private copy: the manifest is untouched
    assert b.sample("val", 0).cells["options"] == ["dog", "rain", s.clip_id]
    from yavih.sample import rehandle

    assert rehandle(s, "x").cells == s.cells and rehandle(s, "x").cells is not s.cells


def test_cell_field_refusals(tmp_path):
    folder = T.make_synthetic_benchmark(tmp_path, "clf", n_train=4, n_eval=2, media="none")
    _with_cells(folder)
    path = folder / "benchmark.yaml"
    good = path.read_text(encoding="utf-8")

    def broken(mutate, match):
        y = yaml.safe_load(good)
        mutate(y)
        path.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
        with pytest.raises(SchemaError, match=match):
            Benchmark.from_dir(folder)

    broken(lambda y: y["fields"]["caption"].__setitem__("column", "label"), "role cell")
    broken(lambda y: y["fields"]["caption"].__setitem__("kind", "Text!"), "kind")
    broken(lambda y: y["fields"]["caption"].__setitem__("storage", "blob"), "storage")
    broken(lambda y: y["manifest"]["columns"]["caption"].pop("field"), "must name its `field`")
    broken(lambda y: y["input_contract"]["handed_over"].pop("caption"), "handed_over")
    broken(lambda y: y["input_contract"]["info"].__setitem__("caption", []), "a cell has no media facts")


def test_schema_2_is_refused_with_the_migrate_hint_and_migrates_without_moving_a_hash(tmp_path):
    folder = T.make_synthetic_benchmark(tmp_path, "clf", n_train=4, n_eval=2, media="none")
    b3 = Benchmark.from_dir(folder)
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y2 = dict(y, schema_version=2)
    y2["streams"] = {n: {k: v for k, v in e.items() if k != "storage"} for n, e in y2.pop("fields").items()}
    text2 = yaml.safe_dump(y2, sort_keys=False)
    (folder / "benchmark.yaml").write_text(text2, encoding="utf-8")
    with pytest.raises(SchemaError, match="bench migrate"):
        Benchmark.from_dir(folder)
    assert migrate_schema(y2)["fields"] == y["fields"]
    (folder / "benchmark.yaml").write_text(migrate_text(text2), encoding="utf-8")
    again = Benchmark.from_dir(folder)
    assert (again.input_hash, again.protocol_hash) == (b3.input_hash, b3.protocol_hash)


class CaptionGenerator(Bridge):
    """A text-to-audio toy: the clip is seeded by the caption cell (it never reads the video)."""

    batch_size = 2
    outputs = ("audio",)

    def describe(self) -> dict:
        return {"model": "toy-t2a", "reading": "t", "weights_sha256": None, "weights_ref": "-", "batch_size": self.batch_size, "views": 1,
                "input": {"video": {"temporal": {"rule": "custom", "description": "not read"}, "spatial": {"rule": "custom", "description": "not read"},
                                    "norm": {"source": "none"}, "decoder": {"lib": "none", "version": "0"}}},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
                "args": {}, "outputs": ["audio"], "sampling": {"seed": 1, "steps": 2, "guidance": None, "n_candidates": 1, "selection": "first"}}

    def encode(self, samples):
        out = []
        for s in samples:
            seed = int.from_bytes(hashlib.sha256(s.cells["caption"].encode()).digest()[:8], "big")
            out.append(C.wav_bytes(np.random.default_rng(seed).standard_normal(1600).astype(np.float32) * 0.1, 16000))
        return {"audio": out, "used": [{"video": {"times": []}} for _ in samples]}


def test_a_text_to_audio_generator_on_a_caption_cell(tmp_path, yavih_home):
    from yavih.extract import bridge_check, run_extract
    from yavih.io import validate_features

    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=6, n_eval=4, media="tiny")
    b = _with_cells(folder)
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["output_contract"].update(schema={"audio": {"type": "wav", "channels": 1, "duration_s": [0.05, 0.2]}}, sampling="required")
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    b = Benchmark.from_dir(folder)
    prof = bridge_check(bridge=CaptionGenerator, bench=b, n=6)
    assert prof["invariance"]["rerun_bit_identical"] is True and prof["invariance"]["batch_rel_l2"] == 0.0
    run = tmp_path / "run"
    assert run_extract(bench=b, bridge=CaptionGenerator, split="val", shard=(0, 1), out=run, canaries=1)["status"] == "ok"
    assert validate_features(run / "features" / "val", bench=b).ok


def _cells_only(folder):
    """Drop every file field: the caption and options cells are the whole input (a text-to-audio benchmark)."""
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    files = [n for n, e in y["fields"].items() if e.get("storage") == "file"]
    for n in files:
        del y["fields"][n]
        y["input_contract"]["handed_over"].pop(n, None)
        y["input_contract"]["info"].pop(n, None)
    for c in y["manifest"]["columns"].values():
        if c.get("role") == "media":
            c.clear()
            c.update(type="str", role="extra")
    y["output_contract"].update(schema={"audio": {"type": "wav", "channels": 1, "duration_s": [0.05, 0.2]}}, sampling="required")
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    return Benchmark.from_dir(folder)


class TextOnlyGenerator(CaptionGenerator):
    def describe(self) -> dict:
        return {**super().describe(), "input": {"caption": {"note": "the prompt, verbatim"}}}

    def encode(self, samples):
        return {**super().encode(samples), "used": [{} for _ in samples]}


def test_a_benchmark_whose_only_inputs_are_cells(tmp_path, yavih_home):
    from yavih.extract import bridge_check, run_extract
    from yavih.io import validate_features

    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=6, n_eval=4, media="tiny")
    _with_cells(folder)
    b = _cells_only(folder)
    assert b.spec.streams == {} and set(b.spec.cells) == {"caption", "options"}
    assert b.view().fields == {"caption": {"storage": "cell", "kind": "text"}, "options": {"storage": "cell", "kind": "options"}}
    s = next(iter(b.reader("val", rows=range(1))))
    assert s.streams == {} and s.cells["caption"].startswith("the sound of ")
    prof = bridge_check(bridge=TextOnlyGenerator, bench=b, n=6)
    assert prof["invariance"]["rerun_bit_identical"] is True
    run = tmp_path / "run"
    assert run_extract(bench=b, bridge=TextOnlyGenerator, split="val", shard=(0, 1), out=run, canaries=1)["status"] == "ok"
    rep = validate_features(run / "features" / "val", bench=b)
    assert rep.ok, rep
    assert rep.rules_checked == {}  # a cell has no rule
    again = validate_features(run / "features" / "val", bench=b)  # reuses _meta.json (the cheap recheck)
    assert again.ok and again.shards.get("reused_meta") is True and again.N == 0


def test_no_field_at_all_is_refused(tmp_path):
    folder = T.make_synthetic_benchmark(tmp_path, "clf", n_train=4, n_eval=2, media="none")
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["fields"] = {}
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    with pytest.raises(SchemaError, match="at least one field"):
        Benchmark.from_dir(folder)


def test_describe_input_cell_entries():
    from yavih.bridge import DescribeError, input_streams, validate_input

    validate_input({"caption": {}}, {}, [], {"caption": "text"})
    validate_input({"caption": {"note": "verbatim"}}, {}, [], {"caption": "text"})
    for bad in ({"caption": {"norm": {"source": "x"}}}, {"caption": {"note": 3}}, {"prompt": {}}, {}):
        with pytest.raises(DescribeError):
            validate_input(bad, {}, [], {"caption": "text"})
    assert input_streams({"caption": {"note": "x"}, "video": {"norm": {}, "decoder": {}}}) == ["video"]


def _with_pointcloud(folder):
    """Add an opaque `cloud` file field (kind `pointcloud`, a .ply per row under the media root)."""
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    media_root = Path(y["data"]["roots"]["media_root"]["default"])
    raw = gzip.decompress((folder / "manifest.csv.gz").read_bytes()).decode("utf-8")
    rows = list(csv.reader(io.StringIO(raw)))
    header, body = rows[0], rows[1:]
    key, split = header.index(y["manifest"]["key"]), header.index("split")
    header += ["rel_cloud"]
    body = [r + [f"{r[split]}/{r[key]}.ply"] for r in body]
    for r in body:
        p = media_root / r[-1]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 " + r[key].encode() + b"\n")
    data = csv_bytes(header, body)
    (folder / "manifest.csv.gz").write_bytes(gzip_bytes(data))
    y["manifest"]["sha256"] = hashlib.sha256(data).hexdigest()
    y["manifest"]["columns"]["rel_cloud"] = {"type": "str", "role": "media", "stream": "cloud"}
    y["fields"]["cloud"] = {"storage": "file", "kind": "pointcloud", "media_column": "rel_cloud", "root": "media_root", "layout": "dir",
                            "layout_params": {}, "required": True}
    y["input_contract"]["handed_over"]["cloud"] = "the clip's point cloud, a .ply file as stored"
    y["input_contract"]["info"]["cloud"] = ["container", "bytes"]
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    return Benchmark.from_dir(folder)


class CloudBridge(Bridge):
    """Reads the opaque point cloud (its bytes key the features) and nothing else."""

    batch_size = 2
    dtype = "float32"
    pos_mode = "none"

    def describe(self) -> dict:
        return {"model": "toy-cloud", "reading": "p", "weights_sha256": None, "weights_ref": "-", "batch_size": self.batch_size, "views": 1,
                "dtype": "float32", "pos_mode": "none", "streams": [], "layer": "last", "pooling": "none",
                "input": {"cloud": {"note": "the .ply bytes, hashed"}},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True}, "args": {}}

    def encode(self, samples):
        feats = []
        for s in samples:
            seed = int.from_bytes(hashlib.sha256(s.streams["cloud"].bytes()).digest()[:8], "big")
            feats.append(np.random.default_rng(seed).standard_normal((2, 8)).astype(np.float32))
        return np.stack(feats), None, [{} for _ in samples]


def test_an_opaque_file_field_is_handed_over_as_bytes(tmp_path, yavih_home):
    from yavih.benchmark import bench_verify
    from yavih.extract import bridge_check, run_extract
    from yavih.io import validate_features

    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=6, n_eval=4, media="tiny")
    b = _with_pointcloud(folder)
    assert b.spec.streams["cloud"].kind == "pointcloud" and b.view().streams["cloud"] == "pointcloud"
    s = next(iter(b.reader("val", rows=range(1))))
    info = s.streams["cloud"].info
    assert info["container"] == "ply" and info["bytes"] == len(s.streams["cloud"].bytes()) and info["duration_s"] is None
    rep = bench_verify(str(folder), decode_sample=4)
    assert rep["decode_sample"]["failed"] == 0 and rep["decode_sample"]["ok"] > 0
    assert bridge_check(bridge=CloudBridge, bench=b, n=6)["invariance"]["rerun_bit_identical"] is True
    run = tmp_path / "run"
    assert run_extract(bench=b, bridge=CloudBridge, split="val", shard=(0, 1), out=run, canaries=1)["status"] == "ok"
    assert validate_features(run / "features" / "val", bench=b).ok


def test_opaque_kind_rules():
    from yavih.bridge import DescribeError, validate_input
    from yavih.fields import info_vocab, is_file_kind

    assert is_file_kind("pointcloud") and is_file_kind("video") and not is_file_kind("Point-Cloud") and not is_file_kind("")
    assert info_vocab("pointcloud") == {"container", "bytes"} and "fps" in info_vocab("video")
    validate_input({"cloud": {"note": "bytes"}}, {"cloud": "pointcloud"}, [])
    with pytest.raises(DescribeError, match="opaque"):
        validate_input({"cloud": {"temporal": {"rule": "custom"}}}, {"cloud": "pointcloud"}, [])


def _with_arrays(folder):
    """Add an `array` file field `cloud`: an .npz per row holding `points` [5, 3] and `normals` [5, 3]."""
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    media_root = Path(y["data"]["roots"]["media_root"]["default"])
    raw = gzip.decompress((folder / "manifest.csv.gz").read_bytes()).decode("utf-8")
    rows = list(csv.reader(io.StringIO(raw)))
    header, body = rows[0], rows[1:]
    key, split = header.index(y["manifest"]["key"]), header.index("split")
    header += ["rel_cloud"]
    body = [r + [f"{r[split]}/{r[key]}.npz"] for r in body]
    for r in body:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(r[key].encode()).digest()[:8], "big"))
        p = media_root / r[-1]
        p.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez(buf, points=rng.standard_normal((5, 3)).astype(np.float32), normals=np.ones((5, 3), np.float32))
        p.write_bytes(buf.getvalue())
    data = csv_bytes(header, body)
    (folder / "manifest.csv.gz").write_bytes(gzip_bytes(data))
    y["manifest"]["sha256"] = hashlib.sha256(data).hexdigest()
    y["manifest"]["columns"]["rel_cloud"] = {"type": "str", "role": "media", "stream": "cloud"}
    y["fields"]["cloud"] = {"storage": "file", "kind": "array", "media_column": "rel_cloud", "root": "media_root", "layout": "dir",
                            "layout_params": {}, "required": True}
    y["input_contract"]["handed_over"]["cloud"] = "the clip's point cloud: points and normals, an .npz as stored"
    y["input_contract"]["info"]["cloud"] = ["container", "n_arrays"]
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    return Benchmark.from_dir(folder)


class ArrayBridge(CloudBridge):
    """Pools the decoded point cloud."""

    def describe(self) -> dict:
        return {**super().describe(), "model": "toy-array", "input": {"cloud": {"note": "points, mean-pooled"}}}

    def encode(self, samples):
        feats = [np.repeat(s.streams["cloud"].arrays()["points"].mean(0, keepdims=True), 2, 0).astype(np.float32) for s in samples]
        return np.stack(feats), None, [{} for _ in samples]


def test_an_array_file_field_is_handed_over_decoded(tmp_path, yavih_home):
    from yavih.benchmark import bench_verify
    from yavih.extract import bridge_check, run_extract
    from yavih.io import validate_features

    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=6, n_eval=4, media="tiny")
    b = _with_arrays(folder)
    s = next(iter(b.reader("val", rows=range(1))))
    st = s.streams["cloud"]
    assert st.info["container"] == "npz" and st.info["n_arrays"] == 2 and st.info["shape"] is None and st.info["duration_s"] is None
    arrs = st.arrays()
    assert list(arrs) == ["points", "normals"] and arrs["points"].shape == (5, 3) and arrs["points"].dtype == np.float32
    with pytest.raises(ValueError, match="use arrays"):
        st.array()
    rep = bench_verify(str(folder), decode_sample=4)
    assert rep["decode_sample"]["failed"] == 0 and rep["decode_sample"]["ok"] > 0
    assert bridge_check(bridge=ArrayBridge, bench=b, n=6)["invariance"]["rerun_bit_identical"] is True
    run = tmp_path / "run"
    assert run_extract(bench=b, bridge=ArrayBridge, split="val", shard=(0, 1), out=run, canaries=1)["status"] == "ok"
    assert validate_features(run / "features" / "val", bench=b).ok


def test_array_containers_and_refusals():
    import struct

    from yavih.media.arrays import ArrayError, array_facts, container_of, load_arrays

    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    buf = io.BytesIO()
    np.save(buf, a)
    assert array_facts(buf.getvalue(), "npy") == {"dtype": "float32", "shape": [4, 3], "n_arrays": 1}
    assert (load_arrays(buf.getvalue(), "npy")["arr_0"] == a).all()
    head = json.dumps({"__metadata__": {"k": "v"}, "x": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]}}).encode()
    st = struct.pack("<Q", len(head)) + head + a.tobytes()
    assert array_facts(st, "safetensors") == {"dtype": "float32", "shape": [4, 3], "n_arrays": 1}
    assert (load_arrays(st, "safetensors")["x"] == a).all()
    for bad in ("pt", "pkl", "ckpt"):
        with pytest.raises(ArrayError, match="unpickle"):
            container_of(bad)
    with pytest.raises(ArrayError, match="extension"):
        container_of(None)
    obj = io.BytesIO()
    np.save(obj, np.array([{}], dtype=object), allow_pickle=True)
    with pytest.raises(ArrayError, match="object"):
        load_arrays(obj.getvalue(), "npy")
    bf16 = json.dumps({"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    with pytest.raises(ArrayError, match="numpy cannot hold"):
        array_facts(struct.pack("<Q", len(bf16)) + bf16 + b"\0" * 4, "safetensors")
    assert F.info_vocab("array") == F.ARRAY_INFO and "dtype" not in F.info_vocab("pointcloud")
