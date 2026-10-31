"""Schema 3 ``fields``: one vocabulary, file and cell storage, ``Sample.cells``, the hashed view, the migration
(docs/benchmark-folder.md)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json

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
