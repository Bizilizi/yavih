"""``yavih.benchmark`` tests: folder schema, manifest, resolution, census, protocol hash, ``bench *``, probe.py/reader.py, label kinds.

Every benchmark folder here is written by the test itself (synthetic, no network, no decoders).
``make_folder`` builds a valid schema-2 folder from a header/rows/classes triple plus a yaml mutation so
each validation rule can be exercised by breaking exactly one thing.
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import os
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from yavih import benchmark as bm
from yavih import hashing
from yavih.benchmark import (
    Benchmark,
    DataNotReady,
    ManifestError,
    NoTargets,
    SchemaError,
    UnknownBenchmark,
    bench_fetch,
    bench_info,
    bench_verify,
    data_status_path,
    list_benchmarks,
    load_yaml,
)
from yavih.config import ConfigError
from yavih.metrics import registry
from yavih.versions import BENCHMARK_SCHEMA_VERSION, READOUT_VERSION

# ----------------------------------------------------------------------------- helpers


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _csv(header: list[str], rows: list[list[str]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def base_spec(name: str, **over: Any) -> dict[str, Any]:
    """A minimal valid schema-2 clf/single yaml; shas are filled by make_folder."""
    spec: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "name": name,
        "aliases": [],
        "title": "synthetic",
        "readout": "clf",
        "protocol_version": 2,
        "changelog": [
            {"version": 1, "date": "2026-10-07", "reason": "test", "invalidates": False},
            {"version": 2, "date": "2026-10-08", "reason": "test, version 2", "invalidates": False},
        ],
        "source": {"kind": "derived", "urls": [], "licence": "synthetic", "bytes": 0, "sha256": {}, "notes": "t"},
        "budget": {"clips": {"train": 4, "val": 2}, "media_gb": 0.0, "extract_min_per_1k_clips_ref": 0.1, "probe_minutes_ref": {"clf": 1}},
        "manifest": {
            "file": "manifest.csv.gz",
            "sha256": "0" * 64,
            "classes": "classes.csv",
            "classes_sha256": "0" * 64,
            "key": "clip_id",
            "label": {"column": "label", "kind": "single", "sep": ";", "class_key": "name", "sets": None, "train_set": None, "allow_unknown": False, "empty_ok": False},
            "columns": {"rel_media": {"type": "str", "role": "media", "stream": "video"}},
            "upstream": {
                "source": "test",
                "sha256": "x",
                "rows": 6,
                "note": None,
                "upstream_files": {},
                "curation": {"total_upstream": 6, "per_split": {"train": {"listed": 4, "staged": 4, "missing": 0}, "val": {"listed": 2, "staged": 2, "missing": 0}}, "excluded": {}, "short_clips": 0, "leakage_audit": None},
            },
        },
        "splits": {"train": ["train"], "eval": ["val"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95},
        "streams": {"video": {"kind": "video", "media_column": "rel_media", "root": "media_root", "layout": "dir", "layout_params": {}, "required": True}},
        "targets": None,
        "data": {"roots": {"media_root": {"default": "${data.root}/" + name, "description": "d"}}, "tar_index": None, "derived": {}},
        "input_contract": {
            "handed_over": {"video": "the stored mp4 exactly as stored; nothing decoded, cut or resized"},
            "info": {"video": ["duration_s", "container", "bytes"]},
            "facts": [],
            "materialize": None,
        },
        "output_contract": {"shape": "fixed", "coordinates": "media", "target_frame": None, "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None}},
        "external": [],
        "protocol": {"clip_seconds": 10.0, "train_draw": None, "views": "none"},
        "metrics": {"headline": "acc", "list": ["acc", "n_eval"], "floors": ["acc_floor"], "controls": [], "chance": {"acc": 0.5}, "ci": {"acc": "binomial"}, "bootstrap": {"samples": 2000, "seed": 0}, "aggregation": None},
        "readout_params": {},
    }
    spec.update(over)
    return spec


def set_streams(s: dict[str, Any], streams: dict[str, dict[str, Any]], info: dict[str, list[str]] | None = None) -> None:
    """Replace ``streams`` and keep ``input_contract.handed_over``/``info`` in step (exactly the stream names)."""
    s["streams"] = streams
    s["input_contract"]["handed_over"] = {n: f"the stored {v['kind']} exactly as stored" for n, v in streams.items()}
    s["input_contract"]["info"] = {n: list((info or {}).get(n, [])) for n in streams}


def dir_stream(kind: str, column: str, root: str = "media_root", **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "media_column": column, "root": root, "layout": "dir", "layout_params": {}, "required": True, **extra}


BASE_HEADER = ["clip_id", "split", "label", "rel_media"]
BASE_ROWS = [
    ["t0", "train", "a", "train/t0.mp4"],
    ["t1", "train", "b", "train/t1.mp4"],
    ["t2", "train", "a", "train/t2.mp4"],
    ["t3", "train", "b", "train/t3.mp4"],
    ["v0", "val", "a", "val/v0.mp4"],
    ["v1", "val", "b", "val/v1.mp4"],
]
BASE_CLASSES = [["0", "a"], ["1", "b"]]


def make_folder(
    root: Path,
    name: str = "synth",
    *,
    header: list[str] = BASE_HEADER,
    rows: list[list[str]] = BASE_ROWS,
    classes: list[list[str]] | None = BASE_CLASSES,
    classes_header: list[str] | None = None,
    spec: dict[str, Any] | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    folder_name: str | None = None,
    fix_sha: bool = True,
    probe_py: str | None = None,
) -> Path:
    folder = root / (folder_name or name)
    folder.mkdir(parents=True, exist_ok=True)
    spec = copy.deepcopy(spec) if spec is not None else base_spec(name)
    mb = _csv(header, rows)
    with open(folder / "manifest.csv.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(mb)
    if fix_sha:
        spec["manifest"]["sha256"] = _sha(mb)
    if classes is not None:
        ch = classes_header or (["index", "name", "mid"] if len(classes[0]) == 3 else ["index", "name"])
        cb = _csv(ch, classes)
        (folder / "classes.csv").write_bytes(cb)
        if fix_sha:
            spec["manifest"]["classes"] = "classes.csv"
            spec["manifest"]["classes_sha256"] = _sha(cb)
    else:
        spec["manifest"]["classes"] = None
        spec["manifest"]["classes_sha256"] = None
    if mutate is not None:
        mutate(spec)
    if probe_py is not None:
        (folder / "probe.py").write_text(probe_py, encoding="utf-8")
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return folder


def touch_media(root: Path, rel_paths: list[str], content: bytes = b"") -> None:
    for rel in rel_paths:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def with_roots(monkeypatch: pytest.MonkeyPatch, name: str, **roots: Path) -> None:
    for k, v in roots.items():
        monkeypatch.setenv(f"YAVIH_DATA__{name.upper()}__{k.upper()}", str(v))


def _expected_protocol_hash(b: Benchmark) -> str:
    raw = b.spec.raw
    return hashing.protocol_hash(
        name=b.name, protocol_version=b.spec.protocol_version, protocol=raw["protocol"], splits=raw["splits"], streams=raw["streams"],
        input_contract=raw["input_contract"], output_contract=raw["output_contract"], external=raw["external"], metrics=raw["metrics"],
        readout_params=raw["readout_params"], targets=raw["targets"], manifest=raw["manifest"], manifest_hash=b.manifest_hash,
        readout=b.spec.readout,
    )


# ----------------------------------------------------------------------------- load, hashes, normalization


def test_load_hashes_match_oracle(tmp_path):
    rows = [[f"{sp}_{k:04d}", sp, f"c{k % 3}", f"{sp}/c{k % 3}/{sp}_{k:04d}.mp4"] for sp, n in (("train", 12), ("val", 6)) for k in range(n)]
    classes = [["0", "c0"], ["1", "c1"], ["2", "c2"]]
    folder = make_folder(tmp_path, "synth_clf", rows=rows, classes=classes,
                         mutate=lambda s: (s.__setitem__("aliases", ["synth_clf_alias"]), s["budget"].__setitem__("clips", {"train": 12, "val": 6}), s["manifest"].__setitem__("upstream", None)))
    b = Benchmark.load(str(folder))
    assert b.name == "synth_clf" and b.folder == folder.resolve()
    mb, cb = _csv(BASE_HEADER, rows), _csv(["index", "name"], classes)
    # manifest-hash oracle computed without yavih.hashing
    oracle = _sha(b"manifest:" + _sha(mb).encode() + b"\nclasses:" + _sha(cb).encode())
    assert b.manifest_hash == oracle
    # protocol payload (yavih-protocol-5), recomputed independently of yavih.hashing: input side, scoring side, combined,
    # from the declaration only (typed views; no source digest)
    raw = yaml.safe_load((folder / "benchmark.yaml").read_text())
    assert hashing.PROTOCOL_SCHEMA == "yavih-protocol-5"

    def canon(x):
        return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    def part(name, comps):
        return _sha(canon({"schema": "yavih-protocol-5", "part": name, "components": {k: _sha(canon(v)) for k, v in comps.items()}}))

    facts = set(raw["input_contract"].get("facts") or [])
    inp = {"name": "synth_clf", "manifest_hash": oracle, "manifest": {"key": raw["manifest"]["key"], "columns": raw["manifest"]["columns"]},
           "splits": raw["splits"], "streams": raw["streams"],
           "input_contract": {k: raw["input_contract"].get(k) for k in ("info", "facts", "materialize")}, "output_contract": raw["output_contract"],
           "readout": "clf", "readout_inputs": {"params": {}, "protocol": {k: v for k, v in sorted(raw["protocol"].items())
                                                                         if k in set(hashing.INPUT_PROTOCOL_KEYS) | facts}}}
    proto = dict(raw["protocol"])
    if isinstance(proto.get("train_draw"), dict):
        proto["train_draw"] = {k: proto["train_draw"].get(k) for k in ("cap", "seed", "balance")}
    metrics = {k: raw["metrics"].get(k) for k in ("headline", "list", "floors", "controls", "chance", "ci", "bootstrap", "aggregation")}
    sco = {"protocol_version": 2, "protocol": proto, "readout_params": {}, "metrics": metrics, "external": [],
           "targets": None, "label": raw["manifest"]["label"], "readout": {"name": "clf", "version": READOUT_VERSION["clf"]}}
    ih, sh = part("input", inp), part("scoring", sco)
    assert (b.input_hash, b.scoring_hash) == (ih, sh)
    assert b.protocol_hash == _sha(json.dumps({"schema": "yavih-protocol-5", "input_hash": ih, "scoring_hash": sh}, sort_keys=True,
                                              separators=(",", ":")).encode())
    assert b.protocol_hash == _expected_protocol_hash(b)
    assert b.spec.raw == raw
    assert b.roles() == {"train": ["train"], "eval": ["val"], "floor": [], "folds": []}
    assert b.n_rows("train") == 12 and b.n_rows("val") == 6
    assert [c["name"] for c in b.classes()] == ["c0", "c1", "c2"] and b.num_classes == 3
    assert repr(b) == "Benchmark('synth_clf')"


def test_source_revision_is_recorded_and_never_hashed(tmp_path):
    """``source.revision``: the pinned release revision is an optional non-empty string, read into
    ``SourceSpec.revision``; like every ``source`` key it is provenance (``bench verify`` guards the staged copy), so
    adding or changing it never changes ``protocol_hash``. The manifest READ RULES do change it."""
    b0 = Benchmark.from_dir(make_folder(tmp_path / "a"))
    assert b0.spec.source.revision is None
    b1 = Benchmark.from_dir(make_folder(tmp_path / "b", mutate=lambda s: s["source"].__setitem__("revision", "4a0055b2" * 5)))
    b2 = Benchmark.from_dir(make_folder(tmp_path / "c", mutate=lambda s: s["source"].__setitem__("revision", "main")))
    assert b1.spec.source.revision == "4a0055b2" * 5 and b2.spec.source.revision == "main"
    assert b0.protocol_hash == b1.protocol_hash == b2.protocol_hash
    for bad, match in (("", "source.revision"), ("   ", "source.revision"), (7, "source.revision")):
        with pytest.raises(SchemaError, match=match):
            Benchmark.from_dir(make_folder(tmp_path / f"bad{bad!r}", mutate=lambda s, v=bad: s["source"].__setitem__("revision", v)))
    # how rows and labels are read is hashed; where the manifest came from is not
    b3 = Benchmark.from_dir(make_folder(tmp_path / "d", mutate=lambda s: s["manifest"]["label"].__setitem__("empty_ok", True)))
    assert b3.protocol_hash != b0.protocol_hash and b3.protocol_hash == _expected_protocol_hash(b3)
    b4 = Benchmark.from_dir(make_folder(tmp_path / "e", mutate=lambda s: s["manifest"].__setitem__("upstream", None)))
    assert b4.protocol_hash == b0.protocol_hash


def test_manifest_normalization(tmp_path):
    b = Benchmark.from_dir(make_folder(tmp_path))
    m = b.manifest("val")
    assert len(m) == 2 and m.split == "val"
    assert m.row.dtype == np.int64 and m.row.tolist() == [0, 1]
    assert m.clip_id == ["v0", "v1"] and m.splits == ["val"] * 2
    assert m.label_ids == [0, 1]  # classes.csv index is the only vocabulary
    assert m.label_sets is None and m.boxes is None and m.segments is None
    assert m.media[0] == {"video": "val/v0.mp4"} and m.media_key[0] == "val/v0.mp4"
    assert m.extra == [{"rel_media": "val/v0.mp4"}, {"rel_media": "val/v1.mp4"}]  # every declared column, typed
    assert m.file_row.tolist() == [4, 5]
    whole = b.manifest()
    assert len(whole) == 6 and whole.split is None and whole.splits[:4] == ["train"] * 4
    assert b.manifest("val") is m  # cached view
    with pytest.raises(KeyError):
        b.manifest("test")
    assert m.index_of("v1") == 1


def test_protocol_hash_is_env_independent_and_ignores_aliases(tmp_path):
    f1 = make_folder(tmp_path / "a", "synth")
    f2 = make_folder(tmp_path / "b", "synth", mutate=lambda s: s.__setitem__("aliases", ["other_name"]))
    b1, b2 = Benchmark.from_dir(f1), Benchmark.from_dir(f2)
    assert b1.protocol_hash == b2.protocol_hash and b1.manifest_hash == b2.manifest_hash
    # every typed view is hashed: streams, the contracts' data keys, external, a label fact; prose and site facts are not
    for n, mut in enumerate([
        lambda s: s["streams"]["video"].__setitem__("required", False),
        lambda s: s["input_contract"]["handed_over"].__setitem__("video", "something else"),
        lambda s: s["input_contract"]["info"].__setitem__("video", ["duration_s"]),
        lambda s: s["output_contract"]["requires"].__setitem__("pos", ["t"]),
        lambda s: s["protocol"].__setitem__("clip_seconds", 9.0),
        lambda s: s["data"]["roots"]["media_root"].__setitem__("description", "changed"),  # NOT hashed
    ]):
        f = make_folder(tmp_path / f"m{n}", "synth", mutate=mut)
        same = Benchmark.from_dir(f).protocol_hash == b1.protocol_hash
        assert same is (n in (1, 5)), n  # handed_over (1) is prose: documentation, not hashed
    assert READOUT_VERSION["clf"] == 1


# ----------------------------------------------------------------------------- schema: unknown keys everywhere


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("source",),
        ("budget",),
        ("manifest",),
        ("manifest", "label"),
        ("manifest", "columns", "rel_media"),
        ("manifest", "upstream"),
        ("manifest", "upstream", "curation"),
        ("splits",),
        ("streams", "video"),
        ("data",),
        ("data", "roots", "media_root"),
        ("input_contract",),
        ("output_contract",),
        ("output_contract", "requires"),
        ("protocol",),
        ("metrics",),
        ("metrics", "bootstrap"),
    ],
)
def test_unknown_key_at_every_level_is_an_error(tmp_path, path):
    def mutate(s):
        d = s
        for k in path:
            d = d[k]
        d["bogus_key"] = 1

    folder = make_folder(tmp_path, mutate=mutate)
    with pytest.raises(SchemaError, match="bogus_key"):
        load_yaml(folder / "benchmark.yaml")


def test_unknown_changelog_key_and_readout_params_key(tmp_path):
    f = make_folder(tmp_path, mutate=lambda s: s["changelog"][0].__setitem__("who", "me"))
    with pytest.raises(SchemaError, match="who"):
        load_yaml(f / "benchmark.yaml")
    f = make_folder(tmp_path / "b", mutate=lambda s: s["readout_params"].__setitem__("T", 8))
    with pytest.raises(SchemaError, match="readout_params"):
        load_yaml(f / "benchmark.yaml")


def test_missing_required_key_is_an_error(tmp_path):
    for n, (mut, match) in enumerate([
        (lambda s: s.pop("budget"), "missing required key"),
        (lambda s: s["protocol"].pop("views"), "views"),
        (lambda s: s.pop("external"), "external"),
        (lambda s: s["input_contract"].pop("facts"), "facts"),
        (lambda s: s["output_contract"].pop("requires"), "requires"),
    ]):
        f = make_folder(tmp_path / str(n), mutate=mut)
        with pytest.raises(SchemaError, match=match):
            load_yaml(f / "benchmark.yaml")


def test_unknown_keys_and_layouts_are_schema_errors(tmp_path):
    f = make_folder(tmp_path / "a", mutate=lambda s: s.__setitem__("requires", {"pos": [], "streams": [], "grid": None, "t_bins": None}))
    with pytest.raises(SchemaError, match="requires"):
        load_yaml(f / "benchmark.yaml")
    f = make_folder(tmp_path / "b", mutate=lambda s: s["streams"]["video"].__setitem__("bridge_decode_allowed", False))
    with pytest.raises(SchemaError, match="bridge_decode_allowed"):
        load_yaml(f / "benchmark.yaml")
    f = make_folder(tmp_path / "c", mutate=lambda s: s["streams"]["video"].__setitem__("layout", "mat"))
    with pytest.raises(SchemaError, match="layout"):
        load_yaml(f / "benchmark.yaml")
    f = make_folder(tmp_path / "d", mutate=lambda s: s["manifest"]["columns"].__setitem__("offset_seconds", {"type": "float", "role": "window"}))
    with pytest.raises(SchemaError, match="role"):
        load_yaml(f / "benchmark.yaml")


# ----------------------------------------------------------------------------- schema validations, one per rule


def _schema_cases() -> list[tuple[str, Callable[[dict], None] | None, str]]:
    return [
        ("schema_version", lambda s: s.__setitem__("schema_version", 99), "schema_version"),
        ("name_charset", lambda s: s.__setitem__("name", "Bad-Name"), "name"),
        ("alias_is_own_name", lambda s: s.__setitem__("aliases", ["synth"]), "aliases"),
        ("readout_enum", lambda s: s.__setitem__("readout", "cls"), "readout"),
        ("changelog_version_mismatch", lambda s: s.__setitem__("protocol_version", 3), "changelog"),
        ("changelog_not_ascending", lambda s: s.__setitem__("changelog", [{"version": 2, "date": "d", "reason": "r", "invalidates": False}, {"version": 1, "date": "d", "reason": "r", "invalidates": False}]), "changelog"),
        ("source_kind", lambda s: s["source"].__setitem__("kind", "ftp"), "source.kind"),
        ("budget_clips_unknown_split", lambda s: s["budget"]["clips"].__setitem__("test", 1), "budget.clips"),
        ("budget_probe_minutes_missing_readout", lambda s: s["budget"].__setitem__("probe_minutes_ref", {"seq": 1}), "probe_minutes_ref"),
        ("media_column_not_declared", lambda s: s["streams"]["video"].__setitem__("media_column", "rel_path"), "media_column"),
        ("media_column_wrong_role", lambda s: s["manifest"]["columns"].__setitem__("rel_media", {"type": "str", "role": "extra"}), "role: media"),
        ("media_column_wrong_stream", lambda s: s["manifest"]["columns"]["rel_media"].__setitem__("stream", "audio"), "stream"),
        ("column_stream_without_media_role", lambda s: s["manifest"]["columns"].__setitem__("x", {"type": "int", "role": "extra", "stream": "video"}), "stream"),
        ("column_type", lambda s: s["manifest"]["columns"].__setitem__("x", {"type": "bool", "role": "extra"}), "type"),
        ("implicit_column_declared", lambda s: s["manifest"]["columns"].__setitem__("label", {"type": "str", "role": "extra"}), "implicit"),
        ("split_twice", lambda s: s["splits"].__setitem__("floor", ["val"]), "both"),
        ("split_train_empty", lambda s: s["splits"].__setitem__("train", []), "train"),
        ("folds_with_train", lambda s: s["splits"].__setitem__("folds", ["1", "2"]), "folds"),
        ("min_eval_coverage_range", lambda s: s["splits"].__setitem__("min_eval_coverage", 1.5), "min_eval_coverage"),
        ("stream_kind", lambda s: s["streams"]["video"].__setitem__("kind", "text"), "kind"),
        ("stream_layout", lambda s: s["streams"]["video"].__setitem__("layout", "zip"), "layout"),
        ("stream_root_undeclared", lambda s: s["streams"]["video"].__setitem__("root", "other_root"), "root"),
        ("layout_params_unknown", lambda s: s["streams"]["video"].__setitem__("layout_params", {"x": 1}), "layout_params"),
        ("tar_index_missing", lambda s: s["streams"]["video"].__setitem__("layout", "tar_index"), "tar_index"),
        ("tar_index_unused", lambda s: s["data"].__setitem__("tar_index", "index.json"), "tar_index"),
        ("views", lambda s: s["protocol"].__setitem__("views", "clips"), "views"),
        ("clip_seconds_non_positive", lambda s: s["protocol"].__setitem__("clip_seconds", 0.0), "clip_seconds"),
        ("train_draw_keys", lambda s: s["protocol"].__setitem__("train_draw", {"cap": 1}), "train_draw"),
        ("protocol_non_json", lambda s: s["protocol"].__setitem__("clip_seconds", float("nan")), "non-finite"),
        ("target_region_shape", lambda s: s["protocol"].__setitem__("target_region", [0.0, 0.0, 1.0]), "target_region"),
        ("target_region_order", lambda s: s["protocol"].__setitem__("target_region", [0.5, 0.0, 0.4, 1.0]), "target_region"),
        ("undeclared_protocol_key", lambda s: s["protocol"].__setitem__("offsets_seconds", [0.0]), "consumed by no registered metric"),
        # input_contract
        ("handed_over_missing_stream", lambda s: s["input_contract"].__setitem__("handed_over", {}), "handed_over"),
        ("handed_over_extra_stream", lambda s: s["input_contract"]["handed_over"].__setitem__("audio", "x"), "handed_over"),
        ("handed_over_empty_sentence", lambda s: s["input_contract"]["handed_over"].__setitem__("video", "  "), "sentence"),
        ("info_missing_stream", lambda s: s["input_contract"].__setitem__("info", {}), "info"),
        ("info_key_wrong_kind", lambda s: s["input_contract"]["info"].__setitem__("video", ["sample_rate"]), "sample_rate"),
        ("info_key_unknown", lambda s: s["input_contract"]["info"].__setitem__("video", ["codec"]), "codec"),
        ("info_rotation_not_guaranteeable", lambda s: s["input_contract"]["info"].__setitem__("video", ["rotation"]), "rotation"),
        ("fact_not_a_protocol_key", lambda s: s["input_contract"].__setitem__("facts", ["annotated_second"]), "top-level protocol key"),
        ("materialize_kind", lambda s: s["input_contract"].__setitem__("materialize", {"kind": "resample"}), "materialize"),
        # output_contract
        ("shape_pinned", lambda s: s["output_contract"].__setitem__("shape", "ragged"), "shape"),
        ("coordinates_pinned", lambda s: s["output_contract"].__setitem__("coordinates", "crop"), "coordinates"),
        ("target_frame_on_clf", lambda s: s["output_contract"].__setitem__("target_frame", "full_frame"), "target_frame"),
        ("target_frame_value", lambda s: s["output_contract"].__setitem__("target_frame", "crop"), "target_frame"),
        ("requires_pos_axis", lambda s: s["output_contract"]["requires"].__setitem__("pos", ["z"]), "pos"),
        ("requires_streams_unknown", lambda s: s["output_contract"]["requires"].__setitem__("streams", ["audio"]), "requires.streams"),
        ("requires_clf_grid", lambda s: s["output_contract"]["requires"].__setitem__("grid", {"h": 14, "w": 14, "min_cell_coverage": 0.5}), "clf"),
        # external
        ("external_not_list", lambda s: s.__setitem__("external", {}), "external"),
        ("external_keys", lambda s: s.__setitem__("external", [{"name": "pkg", "pip": "pkg==1"}]), "external"),
        ("external_empty_value", lambda s: s.__setitem__("external", [{"name": "pkg", "pip": "", "version_pin": "1", "used_for": "x"}]), "non-empty"),
        ("external_without_probe_py", lambda s: s.__setitem__("external", [{"name": "pkg", "pip": "pkg==1", "version_pin": "1", "used_for": "x"}]), "probe.py"),
        # metrics
        ("unknown_metric", lambda s: s["metrics"].__setitem__("list", ["acc", "accuracy"]), "accuracy"),
        ("metric_wrong_readout", lambda s: s["metrics"].__setitem__("list", ["acc", "seg_acc"]), "seg_acc"),
        ("floor_in_list", lambda s: s["metrics"].__setitem__("list", ["acc", "acc_floor"]), "acc_floor"),
        ("metric_in_floors", lambda s: s["metrics"].__setitem__("floors", ["acc"]), "kind"),
        ("control_in_list", lambda s: s["metrics"].__setitem__("controls", ["acc"]), "kind"),
        ("per_set_metric_without_sets", lambda s: s["metrics"].__setitem__("list", ["acc", "auc_pr_audible"]), "audible"),
        ("headline_not_listed", lambda s: s["metrics"].__setitem__("headline", "map"), "headline"),
        ("chance_not_listed", lambda s: s["metrics"].__setitem__("chance", {"map": 0.1}), "chance"),
        ("table_not_listed", lambda s: s["metrics"].__setitem__("table", ["acc", "map"]), "table"),
        ("table_without_headline", lambda s: s["metrics"].__setitem__("table", ["n_eval"]), "headline"),
        ("table_empty", lambda s: s["metrics"].__setitem__("table", []), "table"),
        ("table_not_a_list", lambda s: s["metrics"].__setitem__("table", "acc"), "table"),
        ("ci_kind", lambda s: s["metrics"].__setitem__("ci", {"acc": "jackknife"}), "ci"),
        ("ci_not_listed", lambda s: s["metrics"].__setitem__("ci", {"map": "binomial"}), "ci"),
        ("ci_not_admissible", lambda s: s["metrics"].__setitem__("ci", {"n_eval": "binomial"}), "no binomial interval"),
        ("bootstrap_required", lambda s: (s["metrics"].__setitem__("ci", {"acc": "bootstrap_clip"}), s["metrics"].__setitem__("bootstrap", None)), "bootstrap"),
        ("aggregation_without_folds", lambda s: s["metrics"].__setitem__("aggregation", {"over": "folds", "report": ["mean"]}), "aggregation"),
        ("readout_params_not_exact", lambda s: s["readout_params"].__setitem__("grid", 14), "readout_params"),
        ("targets_for_clf", lambda s: s.__setitem__("targets", {"train": {"kind": "dense_depth", "source": "derived_file", "root": "media_root", "file": "x.npy", "sha256": "0" * 64}, "val": {"kind": "dense_depth", "source": "derived_file", "root": "media_root", "file": "y.npy", "sha256": "0" * 64}}), "targets"),
        ("label_kind_enum", lambda s: s["manifest"]["label"].__setitem__("kind", "multilabel"), "kind"),
        ("label_sets_on_single", lambda s: s["manifest"]["label"].__setitem__("sets", {"a": ["label"]}), "sets"),
        ("label_column_null_for_single", lambda s: s["manifest"]["label"].__setitem__("column", None), "column"),
        ("label_class_key", lambda s: s["manifest"]["label"].__setitem__("class_key", "id"), "class_key"),
        ("classes_null_for_single", lambda s: (s["manifest"].__setitem__("classes", None), s["manifest"].__setitem__("classes_sha256", None)), "classes"),
        ("classes_sha_half_null", lambda s: s["manifest"].__setitem__("classes_sha256", None), "classes"),
        ("manifest_sha_not_hex", lambda s: s["manifest"].__setitem__("sha256", "zz"), "sha256"),
        ("clf_needs_class_labels", lambda s: (s["manifest"]["label"].__setitem__("kind", "box"), s["manifest"].__setitem__("classes", None), s["manifest"].__setitem__("classes_sha256", None)), "clf"),
        ("root_name_charset", lambda s: s["data"].__setitem__("roots", {"Media-Root": {"default": "/x"}}), "root"),
        ("derived_sha", lambda s: s["data"].__setitem__("derived", {"x.npy": "nothex"}), "derived"),
        ("top_level_not_mapping", None, "mapping"),
    ]


@pytest.mark.parametrize("case", _schema_cases(), ids=[c[0] for c in _schema_cases()])
def test_schema_validation_rules(tmp_path, case):
    name, mutate, match = case
    if mutate is None:
        (tmp_path / "benchmark.yaml").write_text("- just\n- a list\n")
        with pytest.raises(SchemaError, match=match):
            load_yaml(tmp_path / "benchmark.yaml")
        return
    folder = make_folder(tmp_path, mutate=mutate)
    with pytest.raises(SchemaError, match=match):
        load_yaml(folder / "benchmark.yaml")


def test_metrics_table_is_optional(tmp_path):
    """``metrics.table`` absent → ``None`` (the report shows all of ``list``); present → the declared subset."""
    assert load_yaml(make_folder(tmp_path / "plain") / "benchmark.yaml").metrics.table is None
    folder = make_folder(tmp_path / "table", mutate=lambda s: s["metrics"].__setitem__("table", ["acc"]))
    assert load_yaml(folder / "benchmark.yaml").metrics.table == ("acc",)


def test_score_hook_gets_the_reading_only_when_it_declares_it():
    """``ReadoutContext.scorer`` passes ``reading=`` to a hook that names it (or takes ``**kwargs``), never to
    one with the five-argument signature."""
    from types import SimpleNamespace

    from yavih.readouts.base import ReadoutContext

    seen = []

    def plain(bench, split, rows, outputs, targets):
        seen.append(("plain", split))
        return {}

    def routed(bench, split, rows, outputs, targets, *, reading=None):
        seen.append(("routed", reading))
        return {}

    def anykw(bench, split, rows, outputs, targets, **kw):
        seen.append(("anykw", kw.get("reading")))
        return {}

    for fn in (plain, routed, anykw):
        ctx = SimpleNamespace(probe_mod=SimpleNamespace(score=fn), bench=None, reading="av")
        ReadoutContext.scorer(ctx, ["test"], [0, 1], None)(None)
    assert seen == [("plain", "test"), ("routed", "av"), ("anykw", "av")]


def test_missing_protocol_key_of_a_probe_py_metric(tmp_path):
    """A folder metric's ``protocol_keys`` are required like a registry metric's."""
    folder = make_folder(tmp_path, probe_py=AVSYNC_PROBE, mutate=lambda s: s["metrics"].__setitem__("list", ["acc", "acc_tol1", "n_eval"]))
    with pytest.raises(SchemaError, match="offsets_seconds"):
        load_yaml(folder / "benchmark.yaml")


def test_facts_consume_protocol_keys_and_reach_the_view(tmp_path):
    """A label fact listed in ``input_contract.facts`` is consumed and copied into the view and
    ``Sample.meta["facts"]`` — and nothing else of the protocol is."""
    f = make_folder(tmp_path, mutate=lambda s: (s["protocol"].__setitem__("annotated_second", 5.0), s["input_contract"].__setitem__("facts", ["annotated_second"])))
    b = Benchmark.from_dir(f, data={"media_root": str(tmp_path / "media")})
    assert b.spec.input_contract.facts == ("annotated_second",)
    v = b.view()
    assert v.facts == {"annotated_second": 5.0} and v.name == "synth" and v.readout == "clf"
    assert v.protocol_hash == b.protocol_hash and v.streams == {"video": "video"}
    assert v.input_contract == b.spec.raw["input_contract"] and v.output_contract == b.spec.raw["output_contract"]
    assert not hasattr(v, "manifest") and not hasattr(v, "splits") and not hasattr(v, "roots")
    json.dumps(v.__dict__)  # JSON primitives only
    v.facts["annotated_second"] = 0.0  # a private copy: the spec is untouched
    assert b.spec.protocol["annotated_second"] == 5.0 and b.view().facts == {"annotated_second": 5.0}
    s = next(b.reader("val"))
    assert s.meta == {"facts": {"annotated_second": 5.0}}


def test_yaml_parse_errors(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        load_yaml(tmp_path / "nope.yaml")
    (tmp_path / "benchmark.yaml").write_text("a: [unclosed\n")
    with pytest.raises(SchemaError, match="yaml"):
        load_yaml(tmp_path / "benchmark.yaml")


def test_name_must_equal_folder_name(tmp_path):
    folder = make_folder(tmp_path, "synth", folder_name="other")
    load_yaml(folder / "benchmark.yaml")  # the yaml alone is fine
    with pytest.raises(SchemaError, match="folder"):
        Benchmark.from_dir(folder)
    with pytest.raises(UnknownBenchmark):
        Benchmark.from_dir(tmp_path)  # no benchmark.yaml


def test_load_yaml_does_not_touch_the_manifest(tmp_path):
    folder = make_folder(tmp_path)
    (folder / "manifest.csv.gz").unlink()
    spec = load_yaml(folder / "benchmark.yaml")
    assert spec.name == "synth" and spec.readout == "clf"
    with pytest.raises(ManifestError, match="missing"):
        Benchmark.from_dir(folder)


# ----------------------------------------------------------------------------- probe.py hooks and external

PROBE_PY = '''"""probe.py of a synthetic folder."""
from yavih.metrics.registry import MetricSpec


def _score(outputs, targets, protocol, split_meta):
    return 0.0


METRICS = {
    "my_score": MetricSpec(name="my_score", fn=_score, readouts=frozenset({"clf"}), protocol_keys=frozenset({"my_key"}), shape="scalar", kind="metric"),
}
PROTOCOL_KEYS = frozenset({"helper_key"})
'''


def _probe_mutate(s: dict[str, Any]) -> None:
    s["metrics"]["list"] = ["acc", "my_score", "n_eval"]
    s["protocol"].update({"my_key": 1, "helper_key": "x"})


def test_probe_py_metrics_overlay_and_protocol_keys(tmp_path):
    f = make_folder(tmp_path, mutate=_probe_mutate, probe_py=PROBE_PY)
    b = Benchmark.from_dir(f)
    assert set(b.metric_specs) == {"acc", "my_score", "n_eval", "acc_floor"}
    assert b.metric_specs["my_score"].fn(None, None, None, None) == 0.0
    assert b.spec.folder_metrics == ("my_score",)
    assert b.probe_sha256 == hashing.source_sha256(PROBE_PY) and b.protocol_hash == _expected_protocol_hash(b)
    assert b.folder_source == {"probe": b.probe_sha256, "reader": None}
    assert b.probe_module().PROTOCOL_KEYS == {"helper_key"}
    # probe.py is provenance, never identity (yavih-protocol-5): a comment edit leaves its digest, a code edit moves
    # the digest, and neither moves protocol_hash — behaviour is pinned by protocol_version and the conformance fixture
    (f / "probe.py").write_text(PROBE_PY + "# comment\n")
    assert Benchmark.from_dir(f).folder_source == b.folder_source
    (f / "probe.py").write_text(PROBE_PY + "EXTRA = 1\n")
    b2 = Benchmark.from_dir(f)
    assert b2.folder_source["probe"] != b.folder_source["probe"]
    assert b2.protocol_hash == b.protocol_hash and b2.input_hash == b.input_hash and b2.scoring_hash == b.scoring_hash
    # without probe.py the folder metric is unknown and the helper key unconsumed
    (f / "probe.py").unlink()
    with pytest.raises(SchemaError, match="my_score"):
        load_yaml(f / "benchmark.yaml")
    # METRICS' protocol keys are required; PROTOCOL_KEYS are required and count as consumed
    f2 = make_folder(tmp_path / "nokey", mutate=lambda s: (_probe_mutate(s), s["protocol"].pop("my_key")), probe_py=PROBE_PY)
    with pytest.raises(SchemaError, match="my_key"):
        load_yaml(f2 / "benchmark.yaml")
    f3 = make_folder(tmp_path / "nohelper", mutate=lambda s: (_probe_mutate(s), s["protocol"].pop("helper_key")), probe_py=PROBE_PY)
    with pytest.raises(SchemaError, match="helper_key"):
        load_yaml(f3 / "benchmark.yaml")


def test_probe_py_may_replace_a_registry_metric_only_keeping_its_shape(tmp_path):
    ok = PROBE_PY.replace('"my_score": MetricSpec(name="my_score"', '"acc": MetricSpec(name="acc"').replace("frozenset({\"my_key\"})", "frozenset()")
    f = make_folder(tmp_path / "ok", mutate=lambda s: s["protocol"].update({"helper_key": "x"}), probe_py=ok)
    b = Benchmark.from_dir(f)
    assert b.metric_specs["acc"].fn(None, None, None, None) == 0.0 and b.spec.folder_metrics == ("acc",)
    assert b.spec.metrics.ci == {"acc": "binomial"}
    bad = ok.replace('kind="metric"', 'kind="disclosure"')
    f2 = make_folder(tmp_path / "bad", mutate=lambda s: s["protocol"].update({"helper_key": "x"}), probe_py=bad)
    with pytest.raises(SchemaError, match="readouts/shape/kind"):
        load_yaml(f2 / "benchmark.yaml")


def test_probe_py_import_errors_are_schema_errors(tmp_path):
    f = make_folder(tmp_path, probe_py="import nonexistent_package_xyz\n")
    with pytest.raises(SchemaError, match="probe.py.*cannot import"):
        load_yaml(f / "benchmark.yaml")
    f2 = make_folder(tmp_path / "b", probe_py="METRICS = {'x': 1}\n")
    with pytest.raises(SchemaError, match="MetricSpec"):
        load_yaml(f2 / "benchmark.yaml")


def test_external_block_with_probe_py(tmp_path, monkeypatch):
    ext = [{"name": "vggsounder_fake_pkg", "pip": "vggsounder_fake_pkg==1.0", "version_pin": "1.0", "used_for": "scoring"}]
    f = make_folder(tmp_path, mutate=lambda s: (_probe_mutate(s), s.__setitem__("external", ext)), probe_py=PROBE_PY)
    b = Benchmark.from_dir(f)
    assert b.spec.external == (bm.ExternalSpec(**ext[0]),)
    assert bench_info(str(f), scan=False)["external"] == ext
    # duplicate names refused; the version pin is hashed
    f2 = make_folder(tmp_path / "dup", mutate=lambda s: (_probe_mutate(s), s.__setitem__("external", ext * 2)), probe_py=PROBE_PY)
    with pytest.raises(SchemaError, match="duplicate"):
        load_yaml(f2 / "benchmark.yaml")
    f3 = make_folder(tmp_path / "pin", mutate=lambda s: (_probe_mutate(s), s.__setitem__("external", [{**ext[0], "version_pin": "1.1"}])), probe_py=PROBE_PY)
    assert Benchmark.from_dir(f3).protocol_hash != b.protocol_hash
    # a folder metric naming an external package must find it declared
    named = PROBE_PY.replace('kind="metric")', 'kind="metric", external="other_pkg")') if "external" in registry.MetricSpec.__dataclass_fields__ else None
    if named is not None:
        f4 = make_folder(tmp_path / "named", mutate=lambda s: (_probe_mutate(s), s.__setitem__("external", ext)), probe_py=named)
        with pytest.raises(SchemaError, match="other_pkg"):
            load_yaml(f4 / "benchmark.yaml")


def _manifest_cases() -> list[tuple[str, dict[str, Any], str]]:
    return [
        ("sha_mismatch", {"mutate": lambda s: s["manifest"].__setitem__("sha256", "1" * 64)}, "manifest sha256"),
        ("classes_sha_mismatch", {"mutate": lambda s: s["manifest"].__setitem__("classes_sha256", "1" * 64)}, "classes sha256"),
        ("duplicate_clip_id", {"rows": BASE_ROWS[:5] + [["v0", "val", "b", "val/v1.mp4"]]}, "duplicate"),
        ("clip_id_charset", {"rows": BASE_ROWS[:5] + [["v/1", "val", "b", "val/v1.mp4"]]}, "charset"),
        ("media_charset", {"rows": BASE_ROWS[:5] + [["v1", "val", "b", "val/v 1.mp4"]]}, "charset"),
        ("media_absolute", {"rows": BASE_ROWS[:5] + [["v1", "val", "b", "/abs/v1.mp4"]]}, "relative"),
        ("media_dotdot", {"rows": BASE_ROWS[:5] + [["v1", "val", "b", "../v1.mp4"]]}, "relative"),
        ("media_empty_required", {"rows": BASE_ROWS[:5] + [["v1", "val", "b", ""]]}, "empty media"),
        ("unassigned_split", {"rows": BASE_ROWS + [["x0", "test", "a", "test/x0.mp4"]]}, "assigned to no role"),
        ("declared_split_without_rows", {"rows": BASE_ROWS[:4]}, "has no rows"),
        ("undeclared_column", {"header": BASE_HEADER + ["extra"], "rows": [r + ["1"] for r in BASE_ROWS]}, "not declared"),
        ("missing_declared_column", {"header": ["clip_id", "split", "label"], "rows": [r[:3] for r in BASE_ROWS]}, "missing from the header"),
        ("ragged_row", {"rows": BASE_ROWS[:5] + [["v1", "val", "b"]]}, "fields"),
        ("unknown_label", {"rows": BASE_ROWS[:5] + [["v1", "val", "zzz", "val/v1.mp4"]]}, "not in classes.csv"),
        ("empty_label", {"rows": BASE_ROWS[:5] + [["v1", "val", "", "val/v1.mp4"]]}, "empty label cell"),
        ("multi_in_single", {"rows": BASE_ROWS[:5] + [["v1", "val", "a;b", "val/v1.mp4"]]}, "exactly one class"),
        ("classes_not_contiguous", {"classes": [["0", "a"], ["2", "b"]]}, "contiguous"),
        ("classes_duplicate_name", {"classes": [["0", "a"], ["1", "a"]]}, "duplicate class names"),
        ("classes_bad_header", {"classes": BASE_CLASSES, "classes_header": ["index", "display_name"]}, "header"),
        ("classes_duplicate_mid", {"classes": [["0", "a", "/m/1"], ["1", "b", "/m/1"]]}, "mids"),
        ("empty_manifest", {"rows": []}, "no rows"),
    ]


@pytest.mark.parametrize("case", _manifest_cases(), ids=[c[0] for c in _manifest_cases()])
def test_manifest_validation_rules(tmp_path, case):
    _, kwargs, match = case
    folder = make_folder(tmp_path, **kwargs)
    with pytest.raises(ManifestError, match=match):
        Benchmark.from_dir(folder)


def test_allow_unknown_and_empty_ok(tmp_path):
    rows = BASE_ROWS[:4] + [["v0", "val", "zzz", "val/v0.mp4"], ["v1", "val", "", "val/v1.mp4"]]
    folder = make_folder(tmp_path, rows=rows, mutate=lambda s: (s["manifest"]["label"].update({"allow_unknown": True, "empty_ok": True})))
    b = Benchmark.from_dir(folder)
    assert b.unknown_labels == 1
    assert b.manifest("val").label_ids == [None, None]
    assert b.manifest("train").label_ids == [0, 1, 0, 1]


def test_classes_with_mid_and_class_key_index(tmp_path):
    classes = [["0", "a", "/m/a"], ["1", "b", "/m/b"]]
    rows = [[r[0], r[1], {"a": "/m/a", "b": "/m/b"}[r[2]], r[3]] for r in BASE_ROWS]
    folder = make_folder(tmp_path, rows=rows, classes=classes, mutate=lambda s: s["manifest"]["label"].__setitem__("class_key", "mid"))
    b = Benchmark.from_dir(folder)
    assert b.manifest("train").label_ids == [0, 1, 0, 1]
    assert b.classes()[1] == {"index": 1, "name": "b", "mid": "/m/b"}
    rows_idx = [[r[0], r[1], {"a": "0", "b": "1"}[r[2]], r[3]] for r in BASE_ROWS]
    folder = make_folder(tmp_path / "idx", rows=rows_idx, mutate=lambda s: s["manifest"]["label"].__setitem__("class_key", "index"))
    assert Benchmark.from_dir(folder).manifest("val").label_ids == [0, 1]
    rows_bad = [[r[0], r[1], "7", r[3]] for r in BASE_ROWS]
    folder = make_folder(tmp_path / "bad", rows=rows_bad, mutate=lambda s: s["manifest"]["label"].__setitem__("class_key", "index"))
    with pytest.raises(ManifestError, match="not in classes.csv"):
        Benchmark.from_dir(folder)


def test_typed_extra_columns(tmp_path):
    header = BASE_HEADER + ["offset_bin", "score", "note"]
    rows = [r + [str(i), f"{i * 0.5}", "n"] for i, r in enumerate(BASE_ROWS)]
    rows[5][5] = ""  # empty typed cell → None

    def mutate(s):
        s["manifest"]["columns"].update({"offset_bin": {"type": "int", "role": "extra"}, "score": {"type": "float", "role": "extra"}, "note": {"type": "str", "role": "extra"}})

    b = Benchmark.from_dir(make_folder(tmp_path, header=header, rows=rows, mutate=mutate))
    m = b.manifest("val")
    assert m.extra[0] == {"rel_media": "val/v0.mp4", "offset_bin": 4, "score": 2.0, "note": "n"}
    assert m.extra[1] == {"rel_media": "val/v1.mp4", "offset_bin": 5, "score": None, "note": "n"}
    rows[0][4] = "x"
    with pytest.raises(ManifestError, match="cannot parse"):
        Benchmark.from_dir(make_folder(tmp_path / "bad", header=header, rows=rows, mutate=mutate))


# ----------------------------------------------------------------------------- label kinds


def test_multi_label_kind(tmp_path):
    rows = [
        ["t0", "train", "a;b", "train/t0.wav"],
        ["t1", "train", "b", "train/t1.wav"],
        ["t2", "train", "a", "train/t2.wav"],
        ["t3", "train", "b;a", "train/t3.wav"],
        ["v0", "val", "a", "val/v0.wav"],
        ["v1", "val", "a;b;a", "val/v1.wav"],
    ]

    def mutate(s):
        s["manifest"]["label"]["kind"] = "multi"
        s["manifest"]["columns"]["rel_media"]["stream"] = "audio"
        set_streams(s, {"audio": dir_stream("audio", "rel_media")}, {"audio": ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]})
        s["metrics"].update({"headline": "map", "list": ["map", "effective_pos_classes", "n_eval"], "floors": ["map_floor"], "chance": {}, "ci": {}})

    b = Benchmark.from_dir(make_folder(tmp_path, "fsd", rows=rows, mutate=mutate))
    assert b.manifest("train").label_ids == [[0, 1], [1], [0], [1, 0]]
    assert b.manifest("val").label_ids == [[0], [0, 1]]  # deduped, order kept
    assert set(b.metric_specs) == {"map", "effective_pos_classes", "n_eval", "map_floor"}
    assert b.spec.input_contract.info["audio"] == ("duration_s", "sample_rate", "n_samples", "channels", "container", "bytes")


def test_multi_sets_kind_vggsounder_shape(tmp_path):
    header = ["clip_id", "label", "labels_a", "labels_v", "labels_av", "split", "rel_video", "rel_audio"]
    rows = [
        ["t0", "a", "", "", "", "train", "t0.mp4", "t0.wav"],
        ["t1", "b", "", "", "", "train", "t1.mp4", "t1.wav"],
        ["t2", "a", "", "", "", "train", "t2.mp4", "t2.wav"],
        ["t3", "b", "", "", "", "train", "t3.mp4", "t3.wav"],
        ["e0", "", "a", "", "b", "test", "e0.mp4", "e0.wav"],
        ["e1", "", "", "b", "", "test", "e1.mp4", "e1.wav"],
    ]

    def mutate(s):
        s["manifest"]["label"].update({"kind": "multi_sets", "sets": {"audible": ["labels_a", "labels_av"], "visible": ["labels_v", "labels_av"], "all": ["labels_a", "labels_v", "labels_av"]}, "train_set": None, "empty_ok": True})
        s["manifest"]["columns"] = {"rel_video": {"type": "str", "role": "media", "stream": "video"}, "rel_audio": {"type": "str", "role": "media", "stream": "audio"}}
        set_streams(s, {"video": dir_stream("video", "rel_video"), "audio": dir_stream("audio", "rel_audio")})
        s["splits"]["eval"] = ["test"]
        s["budget"]["clips"] = {"train": 4, "test": 2}
        s["manifest"]["upstream"]["curation"] = {"total_upstream": 12, "per_split": {"train": {"listed": 4, "staged": 4, "missing": 0}, "test": {"listed": 2, "staged": 2, "missing": 0}}, "excluded": {"leakage": 6}, "short_clips": 0, "leakage_audit": {"against": "VGGSound train ytids", "excluded": 6}}
        s["metrics"].update({"headline": "auc_pr_audible", "list": ["auc_pr_audible", "auc_roc_audible", "auc_pr_visible", "auc_roc_visible", "auc_pr_all", "auc_roc_all", "n_eval"], "floors": ["auc_pr_audible_floor", "auc_pr_visible_floor", "auc_pr_all_floor"], "chance": {}, "ci": {}})

    b = Benchmark.from_dir(make_folder(tmp_path, "vggs", header=header, rows=rows, mutate=mutate))
    tr, ev = b.manifest("train"), b.manifest("test")
    assert tr.label_ids == [[0], [1], [0], [1]]
    assert tr.label_sets == [{"audible": [], "visible": [], "all": []}] * 4
    assert ev.label_ids == [[], []]
    assert ev.label_sets == [{"audible": [0, 1], "visible": [1], "all": [0, 1]}, {"audible": [], "visible": [1], "all": [1]}]
    assert ev.media[0] == {"video": "e0.mp4", "audio": "e0.wav"} and ev.media_key[0] == "audio=e0.wav|video=e0.mp4"
    assert b.metric_specs["auc_pr_all"].fn.bound == {"set_name": "all"}  # expanded at load
    assert b.metric_specs["auc_pr_all_floor"].kind == "floor"
    assert b.spec.manifest.upstream.curation.invariant_holds()
    rows2 = [[r[0], "", r[2] or ("a" if r[5] == "train" else r[2]), r[3], r[4], r[5], r[6], r[7]] for r in rows]
    b2 = Benchmark.from_dir(make_folder(tmp_path / "ts", "vggs", header=header, rows=rows2, mutate=lambda s: (mutate(s), s["manifest"]["label"].__setitem__("train_set", "audible"))))
    assert b2.manifest("train").label_ids == [[0]] * 4
    with pytest.raises(SchemaError, match="sets"):
        load_yaml(make_folder(tmp_path / "nosets", "vggs", header=header, rows=rows, mutate=lambda s: (mutate(s), s["manifest"]["label"].__setitem__("sets", None))) / "benchmark.yaml")
    with pytest.raises(SchemaError, match="train_set"):
        load_yaml(make_folder(tmp_path / "badts", "vggs", header=header, rows=rows, mutate=lambda s: (mutate(s), s["manifest"]["label"].__setitem__("train_set", "nope"))) / "benchmark.yaml")
    with pytest.raises(ManifestError, match="empty label cell"):
        Benchmark.from_dir(make_folder(tmp_path / "noempty", "vggs", header=header, rows=rows, mutate=lambda s: (mutate(s), s["manifest"]["label"].__setitem__("empty_ok", False))))


def _box_spec_mutate(s: dict[str, Any]) -> None:
    s["readout"] = "box"
    s["budget"]["probe_minutes_ref"] = {"box": 1}
    s["manifest"]["label"].update({"kind": "box", "column": "box"})
    s["manifest"]["classes"] = None
    s["manifest"]["classes_sha256"] = None
    s["manifest"]["columns"] = {"rel_video": {"type": "str", "role": "media", "stream": "video"}, "rel_audio": {"type": "str", "role": "media", "stream": "audio"}, "class_name": {"type": "str", "role": "extra"}}
    set_streams(s, {"video": dir_stream("video", "rel_video"), "audio": dir_stream("audio", "rel_audio")}, {"video": ["duration_s", "fps", "width", "height", "container", "bytes"]})
    s["input_contract"]["facts"] = ["annotated_second"]
    s["protocol"].update({"box_population": "single", "gt_geometry": "full_frame", "heatmap_grid": 14, "annotated_second": 5.0})
    s["metrics"].update({"headline": "iou_mean", "list": ["iou_mean", "acc@0.5", "acc@0.3", "center_in_gt", "n_eval", "n_excluded_degenerate"], "floors": ["iou_floor"], "controls": ["iou_shuffled", "audio_blind"], "chance": {}, "ci": {}})
    s["output_contract"].update({"target_frame": "full_frame", "requires": {"pos": ["y", "x"], "streams": ["video", "audio"], "grid": {"h": 14, "w": 14, "min_cell_coverage": 0.25}, "t_bins": None}})
    s["readout_params"] = {"grid": 14, "visual_stream": "video", "conditioning_stream": "audio", "shuffled_control": True, "shuffle_seed": 20260913}


BOX_HEADER = ["clip_id", "split", "box", "rel_video", "rel_audio", "class_name"]
BOX_ROWS = [
    ["t0", "train", "0.1|0.2|0.5|0.6", "t0.mp4", "t0.wav", "dog"],
    ["t1", "train", "0|0|1|1", "t1.mp4", "t1.wav", "dog"],
    ["t2", "train", "0.3|0.3|0.4|0.9", "t2.mp4", "t2.wav", "cat"],
    ["t3", "train", "0.2|0.1|0.3|0.2", "t3.mp4", "t3.wav", "cat"],
    ["v0", "val", "0.5|0.5|0.6|0.6", "v0.mp4", "v0.wav", "dog"],
    ["v1", "val", "0.1|0.1|0.9|0.9", "v1.mp4", "v1.wav", "cat"],
]


def test_box_kind_vggss_shape(tmp_path):
    folder = make_folder(tmp_path, "vggss_box", header=BOX_HEADER, rows=BOX_ROWS, classes=None, mutate=_box_spec_mutate)
    b = Benchmark.from_dir(folder)
    assert b.spec.readout == "box" and b.classes() == [] and b.num_classes == 0
    m = b.manifest("val")
    assert m.label_ids == [None, None]
    assert m.boxes.dtype == np.float32
    np.testing.assert_allclose(m.boxes, [[0.5, 0.5, 0.6, 0.6], [0.1, 0.1, 0.9, 0.9]], rtol=1e-6)
    assert b.manifest().boxes.shape == (6, 4)
    assert set(b.spec.protocol_extra_keys) == {"box_population", "gt_geometry", "heatmap_grid", "annotated_second"}
    assert b.spec.output_contract.target_frame == "full_frame" and b.spec.requires.grid["min_cell_coverage"] == 0.25
    assert b.spec.target_region == (0.0, 0.0, 1.0, 1.0)  # absent = the whole frame
    assert b.view().facts == {"annotated_second": 5.0}
    with pytest.raises(NoTargets):
        b.targets("train")
    bad = make_folder(tmp_path / "bad", "vggss_box", header=BOX_HEADER, rows=BOX_ROWS[:5] + [["v1", "val", "0.1|0.1|0.9", "v1.mp4", "v1.wav", "cat"]], classes=None, mutate=_box_spec_mutate)
    with pytest.raises(ManifestError, match="x1\\|y1\\|x2\\|y2"):
        Benchmark.from_dir(bad)
    for mut, match in [
        (lambda s: s["readout_params"].__setitem__("grid", 7), "requires.grid"),
        (lambda s: s["readout_params"].__setitem__("conditioning_stream", "video"), "differ"),
        (lambda s: s["output_contract"]["requires"].__setitem__("grid", None), "requires.grid"),
        (lambda s: s["output_contract"].__setitem__("target_frame", None), "target_frame"),
        (lambda s: s["input_contract"].__setitem__("facts", []), "annotated_second"),  # box reads it itself? no: consumed by facts or readout
    ]:
        f = make_folder(tmp_path / match.replace(".", "_").replace(" ", "_"), "vggss_box", header=BOX_HEADER, rows=BOX_ROWS, classes=None, mutate=lambda s, mut=mut: (_box_spec_mutate(s), mut(s)))
        if match == "annotated_second":  # the box readout lists annotated_second itself, so dropping the fact is legal
            load_yaml(f / "benchmark.yaml")
            continue
        with pytest.raises(SchemaError, match=match):
            load_yaml(f / "benchmark.yaml")


def _seq_spec_mutate(s: dict[str, Any]) -> None:
    s["readout"] = "seq"
    s["budget"]["probe_minutes_ref"] = {"seq": 1}
    s["manifest"]["label"]["kind"] = "segments"
    s["manifest"]["columns"] = {
        "rel_media": {"type": "str", "role": "media", "stream": ["video", "audio"]},  # one mp4 feeds both streams (ave)
        "event_start": {"type": "int", "role": "label_aux"},
        "event_end": {"type": "int", "role": "label_aux"},
    }
    tar = {"layout": "tar_index"}
    set_streams(s, {"video": dir_stream("video", "rel_media", **tar), "audio": dir_stream("audio", "rel_media", **tar)})
    s["data"]["tar_index"] = "index.json"
    s["input_contract"]["facts"] = ["num_segments", "segment_seconds"]
    s["protocol"].update({"num_segments": 10, "segment_seconds": 1.0, "segments_scored": 10, "segment_offset": 0, "background_index": 2})
    s["metrics"].update({"headline": "seg_acc", "list": ["seg_acc", "clip_acc", "seg_acc_boundary", "n_boundary", "n_eval"], "floors": ["seg_acc_floor", "seg_acc_boundary_floor"], "chance": {"seg_acc": 0.3}, "ci": {"seg_acc": "bootstrap_clip", "seg_acc_boundary": "bootstrap_clip"}})
    s["output_contract"]["requires"] = {"pos": ["t"], "streams": ["video"], "grid": None, "t_bins": 8}
    s["readout_params"] = {"T": 10}
    s["splits"]["unused"] = ["dev"]


SEQ_HEADER = ["clip_id", "split", "label", "rel_media", "event_start", "event_end"]
SEQ_ROWS = [
    ["y0#0", "train", "a", "y0.mp4", "0", "10"],
    ["y0#1", "train", "b", "y0.mp4", "3", "5"],
    ["y1#0", "train", "a", "y1.mp4", "2", "9"],
    ["y2#0", "train", "b", "y2.mp4", "0", "4"],
    ["y3#0", "val", "a", "y3.mp4", "1", "2"],
    ["y4#0", "val", "b", "y4.mp4", "0", "10"],
    ["y5#0", "dev", "a", "y5.mp4", "0", "10"],
]
SEQ_CLASSES = [["0", "a"], ["1", "b"], ["2", "background"]]


def test_segments_kind_ave_shape(tmp_path):
    folder = make_folder(tmp_path, "ave", header=SEQ_HEADER, rows=SEQ_ROWS, classes=SEQ_CLASSES, mutate=_seq_spec_mutate)
    b = Benchmark.from_dir(folder)
    m = b.manifest("train")
    assert m.label_ids == [0, 1, 0, 1]
    assert m.segments.tolist() == [[0, 10], [3, 5], [2, 9], [0, 4]]
    assert m.extra[1] == {"rel_media": "y0.mp4", "event_start": 3, "event_end": 5}
    assert b.spec.manifest.columns["rel_media"].streams() == ("video", "audio")
    # rows of one video share a media_key (features byte-identical by construction, ave)
    assert m.media_key[0] == m.media_key[1] == "audio=y0.mp4|video=y0.mp4"
    assert "audio" in m.media[0]
    assert b.spec.splits.role_of("dev") == "unused" and b.n_rows("dev") == 1
    assert b.manifest("dev").label_ids == [0]
    assert b.view().facts == {"num_segments": 10, "segment_seconds": 1.0}
    with pytest.raises(KeyError, match="not read"):
        next(b.reader("dev"))
    assert registry.protocol_keys_for(b.metric_specs.values()) <= set(b.spec.protocol)
    bad = make_folder(tmp_path / "bad", "ave", header=SEQ_HEADER, rows=SEQ_ROWS[:6] + [["y5#0", "dev", "a", "y5.mp4", "5", "5"]], classes=SEQ_CLASSES, mutate=_seq_spec_mutate)
    with pytest.raises(ManifestError, match="event_start < event_end"):
        Benchmark.from_dir(bad)
    for mut, match in [
        (lambda s: s["manifest"]["columns"].pop("event_end"), "event_end"),
        (lambda s: s["readout_params"].__setitem__("T", 8), "segments_scored"),  # T = segments_scored
        (lambda s: s["output_contract"]["requires"].__setitem__("t_bins", 11), "cannot exceed"),
        (lambda s: s["output_contract"]["requires"].__setitem__("t_bins", None), "t_bins"),
        (lambda s: s["protocol"].pop("segment_offset"), "segment_offset"),
        (lambda s: s["protocol"].__setitem__("background_index", 7), None),
    ]:
        f = make_folder(tmp_path / (match.replace(".", "_").replace(" ", "_") if match else "bi"), "ave", header=SEQ_HEADER, rows=SEQ_ROWS, classes=SEQ_CLASSES, mutate=lambda s, mut=mut: (_seq_spec_mutate(s), mut(s)))
        if match is None:
            with pytest.raises(ManifestError, match="background_index"):
                Benchmark.from_dir(f)
        else:
            with pytest.raises(SchemaError, match=match):
                load_yaml(f / "benchmark.yaml")


def _grid_spec_mutate(s: dict[str, Any]) -> None:
    s["readout"] = "grid"
    s["budget"]["probe_minutes_ref"] = {"grid": 1}
    s["manifest"]["label"].update({"kind": "none", "column": None})
    s["manifest"]["classes"] = None
    s["manifest"]["classes_sha256"] = None
    s["manifest"]["columns"] = {"rel_media": {"type": "str", "role": "media", "stream": "image"}, "eval_row": {"type": "int", "role": "extra"}}
    set_streams(s, {"image": dir_stream("image", "rel_media", root="derived_root")}, {"image": ["width", "height", "container", "bytes"]})
    s["data"]["roots"] = {"derived_root": {"default": "${data.root}/nyuv2_derived"}}
    s["targets"] = {
        "train": {"kind": "dense_depth", "source": "derived_file", "root": "derived_root", "file": "train_depth.npy", "sha256": "0" * 64, "index_column": None},
        "val": {"kind": "dense_depth", "source": "derived_file", "root": "derived_root", "file": "eval_depth.npy", "sha256": "0" * 64, "index_column": "eval_row"},
    }
    s["input_contract"]["facts"] = ["target_region"]
    s["protocol"].update({"max_depth": 10.0, "target_size": 4, "depth_param": "log", "source_size": [480, 640], "target_region": [0.0, 0.125, 1.0, 0.875]})
    s["metrics"].update({"headline": "d1", "list": ["d1", "d2", "d3", "absrel", "rmse", "n_eval"], "floors": ["d1_floor"], "chance": {}, "ci": {}})
    s["output_contract"].update({"target_frame": "full_frame", "requires": {"pos": ["y", "x"], "streams": ["image"], "grid": {"h": 14, "w": 14, "min_cell_coverage": 0.5}, "t_bins": None}})
    s["readout_params"] = {"h": 14, "w": 14, "out": 1, "visual_stream": "image"}


GRID_HEADER = ["clip_id", "split", "rel_media", "eval_row"]
GRID_ROWS = [
    ["s0_f0", "train", "rgb/train/s0_f0.png", ""],
    ["s0_f1", "train", "rgb/train/s0_f1.png", ""],
    ["s1_f0", "train", "rgb/train/s1_f0.png", ""],
    ["s1_f1", "train", "rgb/train/s1_f1.png", ""],
    ["e0", "val", "rgb/eval/e0.png", "2"],
    ["e1", "val", "rgb/eval/e1.png", "0"],
]


def _write_grid_data(derived: Path) -> tuple[np.ndarray, np.ndarray]:
    derived.mkdir(parents=True, exist_ok=True)
    train = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    ev = (np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4) + 1000)
    np.save(derived / "train_depth.npy", train)
    np.save(derived / "eval_depth.npy", ev)
    return train, ev


def _grid_folder(tmp_path: Path, derived: Path) -> Path:
    def mutate(s):
        _grid_spec_mutate(s)
        s["targets"]["train"]["sha256"] = _sha((derived / "train_depth.npy").read_bytes())
        s["targets"]["val"]["sha256"] = _sha((derived / "eval_depth.npy").read_bytes())

    return make_folder(tmp_path, "nyuv2_depth", header=GRID_HEADER, rows=GRID_ROWS, classes=None, mutate=mutate)


def test_grid_kind_nyuv2_shape_and_target_reader(tmp_path, yavih_home, monkeypatch):
    derived = tmp_path / "derived"
    train, ev = _write_grid_data(derived)
    folder = _grid_folder(tmp_path, derived)
    with_roots(monkeypatch, "nyuv2_depth", derived_root=derived)
    b = Benchmark.from_dir(folder)
    assert b.spec.readout == "grid" and b.spec.targets is not None
    assert b.spec.target_region == (0.0, 0.125, 1.0, 0.875) and b.view().facts == {"target_region": [0.0, 0.125, 1.0, 0.875]}
    m = b.manifest("train")
    assert m.label_ids == [None] * 4 and m.media[0] == {"image": "rgb/train/s0_f0.png"}
    assert b.manifest("val").extra == [{"rel_media": "rgb/eval/e0.png", "eval_row": 2}, {"rel_media": "rgb/eval/e1.png", "eval_row": 0}]
    tr = b.spec.stream_for("image", "train")
    assert (tr["layout"], tr["root"], tr["media_column"]) == ("dir", "derived_root", "rel_media")
    # TargetReader: memmap, sha verified once, rows in manifest order through index_column
    tt = b.targets("train")
    assert tt.kind == "dense_depth" and tt.shape == (4, 4, 4) and tt.dtype == np.float32 and len(tt) == 4
    np.testing.assert_array_equal(tt.rows([3, 0]), train[[3, 0]])
    te = b.targets("val")
    assert te.shape == (2, 4, 4)
    np.testing.assert_array_equal(te.rows(np.arange(2)), ev[[2, 0]])
    np.testing.assert_array_equal(te.rows([1]), ev[[0]])
    with pytest.raises(IndexError):
        te.rows([2])
    status = json.loads(data_status_path("nyuv2_depth", b.resolved_roots()).read_text())  # keyed by the roots
    assert any(v["sha256"] == b.spec.targets["train"].sha256 for v in status["targets"].values())
    # index_column is hashed (it decides which memmap row a manifest row is scored against)
    assert hashing.protocol_targets(b.spec.raw["targets"]) == [
        {"split": "train", "kind": "dense_depth", "file": "train_depth.npy", "sha256": b.spec.targets["train"].sha256, "index_column": None},
        {"split": "val", "kind": "dense_depth", "file": "eval_depth.npy", "sha256": b.spec.targets["val"].sha256, "index_column": "eval_row"},
    ]
    with pytest.raises(NoTargets):
        b.targets("dev")


def test_target_reader_errors(tmp_path, yavih_home, monkeypatch):
    derived = tmp_path / "derived"
    _write_grid_data(derived)
    folder = _grid_folder(tmp_path, derived)
    with_roots(monkeypatch, "nyuv2_depth", derived_root=derived)
    b = Benchmark.from_dir(folder)
    np.save(derived / "train_depth.npy", np.ones((4, 4, 4), dtype=np.float32))
    bm._TARGET_SHA_CACHE.clear()
    with pytest.raises(ManifestError, match="sha256"):
        b.targets("train")
    (derived / "eval_depth.npy").unlink()
    with pytest.raises(DataNotReady, match="missing"):
        b.targets("val")


def test_grid_schema_rules(tmp_path):
    for mut, match in [
        (lambda s: s.__setitem__("targets", None), "targets"),
        (lambda s: s["targets"].pop("val"), "missing targets"),
        (lambda s: s["targets"]["val"].__setitem__("index_column", "nope"), "index_column"),
        (lambda s: s["targets"]["val"].__setitem__("root", "nope"), "root"),
        (lambda s: s["targets"]["val"].__setitem__("kind", "sparse"), "kind"),
        (lambda s: s["manifest"]["label"].update({"kind": "single", "column": "rel_media"}), "classes|implicit|none"),
        (lambda s: s["readout_params"].__setitem__("h", 7), "requires.grid"),
        (lambda s: s["readout_params"].__setitem__("visual_stream", "video"), "visual_stream"),
        (lambda s: s["output_contract"].__setitem__("target_frame", None), "target_frame"),
        (lambda s: s["protocol"].__setitem__("target_region", [0.0, 0.875, 1.0, 0.125]), "target_region"),
        (lambda s: s["streams"]["image"].__setitem__("layout", {"train": "dir", "eval": "npy_memmap"}), "layout"),
        (lambda s: s["streams"]["image"].__setitem__("root", {"train": "derived_root"}), "no entry for split"),
        (lambda s: s["streams"]["image"].__setitem__("root", {"train": "derived_root", "test": "derived_root"}), "split values or role words"),
        (lambda s: s["streams"]["image"].__setitem__("root", {"train": "derived_root", "eval": "derived_root"}), None),  # per-split, role words
    ]:
        f = make_folder(tmp_path / (match or "ok").replace(" ", "_").replace(".", "_").replace("|", "_"), "nyuv2_depth", header=GRID_HEADER, rows=GRID_ROWS, classes=None, mutate=lambda s, mut=mut: (_grid_spec_mutate(s), mut(s)))
        if match is None:
            load_yaml(f / "benchmark.yaml")
        else:
            with pytest.raises(SchemaError, match=match):
                load_yaml(f / "benchmark.yaml")


# ----------------------------------------------------------------------------- folds (esc50), floor role + materialize (avsync)


FOLD_ROWS = [[f"c{i}", str(i % 5 + 1), "a" if i % 2 == 0 else "b", f"audio/c{i}.wav"] for i in range(10)]



def _folds_mutate(s: dict[str, Any]) -> None:
    s["splits"] = {"train": [], "eval": [], "floor": [], "folds": ["1", "2", "3", "4", "5"], "unused": [], "min_eval_coverage": 0.95}
    s["budget"]["clips"] = {str(f): 2 for f in range(1, 6)}
    s["manifest"]["upstream"] = None
    s["manifest"]["columns"]["rel_media"]["stream"] = "audio"
    set_streams(s, {"audio": dir_stream("audio", "rel_media")})
    s["protocol"]["clip_seconds"] = 5.0
    s["metrics"].update({"headline": "acc_mean", "list": ["acc_mean", "acc_pstd", "acc_per_fold", "n_folds", "n_eval"], "floors": ["acc_floor"], "chance": {"acc_mean": 0.02}, "ci": {"acc_mean": "pooled_binomial"}, "aggregation": {"over": "folds", "report": ["mean", "pstd", "per_fold"], "ci": "pooled_binomial"}})


def test_folds_esc50_shape(tmp_path):
    b = Benchmark.from_dir(make_folder(tmp_path, "esc50", rows=FOLD_ROWS, mutate=_folds_mutate))
    assert b.roles() == {"train": [], "eval": [], "floor": [], "folds": ["1", "2", "3", "4", "5"]}
    assert b.spec.splits.read_values() == ("1", "2", "3", "4", "5")
    assert all(b.n_rows(f) == 2 for f in "12345")
    assert b.manifest("3").clip_id == ["c2", "c7"]
    assert b.metric_specs["acc_per_fold"].shape == "list"
    with pytest.raises(SchemaError, match="aggregation"):
        load_yaml(make_folder(tmp_path / "noagg", "esc50", rows=FOLD_ROWS, mutate=lambda s: (_folds_mutate(s), s["metrics"].__setitem__("aggregation", None))) / "benchmark.yaml")
    with pytest.raises(SchemaError, match="two folds"):
        load_yaml(make_folder(tmp_path / "onefold", "esc50", rows=FOLD_ROWS, mutate=lambda s: (_folds_mutate(s), s["splits"].__setitem__("folds", ["1"]), s["budget"].__setitem__("clips", {"1": 2}))) / "benchmark.yaml")


AVSYNC_HEADER = ["clip_id", "label", "split", "rel_media", "rel_video", "rel_audio", "offset_bin", "offset_seconds", "arm"]
AVSYNC_ROWS = [
    ["x0#0", "0", "train", "avsync/train/x0#0.mp4", "x0.mp4", "x0.wav", "0", "-2.0", "transient"],
    ["x0#4", "4", "train", "avsync/train/x0#4.mp4", "x0.mp4", "x0.wav", "4", "0.0", "transient"],
    ["x1#8", "8", "train", "avsync/train/x1#8.mp4", "x1.mp4", "x1.wav", "8", "2.0", "transient"],
    ["x2#2", "2", "train", "avsync/train/x2#2.mp4", "x2.mp4", "x2.wav", "2", "-1.0", "transient"],
    ["e0#1", "1", "eval", "avsync/eval/e0#1.mp4", "e0.mp4", "e0.wav", "1", "-1.5", "transient"],
    ["e1#7", "7", "eval", "avsync/eval/e1#7.mp4", "e1.mp4", "e1.wav", "7", "1.5", "transient"],
    ["f0#3", "3", "floor", "avsync/floor/f0#3.mp4", "f0.mp4", "f0.wav", "3", "-0.5", "stationary"],
    ["f1#5", "5", "floor", "avsync/floor/f1#5.mp4", "f1.mp4", "f1.wav", "5", "0.5", "stationary"],
]
AVSYNC_CLASSES = [[str(i), str(i)] for i in range(9)]


def _materialize() -> dict[str, Any]:
    return {
        "kind": "av_offset_mux", "offset_column": "offset_seconds", "source_columns": {"video": "rel_video", "audio": "rel_audio"},
        "pivot": 1.0, "length": 8.0, "video": 0.5, "audio": -0.5,
        "tool": {"name": "ffmpeg", "version_pin": "4.4.2", "video_args": ["-c:v", "libx264"], "audio_args": ["-c:a", "alac"]},
    }


#: avsync's offset metrics and floor-arm controls are its probe.py METRICS, not registry names.
AVSYNC_PROBE = (Path(bm.__file__).parent / "benchmarks" / "avsync" / "probe.py").read_text(encoding="utf-8")


def _avsync_mutate(s: dict[str, Any]) -> None:
    s["splits"] = {"train": ["train"], "eval": ["eval"], "floor": ["floor"], "folds": None, "unused": [], "min_eval_coverage": 0.95}
    s["budget"]["clips"] = {"train": 4, "eval": 2, "floor": 2}
    s["manifest"]["upstream"] = None
    s["manifest"]["columns"] = {
        "rel_media": {"type": "str", "role": "media", "stream": ["video", "audio"]},
        "rel_video": {"type": "str", "role": "extra"},
        "rel_audio": {"type": "str", "role": "extra"},
        "offset_bin": {"type": "int", "role": "extra"},
        "offset_seconds": {"type": "float", "role": "extra"},
        "arm": {"type": "str", "role": "extra"},
    }
    s["data"]["roots"] = {"media_root": {"default": "${data.root}/vggsound"}, "derived_root": {"default": "${data.root}/avsync_derived"}}
    set_streams(s, {"video": dir_stream("video", "rel_media", root="derived_root"), "audio": dir_stream("audio", "rel_media", root="derived_root")})
    s["input_contract"]["materialize"] = _materialize()
    s["protocol"].update(
        {
            "views": "rows",
            "offsets_seconds": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            "nominal_window_start": 1.0,
            "floor_sigmas": 3.0,
            "coarse_grid_seconds": 1.0,
            "curation": {"required_seconds": 10.0, "min_onset_rate": 0.1, "top_frac_peakiness": 0.4, "peakiness_cut": 3.564906, "motion_quantile": 0.25, "motion_cut": 0.065111, "stationary_pool_size": 6000, "offset_seed": 0},
        }
    )
    s["metrics"].update({"headline": "acc", "list": ["acc", "acc_tol1", "mae_seconds", "acc_coarse5", "n_coarse5", "n_eval"], "floors": ["acc_floor"], "controls": ["acc_floor_arm", "floor_violated"], "chance": {"acc": 0.111}, "ci": {"acc": "binomial"}})


def test_avsync_materialized_mp4_shape(tmp_path, yavih_home, monkeypatch):
    derived = tmp_path / "derived"
    for r in AVSYNC_ROWS:
        touch_media(derived, [r[3]], f"mp4-{r[0]}".encode())
    folder = make_folder(tmp_path, "avsync", header=AVSYNC_HEADER, rows=AVSYNC_ROWS, classes=AVSYNC_CLASSES, probe_py=AVSYNC_PROBE, mutate=_avsync_mutate)
    with_roots(monkeypatch, "avsync", media_root=tmp_path / "vggsound", derived_root=derived)
    b = Benchmark.from_dir(folder)
    assert b.roles() == {"train": ["train"], "eval": ["eval"], "floor": ["floor"], "folds": []}
    m = b.manifest("train")
    assert m.label_ids == [0, 4, 8, 2]
    assert m.media_key == [f"audio={r[3]}|video={r[3]}" for r in AVSYNC_ROWS[:4]]  # every row has its own materialized file
    assert m.extra[0]["offset_seconds"] == -2.0
    assert set(b.spec.protocol_extra_keys) == {"offsets_seconds", "nominal_window_start", "floor_sigmas", "coarse_grid_seconds", "curation"}
    assert b.spec.input_contract.materialize["tool"]["version_pin"] == "4.4.2"
    # the reader hands over the materialized mp4 as BOTH streams (one file, read once); the offset is the
    # label and never reaches the bridge
    s = next(b.reader("train"))
    assert s.clip_id == "x0#0" and s.meta == {"facts": {}}
    assert s.streams["video"]._real_path == s.streams["audio"]._real_path == str(derived / "avsync/train/x0#0.mp4")
    assert os.path.realpath(s.streams["video"].path) == os.path.realpath(derived / "avsync/train/x0#0.mp4")  # an opaque handle
    assert s.streams["video"].bytes() is s.streams["audio"].bytes() and s.streams["video"].bytes() == b"mp4-x0#0"
    # the recipe is the pin, not the output bytes: verify carries no output-sha check
    rep = bench_verify(str(folder), decode_sample=0)
    assert rep["per_split"]["train"]["present"] == 4 and not any(k.startswith("materialized") for k in rep)
    # schema: materialize columns must be declared; no output sha256 ships; controls need their protocol key
    for n, (mut, match) in enumerate([
        (lambda s: s["input_contract"]["materialize"].__setitem__("offset_column", "nope"), "offset_column"),
        (lambda s: s["input_contract"]["materialize"]["tool"].pop("version_pin"), "version_pin"),
        (lambda s: s["input_contract"]["materialize"].__setitem__("outputs", {"file": "materialized.sha256.csv.gz", "sha256": None}), "outputs"),
        (lambda s: s["input_contract"]["materialize"].__setitem__("source_columns", {"video": "nope"}), "source_columns"),
        (lambda s: s["input_contract"]["materialize"]["tool"].__setitem__("pins", {"x": {"value": "v", "probe": [], "match": "."}}), "probe"),
        (lambda s: s["input_contract"]["materialize"]["tool"].__setitem__("pins", {"x": {"value": "v", "probe": ["a"], "match": "("}}), "match"),
        (lambda s: s["protocol"].pop("floor_sigmas"), "floor_sigmas"),
    ]):
        f = make_folder(tmp_path / f"bad{n}", "avsync", header=AVSYNC_HEADER, rows=AVSYNC_ROWS, classes=AVSYNC_CLASSES, probe_py=AVSYNC_PROBE, mutate=lambda s, mut=mut: (_avsync_mutate(s), mut(s)))
        with pytest.raises(SchemaError, match=match):
            load_yaml(f / "benchmark.yaml")


# ----------------------------------------------------------------------------- tar index layout


def _make_tar_index(root: Path, members: dict[str, bytes]) -> None:
    """One tar + index.json v1 with ``members: {key: [shard_idx, offset_data, size]}``."""
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "shard-000000.tar"
    with tarfile.open(tar_path, "w") as tf:
        for key, data in members.items():
            ti = tarfile.TarInfo(key)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    index: dict[str, Any] = {"version": 1, "shards": ["shard-000000.tar"], "members": {}}
    with tarfile.open(tar_path, "r") as tf:
        for ti in tf.getmembers():
            index["members"][ti.name] = [0, ti.offset_data, ti.size]
    (root / "index.json").write_text(json.dumps(index))


TAR_HEADER = ["clip_id", "split", "label", "rel_audio"]
TAR_ROWS = [
    ["a0", "train", "/m/x", "audio/train/a0.wav"],
    ["a1", "train", "/m/y;/m/x", "audio/train/a1.wav"],
    ["a2", "train", "/m/y", "audio/train/a2.wav"],
    ["a3", "train", "/m/x", "audio/train/a3.wav"],
    ["b0", "eval", "/m/x", "audio/eval/b0.wav"],
    ["b1", "eval", "/m/y", "audio/eval/b1.wav"],
]
TAR_CLASSES = [["0", "x", "/m/x"], ["1", "y", "/m/y"]]


def _tar_mutate(s: dict[str, Any]) -> None:
    s["aliases"] = ["synth_alias"]
    s["splits"]["eval"] = ["eval"]
    s["budget"]["clips"] = {"train": 4, "eval": 2}
    s["manifest"]["upstream"] = None
    s["manifest"]["label"].update({"kind": "multi", "class_key": "mid"})
    s["manifest"]["columns"] = {"rel_audio": {"type": "str", "role": "media", "stream": "audio"}}
    set_streams(
        s,
        {"audio": {"kind": "audio", "media_column": "rel_audio", "root": "media_root", "layout": "tar_index", "layout_params": {}, "required": True}},
        {"audio": []},
    )
    s["data"]["tar_index"] = "index.json"
    s["metrics"].update({"headline": "map", "list": ["map", "effective_pos_classes", "n_eval"], "floors": ["map_floor"], "chance": {}, "ci": {}})


def test_tar_index_presence_fingerprint_and_reader(tmp_path, yavih_home, monkeypatch):
    from yavih.media import MediaMissing
    from yavih.sample import fetch

    folder = make_folder(tmp_path, "tarbench", header=TAR_HEADER, rows=TAR_ROWS, classes=TAR_CLASSES, mutate=_tar_mutate)
    media = tmp_path / "media"
    members = {r[3]: f"audio-{r[0]}".encode() for r in TAR_ROWS if r[0] not in ("a3", "b1")}  # a3, b1: no member
    _make_tar_index(media, members)
    with_roots(monkeypatch, "tarbench", media_root=media)
    b = Benchmark.from_dir(folder)
    assert b.spec.aliases == ("synth_alias",)
    assert b.manifest("train").label_ids == [[0], [1, 0], [1], [0]]
    st = b.data_status()
    assert st.per_split["train"] == {"present": 3, "nominal": 4, "missing_first10": ["a3"]}
    assert st.per_split["eval"] == {"present": 1, "nominal": 2, "missing_first10": ["b1"]}
    assert st.status_word() == "partial 66%"
    rep = bench_verify(str(folder), decode_sample=0, data={"media_root": str(media)})
    assert rep["per_split"] == st.per_split and rep["media_fingerprint"] is not None
    members["audio/train/a0.wav"] = b"audio-a0-CHANGED"
    _make_tar_index(media, members)
    rep2 = bench_verify(str(folder), decode_sample=0, data={"media_root": str(media)})
    assert rep2["media_fingerprint"] != rep["media_fingerprint"]
    # the reader hands over the member in place
    s = next(b.reader("eval"))
    assert s.streams["audio"].bytes() == b"audio-b0" and s.streams["audio"].member[0] == str(media / "shard-000000.tar")
    # a3's missing member makes the row missing at the read-ahead, never at the bridge
    a3 = b.sample("train", 3)
    with pytest.raises(MediaMissing, match="a3"):
        fetch(a3)


def test_in1k_shape_stored_jpeg(tmp_path, monkeypatch):
    rows = [[r[0], r[1], r[2], r[3].replace(".mp4", ".JPEG")] for r in BASE_ROWS]

    def mutate(s):
        s["manifest"]["columns"]["rel_media"]["stream"] = "image"
        set_streams(s, {"image": dir_stream("image", "rel_media")}, {"image": ["container", "bytes"]})
        s["protocol"]["clip_seconds"] = None

    folder = make_folder(tmp_path, "in1k", rows=rows, mutate=mutate)
    media = tmp_path / "media"
    touch_media(media, [r[3] for r in rows], b"\xff\xd8")
    with pytest.raises(ConfigError, match="YAVIH_DATA__IN1K__MEDIA_ROOT"):
        next(Benchmark.from_dir(folder).reader("val"))  # no root configured → exit-1 territory, named env var
    with_roots(monkeypatch, "in1k", media_root=media)
    b = Benchmark.from_dir(folder)
    s = next(b.reader("val"))
    # every dir-layout stream hands over its path and its bytes
    assert s.streams["image"]._real_path == str(media / "val/v0.JPEG") and s.streams["image"].bytes() == b"\xff\xd8"
    assert "/val/" not in s.streams["image"].path and "val" not in os.path.basename(s.streams["image"].path) and s.streams["image"].path.endswith(".jpeg")  # opaque: no split, no class dir
    assert s.streams["image"].member is None
    assert b.data_status().status_word() == "partial 100%"
    rep = bench_verify(str(folder), decode_sample=0)
    assert rep["status"] == "verified" and b.data_status(refresh=True).status_word() == "verified"


# ----------------------------------------------------------------------------- reader, shards, canaries


def test_reader_order_shards_rows_and_canaries(tmp_path, monkeypatch):
    rows = [[f"r{i:02d}", "train" if i < 20 else "val", "a" if i % 2 else "b", f"m/r{i:02d}.mp4"] for i in range(26)]
    b = Benchmark.from_dir(make_folder(tmp_path, rows=rows, mutate=lambda s: (s["budget"].__setitem__("clips", {"train": 20, "val": 6}), s["manifest"].__setitem__("upstream", None))))
    with_roots(monkeypatch, "synth", media_root=tmp_path / "media")
    samples = list(b.reader("train"))
    assert [s.row for s in samples] == list(range(20)) and [s.clip_id for s in samples] == [r[0] for r in rows[:20]]
    assert all(s.split == "train" and set(s.streams) == {"video"} and s._canary_of is None and s.meta == {"facts": {}} for s in samples)
    assert samples[0].streams["video"]._real_path == str(tmp_path / "media" / "m/r00.mp4")
    assert not hasattr(samples[0], "media_key")  # never handed to the bridge
    assert [s.row for s in b.reader("train", rows=range(5, 8))] == [5, 6, 7]
    with pytest.raises(ValueError):
        list(b.reader("train", rows=range(0, 30)))
    with pytest.raises(ValueError):
        list(b.reader("train", shard=(0, 2), rows=range(0, 2)))
    with pytest.raises(ValueError):
        list(b.reader("train", shard=(2, 2)))
    from yavih.io.shards import plan_shards

    assert [s.row for s in b.reader("train", shard=(1, 2), batch_size=4)] == list(range(*plan_shards(20, 4, 2)[1]))
    assert [s.row for s in b.reader("train", shard=(0, 2), batch_size=4)] == list(range(*plan_shards(20, 4, 2)[0]))
    seven = plan_shards(20, 4, 7)
    empty = next(i for i, (lo, hi) in enumerate(seven) if lo == hi)
    assert list(b.reader("train", shard=(empty, 7), batch_size=4, canaries=8)) == []
    # canaries: min(K, n // 4) copies after the regular rows, deterministic, from rows ≥ 2B before the end
    out = list(b.reader("train", canaries=8, batch_size=2, canary_seed=0))
    regular, canaries = out[:20], out[20:]
    assert len(canaries) == 5 == min(8, 20 // 4)
    assert all(c._canary_of == c.row and c.row < 20 - 4 for c in canaries)
    assert all("canary_of" not in c.meta and c.meta == {"facts": {}} for c in canaries)  # a bridge cannot tell
    assert len({c.row for c in canaries}) == 5
    again = [c.row for c in list(b.reader("train", canaries=8, batch_size=2, canary_seed=0))[20:]]
    assert again == [c.row for c in canaries]
    other = [c.row for c in list(b.reader("train", canaries=8, batch_size=2, canary_seed=1))[20:]]
    assert other != [c.row for c in canaries]
    assert b.canary_rows("train", range(0, 20), 2, 8, 0) == [c.row for c in canaries]
    assert b.canary_rows("train", range(0, 3), 1, 8, 0) == []  # n // 4 == 0
    reg_by_row = {s.row: s for s in regular}
    for c in canaries:
        r = reg_by_row[c.row]
        # the same row, facts and stored bytes; its own opaque path handle: a path-keyed cache misses
        assert (c.row, c.clip_id, c.split, c.meta) == (r.row, r.clip_id, r.split, r.meta)
        assert c.streams["video"] is not r.streams["video"] and c.streams["video"]._real_path == r.streams["video"]._real_path
        assert c.streams["video"].path != r.streams["video"].path


def test_reader_delegates_to_folder_reader_py(tmp_path, monkeypatch):
    folder = make_folder(tmp_path)
    (folder / "reader.py").write_text(
        "from yavih.sample import Sample\n"
        "def iter_samples(bench, split, rows, batch_size, canaries, canary_seed, *, shard=None):\n"
        "    for r in rows:\n"
        "        yield Sample(row=r, clip_id=bench.manifest(split).clip_id[r], split=split, streams={}, meta={'custom': True, 'bs': batch_size, 'k': canaries})\n"
    )
    b = Benchmark.from_dir(folder)
    out = list(b.reader("val", batch_size=3, canaries=2))
    assert [s.row for s in out] == [0, 1] and out[0].meta == {"custom": True, "bs": 3, "k": 2}
    (folder / "reader.py").write_text("x = 1\n")
    b2 = Benchmark.from_dir(folder)
    with pytest.raises(ImportError, match="iter_samples"):
        list(b2.reader("val"))


READER_PY = '''"""reader.py of a synthetic folder: delegates to the generic reader."""


def iter_samples(bench, split, rows, batch_size, canaries, canary_seed, *, shard=None, roots=None):
    from yavih.sample import iter_samples as generic

    return generic(bench, split, rows, batch_size, canaries, canary_seed, shard=shard, roots=roots)
'''


def test_folder_reader_is_provenance_and_receives_the_shard(tmp_path, monkeypatch):
    """An optional ``reader.py`` is recorded as provenance (``folder_source.reader``, normalized source: a comment
    edit is not a change) and never moves a hash; ``Benchmark.reader`` hands it ``shard=(i, n)`` so the canary draw
    is seeded with the shard index."""
    from yavih.io.shards import plan_shards

    rows = [[f"r{i:02d}", "train" if i < 20 else "val", "a" if i % 2 else "b", f"m/r{i:02d}.mp4"] for i in range(26)]
    folder = make_folder(tmp_path, rows=rows, mutate=lambda s: (s["budget"].__setitem__("clips", {"train": 20, "val": 6}), s["manifest"].__setitem__("upstream", None)))
    with_roots(monkeypatch, "synth", media_root=tmp_path / "media")
    b0 = Benchmark.from_dir(folder)
    assert b0.folder_source == {"probe": None, "reader": None}
    (folder / "reader.py").write_text(READER_PY, encoding="utf-8")
    b = Benchmark.from_dir(folder)
    assert b.folder_source["reader"] == hashing.source_sha256(READER_PY)
    assert b.protocol_hash == b0.protocol_hash == _expected_protocol_hash(b)
    assert b.input_hash == b0.input_hash and b.scoring_hash == b0.scoring_hash
    (folder / "reader.py").write_text("# a comment added\n" + READER_PY, encoding="utf-8")
    assert Benchmark.from_dir(folder).folder_source == b.folder_source
    (folder / "reader.py").write_text(READER_PY, encoding="utf-8")
    lo, hi = plan_shards(20, 2, 2)[1]
    out = list(b.reader("train", shard=(1, 2), batch_size=2, canaries=8))
    regular = [s.row for s in out if s._canary_of is None]
    canaries = [s._canary_of for s in out if s._canary_of is not None]
    assert regular == list(range(lo, hi)) and len(canaries) == min(8, (hi - lo) // 4)
    assert canaries == b.canary_rows("train", range(lo, hi), 2, 8, 0, shard_index=1)
    assert canaries != b.canary_rows("train", range(lo, hi), 2, 8, 0, shard_index=0)
    assert canaries == [s._canary_of for s in b0.reader("train", shard=(1, 2), batch_size=2, canaries=8) if s._canary_of is not None]


# ----------------------------------------------------------------------------- resolution, aliases, listing, info, fetch


def test_load_resolution_order_and_aliases(tmp_path, yavih_home, monkeypatch):
    home_bench = make_folder(yavih_home / "benchmarks", "homer", mutate=lambda s: s.__setitem__("aliases", ["homer_alias", "shared"]))
    monkeypatch.chdir(tmp_path)
    # 1. an existing directory path wins
    local = make_folder(tmp_path / "local", "homer", mutate=lambda s: s.__setitem__("title", "LOCAL"))
    assert Benchmark.load(str(local)).spec.title == "LOCAL"
    assert Benchmark.load("local/homer").spec.title == "LOCAL"  # relative path
    # 2. $YAVIH_HOME/benchmarks/<ref>
    assert Benchmark.load("homer").folder == home_bench.resolve()
    # 4. alias lookup
    assert Benchmark.load("homer_alias").name == "homer"
    assert bm.resolve_folder("shared") == home_bench
    with pytest.raises(UnknownBenchmark, match="nope"):
        Benchmark.load("nope")
    with pytest.raises(UnknownBenchmark):
        Benchmark.load("../homer")  # a path that is not a benchmark dir never resolves by name
    assert isinstance(UnknownBenchmark("m"), KeyError) and str(UnknownBenchmark("m")) == "m"
    assert bm.packaged_benchmarks() == sorted(bm.packaged_benchmarks())
    from yavih.benchmarks import packaged_folder, packaged_names, packaged_root

    assert packaged_root().is_dir() and packaged_names() == bm.packaged_benchmarks()
    assert packaged_folder("../etc") is None and packaged_folder("") is None


def test_data_status_cache_presence_and_bench_list(tmp_path, yavih_home, monkeypatch):
    make_folder(yavih_home / "benchmarks", "synth")
    media = tmp_path / "media"
    touch_media(media, [r[3] for r in BASE_ROWS[:5]])  # v1 missing
    with_roots(monkeypatch, "synth", media_root=media)
    b = Benchmark.load("synth")
    st = b.data_status()
    assert st.roots == {"media_root": str(media)} and st.manifest_hash == b.manifest_hash
    assert st.per_split == {"train": {"present": 4, "nominal": 4, "missing_first10": []}, "val": {"present": 1, "nominal": 2, "missing_first10": ["v1"]}}
    assert st.status_word() == "partial 83%" and st.media_fingerprint is None and st.verified is None and st.media_info_sha256 is None
    path = data_status_path("synth", {"media_root": str(media)})
    assert path.parent == yavih_home / "data_status" and path.name.startswith("synth-")  # keyed by the roots
    assert path.is_file() and st.path == str(path)
    touch_media(media, ["val/v1.mp4"])
    b2 = Benchmark.load("synth")
    assert b2.data_status().per_split["val"]["present"] == 1
    assert b2.data_status(refresh=True).per_split["val"]["present"] == 2
    other = tmp_path / "other"
    touch_media(other, [r[3] for r in BASE_ROWS[:1]])
    st3 = Benchmark.load("synth", data={"media_root": str(other)}).data_status()
    assert st3.present() == 1 and st3.status_word() == "partial 16%"
    with pytest.raises(ConfigError):
        Benchmark.load("synth", data={"nope": "/x"}).data_status()
    rows = list_benchmarks()
    mine = [r for r in rows if r["name"] == "synth"]
    assert mine and mine[0]["readout"] == "clf" and mine[0]["clips"] == {"train": 4, "val": 2} and mine[0]["source_kind"] == "derived"
    assert mine[0]["status"].startswith("partial") and mine[0]["aliases"] == []
    monkeypatch.delenv("YAVIH_DATA__SYNTH__MEDIA_ROOT")
    path.unlink()
    assert [r for r in list_benchmarks() if r["name"] == "synth"][0]["status"] == "missing"
    bad = make_folder(yavih_home / "benchmarks", "broken", mutate=lambda s: s.__setitem__("readout", "cls"))
    assert bad.is_dir()
    row = [r for r in list_benchmarks() if r["name"] == "broken"][0]
    assert row["status"] == "invalid" and "SchemaError" in row["error"]


def test_media_info_table_is_found_and_recorded(tmp_path, yavih_home, monkeypatch):
    """The site census lives at ``$YAVIH_HOME/media_info/<name>-<manifest_hash12>.jsonl.gz``; the reader
    prefers it over a live probe and its sha256 goes to ``data_status.json``."""
    from yavih.sample import MediaInfoTable, media_info_path

    folder = make_folder(tmp_path, "synth")
    media = tmp_path / "media"
    touch_media(media, [r[3] for r in BASE_ROWS], b"not-an-mp4")
    with_roots(monkeypatch, "synth", media_root=media)
    b = Benchmark.from_dir(folder)
    assert b.media_info() is None
    info = {"duration_s": 10.0, "fps": 25.0, "n_frames": 250, "width": 320, "height": 240, "container": "mp4", "bytes": 10}
    entries = {("video", r[3]): {**info, "duration_s": 10.0 + i} for i, r in enumerate(BASE_ROWS)}
    table = MediaInfoTable.write(media_info_path("synth", b.manifest_hash), entries)
    assert table.path.endswith(f"synth-{b.manifest_hash[:12]}.jsonl.gz")
    b = Benchmark.from_dir(folder)
    assert len(b.media_info()) == 6 and b.media_info().sha256 == table.sha256
    s = next(b.reader("val"))
    assert s.streams["video"].info["duration_s"] == 14.0 and s.streams["video"].info["sample_rate"] is None  # no probe of the junk bytes
    assert b.data_status(refresh=True).media_info_sha256 == table.sha256
    rep = bench_verify(str(folder), decode_sample=0)
    assert rep["media_info"] == {"present": True, "rows": 6, "sha256": table.sha256, "path": table.path}
    assert json.loads(Path(rep["data_status_path"]).read_text())["media_info_sha256"] == table.sha256


def test_status_word_edges():
    st = bm.DataStatus(name="x", manifest_hash="h", roots={}, per_split={}, media_fingerprint=None, targets_sha_ok=None, verified=None)
    assert st.status_word() == "missing"
    st.per_split = {"train": {"present": 0, "nominal": 10, "missing_first10": []}}
    assert st.status_word() == "missing"
    st.per_split = {"train": {"present": 10, "nominal": 10, "missing_first10": []}}
    assert st.status_word() == "partial 100%"
    st.verified = "2026-10-07T00:00:00Z"
    assert st.status_word() == "verified"
    st.per_split["train"]["present"] = 6
    assert st.status_word() == "partial 60%"
    d = st.to_json()
    assert d["schema"] == 1 and bm.DataStatus.from_json(d).per_split == st.per_split


def test_bench_verify_and_info_and_fetch(tmp_path, yavih_home, monkeypatch):
    folder = make_folder(tmp_path / "src", "synth", mutate=lambda s: s.__setitem__("aliases", ["syn"]))
    (folder / "README.md").write_text("# synth\n\n## Task\nt\n")
    (folder / "setup.md").write_text("# synth — setup\n")
    media = tmp_path / "media"
    touch_media(media, [r[3] for r in BASE_ROWS], b"\x00" * 100)
    with_roots(monkeypatch, "synth", media_root=media)
    rep = bench_verify(str(folder), decode_sample=0, fingerprint_sample=4)
    assert rep["manifest_sha_ok"] and rep["labels_ok"] and rep["curation_invariants_ok"] is True
    assert rep["per_split"]["train"]["present"] == 4
    assert rep["decode_sample"]["ok"] == 0 and "skipped" in rep["decode_sample"] and rep["read"]["ok"] == 0
    assert rep["targets_sha_ok"] is None and rep["derived_sha_ok"] is None
    assert rep["external"] == {} and rep["media_info"]["present"] is False and rep["data_roots"] == {"media_root": str(media)}
    assert len(rep["media_fingerprint"]) == 64 and rep["status"] == "verified"
    status = json.loads(Path(rep["data_status_path"]).read_text())
    assert status["media_fingerprint"] == rep["media_fingerprint"] and status["verified"] and status["roots"] == rep["data_roots"]
    assert bench_verify(str(folder), decode_sample=0, fingerprint_sample=4)["media_fingerprint"] == rep["media_fingerprint"]
    (media / "train/t0.mp4").write_bytes(b"\x01" * 100)
    assert bench_verify(str(folder), decode_sample=0, fingerprint_sample=256)["media_fingerprint"] != rep["media_fingerprint"]
    # read reads every sampled row's bytes; the decode sample (placeholder media) fails per row, never crashes verify
    rep3 = bench_verify(str(folder), decode_sample=2)
    assert rep3["read"] == {"ok": 2, "failed": 0, "failures": []}
    assert rep3["decode_sample"]["ok"] == 0 and (rep3["decode_sample"]["failed"] == 2 or "skipped" in rep3["decode_sample"])
    # a partially staged site: the read / decode sample draws from PRESENT rows only (an absent row is a
    # presence fact, already counted in per_split, not a media-health failure)
    (media / "val/v1.mp4").unlink()
    (media / "train/t1.mp4").unlink()
    rep4 = bench_verify(str(folder), decode_sample=6)
    assert rep4["per_split"]["train"]["present"] == 3 and rep4["per_split"]["val"]["present"] == 1
    assert rep4["read"]["failed"] == 0 and rep4["read"]["ok"] == 4  # 3 of train's present rows + val's one
    # a row that vanishes between the presence scan and the read is a failed read
    import yavih.benchmark as B

    real_scan = B._Presence.scan

    def scan_then_lose_a_file(self):
        out = real_scan(self)
        (media / "train/t2.mp4").unlink()
        return out

    monkeypatch.setattr(B._Presence, "scan", scan_then_lose_a_file)
    rep5 = bench_verify(str(folder), decode_sample=6)
    assert rep5["read"]["failed"] == 1 and rep5["read"]["failures"][0]["clip_id"] == "t2"
    monkeypatch.setattr(B._Presence, "scan", real_scan)
    (media / "train/t2.mp4").write_bytes(b"\x00" * 100)
    bad = make_folder(tmp_path / "cur", "synth", mutate=lambda s: s["manifest"]["upstream"]["curation"]["excluded"].__setitem__("leakage", 3))
    assert bench_verify(str(bad), decode_sample=0)["curation_invariants_ok"] is False
    info = bench_info(str(folder), scan=True)
    assert info["readme"].startswith("# synth") and info["roles"]["train"]["nominal"] == 4
    assert "floor" not in info["roles"] and info["floors"] == ["acc_floor"] and info["metrics"]["headline"] == "acc"
    assert info["history"][-1]["version"] == 2 and info["source"]["kind"] == "derived" and info["budget"]["media_gb"] == 0.0
    assert info["external"] == [] and info["input_contract"]["facts"] == [] and info["output_contract"]["shape"] == "fixed"
    assert bench_info(str(folder), readme_only=True) == {"name": "synth", "readme": info["readme"]}
    assert bench_info(str(folder), setup=True) == {"name": "synth", "setup": "# synth — setup\n"}
    out = bench_fetch(str(folder))
    assert out["folder"] == str(yavih_home / "benchmarks" / "synth") and out["manifest_sha_ok"]
    assert {"manifest.csv.gz", "setup.md"} <= set(out["copied"])
    assert Benchmark.load("synth").folder == (yavih_home / "benchmarks" / "synth").resolve()
    assert Benchmark.load("syn").name == "synth"
    assert bench_fetch("synth")["copied"] == []
    assert Benchmark.load("synth", fetch=True).name == "synth"
    with pytest.raises(UnknownBenchmark):
        bench_fetch("nope")
    (yavih_home / "benchmarks" / "synth" / "classes.csv").write_text("index,name\n0,zzz\n1,b\n")
    with pytest.raises(ManifestError, match="classes sha256"):
        bench_fetch("synth")


def test_bench_verify_derived_sha(tmp_path, yavih_home, monkeypatch):
    derived = tmp_path / "derived"
    touch_media(derived, ["rgb/a.png"], b"png-bytes")
    folder = make_folder(tmp_path, "synth", mutate=lambda s: (s["data"]["roots"].__setitem__("derived_root", {"default": "/x"}), s["data"].__setitem__("derived", {"rgb/a.png": _sha(b"png-bytes")})))
    with_roots(monkeypatch, "synth", media_root=tmp_path / "media", derived_root=derived)
    assert bench_verify(str(folder), decode_sample=0)["derived_sha_ok"] is True
    (derived / "rgb/a.png").write_bytes(b"other")
    rep = bench_verify(str(folder), decode_sample=0)
    assert rep["derived_sha_ok"] is False and rep["derived_errors"] == ["rgb/a.png: sha256 mismatch"]


def test_media_fingerprint_function(tmp_path):
    files = []
    for i in range(5):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(bytes([i]) * 10)
        files.append((p, 0, None))
    fp = bm.media_fingerprint(files, "h" * 64, sample=3)
    assert fp == bm.media_fingerprint(files, "h" * 64, sample=3) and len(fp) == 64
    assert fp != bm.media_fingerprint(files, "g" * 64, sample=3)  # seeded by manifest_hash
    assert bm.media_fingerprint([], "h" * 64) is None and bm.media_fingerprint(files, "h", sample=0) is None
    # offsets select a sub-range (tar members)
    big = tmp_path / "big.bin"
    big.write_bytes(b"A" * 100 + b"B" * 100)
    assert bm.media_fingerprint([(big, 0, 100)], "h") != bm.media_fingerprint([(big, 100, 100)], "h")
    # duplicate locations (shared media) count once
    assert bm.media_fingerprint(files + files, "h" * 64, sample=10) == bm.media_fingerprint(files, "h" * 64, sample=10)


# ----------------------------------------------------------------------------- spec dataclass helpers


def test_stream_spec_for_split_and_curation_helpers():
    s = bm.StreamSpec(kind="image", media_column={"train": "a", "eval": "b"}, root="r", layout={"train": "dir", "val": "tar_index"}, layout_params={"train": {}, "eval": {}}, layout_params_per_split=True)
    assert s.for_split("train", "train") == {"kind": "image", "media_column": "a", "root": "r", "layout": "dir", "layout_params": {}, "required": True}
    assert s.for_split("val", "eval")["layout"] == "tar_index" and s.for_split("val", "eval")["media_column"] == "b"
    with pytest.raises(SchemaError, match="no entry"):
        s.for_split("test", "floor")
    assert s.media_columns() == ("a", "b") and s.layouts() == ("dir", "tar_index") and s.roots() == ("r",)
    cur = bm.CurationSpec(total_upstream=10, per_split={"a": {"listed": 4}, "b": {"listed": 3}}, excluded={"x": 3})
    assert cur.invariant_holds() and not bm.CurationSpec(total_upstream=10, per_split={"a": {"listed": 4}}, excluded={}).invariant_holds()
    sp = bm.SplitsSpec(train=("train",), eval=("val",), floor=("f",), folds=None, unused=("u",))
    assert sp.role_of("f") == "floor" and sp.role_of("u") == "unused" and sp.role_of("zz") is None
    assert sp.all_values() == ("train", "val", "f", "u") and sp.read_values() == ("train", "val", "f")
    lab = bm.LabelSpec(column="label", kind="multi_sets", sets={"a": ("x", "y"), "b": ("y", "z")})
    assert lab.label_columns() == ("label", "x", "y", "z")


def test_spec_is_frozen_and_raw_is_the_hashed_yaml(tmp_path):
    folder = make_folder(tmp_path)
    spec = load_yaml(folder / "benchmark.yaml")
    with pytest.raises(AttributeError):
        spec.name = "x"  # type: ignore[misc]
    raw = yaml.safe_load((folder / "benchmark.yaml").read_text())
    assert spec.raw["protocol"] == raw["protocol"] and spec.raw["input_contract"] == raw["input_contract"]
    assert spec.protocol_extra_keys == frozenset()
    assert spec.requires is spec.output_contract.requires and spec.requires == bm.RequiresSpec()
    assert spec.output_contract == bm.OutputContractSpec(shape="fixed", coordinates="media", target_frame=None, requires=bm.RequiresSpec())
    assert spec.input_contract.handed_over == {"video": raw["input_contract"]["handed_over"]["video"]}
    assert spec.input_contract.info == {"video": ("duration_s", "container", "bytes")} and spec.input_contract.materialize is None
    assert spec.external == () and spec.target_region == (0.0, 0.0, 1.0, 1.0)
    assert not hasattr(spec, "window_for")
    assert spec.manifest.upstream.curation.invariant_holds()
    assert bm.READOUT_PARAM_KEYS["box"] == {"grid", "visual_stream", "conditioning_stream", "shuffled_control", "shuffle_seed"}
    assert bm.PROTOCOL_FIXED_KEYS == set(raw["protocol"]) == {"clip_seconds", "train_draw", "views"}
    assert bm.LAYOUTS == {"dir", "tar_index"} and "window" not in bm.COLUMN_ROLES


def test_core_import_boundary_for_benchmark_module():
    """benchmark.py imports only numpy + pyyaml + stdlib + yavih core at module scope."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['sklearn'] = None\n"
        "import yavih.benchmark, yavih.benchmarks\n"
        "bad = [m for m in ('torch', 'sklearn', 'av', 'PIL', 'soundfile', 'scipy') if sys.modules.get(m)]\n"
        "print(','.join(bad))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "", proc.stderr

