"""Folder tests of the packaged ``in1k`` benchmark.

Run only this folder: ``python -m pytest "$(yavih bench path in1k)/tests" -q -p no:cacheprovider``.
Everything works without media (the reader smoke uses synthetic JPEGs and the freeze test a synthetic devkit and
train list in ``tmp_path``); the real-data smoke and the re-freeze check run when the torchvision root resolves
(``YAVIH_DATA__IN1K__MEDIA_ROOT`` or the config's ``[data.in1k] media_root``) and are skipped otherwise.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_bytes, rows, bench = F.spec, F.manifest_bytes, F.rows, F.bench

TOP_LEVEL = [
    "schema_version", "name", "aliases", "title", "capability", "header", "readout", "protocol_version", "changelog", "source", "obtain",
    "budget", "manifest", "splits", "streams", "targets", "data", "input_contract", "output_contract", "external", "protocol",
    "metrics", "readout_params",
]
IMAGE_INFO_VOCAB = {"width", "height", "container", "bytes"}  # RawStream.info keys valid for kind image


def _load_module(name: str, path: Path):
    return F.load_module(path, name)


# ----------------------------------------------------------------------------- files, schema 2, hashes


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_folder_contents():
    scripts = sorted(p.name for p in (HERE / "prepare").glob("*.sh"))
    assert scripts and all(re.match(r"^\d{2}_", s) for s in scripts), scripts
    assert not (HERE / "probe.py").exists()  # external: [] -> no authors' code to call


def test_yaml_schema2_top_level(spec):
    assert list(spec) == TOP_LEVEL
    assert spec["schema_version"] == 2
    assert spec["name"] == "in1k" == HERE.name and spec["readout"] == "clf"
    assert spec["aliases"] == ["imagenet1k", "imagenet"]


def test_changelog_entry(spec):
    log = spec["changelog"]
    assert [e["version"] for e in log] == [1] and spec["protocol_version"] == 1
    assert log[0]["invalidates"] is False and log[0]["supersedes"] == []


def test_protocol_holds_label_facts_only(spec):
    assert set(spec["protocol"]) == {"clip_seconds", "train_draw", "views"}, "label facts only"
    assert spec["protocol"]["clip_seconds"] is None and spec["protocol"]["views"] == "none"


def test_streams_block(spec):
    assert spec["streams"] == {
        "image": {"kind": "image", "media_column": "rel_media", "root": "media_root", "layout": "dir", "layout_params": {}, "required": True}
    }
    assert spec["manifest"]["columns"] == {"rel_media": {"type": "str", "role": "media", "stream": "image"}}
    assert spec["targets"] is None and spec["data"]["derived"] == {} and spec["data"]["tar_index"] is None


def test_input_contract(spec):
    ic = spec["input_contract"]
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(spec["streams"]) == set(ic["info"])
    assert "exactly as stored" in ic["handed_over"]["image"]
    assert ic["info"]["image"] == ["width", "height", "container", "bytes"]
    assert set(ic["info"]["image"]) <= IMAGE_INFO_VOCAB
    assert ic["facts"] == [] and ic["materialize"] is None


def test_output_contract_and_external(spec):
    assert spec["output_contract"] == {
        "shape": "fixed",
        "coordinates": "media",
        "target_frame": None,  # clf: no spatial target
        "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None},
    }
    assert spec["external"] == [] and spec["readout_params"] == {}


def test_sha256_match(spec):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec, rows):
    assert F.assert_splits_assigned_once(spec, rows) == {"train": 50000, "val": 50000} == spec["budget"]["clips"]
    assert spec["splits"]["min_eval_coverage"] == 0.95


def test_manifest_schema_and_order(rows):
    assert list(rows[0].keys()) == ["clip_id", "rel_media", "label", "split"]
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[A-Za-z0-9_.#:-]+", i) for i in ids)
    assert all(re.fullmatch(r"(train|val)/n\d{8}/[A-Za-z0-9_]+\.JPEG", r["rel_media"]) for r in rows)
    # label = the wnid directory of rel_media; train rows first, then val, each sorted (wnid, filename)
    assert all(r["rel_media"].split("/")[1] == r["label"] for r in rows)
    assert [r["split"] for r in rows] == ["train"] * 50000 + ["val"] * 50000
    for split in ("train", "val"):
        keys = [(r["label"], r["rel_media"]) for r in rows if r["split"] == split]
        assert keys == sorted(keys), split
    for split in ("train", "val"):
        per: dict[str, int] = {}
        for r in rows:
            if r["split"] == split:
                per[r["label"]] = per.get(r["label"], 0) + 1
        assert len(per) == 1000 and set(per.values()) == {50}, split
    assert rows[0]["clip_id"] == "n01440764_10738" and rows[50000]["clip_id"] == "ILSVRC2012_val_00000293"


def test_classes_contiguous_sorted_wnid(spec):
    classes = list(csv.DictReader(io.StringIO((HERE / spec["manifest"]["classes"]).read_text(encoding="utf-8"), newline="")))
    assert list(classes[0].keys()) == ["index", "name"]
    assert [int(c["index"]) for c in classes] == list(range(1000))
    names = [c["name"] for c in classes]
    assert names == sorted(names) and len(set(names)) == 1000
    assert names[0] == "n01440764" and names[999] == "n15075141"


def test_curation_sums(spec, rows):
    cur = spec["manifest"]["upstream"]["curation"]
    assert sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) == cur["total_upstream"] == 1331167
    for split, v in cur["per_split"].items():
        assert v["listed"] == sum(r["split"] == split for r in rows)
    assert cur["excluded"] == {"draw_remainder": 1281167 - 50000}
    td = spec["protocol"]["train_draw"]
    assert td["cap"] == 50000 and td["seed"] == 0 and td["balance"] == "class"
    up = spec["manifest"]["upstream"]
    assert up["rows"] == cur["total_upstream"] and up["sha256"] == up["upstream_files"]["train file list"]


def test_curation_json_matches_yaml(spec):
    """``curation.json`` (written by make_manifest.py) agrees with benchmark.yaml and the freeze constants."""
    mm = _load_module("in1k_make_manifest", HERE / "make_manifest.py")
    side = json.loads((HERE / "curation.json").read_text(encoding="utf-8"))
    up = spec["manifest"]["upstream"]
    assert side["curation"] == up["curation"] and side["upstream_files"] == up["upstream_files"]
    assert side["manifest_sha256"] == spec["manifest"]["sha256"] and side["classes_sha256"] == spec["manifest"]["classes_sha256"]
    td = spec["protocol"]["train_draw"]
    assert side["train_draw"] == {**{k: td[k] for k in ("cap", "seed", "balance")}, "drawn_from": up["draw"]}
    assert up["draw"] == mm.DRAWN_FROM and (td["cap"], td["seed"]) == (mm.TRAIN_CAP, mm.TRAIN_SEED)
    assert up["upstream_files"] == {mm.DEVKIT_NAME: mm.DEVKIT_SHA256, "train file list": mm.TRAIN_LIST_SHA256}
    assert spec["source"]["sha256"][mm.DEVKIT_NAME] == mm.DEVKIT_SHA256


def test_metric_names_resolve(spec):
    F.assert_metric_names_resolve(HERE, spec)
    names = list(spec["metrics"]["list"]) + list(spec["metrics"]["floors"]) + list(spec["metrics"]["controls"])
    assert names == ["acc", "n_eval", "acc_floor"]
    assert spec["metrics"]["headline"] == "acc" and spec["metrics"]["chance"] == {"acc": 0.001}
    assert spec["metrics"]["ci"] == {"acc": "binomial"}


def test_yaml_loads_in_core(bench, spec):
    assert bench.name == "in1k" and bench.spec is not None
    assert bench.protocol_hash and len(bench.protocol_hash) == 64
    assert bench.n_rows("train") == 50000 and bench.n_rows("val") == 50000
    assert bench.roles()["train"] == ["train"] and bench.roles()["eval"] == ["val"]
    man = bench.manifest("val")
    assert man.label_ids[0] == 0 and man.media[0]["image"] == "val/n01440764/ILSVRC2012_val_00000293.JPEG"
    view = bench.view()
    assert view.streams == {"image": "image"} and view.facts == {} and not hasattr(view, "manifest")
    assert view.input_contract == spec["input_contract"] and view.output_contract == spec["output_contract"]


# ----------------------------------------------------------------------------- README.md / setup.md


def test_readme_facts():
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    assert "@article{russakovsky2015imagenet" in F.section(text, 2, "Citation")
    proto = F.section(text, 2, "In yavih")
    assert "setup.md" in proto and "make_manifest.py" in proto and "--check" in proto


def test_setup_md_keeps_the_facts(spec):
    """The setup facts in setup.md: upstream sizes and digests, the frozen-subset size, the root variable, pitfalls."""
    text = (HERE / "setup.md").read_text(encoding="utf-8")
    for needle in (
        *spec["source"]["sha256"].values(), "1d675b47d978889d74fa0da5fadfb00e", "12 432 228 705", "1 331 167",
        "147 897 477 120", "6 744 924 160", "YAVIH_DATA__IN1K__MEDIA_ROOT", "Empty `val/`", "rsync",
        "yavih bench verify in1k",
    ):
        assert needle in text, needle


# ----------------------------------------------------------------------------- no reader.py (generic reader)


def test_folder_ships_no_reader_py():
    """The folder ships no reader.py: ``Benchmark.reader()`` falls back to the generic
    ``yavih.sample.iter_samples``, which hands over what benchmark.yaml declares; the reader smokes in this
    file go through ``bench.reader``."""
    assert not (HERE / "reader.py").exists()


def _write_jpegs(root: Path, rels: dict[str, str]) -> dict[str, tuple[int, int, str]]:
    """Synthetic raw files under the manifest's first paths: RGB 640x480, greyscale 300x500, CMYK 200x200."""
    from PIL import Image

    rng = np.random.default_rng(0)
    out = {}
    shapes = {"train": ((480, 640), "RGB"), "val": ((500, 300), "L")}
    for split, rel in rels.items():
        (h, w), mode = shapes[split]
        arr = rng.integers(0, 256, size=(h, w, 3) if mode == "RGB" else (h, w), dtype=np.uint8)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode).save(root / rel, format="JPEG", quality=90)
        out[split] = (w, h, mode)
    return out


def test_reader_smoke_on_synthetic_raw_files(tmp_path):
    """Raw hand-off: bytes() is the stored file, info has the guaranteed keys, meta holds only facts."""
    from yavih import Benchmark

    rels = {"train": "train/n01440764/n01440764_10738.JPEG", "val": "val/n01440764/ILSVRC2012_val_00000293.JPEG"}
    sizes = _write_jpegs(tmp_path, rels)
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    for split in ("train", "val"):
        samples = list(b.reader(split, rows=range(0, 1), batch_size=1, canaries=0))
        assert len(samples) == 1 and samples[0].row == 0
        s = samples[0]
        assert set(s.streams) == {"image"} and s.meta == {"facts": {}}
        assert not hasattr(s, "media_key")
        st = s.streams["image"]
        stored = (tmp_path / rels[split]).read_bytes()
        assert st.kind == "image" and st.name == "image" and st.member is None
        assert st._real_path == str(tmp_path / rels[split]) and os.path.realpath(st.path) == os.path.realpath(tmp_path / rels[split])
        assert st.bytes() == stored
        with st.open() as fh:
            assert fh.read() == stored
        w, h, _ = sizes[split]
        assert (st.info["width"], st.info["height"]) == (w, h)
        assert st.info["container"] == "jpeg" and st.info["bytes"] == len(stored)
        for gone in ("image", "frame_times", "window", "calls"):
            assert not callable(getattr(st, gone, None)), gone


def test_reader_canaries_are_exact_copies(tmp_path):
    from yavih import Benchmark

    rels = {"train": "train/n01440764/n01440764_10738.JPEG", "val": "val/n01440764/ILSVRC2012_val_00000293.JPEG"}
    _write_jpegs(tmp_path, rels)
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    man = b.manifest("val")
    for r in range(1, 8):  # rows 1..7 share row 0's synthetic file content
        p = tmp_path / man.media[r]["image"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((tmp_path / rels["val"]).read_bytes())
    samples = list(b.reader("val", rows=range(0, 8), batch_size=2, canaries=1))
    assert [s.row for s in samples[:8]] == list(range(8)) and len(samples) == 9
    c = samples[8]
    assert c._canary_of in range(0, 4) and c.meta == {"facts": {}} and "canary_of" not in c.meta
    assert c.streams["image"].bytes() == samples[c._canary_of].streams["image"].bytes()


def test_missing_media_is_not_handed_over(tmp_path):
    from yavih import Benchmark

    try:
        from yavih.sample import MediaMissing
    except ImportError:  # core may define the class in yavih.media instead
        from yavih.media import MediaMissing

    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    with pytest.raises(MediaMissing):
        for s in b.reader("val", rows=range(0, 1), batch_size=1, canaries=0):
            s.streams["image"].bytes()


def _media_root() -> str | None:
    root = F.site_root(HERE, "media_root", needs=("val/n01440764/ILSVRC2012_val_00000293.JPEG", "train/n01440764/n01440764_10738.JPEG"))
    return None if root is None else str(root)


def test_reader_smoke_on_real_data():
    """Optional extra: the raw hand-off over the real torchvision root (skipped where absent)."""
    root = _media_root()
    if root is None:
        pytest.skip("ImageNet torchvision root not present on this machine")
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": root})
    for split in ("train", "val"):
        samples = list(b.reader(split, rows=range(0, 8), batch_size=2, canaries=1))
        assert [s.row for s in samples[:8]] == list(range(8)) and len(samples) == 9
        st = samples[0].streams["image"]
        assert st._real_path == os.path.join(root, b.manifest(split).media[0]["image"])
        assert st.bytes() == Path(st.path).read_bytes() and st.bytes()[:3] == b"\xff\xd8\xff"
        assert st.info["container"] == "jpeg" and min(st.info["width"], st.info["height"]) >= 32


# ----------------------------------------------------------------------------- bridge_template.py + output contract


def test_bridge_template_contents():
    src = (HERE / "bridge_template.py").read_text(encoding="utf-8")
    for gone in ("decode_plan", "BRIDGE_DECODE", "frame_times", '"decode":', "st.image()", "window"):
        assert gone not in src, gone
    for want in ("Preprocessing source", "crop_region", "grid_pos", '"views": 1', "used", "setup.md"):
        assert want in src, want
    tree = ast.parse(src)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert [c.name for c in cls] == ["In1kBridge"]
    assert {f.name for f in cls[0].body if isinstance(f, ast.FunctionDef)} >= {"__init__", "describe", "encode"}


class _FakeStream:
    """The RawStream surface the template touches: kind, path, info, bytes(), open()."""

    def __init__(self, path: Path):
        from PIL import Image

        self.name = self.kind = "image"
        self.path = str(path)
        with Image.open(path) as im:
            w, h = im.size
        self.info = {"width": w, "height": h, "container": "jpeg", "bytes": path.stat().st_size}
        self.member = None

    def bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def open(self):
        return io.BytesIO(self.bytes())


def _template(tmp_path: Path):
    from PIL import Image

    tpl = _load_module("in1k_bridge_template_under_test", HERE / "bridge_template.py")
    rng = np.random.default_rng(1)
    samples = []
    for k, ((h, w), mode) in enumerate((((375, 500), "RGB"), ((500, 333), "L"), ((256, 256), "CMYK"), ((480, 640), "RGB"))):
        p = tmp_path / f"img{k}.JPEG"
        arr = rng.integers(0, 256, size=(h, w) if mode == "L" else (h, w, len(mode)), dtype=np.uint8)
        Image.fromarray(arr, mode).save(p, format="JPEG", quality=90)
        samples.append(SimpleNamespace(row=k, clip_id=f"c{k}", split="val", streams={"image": _FakeStream(p)}, meta={"facts": {}}))
    view = SimpleNamespace(name="in1k", aliases=["imagenet1k", "imagenet"], readout="clf", protocol_version=1, protocol_hash="0" * 64, streams={"image": "image"}, input_contract={}, output_contract={}, facts={})
    br = tpl.In1kBridge(view, {})

    def forward(x):  # stand-in model: 16x16 patch means -> [B, 196, 3] (instance-wise)
        b = x.shape[0]
        return x.reshape(b, 3, 14, 16, 14, 16).mean(axis=(3, 5)).reshape(b, 3, 196).transpose(0, 2, 1)

    br.forward = forward
    return tpl, br, samples


def test_template_describe_declares_rules(tmp_path):
    _, br, _ = _template(tmp_path)
    d = br.describe()
    assert d["views"] == 1 and "decode" not in d
    assert set(d["input"]) == {"image"} and set(d["input"]["image"]) == {"spatial", "norm", "decoder"}
    assert "temporal" not in d["input"]["image"]  # an image stream has no temporal rule
    assert d["input"]["image"]["spatial"]["crop"] == {"kind": "center", "size": [224, 224]}
    assert d["input"]["image"]["norm"]["source"] and d["input"]["image"]["decoder"]["lib"] == "pillow"
    assert all(v is not None for k, v in d["precision"].items() if k != "autocast")
    assert d["args"] == {"layer": -1}
    try:
        from yavih.bridge import DESCRIBE_REQUIRED, validate_describe
    except ImportError:  # pragma: no cover
        return
    if "views" in DESCRIBE_REQUIRED and "decode" not in DESCRIBE_REQUIRED:  # the core's schema check
        bench = SimpleNamespace(**vars(br.bench))
        validate_describe(d, br, bench)


def test_template_output_contract(tmp_path):
    """Output contract on mixed sizes and colour modes: fixed (N, D), finite, `used` per the declared rule, and
    instance-wise (batch vs alone vs reversed order identical, bridge check's hard checks in miniature)."""
    _, br, samples = _template(tmp_path)
    feats, pos, used = F.encode_prepared(br, samples)
    assert feats.shape == (4, 196, 3) and np.isfinite(feats).all()
    assert pos is None and used == [{"image": {}}] * 4
    alone = np.concatenate([F.encode_prepared(br, [s])[0] for s in samples])
    rev = F.encode_prepared(br, samples[::-1])[0][::-1]
    assert np.array_equal(feats, alone) and np.array_equal(feats, rev)
    again = F.encode_prepared(br, samples)[0]
    assert np.array_equal(feats, again)


def _crop_region_ref(w: int, h: int) -> tuple[float, float, float, float]:
    """torchvision Resize(256) + CenterCrop(224) mapped back to fractions of the original image."""
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    top, left = int(round((nh - 224) / 2.0)), int(round((nw - 224) / 2.0))
    return top / nh, left / nw, (top + 224) / nh, (left + 224) / nw


def test_template_positions_in_media_coordinates(tmp_path):
    """per_sample pos: t = -1, (y, x) patch centres inside the crop region of the ORIGINAL image (E_RULE_MISMATCH)."""
    from yavih.bridge import grid_pos

    assert "region" in inspect.signature(grid_pos).parameters
    import yavih.media.rules  # noqa: F401
    _, br, samples = _template(tmp_path)
    br.pos_mode, br.streams = "per_sample", ("image",)
    _, pos, _ = F.encode_prepared(br, samples)
    assert pos.shape == (4, 196, 4) and pos.dtype == np.float32
    for s, p in zip(samples, pos, strict=True):
        y0, x0, y1, x1 = _crop_region_ref(s.streams["image"].info["width"], s.streams["image"].info["height"])
        assert (p[:, 0] == 0).all() and (p[:, 1] == -1).all()
        assert (p[:, 2] >= y0 - 1e-3).all() and (p[:, 2] <= y1 + 1e-3).all()
        assert (p[:, 3] >= x0 - 1e-3).all() and (p[:, 3] <= x1 + 1e-3).all()
        from yavih.media.rules import crop_region

        r = crop_region(br.describe()["input"]["image"]["spatial"], s.streams["image"].info["width"], s.streams["image"].info["height"])
        assert (p[:, 2] >= r[0] - 1e-6).all() and (p[:, 2] <= r[2] + 1e-6).all()
        assert (p[:, 3] >= r[1] - 1e-6).all() and (p[:, 3] <= r[3] + 1e-6).all()


# ----------------------------------------------------------------------------- make_manifest


def _synthetic_devkit(id_to_wnid: dict[int, str], gt: list[int]) -> bytes:
    """A devkit tar.gz with the two files make_manifest.py reads; one non-leaf synset that must be ignored."""
    from scipy.io import savemat

    rec = np.zeros((len(id_to_wnid) + 1,), dtype=[(f, "O") for f in ("ILSVRC2012_ID", "WNID", "words", "num_children")])
    for k, (sid, w) in enumerate(sorted(id_to_wnid.items())):
        rec[k] = (sid, w, "leaf", 0)
    rec[-1] = (len(id_to_wnid) + 1, "n09999999", "inner", 2)
    mat = io.BytesIO()
    savemat(mat, {"synsets": rec})
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tf:
        for name, data in (("data/meta.mat", mat.getvalue()), ("data/ILSVRC2012_validation_ground_truth.txt", "".join(f"{g}\n" for g in gt).encode())):
            ti = tarfile.TarInfo(f"ILSVRC2012_devkit_t12/{name}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return out.getvalue()


def _synthetic_train(root: Path, per_class: dict[str, int]) -> Path:
    """``root/train/<wnid>/<wnid>_<n>.JPEG`` (empty files) and the equivalent ILSVRC2012_img_train.tar."""
    tar_path = root / "ILSVRC2012_img_train.tar"
    with tarfile.open(tar_path, "w") as outer:
        for w, n in per_class.items():
            (root / "train" / w).mkdir(parents=True)
            inner = io.BytesIO()
            with tarfile.open(fileobj=inner, mode="w") as tf:
                for i in range(n):
                    (root / "train" / w / f"{w}_{i + 1}.JPEG").write_bytes(b"")
                    tf.addfile(tarfile.TarInfo(f"{w}_{i + 1}.JPEG"), io.BytesIO(b""))
            ti = tarfile.TarInfo(f"{w}.tar")
            ti.size = len(inner.getvalue())
            outer.addfile(ti, io.BytesIO(inner.getvalue()))
    return tar_path


def test_make_manifest_on_a_synthetic_layout(tmp_path):
    """The public route end to end on 3 classes: devkit -> val rows, root listing == tar listing -> draw."""
    pytest.importorskip("scipy")
    pytest.importorskip("torch")
    mm = _load_module("in1k_make_manifest", HERE / "make_manifest.py")
    id_to_wnid = {1: "n00000003", 2: "n00000001", 3: "n00000002"}
    gt = [2, 1, 3, 1, 2, 3]
    assert mm.parse_devkit(_synthetic_devkit(id_to_wnid, gt)) == (id_to_wnid, gt)
    assert mm.val_rel_media(id_to_wnid, gt)[:2] == ["val/n00000001/ILSVRC2012_val_00000001.JPEG", "val/n00000003/ILSVRC2012_val_00000002.JPEG"]
    tar_path = _synthetic_train(tmp_path, {"n00000001": 5, "n00000002": 4, "n00000003": 6})
    listed = mm.list_train_root(tmp_path)
    assert sorted(listed) == sorted(mm.list_train_tar(tar_path)) and len(listed) == 15
    assert mm.train_list_sha256(listed) == mm.train_list_sha256(listed[::-1])

    d = mm.derive(listed, id_to_wnid, gt, cap=6, seed=0)
    rows = list(csv.DictReader(io.StringIO(d["manifest_csv"].decode("utf-8"), newline="")))
    assert list(rows[0]) == mm.COLUMNS and [r["split"] for r in rows] == ["train"] * 6 + ["val"] * 6
    assert d["per_class"] == {"n00000001": 2, "n00000002": 2, "n00000003": 2}
    for split in ("train", "val"):
        keys = [(r["label"], r["rel_media"]) for r in rows if r["split"] == split]
        assert keys == sorted(keys)
    assert all(r["rel_media"].split("/")[1] == r["label"] and Path(r["rel_media"]).stem == r["clip_id"] for r in rows)
    assert d["classes_csv"] == b"index,name\n0,n00000001\n1,n00000002\n2,n00000003\n"
    cur = d["curation"]
    assert cur["total_upstream"] == 21 and cur["excluded"] == {"draw_remainder": 9}
    assert d == mm.derive(list(reversed(listed)), id_to_wnid, gt, cap=6, seed=0)  # order-free input, same freeze
    with pytest.raises(SystemExit):
        mm.derive([*listed, "train/n00000004/n00000004_1.JPEG"], id_to_wnid, gt, cap=6, seed=0)


def test_make_manifest_check_mode_matches_committed_files():
    """Re-derive from the configured torchvision root and compare; skipped where the root, torch or scipy is absent."""
    mm = _load_module("in1k_make_manifest", HERE / "make_manifest.py")
    root = F.site_root(HERE, "media_root", needs=(mm.DEVKIT_NAME, "train/n01440764"))
    if root is None:
        pytest.skip("ImageNet torchvision root with the devkit not present on this machine")
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("scipy") is None:
        pytest.skip("torch and scipy needed for the re-freeze")
    r = subprocess.run([sys.executable, str(HERE / "make_manifest.py"), "--check"], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"check": "identical"' in r.stdout
