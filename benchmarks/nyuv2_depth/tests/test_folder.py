"""Folder tests of the packaged ``nyuv2_depth`` benchmark (protocol_version 1).

Run only this folder: ``python -m pytest "$(yavih bench path nyuv2_depth)/tests" -q -p no:cacheprovider``.
Three groups:

- file-level checks (yaml, manifest, listing, README.md / setup.md sections, prepare rules, the PNG codec, the
  manifest freeze helpers, the output-contract geometry) that need only numpy + pyyaml (+ scipy / pillow where
  noted);
- checks through the core (``Benchmark``, ``RawStream``, ``eval.py``'s ``MetricSpec``): always run;
- real-data checks, run when the derived root resolves through yavih's config resolver
  (``YAVIH_DATA__NYUV2_DEPTH__DERIVED_ROOT``, the site config ``[data.nyuv2_depth] derived_root``) and is built,
  skipped otherwise; the GeoNet source cross-checks and the re-freeze check additionally need ``geonet_root``.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_bytes, rows, bench = F.spec, F.manifest_bytes, F.rows, F.bench

_geonet = F.site_root(HERE, "geonet_root")
GEONET_ROOT = str(_geonet) if _geonet else None  # prepare-time GeoNet source (nyu_geonet/), for the cross-checks
OFFSETS = np.asarray([244.350, 232.338, 207.016], dtype=np.float32)
REGION = [0.0, 0.125, 1.0, 0.875]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _has(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(scope="module")
def derive():
    return F.load_module(HERE / "prepare" / "derive_targets.py", "nyuv2_derive_targets")


@pytest.fixture(scope="module")
def png():
    return F.load_module(HERE / "prepare" / "recontainer_png.py", "nyuv2_recontainer_png")


# ----------------------------------------------------------------------------- files and yaml


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_folder_specific_files():
    """The folder layout is ``yavih bench lint``'s; these are the files only this folder ships."""
    for name in ("eval.py", "curation.json", "rgb_png.sha256.csv.gz", "prepare/derive_targets.py", "prepare/recontainer_png.py"):
        assert (HERE / name).exists(), name
    assert not (HERE / "classes.csv").exists()  # a grid readout has none
    scripts = sorted(p.name for p in (HERE / "prepare").glob("*.sh"))
    assert [s[:3] for s in scripts] == [f"{i:02d}_" for i in range(1, 8)], scripts


def test_yaml_schema_and_changelog(spec):
    assert spec["schema_version"] == 3 and spec["name"] == "nyuv2_depth" == HERE.name and spec["scorer"] == "grid"
    assert spec["protocol_version"] == spec["changelog"][-1]["version"] == 1
    assert [c["version"] for c in spec["changelog"]] == [1] and spec["changelog"][0]["supersedes"] == []


def test_sha256_and_listing(spec, rows):
    F.assert_manifest_hashes(HERE, spec)
    listing = gzip.open(HERE / "rgb_png.sha256.csv.gz", "rb").read()
    assert _sha(listing) == spec["data"]["derived"]["rgb_png.sha256.csv"]
    items = list(csv.DictReader(io.StringIO(listing.decode("utf-8"), newline="")))
    assert list(items[0]) == ["clip_id", "split", "rel_media", "bytes", "sha256", "pixels_sha256"]
    assert [(i["clip_id"], i["split"], i["rel_media"]) for i in items] == [(r["clip_id"], r["split"], r["rel_media"]) for r in rows]
    assert all(re.fullmatch(r"[0-9a-f]{64}", i["sha256"]) and re.fullmatch(r"[0-9a-f]{64}", i["pixels_sha256"]) for i in items)
    assert sum(int(i["bytes"]) for i in items) == 5_653_048_724


def test_hashes_are_filled(spec):
    hexes = [t["sha256"] for t in spec["targets"].values()] + list(spec["data"]["derived"].values()) + list(spec["source"]["sha256"].values())
    for h in hexes:
        assert re.fullmatch(r"[0-9a-f]{64}", h) and h != "0" * 64, h
    for split, t in spec["targets"].items():
        assert spec["data"]["derived"][t["file"]] == t["sha256"], split


def test_splits_assigned_once(spec, rows):
    assert F.assert_splits_assigned_once(spec, rows) == {"train": 12000, "eval": 654} == spec["budget"]["clips"]


def test_manifest_schema_and_order(spec, rows):
    cols = ["clip_id", "split", "rel_media", "source_mat", "eval_row", "labeled_index", "session", "source"]
    assert list(rows[0].keys()) == cols
    assert set(spec["manifest"]["columns"]) == set(cols) - {"clip_id", "split"}
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(r"[A-Za-z0-9_.#:-]+", i) for i in ids)
    assert all(re.fullmatch(r"[A-Za-z0-9_.#:/-]+", r["rel_media"]) for r in rows)
    assert [r["split"] for r in rows] == ["train"] * 12000 + ["eval"] * 654
    train, ev = rows[:12000], rows[12000:]
    for r in train:
        assert r["rel_media"] == f"rgb/train/{r['clip_id']}.png" and r["source_mat"] == f"nyu_geonet/{r['clip_id']}.mat"
        assert r["eval_row"] == "" and r["labeled_index"] == ""
        assert re.fullmatch(r"(.+?)_r-[0-9.]+-[0-9]+", r["clip_id"]).group(1) == r["session"] and r["source"] == "geonet"
    assert [r["clip_id"] for r in train] == sorted(r["clip_id"] for r in train)
    assert [int(r["eval_row"]) for r in ev] == list(range(654))
    lab = [int(r["labeled_index"]) for r in ev]
    assert lab == sorted(lab) and len(set(lab)) == 654 and max(lab) < 1449
    assert all(r["rel_media"] == f"rgb/eval/{r['clip_id']}.png" and r["source_mat"] == "" for r in ev)
    assert all(r["clip_id"] == f"labeled_{int(r['labeled_index']) + 1:06d}" and r["source"] == "labeled" for r in ev)
    assert not ({r["session"] for r in train} & {r["session"] for r in ev})  # leakage: a freeze-time fact
    assert len({r["session"] for r in train}) == 342


def test_curation_sums_and_json(spec, rows):
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["total_upstream"] == 31570  # listed + excluded == total_upstream == rows: `yavih bench lint` (budget)
    assert cur["excluded"] == {"leakage": 847, "corrupt": 2, "draw_remainder": 18067}
    cj = json.loads((HERE / "curation.json").read_text(encoding="utf-8"))
    up = spec["manifest"]["upstream"]
    assert cj["curation"] == up["curation"] and cj["manifest_sha256"] == spec["manifest"]["sha256"]
    assert cj["upstream_files"] == up["upstream_files"]
    assert cj["train_draw"]["drawn_from"] == up["draw"]
    td = spec["protocol"]["train_draw"]
    assert (td["cap"], td["seed"], td["balance"]) == (12000, 0, "session") and up["sha256"] in up["draw"]
    mm = _make_manifest()
    assert up["draw"] == mm.drawn_from() and up["sha256"] == mm.GEONET_LISTING_SHA256
    assert up["upstream_files"] == {"nyu_geonet listing": mm.GEONET_LISTING_SHA256,
                                    "eval_meta.json (without source_mat)": mm.EVAL_META_SHA256}
    assert cur["leakage_audit"]["against"] == mm.LEAKAGE_AUDIT


def test_metric_names_resolve(spec):
    F.assert_metric_names_resolve(HERE, spec)
    names = list(spec["metrics"]["list"]) + list(spec["metrics"]["floors"]) + list(spec["metrics"]["controls"])
    assert names == ["d1", "d2", "d3", "absrel", "rmse", "n_eval", "d1_floor"]
    assert spec["metrics"]["headline"] == "d1" and spec["metrics"]["ci"] == {"d1": "bootstrap_clip"}


def test_contracts_and_protocol(spec, derive):
    assert F.file_fields(spec) == {"image": {"kind": "image", "media_column": "rel_media", "root": "derived_root",
                                         "layout": "dir", "layout_params": {}, "required": True}}
    ic = spec["input_contract"]
    assert list(ic["handed_over"]) == ["image"] and "PNG" in ic["handed_over"]["image"]
    assert ic["info"] == {"image": ["width", "height", "container", "bytes"]}
    assert ic["facts"] == ["target_region"] and ic["materialize"] is None
    assert spec["output_contract"] == {"shape": "fixed", "coordinates": "media", "target_frame": "full_frame",
                                       "requires": {"pos": ["y", "x"], "streams": ["image"],
                                                    "grid": {"h": 14, "w": 14, "min_cell_coverage": 0.5}, "t_bins": None}}
    p = spec["protocol"]
    assert (p["max_depth"], p["target_size"], p["depth_param"], p["source_size"]) == (10.0, 224, "log", [480, 640])
    assert p["target_region"] == REGION == list(derive.TARGET_REGION) and p["clip_seconds"] is None
    h, w = p["source_size"]
    assert (round(p["target_region"][1] * w), round((1 - p["target_region"][3]) * w)) == (derive.CROP_MARGIN, derive.CROP_MARGIN)
    assert spec["readout_params"] == {"h": 14, "w": 14, "out": 1, "visual_stream": "image"}
    assert {s: (t["file"], t["root"], t["kind"]) for s, t in spec["targets"].items()} == {
        "train": ("train_depth_224.npy", "derived_root", "dense_depth"),
        "eval": ("eval_depth_224.npy", "derived_root", "dense_depth"),
    }
    assert spec["source"]["kind"] == "manual_portal" and set(spec["data"]["roots"]) == {"derived_root", "geonet_root"}


def test_external_entry_matches_probe_py(spec):
    """benchmark.yaml external[0] == eval.py's constants (read with ast: importing eval.py needs the core)."""
    tree = ast.parse((HERE / "eval.py").read_text(encoding="utf-8"))
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
              for t in n.targets if isinstance(t, ast.Name)}
    (ext,) = spec["external"]
    assert set(ext) == {"name", "pip", "version_pin", "used_for"}
    assert (ext["name"], ext["pip"], ext["version_pin"]) == (consts["EXTERNAL_NAME"], consts["EXTERNAL_PIP"], consts["EXTERNAL_VERSION_PIN"])
    # a runnable command (the assert-import prints it as is): pip install loguru -e git+…@<commit>#egg=evals
    assert "a1f14640076e38b8bc07b66d0fe2d01d15691e9d" in ext["pip"] and ext["pip"].split()[:4] == ["pip", "install", "loguru", "-e"]
    assert consts["PROBE3D_METRICS_SHA256"] in ext["used_for"].replace("\n", "")
    # module scope imports: stdlib, numpy, yavih core only
    top = {(n.module or "").split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)} | {
        a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    assert top <= {"__future__", "hashlib", "importlib", "numpy", "yavih"}, top


# ----------------------------------------------------------------------------- README.md / setup.md


def test_readme_facts():
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    assert "@inproceedings{elbanani2024probing" in text and "@inproceedings{silberman2012indoor" in text


def test_setup_facts():
    """setup.md carries no unfilled placeholder (the setup.md structure and contract copies are lint's)."""
    text = (HERE / "setup.md").read_text(encoding="utf-8")
    assert not re.search(r"PNG_[A-Z_]+|CENSUS_SUMMARY|BENCH_VERIFY_OUTPUT", text)


def test_docs_keep_the_facts(spec):
    """Key facts stay documented in README.md + setup.md (digests, rules, the leakage and corrupt-file exclusions)."""
    text = ((HERE / "README.md").read_text(encoding="utf-8") + (HERE / "setup.md").read_text(encoding="utf-8")).lower()
    for t in spec["targets"].values():
        assert t["sha256"] in text
    for h in (spec["manifest"]["sha256"], *spec["source"]["sha256"].values(), spec["data"]["derived"]["rgb_png.sha256.csv"]):
        assert h in text, h
    for needle in ("rawdepths", "session", "variable_names", "nearest-exact", "0.4403", "6919", "21181",
                   "bookstore_0001l", "make_manifest.py"):
        assert needle in text, needle


# ----------------------------------------------------------------------------- prepare rules


def test_nearest_index_tables_pinned(derive):
    assert derive.IDX_EXACT[:8].tolist() == [1, 3, 5, 7, 9, 11, 13, 16]
    assert derive.IDX_NEAREST[:8].tolist() == [0, 2, 4, 6, 8, 10, 12, 15]
    assert derive.IDX_EXACT[-1] == 478 and derive.IDX_NEAREST[-1] == 477


def test_depth_target_rule_on_synthetic_depth(derive):
    rng = np.random.default_rng(0)
    d = rng.uniform(0.0, 12.0, size=(481, 641)).astype(np.float32)
    d[rng.random(d.shape) < 0.1] = 0.0
    d[5, 5] = np.nan
    t = derive.depth_target(d)
    assert t.shape == (224, 224) and t.dtype == np.float32
    assert np.all((t == 0) | ((t > 0) & (t <= 10.0)))
    region = d[:480, :640][:, 80:560]  # protocol.target_region of the full frame
    kept = t > 0
    assert np.array_equal(t[kept], region[derive.IDX_EXACT][:, derive.IDX_EXACT][kept])  # verbatim copies, never interpolated
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch absent: torch reference cross-check skipped")
    assert np.array_equal(t, derive.depth_target_torch(d))


def test_png_codec_roundtrip(png):
    rng = np.random.default_rng(3)
    for shape in ((1, 1, 3), (7, 5, 3), (480, 640, 3)):
        x = rng.integers(0, 256, size=shape, dtype=np.uint8)
        data = png.encode_png(x)
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and data == png.encode_png(x)  # deterministic
        if shape[0] < 100:
            assert np.array_equal(png.decode_png(data), x)
        if importlib.util.find_spec("PIL") is not None:
            from PIL import Image

            im = Image.open(io.BytesIO(data))
            assert im.mode == "RGB" and im.size == (shape[1], shape[0]) and np.array_equal(np.asarray(im), x)
    assert png.pixels_sha256(x) == _sha(x.tobytes())


def test_geonet_recovery_is_exact(png):
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, size=(481, 641, 3), dtype=np.uint8)
    out = png.geonet_rgb(rgb.astype(np.float32) - OFFSETS)
    assert out.dtype == np.uint8 and out.shape == (480, 640, 3) and np.array_equal(out, rgb[:480, :640])


def test_recontainer_end_to_end_on_synthetic_sources(tmp_path, png, rows):
    """recontainer_png.py on a 2-row synthetic manifest (one GeoNet .mat, one eval_rgb.npy row) → PNGs holding the
    exact pixels, a listing, and a passing --verify."""
    scipy_io = pytest.importorskip("scipy.io")
    rng = np.random.default_rng(2)
    tr, ev = rows[0], rows[12000]
    geo = tmp_path / "geonet"
    (geo / "nyu_geonet").mkdir(parents=True)
    rgb = rng.integers(0, 255, size=(481, 641, 3), dtype=np.uint8)
    scipy_io.savemat(geo / tr["source_mat"], {"img": rgb.astype(np.float32) - OFFSETS, "depth": np.ones((481, 641), np.float32)})
    der = tmp_path / "derived"
    der.mkdir()
    evr = rng.integers(0, 256, size=(1, 480, 640, 3), dtype=np.uint8)
    np.save(der / "eval_rgb.npy", evr)
    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tr))
        w.writeheader()
        w.writerows([tr, ev])
    base = [sys.executable, "-I", str(HERE / "prepare" / "recontainer_png.py"), "--manifest", str(man), "--out", str(der)]
    r = subprocess.run([*base, "--geonet-root", str(geo), "--eval-index", str(der), "--workers", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for row, want in ((tr, rgb[:480, :640]), (ev, evr[0])):
        assert np.array_equal(png.decode_png_fast((der / row["rel_media"]).read_bytes()), want)
    r = subprocess.run([*base, "--verify"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["file_sha_ok"] == 2, r.stdout + r.stderr


def _make_manifest():
    return F.load_module(HERE / "make_manifest.py", "nyuv2_make_manifest")


def test_make_manifest_helpers_on_synthetic_inputs():
    """The freeze helpers: GeoNet stem → session → base name, the listing digest (order-free), the eval_meta digest
    (independent of the site-specific ``source_mat``), and the pinned listing / count guards."""
    mm = _make_manifest()
    assert mm.geonet_session("basement_0001a_r-1316653580.484909-1316500621.mat") == "basement_0001a"
    assert mm.session_base("bookstore_0001m") == "bookstore_0001" and mm.session_base("living_room_0064b") == "living_room_0064"
    with pytest.raises(SystemExit):
        mm.geonet_session("not_a_geonet_name.mat")
    assert mm.listing_sha256(["b.mat", "a.mat"]) == _sha(b"a.mat\nb.mat\n")
    meta = {"labeled_indices": [2], "clip_ids": ["labeled_000003"], "sessions": ["kitchen_0003"], "source_mat": "/x/y.mat"}
    assert mm.eval_meta_sha256(meta) == mm.eval_meta_sha256({**meta, "source_mat": "/elsewhere.mat"})
    assert mm.eval_meta_sha256(meta) != mm.eval_meta_sha256({**meta, "sessions": ["kitchen_0004"]})
    with pytest.raises(SystemExit, match="expected 30916"):
        mm.replay_draw(["a_r-1.0-1.mat"])
    with pytest.raises(SystemExit, match="pinned"):
        mm.derive(["a_r-1.0-1.mat"], meta)
    assert len(mm.LEAKAGE_SESSIONS) == 6 and mm.GEONET_TOTAL - mm.LEAKAGE_EXCLUDED - len(mm.PROBE3D_DELETED_INDICES) == mm.POOL


def test_make_manifest_check_mode_matches_committed_files():
    """Re-freeze from the GeoNet listing + eval_meta.json; needs both staged and torch (the draw)."""
    geonet = F.site_root(HERE, "geonet_root", needs=("nyu_geonet",))
    der = F.site_root(HERE, "derived_root", needs=("eval_meta.json",))
    if geonet is None or der is None or not _has("torch"):
        pytest.skip("GeoNet listing (<geonet_root>/nyu_geonet), <derived_root>/eval_meta.json or torch not available here")
    r = subprocess.run([sys.executable, str(HERE / "make_manifest.py"), "--check", "--geonet-dir", str(geonet / "nyu_geonet"),
                        "--eval-meta", str(der / "eval_meta.json")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"check": "identical"' in r.stdout


# ----------------------------------------------------------------------------- output contract (eligibility geometry)


def _cell_coverage(yx: np.ndarray, region=REGION, h: int = 14, w: int = 14) -> float:
    """N_ELIGIBILITY for one row: fraction of the h×w cells tiling ``region`` holding ≥ 1 token centre."""
    y0, x0, y1, x1 = region
    y, x = yx[:, 0], yx[:, 1]
    m = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
    ci = np.floor((y[m] - y0) / (y1 - y0) * h).astype(int)
    cj = np.floor((x[m] - x0) / (x1 - x0) * w).astype(int)
    return len(set(zip(ci.tolist(), cj.tolist(), strict=True))) / (h * w)


def _patch_centres(region, n: int = 14) -> np.ndarray:
    y0, x0, y1, x1 = region
    ys = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
    xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    return np.stack(np.meshgrid(ys, xs, indexing="ij"), -1).reshape(-1, 2)


def test_output_contract_geometry():
    """setup.md "## Output contract" numbers: Resize(224)+CenterCrop(224) of 640×480 shows exactly target_region;
    a squash of the whole frame fills 140/196 cells; resize-256/crop-224 144/196; a pooled token one cell; positions
    reported relative to the crop (the classic mistake) still look eligible — only E_RULE_MISMATCH catches them."""
    centre_square = (0.0, 0.125, 1.0, 0.875)
    assert _cell_coverage(_patch_centres(centre_square)) == 1.0
    assert _cell_coverage(_patch_centres((0.0, 0.0, 1.0, 1.0))) == pytest.approx(140 / 196)
    w = 640 * 256 / 480
    assert _cell_coverage(_patch_centres((0.0625, (w - 224) / 2 / w, 0.9375, 1 - (w - 224) / 2 / w))) == pytest.approx(144 / 196)
    assert _cell_coverage(np.asarray([[0.5, 0.5]])) < 0.5
    from yavih.bridge import grid_pos
    from yavih.media.rules import crop_region

    spatial = {"resize": {"short": 224}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}
    region = crop_region(spatial, 640, 480)
    assert np.allclose(region, centre_square, atol=2e-3)  # 640×480 → 298×224, crop left 37: x0 = 37/298
    pos = grid_pos(1, 14, 14, 0, t_centres=[-1.0], region=region)
    assert _cell_coverage(pos[:, 2:4]) == 1.0


# ----------------------------------------------------------------------------- through the core


def test_benchmark_loads(bench, spec):
    assert bench.name == "nyuv2_depth" and len(bench.protocol_hash) == 64
    assert bench.n_rows("train") == 12000 and bench.n_rows("eval") == 654
    assert bench.roles()["train"] == ["train"] and bench.roles()["eval"] == ["eval"]
    view = bench.view()
    assert view.facts == {"target_region": REGION} and view.streams == {"image": "image"}
    assert not hasattr(view, "manifest")


def _write_synthetic_derived(tmp_path: Path, rows, png) -> tuple[Path, dict]:
    rng = np.random.default_rng(5)
    der = tmp_path / "derived"
    pixels = {}
    for r in (rows[0], rows[1], rows[12000]):
        x = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        (der / r["rel_media"]).parent.mkdir(parents=True, exist_ok=True)
        (der / r["rel_media"]).write_bytes(png.encode_png(x))
        pixels[r["clip_id"]] = x
    return der, pixels


def test_reader_hands_over_raw_png(tmp_path, rows, png):
    """Reader smoke on synthetic raw files: the RawStream is the stored PNG, byte for byte, with facts only."""
    from yavih import Benchmark

    der, _ = _write_synthetic_derived(tmp_path, rows, png)
    b = Benchmark.from_dir(HERE, data={"derived_root": str(der), "geonet_root": str(tmp_path / "absent")})
    for split, rng_rows, first in (("train", range(0, 2), rows[0]), ("eval", range(0, 1), rows[12000])):
        samples = list(b.reader(split, rows=rng_rows, batch_size=1, canaries=0))
        assert [s.row for s in samples] == list(rng_rows) and samples[0].clip_id == first["clip_id"]
        s = samples[0]
        assert list(s.streams) == ["image"] and s.meta == {"facts": {"target_region": REGION}}
        st = s.streams["image"]
        assert st.kind == "image"
        assert Path(st._real_path) == der / first["rel_media"]
        assert st.bytes() == (der / first["rel_media"]).read_bytes() == st.open().read()
        assert (st.info["width"], st.info["height"], st.info["container"]) == (640, 480, "png")
        assert st.info["bytes"] == (der / first["rel_media"]).stat().st_size


def test_reader_missing_png_is_missing_media(tmp_path, rows, png):
    from yavih import Benchmark
    try:
        from yavih.media import MediaMissing
    except ImportError:
        from yavih.sample import MediaMissing

    der, _ = _write_synthetic_derived(tmp_path, rows, png)
    b = Benchmark.from_dir(HERE, data={"derived_root": str(der), "geonet_root": str(tmp_path / "absent")})
    with pytest.raises(MediaMissing):
        list(b.reader("train", rows=range(2, 3), batch_size=1, canaries=0))[0].streams["image"].bytes()


def _eval_module():
    return F.load_module(HERE / "eval.py", "nyuv2_probe")


PROTO = {"max_depth": 10.0, "target_size": 224, "depth_param": "log", "source_size": [480, 640], "target_region": REGION}


class _FakeProbe3D:
    """Stand-in for ``evals.utils.metrics`` with Probe3D's evaluate_depth semantics (plumbing test only)."""

    @staticmethod
    def evaluate_depth(depth_pr, depth_gt, image_average=False):
        import torch

        valid = (depth_gt > 0).float()
        depth_pr = depth_pr * valid
        n = valid.sum(dim=(1, 2)).clamp(min=1)
        th = torch.maximum(depth_gt / depth_pr.clamp(min=1e-9), depth_pr / depth_gt.clamp(min=1e-9))
        out = {f"d{k}": ((th < 1.25**k).float() * valid).sum(dim=(1, 2)) / n for k in (1, 2, 3)}
        out["rmse"] = (((depth_gt - depth_pr) ** 2 * valid).sum(dim=(1, 2)) / n).sqrt()
        return out


def test_probe_py_metrics_and_missing_package():
    """eval.py: METRICS name Probe3D as external; without the package the assert-import fails with the pip line;
    with a Probe3D-semantics module the per-image numbers equal yavih's transcription."""
    pytest.importorskip("torch")
    mod = _eval_module()
    assert set(mod.METRICS) == {"d1", "d2", "d3", "rmse", "d1_floor"}
    assert all(m.external == "evals" and m.readouts == frozenset({"grid"}) for m in mod.METRICS.values())
    assert mod.METRICS["rmse"].higher_is_better is False and mod.METRICS["d1_floor"].kind == "floor"
    if not _has("evals"):
        with pytest.raises(Exception) as ei:  # ExternalMissing → exit 3, message = the pip line
            mod.load_probe3d()
        assert "probe3d" in str(ei.value).lower() or "evals" in str(ei.value)
    from yavih.metrics import depth as D

    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 9.5, size=(5, 224, 224)).astype(np.float32)
    gt[rng.random(gt.shape) < 0.2] = 0.0
    pred = (gt + rng.normal(0, 0.5, gt.shape)).clip(0.1, 12).astype(np.float32)
    per = mod.per_image(pred, gt, PROTO, _FakeProbe3D)
    ref = D.depth_per_image(pred, gt, PROTO)
    for k in ("d1", "d2", "d3", "rmse"):
        assert np.allclose(per[k], ref[k], atol=1e-4), k


def test_probe3d_real_package_matches_transcription():
    """When Probe3D (``evals``) is installed as setup.md says, its numbers equal yavih's transcription."""
    if not _has("evals"):
        pytest.skip("Probe3D not installed (" + F.load_yaml(HERE)["external"][0]["pip"] + ")")
    mod = _eval_module()
    from yavih.metrics import depth as D

    rng = np.random.default_rng(1)
    gt = rng.uniform(0.5, 9.5, size=(4, 224, 224)).astype(np.float32)
    pred = (gt * rng.uniform(0.7, 1.3, gt.shape)).astype(np.float32)
    per = mod.per_image(pred, gt, PROTO)
    ref = D.depth_per_image(pred, gt, PROTO)
    for k in ("d1", "d2", "d3", "rmse"):
        assert np.allclose(per[k], ref[k], atol=1e-4), k


def test_bridge_template_encodes_synthetic_sample(tmp_path, rows, png):
    """The template's decode/positions on a synthetic PNG: (feats, pos, used) shapes, t = -1, all 196 cells over
    target_region filled, used == {"image": {}}."""
    pytest.importorskip("PIL")
    from yavih import Benchmark

    der, _ = _write_synthetic_derived(tmp_path, rows, png)
    b = Benchmark.from_dir(HERE, data={"derived_root": str(der), "geonet_root": str(tmp_path / "absent")})
    tmpl = F.load_module(HERE / "bridge_template.py", "nyuv2_bridge_template")
    br = tmpl.Nyuv2DepthBridge(b.view(), {"layer": -1})
    samples = list(b.reader("train", rows=range(0, 2), batch_size=2, canaries=0))
    feats, pos, used = F.encode_prepared(br, samples)
    assert feats.shape[:2] == (2, 196) and pos.shape == (2, 196, 4) and used == [{"image": {}}] * 2
    assert np.all(pos[:, :, 0] == 0) and np.all(pos[:, :, 1] == -1)
    assert _cell_coverage(pos[0, :, 2:4]) == 1.0
    x = br._preprocess(samples[0].streams["image"])
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    d = br.describe()
    assert d["views"] == 1 and set(d["input"]) == {"image"} and "decode" not in d


# ----------------------------------------------------------------------------- real data


def _derived_root() -> str | None:
    der = F.site_root(HERE, "derived_root", needs=("rgb_png.sha256.csv",))
    return str(der) if der else None


def test_real_pngs_match_listing_and_sources(png):
    """Spot check on the built derived_root: listing pinned, PNG sha256 as listed, pixels = the upstream recovery."""
    der = _derived_root()
    if der is None:
        pytest.skip("no built nyuv2_depth derived_root (rgb/ + rgb_png.sha256.csv) here")
    listing = Path(der, "rgb_png.sha256.csv").read_bytes()
    assert listing == gzip.open(HERE / "rgb_png.sha256.csv.gz", "rb").read()
    items = list(csv.DictReader(io.StringIO(listing.decode(), newline="")))
    for it in (items[0], items[11999], items[12000], items[-1]):
        data = Path(der, it["rel_media"]).read_bytes()
        assert _sha(data) == it["sha256"]
        assert png.pixels_sha256(png.decode_png_fast(data)) == it["pixels_sha256"]
    ev = Path(der, "eval_rgb.npy")
    if ev.is_file():
        evr = np.load(ev, mmap_mode="r")
        assert png.pixels_sha256(np.asarray(evr[653])) == items[-1]["pixels_sha256"]
    mat = Path(GEONET_ROOT or "", "nyu_geonet", items[0]["clip_id"] + ".mat")
    if GEONET_ROOT and mat.is_file() and importlib.util.find_spec("scipy") is not None:
        assert png.pixels_sha256(png.load_geonet_rgb(str(mat))) == items[0]["pixels_sha256"]


def test_targets_on_real_data(spec, derive):
    der = _derived_root()
    if der is None or not os.path.isfile(os.path.join(der, "eval_depth_224.npy")):
        pytest.skip("derived targets not present (prepare/04_derive_targets.sh)")
    te = np.load(os.path.join(der, "eval_depth_224.npy"), mmap_mode="r")
    assert te.shape == (654, 224, 224) and te.dtype == np.float32
    src = os.path.join(der, "eval_depth.npy")
    if os.path.isfile(src):
        s = np.load(src, mmap_mode="r")
        for k in (0, 653):
            assert np.array_equal(te[k], derive.depth_target(np.asarray(s[k])))
    tt = os.path.join(der, "train_depth_224.npy")
    if os.path.isfile(tt) and GEONET_ROOT and os.path.isdir(os.path.join(GEONET_ROOT, "nyu_geonet")):
        import scipy.io

        arr = np.load(tt, mmap_mode="r")
        man = list(csv.DictReader(io.StringIO(gzip.open(HERE / "manifest.csv.gz").read().decode(), newline="")))
        for k in (0, 11999):
            d = scipy.io.loadmat(os.path.join(GEONET_ROOT, man[k]["source_mat"]), variable_names=["depth"])["depth"]
            assert np.array_equal(arr[k], derive.depth_target(d))
