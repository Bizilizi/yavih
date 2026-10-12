"""Shared pytest fixtures for yavih.

- ``YAVIH_TESTING=1`` for every test (non-pinned ProbeConfigs are gated on it).
- ``synthetic_benchmark``: a minimal benchmark folder on disk written DIRECTLY (manifest.csv.gz +
  classes.csv + benchmark.yaml valid under docs/benchmark-folder.md) so it depends on no other yavih module.
- ``write_synthetic_benchmark(folder, ...)``: the same as a parametrizable helper.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


@pytest.fixture(autouse=True)
def _yavih_testing_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Tests run with YAVIH_TESTING=1 and never read the developer's real config (neither the
    user file nor a project yavih.toml), their exported YAVIH_* variables, nor the developer's ``~/.yavih`` (site census and data_status caches; ``yavih_home`` below gives a per-test one)."""
    home = tmp_path_factory.getbasetemp() / "yavih_home_isolated"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("YAVIH_HOME", str(home))
    monkeypatch.setenv("YAVIH_TESTING", "1")
    monkeypatch.setenv("YAVIH_CONFIG", str(REPO / "tests" / "_no_such_config.toml"))  # no project file, no discovery
    xdg = tmp_path_factory.getbasetemp() / "xdg_isolated"  # no user file (~/.config/yavih/config.toml is never read)
    xdg.mkdir(exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    for k in [k for k in os.environ if k.startswith("YAVIH_") and k not in ("YAVIH_HOME", "YAVIH_TESTING", "YAVIH_CONFIG")]:
        monkeypatch.delenv(k, raising=False)  # a developer's exported YAVIH_RUNS / YAVIH_DATA__* never leaks in


@pytest.fixture
def yavih_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated ``$YAVIH_HOME`` under tmp_path."""
    home = tmp_path / "yavih_home"
    home.mkdir()
    monkeypatch.setenv("YAVIH_HOME", str(home))
    return home


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("YAVIH_RUNS", str(runs))
    return runs


# ----------------------------------------------------------------------------- synthetic benchmark


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class SyntheticBenchmark:
    """What the fixture hands a test: paths, the yaml dict as written, the rows, and the hashes
    computed by the fixture itself (independent of ``yavih.hashing``) so hashing/benchmark tests can
    assert against them."""

    folder: Path
    name: str
    spec: dict[str, Any]
    rows: list[dict[str, str]]  # manifest rows in file order
    classes: list[dict[str, str]]
    manifest_bytes: bytes  # decompressed csv bytes
    classes_bytes: bytes
    manifest_sha256: str
    classes_sha256: str
    media_root: Path
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def yaml_path(self) -> Path:
        return self.folder / "benchmark.yaml"

    @property
    def manifest_hash(self) -> str:
        # The manifest hash, computed here on purpose (not via yavih.hashing) as an independent oracle.
        payload = b"manifest:" + self.manifest_sha256.encode() + b"\nclasses:" + self.classes_sha256.encode()
        return _sha256(payload)

    def split_rows(self, split: str) -> list[dict[str, str]]:
        return [r for r in self.rows if r["split"] == split]


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def write_synthetic_benchmark(
    root: Path,
    *,
    name: str = "synth_clf",
    n_train: int = 12,
    n_eval: int = 6,
    n_classes: int = 3,
    readout: str = "clf",
    media_root: Path | None = None,
    write_media_files: bool = False,
) -> SyntheticBenchmark:
    """Write ``<root>/<name>/{benchmark.yaml, manifest.csv.gz, classes.csv}`` valid under docs/benchmark-folder.md for a
    single-label ``clf`` benchmark with one ``video`` stream (``layout: dir``). ``write_media_files`` touches
    empty placeholder files under ``media_root`` so presence checks see them (they are NOT decodable)."""
    if readout != "clf":
        raise NotImplementedError("the scaffold fixture only writes clf benchmarks; use yavih.testing.make_synthetic_benchmark for other readouts")
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    media_root = media_root or (root / "media" / name)

    classes = [{"index": str(i), "name": f"class_{i:02d}"} for i in range(n_classes)]
    classes_bytes = _csv_bytes(["index", "name"], [[c["index"], c["name"]] for c in classes])

    rows: list[dict[str, str]] = []
    for split, n in (("train", n_train), ("val", n_eval)):
        for k in range(n):
            label = classes[k % n_classes]["name"]
            cid = f"{split}_{k:04d}"
            rows.append({"clip_id": cid, "split": split, "label": label, "rel_media": f"{split}/{label}/{cid}.mp4"})
    manifest_bytes = _csv_bytes(["clip_id", "split", "label", "rel_media"], [[r["clip_id"], r["split"], r["label"], r["rel_media"]] for r in rows])

    # mtime=0 so a re-written fixture is byte-identical ("re-freeze is byte-identical").
    with open(folder / "manifest.csv.gz", "wb") as raw_fh, gzip.GzipFile(fileobj=raw_fh, mode="wb", mtime=0) as fh:
        fh.write(manifest_bytes)
    (folder / "classes.csv").write_bytes(classes_bytes)

    if write_media_files:
        for r in rows:
            p = media_root / r["rel_media"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    spec: dict[str, Any] = {
        "schema_version": 2,
        "name": name,
        "aliases": [f"{name}_alias"],
        "title": "Synthetic single-label classification fixture",
        "readout": readout,
        "protocol_version": 2,
        "changelog": [
            {"version": 1, "date": "2026-10-07", "reason": "synthetic test fixture (tests/conftest.py)", "invalidates": False},
            {"version": 2, "date": "2026-10-08", "reason": "synthetic fixture, version 2", "invalidates": False},
        ],
        "source": {"kind": "derived", "urls": [], "licence": "synthetic", "bytes": 0, "sha256": {}, "notes": "written by tests/conftest.py"},
        "budget": {
            "clips": {"train": n_train, "val": n_eval},
            "media_gb": 0.0,
            "extract_min_per_1k_clips_ref": 0.1,
            "probe_minutes_ref": {"clf": 1},
        },
        "manifest": {
            "file": "manifest.csv.gz",
            "sha256": _sha256(manifest_bytes),
            "classes": "classes.csv",
            "classes_sha256": _sha256(classes_bytes),
            "key": "clip_id",
            "label": {
                "column": "label",
                "kind": "single",
                "sep": ";",
                "class_key": "name",
                "sets": None,
                "train_set": None,
                "allow_unknown": False,
                "empty_ok": False,
            },
            "columns": {"rel_media": {"type": "str", "role": "media", "stream": "video"}},
            "upstream": {
                "source": "tests/conftest.py synthetic generator",
                "sha256": _sha256(manifest_bytes),
                "rows": n_train + n_eval,
                "note": "all rows synthetic",
                "upstream_files": {},
                "curation": {
                    "total_upstream": n_train + n_eval,
                    "per_split": {
                        "train": {"listed": n_train, "staged": n_train, "missing": 0},
                        "val": {"listed": n_eval, "staged": n_eval, "missing": 0},
                    },
                    "excluded": {},
                    "short_clips": 0,
                    "leakage_audit": None,
                },
            },
        },
        "splits": {"train": ["train"], "eval": ["val"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95},
        "streams": {
            "video": {
                "kind": "video",
                "media_column": "rel_media",
                "root": "media_root",
                "layout": "dir",
                "layout_params": {},
                "required": True,
            }
        },
        "targets": None,
        "data": {
            "roots": {"media_root": {"default": "${data.root}/" + name, "description": "{train,val}/<class>/*.mp4"}},
            "tar_index": None,
            "derived": {},
        },
        # The placeholder files are not decodable, so no header fact is guaranteed.
        "input_contract": {
            "handed_over": {"video": "the stored mp4 placeholder exactly as stored (an empty file; nothing decodable)"},
            "info": {"video": []},
            "facts": [],
            "materialize": None,
        },
        "output_contract": {
            "shape": "fixed",
            "coordinates": "media",
            "target_frame": None,
            "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None},
        },
        "external": [],
        "protocol": {"clip_seconds": 10.0, "train_draw": None, "views": "none"},
        "metrics": {
            "headline": "acc",
            "list": ["acc", "n_eval"],
            "floors": ["acc_floor"],
            "controls": [],
            "chance": {"acc": round(1.0 / n_classes, 6)},
            "ci": {"acc": "binomial"},
            "bootstrap": {"samples": 2000, "seed": 0},
            "aggregation": None,
        },
        "readout_params": {},
    }
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    return SyntheticBenchmark(
        folder=folder,
        name=name,
        spec=spec,
        rows=rows,
        classes=classes,
        manifest_bytes=manifest_bytes,
        classes_bytes=classes_bytes,
        manifest_sha256=_sha256(manifest_bytes),
        classes_sha256=_sha256(classes_bytes),
        media_root=media_root,
    )


@pytest.fixture
def synthetic_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticBenchmark:
    """A tiny clf benchmark folder (12 train / 6 val rows, 3 classes) with its media root exported
    through ``YAVIH_DATA__SYNTH_CLF__MEDIA_ROOT`` (config precedence)."""
    sb = write_synthetic_benchmark(tmp_path, write_media_files=True)
    monkeypatch.setenv(f"YAVIH_DATA__{sb.name.upper()}__MEDIA_ROOT", str(sb.media_root))
    return sb


@pytest.fixture
def venv_python() -> str:
    """The interpreter running the tests (works for the project venv and any other)."""
    import sys

    return sys.executable


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs a CUDA device (skipped without one)")
    config.addinivalue_line("markers", "slow: longer than a few seconds")
    config.addinivalue_line("markers", "integration: exercises several modules end to end (deselect with -m 'not integration')")
    os.environ.setdefault("YAVIH_TESTING", "1")
