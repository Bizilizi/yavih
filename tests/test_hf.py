"""Hugging Face data roots (``data.roots.<root>.hf``, ``yavih bench snapshot``): provenance, never hashed."""

from __future__ import annotations

import json
import sys
import types

import pytest

from yavih import hf as H

REV = "4a0055b21c183257fd9b65ee8d1cd11f13362f0f"


def test_parse_hf_validates_every_key():
    assert H.parse_hf({"repo": "o/n", "revision": REV}, "x") == {"repo": "o/n", "revision": REV, "repo_type": "dataset", "allow_patterns": None}
    for bad, match in (({"repo": "o/n", "revision": "main"}, "40-hex"), ({"repo": "noslash", "revision": REV}, "owner"),
                       ({"repo": "o/n", "revision": REV, "repo_type": "space"}, "repo_type"),
                       ({"repo": "o/n", "revision": REV, "allow_patterns": []}, "allow_patterns"), ({"revision": REV}, "required"),
                       ({"repo": "o/n", "revision": REV, "branch": "x"}, "unknown")):
        with pytest.raises(ValueError, match=match):
            H.parse_hf(bad, "x")


@pytest.fixture()
def fake_hub(monkeypatch):
    calls = []

    def snapshot_download(*, repo_id, repo_type, revision, local_dir, allow_patterns=None):
        calls.append((repo_id, repo_type, revision, allow_patterns))
        from pathlib import Path

        (Path(local_dir) / "data").mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "data" / "a.tar").write_bytes(b"a")
        (Path(local_dir) / "index.json").write_text("{}")
        (Path(local_dir) / ".cache").mkdir(exist_ok=True)
        (Path(local_dir) / ".cache" / "bookkeeping").write_text("x")
        return local_dir

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


def test_snapshot_stamps_and_status_follows_the_root(tmp_path, fake_hub):
    hf = H.parse_hf({"repo": "o/n", "revision": REV, "allow_patterns": ["*.tar", "index.json"]}, "x")
    root = tmp_path / "root"
    assert H.snapshot_status(root, hf)["state"] == "absent"
    assert H.snapshot(root, hf, dry_run=True) == {"root": str(root), **hf, "dry_run": True} and not fake_hub
    stamp = H.snapshot(root, hf)
    assert fake_hub == [("o/n", "dataset", REV, ["*.tar", "index.json"])]
    assert stamp["files"] == ["data/a.tar", "index.json"]  # the stamp and huggingface_hub's .cache are not data
    assert json.loads((root / H.STAMP).read_text())["revision"] == REV
    assert H.snapshot_status(root, hf)["state"] == "ok"
    (root / "data" / "a.tar").unlink()
    assert H.snapshot_status(root, hf)["state"] == "incomplete"
    other = {**hf, "revision": "f" * 40}
    assert H.snapshot_status(root, other)["state"] == "other_revision"
    (root / H.STAMP).unlink()
    assert H.snapshot_status(root, hf)["state"] == "unstamped"


def test_a_root_hf_block_is_parsed_unhashed_and_checked_against_the_source(tmp_path):
    import yaml

    from yavih import testing as T
    from yavih.benchmark import Benchmark, SchemaError

    folder = T.make_synthetic_benchmark(tmp_path, "clf", n_train=4, n_eval=2, media="none")
    base = Benchmark.from_dir(folder)
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["data"]["roots"]["media_root"]["hf"] = {"repo": "o/n", "revision": REV}
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    b = Benchmark.from_dir(folder)
    assert b.spec.data.roots["media_root"].hf["revision"] == REV
    assert (b.protocol_hash, b.input_hash) == (base.protocol_hash, base.input_hash)  # provenance only
    y["source"]["revision"] = "e" * 40
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    with pytest.raises(SchemaError, match="source.revision"):
        Benchmark.from_dir(folder)
    y["source"]["revision"] = REV
    y["data"]["roots"]["media_root"]["hf"]["revision"] = "main"
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    with pytest.raises(SchemaError, match="40-hex"):
        Benchmark.from_dir(folder)
