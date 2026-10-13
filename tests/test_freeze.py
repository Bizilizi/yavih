"""yavih.freeze: byte-reproducible gzip / CSV writers, ``write_if_changed`` and the seeded class-balanced draw.

The draw decides which train rows six folders freeze, so its kept indices are pinned below as literal vectors
(several label sets x seeds x caps); a change to ``class_balanced_subsample`` that moves any of them would change a
re-frozen manifest.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import random
from pathlib import Path

import pytest

from yavih import freeze

ALL_FOLDERS = ("audioset20k", "ave", "avsync", "esc50", "fsd50k", "in1k", "k400", "nyuv2_depth", "ssv2", "vggsounder", "vggss_box")
BENCHMARKS = Path(freeze.__file__).resolve().parent / "benchmarks"


def test_make_manifests_use_yavih_freeze():
    """Every make_manifest.py uses yavih.freeze instead of a private copy of its helpers."""
    for folder in ALL_FOLDERS:
        src = (BENCHMARKS / folder / "make_manifest.py").read_text(encoding="utf-8")
        for name in ("gzip_bytes", "csv_bytes", "write_if_changed", "class_balanced_subsample"):
            assert f"def {name}(" not in src and f"def _{name}(" not in src, f"{folder}: private copy of {name}"
        assert "from yavih.freeze import" in src, folder


# ----------------------------------------------------------------------------- gzip_bytes

_rng = random.Random(11)
GZIP_PAYLOADS = [
    b"",
    b"a",
    b"clip_id,label,split\n",
    "ünïcödé,\"quoted, field\"\r\n".encode() * 37,
    bytes(_rng.getrandbits(8) for _ in range(5000)),  # incompressible
    b"0123456789abcdef" * 70_000,  # > 1 MiB, several deflate blocks
]


@pytest.mark.parametrize("k", range(len(GZIP_PAYLOADS)))
def test_gzip_bytes_round_trips_and_is_deterministic(k):
    data = GZIP_PAYLOADS[k]
    out = freeze.gzip_bytes(data)
    assert gzip.decompress(out) == data
    assert freeze.gzip_bytes(data) == out


def test_gzip_header_is_reproducible():
    """mtime 0 and no FNAME flag in the header: the same bytes on every machine and every day."""
    out = freeze.gzip_bytes(b"x\n")
    assert out[:3] == b"\x1f\x8b\x08" and out[3] & 0x08 == 0 and out[4:8] == b"\0\0\0\0"
    assert out[8] == 2  # XFL 2 = maximum compression (level 9)


def test_gzip_bytes_of_committed_manifests():
    """Re-gzipping each committed manifest body reproduces the committed .gz bytes."""
    for folder in ALL_FOLDERS:
        gz = (BENCHMARKS / folder / "manifest.csv.gz").read_bytes()
        assert freeze.gzip_bytes(gzip.decompress(gz)) == gz, folder


# ----------------------------------------------------------------------------- csv_bytes

CSV_CASES = [
    (["index", "name"], [], b"index,name\n"),
    (["index", "name"], [["0", "dog"], ["1", "cat"]], b"index,name\n0,dog\n1,cat\n"),
    (["clip_id", "labels", "split"], [["a", "x;y;z", "train"], ["b", "", "eval"]], b"clip_id,labels,split\na,x;y;z,train\nb,,eval\n"),
    (["c"], [["comma, inside"], ['quote " inside'], ["new\nline"], ["carriage\rreturn"], ["crlf\r\nend"], [" lead"]],
     b'c\n"comma, inside"\n"quote "" inside"\n"new\nline"\n"carriage\rreturn"\n"crlf\r\nend"\n lead\n'),
    (["n", "f"], [[1, 2.5], [0, -0.125], [None, True]], b"n,f\n1,2.5\n0,-0.125\n,True\n"),
    (["ünï", "名前"], [["café", "犬"], ["\t tab", "trailing "]], "ünï,名前\ncafé,犬\n\t tab,trailing \n".encode()),
]


@pytest.mark.parametrize("k", range(len(CSV_CASES)))
def test_csv_bytes_pinned(k):
    header, rows, want = CSV_CASES[k]
    assert freeze.csv_bytes(header, rows) == want


def test_csv_bytes_of_committed_classes():
    """Each committed classes.csv is reproduced by csv_bytes from its own parsed rows."""
    for folder in ALL_FOLDERS:
        path = BENCHMARKS / folder / "classes.csv"
        if not path.is_file():
            continue
        raw = path.read_bytes()
        header, *rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
        assert freeze.csv_bytes(header, rows) == raw, folder


# ----------------------------------------------------------------------------- write_if_changed


def test_write_if_changed(tmp_path):
    p = tmp_path / "classes.csv"
    assert freeze.write_if_changed(p, b"a\n") is True and p.read_bytes() == b"a\n"
    mtime = p.stat().st_mtime_ns
    assert freeze.write_if_changed(p, b"a\n") is False and p.stat().st_mtime_ns == mtime
    assert freeze.write_if_changed(p, b"b\n") is True and p.read_bytes() == b"b\n"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["classes.csv"]  # no .tmp left behind


# ----------------------------------------------------------------------------- the seeded draw

LABEL_SETS: dict[str, list[str]] = {
    "single_class": ["only"] * 25,
    "balanced_3x10": [c for c in "abc" for _ in range(10)],
    "interleaved": list("abcabcabcabcab"),
    "uneven": ["a"] * 40 + ["b"] * 3 + ["c"] * 1 + ["d"] * 17,
    "unsorted_names": [n for n in ["zebra", "Apple", "apple", "_x", "10", "9"] for _ in range(4)],
    "singletons": [f"s{i}" for i in range(12)],
}

#: (label set, seed, cap, kept indices), computed with this implementation and pinned.
GOLDEN: list[tuple[str, int, int, list[int]]] = [
    ('single_class', 0, 3, [6, 16, 19]),
    ('single_class', 0, 12, [2, 5, 6, 8, 11, 12, 14, 16, 17, 19, 21, 24]),
    ('single_class', 42, 3, [2, 12, 17]),
    ('single_class', 42, 12, [0, 2, 6, 8, 10, 12, 17, 18, 20, 21, 22, 23]),
    ('single_class', 2147483647, 3, [3, 6, 7]),
    ('single_class', 2147483647, 12, [1, 3, 4, 5, 6, 7, 9, 16, 17, 20, 21, 22]),
    ('balanced_3x10', 0, 3, [4, 13, 24]),
    ('balanced_3x10', 0, 15, [1, 3, 4, 5, 7, 12, 13, 14, 17, 19, 20, 24, 25, 28, 29]),
    ('balanced_3x10', 42, 3, [2, 13, 27]),
    ('balanced_3x10', 42, 15, [1, 2, 4, 6, 8, 13, 14, 15, 18, 19, 20, 21, 23, 25, 27]),
    ('balanced_3x10', 2147483647, 3, [1, 17, 28]),
    ('balanced_3x10', 2147483647, 15, [0, 1, 4, 7, 9, 13, 14, 15, 17, 18, 20, 21, 23, 24, 28]),
    ('interleaved', 0, 3, [5, 10, 12]),
    ('interleaved', 0, 7, [0, 2, 3, 5, 10, 12, 13]),
    ('interleaved', 42, 3, [4, 6, 8]),
    ('interleaved', 42, 7, [4, 6, 8, 9, 11, 12, 13]),
    ('interleaved', 2147483647, 3, [3, 10, 11]),
    ('interleaved', 2147483647, 7, [0, 3, 4, 8, 9, 10, 11]),
    ('uneven', 0, 3, [4, 41, 43]),
    ('uneven', 0, 30, [2, 4, 5, 6, 11, 14, 16, 17, 20, 21, 26, 29, 39, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 58, 59, 60]),
    ('uneven', 42, 3, [22, 40, 43]),
    ('uneven', 42, 30, [0, 2, 3, 7, 8, 9, 13, 14, 15, 18, 22, 32, 38, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 53, 54, 55, 56, 59, 60]),
    ('uneven', 2147483647, 3, [31, 42, 43]),
    ('uneven', 2147483647, 30, [1, 9, 12, 17, 20, 21, 23, 24, 25, 31, 32, 34, 37, 40, 41, 42, 43, 44, 45, 46, 48, 51, 53, 54, 55, 56, 57, 58, 59, 60]),
    ('unsorted_names', 0, 3, [7, 16, 20]),
    ('unsorted_names', 0, 12, [0, 1, 6, 7, 8, 9, 12, 15, 16, 17, 20, 22]),
    ('unsorted_names', 42, 3, [4, 18, 22]),
    ('unsorted_names', 42, 12, [0, 3, 4, 7, 9, 10, 13, 14, 18, 19, 21, 22]),
    ('unsorted_names', 2147483647, 3, [4, 19, 20]),
    ('unsorted_names', 2147483647, 12, [0, 1, 4, 6, 8, 10, 12, 15, 16, 19, 20, 23]),
    ('singletons', 0, 3, [0, 1, 10]),
    ('singletons', 0, 6, [0, 1, 2, 3, 10, 11]),
    ('singletons', 42, 3, [0, 1, 10]),
    ('singletons', 42, 6, [0, 1, 2, 3, 10, 11]),
    ('singletons', 2147483647, 3, [0, 1, 10]),
    ('singletons', 2147483647, 6, [0, 1, 2, 3, 10, 11]),
]


def _large_sets() -> dict[str, list[str]]:
    r = random.Random(7)
    zipf = [f"c{min(int(r.paretovariate(1.2)), 60):02d}" for _ in range(3000)]
    many = [f"k{r.randrange(300)}" for _ in range(2000)]
    return {"zipf": zipf, "many_classes": many}


#: (label set, seed, cap, sha256 of the comma-joined kept indices) for the larger sets.
GOLDEN_DIGESTS: list[tuple[str, int, int, str]] = [
    ('zipf', 0, 100, "3c5007500b682ed47a133dee322f10871eeafa4ae30a5277bcc902f3b5476193"),
    ('zipf', 0, 1000, "02394a6addbc4cffa7945784b0b37fb9bcddef9906d540a13231d9102bc52915"),
    ('zipf', 1234, 100, "40279b4ba94cf8cd9ae25a54c358e0368f61bd7c9687f03c12d4a9af64c9f0bf"),
    ('zipf', 1234, 1000, "4da63ca5e97c68be46586a9bb5792d320e68543443ac2b52fb135c7477c2dc24"),
    ('many_classes', 0, 100, "d1c96686212c25b54597fad6d0d7c30a18926ea31e517b26546c8e863e39ad1e"),
    ('many_classes', 0, 1000, "8861a4e5b137f935b4fca10593e209a1baebfae4a90e436cdd1868e8ed73ba6a"),
    ('many_classes', 1234, 100, "f30048d2c4086dc8ef02ae902f74a0f38bac831eee1f4b7cacad08d8521c4cc6"),
    ('many_classes', 1234, 1000, "705471805f2c0ae472d2fd36d963887ee46ab5a7000eee630189a1f96f4ded82"),
]


@pytest.mark.parametrize("name,seed,cap,want", GOLDEN)
def test_class_balanced_subsample_golden(name, seed, cap, want):
    pytest.importorskip("torch", reason="the draw uses torch.randperm (freeze time only)")
    assert freeze.class_balanced_subsample(list(LABEL_SETS[name]), cap, seed) == want


@pytest.mark.parametrize("name,seed,cap,digest", GOLDEN_DIGESTS)
def test_class_balanced_subsample_golden_large(name, seed, cap, digest):
    pytest.importorskip("torch", reason="the draw uses torch.randperm (freeze time only)")
    kept = freeze.class_balanced_subsample(_large_sets()[name], cap, seed)
    assert len(kept) == cap and kept == sorted(set(kept))
    assert hashlib.sha256(",".join(map(str, kept)).encode()).hexdigest() == digest


@pytest.mark.parametrize("cap", [-1, 0, 25, 30])
def test_class_balanced_subsample_without_subsampling(cap):
    """``max_clips <= 0`` or ``>= len(labels)`` keeps every row (no torch needed)."""
    assert freeze.class_balanced_subsample(["only"] * 25, cap, 0) == list(range(25))
    assert freeze.class_balanced_subsample([], cap, 0) == []


def test_class_balanced_subsample_is_balanced_and_seeded():
    pytest.importorskip("torch", reason="the draw uses torch.randperm (freeze time only)")
    labels = LABEL_SETS["uneven"]
    keep = freeze.class_balanced_subsample(labels, 20, 0)
    per = {c: sum(labels[i] == c for i in keep) for c in "abcd"}
    assert per == {"a": 8, "b": 3, "c": 1, "d": 8}  # round-robin: small classes are exhausted, the rest share
    assert freeze.class_balanced_subsample(labels, 20, 1) != keep
    assert freeze.class_balanced_subsample(labels, 20, 0) == keep


def test_freeze_imports_stdlib_only():
    """yavih.freeze is importable without numpy, yaml or torch (freeze scripts and the core stay light)."""
    import subprocess
    import sys

    code = ("import sys\nfor m in ('numpy', 'yaml', 'torch'):\n    sys.modules[m] = None\n"
            "import importlib.util\nspec = importlib.util.spec_from_file_location('f', sys.argv[1])\n"
            "mod = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(mod)\nprint('ok')\n")
    proc = subprocess.run([sys.executable, "-I", "-c", code, freeze.__file__], capture_output=True, text=True, timeout=60)
    assert proc.stdout.strip() == "ok", proc.stderr
