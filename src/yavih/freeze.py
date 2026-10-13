"""Shared helpers of the folders' ``make_manifest.py`` freeze scripts (a re-freeze is byte-identical).

Every ``benchmarks/<name>/make_manifest.py`` uses these:

- :func:`gzip_bytes` — the ``manifest.csv.gz`` bytes: no file name and ``mtime=0`` in the gzip header, level 9,
  so a re-freeze is byte-identical.
- :func:`csv_bytes` — a header plus rows as UTF-8 CSV with ``\\n`` line ends (``classes.csv``, the manifest body).
- :func:`write_if_changed` — write through ``<name>.tmp`` + ``os.replace``; leave an identical file untouched.
- :func:`class_balanced_subsample` — the seeded class-balanced train draw. It decides which train rows are
  frozen for fsd50k, in1k, k400, nyuv2_depth, ssv2 and vggsounder; ``tests/test_freeze.py`` pins its output.

Stdlib only at import time; ``torch`` is imported inside :func:`class_balanced_subsample` (freeze time only;
core never imports torch).
"""

from __future__ import annotations

import gzip
import io
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["gzip_bytes", "csv_bytes", "write_if_changed", "class_balanced_subsample"]


def gzip_bytes(data: bytes) -> bytes:
    """``data`` gzipped with an empty file name, ``mtime=0`` and level 9: a re-freeze is byte-identical."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(data)
    return buf.getvalue()


def csv_bytes(header: Sequence[str], rows: Iterable[Sequence[object]]) -> bytes:
    """``header`` then ``rows`` as UTF-8 CSV with minimal quoting and ``\\n`` line ends.

    Written by hand rather than with ``csv.writer``, whose minimal quoting differs across Python versions
    (3.13 started quoting a lone ``\\r``): a field is quoted iff it contains ``,``, ``"``, ``\\n`` or ``\\r``,
    with ``"`` doubled; ``None`` is the empty field, everything else is ``str()``.
    """
    def field(v: object) -> str:
        s = "" if v is None else str(v)
        if any(c in s for c in ',"\n\r'):
            return '"' + s.replace('"', '""') + '"'
        return s

    lines = [",".join(field(v) for v in row) for row in (header, *rows)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_if_changed(path: Path, data: bytes) -> bool:
    """Write ``data`` to ``path`` (via ``<name>.tmp`` + ``os.replace``) unless it already holds exactly these bytes.

    Returns whether the file was written.
    """
    if path.is_file() and path.read_bytes() == data:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def class_balanced_subsample(labels: list[str], max_clips: int, seed: int) -> list[int]:
    """Deterministic, seeded, class-balanced subsample of ``range(len(labels))``.

    Groups indices by ``labels[i]``, shuffles each class's index bucket with ONE seeded ``torch.Generator``
    consumed in sorted class order (so the draw does not depend on dict iteration order), then round-robins one
    index at a time across classes (also sorted order) until ``max_clips`` indices are collected or every bucket is
    exhausted. Returns the kept indices SORTED ascending. ``max_clips <= 0`` or ``>= len(labels)`` → no
    subsampling. The "class" is whatever the folder keys the draw by (fsd50k: the first label; nyuv2_depth: the
    capture session).
    """
    import torch  # freeze-time only (core never imports torch)

    n = len(labels)
    if max_clips <= 0 or max_clips >= n:
        return list(range(n))

    by_class: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    generator = torch.Generator()
    generator.manual_seed(seed)
    classes_sorted = sorted(by_class)
    for label in classes_sorted:
        bucket = by_class[label]
        perm = torch.randperm(len(bucket), generator=generator).tolist()
        by_class[label] = [bucket[p] for p in perm]

    pointers = {label: 0 for label in classes_sorted}
    selected: list[int] = []
    while len(selected) < max_clips:
        advanced = False
        for label in classes_sorted:
            if len(selected) >= max_clips:
                break
            bucket = by_class[label]
            pos = pointers[label]
            if pos < len(bucket):
                selected.append(bucket[pos])
                pointers[label] = pos + 1
                advanced = True
        if not advanced:
            break

    selected.sort()
    return selected
