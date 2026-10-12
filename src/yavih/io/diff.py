"""``yavih features diff``: ``{rows_compared, max_abs, max_rel_l2, verdict}`` between two
split directories (bit-identical rerun checks).

Rows are matched by manifest row; ``b`` is random-accessed through its tars while ``a`` streams,
so the comparison never holds more than two clips in memory.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from yavih.io.format import member_basename
from yavih.io.validate import _rel_l2, features_block, iter_members, list_shards

__all__ = ["diff_features", "RowIndex", "VERDICTS"]

VERDICTS: tuple[str, ...] = ("identical", "within_tol", "differs", "incompatible", "empty")


class RowIndex:
    """Random access to ``{row:09d}.feat.npy`` members of a split directory by manifest row,
    through the sidecar ``members`` lists and lazily opened tars."""

    def __init__(self, split_dir: str | os.PathLike) -> None:
        self.split_dir = Path(split_dir)
        self.shards = list_shards(self.split_dir)
        if not self.shards:
            raise FileNotFoundError(f"{self.split_dir}: no finished shards (sidecar + tar)")
        self.row_to_tar: dict[int, Path] = {}
        self.sidecars: list[dict] = []
        for _idx, sc_path, tar_path in self.shards:
            with open(sc_path, encoding="utf-8") as fh:
                sc = json.load(fh)
            self.sidecars.append(sc)
            for m in sc["members"]:
                self.row_to_tar[int(m["row"])] = tar_path
        self._open: dict[Path, tarfile.TarFile] = {}

    @property
    def features(self) -> dict:
        return features_block(self.sidecars)

    def rows(self) -> set[int]:
        return set(self.row_to_tar)

    def feat(self, row: int) -> np.ndarray | None:
        tar_path = self.row_to_tar.get(row)
        if tar_path is None:
            return None
        tf = self._open.get(tar_path)
        if tf is None:
            tf = self._open[tar_path] = tarfile.open(tar_path, "r:")
        try:
            ti = tf.getmember(member_basename(row) + ".feat.npy")
        except KeyError:
            return None
        return np.load(io.BytesIO(tf.extractfile(ti).read()), allow_pickle=False)  # type: ignore[union-attr]

    def close(self) -> None:
        for tf in self._open.values():
            tf.close()
        self._open.clear()

    def __enter__(self) -> RowIndex:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def diff_features(a: str | os.PathLike, b: str | os.PathLike, *, rel_tol: float = 1e-3) -> dict:
    """Compare the rows present in both split directories. ``verdict``: ``identical`` (bit-equal),
    ``within_tol`` (``max_rel_l2 ≤ rel_tol``), ``differs``, ``incompatible`` (N/D/dtype differ) or
    ``empty`` (no common rows)."""
    a_dir, b_dir = Path(a), Path(b)
    with RowIndex(b_dir) as ib:
        fa = RowIndex(a_dir)
        fa.close()
        feats_a, feats_b = fa.features, ib.features
        out: dict[str, Any] = {
            "a": str(a_dir),
            "b": str(b_dir),
            "rows_compared": 0,
            "max_abs": 0.0,
            "max_rel_l2": 0.0,
            "verdict": "empty",
            "rel_tol": rel_tol,
            "only_in_a": len(fa.rows() - ib.rows()),
            "only_in_b": len(ib.rows() - fa.rows()),
            "features_a": {k: feats_a.get(k) for k in ("dtype", "N", "D", "pos_mode")},
            "features_b": {k: feats_b.get(k) for k in ("dtype", "N", "D", "pos_mode")},
            "worst_row": None,
        }
        if not fa.rows() or not ib.rows():
            return out  # a side without any row is "empty", whatever its (null) features block says
        if any(feats_a.get(k) != feats_b.get(k) for k in ("dtype", "N", "D")):
            out["verdict"] = "incompatible"
            return out
        common = fa.rows() & ib.rows()
        if not common:
            return out
        max_abs = 0.0
        max_rel = 0.0
        worst: int | None = None
        n = 0
        for rec in iter_members(a_dir, rows=common, with_pos=False):
            fb = ib.feat(rec.row)
            if fb is None or fb.shape != rec.feat.shape:
                out["verdict"] = "incompatible"
                out["worst_row"] = rec.row
                return out
            n += 1
            d = float(np.abs(rec.feat.astype(np.float64) - fb.astype(np.float64)).max()) if rec.feat.size else 0.0
            r = _rel_l2(rec.feat, fb)
            if d > max_abs:
                max_abs = d
            if r > max_rel:
                max_rel, worst = r, rec.row
        out.update({"rows_compared": n, "max_abs": max_abs, "max_rel_l2": max_rel, "worst_row": worst})
        out["verdict"] = "identical" if max_abs == 0.0 else ("within_tol" if max_rel <= rel_tol else "differs")
        return out
