#!/usr/bin/env python
"""Freeze the ``av_syncbench`` manifest from the two public metadata files of the AV-SyncBench release.

Inputs (read only, nothing is downloaded; sha256-pinned, a different file is refused):

- ``AV_SyncBench/hf_temporal_challenge/metadata.jsonl`` (36 925 lines, sha256 ``f96dcf8a…``) and
  ``AV_SyncBench/hf_timbre_edit/metadata.jsonl`` (821 lines, sha256 ``93dcba15…``), both inside the split archive
  ``AV_SyncBench.tar.gz.part_a{a..g}`` of the ModelScope dataset ``coming245/AVSyncBench`` at revision
  ``86c06579529a6e7b2cafb0dc386a50152a37fb98`` (``prepare/01_download.sh`` fetches and checks the seven parts,
  ``prepare/02_extract.sh`` extracts them). One JSON object per line: ``sample_id``, ``video_id``,
  ``original_video``, ``edited_audio`` (paths relative to the file's directory), ``edit_type``,
  ``target_attribute``, [``source_attribute``], ``category``, ``duration``, ``task_type``.
- ``<media_root>/AV_SyncBench/{hf_temporal_challenge,hf_timbre_edit}/{original_video,edited_audio}/`` for presence
  only (one ``scandir`` per directory; a disclosure, never a filter). ``--no-media`` skips it.

The rules (every row is one original-versus-perturbed pair, the upstream evaluator's unit):

1. **Rows.** Every line of both files, minus the 3 lines of ``hf_timbre_edit/metadata.jsonl`` that repeat an earlier
   line byte for byte (lines 374, 421, 422 repeat 345, 378, 379; ``curation.excluded.duplicate_line``): 36 925 +
   818 = 37 743 rows, one split ``test``, ordered by ``rel_video`` then ``upstream_line`` (the pairs of one
   original video are consecutive).
2. **Columns** ``clip_id,split,task,level,direction,category,video_id,rel_video,rel_edited,duration,
   target_attribute,source_attribute,upstream_line``. ``clip_id`` = ``sample_id``. ``task`` / ``level`` /
   ``direction`` are parsed from ``target_attribute`` (temporal) or ``edit_type`` (semantic):
   ``audio_shift: advance 50ms`` -> ``offset`` / ``50ms`` / ``advance``; ``audio_speed: 0.80x`` -> ``speed`` /
   ``0.80x``; ``audio_local_jitter: ... level=L1 ...`` -> ``jitter`` / ``L1``; ``speech_attribute_edit`` ->
   ``voice_timbre`` / the target attribute; ``instrument_timbre_edit`` -> ``instrument_timbre`` / the target
   instrument. ``rel_video`` / ``rel_edited`` = ``AV_SyncBench/<dir>/`` + the line's own relative path.
   ``target_attribute`` and ``source_attribute`` are the raw strings, JSON-encoded (``null`` when absent);
   ``duration`` is the raw number; ``upstream_line`` is ``<t|s><1-based line>`` (``t`` temporal, ``s`` timbre).
   Together with the file a row came from they re-create every metadata line byte for byte (the folder test does).
3. Asserted on every run: the per-task row, level and video counts below (``EXPECTED``), unique ``sample_id`` after
   the duplicates, every ``edited_audio`` file read by one row only.
4. Writes ``manifest.csv.gz`` (byte-identical re-runs, ``yavih.freeze``) and prints the ``manifest.upstream`` block and
   the digests ``benchmark.yaml`` carries. ``--check`` re-derives, compares with the committed manifest and the yaml
   facts, writes nothing and exits 1 on drift.

The release differs from the paper's Table 1 (15 000 offset / 15 000 speed / 7 569 jitter / 592 + 229 timbre
pairs): it holds 14 600 / 14 570 / 7 755 / 589 + 229 distinct pairs (the 3 repeated lines are voice-timbre pairs). The manifest is the release, as published.

Usage::

    python "$(yavih bench path av_syncbench)/make_manifest.py"                 # re-freeze from the configured media_root
    python "$(yavih bench path av_syncbench)/make_manifest.py" --check         # re-derive and compare, write nothing
    python "$(yavih bench path av_syncbench)/make_manifest.py" --no-media --metadata <dir with the two
        hf_*/metadata.jsonl> --check                                          # from the metadata files alone
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
NAME = "av_syncbench"
SPLIT = "test"
ARCHIVE_DIR = "AV_SyncBench"
MODELSCOPE_REVISION = "86c06579529a6e7b2cafb0dc386a50152a37fb98"
#: the two metadata files: directory -> (sha256, lines, row-id prefix)
FILES: dict[str, tuple[str, int, str]] = {
    "hf_temporal_challenge": ("f96dcf8ab8e445e1519bcfdf10a8a413d0df9aec5da76da08483bf3d80aa3212", 36925, "t"),
    "hf_timbre_edit": ("93dcba15699ce6653ee7540f813b463d963c4c90d4948c9bdcfeadfe36fe6c61", 821, "s"),
}
#: lines (1-based) that repeat an earlier line byte for byte: line -> the line it repeats
DUPLICATE_LINES: dict[str, dict[int, int]] = {"hf_timbre_edit": {374: 345, 421: 378, 422: 379}}
#: the key order of a metadata line, per file (json.dumps of the dict re-creates the line)
KEYS: dict[str, tuple[str, ...]] = {
    "hf_temporal_challenge": ("sample_id", "video_id", "original_video", "edited_audio", "edit_type", "target_attribute", "category",
                              "duration", "task_type"),
    "hf_timbre_edit": ("sample_id", "video_id", "original_video", "edited_audio", "edit_type", "target_attribute", "source_attribute",
                       "category", "duration", "task_type"),
}
TASK_TYPE = {"hf_temporal_challenge": "temporal_challenge", "hf_timbre_edit": "semantic_challenge"}
SEMANTIC_TASK = {"speech_attribute_edit": "voice_timbre", "instrument_timbre_edit": "instrument_timbre"}
EDIT_TYPE = {v: k for k, v in SEMANTIC_TASK.items()}
MANIFEST_COLUMNS = ["clip_id", "split", "task", "level", "direction", "category", "video_id", "rel_video", "rel_edited", "duration",
                    "target_attribute", "source_attribute", "upstream_line"]
OFFSET_LEVELS = ("50ms", "100ms", "200ms", "300ms", "500ms")
SPEED_LEVELS = ("0.80x", "0.83x", "0.87x", "0.91x", "0.95x", "1.05x", "1.10x", "1.15x", "1.20x", "1.25x")
JITTER_LEVELS = ("L1", "L2", "L3")
#: rows per task and per (task, level) in the pinned release
EXPECTED: dict[str, Any] = {
    "tasks": {"offset": 14600, "speed": 14570, "jitter": 7755, "voice_timbre": 589, "instrument_timbre": 229},
    "levels": {"offset": {lv: 2920 for lv in OFFSET_LEVELS}, "speed": {lv: 1457 for lv in SPEED_LEVELS}, "jitter": {lv: 2585 for lv in JITTER_LEVELS}},
    "videos": {"offset": 1460, "speed": 1457, "jitter": 1536, "voice_timbre": 323, "instrument_timbre": 229},
}
TOTAL_UPSTREAM = sum(n for _, n, _ in FILES.values())

_SHIFT = re.compile(r"^audio_shift: (advance|delay) (\d+ms)$")
_SPEED = re.compile(r"^audio_speed: (\d\.\d\dx)$")
_JITTER = re.compile(r"^audio_local_jitter: seg=\d+s level=(L[123]) ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _pinned(path: Path, expected: str) -> bytes:
    if not path.is_file():
        raise SystemExit(f"missing input {path} (setup.md 'Obtain the data': extract the release first)")
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != expected:
        raise SystemExit(f"{path}: sha256 {got} != pinned {expected}; a different input is refused (re-freezing would be a new protocol_version)")
    return raw


def parse_task(directory: str, rec: dict[str, Any]) -> tuple[str, str, str]:
    """``(task, level, direction)`` of one metadata record (``direction`` is ``advance`` / ``delay`` for offsets, else ``""``)."""
    if directory == "hf_timbre_edit":
        task = SEMANTIC_TASK.get(rec["edit_type"])
        if task is None:
            raise SystemExit(f"{rec['sample_id']}: unknown edit_type {rec['edit_type']!r}")
        return task, str(rec["target_attribute"]), ""
    t = str(rec["target_attribute"])
    if m := _SHIFT.match(t):
        return "offset", m.group(2), m.group(1)
    if m := _SPEED.match(t):
        return "speed", m.group(1), ""
    if m := _JITTER.match(t):
        return "jitter", m.group(1), ""
    raise SystemExit(f"{rec['sample_id']}: unparsed target_attribute {t!r}")


def load_lines(directory: str, raw: bytes) -> list[tuple[int, dict[str, Any]]]:
    """``[(line, record)]`` of one metadata file, without the pinned duplicate lines (checked to repeat their line)."""
    lines = raw.decode("utf-8").split("\n")
    if lines[-1] != "":
        raise SystemExit(f"{directory}/metadata.jsonl: no trailing newline")
    lines = lines[:-1]
    if len(lines) != FILES[directory][1]:
        raise SystemExit(f"{directory}/metadata.jsonl: {len(lines)} lines, expected {FILES[directory][1]}")
    dups = DUPLICATE_LINES.get(directory, {})
    out = []
    for i, line in enumerate(lines, start=1):
        if i in dups:
            if line != lines[dups[i] - 1]:
                raise SystemExit(f"{directory}/metadata.jsonl line {i} does not repeat line {dups[i]}")
            continue
        rec = json.loads(line)
        if tuple(rec) != KEYS[directory] or json.dumps(rec) != line:
            raise SystemExit(f"{directory}/metadata.jsonl line {i}: unexpected keys or formatting")
        out.append((i, rec))
    return out


def derive(meta: dict[str, bytes]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = []
    for directory, (_, _, prefix) in FILES.items():
        for line, rec in load_lines(directory, meta[directory]):
            if rec["task_type"] != TASK_TYPE[directory]:
                raise SystemExit(f"{rec['sample_id']}: task_type {rec['task_type']!r} in {directory}")
            if directory == "hf_temporal_challenge" and rec["edit_type"] != "":
                raise SystemExit(f"{rec['sample_id']}: non-empty edit_type {rec['edit_type']!r} in {directory}")
            task, level, direction = parse_task(directory, rec)
            rows.append({
                "clip_id": rec["sample_id"], "split": SPLIT, "task": task, "level": level, "direction": direction, "category": rec["category"],
                "video_id": rec["video_id"], "rel_video": f"{ARCHIVE_DIR}/{directory}/{rec['original_video']}",
                "rel_edited": f"{ARCHIVE_DIR}/{directory}/{rec['edited_audio']}", "duration": json.dumps(rec["duration"]),
                "target_attribute": json.dumps(rec["target_attribute"]), "source_attribute": json.dumps(rec.get("source_attribute")),
                "upstream_line": f"{prefix}{line}",
            })
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate sample_id after dropping the pinned duplicate lines")
    edited = Counter(r["rel_edited"] for r in rows)
    if max(edited.values()) != 1:
        raise SystemExit("an edited_audio file is shared by two rows")
    video_of: dict[str, str] = {}
    for r in rows:
        if video_of.setdefault(r["rel_video"], r["video_id"]) != r["video_id"]:
            raise SystemExit(f"{r['rel_video']}: one file, two video_ids")
    tasks = Counter(r["task"] for r in rows)
    levels = {t: dict(sorted(Counter(r["level"] for r in rows if r["task"] == t).items())) for t in EXPECTED["levels"]}
    videos = {t: len({r["video_id"] for r in rows if r["task"] == t}) for t in EXPECTED["tasks"]}
    if dict(tasks) != EXPECTED["tasks"] or levels != {t: dict(sorted(v.items())) for t, v in EXPECTED["levels"].items()} or videos != EXPECTED["videos"]:
        raise SystemExit(f"counts differ from the pinned release: tasks {dict(tasks)}, levels {levels}, videos {videos}")
    offset_dirs = Counter((r["level"], r["direction"]) for r in rows if r["task"] == "offset")
    if set(offset_dirs.values()) != {1460} or len(offset_dirs) != 10:
        raise SystemExit(f"offset (level, direction) counts {dict(offset_dirs)}")
    rows.sort(key=lambda r: (r["rel_video"], r["upstream_line"][0], int(r["upstream_line"][1:])))
    excluded = {"duplicate_line": sum(len(v) for v in DUPLICATE_LINES.values())}
    stats = {"tasks": dict(tasks), "videos_per_task": videos, "original_video_files": len(video_of),
             "video_ids": len({r["video_id"] for r in rows}), "categories": dict(sorted(Counter(r["category"] for r in rows).items()))}
    return rows, {"excluded": excluded, **stats}


def listing_dir(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise SystemExit(f"media directory missing: {directory} (extract the release first; setup.md 'Obtain the data')")
    return {e.name for e in os.scandir(directory)}


def media_bytes(media_root: Path, rows: list[dict[str, str]]) -> int:
    """Bytes of every distinct media file the rows read (``budget.media_gb``)."""
    paths = {r["rel_video"] for r in rows} | {r["rel_edited"] for r in rows}
    return sum((media_root / p).stat().st_size for p in paths if (media_root / p).is_file())


def freeze(meta: dict[str, bytes], media_root: Path | None) -> dict[str, Any]:
    rows, stats = derive(meta)
    if media_root is not None:
        listed = {}
        for directory in FILES:
            for sub in ("original_video", "edited_audio"):
                listed[f"{ARCHIVE_DIR}/{directory}/{sub}"] = listing_dir(media_root / ARCHIVE_DIR / directory / sub)

        def present(rel: str) -> bool:
            d, f = rel.rsplit("/", 1)
            return f in listed.get(d, ())

        staged = sum(1 for r in rows if present(r["rel_video"]) and present(r["rel_edited"]))
        stats["media_bytes"] = media_bytes(media_root, rows)
    else:
        staged = len(rows)
    manifest = csv_bytes(MANIFEST_COLUMNS, ([r[c] for c in MANIFEST_COLUMNS] for r in rows))
    upstream_files = {f"{ARCHIVE_DIR}/{d}/metadata.jsonl": sha for d, (sha, _, _) in FILES.items()}
    return {
        "manifest_bytes": manifest,
        "manifest": {"sha256": sha256_bytes(manifest), "rows": len(rows)},
        "budget_clips": {SPLIT: len(rows)},
        "upstream": {
            "source": f"ModelScope coming245/AVSyncBench @ {MODELSCOPE_REVISION[:7]}: the two metadata.jsonl files of the release archive",
            "sha256": FILES["hf_temporal_challenge"][0],
            "rows": TOTAL_UPSTREAM,
            "upstream_files": upstream_files,
            "curation": {"total_upstream": TOTAL_UPSTREAM, "per_split": {SPLIT: {"listed": len(rows), "staged": staged, "missing": len(rows) - staged}},
                         "excluded": stats["excluded"]},
        },
        "stats": {k: v for k, v in stats.items() if k != "excluded"},
    }


def _yaml_facts() -> dict[str, Any]:
    import yaml

    y = yaml.safe_load((HERE / "benchmark.yaml").read_text(encoding="utf-8"))
    up = y["manifest"]["upstream"]
    return {"budget_clips": y["budget"]["clips"], "sha256": y["manifest"]["sha256"],
            "upstream": {k: up.get(k) for k in ("source", "sha256", "rows", "upstream_files", "curation")}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media-root", type=Path, default=None,
                    help="the directory holding AV_SyncBench/ (default: the folder's configured media_root)")
    ap.add_argument("--metadata", type=Path, default=None,
                    help="a directory holding hf_temporal_challenge/metadata.jsonl and hf_timbre_edit/metadata.jsonl "
                         "(default: <media-root>/AV_SyncBench)")
    ap.add_argument("--no-media", action="store_true", help="freeze from --metadata alone: no presence check (staged = listed)")
    ap.add_argument("--check", action="store_true", help="re-derive and compare with the committed manifest and yaml facts; write nothing")
    ap.add_argument("--out", type=Path, default=HERE, help="where to write manifest.csv.gz (default: this folder)")
    ns = ap.parse_args(argv)
    if ns.no_media and ns.metadata is None:
        raise SystemExit("--no-media needs --metadata (a directory holding the two hf_*/metadata.jsonl files)")
    media_root = None
    if not ns.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", ns.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure [data.{NAME}] media_root: {e}") from None
    meta_dir = ns.metadata or (media_root / ARCHIVE_DIR)
    meta = {d: _pinned(meta_dir / d / "metadata.jsonl", sha) for d, (sha, _, _) in FILES.items()}
    out = freeze(meta, media_root)
    report = {k: v for k, v in out.items() if k != "manifest_bytes"}
    if ns.check:
        committed = gzip.decompress((HERE / "manifest.csv.gz").read_bytes())
        facts = _yaml_facts()
        drift = []
        if committed != out["manifest_bytes"]:
            drift.append("manifest.csv.gz differs")
        if facts["sha256"] != out["manifest"]["sha256"]:
            drift.append("benchmark.yaml manifest.sha256 differs")
        if facts["budget_clips"] != out["budget_clips"]:
            drift.append(f"budget.clips {facts['budget_clips']} != {out['budget_clips']}")
        if facts["upstream"] != out["upstream"]:
            drift.append(f"manifest.upstream differs: {facts['upstream']} != {out['upstream']}")
        print(json.dumps(report, indent=1))
        print("manifest matches" if not drift else "DRIFT: " + "; ".join(drift), f"({out['manifest']['sha256']})", file=sys.stderr)
        return 0 if not drift else 1
    ns.out.mkdir(parents=True, exist_ok=True)
    wrote = write_if_changed(ns.out / "manifest.csv.gz", gzip_bytes(out["manifest_bytes"]))
    print(json.dumps(report, indent=1))
    print(f"manifest.csv.gz {'written' if wrote else 'unchanged'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
