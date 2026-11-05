#!/usr/bin/env python
"""avsync — step 3: materialize one shifted mp4 per row (``input_contract.materialize``; setup.md).

The offset δ between audio and video IS the label, so it cannot be left to the bridge: for every manifest row this
script runs the pinned ffmpeg recipe of ``benchmark.yaml`` ``input_contract.materialize`` (kind ``av_offset_mux``)
on the VGGSound source pair and writes ``<derived_root>/<rel_media>``, an mp4 holding

- video: the source mp4 over ``[v0, v1) = [1 + δ/2, 9 + δ/2)`` s, every source frame whose pts falls in the window,
  re-encoded with libx264 (pinned arguments, native resolution, ``-vsync passthrough`` + the source time base, so
  each frame keeps its timing: output time = source pts − v0);
- audio: the source wav samples ``[a0, a1) = [round((1 − δ/2)·sr), a0 + 8·sr)``, stored losslessly as ALAC.

Inside the output, audio and video are therefore offset by exactly δ. Every output is VERIFIED by decoding it
(PyAV): the audio is bit-identical to the source window, the video has exactly the source frames of the window, at
the source pts − v0 (tolerance ``--pts-tol``, default 1 ms: the mp4 edit list stores a non-zero first-frame delay in
the movie time scale, 1/1000 s), and a sparse PSNR against the source frames is recorded (re-encode quality). Only a
verified output is renamed into place; its sha256 goes to the materialization log (one JSON line per row, with the
recipe sha256 so a changed recipe never reuses an old file).

The log is one file PER WRITER: ``<derived_root>/_materialize/log.<i>-of-<n>.jsonl`` for ``--shard i/n``, else
``_materialize/log.jsonl``. Shards never append to a shared file (O_APPEND is not atomic across NFS clients). Every reader
(resume, ``--write-record``) merges all ``_materialize/log*.jsonl`` and SKIPS lines it cannot parse (counted in the
summary as ``log_unparseable_lines``): the renamed-into-place file and its logged sha256 are the source of truth, and
a row whose record was lost is simply materialized again.

Refusals by design (the row stays missing, ``"refused": true`` in its record, and the run still exits 0): a source
window outside the source media (no padding: the audio window outside the wav, a video window starting before 0
or ending after the source video's last frame + one frame period, or an output video track that is not
``length`` ± one frame long — the freeze's ``required_seconds: 10.0`` is a CONTAINER duration, so a source whose
video track is shorter, e.g. 9.9 s, is refused for δ ≥ 1.5 here and stays a missing row), a source wav that is
not mono 16 kHz PCM. Build refusals: an ffmpeg build other than ``tool.version_pin`` (exit 2), a libx264 build
other than the pinned ``tool.pins.libx264`` (exit 2, checked in the first output's SEI). Any other failed row (ffmpeg error, a
missing source file, an output that fails the frame / pts / audio verification) exits 1.

Usage (``FOLDER=$(yavih bench path avsync)``)::

    # the validation run (216 rows = 8 per split x offset bin, seeded), ≈ 1 min on 28 cores
    python "$FOLDER/prepare/03_materialize.py" --rows 216 --workers 7
    # the full run, e.g. as a SLURM array of 64 shards (each shard writes its own log and resumes from all logs):
    python "$FOLDER/prepare/03_materialize.py" --all --shard ${SLURM_ARRAY_TASK_ID}/64 --workers 3
    # a per-row sha256 record of this site's files, <derived_root>/_materialize/materialized.sha256.csv.gz
    # (re-hashes every recorded file; --allow-missing records the verified rows when some rows stay missing, e.g.
    # refused by design). Per-site, never shipped: the encoded bytes repeat on one host but can differ between
    # hosts, so the folder pins the recipe (recipe_sha256, carried by every log line), not the output bytes.
    python "$FOLDER/prepare/03_materialize.py" --write-record [--allow-missing]

Roots: ``--source-root`` (VGGSound tree; default: the folder's ``media_root``, ``yavih config get avsync.media_root``)
and ``--out-root`` (default: the folder's ``derived_root``, ``yavih config get avsync.derived_root``), both resolved
by ``yavih.config`` only when the flag is absent. Needs ffmpeg on PATH plus PyAV, soundfile and numpy (``yavih[media]``)
for verification; imports nothing else of yavih (stdlib + pyyaml + those three, plus ``yavih.config`` for a default root). ``--workers 1`` runs the rows in this process (no pool).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
NAME = "avsync"
SOURCE_RATE = 16000
PSNR_EVERY = 40  # compare every 40th frame with the source (re-encode quality, informational)
LOG_GLOB = "log*.jsonl"  # every writer's log under <derived_root>/_materialize/ (one file per shard)
RECORD = "materialized.sha256.csv.gz"  # --write-record's per-site record under <derived_root>/_materialize/ (never shipped)


class Refusal(ValueError):
    """A row refused by design (no padding, wrong source format): it stays a missing row, the run exits 0."""


# ----------------------------------------------------------------------------- recipe


def load_spec() -> dict:
    import yaml

    return yaml.safe_load((FOLDER / "benchmark.yaml").read_text(encoding="utf-8"))


def x264_build(mat: dict) -> str:
    """The libx264 build the recipe pins (``tool.pins.libx264.value``; ``yavih doctor --bench avsync`` checks it too)."""
    return str(mat["tool"]["pins"]["libx264"]["value"])


def recipe_sha256(mat: dict) -> str:
    """sha256 over the canonical JSON of the materialize block without ``tool.pins`` (what decides the bytes; the
    pins are checks of the build, enforced on the first output, so declaring them in the yaml left the recipe of
    every materialized file unchanged). The reproducibility pin of the folder: every log line carries it, and a
    changed recipe never reuses an old file. The output bytes themselves are not pinned (they can differ between
    hosts with the same pinned build)."""
    core = dict(mat)
    if isinstance(core.get("tool"), dict):
        core["tool"] = {k: v for k, v in core["tool"].items() if k != "pins"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def windows(mat: dict, delta: float, rate: int = SOURCE_RATE) -> dict:
    """The row's source windows: video ``[v0, v1)`` seconds, audio ``[a0, a1)`` samples (no rounding of v0/v1:
    δ is a multiple of 0.5, so v0 is a multiple of 0.25 s and exact in binary)."""
    v0 = float(mat["pivot"]) + float(mat["video"]) * delta
    a0s = float(mat["pivot"]) + float(mat["audio"]) * delta
    a0 = round(a0s * rate)
    if abs(a0 - a0s * rate) > 1e-6:
        raise ValueError(f"audio window start {a0s} s is not on the {rate} Hz sample grid")
    return {"v0": v0, "v1": v0 + float(mat["length"]), "a0": a0, "a1": a0 + round(float(mat["length"]) * rate)}


def command(mat: dict, video: str, audio: str, out: str, w: dict) -> list[str]:
    vals = {"video": video, "audio": audio, "v0": repr(w["v0"]), "v1": repr(w["v1"]), "a0": str(w["a0"]), "a1": str(w["a1"])}
    tool = mat["tool"]
    return [tool["name"], *(a.format(**vals) for a in tool["video_args"]), *(a.format(**vals) for a in tool["audio_args"]), out]


def ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout
    m = re.match(r"ffmpeg version (\S+)", out)
    return m.group(1) if m else "unknown"


# ----------------------------------------------------------------------------- manifest / roots


def manifest_rows() -> list[dict[str, str]]:
    with gzip.open(FOLDER / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_root(key: str) -> Path:
    """The folder's root ``key`` via yavih's config resolver (``--data``, ``YAVIH_DATA__AVSYNC__<KEY>``, config
    ``[data.avsync] <key>``, else the yaml default); exits naming the config key when it cannot be resolved."""
    from yavih import config

    try:
        return config.folder_root(FOLDER, key)
    except config.ConfigError as exc:
        raise SystemExit(f"{exc} (or pass --source-root / --out-root)") from None


def stratified(rows: list[dict[str, str]], n_rows: int, seed: int = 0) -> list[int]:
    """Manifest indices, ``n_rows`` round-robin over every (split, offset_bin) cell, seeded order within a cell."""
    import numpy as np

    cells: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        cells[(r["split"], int(r["offset_bin"]))].append(i)
    rng = np.random.default_rng(seed)
    order = {k: [v[j] for j in rng.permutation(len(v))] for k, v in sorted(cells.items())}
    out: list[int] = []
    pos = 0
    while len(out) < n_rows and any(pos < len(v) for v in order.values()):
        for v in order.values():
            if len(out) < n_rows and pos < len(v):
                out.append(v[pos])
        pos += 1
    return sorted(out)


# ----------------------------------------------------------------------------- one row


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode_video(path: str, want_frames: set[int] | None = None):
    """``(times float64 [n], {index: rgb uint8 [H, W, 3]} for want_frames, stream facts)``."""
    import av
    import numpy as np

    with av.open(path) as f:
        st = f.streams.video[0]
        st.thread_type = "AUTO"
        times, imgs = [], {}
        for k, fr in enumerate(f.decode(st)):
            times.append(float(fr.time))
            if want_frames is not None and k in want_frames:
                imgs[k] = fr.to_ndarray(format="rgb24")
        facts = {
            "width": st.codec_context.width,
            "height": st.codec_context.height,
            "fps": float(st.average_rate) if st.average_rate else None,
            "stream_duration_s": float(st.duration * st.time_base) if st.duration is not None else None,
            "container_duration_s": float(f.duration / 1e6) if f.duration is not None else None,
            "pix_fmt": st.codec_context.pix_fmt,
        }
    return np.asarray(times, dtype=np.float64), imgs, facts


def _decode_audio_int16(path: str):
    import av
    import numpy as np

    with av.open(path) as f:
        st = f.streams.audio[0]
        parts = [fr.to_ndarray() for fr in f.decode(st)]
        rate, ch = st.codec_context.sample_rate, st.codec_context.channels
    a = np.concatenate(parts, axis=1) if parts else np.zeros((ch, 0), dtype=np.int16)
    return a, rate, ch


def materialize_row(task: dict) -> dict:
    """Run the recipe on one row, verify, rename into place. Never raises: failures come back as ``ok: false``."""
    import numpy as np
    import soundfile as sf

    r, mat, src_root, out_root = task["row"], task["mat"], Path(task["src_root"]), Path(task["out_root"])
    out = out_root / r["rel_media"]
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.mp4")
    rec: dict = {"clip_id": r["clip_id"], "split": r["split"], "rel_media": r["rel_media"], "recipe_sha256": task["recipe_sha256"], "ok": False}
    try:
        delta = float(r["offset_seconds"])
        video, audio = src_root / r[mat["source_columns"]["video"]], src_root / r[mat["source_columns"]["audio"]]
        info = sf.info(str(audio))
        if info.samplerate != SOURCE_RATE or info.channels != 1 or info.subtype != "PCM_16":
            raise Refusal(f"source wav is {info.samplerate} Hz x {info.channels} {info.subtype}, expected 16000 Hz mono PCM_16")
        w = windows(mat, delta, info.samplerate)
        if w["a0"] < 0 or w["a1"] > info.frames:
            raise Refusal(f"audio window [{w['a0']}, {w['a1']}) outside the source's {info.frames} samples (no padding)")
        if w["v0"] < 0:
            raise Refusal(f"video window starts at {w['v0']} s < 0 (no padding)")
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = command(mat, str(video), str(audio), str(tmp), w)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        rec["encode_s"] = round(time.time() - t0, 3)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exit {proc.returncode}: {proc.stderr.strip()[-400:]}")
        if task.get("check_x264"):
            head = tmp.read_bytes()[: 1 << 20]
            pinned = x264_build(task["mat"])
            if pinned.encode() not in head:
                m = re.search(rb"x264 - core \d+ r\d+ [0-9a-f]+", head)
                raise SystemExit(f"libx264 build {m.group(0).decode() if m else 'unknown'} != pinned {pinned!r}")

        # ---- verify by decoding
        t1 = time.time()
        src_t, _, src_facts = _decode_video(str(video))
        sel = np.where((src_t >= w["v0"] - 1e-9) & (src_t < w["v1"] - 1e-9))[0]
        want = set(range(0, len(sel), PSNR_EVERY))
        out_t, out_imgs, out_facts = _decode_video(str(tmp), want)
        problems = []  # unexpected: the recipe or the encoder did not do what it should (exit 1)
        refusals = []  # by design: the source cannot fill the window (the row stays missing, exit 0)
        # the source video must cover the whole window (no padding): a source that ends before v1 would hand over
        # a SHORT video track whose length correlates with the label δ (only δ >= 1.5 reaches v1 > 9.75 s)
        fps = src_facts.get("fps") or (1.0 / float(np.median(np.diff(src_t))) if len(src_t) > 1 else None)
        src_end = float(src_t[-1] + 1.0 / fps) if len(src_t) and fps else None
        if src_end is None or src_end < w["v1"] - task["pts_tol"]:
            refusals.append(f"video window [{w['v0']}, {w['v1']}) extends past the source video end "
                            f"{'unknown' if src_end is None else f'{src_end:.3f}'} s (no padding)")
        length = float(mat["length"])
        if fps and out_facts.get("stream_duration_s") is not None and abs(float(out_facts["stream_duration_s"]) - length) > 1.0 / fps + task["pts_tol"]:
            refusals.append(f"output video track lasts {out_facts['stream_duration_s']:.3f} s, not the {length} s window (± one frame)")
        if len(out_t) != len(sel):
            problems.append(f"{len(out_t)} output frames != {len(sel)} source frames in [{w['v0']}, {w['v1']})")
        else:
            err = np.abs(out_t - (src_t[sel] - w["v0"]))
            rec["pts_max_err_s"] = float(err.max()) if len(err) else 0.0
            if len(err) and err.max() > task["pts_tol"]:
                problems.append(f"frame times differ from source pts - v0 by {err.max():.6f} s")
            if len(sel) and abs(out_t[0] - (src_t[sel[0]] - w["v0"])) > task["pts_tol"]:
                problems.append("first output frame is not the source frame at or after v0")
        a, rate, ch = _decode_audio_int16(str(tmp))
        ref, _ = sf.read(str(audio), start=w["a0"], stop=w["a1"], dtype="int16", always_2d=True)
        if rate != info.samplerate or ch != 1 or a.shape[1] != ref.shape[0] or not np.array_equal(a[0], ref[:, 0]):
            problems.append(f"audio not bit-identical to the source window (rate {rate}, ch {ch}, {a.shape[1]} vs {ref.shape[0]} samples)")
        # sparse PSNR against the source frames (informational; needs the source frames at the same indices)
        if out_imgs and len(out_t) == len(sel):
            _, src_imgs, _ = _decode_video(str(video), {int(sel[k]) for k in out_imgs})
            ps = []
            for k, img in out_imgs.items():
                ref_img = src_imgs.get(int(sel[k]))
                if ref_img is not None and ref_img.shape == img.shape:
                    mse = float(np.mean((img.astype(np.float64) - ref_img.astype(np.float64)) ** 2))
                    ps.append(99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse))
            if ps:
                rec["psnr_min"], rec["psnr_mean"] = round(min(ps), 2), round(float(np.mean(ps)), 2)
        rec["verify_s"] = round(time.time() - t1, 3)
        rec.update(
            {
                "delta": delta,
                "n_frames": int(len(out_t)),
                "first_frame_s": float(out_t[0]) if len(out_t) else None,
                "video": out_facts,
                "source_video": src_facts,
                "audio_samples": int(a.shape[1]),
            }
        )
        if problems or refusals:
            rec["problems"] = refusals + problems
            rec["refused"] = not problems  # only by-design refusals: not a failure of the run
            tmp.unlink(missing_ok=True)
            rec["t_done"] = time.time()
            return rec
        rec["sha256"] = _sha256_file(tmp)
        rec["bytes"] = tmp.stat().st_size
        os.replace(tmp, out)
        rec["ok"] = True
        rec["t_done"] = time.time()  # orders records of one clip across the per-shard logs (read_log)
        return rec
    except SystemExit as exc:  # pinned-build refusal: surfaces to the driver
        tmp.unlink(missing_ok=True)
        rec["fatal"] = str(exc)
        return rec
    except Refusal as exc:  # by design (no padding / source format): the row stays missing
        tmp.unlink(missing_ok=True)
        rec["problems"] = [f"refused: {exc}"]
        rec["refused"] = True
        rec["t_done"] = time.time()
        return rec
    except Exception as exc:  # noqa: BLE001 — every failing row is reported, none aborts the run
        tmp.unlink(missing_ok=True)
        rec["problems"] = [f"{type(exc).__name__}: {exc}"]
        rec["t_done"] = time.time()
        return rec


# ----------------------------------------------------------------------------- driver


def log_path(out_root: Path, shard: str | None) -> Path:
    """This writer's log: ``_materialize/log.<i>-of-<n>.jsonl`` for ``--shard i/n`` (one writer per file, so no
    two processes ever append to one file), else ``_materialize/log.jsonl``."""
    if shard:
        i, n = parse_shard(shard)
        return out_root / "_materialize" / f"log.{i}-of-{n}.jsonl"
    return out_root / "_materialize" / "log.jsonl"


def parse_shard(shard: str) -> tuple[int, int]:
    try:
        i, n = (int(x) for x in shard.split("/"))
    except ValueError:
        raise SystemExit(f"--shard {shard!r}: expected i/n") from None
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"--shard {shard!r}: expected 0 <= i < n")
    return i, n


def log_files(log: Path) -> list[Path]:
    """The logs to merge: every ``log*.jsonl`` in ``log``'s directory (``log`` is the directory itself or any
    file in it), oldest first (then by name)."""
    d = log if log.is_dir() else log.parent
    if not d.is_dir():
        return []

    def key(p: Path) -> tuple[float, str]:
        try:
            return (p.stat().st_mtime, p.name)
        except OSError:
            return (0.0, p.name)

    return sorted((p for p in d.glob(LOG_GLOB) if p.is_file()), key=key)


def read_log(log: Path, stats: dict | None = None) -> dict[str, dict]:
    """Last verified record per clip_id, merged over every writer's log in ``log``'s directory (``log`` may be the
    ``_materialize`` directory or any log file in it). A line that does not parse (a torn append, NUL runs on a
    network FS) is SKIPPED and counted (``stats["unparseable"]``), never fatal: the renamed-into-place file and its
    sha256 are the source of truth, and a row whose record was lost is materialized again on the next run. Across
    files the record with the larger ``t_done`` wins (legacy records without one: the later file / line wins)."""
    done: dict[str, dict] = {}
    st = {"files": 0, "lines": 0, "unparseable": 0}
    for f in log_files(log):
        st["files"] += 1
        with open(f, "rb") as fh:
            raw = fh.read()
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            st["lines"] += 1
            try:
                rec = json.loads(line.decode("utf-8"))
                if not isinstance(rec, dict) or not isinstance(rec.get("clip_id"), str):
                    raise ValueError("not a row record")
            except (UnicodeDecodeError, ValueError):
                st["unparseable"] += 1
                continue
            if rec.get("ok"):
                prev = done.get(rec["clip_id"])
                if prev is None or float(rec.get("t_done") or 0.0) >= float(prev.get("t_done") or 0.0):
                    done[rec["clip_id"]] = rec
    if stats is not None:
        stats.update(st)
    return done


def open_log_for_append(log: Path):
    """Append handle on this writer's log. A file left without a final newline (the writer was killed mid-line)
    gets one first, so the torn tail stays one unparseable line instead of swallowing the next record."""
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "ab")
    if fh.tell() > 0:
        with open(log, "rb") as r:
            r.seek(-1, os.SEEK_END)
            if r.read(1) != b"\n":
                fh.write(b"\n")
    return fh


def gzip_bytes(data: bytes) -> bytes:
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0, compresslevel=9) as gz:
        gz.write(data)
    return out.getvalue()


def record_bytes(rows: list[dict[str, str]], recorded: dict[str, tuple[str, int]]) -> bytes:
    """``materialized.sha256.csv`` (decompressed): ``clip_id,rel_media,sha256,bytes`` of the recorded rows, manifest order."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["clip_id", "rel_media", "sha256", "bytes"])
    for r in rows:
        if r["clip_id"] in recorded:
            sha, size = recorded[r["clip_id"]]
            w.writerow([r["clip_id"], r["rel_media"], sha, size])
    return buf.getvalue().encode("utf-8")


def _stream_durations(path: Path) -> tuple[float | None, float | None]:
    """``(video stream duration s, fps)`` from a file's header (PyAV), ``(None, None)`` when unreadable."""
    try:
        import av

        with av.open(str(path)) as f:
            st = f.streams.video[0]
            dur = float(st.duration * st.time_base) if st.duration is not None else None
            return dur, (float(st.average_rate) if st.average_rate else None)
    except Exception:  # noqa: BLE001
        return None, None


def check_record(task: dict) -> dict:
    """One logged row before it enters the record: the output track lasts ``length`` ± one frame, the source video
    reaches the window end (both from the log's facts, re-read from the file headers when the log line lacks them),
    and the file on disk hashes to the logged sha256 (the file, not the log line, is recorded)."""
    d, mat, out_root, src_root, tol = task["rec"], task["mat"], Path(task["out_root"]), task["src_root"], task["pts_tol"]
    out = out_root / d["rel_media"]
    if not out.is_file():
        return {"clip_id": d["clip_id"], "problem": "file missing"}
    v = d.get("video") if isinstance(d.get("video"), dict) else {}
    dur, fps = v.get("stream_duration_s"), v.get("fps")
    if dur is None or fps is None:
        dur, fps = _stream_durations(out)
    length = float(mat["length"])
    if dur is None or not fps or abs(float(dur) - length) > 1.0 / float(fps) + tol:
        return {"clip_id": d["clip_id"], "problem": f"output video track {dur} s is not {length} s ± one frame"}
    v1 = float(mat["pivot"]) + float(mat["video"]) * float(task["delta"]) + length
    sv = d.get("source_video") if isinstance(d.get("source_video"), dict) else {}
    src_dur = sv.get("stream_duration_s")
    if src_dur is None and src_root:
        src_dur, _ = _stream_durations(Path(src_root) / task["rel_video"])
    if src_dur is None or float(src_dur) < v1 - tol:
        return {"clip_id": d["clip_id"], "problem": f"source video ({src_dur} s) ends before the window end {v1} s"}
    sha = _sha256_file(out)
    if sha != d.get("sha256"):
        return {"clip_id": d["clip_id"], "problem": "file sha256 differs from the log"}
    return {"clip_id": d["clip_id"], "sha256": sha, "bytes": out.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--rows", type=int, default=None, help="validation run: N rows round-robin over split x offset bin (seed 0)")
    sel.add_argument("--all", action="store_true", help="every manifest row (combine with --shard i/n)")
    sel.add_argument("--write-record", action="store_true", help=f"write <out-root>/_materialize/{RECORD} (per-site, never shipped) once every row is verified")
    p.add_argument("--shard", default=None, help="i/n: this process handles manifest rows i::n")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 4), help="parallel rows (each ffmpeg runs 4 encoder threads); 1 = in this process")
    p.add_argument("--source-root", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--pts-tol", type=float, default=1e-3)
    p.add_argument("--force", action="store_true", help="re-materialize rows the log already holds")
    p.add_argument("--allow-missing", action="store_true", help="--write-record: record the verified rows even when some rows are missing")
    ns = p.parse_args(argv)

    spec = load_spec()
    mat = spec["input_contract"]["materialize"]
    rsha = recipe_sha256(mat)
    rows = manifest_rows()
    out_root = ns.out_root or resolve_root("derived_root")
    log = log_path(out_root, ns.shard)  # this writer's own file; reads merge every log in its directory
    log_stats: dict = {}

    if ns.write_record:
        done = {c: d for c, d in read_log(log, log_stats).items() if d.get("recipe_sha256") == rsha}
        missing = [r["clip_id"] for r in rows if r["clip_id"] not in done]
        if missing and not ns.allow_missing:
            print(json.dumps({"write_record": False, "missing": len(missing), "first": missing[:10], "log_files": log_stats["files"], "log_unparseable_lines": log_stats["unparseable"]}, indent=1))
            return 1
        src_root = ns.source_root
        if src_root is None:
            try:
                src_root = resolve_root("media_root")
            except SystemExit:
                src_root = None
        tasks = [{"rec": done[r["clip_id"]], "mat": mat, "out_root": str(out_root), "src_root": str(src_root) if src_root else None,
                  "pts_tol": ns.pts_tol, "delta": float(r["offset_seconds"]), "rel_video": r[mat["source_columns"]["video"]]}
                 for r in rows if r["clip_id"] in done and done[r["clip_id"]].get("rel_media") == r["rel_media"]]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max(1, ns.workers)) as ex:
            checked = list(ex.map(check_record, tasks))
        recorded = {c["clip_id"]: (c["sha256"], c["bytes"]) for c in checked if "sha256" in c}
        excluded = [c for c in checked if "problem" in c]
        if excluded and not ns.allow_missing:
            print(json.dumps({"write_record": False, "excluded": excluded[:20], "n_excluded": len(excluded)}, indent=1))
            return 1
        raw = record_bytes(rows, recorded)
        target = out_root / "_materialize" / RECORD
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip_bytes(raw))
        print(json.dumps({"write_record": str(target), "rows": len(rows), "recorded": len(recorded), "missing": len(missing),
                          "excluded": excluded, "sha256": hashlib.sha256(raw).hexdigest(),
                          "bytes_total": sum(b for _, b in recorded.values()), "log_unparseable_lines": log_stats["unparseable"]}, indent=1))
        return 0

    src_root = ns.source_root or resolve_root("media_root")
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH (setup.md, Dependencies)", file=sys.stderr)
        return 2
    have = ffmpeg_version()
    if have != mat["tool"]["version_pin"]:
        print(f"ffmpeg {have} != pinned {mat['tool']['version_pin']} (input_contract.materialize.tool.version_pin). The pin is the "
              "point (another build encodes other bytes): compute nodes often run another distro build, so run on a host "
              "with the pinned build, ship that binary plus its ldd closure, or use a container (setup.md, Dependencies).",
              file=sys.stderr)
        return 2

    if ns.all:
        idx = list(range(len(rows)))
    else:
        idx = stratified(rows, ns.rows if ns.rows is not None else 216)
    if ns.shard:
        i, n = parse_shard(ns.shard)
        idx = idx[i::n]
    done = read_log(log, log_stats)
    todo = [k for k in idx if ns.force or done.get(rows[k]["clip_id"], {}).get("recipe_sha256") != rsha or not (out_root / rows[k]["rel_media"]).is_file()]

    t0 = time.time()
    results: list[dict] = []
    tasks = [
        {"row": rows[k], "mat": mat, "src_root": str(src_root), "out_root": str(out_root), "recipe_sha256": rsha, "pts_tol": ns.pts_tol, "check_x264": j == 0}
        for j, k in enumerate(todo)
    ]
    if tasks:
        first = materialize_row(tasks[0])  # alone: the x264 build check runs before any parallel work
        results.append(first)
        if first.get("fatal"):
            print(first["fatal"], file=sys.stderr)
            return 2
        with open_log_for_append(log) as fh:

            def put(rec: dict) -> None:
                fh.write((json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))  # one write per line
                fh.flush()

            put(first)
            if ns.workers <= 1:
                for t in tasks[1:]:
                    rec = materialize_row(t)
                    results.append(rec)
                    put(rec)
            else:
                with ProcessPoolExecutor(ns.workers) as ex:
                    futs = [ex.submit(materialize_row, t) for t in tasks[1:]]
                    for fut in as_completed(futs):
                        rec = fut.result()
                        results.append(rec)
                        put(rec)
            os.fsync(fh.fileno())
    wall = time.time() - t0

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    refused = [r for r in bad if r.get("refused")]
    failed = [r for r in bad if not r.get("refused")]
    n_all = len(rows)
    mean_bytes = sum(r["bytes"] for r in ok) / len(ok) if ok else None
    by_split = defaultdict(list)
    for r in ok:
        by_split[r["split"]].append(r["bytes"])
    summary = {
        "recipe_sha256": rsha,
        "ffmpeg": have,
        "x264": x264_build(mat),
        "source_root": str(src_root),
        "out_root": str(out_root),
        "rows_selected": len(idx),
        "rows_skipped_already_done": len(idx) - len(todo),
        "rows_run": len(results),
        "ok": len(ok),
        "refused": len(refused),  # by design (no padding / source format): missing rows, not a failure (exit 0)
        "failed": len(failed),  # unexpected (exit 1)
        "failures": [{k: r.get(k) for k in ("clip_id", "problems", "fatal")} for r in failed[:20]],
        "refusals": [{k: r.get(k) for k in ("clip_id", "problems")} for r in refused[:20]],
        "wall_s": round(wall, 1),
        "workers": ns.workers,
        "rows_per_s": round(len(results) / wall, 3) if wall > 0 and results else None,
        "encode_s_mean": round(sum(r.get("encode_s", 0) for r in ok) / len(ok), 3) if ok else None,
        "verify_s_mean": round(sum(r.get("verify_s", 0) for r in ok) / len(ok), 3) if ok else None,
        "bytes_mean": round(mean_bytes) if mean_bytes else None,
        "bytes_mean_per_split": {s: round(sum(v) / len(v)) for s, v in sorted(by_split.items())},
        "est_total_gb_all_rows": round(sum(sum(v) / len(v) * sum(1 for r in rows if r["split"] == s) for s, v in by_split.items()) / 1e9, 1) if by_split else None,
        "pts_max_err_s": max((r.get("pts_max_err_s", 0.0) for r in ok), default=None),
        "psnr_min": min((r["psnr_min"] for r in ok if "psnr_min" in r), default=None),
        "psnr_mean": round(sum(r["psnr_mean"] for r in ok if "psnr_mean" in r) / max(1, sum(1 for r in ok if "psnr_mean" in r)), 2) if ok else None,
        "first_frame_nonzero": sum(1 for r in ok if (r.get("first_frame_s") or 0) > 0),
        "video_stream_duration_s": sorted({round(r["video"]["stream_duration_s"], 4) for r in ok if r.get("video", {}).get("stream_duration_s") is not None})[:12],
        "n_frames_range": [min((r["n_frames"] for r in ok if "n_frames" in r), default=None), max((r["n_frames"] for r in ok if "n_frames" in r), default=None)],
        "fps_seen": sorted({round(r["video"]["fps"], 3) for r in ok if r.get("video", {}).get("fps")}),
        "sizes_seen": sorted({f"{r['video']['width']}x{r['video']['height']}" for r in ok if "video" in r}),
        "est_all_rows_h_at_these_workers": round(n_all / (len(results) / wall) / 3600, 2) if wall > 0 and results else None,
        "log": str(log),
        "log_files_read": log_stats.get("files", 0),
        "log_unparseable_lines": log_stats.get("unparseable", 0),  # skipped at resume (torn appends); never fatal
    }
    print(json.dumps(summary, indent=1))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
