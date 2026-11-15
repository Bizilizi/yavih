#!/usr/bin/env python
"""avrobustbench — step 4: materialize every manifest row: the clean or corrupted clip the reader hands over.

The corruption IS the label's condition (a row is a VGGSound clip under one of the 15 AVROBUSTBENCH corruptions at
severity 5, or ``clean``), so it is applied here, at prepare time, never by the bridge
(``benchmark.yaml`` ``input_contract.materialize``, kind ``avrobustbench_corrupt``). Per SOURCE clip (the
manifest's ``src_video`` / ``src_audio``, read from the VGGSound tree, ``media_root``):

1. frames — CAV-MAE's ``extract_frame`` (prepare/cavmae_frames.py, the preprocessing the AVROBUSTBENCH README
   names): 10 frames at ``int(i * total / 10)`` (``total = min(frame count, 10 * fps)``), ``Resize(224)`` +
   ``CenterCrop(224)``, each stored and re-read as a JPEG, as the authors' ``dataset.py`` reads them;
2. audio — the stored 16 kHz mono wav, loaded with ``torchaudio.load`` as ``dataset.py``'s ``get_wav`` does.
   An environmental-noise corruption mixes in a noise recording through ``torchaudio.functional.add_noise``,
   which broadcasts: a stereo recording (7 of the 8 ``crowd_speech`` files) makes the output two-channel (channel
   ``c`` = source + noise channel ``c``). That upstream behaviour is kept: those rows' wavs have 2 channels.

Per ROW (one condition of that clip): the visual corruption ``corruption_dict[c][0](frame, severity=5)`` on each of
the 10 frames in order, after seeding numpy / ``random`` / torch with ``seed("video")``; the audio corruption
``corruption_dict[c][1](waveform=wav, intensity=5)`` after seeding with ``seed("audio")``; ``clean`` applies
neither. ``seed(part) = int(sha256("avrobustbench:<recipe seed>:<clip_id>:<part>")[:8], 16)``: every row draws its
own noise file, crop, angle and noise samples, reproducibly. The functions are the authors' own
(prepare/corruptions.py, copied with the changes its header lists). Outputs, under ``derived_root``:

- ``<rel_video>`` = ``media/<split>/<condition>/<clip>.mp4``: the 10 frames as a 1 fps, 10 s H.264 mp4, encoded
  with the pinned ffmpeg arguments (``tool.video_args``: libx264, preset medium, crf 23, yuv420p — the settings
  moviepy's ``write_videofile`` used for the released VGGSound-2C mp4s; no audio track);
- ``<rel_audio>`` = ``media/<split>/<condition>/<clip>.wav``: the (corrupted) waveform written with
  ``soundfile.write(..., sr, format='WAV')`` (PCM-16, 16 kHz, mono or the 2 channels above), as ``dataset.py``'s
  ``get_wav`` returns it.

Every output is VERIFIED before it is renamed into place: the mp4 decodes (PyAV) to exactly 10 frames of
224 x 224, and the wav has the source's sample count, 16 kHz and the corruption's channel count. Its sha256s go to the materialization
log (one JSON line per row, with the recipe sha256 so a changed recipe never reuses an old file), one file per
writer: ``<derived_root>/_materialize/log[.<i>-of-<n>].jsonl``; readers merge every ``log*.jsonl``.

Refusals by design (the row stays missing, ``"refused": true`` in its record, exit 0): a source video in which cv2
cannot read one of the 10 frame indices, a source wav that is not 16 kHz mono, an audio corruption that fails on a
very short wav (``interference`` samples chunk positions in ``range(total - chunk)``). Build refusals (exit 2): an
ffmpeg build other than ``tool.version_pin``, an ImageMagick or libx264 build other than ``tool.pins``, a Python
package whose version differs from ``tool.python`` (``--allow-version-drift`` runs anyway and records the versions
in every log line), a prepare/corruptions.py or prepare/cavmae_frames.py whose sha256 differs from
``code_sha256``, corruption assets that step 3 did not stage.

Usage (``FOLDER=$(yavih bench path avrobustbench)``; ``yavih bench prepare avrobustbench 04 -- <args>``)::

    # a sample: K eval clips under all 16 conditions + T train clips (seeded choice by sha256 of the clip id)
    python "$FOLDER/prepare/04_materialize.py" --sample-clips 64 --sample-train 640 --workers 4
    # the full run (≈ 245 000 rows), e.g. as a SLURM array of 32 shards (each shard writes its own log):
    python "$FOLDER/prepare/04_materialize.py" --all --shard ${SLURM_ARRAY_TASK_ID}/32 --workers 4
    # what is staged and verified (rows with a current log record and both files present)
    python "$FOLDER/prepare/04_materialize.py" --status

Roots: ``--source-root`` (VGGSound tree; default: the folder's ``media_root``) and ``--out-root`` (default: the
folder's ``derived_root``), resolved by ``yavih.config`` only when the flag is absent; ``--assets`` (default:
``<derived_root>/upstream/corruptions``, prepare step 3). Needs ffmpeg on PATH, ImageMagick (libMagickWand) and the
Python packages of ``tool.python`` (setup.md "Dependencies"): opencv-python-headless, scikit-image and Wand on top
of yavih's torch, torchaudio, torchvision, scipy, pillow, soundfile and PyAV.
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
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
NAME = "avrobustbench"
LOG_GLOB = "log*.jsonl"
CODE_FILES = {"corruptions": HERE / "corruptions.py", "cavmae_frames": HERE / "cavmae_frames.py"}
PY_DISTS = {  # tool.python entry -> importlib.metadata distribution name
    "numpy": "numpy", "torch": "torch", "torchaudio": "torchaudio", "torchvision": "torchvision", "scipy": "scipy",
    "pillow": "pillow", "opencv-python-headless": "opencv-python-headless", "scikit-image": "scikit-image",
    "Wand": "Wand", "soundfile": "soundfile",
}


class Refusal(ValueError):
    """A row refused by design: it stays missing, the run still exits 0."""


# ----------------------------------------------------------------------------- the recipe


def load_recipe() -> dict:
    import yaml

    spec = yaml.safe_load((FOLDER / "benchmark.yaml").read_text(encoding="utf-8"))
    mat = spec["input_contract"]["materialize"]
    if mat.get("kind") != "avrobustbench_corrupt":
        raise SystemExit(f"benchmark.yaml input_contract.materialize.kind is {mat.get('kind')!r}, expected avrobustbench_corrupt")
    return mat


def recipe_sha256(mat: dict) -> str:
    return hashlib.sha256(json.dumps(mat, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def row_seed(recipe_seed: int, clip_id: str, part: str) -> int:
    return int(hashlib.sha256(f"{NAME}:{recipe_seed}:{clip_id}:{part}".encode()).hexdigest()[:8], 16)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_build(mat: dict, *, allow_drift: bool) -> dict:
    """Refuse (exit 2) a build the recipe does not pin; returns the versions found (recorded in the log)."""
    import importlib.metadata as md

    problems: list[str] = []
    for key, path in CODE_FILES.items():
        got = _sha256_file(path)
        if got != mat["code_sha256"][key]:
            problems.append(f"{path.name}: sha256 {got} != code_sha256.{key} {mat['code_sha256'][key]}")
    tool = mat["tool"]
    out = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True, check=False).stdout
    m = re.search(r"ffmpeg version (\S+)", out)
    found = {"ffmpeg": m.group(1) if m else None}
    if found["ffmpeg"] != tool["version_pin"]:
        problems.append(f"ffmpeg {found['ffmpeg']} != tool.version_pin {tool['version_pin']}")
    for name, pin in (tool.get("pins") or {}).items():
        if name == "imagemagick":  # the library Wand loads, which need not be the `convert` on PATH that the probe runs
            try:
                from wand.version import MAGICK_VERSION as text
            except ImportError as e:
                text = f"Wand / libMagickWand not importable: {e}"
        else:
            cmd = [a.replace("{tool}", "ffmpeg") for a in pin["probe"]]
            r = subprocess.run(cmd, capture_output=True, check=False)
            text = (r.stdout + r.stderr).decode("utf-8", "replace")
        mm = re.search(pin["match"], text)
        found[name] = mm.group(0) if mm else None
        if found[name] != pin["value"]:
            problems.append(f"{name} {found[name]} != tool.pins.{name} {pin['value']}")
    for entry in tool["python"]:
        dist, want = entry.split("==")
        try:
            got = md.version(PY_DISTS.get(dist, dist))
        except md.PackageNotFoundError:
            got = None
        found[dist] = got
        if got is None:
            problems.append(f"python package {dist} is not installed (setup.md, Dependencies)")
        elif got != want:
            problems.append(f"python package {dist} {got} != tool.python {want}")
    hard = [p for p in problems if "not installed" in p or "code_sha256" in p]
    if hard or (problems and not allow_drift):
        for p in problems:
            print(f"[{NAME}] refused: {p}", file=sys.stderr)
        if not hard:
            print(f"[{NAME}] --allow-version-drift runs anyway and records the versions in every log line", file=sys.stderr)
        raise SystemExit(2)
    for p in problems:
        print(f"[{NAME}] warning (--allow-version-drift): {p}", file=sys.stderr)
    return found


# ----------------------------------------------------------------------------- rows


def read_rows() -> list[dict[str, str]]:
    with gzip.open(FOLDER / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _rank(text: str) -> str:
    return hashlib.sha256(f"{NAME}:sample:{text}".encode()).hexdigest()


def select(rows: list[dict[str, str]], ns: argparse.Namespace) -> list[dict[str, str]]:
    if ns.all:
        chosen = rows
    else:
        by_split: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            src = r["src_video"]
            if src not in by_split[r["split"]]:
                by_split[r["split"]].append(src)
        keep: set[tuple[str, str]] = set()
        for split, k in (("test", ns.sample_clips), ("train", ns.sample_train)):
            srcs = sorted(dict.fromkeys(by_split.get(split, [])), key=_rank)[: max(0, k)]
            keep |= {(split, s) for s in srcs}
        chosen = [r for r in rows if (r["split"], r["src_video"]) in keep]
    if ns.split:
        chosen = [r for r in chosen if r["split"] in ns.split]
    if ns.shard:
        i, n = (int(x) for x in ns.shard.split("/"))
        chosen = [r for r in chosen if int(_rank(r["src_video"])[:8], 16) % n == i]
    return chosen


def read_logs(out_root: Path) -> tuple[dict[str, dict], int]:
    recs: dict[str, dict] = {}
    bad = 0
    d = out_root / "_materialize"
    if d.is_dir():
        for p in sorted(d.glob(LOG_GLOB)):
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    recs[rec["clip_id"]] = rec
                except (json.JSONDecodeError, KeyError, TypeError):
                    bad += 1
    return recs, bad


def is_done(rec: dict | None, row: dict[str, str], out_root: Path, rsha: str) -> bool:
    if not rec or rec.get("recipe_sha256") != rsha or rec.get("refused"):
        return False
    for col in ("rel_video", "rel_audio"):
        p = out_root / row[col]
        if not p.is_file() or p.stat().st_size != rec["outputs"][col]["bytes"]:
            return False
    return True


# ----------------------------------------------------------------------------- one source clip (worker)

_STATE: dict = {}


def _init_worker(assets: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    import cavmae_frames
    import corruptions

    corruptions.ASSETS = assets
    _STATE.update(corruptions=corruptions, cavmae=cavmae_frames)


def _seed_all(seed: int) -> None:
    import random

    import numpy as np
    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _encode_mp4(frames: list, path: Path, video_args: list[str]) -> None:
    import numpy as np

    raw = b"".join(np.asarray(f.convert("RGB"), dtype=np.uint8).tobytes() for f in frames)
    cmd = ["ffmpeg", *video_args, str(path)]
    r = subprocess.run(cmd, input=raw, capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg exit {r.returncode}: {r.stderr.decode('utf-8', 'replace')[-400:]}")


def _verify(mp4: Path, wav: Path, n_frames: int, size: int, n_samples: int, channels: int) -> None:
    import av
    import soundfile as sf

    with av.open(str(mp4)) as c:
        if len(c.streams.audio):
            raise RuntimeError(f"{mp4}: has an audio track")
        shapes = [(f.height, f.width) for f in c.decode(video=0)]
    if len(shapes) != n_frames or any(s != (size, size) for s in shapes):
        raise RuntimeError(f"{mp4}: decoded {len(shapes)} frames {sorted(set(shapes))}, expected {n_frames} x {size}x{size}")
    info = sf.info(str(wav))
    if (info.frames, info.samplerate, info.channels, info.subtype) != (n_samples, 16000, channels, "PCM_16"):
        raise RuntimeError(f"{wav}: {info.frames} samples {info.samplerate} Hz {info.channels} ch {info.subtype}")


def process_clip(job: dict) -> list[dict]:
    """All rows of one source clip: decode once, then one (corrupted) mp4 + wav per row."""
    import soundfile
    import torchaudio

    C, cav = _STATE["corruptions"], _STATE["cavmae"]
    mat, out_root, src_root = job["recipe"], Path(job["out_root"]), Path(job["source_root"])
    sev, nf, size = int(mat["severity"]), int(mat["frames"]), int(mat["size"])
    t0 = time.time()
    recs: list[dict] = []
    base = {"recipe_sha256": job["recipe_sha256"], **({"versions": job["versions"]} if job.get("drift") else {})}
    try:
        try:
            frames = cav.extract_frame(str(src_root / job["src_video"]), extract_frame_num=nf)
        except cav.FrameReadError as e:
            raise Refusal(str(e)) from e
        wav_path = src_root / job["src_audio"]
        info = soundfile.info(str(wav_path))
        if info.samplerate != 16000 or info.channels != 1:
            raise Refusal(f"{wav_path}: {info.samplerate} Hz, {info.channels} channels (expected 16 kHz mono)")
        waveform, sr = torchaudio.load(str(wav_path))  # dataset.py get_wav
    except Refusal as e:
        return [{**base, "clip_id": r["clip_id"], "refused": True, "reason": str(e)} for r in job["rows"]]
    for r in job["rows"]:
        cond = r["condition"]
        mp4, wav = out_root / r["rel_video"], out_root / r["rel_audio"]
        tmp_v, tmp_a = mp4.with_name(mp4.stem + ".tmp.mp4"), wav.with_name(wav.stem + ".tmp.wav")
        try:
            mp4.parent.mkdir(parents=True, exist_ok=True)
            if cond == "clean":
                out_frames, out_wav = [f.copy() for f in frames], waveform.clone()
            else:
                visual, audio = C.corruption_dict[cond]
                _seed_all(row_seed(mat["seed"], r["clip_id"], "video"))
                out_frames = [visual(f, severity=sev) for f in frames]  # dataset.py get_image, frame by frame
                _seed_all(row_seed(mat["seed"], r["clip_id"], "audio"))
                try:
                    out_wav = audio(waveform=waveform.clone(), intensity=sev)  # dataset.py get_wav
                except (ValueError, RuntimeError) as e:
                    raise Refusal(f"{cond} audio on {waveform.shape[-1]} samples: {type(e).__name__}: {e}") from e
            _encode_mp4(out_frames, tmp_v, list(mat["tool"]["video_args"]))
            buf = io.BytesIO()
            soundfile.write(buf, out_wav.numpy().T, sr, format="WAV")  # dataset.py get_wav: PCM-16
            tmp_a.write_bytes(buf.getvalue())
            if out_wav.dim() != 2 or out_wav.shape[0] not in (1, 2):
                raise RuntimeError(f"{cond} audio returned shape {tuple(out_wav.shape)}")
            _verify(tmp_v, tmp_a, nf, size, int(waveform.shape[-1]), int(out_wav.shape[0]))
            os.replace(tmp_v, mp4)
            os.replace(tmp_a, wav)
            recs.append({**base, "clip_id": r["clip_id"], "refused": False, "outputs": {
                "rel_video": {"sha256": _sha256_file(mp4), "bytes": mp4.stat().st_size},
                "rel_audio": {"sha256": _sha256_file(wav), "bytes": wav.stat().st_size,
                              "channels": int(out_wav.shape[0])}}})
        except Refusal as e:
            recs.append({**base, "clip_id": r["clip_id"], "refused": True, "reason": str(e)})
        except Exception as e:  # noqa: BLE001 - recorded as a failure (exit 1), the other rows go on
            recs.append({**base, "clip_id": r["clip_id"], "refused": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            for t in (tmp_v, tmp_a):
                t.unlink(missing_ok=True)
    for rec in recs:
        rec["clip_seconds"] = round((time.time() - t0) / max(1, len(recs)), 3)
    return recs


# ----------------------------------------------------------------------------- main


def _root(flag: Path | None, root: str) -> Path:
    if flag is not None:
        return flag
    from yavih import config

    try:
        return Path(config.folder_root(FOLDER, root))
    except config.ConfigError as e:
        raise SystemExit(f"pass --{'source' if root == 'media_root' else 'out'}-root or configure {NAME}.{root} ({e})") from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="every manifest row")
    g.add_argument("--sample-clips", type=int, help="this many eval source clips (all their conditions), seeded")
    g.add_argument("--status", action="store_true", help="count verified rows per split / condition and exit")
    ap.add_argument("--sample-train", type=int, default=0, help="with --sample-clips: this many train clips")
    ap.add_argument("--split", action="append", help="restrict to this split (repeatable)")
    ap.add_argument("--shard", default=None, help="i/n: this shard of the source clips (its own log file)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--source-root", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--assets", type=Path, default=None)
    ap.add_argument("--allow-version-drift", action="store_true")
    ns = ap.parse_args(argv)

    mat = load_recipe()
    rsha = recipe_sha256(mat)
    out_root = _root(ns.out_root, "derived_root")
    rows = read_rows()
    logs, bad = read_logs(out_root)
    if ns.status:
        count: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
        for r in rows:
            c = count[r["split"]][r["condition"]]
            c[1] += 1
            rec = logs.get(r["clip_id"])
            c[0] += is_done(rec, r, out_root, rsha)
            c[2] += bool(rec and rec.get("refused") and rec.get("recipe_sha256") == rsha)
        print(json.dumps({"recipe_sha256": rsha, "log_unparseable_lines": bad,
                          "verified/listed/refused": {s: {k: v for k, v in d.items()} for s, d in count.items()}}, indent=1))
        return 0

    src_root = _root(ns.source_root, "media_root")
    assets = ns.assets or out_root / "upstream" / "corruptions"
    if not (assets / "audio_corruptions").is_dir():
        print(f"[{NAME}] refused: no corruption assets at {assets} (yavih bench prepare {NAME} 03)", file=sys.stderr)
        return 2
    versions = check_build(mat, allow_drift=ns.allow_version_drift)
    todo = [r for r in select(rows, ns) if not is_done(logs.get(r["clip_id"]), r, out_root, rsha)]
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in todo:
        by_src[r["src_video"]].append(r)
    jobs = [{"src_video": s, "src_audio": rs[0]["src_audio"], "rows": rs, "recipe": mat, "recipe_sha256": rsha,
             "out_root": str(out_root), "source_root": str(src_root), "versions": versions,
             "drift": ns.allow_version_drift} for s, rs in by_src.items()]
    jobs.sort(key=lambda j: j["rows"][0]["split"] != "test")  # eval clips first (stable: manifest order within)
    log_dir = out_root / "_materialize"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (f"log.{ns.shard.replace('/', '-of-')}.jsonl" if ns.shard else "log.jsonl")
    print(f"[{NAME}] {len(todo)} rows of {len(by_src)} source clips to materialize (recipe {rsha[:12]}); log {log_path}",
          file=sys.stderr)
    n_ok = n_ref = n_err = 0
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log, ProcessPoolExecutor(
            max_workers=max(1, ns.workers), initializer=_init_worker, initargs=(str(assets),)) as ex:
        futs = [ex.submit(process_clip, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            for rec in fut.result():
                log.write(json.dumps(rec, sort_keys=True) + "\n")
                n_ok += not rec.get("refused") and "error" not in rec
                n_ref += bool(rec.get("refused"))
                n_err += "error" in rec
                if "error" in rec:
                    print(f"[{NAME}] FAILED {rec['clip_id']}: {rec['error']}", file=sys.stderr)
            log.flush()
            if k % 50 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"[{NAME}] {k}/{len(futs)} clips, {n_ok} rows ok, {n_ref} refused, {n_err} failed, {el:.0f} s", file=sys.stderr)
    print(json.dumps({"rows_ok": n_ok, "rows_refused": n_ref, "rows_failed": n_err, "seconds": round(time.time() - t0, 1),
                      "recipe_sha256": rsha, "log": str(log_path)}))
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
