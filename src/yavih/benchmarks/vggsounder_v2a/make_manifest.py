#!/usr/bin/env python
"""Freeze the ``vggsounder_v2a`` manifest from the frozen ``vggsounder`` manifest (README.md, "Re-freezing the manifest").

Input (read only, nothing is downloaded): the packaged ``vggsounder`` folder's ``manifest.csv.gz`` (sha256-pinned:
VGGSounder 1.0 eval rows with the package's disjoint ``labels_a`` / ``labels_v`` / ``labels_av`` cells).

Eval rows = the ``split == test`` rows of that manifest with at least one audible label (``labels_a`` ∪
``labels_av``) that av-benchmark can score over ``AUDIO_LENGTH`` (8 s, the scored window; a shorter clip is refused by its
synchrony model or video loader, which then drops or substitutes files silently), in file order; then
``yavih.freeze.class_balanced_subsample`` keyed by the first audible label (cap 3 000, seed 0). Each row keeps its VGGSound clip id, the source mp4 and wav paths of the VGGSound tree, and its
audible labels (``;``-joined, ``labels_a`` tokens then ``labels_av`` tokens, file order). ``rel_media`` names the
muted copy prepare/01 writes under ``derived_root``.

Outputs, byte-identical on re-run: ``manifest.csv.gz`` and a printed JSON block with every value ``benchmark.yaml``
needs. ``--check`` compares against the committed file instead of writing (exit 1 on drift). torch is needed here
(the seeded draw; freeze time only).

The exclusions come from ``short_clips.csv`` (sha256-pinned, committed): every eval row whose audio (wav header) is
shorter than ``AUDIO_LENGTH``, or whose video av-benchmark's own loader refuses for that window (``VideoDataset``: too
few frames at 25 fps or 0.5 fps from time 0; a container can outlast its video stream), in the VGGSound corpus as staged
when the folder was frozen. ``--measure <vggsound root>`` re-measures the corpus with av_bench installed and rewrites
that file (its new sha256 is printed; a changed list is a new protocol).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

from yavih.freeze import class_balanced_subsample, csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "vggsounder" / "manifest.csv.gz"
#: sha256 of the decompressed vggsounder manifest (its benchmark.yaml manifest.sha256).
SOURCE_SHA256 = "998c246d2c37bab4ca7d6285b2039f49e8793deb61b5e21bce7fc4082382438d"
CAP, SEED = 3000, 0
AUDIO_LENGTH = 8.0
SHORT = HERE / "short_clips.csv"
#: sha256 of short_clips.csv (eval rows with audio or video shorter than AUDIO_LENGTH; ``--measure`` rewrites it)
SHORT_SHA256 = "e24a4f58d3e8e3e03685720c6c8f94593d81b6c8ba74b6e84e4ff5f2677762d1"
HEADER = ["clip_id", "split", "rel_media", "rel_video", "rel_audio", "labels_audible"]


def _tokens(cell: str) -> list[str]:
    return [t for t in (cell or "").split(";") if t]


def _source_rows() -> list[dict[str, str]]:
    raw = gzip.decompress(SOURCE.read_bytes())
    got = hashlib.sha256(raw).hexdigest()
    if got != SOURCE_SHA256:
        sys.exit(f"{SOURCE}: decompressed sha256 {got} != pinned {SOURCE_SHA256}")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))


def _check_clip(job: tuple[str, str, str]) -> list[str] | None:
    """One eval row: ``[clip_id, audio_s, reason]`` when av-benchmark cannot score its ``AUDIO_LENGTH`` window, else
    None. Audio: the wav header's duration. Video: av-benchmark's own ``VideoDataset`` sample (frames resampled to
    25 fps for Synchformer and 0.5 fps for ImageBind from time 0, each stream must fill the window)."""
    import torch

    torch.set_num_threads(1)
    import soundfile as sf
    from av_bench.data.video_dataset import VideoDataset

    clip_id, audio, video = job
    a = sf.info(audio).duration
    if a < AUDIO_LENGTH:
        return [clip_id, f"{a:.4f}", "audio shorter than the window"]
    try:
        VideoDataset([Path(video)], duration_sec=AUDIO_LENGTH).sample(0)
    except Exception as e:  # noqa: BLE001 - every refusal is a reason to exclude the clip
        return [clip_id, f"{a:.4f}", " ".join(str(e).replace(video, "<video>").split())[:120]]
    return None


def measure(root: Path, workers: int = 24) -> bytes:
    """``short_clips.csv`` bytes: the eval rows av-benchmark cannot score over ``AUDIO_LENGTH`` seconds (needs av_bench,
    freeze time only)."""
    from concurrent.futures import ProcessPoolExecutor

    test = [r for r in _source_rows() if r["split"] == "test"]
    jobs = [(r["clip_id"], str(root / r["rel_audio"]), str(root / r["rel_video"])) for r in test]
    with ProcessPoolExecutor(workers) as ex:
        short = [x for x in ex.map(_check_clip, jobs, chunksize=16) if x is not None]
    return csv_bytes(["clip_id", "audio_s", "reason"], short)


def _short_ids() -> set[str]:
    data = SHORT.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != SHORT_SHA256:
        sys.exit(f"{SHORT}: sha256 {got} != pinned {SHORT_SHA256} (re-measure with --measure and pin the new value)")
    return {r["clip_id"] for r in csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))}


def freeze() -> tuple[bytes, dict]:
    rows = _source_rows()
    short = _short_ids()
    test = [r for r in rows if r["split"] == "test"]
    audible_all = [r for r in test if _tokens(r["labels_a"]) + _tokens(r["labels_av"])]
    audible = [r for r in audible_all if r["clip_id"] not in short]
    first = [(_tokens(r["labels_a"]) + _tokens(r["labels_av"]))[0] for r in audible]
    keep = class_balanced_subsample(first, CAP, SEED)
    out = []
    for i in keep:
        r = audible[i]
        labs = _tokens(r["labels_a"]) + _tokens(r["labels_av"])
        out.append([r["clip_id"], "test", f"muted/{r['clip_id']}.mp4", r["rel_video"], r["rel_audio"], ";".join(dict.fromkeys(labs))])
    data = csv_bytes(HEADER, out)
    facts = {
        "manifest": {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(out)},
        "upstream": {
            "source": "the packaged vggsounder folder's manifest.csv.gz (VGGSounder 1.0 eval rows)",
            "sha256": SOURCE_SHA256,
            "rows": len(test),
            "draw": f"class_balanced_subsample over the first audible label of the {len(audible)} eval rows with an audible label that av-benchmark can score over {AUDIO_LENGTH:g} s, cap {CAP}, seed {SEED}",
        },
        "curation": {
            "total_upstream": len(test),
            "per_split": {"test": {"listed": len(out), "staged": len(out), "missing": 0}},
            "excluded": {"no_audible_label": len(test) - len(audible_all), "shorter_than_audio_length": len(audible_all) - len(audible),
                         "draw_remainder": len(audible) - len(out)},
        },
        "classes_in_draw": len(set(first[i] for i in keep)),
    }
    return data, facts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed manifest instead of writing")
    ap.add_argument("--measure", type=Path, default=None, help="re-measure the durations in this VGGSound root and rewrite short_clips.csv")
    args = ap.parse_args()
    if args.measure is not None:
        data = measure(args.measure)
        write_if_changed(SHORT, data)
        print(json.dumps({"short_clips": data.decode().count("\n") - 1, "sha256": hashlib.sha256(data).hexdigest()}, indent=1))
        return
    data, facts = freeze()
    path = HERE / "manifest.csv.gz"
    if args.check:
        same = path.is_file() and gzip.decompress(path.read_bytes()) == data
        print(json.dumps({"ok": same, **facts}, indent=1))
        sys.exit(0 if same else 1)
    write_if_changed(path, gzip_bytes(data))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
