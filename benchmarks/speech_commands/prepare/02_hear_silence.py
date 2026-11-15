"""speech_commands step 2/3: cut HEAR's ``_silence_`` clips from the six background-noise recordings.

HEAR's Speech Commands task has a ``_silence_`` class whose train and valid clips are not files of the release: its
preprocessing cuts them from ``_background_noise_/*.wav``, a 1 s window every 0.5 s (the TensorFlow Datasets recipe).
This step does the same with HEAR's own loop, writes the 789 clips to ``<media_root>/hear_silence/<name>-<start>.wav``
(``<start>`` in samples at 16 kHz; the ``running_tap`` ones are HEAR's valid silence, the rest its train silence) and
a ``SHA256SUMS`` file over them, and checks every digest against ``EXPECTED_SUMS_SHA256`` (``benchmark.yaml``
``data.derived``). The cut is a sample-exact copy of the source PCM: run it with any yavih python that has
``soundfile`` (libsndfile >= 1.1, whose int16 -> float64 -> int16 round trip is lossless; libsndfile 1.0.x scaled the
written samples by 32767/32768 and fails the digest check).

    yavih bench prepare speech_commands 02          # media root = $(yavih config get speech_commands.media_root)
    ROOT=/path/to/datasets/speech_commands yavih bench prepare speech_commands 02

Needs step 01 (``<media_root>/train/_background_noise_/``). Re-running rewrites nothing that already matches.
"""

# The loop in ``cut`` is copied from hear-preprocess (HEAR 2021, https://github.com/hearbenchmark/hear-preprocess
# @ 610d864b2c0f7064eaeced98ee8992d3914d58f1, tag v2021.1.1), file hearpreprocess/speech_commands.py,
# GenerateTrainDataset.run, lines 106-124; licence Apache License 2.0 as found upstream. Changes: the luigi task, the
# tqdm progress bar and the symlinks of the word folders are dropped; the output directory is
# <media_root>/hear_silence instead of the task's workdir/_silence_; each file is written to a temporary name and
# renamed, and an existing file whose digest already matches is kept.

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import soundfile as sf

from yavih import config

#: ``train/_background_noise_/<file>`` -> sha256 of the release file (speech_commands_v0.02.tar.gz).
BACKGROUND_NOISE_SHA256 = {
    "doing_the_dishes.wav": "099eafcd7c4c266612012b5622b97157042e5288acaf4cbf3dc475814bfcdaa8",
    "dude_miaowing.wav": "1acd62f115d4c3f9daca9c5ec0c2e0c3e174a2a08be30200003a51de52b288ff",
    "exercise_bike.wav": "e453813ed45b2f9f81d5600ec3ac2a5e22d3f2c947fe6f0a3ff1f3d6018c014d",
    "pink_noise.wav": "b6e038c83fb342e39267d4fe69663f76ef0c1121ff60232e7b79171c00318cd6",
    "running_tap.wav": "c199c5fd61f5bf9fd57f1c346c8a51c0c035d63d176348dd9f0f7d671d7eb8eb",
    "white_noise.wav": "aa97784fe735002c407aa66efd0e44d87d9d2da97e7a35fc6bac48632be11f2d",
}
N_SLICES = 789
#: sha256 of ``hear_silence/SHA256SUMS`` (``<sha256>  <file>`` lines, sorted by file); == benchmark.yaml data.derived.
EXPECTED_SUMS_SHA256 = "9e3cef9ceb50fcba95cf909e4e4215aa0e22a05f8b991ae61785e5ea212c8533"
OUT_DIR = "hear_silence"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def media_root() -> Path:
    if os.environ.get("ROOT"):
        return Path(os.environ["ROOT"])
    folder = Path(os.environ.get("FOLDER") or Path(__file__).resolve().parent.parent)
    try:
        return config.folder_root(folder, "media_root", None)
    except config.ConfigError as e:
        raise SystemExit(f"set ROOT or configure [data.speech_commands] media_root: {e}") from None


def _write(filename: str, audio_segment, sr: int) -> None:
    tmp = f"{filename}.part.wav"
    sf.write(tmp, audio_segment, sr)
    os.replace(tmp, filename)


def cut(audio_path: Path, silence_dir: str, existing: dict[str, str]) -> list[str]:
    """One background recording -> its slices (HEAR's loop); returns the file names written or kept."""
    names = []
    audio, sr = sf.read(str(audio_path))
    assert audio.ndim == 1

    basename = os.path.basename(audio_path)
    name, ext = os.path.splitext(basename)

    for start in range(0, len(audio) - sr, sr // 2):
        audio_segment = audio[start : start + sr]
        filename = f"{name}-{start}{ext}"
        names.append(filename)
        filename = os.path.join(silence_dir, filename)
        if filename in existing:
            continue
        _write(filename, audio_segment, sr)
    return names


def main() -> int:
    root = media_root()
    bg = root / "train" / "_background_noise_"
    if not bg.is_dir():
        raise SystemExit(f"{bg} not found: run step 01 first")
    for fname, want in sorted(BACKGROUND_NOISE_SHA256.items()):
        got = sha256_file(bg / fname)
        if got != want:
            raise SystemExit(f"{bg / fname}: sha256 {got}, expected {want}")
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    sums_path = out / "SHA256SUMS"
    # keep a slice only when SHA256SUMS lists it with its current digest
    listed: dict[str, str] = {}
    if sums_path.is_file():
        for line in sums_path.read_text().splitlines():
            sha, name = line.split("  ", 1)
            listed[name] = sha
    existing = {os.path.join(str(out), n): s for n, s in listed.items() if (out / n).is_file() and sha256_file(out / n) == s}
    names: list[str] = []
    for fname in sorted(BACKGROUND_NOISE_SHA256):
        names.extend(cut(bg / fname, str(out), existing))
    if len(names) != N_SLICES or len(set(names)) != N_SLICES:
        raise SystemExit(f"{len(names)} slices, expected {N_SLICES}")
    lines = "".join(f"{sha256_file(out / n)}  {n}\n" for n in sorted(names))
    sums_path.write_text(lines)
    digest = hashlib.sha256(lines.encode()).hexdigest()
    if digest != EXPECTED_SUMS_SHA256:
        print(f"FAIL: {sums_path} has sha256 {digest}, expected {EXPECTED_SUMS_SHA256} "
              "(an old libsndfile rescales the samples; see the docstring)", file=sys.stderr)
        return 1
    print(f"=== OK: {len(names)} silence slices under {out}; SHA256SUMS sha256 {digest} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
