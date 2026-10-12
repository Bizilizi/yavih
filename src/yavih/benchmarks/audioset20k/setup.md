# audioset20k — setup

`benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

Public, ungated, no account: the Hugging Face dataset
[`11hu83/audioset20k`](https://huggingface.co/datasets/11hu83/audioset20k) at the pinned revision
`4a0055b21c183257fd9b65ee8d1cd11f13362f0f` (`source.revision`).

```sh
yavih bench prepare audioset20k      # lists the prepare steps
yavih bench prepare audioset20k 01   # hf download at the pinned revision into media_root (DEST=… to override)
yavih bench prepare audioset20k 02   # sha256sum -c SHA256SUMS, the three pinned shas, then bench verify --census
```

The same by hand (`hf` is the Hugging Face CLI; a re-run resumes):

```sh
hf download 11hu83/audioset20k --repo-type dataset --revision 4a0055b21c183257fd9b65ee8d1cd11f13362f0f --local-dir <data_root>/audioset20k-wds-v1
cd <data_root>/audioset20k-wds-v1 && sha256sum -c SHA256SUMS
yavih bench verify audioset20k --census
```

What arrives (67 files, 66.1 GB; `source.bytes` 66 081 637 551 B = the 65 files `SHA256SUMS` lists):

- 62 WebDataset tars `audioset20k-train-000000.tar` … `-000031.tar` and `audioset20k-eval-000000.tar` …
  `-000029.tar` (~1.07 GB each), members `<key>.mp4` (the clip as downloaded) + `<key>.json` (its `clips.csv`
  row); `<key>` = `<ytid>_<start in ms, 7 digits>`, e.g. `--PJHxphWEs_0030000`.
- `index.json`: `{"version": 1, "shards": [...], "members": {"<split>/<key>.mp4": [shard, offset, size]}}` with
  relative shard names (resolved against the index's directory); sha256
  `6976d509de988883a7e00783cec5dc438c1d38517d877518212a7ec88317b441`.
- `clips.csv`: one row per published clip (`key, split, ytid, start_s, end_s, labels, bytes, duration_s, width,
  height, vcodec, shard`; 33 949 rows); sha256 `96e308e4e01647dafded79207663837b82e5e124371dbcf3d752bbfeea99fbad`.
- `MISSING.csv`: `split, key` of the 8 582 listed segments the dataset could not provide (train 4 434, eval
  4 148); sha256 `bbc4f0320faa852372a9244d07dcf687bfb6d46bbfa7e154a5b81424d20ce457`.
- `SHA256SUMS` (sha256 `a6f22393b4f813af63e241b4a407771e0d98ade764a8399c4dd70951861b177c`) and `README.md` (the
  dataset card).

Licence: the AudioSet annotations are Google's, CC BY 4.0; the media are YouTube content that stays its
uploaders', redistributed for non-commercial research only. Do not publish features keyed to it beyond research
use.

Disk: 66.1 GB in 67 files. **Never extract the tars**: every read is a `pread` at an index offset. Nothing is
materialized (`data.derived: {}`); the media-info census is one small gzip under `$YAVIH_HOME/media_info/`.

The dataset card says "h264 video + AAC audio" and "original 10-second clips"; the pinned copy also holds av1
and vp9 video, Opus and E-AC-3 audio, 3 clips without an audio track, and clips longer and shorter than 10 s
("## Input contract").

## Dependencies

None: `external: []` (AudioSet ships no scorer; the headline is yavih's registry `map`).

- Download: the Hugging Face CLI (`pip install -U huggingface_hub`, which provides `hf`) and `sha256sum`.
- `yavih bench verify audioset20k --census` uses `yavih[media]` (PyAV, `pip install av`) to probe the mp4
  headers. With the census present the model env needs no yavih decoder.
- Bridge side: decode with the model's own stack (PyAV, torchaudio, decord, …) or `yavih.media.decode`. **The
  decoder must handle av1 and vp9 video and Opus audio** (PyAV wheels do; check `ffmpeg -decoders` of a system
  build). The ffmpeg CLI cannot read a tar member at an offset: use `RawStream.open()` / `bytes()`. Declare the
  decoder in `describe()["input"][stream]["decoder"]`.

## Input contract

Per row (all 33 949 published clips: 17 726 train, 16 223 eval), the generic reader hands over two RawStreams
over the same tar member `<split>/<key>.mp4`, read in place once (`RawStream.member` = `(tar, offset, size)`,
`path=None`). `bytes()` / `open()` are the published mp4 byte for byte; each stream's `info` describes its own
track (`has_track: true`, or `false` when the file has no such track).

- **`video`**: h264 (31 056 rows), av1 (1 662) or vp9 (1 231); 128 × 120 to 3840 × 2160, mostly 640 × 480,
  1280 × 720 and 1920 × 1080, about 60 clips at 4K, 502 portrait, 84 square; 6–60 fps (30 fps on 20 055 rows,
  25 on 7 697); track duration 0.73–63.73 s around the nominal 10 s. Guaranteed `info`: `duration_s`, `fps`,
  `n_frames`, `width`, `height`, `container` (`mp4`), `bytes`.
- **`audio`**: the same mp4's audio track: AAC (21 193 rows), Opus (12 750) or E-AC-3 (3); 48 kHz (24 444),
  44.1 kHz (9 501) or 32 kHz (1); stereo (33 605), mono (336) or 5.1 (5); track duration 0.02–10.02 s.
  Guaranteed `info` (on the 33 946 rows with an audio track): `duration_s` (= `n_samples / sample_rate`),
  `sample_rate`, `n_samples`, `channels`, `container` (`mp4`), `bytes` (the whole mp4).
- **Three clips have no audio track**: `train/NlBQ4kLpdzk_0260000`, `train/d8WgfWSf1VM_0280000` and
  `eval/chc33hy6iIU_0510000`. They are rows like every other: the same mp4 is handed over as both streams,
  `video.info` is complete, `audio.info` says `has_track: false` with `duration_s`, `sample_rate`, `n_samples`
  and `channels` null. **An audio or av bridge must still return one feature row of the fixed `[N, D]` for each
  of them**, feeding its own declared choice (e.g. its declared span of silence through its own front end),
  declared as a top-level `"no_track": {"audio": "<one line>"}` in `describe()` as `bridge_template.py` does. It
  reports `used["audio"] = {"samples": [], "pad": [0, 0]}` and puts the audio tokens at `t = -1`. Do not decode
  the absent track (`yavih.media.decode.audio` raises `DecodeError`) and do not raise `MediaUndecodable` for these
  rows (that turns them into missing rows).
- **facts**: none (`Sample.meta == {"facts": {}}`); rows, clip ids and splits must not influence the bridge.
- **materialization**: none.

The bridge declares in `describe()["input"]`: which frames at which times (`uniform` / `fps` / `index_linspace` /
`explicit`, with `edge` for clips shorter than the span), resize and crop (the frames are the original ones), the
audio extent (`span` / `full` / `segments`) and its padding, resampling, downmix, the front end, normalization
constants with their source, the tap and the pooling. `{"rule": "custom", "description": "<one line>"}` is a
legitimate opt-out when the vocabulary does not fit the model's sampler (validate records `N_RULE_CUSTOM`). The
8 × 1 s geometry of `bridge_template.py`:

```python
"video": {"temporal": {"rule": "uniform", "n": 8, "span": [0.0, 8.0], "edge": "clamp"}, ...}   # 0.5, 1.5, …, 7.5 s
"audio": {"temporal": {"rule": "span", "start_s": 0.0, "length_s": 8.0, "pad": "zeros"}, ...}  # the first 8 s, zero-padded
```

```yaml
handed_over:
  video: "the original clip mp4 of the Hugging Face dataset 11hu83/audioset20k at revision 4a0055b21c183257fd9b65ee8d1cd11f13362f0f (tar member <split>/<key>.mp4, read in place by offset) exactly as published; its video track (h264, av1 or vp9; 128x120 to 3840x2160, mixed aspect ratios; 6-60 fps; nominal 10 s, 0.73-63.73 s) is described by info; nothing decoded, cut, resized or cropped"
  audio: "the same mp4 bytes as video (one tar member, read once); its audio track (AAC, Opus or E-AC-3; 48, 44.1 or 32 kHz; 1, 2 or 6 channels; 0.02-10.02 s, its length may differ from the video track's) is described by info; 3 clips' mp4 holds no audio track and is handed over all the same, with info has_track false and the track facts null; nothing decoded, resampled, cut or padded"
info:
  video: [duration_s, fps, n_frames, width, height, container, bytes]
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

`[N, D]` per row, `N` and `D` fixed for every row of both splits (pooled `N = 1` or tokens). `clf` needs no
positions (`requires.pos: []`), so `pos_mode: none` is legal. Positions a token bridge does write are in media
coordinates: `t` = media seconds / `info["duration_s"]` of the token's own stream (`-1` for a token centred in
padding the bridge added), `(y, x)` = fractions of the original frame (`grid_pos(..., region=crop_region(...))`),
`-1` for audio. Validate checks every member for `E_TIMES_OUT_OF_MEDIA` (a frame time outside `[0, duration_s]`,
an audio range beyond `n_samples`, any audio range on a clip without an audio track), `E_RULE_MISMATCH` (the
reported `used` vs the declared rule), `E_POS_RANGE`, and on the canaries `E_BATCH_INVARIANCE`.

```yaml
shape: fixed
coordinates: media
target_frame: null
requires:
  pos: []
  streams: []
  grid: null
  t_bins: null
```

## Prepare & verify

```sh
# media_root defaults to ${data.root}/audioset20k-wds-v1; point it at the download directory if it lives elsewhere:
export YAVIH_DATA__AUDIOSET20K__MEDIA_ROOT=/path/to/audioset20k-wds-v1   # or [data.audioset20k] media_root = "…" in config.toml

yavih bench verify audioset20k --census     # presence, labels, curation sums, read sample, media-info census
yavih bench verify audioset20k --json
```

Success: `manifest sha ok; labels ok; curation ok; … status verified`, `train present 17726/17726`,
`eval present 16223/16223`, read and decode samples with 0 failures, and `media_info present rows=67898` (33 949
`video` + 33 949 `audio` cells; the three track-less clips' audio cells say `has_track: false`). The census
probes every member's two tracks by offset (about 11 min for 66 GB). Presence is index membership; the
`media_fingerprint` hashes a seeded sample of members and is the same for any byte-identical copy.

## Gotchas

1. **av1, vp9 and Opus.** 1 662 clips are av1 and 1 231 vp9; about a third of the audio is Opus. A decoder
   without dav1d/libaom or libvpx fails on those rows only, and `bridge check` on 32 rows can miss it: decode one
   av1 clip before extracting. Undecodable rows become missing rows → `E_COVERAGE` below 95 %.
2. **Longer and shorter than 10 s.** 3 025 video tracks run past 10.1 s (three past 12 s, the longest 63.7 s,
   whose audio still stops near 10 s) and 2 590 are shorter than 9.9 s (461 shorter than 8 s, four shorter than
   1 s); 2 469 audio tracks are shorter than 9.9 s. "The whole clip" (`span: "full"`) and "the first 10 s" are
   different rules; declare `edge: clamp` (frames) or `pad: zeros` (audio) for short clips, or validate refuses
   them (`E_TIMES_OUT_OF_MEDIA`).
3. **Two durations.** The video and audio tracks of one mp4 differ in length (71 rows by more than 0.5 s, up to
   54 s). Video `t` divides by the video stream's `duration_s`, audio `t` by the audio stream's.
4. **Mixed resolutions and aspect ratios.** A centre crop keeps a different region per clip, so positions are
   `per_sample` (`crop_region`). The frames are not pre-cropped: declare the model's own resize + crop
   (`spatial`). 4K decode is slow: resize in the decoder when the model allows it, and declare it.
5. **Mixed sample rates and channels.** Declare the resampler and the downmix (`front_end`). AAC/Opus decode a
   few hundred samples more or fewer than the header's `n_samples` (priming): fit the decoded audio to the
   declared ranges.
6. **Normalization.** Use the model's fixed normalization constants: per-clip statistics over a mostly-zero clip
   rescale silence into signal, and per-batch statistics break batch invariance. A model pretrained with per-clip
   statistics keeps its recipe, declared in `norm` with its `source`.
7. **Labels never reach the bridge**; a video reading is scored on audio events (the same 527 targets). Eval
   clips that are also in VGGSound-train matter if the model was pretrained on VGGSound: disclose, never exclude.
8. **Start-up.** Each process parses the 2 MB `index.json` once (< 1 s); rows are grouped by split and sorted by
   key, so a shard reads a contiguous run of members of a few tars.
9. **Three clips without an audio track.** `train/NlBQ4kLpdzk_0260000` (train row 7 671),
   `train/d8WgfWSf1VM_0280000` (train row 12 049) and `eval/chc33hy6iIU_0510000` (eval row 11 022). A bridge
   that calls `rules.audio_ranges(...)` or decodes the audio unconditionally crashes on them (`ValueError` /
   `DecodeError`), and one whose error handler raises `MediaUndecodable` turns them into missing rows. Branch on
   `has_track` first. The 32-row `bridge check` smoke set rarely contains them, so encode one yourself
   (`Benchmark.load("audioset20k").sample("eval", 11022)`).
