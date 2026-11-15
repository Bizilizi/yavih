# av_syncbench — setup

Read this first. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

The release is one public tar.gz in seven parts on ModelScope (`coming245/AVSyncBench`, revision
`86c06579529a6e7b2cafb0dc386a50152a37fb98`, the revision the authors' toolkit pins), downloadable anonymously over
HTTPS. The manifest is frozen from the two `metadata.jsonl` files inside it; nothing else is needed.

```sh
yavih bench prepare av_syncbench      # lists the steps
yavih bench prepare av_syncbench 01   # download the seven parts into <media_root>/raw/ (68.19 GB, resumable, sha256-checked)
yavih bench prepare av_syncbench 02   # extract them into <media_root>/AV_SyncBench/ (~70 GB); DELETE_PARTS=1 drops raw/ after
yavih bench prepare av_syncbench 03   # make_manifest.py --check: the manifest re-derived byte for byte, presence of every file
yavih bench prepare av_syncbench 04   # yavih bench verify av_syncbench --census
```

1. `01_download.sh`: `curl` from `https://modelscope.cn/api/v1/datasets/coming245/AVSyncBench/repo?Revision=<rev>&FilePath=AV_SyncBench.tar.gz.part_a<x>`
   for `x` in `a`..`g`: six parts of 10 737 418 240 bytes and one of 3 767 330 543 (68 191 839 983 in all), each
   checked against its size and sha256 (`source.sha256` in `benchmark.yaml`; the same values as the authors'
   `scripts/download_dataset.sh`). A verified part is marked `<part>.ok` and skipped on a rerun. No account, no
   licence click-through. The Hugging Face mirror holds only a README.
2. `02_extract.sh`: `cat raw/AV_SyncBench.tar.gz.part_a? | tar -xz` into the media root, without a concatenated
   copy. Result: `AV_SyncBench/hf_temporal_challenge/` (2 676 `original_video/*.mp4`, 36 925 `edited_audio/*.wav`,
   `metadata.jsonl`) and `AV_SyncBench/hf_timbre_edit/` (818 mp4, 818 wav, `metadata.jsonl`). It refuses a non-empty
   `AV_SyncBench/`.
3. `03_check_manifest.py`: `make_manifest.py --check` against the extracted metadata (sha256
   `f96dcf8ab8e445e1519bcfdf10a8a413d0df9aec5da76da08483bf3d80aa3212` temporal,
   `93dcba15699ce6653ee7540f813b463d963c4c90d4948c9bdcfeadfe36fe6c61` timbre).
4. `04_verify.sh`: `yavih bench verify av_syncbench --census`.

Disk budget: 68.2 GB of parts plus 69.94 GB extracted (the rows read 3 494 mp4 and 37 743 wav, 41 237 files), so
about 140 GB at the peak and 70 GB once `raw/` is deleted. Nothing is materialized (`data.derived: {}`). The media
root is `${data.root}/av_syncbench` unless configured (`yavih config get av_syncbench.media_root`); a copy
extracted elsewhere is read in place when the root points at the directory that holds `AV_SyncBench/`.

## Dependencies

None: `external: []`. The scorer is the authors' own evaluation code, which is not on PyPI (the toolkit at
https://github.com/fgt7t6g/AV-SyncBench installs only from its repository), so it is copied into the folder
(grounding rung 2), at commit `82e13536074377b71a51d81b2f5559a43cf4a122`, licence: none stated:

- scorer: `avsyncbench/metrics.py` (`PairwiseResult`, `_group_summary`, `summarize_results`: the strict decision,
  ties wrong, valid-only accuracy) copied whole into `pairwise.py`;
- procedure: the scoring part of `avsyncbench/evaluators/imagebind.py` `ImageBindScorer.score_record` (common
  duration, at most 18 segments chosen by `linspace(...).round()`, the mean of per-segment cosines) transcribed from
  torch to numpy as `pairwise.pair_scores`; `avsyncbench/media.py` `non_overlapping_starts` copied whole. Its
  model-side transforms (two frames per segment warped to 224 x 224, a 128 x 204 log mel) are only the defaults of
  `bridge_template.py`.

`pairwise.py` names every change in its comments; `tests/test_folder.py` pins the segment selection against the
authors' torch arithmetic and `acc_all` against the copied `summarize_results`. Prepare-time tools: `curl`,
`sha256sum`, GNU `tar` with gzip. The bridge decodes with its own stack or with `yavih.media.decode` and PyAV
(`pip install av soundfile`).

## Input contract

What the reader hands each bridge per row (the generic reader; the folder ships no `reader.py`). A row is one
pair: an original video and one perturbed copy of its audio.

- `video`: a `RawStream` over the **stored original mp4**, byte for byte; nothing decoded, cut or resized. `info`
  guarantees `duration_s, fps, width, height, container, bytes`. H.264, native sizes mixed (landscape and portrait,
  360p to 1080p), 25 to 30 fps (some 30000/1001), 2.5 to 13 s. The file also holds the original AAC track:
  `video_seg` must not read it.
- `audio`: the **same mp4**, handed over for its own AAC track (44.1 or 48 kHz, mostly stereo): the original sound,
  which `audio_seg` embeds. `info` guarantees `duration_s, sample_rate, channels, container, bytes` of the track.
- `edited_audio`: a `RawStream` over the **stored perturbed wav** (16-bit PCM mono: 16 kHz for every temporal pair,
  16 or 22.05 kHz for the timbre pairs); nothing resampled, trimmed or padded. `info` guarantees `duration_s,
  sample_rate, n_samples, channels, container, bytes`. It starts at the video's 0 s; its length differs from the
  video's (an advanced or sped-up copy is shorter, a slowed-down one longer, a jittered 5 s segment is a cut of
  the start).
- Facts: `meta["facts"] = {"segment_seconds": 0.64}`, the protocol's segment grid. No task, level or row identity.

What the bridge decides and declares in `describe()["input"]`: the frames of each segment, the frame size and
normalization, the audio front end and its rate, all from the model's own evaluation code. Every rule is `custom`
(the segment grid is not in the rule vocabulary); validate still checks that the reported frame times and sample
ranges lie inside the media.

```yaml
handed_over:
  video: the original mp4 exactly as stored (H.264, native and mixed frame size, 25 to 30 fps, 2.5 to 13 s, with
    its own AAC track, which video_seg must not read); nothing decoded, cut, resized or padded
  audio: the same mp4, read for its own AAC track (the original sound, 44.1 or 48 kHz stereo as stored), which audio_seg
    reads; the frames must not be read
  edited_audio: the perturbed wav exactly as stored (16-bit PCM mono at 16 kHz, or 22.05 kHz for most timbre pairs;
    starts with the video, often shorter or longer than it); nothing resampled, trimmed or padded
info:
  video:
  - duration_s
  - fps
  - width
  - height
  - container
  - bytes
  audio:
  - duration_s
  - sample_rate
  - channels
  - container
  - bytes
  edited_audio:
  - duration_s
  - sample_rate
  - n_samples
  - channels
  - container
  - bytes
facts:
- segment_seconds
materialize: null
```

## Output contract

Per row three arrays, one row per COMPLETE 0.64 s segment of the stream, in time order: segment `k` covers
`[0.64 k, 0.64 (k + 1))` s of that stream (`floor((duration + 1e-9) / 0.64)` segments, the authors'
`non_overlapping_starts`). `video_seg [Tv, D]` from the frames, `audio_seg [Ta, D]` from the mp4's own track,
`edited_seg [Te, D]` from the wav, all three in the model's one shared space (same `D`; un-normalized is fine). The
scorer keeps the first `min(Tv, Ta, Te)` segments, scores at most `max_segments: 18` of them, and compares the mean
cosine of the video with each audio. No positions, no `pos` requirements, no eligibility rule. `bridge check` runs the
modality-isolation gate of `isolation` (`E_MODALITY_LEAK`); validate checks every key's shape and dtype and runs the
sanity statistics on `edited_seg` (`video_seg` and `audio_seg` repeat across the rows of one original, so they are not
checked for duplicates).

```yaml
shape: fixed
coordinates: media
target_frame: null
requires:
  pos: []
  streams: []
  grid: null
  t_bins: null
schema:
  video_seg:
    type: array
    dtype:
    - float16
    - float32
    shape:
    - '*'
    - D
  audio_seg:
    type: array
    dtype:
    - float16
    - float32
    shape:
    - '*'
    - D
  edited_seg:
    type: array
    dtype:
    - float16
    - float32
    shape:
    - '*'
    - D
    stats: true
isolation:
  video_seg:
  - video
  audio_seg:
  - audio
  edited_seg:
  - edited_audio
```

## Prepare & verify

`yavih bench prepare av_syncbench 03` prints `manifest matches (dc5444fb…)` with `staged: 37743, missing: 0`.
`yavih bench verify av_syncbench --census` then reports status verified, `test 37743/37743` present, a successful read
sample and the media-info table written (44 731 cells: 3 494 mp4 read for two streams, 37 743 wav; the census
probes every cell, about 20 minutes at 35 cells/s on a busy network file system). Eval coverage below
`min_eval_coverage: 0.95` makes the scorer refuse; a partial extraction is the usual cause.

## Gotchas

- `yavih bridge check <bridge> --bench av_syncbench --synthetic` is refused with `E_SYNTHETIC`: the core's synthetic
  check covers the core scorers only, not this folder's own pairwise scorer. Check the bridge on the staged data
  (`yavih bridge check <bridge> --bench av_syncbench --out <profile>`, the line `plan init` prints); the folder's own
  synthetic stand-in runs through `yavih bench trial av_syncbench --synthetic`, with its oracle bridge, not yours.

- Decode the mp4's audio track to **16-bit PCM at its own rate and layout first** (PyAV `AudioResampler(format="s16")`,
  as `bridge_template.py` `_pcm16` does), then downmix and resample. `bridge check` hands the isolation gate a
  lossless copy of the mp4 with the frames blanked whose track is 16-bit FLAC; a float decode of the original AAC
  differs from it in the last bit, so a deterministic bridge fails `E_MODALITY_LEAK` on `audio_seg` although it
  reads no frames.
- Embed every complete segment of each stream on its own clock and never align, cut or pad one stream to another:
  the shorter, shifted or sped-up copy IS the test. An `edited_seg` cut to the video's length, or a `video_seg`
  limited to the edited audio's span, scores something else.
- One forward per modality per segment: a fused audio-visual forward, or a normalization over a pair's streams,
  makes the original audio leak into `video_seg` (`E_MODALITY_LEAK`).
- Take the vectors of the model's shared space (after its projection heads), the ones its contrastive objective
  compares. Offset classifiers (Synchformer, SparseSync) are scored in the paper by `p(offset = 0)`, which this
  scorer does not implement: they have no bridge here.
- The authors decode at 25 fps in the paper; their ImageBind adapter decodes at the native rate. Either is the
  bridge's declared choice.
- The rows of one original video are consecutive (sorted by `rel_video`): a bridge may cache the video and original
  audio embeddings of the last mp4 it saw, keyed on the file's bytes, but the cached value must be bit-identical to a
  fresh forward (`bridge check` re-encodes in other batch compositions).
