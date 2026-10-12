# fsd50k — setup

Read this before writing a bridge. `yavih bench info fsd50k --setup` prints it; the
benchmark itself is described in [`README.md`](README.md).

## Obtain the data

Source: Zenodo record 4060432, <https://zenodo.org/records/4060432>, plain HTTP, no account or agreement.
Twelve files, 24 678 734 449 bytes (24.7 GB): `FSD50K.dev_audio.{z01..z05,zip}` (5 × 3 221 225 472 +
2 306 663 327), `FSD50K.eval_audio.{z01,zip}` (3 221 225 472 + 3 037 675 767), `FSD50K.ground_truth.zip`
(334 701), `FSD50K.metadata.zip` (6 700 838), `FSD50K.doc.zip` (6 984). Step 1 verifies Zenodo's md5s
(e.g. `FSD50K.dev_audio.zip` md5 `c480d119b8f7a7e32fdb58f3ea4d6c5a`); the sha256 of every file is in
`benchmark.yaml source.sha256`.

Licence: the dataset as a whole CC BY 4.0 (`FSD50K.doc/LICENSE-DATASET`); each clip carries its
uploader's licence (CC0 / CC BY / CC BY-NC / CC Sampling+) listed in
`metadata/dev_clips_info_FSD50K.json` and `eval_clips_info_FSD50K.json`; attribution of CC BY clips is
required if you redistribute audio.

Disk: 24.7 GB of archives (keep them) + 33 GB / 51 210 files extracted + a temporary 18.4 GB merged dev
zip during step 2 (the six dev parts, 18 412 790 687 bytes; plan 43 GB free in the archive directory).
Multi-part zips (`.z01…` + `.zip`) must be merged with `zip -s 0` before unzipping; step 2 does it.
Never extract into the archive directory: `rel_path` is relative to the media root `OUT`.

```bash
yavih bench prepare fsd50k   # lists the prepare steps
# 1. the 12 Zenodo files with md5 verification (resumable; ~45 min on a fast link)
DL=/path/to/fsd50k_download yavih bench prepare fsd50k 01
# 2. merge the split zips, extract, rename to the short names the manifest uses
DL=/path/to/fsd50k_download OUT=/path/to/datasets/fsd50k yavih bench prepare fsd50k 02
# expect: dev_audio 40 966 wav (24 GB), eval_audio 10 231 wav (8.3 GB); ground_truth/vocabulary.csv sha256 992b6e11…
```

Result layout (= `data.roots.media_root`, default `${data.root}/fsd50k`): `dev_audio/`, `eval_audio/`,
`ground_truth/{dev.csv,eval.csv,vocabulary.csv}`, `metadata/`, `FSD50K.doc/`. The 30 231 frozen wavs
are 22.9 GB of PCM. To point yavih at another location:
`export YAVIH_DATA__FSD50K__MEDIA_ROOT=/path/to/fsd50k` or `[data.fsd50k] media_root = "/path/to/fsd50k"`
in the site config.

## Dependencies

None (`external: []`; `map` is yavih's registry macro AP). To obtain the data: `wget`, `md5sum`/`sha256sum`,
Info-ZIP `zip` + `unzip`. Decoding the wav is the bridge's job, in the model env, with the model's own stack
(the template uses soundfile; `yavih[media]`'s `yavih.media.decode.audio` is an alternative);
`yavih bench verify --census` needs `yavih[media]` for the header probes.

## Input contract

One stream, `audio` (kind `audio`, `layout: dir`, file = `<media_root>/<rel_path>`). Per row the
reader hands over a `RawStream` whose `bytes()` / `open()` are the stored wav byte for byte: RIFF WAV,
PCM16, 44 100 Hz, mono, 0.300–64.133 s (frames 13 230 – 2 828 257). Of the frozen rows 20 315 are
shorter than 10 s (14 058 train / 6 257 eval; 2 515 train clips < 1 s) and 9 829 longer (max 64.133 s
train, 30.0 s eval). Nothing is decoded, resampled, cut, padded or normalized.

Guaranteed `info` keys (from the site census written by `yavih bench verify fsd50k --census`,
else a live header probe): `duration_s` (= `n_samples / sample_rate`), `sample_rate` (44100),
`n_samples`, `channels` (1), `container` (`wav`), `bytes`. `meta == {"facts": {}}`. No materialization.

The bridge decides, and declares in `describe()["input"]["audio"]`: how much of the clip it
reads (`temporal`: `span` with `start_s`/`length_s`/`pad`, `full` with `max_s`/`pad_to_s`, or
`segments`), how short clips are padded, the resampler and target rate, the front end (fbank, mel
count, window/hop, `dither: 0`), the channel handling, the normalization constants with their source,
and the decoder. It reports per clip in `used["audio"]` the native-rate sample ranges it read and the
padding it added (`{"samples": [[lo, hi]], "pad": [left, right]}`), which validate re-evaluates on the
row's `info`. Example: a 10 s head window zero-padded at the end is
`{"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"}`.

```yaml
handed_over:
  audio: "the FSD50K wav exactly as stored (RIFF WAV, PCM16, 44.1 kHz, mono; 0.300–64.133 s over the frozen rows); nothing decoded, resampled, cut or padded"
info:
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

`feats [N, D]` per row with ONE `(N, D)` for every row of both splits (`shape: fixed`): with
durations from 0.3 to 64 s, a bridge whose token count follows the clip length must fix it by its
declared rule (`span` with `pad: zeros`, or `full` with `pad_to_s`) or by pooling, never by padding to
the longest clip of the batch. `clf` ignores positions, so every well-formed feature set is eligible
and `pos_mode: none` is the default. If a bridge writes positions anyway they are media coordinates:
`t` = token centre in seconds / `info.duration_s` (so `per_sample`, durations differ), `-1` for a token
centred in added padding, `y = x = -1`. Validate checks, on every member: `E_TIMES_OUT_OF_MEDIA` (an
audio range outside `[0, n_samples]`), `E_RULE_MISMATCH` (`used` differs from the declared `temporal`
rule evaluated on the row's `info` by more than one sample), `E_BATCH_INVARIANCE`, `E_POS_RANGE`,
`E_DEAD_ROWS`, fixed `N`/`D`; `bridge check` adds rerun bit-identity and the alone / shuffled
invariance passes.

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

No derived files, nothing to materialize. Step 3 re-derives the manifest from the staged ground truth,
checks it against the committed files and writes the media-info census:

```bash
yavih bench prepare fsd50k 03          # make_manifest.py --check + the census
yavih bench verify fsd50k --census     # or just the census and the checks
```

Success: `manifest sha ok; labels ok; curation ok; … status verified`, `train present 20000/20000`,
`eval present 10231/10231`, `read sample` and `decode sample` with `failed=0`, and
`media_info present rows=30231`. `manifest_hash` is
`7ad288bbfb6b957ba83579f33feb44bce2fbad932bdd1fa5dc51c36776369c55`.

## Gotchas

1. **Resampling is yours.** The file is **44 100 Hz**. Resample with a proper filter (torchaudio/soxr),
   never decimate by slicing; declare the target rate and the resampler in `front_end`.
2. **Padding is yours, and it is in the signal.** 67 % of the rows are shorter than 10 s and 2 515
   train clips are under 1 s: with `span 10 s, pad: zeros` they are ≥ 90 % zeros. Per-clip mean/std
   normalisation over such a clip rescales silence into signal: use the model's pretraining constants
   (`norm` with its `source`), unless per-clip standardisation over the padded buffer is the model's own
   pretraining recipe; then keep it and declare it. Batch statistics fail `E_BATCH_INVARIANCE`.
3. **Window choice bounds mAP.** 33 % of the rows are longer than 10 s (up to 64.1 s in train, 30.0 s
   in eval). A head window (the AST convention) makes an event in the second half of a 30 s field
   recording invisible but still labelled. Reading more (`full` with `max_s`, or `segments`) is legal;
   whatever you read is declared, recorded and must be what `used` reports.
4. **`used` is in native samples.** Ranges are half-open `[lo, hi)` at 44 100 Hz of the stored file and
   `pad` counts the native-rate samples the rule adds (`round(length_s · 44100) − (hi − lo)` for `span`),
   even if you pad after resampling. Off-by-one rounding of `round(length_s · sr)` is `E_RULE_MISMATCH`.
5. **Fixed N.** Variable-length front ends (`full` without `pad_to_s`) give a different token count per
   clip → `FeatureShapeError`; padding to the batch maximum makes N depend on the batch and fails
   `bridge check`.
6. **No video.** `streams == ("audio",)`; a video-only model must raise `yavih.NotApplicable` in
   `__init__`.
7. **Labels are a set and smeared** (leaf first, ancestors after); the probe loss is BCE. The bridge
   never sees labels.
8. **`val` is not the eval set.** The official dev `val` sub-split is part of the probe-train
   population (column `dev_split`); the scored split is `eval`.
9. **Dead-pad rows.** A model that masks padded frames to zeros emits identical tokens for the pad
   region of every short clip; with `pooling: none` validate may flag `E_DEAD_ROWS`/`W_CONSTANT_CLIP`.
   Mean-pool or drop pad tokens and declare it.
10. **Read only the declared range** (`soundfile.read(fh, start=lo, stop=hi)`, as the template does):
    reading a whole 30 s clip costs 3× a 10 s one. `bridge check` measures `clips_per_s` at the plan's
    worker count.
