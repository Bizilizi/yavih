# avsync — setup

avsync needs the **VGGSound source media** (shared with `vggsounder`, read only at prepare time) and the
**materialized mp4s** that `prepare/03_materialize.py` writes from them, one per manifest row: the files the reader
hands over. People-facing description and metrics: [`README.md`](README.md); machine truth: `benchmark.yaml`.

## Obtain the data

Source = **VGGSound** (annotations CC BY 4.0, https://www.robots.ox.ac.uk/~vgg/data/vggsound/; media = YouTube
clips fetched by the user; listing `metadata/vggsound.csv` sha256
`ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8`, 199 176 rows, no header). The media decay on
YouTube: prefer an existing copy of the corpus; otherwise download it as the vggsounder folder's setup.md describes
(same corpus, same tree), restricted to the clips step 01 lists.

```sh
yavih bench prepare avsync                                        # lists the steps
LIST_ONLY=1 yavih bench prepare avsync 01 > avsync_clips.txt      # the 20 997 source clips the manifest needs
yavih bench prepare avsync 01                                     # verifies metadata/vggsound.csv
yavih bench prepare avsync 03 -- --all --workers 7                # materialize every row (see "Prepare & verify")
yavih bench prepare avsync 04                                     # re-derivation check + bench verify --census
```

- **Source media** (`media_root`, default `${datasets.vggsound}`): `video/<clip>.mp4` (H.264, native resolution,
  mixed ≈ half 720×1280, 6–60 fps), `audio/<clip>.wav` (16 kHz mono PCM_16; every frozen clip measures
  10.000–10.472 s), `metadata/vggsound.csv`. The frozen rows touch **20 997 clips = 41 994 files = 46.3 GB**
  (39.6 GB mp4 + 6.7 GB wav). A re-downloaded clip is new media: a clip that lost audio length makes
  `03_materialize.py` refuse its row (no padding), and re-downloaded bytes change the materialized files.
- **Materialized media** (`derived_root`, default `${data.root}/avsync/materialized`): one mp4 per row in 256
  directories `<h[:2]>/<h>.mp4` plus the logs `_materialize/log*.jsonl`; ≈ 74.4 GB, ≈ 33 000 inodes. The source
  tree is not read after materialization.

## Dependencies

None (`external: []`); `yavih doctor --bench avsync` lists nothing to install.

Prepare-time tools (the materialization only; never in the model env):

- **ffmpeg `4.4.2-0ubuntu0.22.04.1`** exactly (`input_contract.materialize.tool.version_pin`; Ubuntu 22.04's
  `apt install ffmpeg`), built with **libx264 `core 163 r3060 5db6aa6`** (`tool.pins.libx264`, checked in the
  first output's SEI). Another build encodes other bytes, so `03_materialize.py` refuses it (exit 2). This ffmpeg
  cannot mux PCM into mp4 and calls FLAC-in-mp4 experimental, hence ALAC (bit-exact round trip, checked). Cluster
  compute nodes may run another distro build: run on a host with the pinned build, ship the pinned binary plus its
  shared-library closure (`ldd /usr/bin/ffmpeg`, ≈ 200 MB) first on `PATH`, or use an Ubuntu 22.04 container.
  Check on the node that will run step 03 (`yavih doctor` on a login node says nothing about compute nodes).
- PyAV, soundfile, numpy for the decode verification (`pip install av soundfile`; tested with PyAV 14.3.0,
  soundfile 0.13.1), pyyaml (yavih core).

## Input contract

What the generic reader (`yavih.sample.iter_samples`; no `reader.py`) hands the bridge per row:

- **`video`** — a `RawStream` over the row's **materialized mp4** (`path` = `<derived_root>/<rel_media>`,
  `bytes()`, `open()`), describing its video track: the VGGSound source video over `[1 + δ/2, 9 + δ/2]` s, every
  source frame whose pts falls in that window, re-encoded with libx264 (CRF 18, preset medium, yuv420p, native
  resolution, `-vsync passthrough` with the source time base), each frame at **source pts − (1 + δ/2)** (±1 ms; the
  mp4 edit list stores a first-frame delay in 1/1000 s). `info`: `duration_s` (8.0 ± one frame), `fps` (6–60),
  `width`, `height` (128×96 to 1280×720, portrait e.g. 406×720), `container` (`mp4`), `bytes`.
- **`audio`** — a `RawStream` over the **same file**, describing its audio track: the VGGSound 16 kHz mono wav over
  `[1 − δ/2, 9 − δ/2]` s, cut sample-exactly, stored as lossless ALAC (decodes to int16 bit-identical to the
  source). `info`: `duration_s` 8.0, `sample_rate` 16000, `n_samples` 128 000, `channels` 1, `container`, `bytes`.
- **`meta = {"facts": {}}`**. The offset is the label: it is not in `meta`, not in `info`, not in the path (the file
  name is `sha256("avsync/v2:" + clip_id)[:24]`). `clip_id` (`<clip>#<bin>`) does end in the label: `bridge check`
  scrambles it; a bridge must never read it.

**Materialization** (`input_contract.materialize`, kind `av_offset_mux`, hashed): for a row with δ =
`offset_seconds`, `03_materialize.py` runs `ffmpeg *video_args *audio_args <out>.mp4` with `v0 = 1 + δ/2`,
`v1 = v0 + 8`, `a0 = round((1 − δ/2)·16000)`, `a1 = a0 + 128000` (at δ = ±2 the windows are `[0, 8]` / `[2, 10]`,
inside the ≥ 10.0 s clips), verifies every output by decoding it (audio bit-identical to the source window; exactly
the source frames of the window at source pts − v0) and logs its sha256. The output bytes repeat on one host but
can differ between hosts with the same pinned build, so the folder pins the recipe, not the bytes: the recipe
sha256 (`recipe_sha256` in `03_materialize.py`: the materialize block without `tool.pins`, canonical JSON) is
`feada8d8cb91dff65ad14354f7856738afb9f0e43b1018890de585d84f6b2847`, every log line carries it, and a changed
recipe never reuses an old file. No per-row output sha256 ships; the decode verification is the check.
`03_materialize.py --write-record` writes a per-site record `<derived_root>/_materialize/materialized.sha256.csv.gz`
(never shipped) for checking your own copy later. Inside each file audio and video are offset by exactly δ.

The bridge decides, and declares in `describe()["input"]`: which frames at which times (a `uniform` rule with
`n = 16, span: [0.0, 8.0]` gives the 2 fps bucket centres `(k + 0.5)·0.5 s`; `n = 8` gives 1 fps), resize/crop of
mixed and portrait resolutions, the audio front end (already 16 kHz mono; `[0, 8]` needs no padding),
normalization constants (from the model's pretraining recipe, never from the data) and the decoder.

```yaml
input_contract:
  handed_over:
    video: >-
      the video track of the row's materialized mp4 (prepare/03_materialize.py): the VGGSound source video over
      [1 + δ/2, 9 + δ/2] s, re-encoded frame-accurately (libx264, native resolution and pixel format, source
      frame timing kept, times relative to the window start, ≈ 8.0 s); nothing else decoded, cut, resized or
      padded
    audio: >-
      the audio track of the same mp4: the VGGSound 16 kHz mono wav over [1 − δ/2, 9 − δ/2] s, cut
      sample-exactly (128 000 samples, 8.0 s) and stored losslessly as ALAC (decodes bit-identical to the
      source PCM); audio and video inside the file are offset by exactly δ, which is the label and is not
      handed over
  info:
    video: [duration_s, fps, width, height, container, bytes]
    audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
  facts: []
  materialize:
    kind: av_offset_mux
    offset_column: offset_seconds
    source_columns: {video: rel_video, audio: rel_audio}
    pivot: 1.0
    length: 8.0
    video: 0.5
    audio: -0.5
    tool:
      name: ffmpeg
      version_pin: "4.4.2-0ubuntu0.22.04.1"
      # command = ["ffmpeg", *video_args, *audio_args, "<out>.mp4"]; placeholders: {video} {audio} = source paths,
      # {v0} {v1} = video window [pivot + video·δ, v0 + length] s (repr of the float), {a0} {a1} = audio window in
      # source samples [round((pivot + audio·δ)·sr), a0 + round(length·sr)]. video_args holds the input side and
      # the video stream's options, audio_args the audio stream's options and the muxer's. `pins`: further builds the
      # bytes depend on, checked by `yavih doctor --bench avsync` and by prepare/03_materialize.py on its first output
      # (not recipe arguments: the recipe sha256 of the materialize log leaves them out).
      video_args: [-nostdin, -hide_banner, -loglevel, error, -y, -copyts, -noautorotate, -i, "{video}", -i, "{audio}",
                   -map, "0:v:0", -filter:v, "trim=start={v0}:end={v1},setpts=PTS-{v0}/TB", -vsync, passthrough,
                   "-enc_time_base:v", "-1", -c:v, libx264, -preset, medium, -crf, "18", -pix_fmt, yuv420p,
                   -threads, "4", "-flags:v", +bitexact]
      audio_args: [-map, "1:a:0", -filter:a, "atrim=start_sample={a0}:end_sample={a1},asetpts=PTS-STARTPTS",
                   -c:a, alac, "-flags:a", +bitexact, -map_metadata, "-1", -map_chapters, "-1", -fflags, +bitexact,
                   -movflags, +faststart]
      pins:
        libx264:
          value: "x264 - core 163 r3060 5db6aa6"
          probe: ["{tool}", -nostdin, -hide_banner, -loglevel, error, -f, lavfi, -i, "color=c=black:s=64x64:d=0.04",
                  "-frames:v", "1", -c:v, libx264, -f, h264, "-"]
          match: "x264 - core \\d+ r\\d+ [0-9a-f]+"
```

## Output contract

- **`[N, D]` fixed** for every row of every split of a feature set (`shape: fixed`); `N ≥ 1`, float16 or float32.
  A pooled bridge (`N = 1`, `pos_mode: none`) is eligible (`requires: {pos: [], streams: []}`); validity is the
  floor arm, not eligibility.
- **`pos`, when written, in media coordinates**: column 0 = stream index into `describe()["streams"]` (`video`,
  `audio`), `t` = token time / that stream's `info.duration_s`, `(y, x)` = fraction of the original frame, `-1`
  where a coordinate does not apply (audio tokens' `y, x`). Mixed frame sizes → `pos_mode: per_sample`. Tagging both
  streams in one token set is what lets the attentive probe compare their timing.
- **No spatial target** (`target_frame: null`).
- validate checks: `E_TIMES_OUT_OF_MEDIA` (a reported frame time outside `[0, duration_s]`, e.g. `span: [0, 8]`
  with `edge: error` on a row whose video is a frame short: declare `edge: clamp` or keep `n ≤ 16`),
  `E_RULE_MISMATCH` (reported `used` times vs the declared rule, ±(0.5/fps + 1 ms); audio ranges ±1 sample),
  `E_BATCH_INVARIANCE`, fixed `N`/`D`, `E_POS_RANGE`. `bridge check` adds rerun bit-identity and batch-/order-
  invariance (scrambled `row`/`clip_id`/`split`).

```yaml
output_contract:
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
# roots: [data] root in ~/.config/yavih/config.toml gives media_root ${data.root}/vggsound (via datasets.vggsound)
# and derived_root ${data.root}/avsync/materialized; per-site override:
export YAVIH_DATA__AVSYNC__MEDIA_ROOT=/path/to/vggsound
export YAVIH_DATA__AVSYNC__DERIVED_ROOT=/path/to/avsync/materialized

# 1. validate the recipe on 216 rows (8 per split x offset bin, seeded): ≈ 2 min on 28 cores
yavih bench prepare avsync 03 -- --rows 216 --workers 7
# 2. every row (32 997): ≈ 4.4 h at 7 workers x 4 encoder threads on one 30-core node; as an array, each shard
#    writes its own _materialize/log.<i>-of-<n>.jsonl and resumes from all logs:
#    yavih bench prepare avsync 03 -- --all --shard <i>/<n> --workers 3
yavih bench prepare avsync 03 -- --all --workers 7
# 3. presence, read sample, media-info census
yavih bench verify avsync --census
yavih bench verify avsync --json
```

Step 03 exits 0 when every row is verified or refused **by design** (`refused`: the source cannot fill the window
without padding, about 0.8 % of rows, or a source wav that is not 16 kHz mono PCM), 1 on an unexpected failure
(`failures`: ffmpeg error, missing source, frame/pts/audio verification), 2 on a wrong ffmpeg/x264 build. Refused
and failed rows stay missing (`min_eval_coverage` 0.95 absorbs them).

Success: `yavih bench verify avsync` prints `manifest sha ok; labels ok; curation ok` and per split `present` close to 24000/24000, 3000/3000, 5997/5997 (the refused rows are missing); `--json` adds
`manifest_hash 22a4e223…`, `curation_invariants_ok true`, `unknown_labels 0`. Then smoke the bridge:
`yavih bridge check my_avsync_bridge.py --bench avsync --arg reading=av --out avsync.profile.json`.

## Gotchas

Bridge-side mistakes that pass validation:

1. **The first-frame delay encodes δ's phase: take `pos` times from your rule, not from decoded pts.** The video
   keeps the source frame timing relative to the window start `1 + δ/2`, so its first frame sits at 0 or a
   fraction of a frame depending on δ mod (1/fps): at 30 fps every half-second offset starts at ≈ 1/60 s and every
   whole-second offset at 0. The container duration differs the same way (8.017 vs 8.000 s). A bridge that writes
   pts-derived `t` into `pos`, or a model that consumes timestamps, hands the probe δ's parity, and the floor arm
   fires. Use the declared rule's times (`rule_times`, identical for every row) for `pos`; report decoded pts in
   `used`.
2. **`span: [0.0, 8.0]`, not `"full"`.** The video track lasts 8 s ± one frame; `"full"` scales your times by a
   duration that can correlate with δ. With `[0, 8]` and `n ≤ 16` the last time (7.75 s) lies inside every frozen
   row (the curation's `grid_ok`); above `n = 24` declare `edge: clamp` and know that a clamped frame is a duplicate.
3. **1 fps cannot resolve a 0.5 s grid.** `n = 8` over 8 s scores at chance; report `acc_coarse5` as the comparable
   number of a 1 fps model and say so. At large `n` a 6 fps clip repeats frames (δ-independent, harmless).
4. **Normalization.** Constants from the model's own pretraining recipe; per-clip statistics only when they ARE the
   model's recipe (declare them in `norm` with their `source`; loudness/mel statistics of the 8 s window move with
   δ); never batch or dataset statistics.
5. **No path-, clip-id- or content-keyed caches.** Two train rows of one clip are two different files and
   `clip_id`'s `#<bin>` suffix IS the label: a cache keyed on the clip serves one offset's tensors for all nine bins
   while reporting plausible numbers.
6. **Never shift, pad, trim or re-centre one track against the other**: read both over the same span `[0, 8]`.
7. **One feature set for `av`.** Concatenate the two token sets and tag streams in `pos` (video via `grid_pos(...,
   stream=0, t_centres=…, region=crop_region(...))`, audio tokens with `y = x = -1`); two independently pooled
   vectors lose the cross-stream timing the probe needs. Same `N` for every sample.
8. **Mixed and portrait resolutions** (128×96 to 1280×720, portrait 204×360 / 406×720): a resize-short +
   centre-crop recipe shows a different part of each frame; declare it (`spatial`) and use `per_sample` pos.
9. **Whole-file recipes are fine.** The materialized file is the window. Only a model that cannot take 8 s of input
   raises `yavih.NotApplicable(reason)` from `__init__`.
