# avrobustbench — setup

avrobustbench is the VGGSOUND-2C dataset of AVROBUSTBENCH: VGGSound clips under 15 audio-visual corruptions. It
needs the **VGGSound source media** (the tree `vggsounder` reads, used only at prepare time), the authors'
**corruption assets** (public, MIT), and the **materialized clips** that `prepare/04_materialize.py` writes from
them, one mp4 + one wav per manifest row: the files the reader hands over. People-facing description:
[`README.md`](README.md); machine truth: `benchmark.yaml`.

## Obtain the data

Pinned upstream: `AV-C-Robustness-Benchmark@f6c3f2e12de918373a36575f8fec14d84adc4707 + av_robust_data@42b5e0789ccd1a1545a9e2a4c68e6ecb499541bb`
(the authors' code repository at that commit, and their Hugging Face dataset at that revision).

- **VGGSound source media** (`media_root`, default `${datasets.vggsound}`): `video/<clip>.mp4` (H.264, native
  resolution), `audio/<clip>.wav` (16 kHz mono PCM-16), `metadata/vggsound.csv` (sha256
  `ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8`, 199 176 rows, no header). VGGSound media are
  YouTube clips (annotations CC BY 4.0, https://www.robots.ox.ac.uk/~vgg/data/vggsound/); they decay, so use an
  existing copy of the corpus (the `vggsounder` folder's tree). The manifest touches 34 046 source clips: the
  14 046 VGGSOUND-2C eval clips and 20 000 train clips (`LIST_ONLY=1 yavih bench prepare avrobustbench 01` lists
  them). A re-downloaded clip is new media and makes other corrupted clips.
- **Corruption assets** (`derived_root/upstream/corruptions`, step 03): the `corruptions/` directory of
  https://github.com/sarthaxxxxx/AV-C-Robustness-Benchmark at the commit above (ESC-50 / Freesound noise wavs, frost
  and crowd images; 146 files, 64 MB, every one sha256-checked; MIT).
- **VGGSOUND-2C clip list** (re-freezing only, step 02): the member names of `vgg_data/clean_vgg.zip` (3.0 GB,
  sha256 `74ca5d49efa3dc340695a16f2176d81082e868e48ba2ca24ad493ab863d48b3e`) of
  https://huggingface.co/datasets/sakshamsingh1/av_robust_data, read through HTTP range requests (no full
  download); the list's sha256 is `d8a5754e7a2824848209986f363dc92bc5802ea10c4b44d431f5e5a2583c244a`. The release's
  corrupted mp4s are not used (they lack `frost` and carry AAC audio).
- **Materialized media** (`derived_root`, default `${data.root}/avrobustbench`): `media/train/clean/<clip>.{mp4,wav}`
  (20 000 rows) and `media/test/<condition>/<clip>.{mp4,wav}` (16 x 14 046 = 224 736 rows), plus the logs
  `_materialize/log*.jsonl`. Size: see `budget.media_gb` (≈ 489 000 files).

```sh
yavih bench prepare avrobustbench                          # lists the steps
yavih bench prepare avrobustbench 01                       # verifies the VGGSound tree (metadata/vggsound.csv)
yavih bench prepare avrobustbench 02                       # re-freezing only: the VGGSOUND-2C clip list
yavih bench prepare avrobustbench 03                       # the authors' corruption assets, sha256-checked
yavih bench prepare avrobustbench 04 -- --all --workers 4  # materialize every row ("Prepare & verify")
```

## Dependencies

None (`external: []`); `yavih doctor --bench avrobustbench` lists nothing to install for scoring.

Grounding, per part of the evaluation:

- **Data preparation (the corruptions)**, rung 2 (official code, not a wheel): the authors' `corruptions/corruptions.py`
  of https://github.com/sarthaxxxxx/AV-C-Robustness-Benchmark @ `f6c3f2e12de918373a36575f8fec14d84adc4707` (MIT),
  copied to `prepare/corruptions.py` with five marked changes (asset root; sorted noise-file listing; `np.frombuffer`
  for the removed `np.fromstring`; `wind_visual` without the line that makes it raise on every input; a seeded `rng`
  for scikit-image's salt-and-pepper noise). The per-row pipeline of their `dataset.py` (`get_image`, `get_wav`) is
  transcribed in `prepare/04_materialize.py`. The frames follow CAV-MAE's `extract_frame`, the preprocessing the
  AVROBUSTBENCH README names: https://github.com/YuanGongND/cav-mae @ `68fe8c2a3917dc2926e41f796bfdcb331a64b42c`
  `src/preprocess/extract_video_frame.py` (BSD 2-Clause), copied to `prepare/cavmae_frames.py`.
- **Scorer**: top-1 accuracy per condition is the core `clf` accuracy (`yavih.metrics.classification.correct`); the
  mean over the 15 corruptions is the paper's Table 4 definition (rung 4: the authors released no evaluation code);
  `alpha` / `rho` are the formulas of the implementation the paper cites for them, Schiappa et al.
  `robustness_scores.py` (https://github.com/Maddy12/VideoLanguageModelRobustness @
  `005fd427b7a393adbc050a8ac7f32a48f4388aa6`, licence none stated), transcribed in `eval.py` (rung 3).
- **Readout**: the core `clf` attentive probe trained on the clean train clips (no procedure).

Prepare-time tools (steps 03-04 only; never in the model env), each pinned in `input_contract.materialize.tool`
and checked by step 04 (exit 2 on a mismatch; `--allow-version-drift` runs anyway and records the versions):

- **ffmpeg `4.4.2-0ubuntu0.22.04.1`** with **libx264 `core 163 r3060 5db6aa6`** (Ubuntu 22.04's `apt install
  ffmpeg`), **ImageMagick `6.9.11-60`** (`libmagickwand-6.q16-6`, what Wand loads).
- Python, in the yavih environment: `pip install opencv-python-headless==4.11.0.86 scikit-image==0.25.2 Wand==0.6.13`
  on top of yavih's own numpy 2.4.6, torch 2.3.1, torchaudio 2.3.1, torchvision 0.18.1, scipy 1.17.1, pillow 11.2.1,
  soundfile 0.13.1 (`tool.python`), plus PyAV for the verification. The authors pinned opencv-python 4.11.0.86,
  scikit-image 0.25.2, pillow 11.2.1 and Wand 0.6.13 too.

## Input contract

What the generic reader (`yavih.sample.iter_samples`; no `reader.py`) hands the bridge per row:

- **`video`** — a `RawStream` over the row's materialized mp4 (`<derived_root>/<rel_video>`): **exactly 10 frames of
  224 x 224 at 1 fps** (pts 0, 1, ..., 9 s; `duration_s` 10.0), H.264 yuv420p (crf 23), no audio track. The frames
  are CAV-MAE's: frame `i` is source frame `int(i * total / 10)` (`total = min(frame count, 10 * fps)`), resized
  (short side 224) and centre-cropped, stored and re-read as a JPEG, then passed through the row's visual corruption
  at severity 5 (none for `clean`). `info`: `duration_s`, `fps` (1.0), `n_frames` (10), `width`, `height` (224),
  `container` (`mp4`), `bytes`.
- **`audio`** — a `RawStream` over the row's materialized wav (`<derived_root>/<rel_audio>`): the VGGSound 16 kHz
  mono wav through the row's audio corruption at severity 5 (none for `clean`), PCM-16, the source's length (nominal
  10 s; ~11 % of eval clips are shorter, down to ~1 s). **Mono, except most `crowd` rows** (7 of the 8 crowd
  recordings are stereo; the authors' `add_noise` broadcasts them, so channel `c` = clip + noise channel `c`).
  `info`: `duration_s`, `sample_rate` (16000), `n_samples`, `channels` (1 or 2), `container` (`wav`), `bytes`.
- **`meta = {"facts": {}}`**. A row's condition is not in `meta`, `info` or the stream; it is in the file path and
  in `clip_id` (`<clip>#<condition>`), which a bridge never reads (`bridge check` scrambles `clip_id`).

**Materialization** (`input_contract.materialize`, kind `avrobustbench_corrupt`, hashed): for each source clip,
step 04 extracts the 10 frames and loads the wav once, then per row seeds numpy, `random` and torch with
`int(sha256("avrobustbench:0:<clip_id>:video")[:8], 16)` and applies `corruption_dict[condition][0](frame,
severity=5)` to the 10 frames in order, seeds again with `...:audio` and applies `corruption_dict[condition][1]
(waveform=wav, intensity=5)`, encodes the frames with `ffmpeg *video_args <out>.mp4` (the settings moviepy used for
the authors' released mp4s) and writes the wav with `soundfile.write(..., format='WAV')` as the authors' `get_wav`
does. Every output is decode-verified before it is renamed into place, and its sha256 is logged with the recipe
sha256 (canonical JSON of the materialize block): a changed recipe never reuses an old file. Re-encoded bytes can
differ between hosts with the same builds, so no per-row sha256 ships; the recipe is the pin. The recipe also pins
the sha256 of the two copied code files (`code_sha256`) and of the asset list (`assets.list_sha256`).

The bridge decides, and declares in `describe()["input"]`: which of the 10 frames (an image model: one frame, e.g.
`fps` rule `n: 1, start: 4.0` for CAV-MAE's default frame 4; a video model: all 10 at 1 fps), the resize to its own
input size, the audio window (a 10 s window from 0, zero-padded after a short file), the front end, the channel (0)
of a 2-channel wav, normalization constants from the model's own pretraining recipe, and the decoder.

```yaml
input_contract:
  handed_over:
    video: >-
      the row's materialized mp4 (prepare/04_materialize.py): the 10 CAV-MAE frames of the VGGSound clip (224 x 224,
      resize-224 + centre crop, each a JPEG round trip), each passed through the row's visual corruption at severity 5
      (none for clean), encoded as a 1 fps, 10.0 s H.264 yuv420p mp4 (crf 23, no audio track)
    audio: >-
      the row's materialized wav: the VGGSound 16 kHz mono wav through the row's audio corruption at severity 5 (none
      for clean), written as 16-bit PCM at 16 kHz mono with the source's length (nominal 10 s; ~11 % of clips are
      shorter, down to ~1 s); nothing resampled, trimmed or padded
  info:
    video: [duration_s, fps, n_frames, width, height, container, bytes]
    audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
  facts: []
  materialize:   # the pin is this recipe (its sha256 is in every materialization log line), never the output bytes
    kind: avrobustbench_corrupt
    condition_column: condition
    source_columns: {video: src_video, audio: src_audio}
    severity: 5
    seed: 0
    frames: 10
    size: 224
    fps: 1
    code_sha256:   # prepare/corruptions.py (authors' code, copied) and prepare/cavmae_frames.py (CAV-MAE, copied)
      corruptions: 9754ffd0edd9332389c45f114514df85bc206da371366bafe7443271abb1783b
      cavmae_frames: 8a151d5b899d449f9ef922e287437f074e0a8233558f4cd838172a3dae6c95cc
    assets:   # the upstream corruptions/ directory, prepare/03_corruption_assets.sh (146 files, sha256 list)
      commit: f6c3f2e12de918373a36575f8fec14d84adc4707
      files: 146
      list_sha256: 98b79e6feed85577d45183433ac0458d7c5683ec0b5011464d9d8ff33975bf2a
    tool:
      name: ffmpeg
      version_pin: "4.4.2-0ubuntu0.22.04.1"
      # command = ["ffmpeg", *video_args, "<out>.mp4"], the 10 corrupted frames piped as rgb24; the encoder settings of
      # moviepy 1.0.3 write_videofile (libx264, preset medium, crf 23 default, yuv420p) that made the released mp4s
      video_args: [-nostdin, -hide_banner, -loglevel, error, -y, -f, rawvideo, -vcodec, rawvideo, -s, 224x224,
                   -pix_fmt, rgb24, -r, "1", -an, -i, "-", -vcodec, libx264, -preset, medium, -crf, "23",
                   -pix_fmt, yuv420p, -threads, "1", "-flags:v", +bitexact, -fflags, +bitexact,
                   -map_metadata, "-1"]
      python: [numpy==2.4.6, torch==2.3.1, torchaudio==2.3.1, torchvision==0.18.1, scipy==1.17.1, pillow==11.2.1,
               opencv-python-headless==4.11.0.86, scikit-image==0.25.2, Wand==0.6.13, soundfile==0.13.1]
      pins:
        libx264:
          value: "x264 - core 163 r3060 5db6aa6"
          probe: ["{tool}", -nostdin, -hide_banner, -loglevel, error, -f, lavfi, -i, "color=c=black:s=64x64:d=0.04",
                  "-frames:v", "1", -c:v, libx264, -f, h264, "-"]
          match: "x264 - core \\d+ r\\d+ [0-9a-f]+"
        imagemagick:
          value: "6.9.11-60"
          probe: [convert, -version]
          match: "\\d+\\.\\d+\\.\\d+-\\d+"
```

## Output contract

- **`[N, D]` fixed** for every row of a feature set (`shape: fixed`); `N ≥ 1`, float16 or float32. A pooled bridge
  (`N = 1`, `pos_mode: none`) is eligible (`requires: {pos: [], streams: []}`).
- **`pos`, when written, in media coordinates**: column 0 = stream index into `describe()["streams"]`, `t` = token
  time / that stream's `info.duration_s`, `(y, x)` = fraction of the stored 224 x 224 frame, `-1` where a coordinate
  does not apply. The `clf` probe ignores `pos`.
- **No spatial target** (`target_frame: null`).
- validate checks: fixed `N` / `D`, `E_RULE_MISMATCH` (reported `used` times vs the declared rule), `E_TIMES_OUT_OF_MEDIA`,
  `E_BATCH_INVARIANCE`, `E_POS_RANGE`; `bridge check` adds rerun bit-identity and batch- and order-invariance.

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
# and derived_root ${data.root}/avrobustbench; per-site overrides:
export YAVIH_DATA__AVROBUSTBENCH__MEDIA_ROOT=/path/to/vggsound
export YAVIH_DATA__AVROBUSTBENCH__DERIVED_ROOT=/path/to/avrobustbench

yavih bench prepare avrobustbench 01
yavih bench prepare avrobustbench 03
# a validation sample first: 64 eval clips x 16 conditions + 640 train clips (seeded choice)
yavih bench prepare avrobustbench 04 -- --sample-clips 64 --sample-train 640 --workers 4
# or exactly the rows a capped run reads (`yavih plan build --sample 64`): the first 64 rows of each split; eval rows
# are clip-major (a clip's 16 conditions are consecutive), so 64 eval rows = 4 whole clips
yavih bench prepare avrobustbench 04 -- --first 64 --workers 4
# then every row (244 736 rows of 34 046 source clips), e.g. as an array of shards, each with its own log:
#   yavih bench prepare avrobustbench 04 -- --all --shard <i>/<n> --workers 4
yavih bench prepare avrobustbench 04 -- --all --workers 4
yavih bench prepare avrobustbench 04 -- --status     # verified / listed / refused per split and condition
yavih bench verify avrobustbench --census
```

Step 04 exits 0 when every selected row is verified or refused **by design** (`refused`: cv2 cannot read one of the
10 frame indices, a source wav that is not 16 kHz mono, an audio corruption that cannot run on a very short wav), 1
on an unexpected failure (logged with `error`), 2 on a build that differs from the pins or missing corruption assets.
It resumes: rows with a log record of the current recipe and both files present are skipped. A capped run
(`plan build --sample N`) needs `N >= 16`, best a multiple of 16: every metric needs at least one scored row under
each of the 16 conditions, and the probe refuses (`E_METRICS_INCOMPLETE`) otherwise. Refused rows stay
missing; `min_eval_coverage` 0.95 absorbs them.

Success: `yavih bench verify avrobustbench` prints `manifest sha ok; labels ok; curation ok` and per split `present`
close to 20000/20000 and 224736/224736. Then smoke the bridge:
`yavih bridge check my_avrobustbench_bridge.py --bench avrobustbench --arg reading=av --out avrobustbench.profile.json`.

## Gotchas

Bridge-side mistakes that pass validation:

1. **Never read `clip_id` or the path.** Both name the condition (`#gaussian`, `media/test/gaussian/...`). A bridge
   that branches on them, or caches by clip, hands the probe the condition and inflates or deflates every corrupted
   number. Every row is its own file; treat it as such.
2. **The 16 conditions of a clip share their label and source.** A cache keyed on content hashes of the *source*
   (e.g. a feature cache by YouTube id) serves clean features for corrupted rows: the corruption must reach the model.
3. **Frames are 224 px already.** Resize to the model's input size only; a second `Resize(256) + CenterCrop(224)`
   zooms in and crops away part of the corruption (frost, crowd occluders and spatter sit anywhere in the frame).
4. **Read channel 0 of a 2-channel wav** (most `crowd` rows), as CAV-MAE's Kaldi fbank does; averaging the channels
   halves that noise and changes the number.
5. **Short wavs.** ~11 % of clips are shorter than 10 s; pad after the end (as CAV-MAE does), never loop or stretch,
   and declare `pad`.
6. **Normalization.** Constants from the model's pretraining recipe; per-clip statistics only when they ARE the
   model's recipe (declared in `norm` with their `source`). Per-clip audio normalization rescales the noise the
   corruption added and must be the model's own, never added for this benchmark.
7. **One feature set for `av`.** Concatenate the video and audio token sets (tag streams in `pos`); the probe is
   trained once and scored on every condition, so `N` and `D` must not depend on the row.
