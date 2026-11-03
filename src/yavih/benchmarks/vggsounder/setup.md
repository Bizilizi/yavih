# vggsounder — setup

Read this before anything else for this benchmark. `benchmark.yaml` is the machine truth; README.md says
what the benchmark is.

## Obtain the data

Three parts: the VGGSounder labels (PyPI), the official VGGSound listing, and the VGGSound media (YouTube).

```sh
yavih bench prepare vggsounder      # lists the steps
yavih bench prepare vggsounder 01   # labels: pip install + sha256 of the bundled CSVs
yavih bench prepare vggsounder 02   # vggsound.csv (fetched + verified); media download is manual (below)
yavih bench prepare vggsounder 03   # layout and presence check of the staged tree
```

- **VGGSounder annotations** (public, Apache-2.0, bundled in the PyPI package, offline): `classes.csv`
  (309 classes, sha256 `90a5cd19345019f50d0c3540629c1f4cde33266f00c57ce0cf67cd69835993c2`), `vggsounder.csv`
  (`400979ea3dee34bed145d3fd96966e16689352df63b645c4730caf8eb2cacb06`), `vggsounder+background-music.csv`
  (`6279875ac93df11d90b1a9d60da7dee89073ce233b0843759856bc72e00396ba`). The frozen eval labels equal
  `VGGSounder(background_music=False)` exactly (the package's `a`, `v`, `all` ground truths equal the manifest's
  `audible`, `visible`, `all` sets, its `a only` / `v only` / `av` ground truths the `labels_a` / `labels_v` /
  `labels_av` cells; the folder test pins this).
- **VGGSound listing** (public, CC BY 4.0): `metadata/vggsound.csv` from
  https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv (sha256
  `ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8`, 199 176 rows, no header). Step 02 fetches
  and verifies it; the train draw depends on its row order, so another copy (e.g. the one in the authors' GitHub
  repository, a different sha256) is refused.
- **VGGSound media** (YouTube; decays; you download it yourself): 199 176 mp4 + 199 176 wav, 318 GiB + 61 GiB
  (`source.bytes` ≈ 407 GB). The frozen manifest touches 32 008 clips: **67.0 GB** (`budget.media_gb`). A fresh
  download will miss decayed clips; prefer an existing copy of the corpus if you have access to one. Step 02's
  yt-dlp loop (`VGGSOUND_DOWNLOAD=1`) is illustrative: drop its `[height<=360]` filter to match the frozen media
  (native resolution, about half 720p), and use a downloader with a proxy pool, since bare yt-dlp is rate-limited
  within minutes. Each clip is the 10 s window `[start, start + 10)` → `video/<ytid>_<start6>.mp4` (H.264) +
  `audio/<ytid>_<start6>.wav` (16 kHz mono). Expect decay: a re-download will not cover every row. The package can
  also fetch test-split media from HF `11hu83/vggsound` (`VGGSounder(download_samples=True)`); not used by this
  benchmark (it does not cover the train split).

Disk / inode budget: loose tree 379 GiB and 398 352 inodes for the whole corpus; the frozen rows alone are
67.0 GB / 64 016 inodes (train 42.3 GB, eval 24.7 GB). Never extract bulk tars in place next to the tree.

Data root: `[datasets] vggsound` in your yavih config (the yaml default is `${datasets.vggsound}`), or
`export YAVIH_DATA__VGGSOUNDER__MEDIA_ROOT=/path/to/vggsound`. The root holds `video/`, `audio/` and
`metadata/vggsound.csv`.

## Dependencies

The probe env needs the authors' package, installed independently (it is not a yavih dependency):

```sh
pip install --no-deps vggsounder==1.0
```

- `--no-deps` is load-bearing: the package declares `datasets>=4.5.0`, whose dependencies can upgrade numpy past
  what the installed torch supports. Its scorer (`vggsounder.benchmark`) imports **pandas, scikit-learn and
  tqdm**, which `--no-deps` does not install: `pip install pandas scikit-learn tqdm` if absent.
- Assert-import (`benchmark.yaml` `external`, `eval.py` → `yavih.external.require`): `import vggsounder` must
  succeed and `importlib.metadata.version("vggsounder")` must equal `1.0`, else the probe exits 3
  (`E_EXTERNAL_MISSING` / `E_EXTERNAL_VERSION`) before training, printing the pip line. `yavih doctor --bench
  vggsounder` shows it. The pin is the **PyPI release**: a source checkout of the authors' repo reports another
  version (e.g. `1.0.0`) and its HEAD renamed the `auc_pr` result key to `mAP`, so it is refused.
- The version used is recorded in `result.json[external]` and the results file's `config.external`.
- The model env (bridges) needs none of this: bridges see media only. Decoding there is the bridge's own stack
  (the template uses PyAV + soundfile + pillow: `pip install av soundfile pillow`).

## Input contract

Each row hands over two `RawStream`s of the stored files, nothing decoded, plus `meta == {"facts": {}}` (no label
facts):

- `video` — the VGGSound mp4 exactly as stored: H.264 with its **own 44.1 kHz AAC track** (stereo or mono),
  native and mixed frame size: train / eval 720×1280 9 853 / 5 960, 360×480 2 867 / 1 948, 360×640
  2 851 / 1 221, 240×320 947 / 803, then 720×406, 360×540, 720×960, 360×202, 320×568, … with 1 029 / 503 portrait
  clips and no rotation tags; frame rate 30 (9 100 / 6 941), 29.97, 25, 23.976, 24, 15, 29, down to 6 fps (469 / 197
  clips below 20 fps; one 60 fps train clip); 30–839 frames; duration median 10.0 s, max 10.23 s, shortest 2.0 s
  (train) and **1.0 s (eval)**; 472 / 541 videos shorter than 10.0 s.
  Guaranteed `info`: `duration_s, fps, width, height, container, bytes` (`n_frames` is also non-null on every
  frozen row, but not guaranteed).
- `audio` — the VGGSound wav exactly as stored: 16 kHz mono PCM-16 on every row; median 10.008 s, max 11.35 s, shortest 2.07 s (train)
  and **1.02 s (eval)**; **989 / 20 000 train and 1 334 / 12 008 eval files are shorter than 10.0 s** (334 / 420
  below 9.5 s, 37 / 105 below 8 s).
  Guaranteed `info`: `duration_s, sample_rate, n_samples, channels, container, bytes`.
- The two files of a row are separate cuts of one YouTube window and need not have equal durations
  (|audio − video| median 0.014 s, 99th percentile 0.18 / 0.36 s, max 3.3 s).
- Facts above: header census of the 32 008 frozen rows (`yavih bench prepare vggsounder 04`; 0 unreadable
  headers); `manifest.upstream.curation.short_clips: 2350` = rows with either file < 10.0 s (train 995, test 1 355).
  Nothing is cut or padded for them; every row stays scored.
- The bridge decides and declares (`describe()["input"]`): which frames (`uniform` over each clip's own duration
  is the natural rule; a fixed `[0, 10]` window needs `edge: clamp` or fails `E_TIMES_OUT_OF_MEDIA` on the short
  clips), the resize / crop for the mixed and portrait geometry, the audio window and what happens to a short file
  (`span` with `pad: zeros`, or `full`), resampling and the front end, the normalization constants (the model's
  pretraining recipe), the decoder.

```yaml
handed_over:
  video: >-
    the VGGSound mp4 exactly as stored (H.264, native and mixed frame size — about half 720x1280, the
    rest 360x480/360x640/240x320 and some portrait —, ~10 s at ~30 fps, with its own AAC track);
    nothing decoded, cut, resized or padded
  audio: >-
    the VGGSound wav exactly as stored (16 kHz mono 16-bit PCM, nominal 10 s; ~11 % of eval and ~5 %
    of train files are shorter, down to ~1 s); nothing resampled, trimmed or padded
info:
  video: [duration_s, fps, width, height, container, bytes]
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

- `feats`: one fixed `[N, D]` for every row of every split (`shape: fixed`); `float16` or `float32`.
- `pos`: not required — `clf` reads no positions (`requires: pos [], streams []`), so every bridge with `N ≥ 1` is
  eligible. When a bridge writes `pos`, it is in media coordinates (`coordinates: media`): `t` = fraction of that
  stream's own `duration_s`, `(y, x)` = fraction of the ORIGINAL frame (before any resize or crop), `-1` = no
  coordinate, `t = -1` for tokens centred in padding the bridge added. Since durations and frame sizes differ
  between clips, such positions are `per_sample`.
- No spatial targets (`target_frame: null`); targets are label sets from the manifest, scored by the official
  package.
- `validate` checks the reported `used` against the declared rules and the row's facts (`E_RULE_MISMATCH`,
  `E_TIMES_OUT_OF_MEDIA`).

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
yavih bench verify vggsounder --census     # presence, read/decode sample, labels, curation sums, media-info table
yavih doctor --bench vggsounder            # in the probe env: the vggsounder==1.0 assert-import
```

Expected on a fully staged site:

```
vggsounder: manifest sha ok; labels ok; curation ok; targets_sha_ok=None; derived_sha_ok=None; status verified
  train    present 20000/20000  missing_first10=[]
  test     present 12008/12008  missing_first10=[]
  read sample ok=32 failed=0
  decode sample ok=32 failed=0 ...
  media_info present rows=64016 ...
```

A copy with decayed clips reports them as missing; below 95 % eval coverage the probe refuses.

Scoring: `yavih eval run <run>` calls `eval.py` `score()` with the run's `describe()["reading"]`, which routes
exactly as the package does: reading `a` → its logits under pkl key `a`, scored on the sets `a` (ids `<metric>`)
and `a only` (`<metric>_only`); `v` → `v` and `v only`; `av` → `av` only (its `*_only` ids are `null`). The 30
metric names are the package's `compare_labels_logits` keys (`f1@k`, `f1_macro@k`, `accuracy@k`, `jaccard@k`,
`precision@k`, `recall@k`, `hit_rate@k` for k = 1, 3, 5, 10; `auc_roc`, `auc_pr`). After probing the three
readings of one model, `yavih report build runs/* --join vggsounder a=<run_a> v=<run_v> av=<run_av>` scores them
together with `compute_all_results` (the 30 metrics on all five sets plus `mu` for `a`, `v`, `av`, under the key
`vggsounder:a+v+av`; markdown: rows A / V / AV with mAP, HIT, F1 and mu A / V / A∩V, everything under
`--detail`). The three runs must be the
audio-only (`a`), video-only (`v`) and audio-visual (`av`) readings of ONE model (one `model` name) under one
standard probe, with one `protocol_hash` and `eval_hash`; `join()` refuses anything else (exit 2). Always score the
full eval population: a sparse `subset_ids` that empties a package ground-truth category raises an opaque pandas
`KeyError` inside the package.

## Gotchas

Bridge-side mistakes that pass validation:

1. **Reading the mp4's audio track.** Every mp4 carries its own 44.1 kHz AAC track. The benchmark's audio is the
   `audio` stream (the 16 kHz wav); an "av" bridge that decodes audio from the mp4 reads a different, lossy signal
   and declares the wrong stream in `describe()["input"]`.
2. **A fixed 10 s window on short media.** 1 334 eval wavs and 541 eval mp4s are shorter than 10 s (down to ~1 s).
   A rule over `[0, 10]` with `edge: error` fails `E_TIMES_OUT_OF_MEDIA` on them; with `edge: clamp` it repeats the
   last frame. `uniform` with `span: full` spreads the frames over each clip's own duration instead (n = 8 gives bin
   centres ≈ 0.625 … 9.375 s on a 10 s clip; declare an `explicit` / `fps` rule if the model was trained on another
   recipe). For audio, `span` with `pad: zeros` zero-pads the tail: the model's fbank then sees `log(eps)` frames; a
   per-clip normalization would shift with the padding — use the model's fixed constants, never per-batch
   statistics, and per-clip statistics only when they are the model's own pretraining recipe (then declare them in
   `norm` with their `source`).
3. **Geometry.** Frames are native and mixed (about half 720×1280, ~5 % portrait). Resize short side → 256 →
   centre-crop 224 is a common choice; declare your geometry in `describe()["input"]["video"]["spatial"]` — it is
   recorded with every number. Size batch memory for 720p (16 decoded 720×1280 frames are 44 MB uint8, 5.3× a
   360×480 clip); 720p decode dominates extraction cost.
4. **Readings.** `describe()["reading"]` ∈ {`a`, `v`, `av`}; `describe()["input"]` and `streams` must list only
   what the reading reads (an audio-only bridge that also decodes video wastes ~90 % of decode time and confuses the
   join). One `model` name for the three readings, or `report --join` cannot pair them (`E_JOIN_MODEL`); each
   reading lands in its own run directory `runs/vggsounder/<model>/<reading>/`.
5. **Dead rows / duplicates.** No two rows share media here: `E_DUPLICATE_CLIPS` on this benchmark means the bridge
   ignores its input. A constant feature on short-audio rows (a model that reads only the padded tail) shows up as
   `E_DEAD_ROWS`.
6. **Reading the numbers.** Each reading's `auc_pr` is on a different set (`a`: 11 911 clips, `v`: 10 182, `av`:
   9 883; `a only`: 6 392, `v only`: 850 — the package scores a set over the clips holding a label of it), so compare a reading with the same reading of another model, and
   the three readings of one model only through the join. `auc_roc` saturates early on 309-way sparse labels (0.9+
   is ordinary): rank on `auc_pr`. AP of a constant score is the class prevalence, so a probe that learned nothing
   scores the mean class prevalence of its set (about 0.008 on `a`).
