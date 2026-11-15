# mimetics — setup

Read this first. `benchmark.yaml` is the machine truth; `README.md` describes the benchmark.

## Obtain the data

Two data roots, two sources (`source.kind: public_url`; `obtain.tier: auto`, nobody signs or asks for anything):

1. **Mimetics, the test split** (`mimetics.media_root`, default `${data.root}/mimetics`). The release pinned by this
   folder is `source.revision` = Mimetics_release_v1.0.zip (December 2019, sha256 6ddc9c98c3f720baa5c70f1dc6cc05b80f1a16deb679a728082a698afb8680ab),
   21 869 B, from <https://europe.naverlabs.com/wp-content/uploads/2019/12/Mimetics_release_v1.0.zip> (the page
   <https://europe.naverlabs.com/research/computer-vision/mimetics/> links it). It holds `mimetics_v1.0.csv` (713
   clips: YouTube id, start and end second, label), `mimetics_v1.0_objectsceneannot.txt`,
   `mimetics_v1.0_clsannot.txt` and a README. The clips themselves are YouTube videos: the release README says to
   download and trim them with the Kinetics crawler (<https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics>),
   and step 02 is that crawler's `download_clip`, copied (MIT licence) and run over the 560 videos.
2. **Kinetics-400, the train split** (`mimetics.k400_root`, default `${datasets.kinetics400}`, i.e.
   `${data.root}/kinetics400` unless `[datasets] kinetics400` is set). The 50 Mimetics classes of the
   Academic-Torrents Kinetics-400 `_256` release (infohash `184d11318372f70018cf9a72ef867e2fb9ce1d26`), read in place
   as the `k400` benchmark stages it: `train/<class>/<clip_id>.mp4`. If it is not staged yet, stage it with the k400
   folder (`yavih bench fetch k400`, then `yavih bench prepare k400 01` and `02`: a 157 GB torrent download; read
   `yavih bench show k400 setup`). Only the 10 000 drawn train files (STAGED_TRAIN_GB GB) are read. Point
   `mimetics.k400_root` at an existing copy instead of making a second one.

```bash
yavih bench prepare mimetics          # lists the steps
yavih bench prepare mimetics 01       # the release zip: download, sha256, unzip into <media_root>/release/
pip install yt-dlp==2026.10.26        # the downloader step 02 calls (or set YTDLP=/path/to/yt-dlp)
yavih bench prepare mimetics 02       # 560 YouTube videos -> 713 clips <media_root>/clips/<clip_id>.mp4
yavih bench verify mimetics --census  # presence per split, a decode sample, the media-info table
```

- `01_release.sh` needs `curl`, `unzip`, `sha256sum`; it refuses a zip or csv with another sha256.
- `02_clips.py [--workers 4] [--limit N]` downloads each YouTube video once (`yt-dlp -f mp4`, five attempts) and cuts
  every clip of it with ffmpeg (`-ss <start> -t <end - start> -c:v libx264 -c:a copy`, plus a timestamp shift so the
  first video frame is at 0; ffmpeg >= 4.4 with libx264). Clips already staged are skipped, so it can be re-run to
  retry failures. It writes `<media_root>/download_report.csv` (clip, status, message, UTC time). It took about
  DOWNLOAD_HOURS hours with 4 workers here.
- **YouTube videos disappear.** On DOWNLOAD_DATE, STAGED_TEST of the 713 clips could still be downloaded here
  (`curation.per_split.test` records that census); the others report `unavailable` (removed, private or
  region-blocked videos). A missing test clip is simply absent: the probe scores the clips present and records
  the coverage; it refuses below `splits.min_eval_coverage` (MIN_COV) unless `--allow-low-coverage` is given. Never
  substitute another upload of a video.
- YouTube may also refuse a datacenter address ("Sign in to confirm you're not a bot"): those clips report
  `unavailable` with that message; re-run from another address, or with yt-dlp's own cookie options in `$YTDLP`.

Disk: the release is 22 kB; the staged Mimetics clips take TEST_MB MB (`<media_root>/clips/`); the download
needs room for one full YouTube video per worker at a time under `<media_root>/tmp/` (removed at the end). The
frozen rows touch MEDIA_GB GB in all (`budget.media_gb`). Nothing is materialized or derived (`data.derived: {}`).

## Dependencies

None. `external: []`: no package of the benchmark's authors is imported, and `yavih doctor --bench mimetics` lists
no external package. The scoring is grounded, part by part:

- **scorer**, rung 3 (no official evaluation code was released with Mimetics): IntegralAction,
  <https://github.com/mks0601/IntegralAction_RELEASE> at commit `50bc02ae8ad30d55628d6aecbd46f2dfa6173117`,
  `data/Mimetics/Mimetics.py` `Mimetics.evaluate` (no licence stated in that repository): its top-1 / top-5 loop is
  copied into this folder's `eval.py` (`integralaction_topk`). Its top-1 is exactly yavih's `acc` (argmax equals the
  label, over every scored clip), which is the headline; `top5` is computed by the copied loop. The paper's
  class-averaged metrics (`mean_top1`, `mean_top5`, `mean_ap`) are written from the paper's text (rung 4) in the
  same file.
- **readout**: the core `clf` probe on frozen features (no folder head).
- **prepare**: the ActivityNet Kinetics crawler (<https://github.com/activitynet/ActivityNet>, commit
  `82304fa9f4a7219421df13dae6347a445c4cf5bc`, `Crawler/Kinetics/download.py`, MIT) copied into
  `prepare/02_clips.py`, with the changes listed in its docstring.

Tools outside yavih: `curl`, `unzip`, `sha256sum` (step 01); `yt-dlp` (2026.10.26 here), `ffmpeg` and `ffprobe`
>= 4.4 with libx264 (step 02); torch at freeze time only (`make_manifest.py`'s seeded draw). Model env: the bridge's
own video decoder (PyAV, decord, torchvision.io, …) or `yavih[media]` (`yavih.media.decode`), declared in
`describe()["input"]["video"]["decoder"]`.

## Input contract

There is one stream, `video` (`kind: video`, `layout: dir`, `required: true`), read from two roots: train rows from
`k400_root` (`train/<class>/<clip_id>.mp4`), test rows from `media_root` (`clips/<clip_id>.mp4`). It is the stored
mp4 **as is**, handed over as a `RawStream` (`path`, `bytes()`, `open()`, `info`); nothing is decoded, cut,
resized, padded or normalized. `meta == {"facts": {}}`: the label, the class list, the split, the clip id and the
per-clip `object` / `scene` / `mime` flags never reach the bridge.

The guaranteed `info` keys are non-null on every row: `duration_s` (the video stream's), `fps`, `width`, `height`,
`container`, `bytes`; `n_frames` may be null. Facts of the staged rows (`yavih bench verify mimetics --census`):

| fact | train (Kinetics-400, 10 000) | test (Mimetics, STAGED_TEST staged) |
|---|---|---|
| container / codec | CENSUS_TRAIN_CODEC | CENSUS_TEST_CODEC |
| size | CENSUS_TRAIN_SIZE | CENSUS_TEST_SIZE |
| fps | CENSUS_TRAIN_FPS | CENSUS_TEST_FPS |
| duration (video stream) | CENSUS_TRAIN_DUR | CENSUS_TEST_DUR |

The bridge decides, and declares in `describe()["input"]["video"]`:
- **temporal**: which frames. The template's default is the reference's: 8 frames 8 stored frames apart (an `fps`
  rule at 3.75 per second), one window centred on the clip, clamped on clips shorter than 2.13 s. `uniform` over
  `span: "full"` is the other natural choice. One rule for both splits.
- **spatial**: resize and crop; the template's default (short side 256, centre crop 224) is the reference's.
- **norm**: the model's own pretraining constants with their source. **decoder**: the library and its version.

```yaml
handed_over:
  video: >-
    the stored mp4 as is: for train rows a Kinetics-400 clip of the Academic-Torrents _256 release (h264, 256
    pixels high, about 10 s), for test rows a Mimetics clip as the Kinetics crawler trims it from the YouTube
    video (h264 re-encode at the downloaded resolution, mostly 360 pixels high, 1 to 10 s); nothing decoded, cut,
    resized, padded or normalized
info:
  video: [duration_s, fps, width, height, container, bytes]
facts: []
materialize: null
```

## Output contract

- **Shape**: one fixed `[N, D]` per clip for every row of both splits, `float16` or `float32`.
- **Positions**: `clf` requires none, so pooled features (`N = 1`, `pos_mode: none`) or tokens for the attentive
  probe are both eligible. Positions a bridge does write are in media coordinates: `t` is the token centre in
  seconds over `info.duration_s`, `(y, x)` fractions of the ORIGINAL frame (`crop_region`, `grid_pos(..., region=)`);
  clips differ in duration and size, so positions are `per_sample`.
- **No spatial target** (`target_frame: null`).
- **What validate checks**: `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH` (the declared rule re-evaluated on each clip's
  facts against the reported `used` times, tolerance `0.5 / fps + 1 ms`), `E_POS_RANGE`, `E_BATCH_INVARIANCE`, a
  fixed `N`/`D`, `E_DEAD_ROWS` / `E_DUPLICATE_CLIPS`.

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
# per site, either environment variables
export YAVIH_DATA__MIMETICS__MEDIA_ROOT=/path/to/mimetics
export YAVIH_DATA__MIMETICS__K400_ROOT=/path/to/kinetics400
# or ~/.config/yavih/config.toml:
# [data.mimetics]
# media_root = "/path/to/mimetics"
# k400_root = "/path/to/kinetics400"

yavih bench verify mimetics --census
yavih bridge check my_mimetics_bridge.py --bench mimetics --out mimetics.profile.json
```

A pass prints `train present 10000/10000` and `test present STAGED_TEST/713` (or however many your download got),
a decode sample with 0 failures, and with `--census` the media-info table. `bridge check` must report a
bit-identical rerun and batch and order invariance; its smoke sample may report `W_MISSING_ROWS` for test clips
that YouTube no longer serves.

## Gotchas

1. **Two kinds of media in one benchmark.** Train clips are Kinetics-400 re-encodes 256 pixels high, ~10 s; test
   clips are fresh YouTube trims, mostly 360 pixels high, 1-10 s. One preprocessing for both; never branch on the
   size or the duration. The domain gap is part of the task.
2. **Short test clips.** 59 release clips last 1 s and 134 last 2 s. A fixed-length window (the template's 2.13 s)
   is clamped to the last frame on them (`edge: "clamp"`); `edge: "error"` fails with `E_TIMES_OUT_OF_MEDIA`. Never
   drop a clip: every staged clip is scored (the reference implementation drops clips shorter than its window;
   yavih does not).
3. **One view.** `views: 1`; the reference averages 10 temporal windows per clip, a yavih bridge declares one.
4. **The first frame is at 0.** Step 02 shifts each trimmed clip's timestamps so its first video frame is at pts 0.
   A clip cut by another tool (the crawler as published) often starts at the second frame of its second (16-47 ms),
   and a bridge's declared frame times then miss by one frame (`E_RULE_MISMATCH` in `bridge check`). Re-trim with
   step 02 rather than editing the bridge.
5. **Classes are 50, not 400.** The probe is 50-way; class names are the Academic-Torrents directory names
   (`punching_person_-boxing-`). Post-process `eval_logits.npy` with this folder's `classes.csv`, never with the
   400-class list of `k400`.
6. **Missing test clips.** Coverage below 1 is expected (YouTube removals) and recorded with every number; two
   sites with different downloads score different subsets.
