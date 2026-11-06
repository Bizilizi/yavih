# epic100_anticipation_stream — setup

The agent's first read for epic100_anticipation_stream. `benchmark.yaml` is the machine truth; `README.md` says what
the benchmark is; [streaming benchmarks](https://github.com/Bizilizi/yavih/blob/main/docs/streaming.md) describes streaming benchmarks in general.

## Obtain the data

`source.kind: public_url`, no account. Annotations: the `epic-kitchens-100-annotations` repository at commit
`ea8b40457a400c3fffa1c7f406ef3dc169cc2522` (step 01, sha256-checked). Videos: the official downloader
(`epic-kitchens-download-scripts` @ `4f11fb2b`) from data.bris, about 740 GB for the 633 train and validation videos
(step 02: the action-recognition train and validation videos, exactly the 633 the manifest names). Licence: CC BY-NC 4.0, non-commercial research.

```bash
yavih bench prepare epic100_anticipation_stream 01     # the 8 annotation csvs, sha256-checked
yavih bench prepare epic100_anticipation_stream 02     # the official downloader: --videos --action-recognition --train --val
yavih bench prepare epic100_anticipation_stream 03     # yavih bench verify epic100_anticipation_stream --census
```

## Dependencies

None: `external: []`. The `stream` scorer needs torch in the scoring environment; step 02 needs git and the
downloader's own requirements.

## Input contract

Per row, two streams over ONE stored file `EPIC-KITCHENS/<participant>/videos/<video_id>.MP4` (head-mounted GoPro,
mostly 1920x1080 at 50 or 60 fps, from a minute to about an hour). No facts. The bridge reads the recording causally:
each update with cutoff `c` uses input at or before `c` only.

```yaml
handed_over:
  video: the EPIC-KITCHENS-100 MP4 of the video exactly as stored (head-mounted camera, mostly 1920x1080 at 50 or 60 fps,
    minutes to an hour); the bridge reads it causally, update by update
  audio: the same MP4 bytes as video (read once); its audio track is described by info; the bridge reads it causally, update
    by update
info:
  video:
  - duration_s
  - fps
  - width
  - height
  - container
  - bytes
  audio:
  - container
  - bytes
facts: []
materialize: null
```

## Output contract

Per recording, its update sequence at the cutoffs `0.5, 1.0, ...` up to the recording's duration (`yavih.stream`):
`emb [n, D]`, `support [n, 4]`, `cutoff [n]`, and in a real-time (`-rt`) reading the replay log `timing`.
`describe()["stream"]` declares the windows, frame sampling, startup and gap policies and the device.

```yaml
shape: sequence
coordinates: media
target_frame: null
requires:
  pos: []
  streams: []
  grid: null
  t_bins: null
stream:
  update_interval_s: 0.5
schema:
  emb:
    type: array
    dtype:
    - float16
    - float32
    shape:
    - '*'
    - D
    stats: true
  support:
    type: array
    dtype: float64
    shape:
    - '*'
    - 4
    exact: true
    finite: false
  cutoff:
    type: array
    dtype: float64
    shape:
    - '*'
    exact: true
  timing:
    type: json
    kind: object
    optional: true
    volatile: true
```

## Prepare & verify

```bash
yavih bench prepare epic100_anticipation_stream 03     # = yavih bench verify epic100_anticipation_stream --census
```

Success: `yavih bench verify epic100_anticipation_stream` reports train 495/495 and validation 138/138 present and
the census `1266/1266 cells done (0 missing, 0 failed)`. A failed cell is a truncated download (the official
downloader does not re-verify the files it has just written; a dropped connection leaves an MP4 without its `moov`
atom, which `ffprobe` reports as `moov atom not found`): delete the file and re-run step 2 with
`EPIC_VIDEOS=<id>,<id>` to fetch only those videos, then re-run the census.

## Gotchas

- **Size.** Full-resolution 50/60 fps video: decoding dominates any bridge; decode at the model's rate and size.
- **Long recordings.** About 1 100 updates for a 9 min video and up to 8 000 for the longest: an offline window
  encoder recomputes every window.
- **Startup.** Segments that start within the first second are scored from the prior; keep the update grid from the
  start of the recording.
- **Existing downloads.** A torrent or partial download in another layout can be linked into
  `EPIC-KITCHENS/<participant>/videos/` (one symlink per participant directory); `bench verify` checks the files.
- **Future input.** No look-ahead of any kind: the future-invariance gate compares updates on copies blanked after a
  cutoff (`E_FUTURE_LEAK`).
