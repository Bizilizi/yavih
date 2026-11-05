# thumos14_oad — setup

The agent's first read for thumos14_oad. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark
is; [streaming benchmarks](https://github.com/Bizilizi/yavih/blob/main/docs/streaming.md) describes streaming benchmarks in general.

## Obtain the data

`source.kind: public_url`, no account: the official temporal annotations of THUMOS'14 and the 413 videos they name,
from the UCF server, one file each (34.8 GB). The server sends its certificate without the intermediate CA; step 01
completes the chain from the certificate's own issuer URL and keeps verification on. `--sha` checks every video
against `video_sha256.csv`.

```bash
yavih bench prepare thumos14_oad 01 -- --workers 6 --sha    # annotations + 413 videos into the media root
yavih bench prepare thumos14_oad 02                          # yavih bench verify thumos14_oad --census
```

## Dependencies

None: `external: []`. Step 01 needs `openssl`. The `stream` scorer needs torch in the scoring environment.

## Input contract

Per row, two streams over ONE stored file `videos/{validation,test}/<name>.mp4`: 320x180, 30 fps (18 at 29.97, 11
at 25, 2 at 24), 9 s to 28 min, each with an audio track. No facts. The bridge reads the recording causally: each
update with cutoff `c` uses input at or before `c` only.

```yaml
handed_over:
  video: the THUMOS'14 mp4 exactly as stored (320x180, mostly 30 fps, 9 s to 28 min); the bridge reads it causally, update
    by update
  audio: the same mp4 bytes as video (read once); its audio track is described by info; the bridge reads it causally, update
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

Per recording, its update sequence at the cutoffs `0.25, 0.5, ...` up to the recording's duration (`yavih.stream`):
`emb [n, D]`, `support [n, 4]`, `cutoff [n]`, and in a real-time (`-rt`) reading the replay log `timing`.

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
  update_interval_s: 0.25
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
yavih bench prepare thumos14_oad 02     # = yavih bench verify thumos14_oad --census
```

Success: `yavih bench verify thumos14_oad` reports validation 200/200 and test 213/213 present.

## Gotchas

- **Long videos.** Up to 28 minutes, about 850 updates on average: an offline window encoder recomputes every window.
- **Low resolution.** Every video is 320x180: upsampling to a model's input size is the bridge's declared choice.
- **Not the flow leaderboard.** Published THUMOS OAD numbers combine RGB and optical flow; a frozen RGB encoder is a
  different comparison.
- **Ambiguous chunks** are neither trained on nor scored; the bridge still processes them (the stream is continuous).
