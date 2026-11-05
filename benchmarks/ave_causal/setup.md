# ave_causal — setup

The agent's first read for ave_causal. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is;
[Streaming benchmarks](https://github.com/Bizilizi/yavih/blob/main/docs/streaming.md) describes streaming benchmarks in general.

## Obtain the data

The data are the `ave` folder's: one public HF tar (`AVE_Datset.tar`, 13.35 GB, sha256
`479905dc043ebae0e445c6ed1a109e5295a8f72b65ae959511db0a2bfc3a02c3`) read in place through its `index.json`. Stage them
with the `ave` folder's prepare steps (`yavih bench info ave --setup`), then point this folder at the same root:

```bash
yavih bench prepare ave 01 && yavih bench prepare ave 02 && yavih bench prepare ave 03 -- --root $ROOT
export YAVIH_DATA__AVE_CAUSAL__MEDIA_ROOT=$ROOT    # or [data] root so that ${data.root}/ave is the AVE root
yavih bench prepare ave_causal 01                   # yavih bench verify ave_causal --census
```

Licence: as for `ave` (none recorded upstream; YouTube clips; research use).

## Dependencies

None: `external: []`. The `stream` scorer needs torch in the scoring environment; the bridge decodes with its own
stack (PyAV, torchaudio, ...) or `yavih.media.decode`.

## Input contract

Per row, two streams over ONE stored file, the member `video/<ytid>.mp4` of `AVE_Datset.tar`, read in place (as in
`ave`): `video` (16-46 fps, 128x96 to 1280x720, 9.96-10.08 s) and `audio` (AAC, 44.1 kHz, mono or stereo, 9.7-10.1 s).
No facts. The bridge reads the recording causally: each update with cutoff `c` uses input at or before `c` only.

```yaml
handed_over:
  video: the AVE mp4 (tar member video/<ytid>.mp4 of AVE_Datset.tar) exactly as stored, read in place; its video track (16-46
    fps, 128x96 to 1280x720, 9.96-10.08 s) is described by info; the bridge reads it causally, update by update
  audio: the same mp4 bytes as video (one tar member, read once); its AAC audio track (44.1 kHz, mono or stereo, 9.7-10.1
    s) is described by info; the bridge reads it causally, update by update
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
  - n_samples
  - channels
  - container
  - bytes
facts: []
materialize: null
```

## Output contract

Per recording, its update sequence at the cutoffs `0.5, 1.0, ..., ` up to the recording's duration (`yavih.stream`):
`emb [n, D]`, `support [n, 4]` (video and audio start/end in seconds of what the update read; NaN for an unread
modality), `cutoff [n]`, and in a real-time (`-rt`) reading the replay log `timing`. `describe()["stream"]` declares
the windows, frame sampling, startup and gap policies and the device.

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
yavih bench prepare ave_causal 01      # = yavih bench verify ave_causal --census
```

Success: `yavih bench verify ave_causal` reports train 2592/2592 and test 316/316 present.

## Gotchas

- **Future input.** A centered window, an audio frame that extends past the cutoff, or normalization over the whole
  clip reads the future; `bridge check` refuses it (`E_FUTURE_LEAK`) by comparing updates on copies blanked after a
  cutoff.
- **Support.** Report the input each update actually read (decoded frame pts, audio sample range); a window that starts
  before 0 is padded with constants and its support starts at 0. A support that ends after its cutoff is
  `E_FUTURE_SUPPORT`.
- **The last segment.** Some tracks end before 10 s (as in `ave`): segment 9 is then scored from the last update
  before the end.
- **Real time.** A `-rt` reading replays at 1x on one device (`batch_size` 1); its number holds for that device only.
