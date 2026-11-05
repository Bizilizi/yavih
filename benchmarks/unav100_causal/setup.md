# unav100_causal — setup

The agent's first read for unav100_causal. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark
is; [streaming benchmarks](https://github.com/Bizilizi/yavih/blob/main/docs/streaming.md) describes streaming benchmarks in general.

## Obtain the data

`source.kind: public_url`: the Hugging Face dataset `ttgeng233/UnAV-100` at revision
`daeb88bc14eeff48f8a4ac2836b67f6c350f847b`, public, no account. 66 GB: three annotation parquets and the authors'
video archive in 13 parts (sha256 of every part checked by step 01; `SKIP_SHA=1` checks sizes only). Licence: treat
it as CC BY-NC-SA 4.0 (the dataset card's; the project page says CC BY 4.0); research use.

```bash
yavih bench prepare unav100_causal 01     # yavih bench snapshot unav100_causal + the 13 parts' sizes and sha256
yavih bench prepare unav100_causal 02     # extract raw_videos/<video_id>.mp4 (LINK=<dir> links an existing extraction)
yavih bench prepare unav100_causal 03     # yavih bench verify unav100_causal --census
```

Disk: 66 GB for the snapshot plus 66 GB for the extracted videos (delete the parts afterwards if space is short; the
manifest reads only `raw_videos/`).

## Dependencies

None: `external: []`. Preparing needs `pyarrow` and `huggingface_hub` (the `hf` extra); the `stream` scorer needs torch
in the scoring environment; the bridge decodes with its own stack or `yavih.media.decode`.

## Input contract

Per row, two streams over ONE stored file `raw_videos/<video_id>.mp4` (h264 + AAC, 9.5 to 60.1 s, 6 to 52 fps, 242
resolutions from 128x96 to 1280x720, 44.1 kHz audio except one 48 kHz file). No facts. The bridge reads the recording
causally: each update with cutoff `c` uses input at or before `c` only.

```yaml
handed_over:
  video: the UnAV-100 mp4 raw_videos/<video_id>.mp4 exactly as stored (h264, 6-52 fps, 9.5-60.1 s, up to 1280x720); the bridge
    reads it causally, update by update
  audio: the same mp4 bytes as video (read once); its AAC track (44.1 kHz) is described by info; the bridge reads it causally,
    update by update
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
yavih bench prepare unav100_causal 03     # = yavih bench verify unav100_causal --census
```

Success: `yavih bench verify unav100_causal` reports train 6487/6487 and test 2166/2166 present.

## Gotchas

- **Future input.** A centered window or an audio frame that extends past the cutoff reads the future; `bridge check`
  refuses it (`E_FUTURE_LEAK`).
- **Durations.** Use the recording's own duration (`info`), never the annotated one; 128 videos differ from it by 0.5 s
  or more. The bins follow the file.
- **Long recordings.** About 84 updates per video: an offline window encoder recomputes every window; cache only what
  is exactly reusable, and count the recomputation in a real-time reading.
- **Concurrent events.** Several classes can be active in one bin: the heads are multilabel, never a softmax.
- **Pretraining overlap.** UnAV-100 videos come from VGGSound; a model pretrained on VGGSound may have seen test videos.
  Audit before claiming a number.
