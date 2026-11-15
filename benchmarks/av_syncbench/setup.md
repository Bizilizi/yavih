# av_syncbench — setup

## Obtain the data
TODO: where the media come from, the exact download commands, sizes and sha256s; who must act first
(`obtain.tier` in `benchmark.yaml`: auto, manual-download, needs-licence, needs-data-owner or site-only). Commands (`yavih bench prepare` runs the folder's numbered prepare/ scripts):

```bash
yavih bench prepare av_syncbench      # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
yavih bench prepare av_syncbench 01   # TODO
```

## Dependencies
None. TODO: list the packages of the benchmark's authors the probe env needs (exact pip lines), or keep None.

## Input contract
TODO: what the reader hands over, per stream, in one sentence each; everything else is the bridge's declared decision.

```yaml
handed_over:
  video: the VGGSound mp4 exactly as stored (H.264, native and mixed frame size, ~10
    s at ~30 fps, with its own AAC track, which the video embedding must not read);
    nothing decoded, cut, resized or padded
  audio: the VGGSound wav exactly as stored (16 kHz mono 16-bit PCM, nominal 10 s;
    some files shorter, down to ~1 s); nothing resampled, trimmed or padded
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
TODO: what the probe needs from the features (positions, streams, grid, time bins).

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
  audio_emb:
    type: array
    dtype:
    - float16
    - float32
    shape:
    - D
    stats: true
  video_emb:
    type: array
    dtype:
    - float16
    - float32
    shape:
    - D
    stats: true
isolation:
  audio_emb:
  - audio
  video_emb:
  - video
```

## Prepare & verify
TODO: the expected output of `yavih bench verify av_syncbench` once the data is staged (presence per split, census).

## Gotchas
TODO: the setup and bridge mistakes that pass validation on this benchmark.
