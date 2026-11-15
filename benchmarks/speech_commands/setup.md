# speech_commands — setup

## Obtain the data
TODO: where the media come from, the exact download commands, sizes and sha256s; who must act first
(`obtain.tier` in `benchmark.yaml`: auto, manual-download, needs-licence, needs-data-owner or site-only). Commands (`yavih bench prepare` runs the folder's numbered prepare/ scripts):

```bash
yavih bench prepare speech_commands      # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
yavih bench prepare speech_commands 01   # TODO
```

## Dependencies
None. TODO: list the packages of the benchmark's authors the probe env needs (exact pip lines), or keep None.

## Input contract
TODO: what the reader hands over, per stream, in one sentence each; everything else is the bridge's declared decision.

```yaml
handed_over:
  audio: the ESC-50 wav exactly as stored (RIFF PCM_16, 44.1 kHz, mono, 220 500 samples
    = 5.000 s); nothing decoded, resampled, trimmed or padded
info:
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
```

## Prepare & verify
TODO: the expected output of `yavih bench verify speech_commands` once the data is staged (presence per split, census).

## Gotchas
TODO: the setup and bridge mistakes that pass validation on this benchmark.
