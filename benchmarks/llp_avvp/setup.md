# llp_avvp — setup

## Obtain the data
TODO: where the media come from, the exact download commands, sizes and sha256s; who must act first
(`obtain.tier` in `benchmark.yaml`: auto, manual-download, needs-licence, needs-data-owner or site-only). The pinned upstream revision (`source.revision`) is `daeb88bc14eeff48f8a4ac2836b67f6c350f847b` (TODO: still unav100's). Commands (`yavih bench prepare` runs the folder's numbered prepare/ scripts):

```bash
yavih bench prepare llp_avvp      # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
yavih bench prepare llp_avvp 01   # TODO
```

## Dependencies
None. TODO: list the packages of the benchmark's authors the probe env needs (exact pip lines), or keep None.

## Input contract
TODO: what the reader hands over, per stream, in one sentence each; everything else is the bridge's declared decision.

```yaml
handed_over:
  video: the UnAV-100 mp4 (tar member video/<ytid>.mp4 of unav_raw_videos.tar) exactly
    as stored, read in place; its H.264 track (10-60 s, about 12-30 fps, up to 1280x720)
    is described by info; nothing decoded, cut, resized or padded
  audio: the same mp4 bytes as video (one tar member, read once); its AAC track (44.1
    kHz, mono or stereo) is described by info; nothing decoded, resampled, cut or
    padded
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
  pos:
  - t
  streams: []
  grid: null
  t_bins: null
```

## Prepare & verify
TODO: the expected output of `yavih bench verify llp_avvp` once the data is staged (presence per split, census).

## Gotchas
TODO: the setup and bridge mistakes that pass validation on this benchmark.
