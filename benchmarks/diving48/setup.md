# diving48 — setup

## Obtain the data
TODO: where the media come from, the exact download commands, sizes and sha256s; who must act first
(`obtain.tier` in `benchmark.yaml`: auto, manual-download, needs-licence, needs-data-owner or site-only). Commands (`yavih bench prepare` runs the folder's numbered prepare/ scripts):

```bash
yavih bench prepare diving48      # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
yavih bench prepare diving48 01   # TODO
```

## Dependencies
None. TODO: list the packages of the benchmark's authors the probe env needs (exact pip lines), or keep None.

## Input contract
TODO: what the reader hands over, per stream, in one sentence each; everything else is the bridge's declared decision.

```yaml
handed_over:
  video: the SSv2 webm exactly as stored (vp9, 12 fps nominal with leading offsets
    and holes in ~8 % of clips, height 240, width 240-483, no audio track, 0.83-14.17
    s, median 4.0 s), byte for byte; nothing decoded, cut, re-timed, resized or cropped
info:
  video:
  - duration_s
  - fps
  - n_frames
  - width
  - height
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
TODO: the expected output of `yavih bench verify diving48` once the data is staged (presence per split, census).

## Gotchas
TODO: the setup and bridge mistakes that pass validation on this benchmark.
