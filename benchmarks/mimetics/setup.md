# mimetics — setup

## Obtain the data
TODO: where the media come from, the exact download commands, sizes and sha256s; who must act first
(`obtain.tier` in `benchmark.yaml`: auto, manual-download, needs-licence, needs-data-owner or site-only). Commands (`yavih bench prepare` runs the folder's numbered prepare/ scripts):

```bash
yavih bench prepare mimetics      # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
yavih bench prepare mimetics 01   # TODO
```

## Dependencies
None. TODO: list the packages of the benchmark's authors the probe env needs (exact pip lines), or keep None.

## Input contract
TODO: what the reader hands over, per stream, in one sentence each; everything else is the bridge's declared decision.

```yaml
handed_over:
  video: the Kinetics-400 mp4 exactly as stored in the Academic-Torrents _256 release
    (h264, 256 pixels high, 138-1424 wide, 4 013 portrait; 6-60 fps; train ~10 s,
    val ~12 s, 4 555 frozen clips shorter than 10 s, min 0.53 s); nothing decoded,
    cut, resized, padded or normalized
info:
  video:
  - duration_s
  - fps
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
TODO: the expected output of `yavih bench verify mimetics` once the data is staged (presence per split, census).

## Gotchas
TODO: the setup and bridge mistakes that pass validation on this benchmark.
