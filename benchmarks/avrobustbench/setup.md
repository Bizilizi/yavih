# avrobustbench — setup

## Obtain the data
TODO: where the media come from, the exact download commands, sizes and sha256s; who must act first
(`obtain.tier` in `benchmark.yaml`: auto, manual-download, needs-licence, needs-data-owner or site-only). Commands (`yavih bench prepare` runs the folder's numbered prepare/ scripts):

```bash
yavih bench prepare avrobustbench      # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
yavih bench prepare avrobustbench 01   # TODO
```

## Dependencies
None. TODO: list the packages of the benchmark's authors the probe env needs (exact pip lines), or keep None.

## Input contract
TODO: what the reader hands over, per stream, in one sentence each; everything else is the bridge's declared decision.

```yaml
handed_over:
  video: 'the video track of the row''s materialized mp4 (prepare/03_materialize.py):
    the VGGSound source video over [1 + δ/2, 9 + δ/2] s, re-encoded frame-accurately
    (libx264, native resolution and pixel format, source frame timing kept, times
    relative to the window start, ≈ 8.0 s); nothing else decoded, cut, resized or
    padded'
  audio: 'the audio track of the same mp4: the VGGSound 16 kHz mono wav over [1 −
    δ/2, 9 − δ/2] s, cut sample-exactly (128 000 samples, 8.0 s) and stored losslessly
    as ALAC (decodes bit-identical to the source PCM); audio and video inside the
    file are offset by exactly δ, which is the label and is not handed over'
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
materialize:
  kind: av_offset_mux
  offset_column: offset_seconds
  source_columns:
    video: rel_video
    audio: rel_audio
  pivot: 1.0
  length: 8.0
  video: 0.5
  audio: -0.5
  tool:
    name: ffmpeg
    version_pin: 4.4.2-0ubuntu0.22.04.1
    video_args:
    - -nostdin
    - -hide_banner
    - -loglevel
    - error
    - -y
    - -copyts
    - -noautorotate
    - -i
    - '{video}'
    - -i
    - '{audio}'
    - -map
    - 0:v:0
    - -filter:v
    - trim=start={v0}:end={v1},setpts=PTS-{v0}/TB
    - -vsync
    - passthrough
    - -enc_time_base:v
    - '-1'
    - -c:v
    - libx264
    - -preset
    - medium
    - -crf
    - '18'
    - -pix_fmt
    - yuv420p
    - -threads
    - '4'
    - -flags:v
    - +bitexact
    audio_args:
    - -map
    - 1:a:0
    - -filter:a
    - atrim=start_sample={a0}:end_sample={a1},asetpts=PTS-STARTPTS
    - -c:a
    - alac
    - -flags:a
    - +bitexact
    - -map_metadata
    - '-1'
    - -map_chapters
    - '-1'
    - -fflags
    - +bitexact
    - -movflags
    - +faststart
    pins:
      libx264:
        value: x264 - core 163 r3060 5db6aa6
        probe:
        - '{tool}'
        - -nostdin
        - -hide_banner
        - -loglevel
        - error
        - -f
        - lavfi
        - -i
        - color=c=black:s=64x64:d=0.04
        - -frames:v
        - '1'
        - -c:v
        - libx264
        - -f
        - h264
        - '-'
        match: x264 - core \d+ r\d+ [0-9a-f]+
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
TODO: the expected output of `yavih bench verify avrobustbench` once the data is staged (presence per split, census).

## Gotchas
TODO: the setup and bridge mistakes that pass validation on this benchmark.
