# epic_sounds_det — setup

The agent's first read for EPIC-SOUNDS sound detection. `benchmark.yaml` is the machine truth; `README.md` says
what the benchmark is.

## Obtain the data

`source.kind: public_url`, ungated, no account. Pinned revision:
`epic-sounds-annotations@57a922f0d352e9429f1ef8a37eee21758dd3a33c; C10-epic-sounds-detection@196e7cf2c4f17946279fb57d9ff24516b7403344; epic-kitchens-download-scripts@4f11fb2b579833f360c3c7bb917bf1e24a9787b5`.

- Labels: `EPIC_Sounds_train.csv` (6 184 177 B, sha256 `225427ab…20dd32`) and `EPIC_Sounds_validation.csv`
  (802 681 B, sha256 `fff66485…2e5a`) from https://github.com/epic-kitchens/epic-sounds-annotations, plus the two
  test files (`EPIC_Sounds_recognition_test_timestamps.csv`, `EPIC_Sounds_detection_test_videos.csv`) whose labels
  are withheld (counted as excluded, never read as rows). Full digests in `benchmark.yaml` `source.sha256`.
- Media: the EPIC-KITCHENS-100 MP4s (https://data.bris.ac.uk/data/dataset/3h91syskeag572hl6tvuovwv4d for
  EPIC-55, https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m for the extension), public HTTPS, md5
  per file in `epic-kitchens-download-scripts/data/md5.csv` (sha256 `3f9c3189…6746`). The 632 labelled
  recordings are 1.11 TB of MP4 (train 495 files / 930.9 GB, validation 137 / 183.8 GB, 40 MB to 13.9 GB each).
  Only each MP4's AAC track is kept (`audio/<video_id>.m4a`, ~6 GB in total, `budget.media_gb`): the track is
  copied packet for packet, never re-encoded, and step 02 checks the packet digest against the MP4's.
- Licence: CC BY-NC 4.0 for both the annotations and the videos (non-commercial; attribute EPIC-SOUNDS and
  EPIC-KITCHENS-100).
- Not downloadable: the labels of the 23 detection-test and 44 recognition-test recordings (Codabench challenge
  server). The authors' HDF5 of the audio is available by e-mail only and is not a supported source.

```bash
yavih bench prepare epic_sounds_det                  # lists the steps (FOLDER and YAVIH_BENCH are set for each)
export ROOT=/path/to/datasets/epic_sounds_det        # default: `yavih config get epic_sounds_det.media_root`
yavih bench prepare epic_sounds_det 01               # 6 files, 35.5 MB, sha256-checked (+ the C10 baseline json)
yavih bench prepare epic_sounds_det 02 -- --plan     # what step 02 would fetch (video id, URL, md5), nothing downloaded
yavih bench prepare epic_sounds_det 02 -- --jobs 8   # every MP4: download, md5, copy the AAC track, delete the MP4
```

Disk: `$ROOT/audio/` ~6 GB (632 files); step 02's `--tmp` (default `$ROOT/mp4_tmp`) holds up to `--jobs` MP4s at a
time (up to 14 GB each). Time: data.bris.ac.uk served 0.8 to 1.9 MB/s per connection on 2026-10-29, so the 1.11 TB
take about a week on one connection and about a day at `--jobs 8` if the server keeps that rate per connection.
Step 02 resumes partial downloads (HTTP Range) and skips staged files; rerun it after an interruption.
`--keep-mp4` keeps the MP4s under `$ROOT/mp4/` (needed for nothing in this protocol).

## Dependencies

None. `external: []`: the authors' scorer (C10 `EvaluationCode/audio_based_interaction_detection.py`, sha256
`7c2c2509…4338`) is a script, not a package, and needs `np.float` (numpy < 1.24); `eval.py` transcribes it in numpy
(README "## In yavih").

- Step 02 needs PyAV (`pip install av`, the `yavih[media]` extra) and network access; step 01 needs `curl` and
  `sha256sum`. The census (`yavih bench verify epic_sounds_det --census`) needs `yavih[media]`.
- The bridge decodes the m4a with its own stack (PyAV, torchaudio with an FFmpeg backend) or `yavih.media.decode`.
  `soundfile` alone cannot read AAC.

## Input contract

Per row (one untrimmed recording), one stream, `audio`: the file `audio/<video_id>.m4a`, the AAC track of the
EPIC-KITCHENS-100 MP4 copied packet for packet into an m4a. It is the whole recording: 10.5 s to 3 708 s (median
5 min on train, 3.7 min on validation).

- `audio` — `info`: `duration_s` (`= n_samples / sample_rate`), `sample_rate` (48 000 everywhere), `n_samples` (the
  header count, AAC priming included: the decoded count differs by a few hundred samples), `channels` (2),
  `container` (`m4a`), `bytes` (0.2 to 90 MB).
- `meta["facts"] == {}`. Nothing else reaches the bridge: no label, event, split or recording id.
- Materialization: none. The m4a is a re-containering of the stored audio track, not a transform: nothing is
  decoded, resampled, cut, windowed or padded.

The bridge decides, and declares in `describe()["input"]`, how a recording of up to an hour becomes ONE fixed
`[N, D]`: the window its encoder sees, the hop between windows (the EPIC-SOUNDS baselines use 1 s windows every
0.2 s), the resampler and front end, mono downmix, and how N is fixed across lengths: a fixed rate padded to the
longest recording (`full` with `max_s`/`pad_to_s` and padding tokens at `t = -1`) or a fixed count spread over each
recording (`custom`). The template does the first, stream-decoding the file with PyAV.

```yaml
handed_over:
  audio: "the whole untrimmed recording's AAC track (48 kHz, stereo, 10 s to 62 min), copied packet for packet from the EPIC-KITCHENS-100 MP4 into an m4a; nothing decoded, resampled, cut, windowed or padded"
info:
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

`encode` returns `(feats, pos, used)`: `feats` `[B, N, D]` with one `(N, D)` for every recording of both splits;
`pos` `[B, N, 4]` = `(stream, t, y, x)` with `t` = token centre seconds / `info.duration_s` and `y = x = -1`;
padding tokens have `t = -1`. Durations differ, so `pos_mode` is `per_sample`. `used` reports the sample range(s)
read and the padding added.

Eligibility (`yavih features validate`): tokens carry a `t` axis (`pos: [t]`) and the `audio` stream appears among
the timed tokens (`streams: [audio]`). The folder's detection scorer reads every token at its absolute second `t · duration_s` on
`readout_params.T` = 1024 bins over the recording, so any window, hop or token count is read where it lies; a
pooled token set cannot localize anything. Targets live in time only (`target_frame: null`): every labelled
segment of the recording, in seconds.

```yaml
shape: fixed
coordinates: media
target_frame: null
requires:
  pos: [t]
  streams: [audio]
  grid: null
  t_bins: null
```

## Prepare & verify

```bash
export YAVIH_DATA__EPIC_SOUNDS_DET__MEDIA_ROOT=$ROOT   # or [data.epic_sounds_det] media_root in the config
yavih bench prepare epic_sounds_det 01
yavih bench prepare epic_sounds_det 02 -- --jobs 8
yavih bench prepare epic_sounds_det 03               # re-freeze check against the staged copy (presence + --check)
yavih bench verify epic_sounds_det --census
```

There are no derived files and no materialization. Success looks like:

```
epic_sounds_det: manifest sha ok; labels ok; curation ok; ... status verified
  train    present 495/495  missing_first10=[]
  validation present 137/137  missing_first10=[]
  read sample ok=32 failed=0
  media_info present rows=632 sha256=...
```

Budget: a bridge reads every sample of ~88 h of audio; the placeholder template stages ~18 recordings/s on 15
decode workers for 10-s recordings, i.e. it is decode-bound at roughly 200x real time.

## Gotchas

- Fixed `N` over 10 s to 62 min recordings: at a 0.2 s hop the longest recording needs 18 540 tokens, so a median
  recording is mostly padding. Tokens past the end of every recording in a sample are identical, which validate
  calls `E_DEAD_ROWS`: waive it with that reason (the template does), never fill padding with noise.
- Stream-decode: one hour of 48 kHz stereo float32 is 1.4 GB, and `prepare` runs on `decode_workers` samples at
  once. `yavih.media.decode.audio(stream, ranges)` decodes the whole track and then slices it, so calling it once
  per chunk is quadratic in the recording length; iterate PyAV frames instead (the template does).
- `n_samples` is the header count; the decoder returns a few hundred samples fewer or more (AAC priming). Fit the
  decoded audio to the declared range so `used` matches what was fed.
- Segments are short (median 1.0 s, 90 % under 7 s) but some last minutes: a hop above 0.5 s or a window above a few
  seconds blurs most boundaries without failing any check.
- 285 labelled segments end after the recording's duration as EPIC-KITCHENS-100 lists it (`EPIC_100_video_info.csv`;
  by 6.7 minutes on P01_03): they are scored as upstream defines them, and no bridge choice can reach them.
- Background segments (class 13) are training targets but not scored (the official C10 rule); the paper's Table V
  keeps them (`map_avg_paper`).
