# perception_test_action_loc — setup

The agent's first read for the Perception Test temporal action localisation benchmark. `benchmark.yaml` is the
machine truth; `README.md` says what the benchmark is.

## Obtain the data

`source.kind: public_url`, `obtain.tier: auto`: everything comes from the public Google Cloud Storage bucket
`https://storage.googleapis.com/dm-perception-test/` (no account, no licence click; CC-BY 4.0), as listed by the
README of `google-deepmind/perception_test@3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.

- annotations: `zip_data/challenge_action_localisation_train_annotations.zip` (213 761 B, sha256
  `61304d4a2577c9fac449cdcffaa9060cc8d9985063c9aeae72776f605f0785ea`) and
  `zip_data/challenge_action_localisation_valid_annotations.zip` (571 135 B, sha256
  `93235a985bc9f47aa9c6d4e6ae52332424e6a4514d49ffc9de7c0dde7a1b3d85`). Only `make_manifest.py --check` reads them
  (the manifest is shipped).
- media: members of the official split zips `zip_data/{train,valid}_videos.zip` (28.4 + 75.4 GB) and
  `zip_data/{train,valid}_audios.zip` (13.2 + 35.5 GB). The bucket publishes no sha256 for them. Step 01 never
  downloads a whole zip: it reads each zip's central directory through HTTP range requests, checks that the listing
  (`name,size,crc32` per member) hashes to the sha256 pinned in `prepare/01_fetch.py`, streams out only the members
  the manifest needs (2 009 train + 5 359 valid videos and wavs) and checks each one's size and CRC-32 before moving
  it into place. Resumable; a file already in place with the listed size is kept.

Disk: 149 GB under the media root (95.4 GB mp4 + 53.6 GB wav), 14 736 files + 2 zips; about 1.5 h at 30 MB/s.
The root is `${datasets.perception_test}`, shared with `perception_test_sound_loc` (same videos): staging one
stages both.

```bash
yavih bench prepare perception_test_action_loc              # lists the steps (FOLDER and YAVIH_BENCH are set for each)
yavih config get perception_test_action_loc.media_root      # where step 01 writes (set [datasets] perception_test to move it)
yavih bench prepare perception_test_action_loc 01           # annotations + every needed mp4 and wav (149 GB), resumable
yavih bench prepare perception_test_action_loc 01 -- --first 32   # only the first 32 rows of each split (a capped trial run)
yavih bench prepare perception_test_action_loc 01 -- --verify     # re-check the CRC-32 of every file in place
yavih bench prepare perception_test_action_loc 02           # yavih bench verify --census (needs yavih[media])
```

Step 01 options: `--workers N` (parallel members, default 8), `--only videos|audios`, `--root DIR`, `--from-zips DIR`
(extract from split zips you downloaded yourself; the same listing pins apply). Exit 2 = a pin or CRC mismatch
(nothing moved into place: the upstream changed, report it), exit 3 = a network or disk error (rerun to resume).

## Dependencies

None: `external: []`. The authors ship no installable scorer: the challenge metric lives in the baseline fork
github.com/ptchallenge-workshop/actionformer_release_PT (`libs/utils/metrics.py`, pandas + joblib, no package), so
the folder's own scorer (`eval.py` and `detection.py`) is a transcription of it (README "## In yavih" lists the one difference).

- Step 01 needs only the Python standard library and yavih core; step 02 (`bench verify --census`) needs
  `yavih[media]` (PyAV, soundfile).
- The bridge decodes with its own stack or `yavih.media.decode` (`yavih[media]`). The wavs are 32-bit integer PCM
  (`pcm_s32le`): soundfile, torchaudio and PyAV read them; some minimal wav readers (Python's `wave`) do not.

## Input contract

Two streams over two stored files per row, both read as stored (`layout: dir` under the media root):

- `video` — `videos/<video_id>.mp4`: H.264, 1920x1080 or 1280x720 (also portrait), about 25-30 fps (the frame rate
  differs per video, e.g. 29.6, 29.95, 30.01), 6.7-36 s (median about 23 s), with its own AAC 48 kHz stereo track
  (the `audio` stream of an `av` reading is the wav, not this track). `info`: `duration_s`, `fps`, `width`, `height`,
  `container`, `bytes`.
- `audio` — `audios/<video_id>.wav`: the official audio, 32-bit PCM at 44.1 kHz or 48 kHz, mono or stereo (on the
  first 64 rows: 25 at 44.1 kHz mono, 8 at 48 kHz mono, 31 at 48 kHz stereo), about as long as the video (it may
  differ by up to ~0.1 s). `info`: `duration_s`, `sample_rate`, `n_samples`, `channels`, `container`,
  `bytes`.

No label facts (`facts: []`): the bridge never sees the events. The event times are in seconds of the video
timeline; the probe reads each token at `t · duration_s` of its own stream, so the bridge only has to report
positions in media coordinates. Frames, windows, resolution, the audio front end and the token count are the
bridge's declared decisions; the count must be fixed while clips last 6.7 to 36 s (spread a fixed number of tokens
over each clip, as the template does).

```yaml
handed_over:
  video: the Perception Test mp4 exactly as stored (videos/<video_id>.mp4, H.264 with its own audio track, 6.7-36 s,
    up to 1920x1080, about 30 fps); nothing decoded, cut or resized
  audio: the official wav exactly as stored (audios/<video_id>.wav, 32-bit PCM, 44.1 or 48 kHz, mono or stereo); nothing
    decoded, resampled, cut or padded
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

`[N, D]` fixed for every row of every split; `pos = (stream, t, y, x)` in media coordinates (`t` = token centre /
that stream's `duration_s`, `-1` = no axis). The folder's detection scorer needs `pos.t` and tokens tagged `video`: `features
validate` records `eligible: false` (and the probe `not_applicable`) for a set without them, which is what an `a`
reading gets. Audio tokens of an `av` reading join the memory with their own `t`. Time resolution matters: actions
last 0.3 s (5th percentile) to 7.2 s (95th), median 1.8 s, and the probe predicts on 128 bins over the clip (~0.18 s
each on a 23 s clip); a bridge with one token per few seconds or a time-pooled token set is valid but cannot localise
short actions. `(y, x)` are not read: per-patch video tokens are fine (they keep their `t`), so are per-frame means.

```yaml
shape: fixed
coordinates: media
target_frame: null
requires:
  pos:
  - t
  streams:
  - video
  grid: null
  t_bins: null
```

## Prepare & verify

`yavih bench prepare perception_test_action_loc 02` runs `yavih bench verify perception_test_action_loc --census`.
Once step 01 has finished it reports presence 2 009 / 2 009 (train) and 5 359 / 5 359 (valid), `labels_ok`,
`manifest_sha_ok`, a successful read sample and the media-info table written (14 736 files probed; a few minutes on
local disk). After `01 -- --first N` it reports N rows per split present: enough for `yavih plan build --sample N`,
which reads exactly the first N manifest rows of each split, never for a full run (the probe refuses eval coverage
below 0.95).

## Gotchas

- The wav sample rate and channel count vary per clip (44.1 or 48 kHz, mono or stereo): resample and mix down to
  the model's input, never assume 48 kHz.
- The `audio` stream is the wav, not the mp4's AAC track: decoding audio from the `video` stream's file passes
  validation but declares the wrong stream (and its AAC priming shifts samples).
- Every clip has a different length: a bridge that cuts or pads to a fixed number of seconds silently drops the end
  of long clips (events there become unreachable) or adds padding tokens (give tokens centred in padding `t = -1`).
- A token's `t` must be its real time: compute it from the decoded pts / sample ranges, not from bin indices that
  assume a fixed duration.
- 1080p decoding dominates extraction (about 5 clips/s for 32 frames on CPU): give the extractor decode workers.
- Three eval classes never occur in train (7 eval events); they count in the class mean with AP 0 for every probe.
- Even perfect features stop near `tmap` 0.96 under the pinned decoding (overlapping actions of one class, README).
- A model pretrained on Perception Test videos (or fine-tuned on its tasks) has seen these clips; audit before
  claiming a number.
