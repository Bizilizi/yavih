# diving48 — setup

## Obtain the data

Source: the official Diving48 page, http://www.svcl.ucsd.edu/projects/resound/dataset.html (UCSD SVCL), whose
download links are Nextcloud shares (migrated 2023-09-30). Pinned revision: `Diving48 V2 annotations (2020-10-30 update) + 2018 RGB release`.
No account, no agreement (`obtain.tier: auto`). The page states no licence: the clips are cut from online videos
of diving competitions and are distributed for research; cite the ECCV 2018 paper.

| file | share | bytes | sha256 |
|---|---|---|---|
| `Diving48_rgb.tar.gz` | `https://nextcloud.nrp-nautilus.io/s/eqKMRFHqNCrP77L/download` | 10 287 167 940 | `24e1cf350f2ec16fd1212ed57c90c148aab2c40fe61f43a7c4a68b8b0ae7c3e0` |
| `Diving48_V2_train.json` | `https://nextcloud.nrp-nautilus.io/s/Z3CfTwcnTtp5d4p/download` | 1 586 934 | `62bcff1537b24162b1c9be49f376b8f9b6fccd222734f10d4a5d545b38c00e93` |
| `Diving48_V2_test.json` | `https://nextcloud.nrp-nautilus.io/s/mfr5pfaSNp5Zx8X/download` | 207 545 | `658fe0d84ebea61414548bb69a996d668d9f249a82919e276ffc0f424d9371a3` |
| `Diving48_vocab.json` | `https://nextcloud.nrp-nautilus.io/s/oWjtCkbw9G4qdSy/download` | 3 094 | `cc4238904e3ab274ae3ab36d26a09afd41e8c7891aa1f7c773d43fecb9388696` |

Use the **V2** files (`Diving48_V2_train.json`, `Diving48_V2_test.json`), not the V1 `Diving48_train.json` /
`Diving48_test.json` the same page links first; the optical-flow archive is not used.

```bash
yavih bench prepare diving48                    # lists the prepare steps
yavih bench prepare diving48 01                 # download the four files into the media root, check sizes + sha256
CONNECTIONS=4 yavih bench prepare diving48 01   # the same, the archive as 4 parallel byte ranges (faster)
yavih bench prepare diving48 02                 # extract rgb/ (18 404 files), check the count and the tree digest
KEEP_ARCHIVE=0 yavih bench prepare diving48 02  # the same, then delete the archive
yavih bench prepare diving48 03                 # make_manifest.py --check with presence, then bench verify --census
```

* Disk: 10.3 GB for the archive, 10.3 GB once extracted (18 404 mp4, 10 313 555 478 B; `budget.media_gb: 9.64` counts
  the 16 997 files the manifest reads); about 21 GB while both exist. The archive can be deleted after step 02 (`KEEP_ARCHIVE=0`).
* Time: the download depends on the share; one connection ran at 2 to 15 MB/s here, `CONNECTIONS=4` helps.
  Extraction takes a few minutes; the tree digest of step 02 reads every file once (`VERIFY_DIGEST=0` skips it).
* Data root: `media_root` defaults to `${data.root}/diving48`. Elsewhere set
  `YAVIH_DATA__DIVING48__MEDIA_ROOT=/path/to/diving48` (or `ROOT=...` for steps 01 and 02 only), or, in
  `~/.config/yavih/config.toml`:

```toml
[data.diving48]
media_root = "/path/to/datasets/diving48"
```

Staged layout: `<media_root>/rgb/<vid_name>.mp4` (one flat directory) next to the three JSON files.

## Dependencies

None. Diving48 uses no original-author code (`external: []`). The protocol (the V2 splits, the 48-class label space,
top-1 accuracy over all 1 970 test clips) is VideoGLUE's: Yuan et al., `tensorflow/models`
`official/projects/videoglue` at commit `19f287775ae950e22a9b5fb82727e72cff2e3664` (Apache-2.0),
`configs/yaml/vmae/frozen/vc_vmae_vit3d_diving48.yaml` and `datasets/video_classification.py` (`Diving48Factory`,
48 classes). Per part of the evaluation:

* scorer: VideoGLUE's official code (rung 2; not a wheel) computes the number with
  `tf_keras.metrics.TopKCategoricalAccuracy(k=1)` in `tasks/multihead_video_classification.py`; yavih's core `acc`
  (top-1 argmax accuracy of the `clf` scorer) is the same quantity, so nothing is copied into this folder (the
  upstream metric needs TensorFlow, which a torch environment should not need for it);
* readout: the core `clf` attentive probe (no folder code). VideoGLUE's frozen-backbone head (one learnable-query
  cross-attention pooler) and its training recipe are not reproduced; `metrics.notes` lists the differences.

Prepare-time tools: `curl`, `sha256sum`, `tar` with gzip, `find`; `make_manifest.py` needs only yavih. The model env
needs no yavih decoder once the census exists (`yavih bench verify diving48 --census`); the bridge decodes with its
own stack (the template uses yavih's PyAV helpers, the `[media]` extra).

## Input contract

One stream, `video` (`kind: video`, `layout: dir`, manifest column `rel_media` = `rgb/<clip_id>.mp4` under
`media_root`). Per row the reader hands over a `RawStream` over the stored mp4, byte for byte (`path`, `bytes()`,
`open()`): h264, 640 x 480, no audio track, 1.0-16.96 s (median 4.08 s), 25 to 823 frames, at the source's own rate: 25 fps for 8 743 clips, 60 for 6 092, about 30 (29.83-30) for 2 027, 24.92 for 135; every clip's `n_frames` is its annotation's `end_frame` + 1. Nothing is decoded, cut, re-timed, resized or cropped, and no statistic is computed.

* Guaranteed `info` keys: `duration_s`, `fps`, `n_frames`, `width`, `height`, `container` (`mp4`), `bytes`, from the
  site census (`$YAVIH_HOME/media_info/diving48-<manifest_hash12>.jsonl.gz`) or a live header probe.
* Facts (`meta["facts"]`): none. The manifest's `end_frame` (the annotation's frame count) never reaches the bridge.
* Materialization: none. Every clip is one trimmed dive; the label applies to the whole file (every upstream
  `start_frame` is 0).
* The bridge decides and declares in `describe()["input"]["video"]` (recorded with every number): the temporal rule
  (which frames, how many, at which times), the resize and crop, the interpolation, the normalisation constants with
  their source, the decoder. It reports per clip `used = {"video": {"times": [pts of the frames fed]}}` in media
  seconds. The template's defaults are VideoGLUE's: 16 frames at 6.25 Hz (a stride of 4 frames at 25 fps) from a
  2.56 s window centred on the clip, short side 256, centre crop 224, pixels in [0, 1].

```yaml
handed_over:
  video: "the Diving48 mp4 exactly as stored (h264, 640 x 480, no audio track, 1.0-16.96 s (median 4.08 s), 25 to 823 frames, at the source's own rate: 25 fps for 8 743 clips, 60 for 6 092, about 30 (29.83-30) for 2 027, 24.92 for 135), byte for byte; nothing decoded, cut, re-timed, resized or cropped"
info:
  video: [duration_s, fps, n_frames, width, height, container, bytes]
facts: []
materialize: null
```

## Output contract

`[N, D]` per clip, `N` and `D` fixed for every row of both splits (`shape: fixed`); `dtype` float16 or float32.
The `clf` readout needs no positions (`requires.pos: []`, `requires.streams: []`), so `pos_mode: none` with pooled
features is fine; a bridge that keeps tokens may write `per_sample` pos in media coordinates (`t` = token time /
`duration_s` of the stored clip; `(y, x)` = fractions of the original 640 x 480 frame inside the declared crop). No
spatial targets (`target_frame: null`). What validate checks on every member: the reported times lie inside
`[0, duration_s]` (`E_TIMES_OUT_OF_MEDIA`) and equal the declared rule evaluated on the row's `info` (`E_RULE_MISMATCH`;
a `custom` rule is the opt-out, noted `N_RULE_CUSTOM`), token centres inside the crop region, canaries re-encoded in
another batch composition agree within `1e-5` rel-L2 (`E_BATCH_INVARIANCE`), pos in `[0, 1] ∪ {-1}`. Eligibility for
`clf` is the default (always eligible for a well-formed set).

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

```bash
yavih bench prepare diving48 03              # make_manifest.py --check (presence) + bench verify --census
yavih bench verify diving48 --json           # later checks
```

No derived files, no prepare-time transform. Success of step 03: `make_manifest.py` logs `--check: manifest.csv
identical ... classes.csv identical` and `missing per split 0`; `yavih bench verify diving48` reports `status
verified`, `train present 15027/15027`, `test present 1970/1970`, a decode sample with no failure and the media-info
table written; with `--json`, `labels_ok`, `manifest_sha_ok` and `curation_invariants_ok` are true and
`unknown_labels` is 0.

Then `yavih bridge check <bridge.py> --bench diving48 --out diving48.profile.json` must report
`invariance.rerun_bit_identical: true`, `batch_rel_l2` and `order_rel_l2` ≤ `1e-5`, and
`rules checked: video <rule> + spatial`.

## Gotchas

* **V1 vs V2.** The official page lists the V1 split files first. The protocol is V2 (15 027 / 1 970 clips, 6 639
  labels differ from V1); a manifest or a published number on V1 (15 943 / 2 096) is another benchmark.
* **Class 30 has no clip.** `Forward Dive NoTwis STR` is in the 48-class vocabulary but in neither V2 file. The
  probe keeps its logit (48-way, as VideoGLUE); nothing to do in the bridge, but do not "fix" `classes.csv`.
* **Frame rates differ per clip** (25 fps for 8 743 clips, 60 for 6 092, about 30 for 2 027, 24.92
  for 135). A stride counted in frames covers a different span of the dive on
  each clip; the template states times in seconds (`fps` rule). Read `info["fps"]` and `duration_s` per clip, never
  assume 25 fps, and report the pts you fed.
* **The window can miss part of the dive.** VideoGLUE's 16 x stride 4 window is 2.56 s at 25 fps; the median clip is
  4.08 s and the longest 16.96 s (60-fps clips: 2.56 s of media is only 154 frames), so a centred window may cut the takeoff or the entry. `{"rule": "uniform", "n": 16, "span":
  "full"}` covers the whole clip. Either is legal: use what your model was evaluated with and declare it.
* **Short clips.** A clip shorter than the window (2 192 clips are under 2.56 s, the shortest 1.0 s) gets its last frame repeated under `edge: "clamp"`;
  do not loop or pad to the batch maximum.
* **4:3 frames.** Every clip is 640 x 480; short side 256 + centre crop 224 keeps 66 % of the width. Declare the
  crop; do not adapt it per clip.
* **No audio track.** The reading is `v`; there is nothing to read as `a` or `av`.
* **No dataset or batch statistics.** Normalisation constants come from the model's pretraining recipe only
  (`norm.source`); per-batch normalisation fails `bridge check` batch invariance.
