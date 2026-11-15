# crema_d — setup

## Obtain the data

Source: GitHub `CheyneyComputerScience/CREMA-D` (https://github.com/CheyneyComputerScience/CREMA-D), the CREMA-D
authors' repository, media in Git LFS, pinned commit `1658cd342dff90010aa843eaeebd53610a08b1dc` (2025-03-21; the
7 438 wavs used here are unchanged since commit `f06ffd7`, 2017). No account or agreement needed, and no `git-lfs`:
the step clones the LFS pointers and downloads each wav from GitHub's media endpoint.

Licence: **Open Database License (ODbL) 1.0** for the database and **Database Contents License (DbCL) 1.0** for its
contents (upstream `LICENSE.txt`): use, share and adapt with attribution; a shared adapted database stays under ODbL.

```bash
yavih bench prepare crema_d       # lists the prepare steps
yavih bench prepare crema_d 01    # root = $(yavih config get crema_d.media_root); ROOT=/path overrides
#   sparse clone (AudioWAV/ pointers, processedResults/, LICENSE.txt) at the pinned commit, then
#   sha256 processedResults/summaryTable.csv == fb0e382700d32c0d9adc0c6815292215a36bcf912f2657048862c91e3bd2b317
#   sha256 LICENSE.txt                       == e76747a4efa4b338914bb3f930b5d3d3c3ac15b886bbabf897f6a19731f987f8
#   TFDS's dataset_info.json of crema_d 1.0.0 -> tfds/crema_d-1.0.0-dataset_info.json
#   sha256                                   == 5e748e0ecf01210ebcc1e2a4bcb80a59af47b89bfdee6521d18efa4ff0a7b6b0
#   TFDS's manual checksums of crema_d (size + sha256 per wav) -> tfds/crema_d-manual_checksums.txt
#   sha256                                   == 66b5d959dcee997321baf9080c80022a6937279419446db3a9d4827d839b0abb
#   7 438 wavs from https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/<commit>/AudioWAV/<name>.wav,
#   each checked against the sha256 and size of its LFS pointer (JOBS=4 parallel downloads by default)
yavih bench prepare crema_d 02    # verify: 7 438 wavs, 16 kHz mono PCM_16 headers, TFDS's sha256 per wav, the tree digest
yavih bench prepare crema_d 03    # re-freeze (byte-identical) and yavih bench verify
```

* Size: the 7 438 wavs are TODO bytes (`budget.media_gb`); the sparse clone adds a few MB. Downloads are one
  HTTP request per wav: about an hour at four parallel downloads on a slow file system, minutes on a local disk.
  Re-running step 01 skips every wav already present with the right sha256.
* Which wavs: every `FileName` of `processedResults/summaryTable.csv` (7 442) except the four that TensorFlow
  Datasets' `crema_d` 1.0.0 skips as absent from GitHub (`1040_ITH_SAD_XX`, `1006_TIE_NEU_XX`, `1013_WSI_DIS_XX`,
  `1017_IWW_FEA_XX`), so exactly HEAR 2021's clips. The MP3 and video files of the repository are not needed.
* Data root: `media_root` defaults to `${data.root}/crema_d`. Elsewhere set
  `YAVIH_DATA__CREMA_D__MEDIA_ROOT=/path/to/crema_d` or, in `~/.config/yavih/config.toml`:

```toml
[data.crema_d]
media_root = "/path/to/datasets/crema_d"
```

## Dependencies

None to install (`external: []`). The evaluation is HEAR 2021's, copied into the folder (fleet rung 2: the
suite's official code, not usable as a wheel, since `heareval` pins `numpy==1.19.2` and TensorFlow and its
`score.py` imports torch, `sed_eval` and `dcase_util`):

* scorer: hear-eval-kit (https://github.com/hearbenchmark/hear-eval-kit @ `855964977238e89dfc76394aa11c37010edb6f20`,
  Apache-2.0) `heareval/score.py` (`Top1Accuracy`) and `heareval/predictions/task_predictions.py`
  (`aggregate_test_results`, `data_splits_from_folds`) -> `hear_score.py`, called by `eval.py`;
* folds: hear-preprocess (https://github.com/hearbenchmark/hear-preprocess @ `610d864b2c0f7064eaeced98ee8992d3914d58f1`,
  v2021.1.1, Apache-2.0) `hearpreprocess/pipeline.py` (`split_k_folds`) and `hearpreprocess/tfds_pipeline.py` (the
  file stem), with TensorFlow Datasets (https://github.com/tensorflow/datasets @ v4.4.0 =
  `76f8591def26afaca16340b06d057553582f6163`, Apache-2.0) `audio/crema_d.py`, `core/hashing.py`, `core/shuffle.py`,
  `core/tfrecords_writer.py`, `core/tfrecords_reader.py`, and python-slugify v6.1.2 (MIT) -> `hear_split.py`, called by
  `make_manifest.py` (prepare time only);
* readout: yavih's core `clf` probe (no HEAR code: HEAR trains its own MLP with a grid search).

Prepare-time tools: `git`, `curl`, `sha256sum`, and a python with `numpy` and `soundfile` for steps 02 and 03
(the yavih `[media]` extra has it). The bridge decodes with its own stack (the template uses `soundfile`).

## Input contract

One stream, `audio` (`kind: audio`, `layout: dir`, manifest column `rel_path` = `AudioWAV/<clip_id>.wav` under
`media_root`). Per row the reader hands over a `RawStream` over the stored wav, byte for byte (`path`, `bytes()`,
`open()`): RIFF PCM_16, 16 kHz, mono, 1.3 to 5.0 s long (it varies per clip). Nothing is decoded, resampled,
trimmed, padded or normalized, and no statistic is computed.

* Guaranteed `info` keys: `duration_s`, `sample_rate` (16000), `n_samples`, `channels` (1), `container` (`wav`),
  `bytes`, from the site census (`$YAVIH_HOME/media_info/crema_d-<manifest_hash12>.jsonl.gz`) or a live header
  probe. `duration_s` and `n_samples` differ from clip to clip.
* Facts (`meta["facts"]`): none. `protocol.clip_seconds` is null: the clips have no common length.
* Materialization: none. HEAR's preprocessing trims or zero-pads every clip to 5.0 s and resamples it to the
  model's rate; here the bridge does both, and declares them (the template's default is HEAR's 5.0 s).
* Folds: each manifest `split` value `fold00` … `fold04` is read as its own split (1 487 or 1 488 rows); features
  must not depend on the fold or the batch.
* The bridge decides and declares in `describe()["input"]["audio"]` (recorded with every number): the temporal
  rule (`full` with `max_s` and `pad_to_s` = HEAR's trim and zero-pad, `full` alone = the stored clip, `span` = a
  cut, `custom` = anything else such as repeat padding), the resampler and target rate, the front end, the
  normalization constants with their source, and the decoder. It reports per clip
  `used = {"audio": {"samples": [[lo, hi]], "pad": [left, right]}}` in native-rate samples.

```yaml
handed_over:
  audio: "the CREMA-D wav exactly as stored (RIFF PCM_16, 16 kHz, mono, 1.3 to 5.0 s); nothing decoded, resampled, trimmed or padded"
info:
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

`[N, D]` per clip, `N` and `D` fixed for every row of every fold (`shape: fixed`); `dtype` float16 or float32.
Readout `clf` needs no positions (`requires.pos: []`, `requires.streams: []`), so `pos_mode: none` is fine; a
bridge that keeps time × frequency tokens may write `per_sample` pos in media coordinates (`t` = token centre /
`duration_s` of the stored clip, `-1` for a token centred in padding the bridge added; `y = x = -1` for audio),
per sample because the clips differ in length. No spatial targets (`target_frame: null`). What validate checks on
every member: the reported audio ranges lie inside `[0, n_samples]` (`E_TIMES_OUT_OF_MEDIA`), they and the padding
equal the declared rule evaluated on the row's `info` within one sample (`E_RULE_MISMATCH`; a `custom` rule is a
legitimate opt-out, noted as `N_RULE_CUSTOM`), canaries re-encoded in another batch composition agree within
`1e-5` rel-L2 (`E_BATCH_INVARIANCE`), pos in `[0, 1] ∪ {-1}`. Eligibility for `clf` is the default.

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
yavih bench prepare crema_d 02               # 7438 wavs, headers, digest
yavih bench prepare crema_d 03               # make_manifest.py (asserts TFDS's byte counts) + bench verify
yavih bench verify crema_d --census --json   # writes the media-info table once per site
```

No derived files, no prepare-time transform. Success: step 03 prints `manifest.csv.gz unchanged`, the TFDS
serialized sizes equal to `dataset_info.json` (train 1 223 332 734 B in 16 shards, validation 181 972 381 B in 2,
test 365 486 261 B in 4), and `bench verify` reports `status verified`, every fold present (1 488 / 1 488 / 1 488 /
1 487 / 1 487), `decode sample ok=30 failed=0`; with `--json`, `labels_ok`, `manifest_sha_ok` and
`curation_invariants_ok` are true.

Then `yavih bridge check <bridge.py> --bench crema_d --out crema_d.profile.json` must report
`invariance.rerun_bit_identical: true`, `batch_rel_l2` and `order_rel_l2` ≤ `1e-5`, and
`rules_checked: {audio: {temporal: "full"}}` (or your declared rule).

## Gotchas

* **The clips differ in length.** 1.3 to 5.0 s, median about 2.5 s. A fixed-length model input pads most clips:
  pad every clip to the same length (HEAR: 5.0 s, the template's default), never to the longest clip of the
  batch (that fails batch invariance) and never with per-batch statistics.
* **Already 16 kHz.** Resample only when the model's rate differs, per clip, and declare the resampler; HEAR
  resampled with SoX (`soxr`).
* **The folds are HEAR's, not speaker-independent.** HEAR draws its five folds over files, so every actor appears
  in several folds and a model may partly recognise the speaker. Do not re-split by actor, and do not use the
  `tfds_split` column (TFDS's own speaker-grouped split) as a fold: both give other numbers.
* **Vocabulary.** The logit index is `classes.csv` (the six emotion codes sorted: ANG, DIS, FEA, HAP, NEU, SAD, as
  HEAR's `labelvocabulary.csv`); TFDS numbers them in another order (NEU = 0). The bridge never sees labels anyway.
* **Never cut a centred window** unless that is the model's documented recipe (then declare a `span` rule): the
  speech starts near the beginning of most clips and many end before 2 s.
