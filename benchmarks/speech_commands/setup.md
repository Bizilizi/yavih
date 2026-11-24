# speech_commands — setup

## Obtain the data

Source: Speech Commands v0.02 (Warden 2018), two public archives on Google's download server, no account or
agreement. The task is HEAR 2021's `speech_commands-v0.0.2-5h`, built from exactly these two files as
hear-preprocess builds it. Pinned revision: `speech_commands_v0.02 + speech_commands_test_set_v0.02 (HEAR task speech_commands-v0.0.2, hear-preprocess v2021.1.1 @ 610d864b2c0f7064eaeced98ee8992d3914d58f1)`.

| file | bytes | sha256 | md5 (as HEAR pins it) |
|---|---|---|---|
| `speech_commands_v0.02.tar.gz` | 2 428 923 189 | `af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58` | `6b74f3901214cb2c2934e98196829835` |
| `speech_commands_test_set_v0.02.tar.gz` | 112 563 277 | `cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df` | `854c580ee90bff80c516491c84544e32` |

Licence: **CC BY 4.0** (the `LICENSE` file in both archives).

```bash
yavih bench prepare speech_commands        # lists the steps
ROOT=/path/to/datasets/speech_commands yavih bench prepare speech_commands 01
#   downloads both archives into $ROOT/_downloads (DOWNLOADS=... to reuse a copy), checks sha256 and md5,
#   extracts them to $ROOT/train/ (105 829 word wavs in 35 folders, _background_noise_/, validation_list.txt,
#   testing_list.txt) and $ROOT/test/ (4 890 wavs in 12 folders)
ROOT=/path/to/datasets/speech_commands yavih bench prepare speech_commands 02
#   cuts HEAR's 789 one-second _silence_ clips from the six background-noise recordings into $ROOT/hear_silence/
#   and checks them: hear_silence/SHA256SUMS sha256 == 9e3cef9ceb50fcba95cf909e4e4215aa0e22a05f8b991ae61785e5ea212c8533
ROOT=/path/to/datasets/speech_commands yavih bench prepare speech_commands 03
#   make_manifest.py --check against the staged media, then yavih bench verify speech_commands --census
```

* Disk: 2.5 GB of archives (delete `_downloads/` after step 01 if space is short) and 3.4 GB extracted. The
  rows the benchmark reads (train 16 000, valid 2 000, test 4 890) are 0.72 GB (`budget.media_gb`); the rest of
  the train archive is in the manifest as unused splits and stays staged for `make_manifest.py --check`.
* Time: the download takes minutes; extracting ~110 000 small files takes minutes on a local disk and up to an
  hour on a network file system. Step 02 takes about a minute, step 03 a few minutes (header reads).
* Data root: `media_root` defaults to `${data.root}/speech_commands`. Elsewhere set
  `YAVIH_DATA__SPEECH_COMMANDS__MEDIA_ROOT=/path/to/speech_commands` or, in `~/.config/yavih/config.toml`:

```toml
[data.speech_commands]
media_root = "/path/to/datasets/speech_commands"
```

## Dependencies

None to install: `external: []`. Every part of the evaluation that comes from HEAR is copied into the folder
(grounding rung 2: HEAR's code is not a usable wheel; `heareval` on PyPI pins `numpy==1.19.2`, `numba==0.48`
and `tensorflow>=2.0`, and `heareval.score` imports `sed_eval` and `dcase_util` at module scope):

* scorer: hear-eval-kit (https://github.com/hearbenchmark/hear-eval-kit) @
  `855964977238e89dfc76394aa11c37010edb6f20` (v2021.1.3), Apache-2.0: `heareval/score.py` lines 57-155
  (`validate_score_return_type`, `ScoreFunction`, `Top1Accuracy`) verbatim in `hear_score.py`; the glue `eval.py`
  `hear_inputs` hands it what `heareval/predictions/task_predictions.py` hands it (softmax predictions, one-hot
  targets).
* task definition: hear-preprocess (https://github.com/hearbenchmark/hear-preprocess) @
  `610d864b2c0f7064eaeced98ee8992d3914d58f1` (v2021.1.1), Apache-2.0: `hearpreprocess/speech_commands.py`
  (splits, labels, silence cutting) and `hearpreprocess/pipeline.py` (`SubsampleSplit`, the 5 h draw),
  transcribed in `make_manifest.py` and `prepare/02_hear_silence.py` (cited by line in each).
* readout: yavih's core `clf` probe (no HEAR code; HEAR's own readout is a grid-searched MLP, see the
  `metrics.notes`).

Prepare-time tools: `curl`, `sha256sum`, `md5sum`, `tar`, and a python with `numpy` and `soundfile` built on
libsndfile >= 1.1 (the yavih `[media]` extra has them) for steps 02 and 03. libsndfile 1.0.x rescales the
samples HEAR's cutting loop writes by 32767/32768, and step 02 then fails its digest check. The bridge decodes
with its own stack (the template uses `soundfile`).

## Input contract

One stream, `audio` (`kind: audio`, `layout: dir`, manifest column `rel_path` under `media_root`:
`train/<word>/<speaker>_nohash_<n>.wav`, `test/<label>/<file>.wav` or `hear_silence/<noise>-<start>.wav`). Per
row the reader hands over a `RawStream` over the stored wav, byte for byte (`path`, `bytes()`, `open()`): RIFF
PCM_16, 16 kHz, mono, at most 16 000 samples. Nothing is decoded, resampled, trimmed, padded or normalized.

* Guaranteed `info` keys: `duration_s`, `sample_rate` (16000), `n_samples` (at most 16000), `channels` (1),
  `container` (`wav`), `bytes`, from the site census or a live header probe. 1 804 of the 22 890 read clips
  are shorter than 1.0 s (`curation.short_clips`); none is longer.
* The 789 `_silence_` clips of HEAR's train and valid are not files of the release: HEAR cuts them from the
  `_background_noise_` recordings (a 16 000-sample window every 8 000 samples); step 02 cuts them with HEAR's
  own loop as sample-exact copies of the source PCM. The 408 test `_silence_` clips are files of the test archive.
* Facts (`meta["facts"]`): none. `protocol.clip_seconds: 1.0` is HEAR's `sample_duration`, never used by the
  reader to cut or pad.
* Materialization: none (the silence clips are pinned by sha256 through `data.derived`, not re-encoded).
* The bridge decides and declares in `describe()["input"]["audio"]`: the temporal rule, the resampler and
  target rate, the front end, the normalization constants with their source, and the decoder. HEAR feeds
  every model the clip trimmed or zero-padded at its end to exactly 1.0 s; the template's default rule
  `{"rule": "full", "max_s": 1.0, "pad_to_s": 1.0}` is that. It reports per clip
  `used = {"audio": {"samples": [[lo, hi]], "pad": [left, right]}}` in native-rate samples.

```yaml
handed_over:
  audio: >-
    the stored 16 kHz mono PCM_16 wav exactly as stored: a release clip of at most 1.000 s (some are shorter), or one
    of HEAR's 1.000 s silence clips that prepare/02 cuts sample-exactly from a background recording; nothing
    decoded, resampled, trimmed or padded
info:
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

`[N, D]` per clip, `N` and `D` fixed for every row of every split (`shape: fixed`); `dtype` float16 or float32.
Readout `clf` needs no positions (`requires.pos: []`, `requires.streams: []`), so `pos_mode: none` is the
default. No spatial targets (`target_frame: null`). What validate checks on every member: the reported audio
ranges lie inside `[0, n_samples]` (`E_TIMES_OUT_OF_MEDIA`), they and the padding equal the declared rule evaluated
on the row's `info` within one sample (`E_RULE_MISMATCH`; a `custom` rule is noted as `N_RULE_CUSTOM`), canaries
re-encoded in another batch composition agree within `1e-5` rel-L2 (`E_BATCH_INVARIANCE`). Eligibility for `clf`
is the default.

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
yavih bench prepare speech_commands 03          # make_manifest.py --check, then bench verify --census
yavih bench verify speech_commands --json
```

Success: `make_manifest.py --check` reports `manifest.csv identical` and `classes.csv identical` (every pin of
the archives' contents matches, every listed wav present, 22 890 read-split headers 16 kHz mono PCM_16, 1 804
shorter than 1.0 s); `bench verify` reports `train 16000/16000`, `valid 2000/2000`, `test 4890/4890` present, a
clean decode sample and `derived_sha_ok: true` (the silence `SHA256SUMS`), with `labels_ok`, `manifest_sha_ok`
and `curation_invariants_ok` true.

Then `yavih bridge check <bridge.py> --bench speech_commands --out speech_commands.profile.json` must report
`invariance.rerun_bit_identical: true`, `batch_rel_l2` and `order_rel_l2` ≤ `1e-5`, and
`rules_checked: {audio: {temporal: "full"}}` (or your declared rule).

## Gotchas

* **Short clips.** One read clip in thirteen is shorter than 1.0 s. Pad every clip to one fixed model length
  (HEAR: zeros at the end up to 1.0 s, the template's default), never to the batch maximum: batch-dependent
  padding fails batch invariance, and a token count that follows the clip length breaks the fixed `N`.
* **Padding much further.** A model built for 10 s input fed 1 s of speech and 9 s of zeros sees mostly digital
  silence; size the input to the clip when the model allows it, and declare the fill you choose.
* **Silence clips come from step 02.** Cutting them another way (another hop, float rounding, a resampled copy)
  changes HEAR's `_silence_` class; step 02 refuses any digest but the pinned one.
* **The splits are HEAR's, not the release's lists.** The probe trains on HEAR's 5 h draw (`train` + `valid`)
  and is scored on the separate test archive, not on `testing_list.txt`. Do not re-split, and do not train on
  the unused `train_rest`, `valid_rest` or `testing_list` rows.
* **Vocabulary.** The logit index is `classes.csv`, HEAR's sorted label vocabulary (`_silence_`, `_unknown_`,
  then the ten words); 25 of the 35 recorded words are folded into `_unknown_`.
