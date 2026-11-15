# nsynth — setup

## Obtain the data

Source: the NSynth dataset (Google Magenta, https://magenta.tensorflow.org/datasets/nsynth), "json/wav"
release. Pinned revision: `NSynth json/wav release of 2017-04 (archive md5 fde6665a / 87e94a00 / 5e6f8719); task hear-preprocess v2021.1.1 nsynth_pitch v2.2.3 mode 5h`.
Three public archives on Google Cloud Storage, no account or agreement. Licence: **CC BY 4.0**.

The benchmark is HEAR's `nsynth_pitch-v2.2.3-5h` task: the frozen manifest lists the 5 000 notes HEAR's
preprocessing draws (4 000 train, 500 valid, 500 test). The archives must be downloaded whole (gzip is not
seekable and the train notes are spread through the 23.8 GB train archive); only the 5 000 listed wavs and the three
`examples.json` files are extracted.

```bash
yavih bench prepare nsynth          # lists the steps
yavih bench prepare nsynth 01       # download the three archives to <media_root>/archives/, md5 + sha256 checked
#   nsynth-train.jsonwav.tar.gz  23 815 298 079 B  md5 fde6665a93865503ba598b9fac388660
#                                sha256 15c5100fc40c3a262f9b42707c15ffed30b8fec58c0d9b88d4136f6cb1fddefe
#   nsynth-valid.jsonwav.tar.gz   1 068 767 009 B  md5 87e94a00a19b6dbc99cf6d4c0c0cae87
#                                sha256 00dea2645fbe0069258567da30807a90825e0bab54c077d6481f253096c4e2a0
#   nsynth-test.jsonwav.tar.gz      349 501 546 B  md5 5e6f8719bf7e16ad0a00d518b78af77d
#                                sha256 0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c
yavih bench prepare nsynth 02       # extract examples.json + the manifest's 5 000 wavs into <media_root>/nsynth-<split>/
PY=python yavih bench prepare nsynth 03   # 5 000 wav headers (64 000 frames, 16 kHz, mono, PCM_16) + audio digest
yavih bench prepare nsynth 04       # make_manifest.py --check from the staged examples.json, then bench verify
```

* Sizes: 25.2 GB of archives (the download; keep or delete them after step 02), then 0.64 GB of staged wavs
  (5 000 × 128 044 B) and the three `examples.json` (train 270 MB). Budget about 26 GB of disk while staging.
* Time: the downloads ran at about 40 MB/s here (about 11 minutes); step 02 reads every archive once
  (about 14 minutes here, almost all of it the train archive).
* Digests of the extracted metadata: `nsynth-train/examples.json` a8182e084aa338d764f64f6a283bd1778d358e913a7074294110e9ec601e0087,
  `nsynth-valid/examples.json` 050e0bf55d1a87eee2abbc3d273c7f525404c0a51ac378561b905f2da31a7286,
  `nsynth-test/examples.json` 74df2dd960c156cd5e8757d8afb1fe30e6a233aa99043a29037e7b54cde6908f. The staged
  wav set's digest (sha256 of `cd <media_root> && export LC_ALL=C && sha256sum nsynth-*/audio/*.wav`) is
  `d4e890c1dc76f7dbecc13ebfe7f7274cabcde8578c4c15b892d6a60a0794703a`.
* Data root: `media_root` defaults to `${data.root}/nsynth`. Elsewhere set
  `YAVIH_DATA__NSYNTH__MEDIA_ROOT=/path/to/nsynth`, pass `ROOT=/path/to/nsynth` to a step, or, in
  `~/.config/yavih/config.toml`:

```toml
[data.nsynth]
media_root = "/path/to/datasets/nsynth"
```

## Dependencies

None to install: `external: []`. The evaluation is HEAR's, carried as copies in the folder (yavih's grounding
rule: only an authors' wheel that installs in a current environment is imported):

* **Scorer** (rung 2, the suite's official code, copied): `pitch_acc` and `chroma_acc` are `Top1Accuracy` and
  `ChromaAccuracy` of the HEAR evaluation kit, https://github.com/hearbenchmark/hear-eval-kit at commit
  `855964977238e89dfc76394aa11c37010edb6f20` (v2021.1.3, Apache-2.0), file `heareval/score.py`, copied into
  `eval.py`. The PyPI wheel `heareval==2021.1.3` is not imported: it pins `numpy==1.19.2` and `numba==0.48` and
  imports `torch`, `sed_eval` and `dcase_util` at module scope.
* **Task definition** (rung 2, copied): the pitch range, the 5h subsample and the label vocabulary are HEAR's
  preprocessing, https://github.com/hearbenchmark/hear-preprocess at commit
  `610d864b2c0f7064eaeced98ee8992d3914d58f1` (v2021.1.1, Apache-2.0), files `hearpreprocess/nsynth_pitch.py`,
  `hearpreprocess/pipeline.py` (`SubsampleSplit.run`) and `hearpreprocess/util/luigi.py` (`str2int`), copied
  into `make_manifest.py` (pandas replaced by numpy; `tests/test_folder.py` checks the equivalence).
* **Readout**: yavih's `clf` probe, not HEAR's. HEAR trains an MLP (1 or 2 hidden layers of 1 024, a grid over
  four learning rates and two initialisations, selected on `valid`) on one scene embedding per clip; yavih trains
  its pinned attentive probe on the bridge's tokens and selects it on a 10 % carve of `train` (`metrics.notes`).

Prepare-time tools: `curl`, `md5sum`, `sha256sum`, GNU `tar` with gzip, and a python with `numpy` (the
manifest) and `soundfile` (step 03's header check; the yavih `[media]` extra has it). The model environment
needs no yavih decoder once the census exists (`yavih bench verify nsynth --census`).

## Input contract

One stream, `audio` (`kind: audio`, `layout: dir`, manifest column `rel_path` =
`nsynth-<split>/audio/<note_str>.wav` under `media_root`). Per row the reader hands over a `RawStream` over the
stored NSynth wav, byte for byte (`path`, `bytes()`, `open()`): RIFF PCM_16, 16 kHz, mono, 64 000 samples
(4.000 s). Nothing is decoded, resampled, trimmed, padded or normalized, and no statistic is computed. HEAR's own
preprocessing of these files (convert to mono PCM_16 wav, trim or pad to 4.0 s) leaves every NSynth note
unchanged, so the stored file is exactly what HEAR hands a model before it resamples.

* Guaranteed `info` keys: `duration_s` (4.0), `sample_rate` (16000), `n_samples` (64000), `channels` (1),
  `container` (`wav`), `bytes` (128044), from the site census or a live header probe. Every one of the 5 000
  frozen notes has exactly these values (`curation.short_clips: 0`).
* Facts (`meta["facts"]`): none. `protocol.clip_seconds: 4.0` is a fact of the corpus, never used to cut or pad.
* Materialization: none.
* Splits: `train` (4 000) trains the probe and `test` (500) is scored; `valid` (500) is `unused` (it shares all
  53 instruments with `test`; HEAR only selects its MLP on it), staged by step 02 but never extracted or read.
  Features must not depend on the split or the batch.
* The bridge decides and declares in `describe()["input"]["audio"]` (recorded with every number): the temporal
  rule (`full` = the whole clip, `full` with `pad_to_s` = zero padding, `span` = a cut, `custom` = anything
  else), the resampler and target rate (HEAR's reference: ffmpeg `aresample` with `resampler=soxr`), the front end,
  the normalization constants with their source, and the decoder. It reports per clip `used = {"audio":
  {"samples": [[lo, hi]], "pad": [left, right]}}` in native-rate samples.

```yaml
handed_over:
  audio: "the NSynth wav exactly as stored (RIFF PCM_16, 16 kHz, mono, 64 000 samples = 4.000 s); nothing decoded, resampled, trimmed or padded"
info:
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

`[N, D]` per clip, `N` and `D` fixed for every row of every split (`shape: fixed`); `dtype` float16 or float32.
Readout `clf` needs no positions (`requires.pos: []`, `requires.streams: []`), so `pos_mode: none` is fine; a
bridge that keeps time × frequency tokens may write `constant` pos in media coordinates (`t` = token centre /
`duration_s`, `-1` for a token centred in padding the bridge added; `y = x = -1` for audio), legal as
`constant` because every clip is 4.0 s. A single pooled vector per clip (HEAR's scene embedding, `N = 1`) is
accepted too. No spatial targets (`target_frame: null`). What validate checks on every member: the reported audio
ranges lie inside `[0, n_samples]` (`E_TIMES_OUT_OF_MEDIA`), they and the padding equal the declared rule
evaluated on the row's `info` within one sample (`E_RULE_MISMATCH`; `custom` is noted as `N_RULE_CUSTOM`),
canaries re-encoded in another batch composition agree within `1e-5` rel-L2 (`E_BATCH_INVARIANCE`), pos in
`[0, 1] ∪ {-1}`. Eligibility for `clf` is the default.

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
yavih bench verify nsynth --census --json     # writes the media-info table once per site
yavih bench verify nsynth --json
```

No derived files, no prepare-time transform. Success: `status verified`, `train present 4000/4000`, `test
present 500/500`, `decode sample ok=30 failed=0`; with `--json`, `labels_ok`,
`manifest_sha_ok` and `curation_invariants_ok` are true and `unknown_labels` is 0.

Then `yavih bridge check <bridge.py> --bench nsynth --out nsynth.profile.json` must report
`invariance.rerun_bit_identical: true`, `batch_rel_l2` and `order_rel_l2` ≤ `1e-5`, and
`rules_checked: {audio: {temporal: "full"}}` (or your declared rule).

## Gotchas

* **The label is the pitch: never move it.** Resampling at a wrong ratio, reading the 16 kHz file as another
  rate, pitch-shift or speed augmentation left on in `prepare()`: every one passes validation and moves the
  spectrum by semitones. Resample 16 kHz → the model rate with a declared resampler.
* **Resampling is the bridge's job.** The stored rate is 16 kHz; a model trained at 32 or 48 kHz fed 16 kHz samples
  as if they were its own rate hears every note up to an octave and a half too low.
* **Decaying notes.** Many notes fade to near silence before 4 s; a model that normalizes per clip (when that is
  its recipe) amplifies the tail noise. Zero-padding to a 10 s model input adds silence; declare it (`full` with
  `pad_to_s`) or size the input to 4 s.
* **The note name spells the label.** `clip_id` and the file name are `<instrument>-<pitch>-<velocity>`
  (`guitar_acoustic_010-060-100` is pitch 60): a bridge never reads them (`bridge check` scrambles ids).
* **Vocabulary.** The logit index is `classes.csv`: names `21` … `108`, index = pitch − 21. `chroma_acc` is HEAR's
  class-index-mod-12 rule, which is the pitch class only because the vocabulary is contiguous.
* **The test split is HEAR's 500-note draw, not NSynth's 4 096-note test set.** Do not stage or score the full test
  set: its numbers are another benchmark (HEAR's 50h mode keeps all 4 060 in-range test notes).
