# esc50 — setup

## Obtain the data

Source: GitHub `karolpiczak/ESC-50` (https://github.com/karolpiczak/ESC-50), pinned commit
`33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6` (2023-10-13). Also as
`https://github.com/karoldvl/ESC-50/archive/master.zip` (~600 MB) and DOI 10.7910/DVN/YDEPUT. No account
or agreement needed.

Licence: **CC BY-NC 3.0** for the dataset; the ESC-10 subset (`esc10 == True`, 400 clips) is CC BY 3.0;
per-clip Freesound attributions in the upstream `LICENSE`. Non-commercial use only.

```bash
yavih bench prepare esc50   # lists the prepare steps
OUT=/path/to/datasets/esc50 yavih bench prepare esc50 01
#   git clone --depth 1 https://github.com/karolpiczak/ESC-50.git "$OUT", checks out the pinned commit, then
#   sha256 meta/esc50.csv == ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246
#   sha256 LICENSE        == f6aec3a3283a2efdcbb1feff334adad7ab17c097bdec0cbb4eebdb9e16886074
YAVIH_DATA__ESC50__MEDIA_ROOT=/path/to/datasets/esc50 PY=python yavih bench prepare esc50 02
#   2000 wavs, every header (220500 frames, 44100 Hz, 1 ch, PCM_16), audio tree digest
#   52754b9099c1c48a9fec379bd7dd75dbc6d78387faecc2a9318fc0c43332d398
#   (= sha256 of `cd audio && export LC_ALL=C && sha256sum *.wav`)
```

* Size: shallow clone pack 645 520 453 B; the clone is 1.53 GB / ~2 040 files on disk (0.88 GB of wav
  without `.git`, `budget.media_gb: 0.88`); every wav is exactly 441 044 B. Nothing to extract.
* The `.zip` route works too (rename `ESC-50-master/` to the media root); `02_verify.sh` is the arbiter
  either way. Cloning takes a few minutes; `02_verify.sh` reads 882 MB (a few minutes on NFS).
* Data root: `media_root` defaults to `${data.root}/esc50`. Elsewhere set
  `YAVIH_DATA__ESC50__MEDIA_ROOT=/path/to/esc50` or, in `~/.config/yavih/config.toml`:

```toml
[data.esc50]
media_root = "/path/to/datasets/esc50"
```

## Dependencies

None. ESC-50 uses no original-author code (`external: []`); accuracy is yavih's. Prepare-time tools:
`git`, `sha256sum`, and a python with `soundfile` for `02_verify.sh`'s header check (`pip install
soundfile`, or the yavih `[media]` extra). The model env needs no yavih decoder once the census exists
(`yavih bench verify esc50 --census`); the bridge decodes with its own stack (the template uses
`soundfile`).

## Input contract

One stream, `audio` (`kind: audio`, `layout: dir`, manifest column `rel_path` = `audio/<clip_id>.wav`
under `media_root`). Per row the reader hands over a `RawStream` over the stored wav, byte for byte
(`path`, `bytes()`, `open()`): RIFF PCM_16, 44.1 kHz, mono, 220 500 samples (5.000 s). Nothing is decoded,
resampled, trimmed, padded or normalized, and no statistic is computed.

* Guaranteed `info` keys: `duration_s` (5.0), `sample_rate` (44100), `n_samples` (220500), `channels` (1),
  `container` (`wav`), `bytes` (441044), from the site census (`$YAVIH_HOME/media_info/esc50-<manifest_hash12>.jsonl.gz`)
  or a live header probe. Every one of the 2 000 stored clips has exactly these values
  (`curation.short_clips: 0`).
* Facts (`meta["facts"]`): none. `protocol.clip_seconds: 5.0` is a fact of the corpus, never used to cut
  or pad.
* Materialization: none.
* Folds: each manifest `split` value `"1"…"5"` is read as its own split (400 rows); features must not
  depend on the fold or the batch.
* The bridge decides and declares in `describe()["input"]["audio"]` (recorded with every number): the temporal
  rule (`full` = the whole clip, `full` with `pad_to_s` = zero padding to the model length, `span` = a
  cut, `custom` = anything else such as repeat padding), the resampler and target rate, the front end
  (fbank/mel parameters, channels), the normalization constants with their source (the model's own
  pretraining recipe), and the decoder. It reports per clip `used = {"audio": {"samples": [[lo, hi]],
  "pad": [left, right]}}` in native-rate samples.

```yaml
handed_over:
  audio: "the ESC-50 wav exactly as stored (RIFF PCM_16, 44.1 kHz, mono, 220 500 samples = 5.000 s); nothing decoded, resampled, trimmed or padded"
info:
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

`[N, D]` per clip, `N` and `D` fixed for every row of every fold (`shape: fixed`); `dtype` float16 or
float32. Readout `clf` needs no positions (`requires.pos: []`, `requires.streams: []`), so `pos_mode:
none` is fine; a bridge that keeps time × frequency tokens may write `constant` pos in media coordinates
(`t` = token centre / `duration_s` of the stored clip, `-1` for a token centred in padding the bridge
added; `y = x = -1` for audio), legal as `constant` because every clip is 5.0 s. No spatial targets
(`target_frame: null`). What validate checks on every member: the reported audio ranges lie inside
`[0, n_samples]` (`E_TIMES_OUT_OF_MEDIA`), they and the padding equal the declared rule evaluated on the
row's `info` within one sample (`E_RULE_MISMATCH`; a `custom` rule is a legitimate opt-out, noted as
`N_RULE_CUSTOM`, and keeps the in-media check), canaries re-encoded in another batch composition agree
within `1e-5` rel-L2 (`E_BATCH_INVARIANCE`), pos in `[0, 1] ∪ {-1}`. Eligibility for `clf` is the default
(`N_ELIGIBILITY` always eligible for a well-formed set).

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
yavih bench verify esc50 --census --json     # writes the media-info table once per site
yavih bench verify esc50 --json
```

No derived files, no prepare-time transform. Success: `status verified`, every fold `present 400/400`,
`decode sample ok=30 failed=0`, `media_fingerprint e413d9039c94…`; with `--json`, `labels_ok`,
`manifest_sha_ok` and `curation_invariants_ok` are true and `unknown_labels` is 0.

Then `yavih bridge check <bridge.py> --bench esc50 --out esc50.profile.json` must report
`invariance.rerun_bit_identical: true`, `batch_rel_l2` and `order_rel_l2` ≤ `1e-5`, and
`rules_checked: {audio: {temporal: "full"}}` (or your declared rule).

## Gotchas

* **Padding the 5 s clip to a longer model input.** Zero-padding to 8–10 s makes ≥ 3/8 of the input
  digital silence and skews per-clip normalization; zero vs repeat fill can differ by ~10 points. The
  reader hands over exactly the stored 5 s and never pads; **if your model needs 10 s / 1024 frames, the
  fill is YOUR choice** (zero = `full` with `pad_to_s`; repeat = `custom`; or size the input to the clip,
  as AudioMAE does with 512 frames). Declare it in `describe()["input"]["audio"]["temporal"]`. Normalize
  as the model was pretrained: with its fixed dataset constants, or per clip over the padded buffer only
  if that is the model's own pretraining recipe, then declared in `norm` with its `source`. Drop
  padding-only tokens when the model allows it.
* **Resampling is the bridge's job.** The stored rate is 44.1 kHz; a model trained at 16 kHz fed 44.1 kHz
  passes every check and loses most of its accuracy. Declare the resampler in `front_end`.
* **Do not re-window** unless that is the model's documented ESC recipe (then declare a `span` rule): no
  centred 4 s, no first-N-s crop by default; the stored clip is the whole recording.
* **Decode the whole file.** Use `info["n_samples"]` / `duration_s`, never a hard-coded 220 500: the
  facts come from the stored file, and `used` must match them within one sample.
* **No dataset or batch statistics.** Per-batch or per-corpus normalization (or BatchNorm in train mode)
  fails `bridge check` batch invariance (`invariance.batch_rel_l2`); normalization constants come from the
  model's pretraining recipe only (`norm.source`).
* **Vocabulary.** The logit index is `classes.csv` (sorted names). The upstream `target` digit in the
  filename / `meta/esc50.csv` is a different order (dog = 0, rooster = 1, …); never map through it (the
  bridge never sees labels anyway).
* **Folds are the official ones.** Four Freesound sources (9 clips) appear in two folds under different
  classes (`131943` clock_tick / clock_alarm, `134049` hen / rooster, `209698` clock_alarm / clock_tick,
  `234879` hen / rooster); kept as upstream ships them, since re-splitting would break comparability with
  every published ESC-50 number.
