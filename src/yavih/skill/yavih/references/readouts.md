# Readouts reference — what is measured, floors, controls, results

One protocol, one readout per benchmark (`benchmark.yaml[readout]`): `clf`, `seq`, `grid`, `box`.
The probe is pinned (`ProbeConfig.pinned(readout)`); any other config needs `YAVIH_TESTING=1` and
marks the result `standard: false` (never comparable). Features are read from validated shards
(`_meta.json` required); train rows = union of `splits.train` present rows, eval = `splits.eval`,
floor = `splits.floor`, `folds` → one probe per held-out fold. Validation = a seeded 10 % of TRAIN;
selection on val loss; early stopping; AdamW, cosine; fp32 probe math; deterministic algorithms
(`warn_only=True`: on CUDA the attention backward has no deterministic fp32 kernel, so probe reruns
agree to float rounding, not bit for bit; yavih captures torch's warning, names the ops in
`probe_fit.json` `determinism` and adds `W_NONDETERMINISTIC` to `result.json`; nothing to do); seed `k` → `<runs root>/<b>/<model>/<reading>/probe/<eval_hash12>/seed<k>/`. A folder's `eval.py` may supply targets,
metrics and the authors' scorer; it never defines a head, loss or training loop.

Readouts work in media coordinates and never need to know what your bridge did to the media:
`seq` places a token at absolute second `t · duration_s`; `grid` and `box` place it at its
full-frame `(y, x)`, where their targets live too. A bridge that cropped leaves the cells it did
not see empty (they get a learned null embedding); a bridge that read 8 frames and one that read 20
over the same seconds see the same targets.

<!-- yavih:bench-table readout_groups begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
Packaged benchmarks per readout: `clf` audioset20k, avsync, esc50, fsd50k, in1k, k400, ssv2, vggsounder · `seq` ave · `grid` nyuv2_depth · `box` vggss_box · `audiocaps_t2a` audiocaps_t2a · `vggsounder_v2a` vggsounder_v2a.
<!-- yavih:bench-table readout_groups end -->

| | `clf` | `seq` | `grid` | `box` |
|---|---|---|---|---|
| head | attentive probe (3 self-attn + 1 cross-attn, 16 heads, 1 learned query) → `Linear(D, C)` | dense query head, `T = 10` queries (one per scored second), memory tokens + `sincos` of their absolute second | dense query head, 14×14 queries tiling the target region, memory tokens + `sincos` of their region-relative full-frame `(y, x)`; `exp` → bilinear to 224² | dense-av: visual tokens mean-pooled per cell of a 14×14 full-frame grid (empty cell → learned null cell), 2 self-attn over cells, 2 cross-attn cells → conditioning stream, 1 box query → sigmoid `(cx, cy, w, h)` |
| needs (`output_contract.requires`) | nothing | `pos.t` of `video`; ≥ 8 of the 10 scored seconds hold a timed token | `pos.(y, x)` of `image`; ≥ 50 % of the 14×14 cells of the target region hold a token | `pos.(y, x)` of `video` covering ≥ 25 % of the full frame's 14×14 cells + tokens tagged `audio` |
| loss | `ce` (single) / `bce` (multi, multi_sets) | `ce_seq` | `l1_log` | `box_l1_giou` (full-frame fractions) |
| lr / grid | 1e-4, none | 3e-4, (1e-4, 3e-4, 1e-3) | same as seq | same as seq |
| epochs / batch / patience | 30 / 256 / 5 | 50 / 64 / 10 | 50 / 64 / 10 | 50 / 64 / 10 |
| metrics (`metrics.list`) | `acc`, `n_eval` (single); `map`, `effective_pos_classes` (multi); `auc_pr_<set>`, `auc_roc_<set>` (multi_sets); `acc_mean`, `acc_pstd`, `acc_per_fold`, `n_folds` (folds); avsync adds `acc_tol1`, `mae_seconds`, `acc_coarse5`, `n_coarse5` | `seg_acc`, `clip_acc`, `seg_acc_boundary`, `n_boundary`, `n_eval` | `d1`, `d2`, `d3`, `absrel`, `rmse`, `n_eval` | `iou_mean`, `acc@0.5`, `acc@0.3`, `center_in_gt`, `n_eval`, `n_excluded_degenerate` |
| floor (`metrics.floors`) | `acc_floor` = train majority class; `map_floor` = constant train-prevalence score; `auc_pr_<set>_floor`; vggsounder: none | `seg_acc_floor`, `seg_acc_boundary_floor` = most frequent train segment label (over all 10 seconds) | `d1_floor` = per-pixel mean train log-depth map (0.4403) | `iou_floor` = mean train box in full-frame fractions (strong) |
| control (`metrics.controls`) | avsync: `acc_floor_arm` (same probe on stationary rows), `floor_violated` | — | — | `iou_shuffled` (conditioning tokens deranged across clips), `audio_blind` (McNemar z over paired `acc@0.5` ≤ 3) |
| intervals (`metrics.ci`) | `acc: binomial` (Wilson); esc50 `acc_mean: pooled_binomial` over 2 000 rows; fsd50k `map: bootstrap_clip` | `seg_acc`, `seg_acc_boundary: bootstrap_clip` (2 000 resamples, seed 0) | `d1: bootstrap_clip` | `iou_mean: bootstrap_clip`; `acc@0.5`, `acc@0.3`, `center_in_gt: binomial` |

## Floors first

Every readout computes its floor from TRAIN targets only and scores it through the same metric
function (the authors' scorer included) on the eval rows. A model number is a result only when its
CI (or ~1 pt cross-hardware noise where there is no CI) clears the floor. Chance values (`chance`)
are weaker than floors (AVE: chance 1/29 ≈ 0.034, the floor several times higher; vggss_box: the
mean train box already has a sizeable IoU). Controls void a headline: avsync `floor_violated: true`
means the probe reads a cue that survives on stationary clips; vggss_box `audio_blind: true` means
the box head does as well with another clip's audio (visual saliency, not grounding). The report
greys such rows (`~~value~~` in md, `flag=below_floor` in csv) — never un-grey them by hand.

## Eligibility (decided by `validate`, never by the probe)

`features validate` evaluates `output_contract.requires` on the features and writes the verdict to
`_meta.json.eligibility`; the text line reads `eligible for <readout>: yes` or, e.g., `eligible for
box: no (grid coverage median 0.18 < 0.25 over the full frame) → probe will record
not_applicable`. Checks: the `pos` axes are present; every `requires.streams` name appears as a
tag; grid/box: per sampled row, the fraction of the `h × w` cells tiling the target region
(`protocol.target_region`, else the full frame) that hold ≥ 1 token centre of the visual stream —
eligible iff the MEDIAN over the sample ≥ `min_cell_coverage`; seq: per sampled row, the number of
scored seconds holding ≥ 1 token with `t ≥ 0` — eligible iff the median ≥ `t_bins`. Median and
min are recorded. An ineligible set is well-formed (exit 0) and the probe writes `status:
not_applicable` with `not_applicable.missing`: a recorded fact (a pooled bridge cannot localize),
not something to fix by changing the readout. `plan` predicts the obvious cases from the profile
(`N == 1`, `pos_mode: none`, an untagged required stream) and emits a plan-time `not_applicable`.

Coverage arithmetic to check before extracting: a Resize(256) + CenterCrop(224) bridge with 14×14
patches covers 96 of 196 full-frame cells (0.49) on 16:9 clips and 120 (0.61) on 4:3, which is why
vggss_box asks for 0.25; on NYUv2 Resize(224) + CenterCrop(224) covers the target region exactly
(1.0). On AVE, 8 temporal tokens spread over the whole clip touch 8 of the 10 seconds: eligible.

## The benchmark authors' code (`external`)

Where the authors ship a scorer, the folder's `eval.py` calls it through
`yavih.external.require(name, pip, version_pin)` — an assert-import, never a yavih dependency:

- **vggsounder**: every number is the `vggsounder` package's own (`pip install --no-deps
  vggsounder==1.0`; the exact line is in setup.md "## Dependencies"): its 30 metrics under its own
  names (`auc_pr`, `hit_rate@1`, `f1@1`, …) per reading, and `mu` in the join. No floor, no chance, no
  `n_eval` (`metrics.floors: []`). yavih has no extra for it.
- **nyuv2_depth**: Probe3D's depth metrics when importable at the pinned commit; otherwise the
  folder documents yavih's transcription (README "## In yavih" says which).
- ave (the authors' accuracy is a few lines of a research repo) and vggss_box (the official VGG-SS
  code scores heatmaps, not boxes) use yavih's own metric functions, `external: []`.

The probe checks every `external` entry BEFORE training: absent → exit 3 `E_EXTERNAL_MISSING`,
wrong version → exit 3 `E_EXTERNAL_VERSION`, both with the pip line. Nothing is silently
re-implemented. The versions used land in `result.json[external]` and the results file's
`config.external`.

## Refusals you will meet (exit 2 unless noted)

train ∩ eval clip_ids ≠ ∅ (`E_LEAKAGE`) · eval coverage < `min_eval_coverage` (`E_COVERAGE`;
`--allow-low-coverage` records `W_COVERAGE` instead) · `_meta.json` missing or `validation.ok ==
false` (`E_META_MISSING`, `E_VALIDATION` — run `features validate`) · stale
`manifest_hash` / input hash (`E_HASH_MANIFEST`, `E_HASH_INPUT` — the benchmark's input side
changed since extraction; the message names what changed; a scoring-side change — a `protocol_version` bump
after a `eval.py` or metric fix, a pin, a readout version — only re-probes) · role splits extracted under different hashes (`E_HASH_MISMATCH`) ·
a feature format this yavih does not read (`E_FORMAT`: re-extract) · `n_folds ≠ 5` for ESC-50 (`E_FOLDS`) · a `metrics.list`
name it cannot produce (`E_METRICS_INCOMPLETE`) · features missing (`E_NO_FEATURES`, exit 3) ·
scratch short (`E_SCRATCH_SPACE`, exit 3) · authors' package missing or mis-versioned (exit 3).
Large sets: above the RAM budget the probe stages shards to `YAVIH_SCRATCH` and uses the memmap
backend (`validate` warned `W_SIZE`); request `--mem` ≥ `needs.mem_gb` and scratch ≥
`needs.scratch_gb`. Deadline / SIGTERM between epochs → `probe_ckpt.pt` and exit 4; rerun resumes.
The checkpoint of the last completed epoch is kept in memory and written to disk at most every
10 min (`YAVIH_PROBE_CKPT_EVERY_S`, `0` = every epoch), at the end of every lr arm, and always on
the deadline or SIGTERM; only a hard kill (SIGKILL, OOM) loses up to that interval.

## Reading `result.json`

```json
{"schema_version": 4, "status": "ok", "reason": null, "standard": true, "eval_hash": "…",
 "benchmark": {"name": "ave", "protocol_version": 2, "readout": "seq", "readout_version": 2, "manifest_hash": "…", "protocol_hash": "…", "folder_source": {"probe": "…", "reader": null}},
 "model": "vjepa2-l", "reading": "av",
 "bridge": {"name": "vjepa-av-ave", "describe": {…}, "describe_sha256": "…", "source_sha256": "…"},
 "features": {"fingerprint": "…", "dtype": "float16", "N": 1568, "D": 1024, "pos_mode": "per_sample", "bytes": 5.2e9,
              "backend": "ram", "canary_max_rel_l2": 3.1e-6, "batch_tolerance": 1e-05,
              "rules_checked": {"video": {"temporal": "uniform", "spatial": true}}},
 "probe": {"hash": "…", "config": {…}, "seed": 0, "selected_lr": 3e-4, "epochs_run": 23, "best_epoch": 13, "stop": "patience", "n_train": 2362, "n_val": 262},
 "metrics": {"seg_acc": 0.412, "clip_acc": 0.412, "seg_acc_boundary": 0.531, "n_eval": 316, "n_boundary": 104},
 "ci95": {"seg_acc": [0.36, 0.47], "seg_acc_boundary": [0.44, 0.63]},
 "floors": {"seg_acc_floor": 0.164, "seg_acc_boundary_floor": 0.499},
 "controls": {}, "chance": {"seg_acc": 0.0345}, "external": {},
 "coverage": {"train": {"present": 2624, "nominal": 2624, "ids_sha256": "…"}, "eval": {"present": 316, "nominal": 316, "ids_sha256": "…"}, "floor": null},
 "not_applicable": null, "timing": {…}, "env": {…}, "warnings": [], "created": "2026-10-08T14:00:00Z"}
```

(Values illustrative. On AVE `clip_acc` equals `seg_acc` by construction: every clip has the same
10 scored segments, so the mean of per-clip accuracies is the pooled accuracy; the two differ only
where clips carry different numbers of scored segments. Equal values are not a bug.) Read in this order: `status` (`ok` | `not_applicable` → `reason`,
`not_applicable.missing`, `decided_by ∈ {validate, plan, bridge}` | `failed`: no DONE, never
reported) → `standard` → `eval_hash` (the comparability key: two numbers of one benchmark compare iff
it is equal; it folds `protocol_hash` and `probe_hash`, what yavih controls, and nothing of your bridge)
→ `coverage.eval.present/nominal` (and `ids_sha256`, which must agree across
the models you compare) → `floors` and `controls` → the headline with its `ci95` → `external`
(the authors' package versions that produced numbers) → `probe.stop` / `epochs_run` (`max_epochs`
with `best_epoch == epochs_run` = still improving) → `features.canary_max_rel_l2` and
`batch_tolerance` → `features.rules_checked` (per stream; `"custom"` = your declared opt-out, never a
defect) → `warnings` (`W_COVERAGE`, `W_WEIGHTS_UNFINGERPRINTED`). `model`, `reading` and `bridge` are
recorded so a reader sees what produced the number; they are never part of the comparison. The result
key is `(benchmark, eval_hash, model, reading, seed)`. Metric keys of a non-`ok` result are ABSENT, never zero. Only `result.json`
with a sibling `DONE` enters `yavih report`. `eval_logits.npy` and `eval_rows.npy` beside it feed
`report build --join vggsounder a=RUN v=RUN av=RUN`, which needs the three readings of one `model`
with one `eval_hash`, identical eval `ids_sha256` and the installed `vggsounder` package (different
weights fingerprints are the warning `W_JOIN_WEIGHTS`).

## Reading `results_<timestamp>.json`

`yavih eval run` (and every probe unit) writes, after its DONE markers,
`<results_root>/<model>/results_<timestamp>.json` (one directory per model NAME, every reading of it
inside); with `--log-samples` also
`samples_<benchmark>_<reading>_<timestamp>.jsonl` (one line per scored row: `clip_id`, `row`, `seed`,
`role`, `prediction`, `target`). The results root is `--results DIR`, else `YAVIH_RESULTS`, else
config `[results] root`, else `yavih-bench/results` in a project, else `results/` next to the runs root; `--no-results` skips it
(`"results_version": 2`). Entries are keyed `<benchmark>:<reading>` (e.g. `ave:av`; a join
`vggsounder:a+v+av`); take `benchmark`, `reading` and `eval_hash` from `configs[key]`, never by parsing
the key. Keys:

- `results.<benchmark>:<reading>`: every `metrics.list` value, `<m>_ci95`, `<m>_floor` (the floor whose
  `floor_of` names the metric), controls, `status`, `headline`, `n_seeds`, `coverage_eval`; with
  several seeds the mean, `<m>_std`, and the lowest seed's interval. A `not_applicable` entry holds
  only `status` and `reason`.
- `configs.<benchmark>:<reading>`: `benchmark`, `reading`, `eval_hash` (the comparability key),
  readout and its version, `protocol_version`, `protocol_hash`, `probe_hash`, `manifest_hash`, roles
  with counts, the metrics block, the input/output contracts, `external`, the probe config and seeds,
  the bridge (`describe()`, `describe_sha256`: provenance, per entry because one model directory holds
  several readings), `features_fingerprint`, and the `result.json` paths it was built from.
- `versions` (the `protocol_version`, lm-eval's task version), `higher_is_better`, `n_samples`,
  `config` (`model`, the bridge of a single-entry file, the probe config and seeds, `external` package
  versions), `git_hash`, `yavih_version`, `date`, `env`.

These files are derived, never read to decide anything; `report build --merge-results` merges a
model's files (newest entry per `(benchmark, reading, eval_hash)` wins; older `eval_hash`es stay under
`<benchmark>:<reading>@<eval_hash12>`). `report build` prints, per model,
`benchmark | reading | version | eval_hash | readout | metric | value | ci95 | floor` and then the
comparability columns: one column per `(benchmark, eval_hash)`, rows by `(model, reading)`, `value
(floor)` with CI, greyed rows (below floor, `floor_violated`, `audio_blind` — never a custom rule or
any other bridge decision), `population differs`, `MISSING` readings per model.

Two numbers are comparable iff benchmark, `eval_hash` and `standard` agree. When one `(benchmark,
eval_hash, model, reading)` row holds results of several bridge configurations (the same model name
re-run with another layer, crop or checkpoint: several `describe_sha256`), the report warns
`W_BRIDGE_CHANGED`, shows the newest and lists the others under `--detail` (a configuration re-run in place,
whose old output went to `probe/_stale/`, included: never shown, `moved_aside`); several feature sets of one
configuration → `W_FEATURES_DIFFER`, newest shown. Seeds are averaged within one configuration and one
feature set only. To show two configurations side by side, give them two `model` names.

## Per-benchmark reading notes

- **esc50** (`clf`, folds): features extracted once over all 2 000 rows (`features/<fold>/`), five
  probes; `acc_mean ± acc_pstd`, `acc_per_fold` is a list.
- **audioset20k** (`clf`, multi, 527 mids): macro AP over the fixed 527
  classes, zero-positive classes count 0, `effective_pos_classes` disclosed. It reads the
  published `11hu83/audioset20k` release (the original clip mp4, `video` + `audio`): train 17 726 /
  eval 16 223 (every published clip; 3 have no audio track, `info.has_track: false`: an a/av bridge
  still emits one row for them, its declared choice). `map_floor` = 0.004855; no interval. Numbers on another AS-20K download are not comparable
  (another `eval_hash`); write a/v/av readings.
- **fsd50k** (`clf`, multi, 200): `map` with a clip-level bootstrap CI; `map_floor` 0.0189.
- **vggsounder** (`clf`, multi_sets): scored by the `vggsounder` package with its routing — reading
  `a` on the sets `a` (ids `<metric>`) and `a only` (`<metric>_only`), `v` on `v` / `v only`, `av` on
  `av` (its `*_only` values are `null`); headline `auc_pr` (mAP); the report shows `auc_pr`,
  `hit_rate@1`, `f1@1` by default, the other 57 in json/csv and `--detail`. Train = 20 000
  VGGSound-train single labels (BCE one-hot). `mu` (A, V, A∩V) only via `report --join`, which stores
  the package's whole `compute_all_results` output. Write a/v/av readings, never av alone.
- **in1k** (`clf`): `acc` on 50 000 val; floor 0.001. **ssv2** (`clf`): 174 classes, eval 24 777.
  **k400** (`clf`): eval 13 492 (every `.mp4` of the Academic-Torrents `val_256`; "internal
  comparison only"). The frame rule is the bridge's declared choice.
- **avsync** (`clf` + floor role): 9 offset classes; `acc` (chance 0.111) is void when
  `floor_violated` (threshold 0.1233 at n_floor = 5 997, 3σ); `acc_coarse5` is the number least
  sensitive to the bridge's frame rate. 1 fps / 8 frames cannot resolve a 0.5 s grid: a chance
  result at low frame rates is valid, not broken.
- **ave** (`seq`, T = 10): the official AVE accuracy, all ten one-second segments, background
  included = `seg_acc`; not comparable to numbers scored on 8 segments. `seg_acc_boundary` only on the 104
  short-event test clips (±~10 pp): never rank on it below a ~10 pp gap. A bridge without `t`
  (pooled) is `not_applicable`.
- **nyuv2_depth** (`grid`): `d1` headline over the target region (Probe3D's 480 × 480 centre square
  of the 640 × 480 frame); `d1_floor` 0.4403 is strong.
- **vggss_box** (`box`): `iou_mean` in full-frame fractions against `iou_floor` AND `audio_blind`;
  `n_excluded_degenerate` disclosed; a video-only bridge is `not_applicable`; a box outside what
  your crop saw is simply hard for you.
