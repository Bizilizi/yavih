# Readouts

A benchmark is scored by exactly one standardized readout, named by `benchmark.yaml[readout]`: `clf`
(classification), `seq` (per-segment classification), `grid` (dense regression, depth) or `box` (one box per
clip). The readout trains a pinned probe head on the extracted features and scores it with the metric functions
the benchmark lists. The heads, the trainer and `ProbeConfig` below are the only ones: a benchmark folder's
`probe.py` may supply targets, metrics, an external scorer and a cross-reading join,
but never a head, a loss or a training loop.

What the probe writes is described in [results](results.md); the hashes that key it (`probe_hash`,
`eval_hash`) in [hashing](hashing.md); the eligibility verdict it reads in [validation](validation.md); the
`metrics` block of `benchmark.yaml` in [benchmark-folder](benchmark-folder.md#benchmarkyaml); the feature files it consumes in
[features](features.md).

## Entry point

Module `yavih.readouts` (the `[probe]` extra: torch). CLI: `yavih probe run RUN_DIR`.

```python
def probe(run_dir, *, seeds=(0,), readout=None, cfg=None, deadline=None, scratch=None,
          allow_low_coverage=False, loader_workers=None, log_samples=False,
          results_root=None, write_results=True, bench=None, unit=None, profile=None,
          validate_first=False, device=None, out=None, waive=None) -> list[ProbeResult]
```

| argument | CLI | meaning |
|---|---|---|
| `seeds` | `--seeds 0,1,2` | one probe (one `result.json`) per seed |
| `readout` | | must equal `benchmark.yaml[readout]` (else `E_READOUT`, exit 1) |
| `cfg` | | `None` = `ProbeConfig.pinned(readout, bench)`; anything else needs `YAVIH_TESTING=1` |
| `deadline` | `--deadline slurm\|<epoch_s>` | checked between epochs; see [execution](execution.md) |
| `scratch` | `--scratch DIR` | staging root for feature sets above the RAM budget |
| `allow_low_coverage` | `--allow-low-coverage` | record `W_COVERAGE` instead of refusing `E_COVERAGE` |
| `loader_workers` | `--loader-workers N` | `None` (default) or `>= 1`: a loader thread gathers the next two training batches; `0` = inline (same numbers; see Training mechanics) |
| `log_samples` | `--log-samples` | also write a samples file to the results store |
| `results_root` / `write_results` | `--results DIR` / `--no-results` | the results-store root ([results](results.md#results-store)) |
| `bench` | `--bench REF\|DIR` | a benchmark not found by the name in `_meta.json` |
| `validate_first` | on by default, `--no-validate` skips | run `features validate` on every role split first |
| `waive` | `--waive CODE=reason` | validation waivers passed to the validate prologue |
| `device` | `--device cpu\|cuda\|cuda:N` | overrides cuda/cpu auto-detection |
| `out` | `--out DIR` | probe a read-only feature set (below) |

The benchmark is read from `features/*/_meta.json` (every split must name the same one). `--out DIR` mirrors
the source's `features/<split>/` shards, sidecars and `NOT_APPLICABLE.json` into `DIR/features/<split>/` as
symlinks (refreshed on every run), copies an existing `_meta.json`, and makes `DIR` the run directory for
everything the probe writes; the source tree is never written, and the results-store root is still judged on
the source run directory.

## Readout registry

Readouts are not hard-coded. `yavih.readouts.registry` holds one `Readout` descriptor per readout; a readout
module exports `READOUT = Readout(...)` and a driver `run_<name>(ctx)`. The built-in modules are
`yavih.readouts.{clf, seq, grid, box}`; a package adds one through the entry-point group `yavih.readouts`.
`benchmark.yaml` validation, `protocol_hash` (the readout `version`), the pinned `ProbeConfig`, the floor
outputs, the dense-readout checks of `features validate` and `plan`, and the metric table all ask the registry
(`registry.get(name)`, `registry.names()`). Importing a readout module never imports torch; the driver imports
its torch code lazily.

| field | role |
|---|---|
| `name`, `version` | the readout name; `version` enters `protocol_hash` (scoring side) |
| `label_kinds` | the `manifest.label.kind` values it scores |
| `protocol_keys`, `param_keys` | the per-benchmark `protocol` keys its driver consumes; the exact `readout_params` key set |
| `targets`, `full_frame` | labels come from `benchmark.yaml[targets]`; `output_contract.target_frame` must be `full_frame` |
| `requires_t_bins`, `requires_grid` | `output_contract.requires` keys: required (`True`), forbidden (`False`), optional (`None`) |
| `validate_spec` | extra cross-checks of `readout_params` against `requires` and `protocol` |
| `dense` | features need media positions (`features validate`, `plan`) |
| `train`, `head`, `loss`, `task_sizes`, `geometry` | the pinned training block, head descriptor and loss; sizes taken from a benchmark |
| `head_families`, `losses` | values a `ProbeConfig` of this readout may carry |
| `floor_outputs` | the label-only constant output the floor metrics score |
| `check_outputs`, `input_keys`, `synthetic` | optional output-contract check, the keys it reads (they enter `input_hash`), a test fixture builder |

Adding a readout: [adding-a-readout](adding-a-readout.md).

## Metric registry

`yavih.metrics.REGISTRY` maps a metric name to a `MetricSpec`. The registry is importable without torch or
scikit-learn: every `fn` is a lazy reference resolved on first call.

```python
@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable                       # fn(outputs, targets, protocol, split_meta) -> float | list[float]
    readouts: frozenset[str]           # readouts that may list it; "*" = every readout
    protocol_keys: frozenset[str] = frozenset()   # protocol keys fn reads; Benchmark.load requires them
    shape: str = "scalar"              # "scalar" | "list"
    kind: str = "metric"               # "metric" | "floor" | "control" | "disclosure"
    higher_is_better: bool | None = True          # None for disclosures and controls
    floor_of: tuple[str, ...] = ()     # floors only: the metrics this floor is the floor of
    external: str | None = None        # the external[].name whose code computes it (folder metrics)
    per_clip: Callable | None = None   # folder metrics: per-clip values for intervals
    ci: frozenset[str] = frozenset()   # folder metrics: the interval kinds per_clip supports
```

`benchmark.yaml[metrics.list/floors/controls]` names are resolved at `Benchmark.load`: an unknown name, a name
whose `readouts` excludes the benchmark's readout, a name of the wrong `kind` for its list (`list` takes
`metric` and `disclosure`, `floors` takes `floor`, `controls` takes `control`), or a missing `protocol_keys`
entry is a load error. The readout driver calls every listed `fn` with the head's raw outputs; it never
hard-codes a metric name.

| name | readouts | kind | protocol keys | definition |
|---|---|---|---|---|
| `acc` | clf | metric | | exact-argmax accuracy |
| `n_eval` | every readout | disclosure | | number of scored eval rows |
| `n_train` | clf | disclosure | | number of train rows |
| `acc_floor` | clf | floor of `acc`, `acc_mean` | | accuracy of the train majority class on eval |
| `map` | clf | metric | | macro AP over the fixed `C`; zero-positive classes count `AP = 0` |
| `effective_pos_classes` | clf | disclosure | | classes with at least one eval positive |
| `map_floor` | clf | floor of `map` | | macro AP of the constant train-prevalence score |
| `auc_pr_<set>`, `auc_roc_<set>` | clf | metric | | per `label.sets` name, expanded at load |
| `auc_pr_<set>_floor` | clf | floor of `auc_pr_<set>` | | the train-prevalence score on that set |
| `acc_mean`, `acc_pstd` | clf | metric | | mean and population std of the held-out fold accuracies |
| `acc_per_fold` | clf | metric, `shape: list` | | held-out accuracy per fold, in `splits.folds` order |
| `n_folds` | clf | disclosure | | number of folds |
| `seg_acc`, `clip_acc`, `seg_acc_boundary` | seq | metric | `num_segments`, `segments_scored`, `segment_offset`, `background_index` | [seq](#seq) |
| `n_boundary` | seq | disclosure | same | rows whose event is shorter than the clip |
| `seg_acc_floor`, `seg_acc_boundary_floor` | seq | floor | same | most frequent train segment label |
| `d1`, `d2`, `d3`, `absrel`, `rmse` | grid | metric (`absrel`, `rmse` lower is better) | `max_depth`, `target_size` | [grid](#grid) |
| `d1_floor` | grid | floor of `d1` | same | per-pixel mean train log-depth map |
| `iou_mean`, `acc@0.5`, `acc@0.3`, `center_in_gt` | box | metric | | [box](#box) |
| `n_excluded_degenerate` | box | disclosure | | degenerate eval boxes excluded |
| `iou_floor` | box | floor of `iou_mean` | | the mean train box |
| `iou_shuffled`, `audio_blind` | box | control | | the shuffled-conditioning control |

### Folder metrics

A benchmark folder's `probe.py` may export `METRICS: dict[str, MetricSpec]`, which overlays the registry for
that benchmark only. A folder metric may replace a registry entry of the same name only if it keeps its
`readouts`, `shape` and `kind`. `PROTOCOL_KEYS: frozenset[str]` names the protocol keys the folder consumes
that no metric declares. `Benchmark.load` requires every key a resolved metric or `PROTOCOL_KEYS` declares and
refuses a per-benchmark protocol key that nothing consumes (a metric, the readout's `protocol_keys`,
`PROTOCOL_KEYS` or `input_contract.facts`). Examples: avsync's offset metrics (`acc_tol1`, `mae_seconds`,
`acc_coarse5`, `n_coarse5`) and floor-arm controls (`acc_floor_arm`, `floor_violated`); vggsounder's
`auc_pr_<set>`, `auc_roc_<set>` and floors, computed by the official package; nyuv2_depth's depth metrics,
computed by Probe3D's code. The other `probe.py` hooks (`targets`, `score`, `join`) are listed
in [benchmark-folder](benchmark-folder.md#probepy-hooks).

### External packages

A spec with `external` set is computed by the folder's `score` hook, which reaches the benchmark authors'
package only through `yavih.external.require(name, pip, version_pin)`, an assert-import:

- package absent: `E_EXTERNAL_MISSING`, exit 3 (not ready); the message carries the `pip` line of
  `benchmark.yaml[external]`;
- installed version different from `version_pin`: `E_EXTERNAL_VERSION`, exit 3;
- importable without distribution metadata: accepted only as a git checkout when the `pip` line pins a commit
  (`git+<url>@<sha>`). The checkout must be at that commit (read from `.git`), its declared version (if
  `setup.py` / `pyproject.toml` states one) must equal `version_pin`, and `git status` (when it runs) must show
  no modified tracked file in the package. The recorded version is `"<pin>+<sha8> (git checkout)"`.
  Distribution metadata, when present, decides alone.

The probe checks every `external` entry before staging or training, so a missing package costs nothing. Such
packages are never yavih dependencies, and nothing is silently re-implemented when one is missing. The versions
actually used go to `result.json[external]` and the results file's `config.external`.

## Common rules

### What the probe reads

- Features: one `FeatureReader` per role split, consumed only through `rows(idx)` / `matrix()` and, for `seq`,
  `durations(idx, stream)`; never gathered whole into Python lists. Rows are upcast to fp32 per mini-batch.
- `_meta.json` of every role split ([features](features.md)): it must exist (`E_META_MISSING`), report
  `validation.ok` (`E_VALIDATION`), be fresh against its sidecars (`E_META_STALE`) and have `format_version` 3
  or 4 (`E_FORMAT`).
- Labels: from `Benchmark.manifest`, from `Benchmark.targets(split)` when `benchmark.yaml[targets]` is set
  (grid), or from the folder's `targets` hook.
- Coordinates: `pos` is in media coordinates; spatial targets are in full-frame fractions. A readout never needs
  to know what the bridge cropped or how many frames it read ([coordinates](#coordinates)).

### Roles

| role | rows |
|---|---|
| train | the union of the present rows of `splits.train` |
| eval | the present rows of `splits.eval` |
| floor | the present rows of `splits.floor` (avsync): scored by the trained probe, feeds controls |
| folds | for each fold value `f` of `splits.folds`: train = every other fold, eval = `f` |

A clip id in both train and eval is `E_LEAKAGE`. Leakage between the eval source and the train draw is a
freeze-time fact (`manifest.upstream.curation.leakage_audit`), not a probe-time check.

### Validation carve and selection

The validation rows are carved from TRAIN only: a seeded permutation (`torch.Generator().manual_seed(seed)`)
of the train rows, `n_val = max(1, round(val_fraction · n_train))` capped at `n_train − 1`. Every lr of
`lr_grid` (or the single `lr` when the grid is empty) trains its own arm on the same split and init; each arm
keeps its best state by val loss and stops early after `patience` epochs without improvement; the arm with
the lowest val loss is selected and its best state restored. `probe.selected_lr`, `epochs_run`,
`best_epoch` and `stop` record the outcome.

### Coverage

For every eval split, `present / nominal` (from `_meta.json.coverage`) must reach
`splits.min_eval_coverage`, else `E_COVERAGE`; `--allow-low-coverage` records `W_COVERAGE` in the result
instead, and the report then marks the row `low_coverage`. `result.coverage` records `present`, `nominal` and
`ids_sha256` (sha256 over the sorted present clip ids) per role, plus a `folds` block under `splits.folds`.
Under folds, the number of fold splits with features must equal `len(splits.folds)`, else `E_FOLDS`.

### Eligibility and not_applicable

The probe never re-derives eligibility. It reads `_meta.json.eligibility` (every split must agree on
`eligible`, else `E_ELIGIBILITY_MISMATCH`); `eligible: false` writes a `not_applicable` result with
`not_applicable.missing` from that block and exits 0 ([validation](validation.md)). Two other sources produce
`not_applicable`:

- the bridge declared it (`NOT_APPLICABLE.json` in a split): that split contributes its record only, a
  `_meta.json` beside it is never read, and every `NOT_APPLICABLE.json` and every other split's `_meta.json`
  must name one `(model, reading, describe_sha256)`, else `E_DESCRIBE_MISMATCH` ("re-extract every split
  with the current bridge");
- the plan predicted it (`expected_status: not_applicable` on the unit): no features exist; the result is
  built from the planned profile with `features: null` and `coverage: null`. A unit planned under another
  `protocol_hash` is `E_HASH_PROTOCOL`, under another `probe_hash` or `eval_hash` `E_HASH_PROBE`.

### Floors, controls, chance

- Floors are computed from TRAIN targets only. Each readout's `floor_outputs` builds a constant output in the
  head's raw-output shape (train majority class, train prevalence, most frequent segment label, mean log-depth
  map, mean box), and the floor metric scores it on the eval rows through the same metric function, an
  external scorer included. `floor_of` names the metric each floor belongs to (the report's floor column).
- Controls are scored on the trained head's outputs and can void a headline: avsync `floor_violated`, vggss_box
  `audio_blind`. The report greys such rows.
- Chance values are copied from `metrics.chance` into `result.chance`; they are weaker than floors.

### Intervals

Intervals are not metrics: `metrics.ci[name]` makes the driver emit `result.ci95[name] = [lo, hi]`.

| kind | definition | admissible names |
|---|---|---|
| `binomial` | Wilson 95 % on a per-row 0/1 quantity | `acc`, `acc@0.5`, `acc@0.3`, `center_in_gt` |
| `bootstrap_clip` | seeded clip-level percentile bootstrap (2.5 / 97.5), `metrics.bootstrap` `{samples, seed}` (required with this kind) | the binomial names, `seg_acc`, `seg_acc_boundary`, `clip_acc`, `iou_mean`, `d1`, `d2`, `d3`, `absrel`, `rmse`, `map` |
| `pooled_binomial` | Wilson on the pooled correct count over all held-out fold rows (ESC-50: 2 000) | the binomial names, `acc_mean`, `acc_pstd`, `acc_per_fold` |

The resampling unit is the clip: a per-clip quantity (for `seg_acc` the clip's segment cells, pooled exactly as
the point estimate pools them) or, for `map`, the macro AP recomputed on every clip resample (fixed `C`,
zero-positive classes 0). A folder metric admits the kinds its `MetricSpec.ci` names, computed from its own
`per_clip`. `Benchmark.load` refuses any other pairing (`registry.CI_ADMISSIBLE`).

### Aggregation

`metrics.aggregation` is required with `splits.folds` and forbidden without it: `{over: folds, report: [mean,
pstd, per_fold]}` (esc50), optional `ci`. One probe is trained per held-out fold; outputs are pooled in fold
order for `acc_mean`, `acc_pstd`, `acc_per_fold`, `n_folds` and `pooled_binomial`. Across seeds, aggregation
happens in the report ([results](results.md#yavih-report)), never in the probe.

### Refusals

| code | exit | cause |
|---|---|---|
| `E_LEAKAGE` | 2 | train ∩ eval clip ids non-empty |
| `E_COVERAGE` | 2 | eval coverage below `min_eval_coverage` without `--allow-low-coverage` |
| `E_META_MISSING`, `E_VALIDATION`, `E_META_STALE` | 2 | `_meta.json` absent or without an eligibility block, `validation.ok == false`, stale against the sidecars |
| `E_FORMAT` | 2 | `_meta.json` `format_version` above the probe's or below 3; a feature set naming no model/reading |
| `E_HASH_MISMATCH` | 2 | role splits disagree on `benchmark.name`, `manifest_hash` or `input_hash` |
| `E_DESCRIBE_MISMATCH` | 2 | role splits disagree on `model`, `reading`, `bridge.describe_sha256` or feature `dtype`/`N`/`D`/`pos_mode` |
| `E_HASH_MANIFEST`, `E_HASH_INPUT` | 2 | the features' manifest or input hash differs from the loaded benchmark's; the message names the writer versions |
| `E_FOLDS` | 2 | fold splits present ≠ `len(splits.folds)` |
| `E_SPLIT_MISSING` | 2 | a role split has no features directory or no present rows |
| `E_METRICS_INCOMPLETE` | 2 | a listed metric, floor, control or interval was not produced |
| `E_POS_MISSING`, `E_DURATIONS` | 2 | a dense readout got no `pos`; a timed token's stream has no `duration_s` |
| `E_NO_FEATURES` | 3 | no `features/` directory or no split directory |
| `E_SCRATCH_SPACE` | 3 | scratch free space below 1.1 × the train bytes |
| `E_EXTERNAL_MISSING`, `E_EXTERNAL_VERSION` | 3 | authors' package absent or mis-versioned |
| `E_RESULTS_ROOT`, `E_NONSTANDARD_CONFIG`, `E_READOUT`, `E_PROBE_PY` | 1 | no results-store root; non-pinned config outside testing; readout mismatch; `probe.py` failed to import |

Features are never checked against `protocol_hash` or `scoring_hash`: a scoring edit re-probes, it never
refuses the features ([hashing](hashing.md#scoring-side)). An exception inside the readout driver writes a `failed`
`result.json` (no DONE) and is re-raised.

### Training mechanics

- Determinism: construction and training run inside `torch.random.fork_rng` + `manual_seed(seed)`, with
  `use_deterministic_algorithms(True, warn_only=True)`, `cudnn.benchmark = False` and
  `CUBLAS_WORKSPACE_CONFIG` set when unset. `warn_only` lets ops without a deterministic CUDA kernel run (the
  fp32 attention backward), so CUDA reruns agree to float rounding, not bit for bit. Torch's warnings about
  such ops are captured, not printed: `probe_fit.json` records `determinism: {mode, device,
  nondeterministic_ops, bitwise_reproducible}` and the result gets `W_NONDETERMINISTIC` naming the ops.
- Precision: training and scoring in fp32, `allow_tf32 = False`, matmul precision `highest`.
- Micro-batching: token-capped (`max_tokens_per_forward`) with exact per-sample-sum accumulation for every loss.
- lr arms: the arms of `lr_grid` train one after the other, each with its own passes over the data. Every arm
  re-forks and re-seeds, so every arm sees the same split, the same initialisation and the same batch order;
  selection compares their best val losses.
- Feature access: a split whose stored bytes fit the RAM budget is read once into one RAM array (stored dtype)
  that every arm, fold and seed of the probe shares; above the budget the rows are gathered from the memmapped
  shards per batch. The budget is `min(16 GiB, 0.45 × memory limit)` (SLURM's, else the cgroup's), else 16 GiB;
  `YAVIH_RAM_BUDGET_GB` or `[resources] ram_budget_gb` overrides it. The backend never changes a number.
- Loader: with `loader_workers` `None` (default) or `>= 1`, one background thread gathers the next two training
  batches while the current one trains, consumed in epoch order; `0` gathers inline. Either way the batches and the
  numbers are the same. Rows are upcast to fp32 per batch: on CPU in the gather, on CUDA after the host→device copy
  (fp16 features cross in their stored dtype from pinned memory); the fp16→fp32 cast is exact, so both give the same
  values.
- Deadline and signals: SIGTERM / SIGUSR1 set a flag checked between micro-batches; the deadline is checked
  between epochs. `probe_ckpt.pt` of the last completed epoch is kept in memory and written at most every 600 s
  (`YAVIH_PROBE_CKPT_EVERY_S`, `0` = every epoch), at the end of every lr arm, and always on deadline or signal,
  before exit 4; a rerun resumes when the hashes match. Every checkpoint of a seed is deleted once its result is
  written.
- Scratch: when the train bytes exceed the RAM budget, the role splits are staged (up to four files copied at once,
  each to a tmp name then renamed) to
  `<scratch>/<unit_id>.<host>.<pid>/` (root: `--scratch`, else `YAVIH_SCRATCH`, `[scratch] root`,
  `SLURM_TMPDIR`) after a free-space preflight; stale siblings (dead pid on this host, or older than 36 h) are
  swept, and the copy is removed on exit, deadline or signal. With no scratch configured the probe records
  `W_SIZE` and reads in place through the memmap backend. A `--scratch` directory that already holds a
  `features/` copy (from `yavih features stage`) is used and never deleted. See [execution](execution.md).

### Outputs

Per seed, under `<run_dir>/probe/<eval_hash12>/seed<k>/`: `result.json` and `DONE` ([results](results.md)),
`eval_logits.npy` (float32, the head's raw eval output), `eval_rows.npy` (int64 feature rows, for joins),
`probe_fit.json` (per-arm epochs, s/epoch, selected lr, backend, determinism) and readout extras
(`floor_logits.npy` / `floor_rows.npy` for a floor role, `eval_fold_index.npy` under folds,
`eval_outputs_shuffled.npy` for the box control). A seed whose DONE matches the current hashes, model, reading,
`describe_sha256` and features fingerprint is skipped; any other earlier output is moved to `probe/_stale/`.
After the DONE markers, unless `--no-results`, one results-store file for the invocation (and with
`--log-samples` one samples file) is written; a failure to write the store is reported on stderr and never
fails the probe.

## ProbeConfig

```python
@dataclass(frozen=True, kw_only=True)
class ProbeConfig:
    readout: str
    head: dict                     # architecture descriptor (JSON primitives)
    lr: float; lr_grid: tuple[float, ...]; weight_decay: float; max_epochs: int; batch_size: int
    patience: int; warmup_fraction: float
    schedule: str = "cosine"; val_fraction: float = 0.1
    loss: str; selection: str = "val_loss"; max_tokens_per_forward: int = 131072; precision: str = "fp32"

    @staticmethod
    def pinned(readout, bench=None, *, num_classes=None, label_kind=None, T=None, grid=None, target_size=None)
```

`pinned(readout, bench)` fills the task geometry from the benchmark: the class count, the single/multi loss
choice, `T`, the grid and the target size. Pinned values:

| field | `clf` | `seq` | `grid` | `box` |
|---|---|---|---|---|
| `head.family` | `attentive_probe_v1` | `dense_query_v1` | `dense_query_v1` | `dense_av_v1` |
| `head.self_attn_depth` / `cross_attn_depth` | 3 / 1 | 3 / 1 | 3 / 1 | 2 (over cells) / 2 (cells → conditioning) |
| `head.num_heads` / `mlp_ratio` | 16 / 4 | 16 / 4 | 16 / 4 | 16 / 4 |
| `head.memory_pos` | `none` | `sincos(t)+null_embed` | `sincos(y,x)+null_embed` | `sincos(y,x) on binned grid` |
| `head.layout` | `point` | `time:T` (`readout_params.T`) | `grid:HxW` (`readout_params.h/w`) | `box`, `grid: G` (`readout_params.grid`) |
| `head.out` | `num_classes` | `num_classes` | `1` (log-depth), `upsample: target_size`, `activation: exp` | `4` (cx, cy, w, h), `activation: sigmoid` |
| `loss` | `ce` (single) / `bce` (multi, multi_sets) | `ce_seq` (masked) | `l1_log` | `box_l1_giou` (mean L1 over 4 coords + (1 − GIoU)) |
| `lr`, `lr_grid` | 1e-4, () | 3e-4, (1e-4, 3e-4, 1e-3) | same as seq | same as seq |
| `weight_decay` | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| `max_epochs` / `batch_size` / `patience` | 30 / 256 / 5 | 50 / 64 / 10 | 50 / 64 / 10 | 50 / 64 / 10 |
| `warmup_fraction` | 0.0 | 0.05 | 0.05 | 0.05 |
| `max_tokens_per_forward` | 131072 | 131072 | 131072 | 131072 |

Optimizer AdamW (betas 0.9 / 0.999); linear warmup over `ceil(warmup_fraction · total)` steps, then cosine
1 → 0 over the rest of `steps_per_epoch · max_epochs`. `ProbeConfig` validates itself on construction (head
family, loss, schedule, selection, precision, positive sizes) so a bad config is refused before it is hashed.
`to_json()` is what `result.probe.config` records; `probe_hash` hashes an explicit field list
([hashing](hashing.md#probe_hash)).

### Standard and non-standard

`cfg=None` uses the pinned config and the result is `standard: true`. Any other `cfg` requires
`YAVIH_TESTING=1` (else `E_NONSTANDARD_CONFIG`, exit 1); the result is standard only when its `probe_hash`
equals the pinned one. `ProbeConfig.fast()` (test-only, also gated by `YAVIH_TESTING=1`) keeps the head family
and task geometry but uses depth 1/1, 4 heads, `lr = max(lr, 3e-3)`, no lr grid, no warmup, 25 epochs and
patience, `batch_size ≤ 16`, `max_tokens_per_forward ≤ 32768`. A non-standard result has its own `eval_hash`
and is never in a comparable report column.

## Coordinates

The bridge writes `pos = (stream, t, y, x)` in media coordinates: `t` a fraction of the token's own stream
duration, `(y, x)` fractions of the original frame, `-1` for no coordinate ([bridge](bridge.md)). The readouts
re-express them in the task frame before the head sees them:

- `seq`: `u = (τ − segment_offset · segment_seconds) / (T · segment_seconds)` with `τ = t · duration_s` of the
  token's own stream;
- `grid` and `box`: `((y − y0) / (y1 − y0), (x − x0) / (x1 − x0))` over `protocol.target_region` `[y0, x0, y1,
  x1]` (the full frame when absent).

Tokens outside the scored span or region keep their out-of-range coordinates (no clamping); a missing
coordinate becomes `NaN`, which the heads map to a learned null embedding per axis. A bridge that cropped
leaves the cells it did not see empty; a bridge that read 8 frames and one that read 20 over the same seconds
are read at the same seconds.

Dense query head (`seq`, `grid`): memory tokens get `Linear(D, D)` plus a fixed 1-D/2-D sincos embedding (axis
scale 64) of their task coordinate; queries are initialized from sincos of the task coordinate (segment centres
`(k + 0.5)/T`, cell centres of the region); then 3 self-attention + 1 cross-attention blocks and a shared
`Linear`.

## clf

- Inputs: features `[n, N, D]`; `pos` is ignored (`memory_pos: none`: the head is permutation invariant over
  tokens). `readout_params: {}`.
- Head: attentive probe, 3 pre-norm self-attention blocks over the token set, one cross-attention block with a
  single learned query (`trunc_normal(std=0.02)`), `Linear(D, C)`.
- Targets: `int64 [n]` (single) or `float32 [n, C]` multi-hot (multi). `multi_sets`: train targets come from
  `manifest.label.column` (one-hot) or, with `manifest.label.train_set`, from that set's columns; eval builds
  one target matrix per `label.sets` entry, scored separately; cells of columns that do not apply to a row are
  empty (`label.empty_ok: true`).
- Metrics: whatever `metrics.list` names: single → `acc`; multi → `map`, `effective_pos_classes`; multi_sets →
  `auc_pr_<set>`, `auc_roc_<set>`; folds → `acc_mean`, `acc_pstd`, `acc_per_fold`, `n_folds`; plus folder
  metrics (avsync: `acc_tol1`, `mae_seconds`, `acc_coarse5`, `n_coarse5`).
- Floors: `acc_floor` (one-hot of the train majority class, ties to the lowest index), `map_floor` /
  `auc_pr_<set>_floor` (train prevalence per class as a constant score). Under folds each fold's floor is its
  own train majority.
- Floor role (avsync): the same trained probe scores the floor rows → `acc_floor_arm`; `floor_violated =
  acc_floor_arm > chance + floor_sigmas · sqrt(chance (1 − chance) / n_floor)` (avsync: 0.1233 at
  n_floor = 5 997, 3 σ).

## seq

- Inputs: features, `pos.t` and each row's `duration_s` per stream (`FeatureReader.durations`). A token's
  absolute time is `τ = t · duration_s` of its own stream; tokens with `t = -1` carry no time. A timed token
  whose stream has no `duration_s` is `E_DURATIONS`.
- `readout_params: {T}` with `T = protocol.segments_scored`; protocol key `segment_seconds`;
  `output_contract.requires.t_bins` is required and at most `T`.
- Targets `int64 [n, T]`: scored index `k ∈ [0, T)` is media second `s = segment_offset + k` (the segment
  `[s · segment_seconds, (s + 1) · segment_seconds)`) and takes the row's class iff `event_start ≤ s <
  event_end`, else `background_index`. With `segment_offset: 0` and `segments_scored == num_segments` (AVE)
  every second of the clip is scored, the official AVE accuracy. A last segment the media only partly covers is
  still scored.
- Head: dense query head, layout `time:T`, one query per scored segment; loss `ce_seq`.
- Metrics: `seg_acc` (pooled over valid (clip, segment) pairs), `clip_acc` (mean of per-clip accuracies; equals
  `seg_acc` when every clip has the same number of scored segments), `seg_acc_boundary` on rows with
  `event_end − event_start < num_segments` (AVE test: 104 of 316), `n_boundary`, `n_eval`.
- Floors: `seg_acc_floor`, `seg_acc_boundary_floor`: the most frequent train segment label over all
  (clip, segment) cells, read on all rows and on the boundary rows.
- Eligibility (validate): the median number of scored seconds holding a timed token reaches `t_bins`.

## grid

- Inputs: features and `pos.(y, x)` (full-frame fractions) of the stream named `readout_params.visual_stream`.
  `readout_params: {h, w, out, visual_stream}`; protocol keys `depth_param`, `source_size`, `target_region`,
  plus the metrics' `max_depth`, `target_size`. Exactly one split per role.
- Targets: `Benchmark.targets(split)`, depth `float32 [n, S, S]` (`S = protocol.target_size`) = the raster of
  `protocol.target_region` of the full frame (NYUv2: `[0.0, 0.125, 1.0, 0.875]`, the centre 480 × 480 of the
  480 × 640 frame). Valid pixels: `0 < d ≤ max_depth`; invalid train pixels are masked in the loss.
- Head: dense query head, layout `grid:HxW`; the `h × w` query cells tile the target region; memory tokens carry
  region-relative `(y, x)`, tokens outside the region included. One log-depth output per cell, bilinear upsample
  to `S × S`, then `exp`; loss `l1_log`.
- Metrics, per image then mean over images (`pred · valid` first): `d1`, `d2`, `d3` (fraction of valid pixels
  with `max(gt/pred, pred/gt) < 1.25^k`), `absrel`, `rmse`, `n_eval`. nyuv2_depth computes them with Probe3D's
  code through its `probe.py`.
- Floor: `d1_floor` of `exp` of the per-pixel mean train log-depth map, computed in one streaming pass over the
  train targets (pixels valid in no train image take the global mean).
- Eligibility (validate): the median fraction of the `h × w` region cells holding a visual token centre reaches
  `requires.grid.min_cell_coverage`.

## box

- Inputs: features and `pos` with the two stream tags named by `readout_params.visual_stream` and
  `conditioning_stream` (resolved through `describe()["streams"]`; they must differ). `readout_params: {grid,
  visual_stream, conditioning_stream, shuffled_control, shuffle_seed}`; `readout_params.grid` must equal
  `requires.grid` h and w.
- Targets `float32 [n, 4]` `(cx, cy, w, h)` in full-frame fractions from `x1|y1|x2|y2` (sanitized `max(x, 0)`);
  degenerate boxes (`x2 ≤ x1` or `y2 ≤ y1`) are excluded and counted (`n_excluded_degenerate`).
- Head `dense_av_v1`: visual tokens (stream tag of `visual_stream`, `y, x ≥ 0`) are mean-pooled per cell of the
  `G × G` grid tiling the target region by their full-frame `(y, x)`; a cell with no token gets a learned
  null-cell embedding; fixed 2-D sincos added; 2 self-attention blocks over the `G²` cells; 2 cross-attention
  blocks from cells into all tokens of `conditioning_stream`; one learned box query cross-attends the cells;
  `MLP(D → D → 4)` + sigmoid. Loss `box_l1_giou`.
- Metrics, in full-frame fractions (IoU is invariant under the per-axis scaling to pixels): `iou_mean`,
  `acc@0.5`, `acc@0.3` (fraction of rows with IoU at or above the threshold), `center_in_gt` (predicted centre
  inside the ground-truth box, inclusive), `n_eval`, `n_excluded_degenerate`.
- Floor: `iou_floor`, the IoU of the mean train box.
- Control (`shuffled_control: true`): a second head is trained and scored with the conditioning stream's tokens
  replaced by another clip's under a seeded derangement (`shuffle_seed` for train, `shuffle_seed + 1` for eval;
  partners must hold the same number of conditioning tokens, else `E_POS_SHUFFLE`) → `iou_shuffled`;
  `audio_blind = not (McNemar z over paired acc@0.5 successes > 3)`.
- Eligibility (validate): both named streams are present, the visual stream covers at least
  `min_cell_coverage` of the cells (median over the sample), and any `requires.annotated_frame` condition holds.
