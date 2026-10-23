# Results

A probe writes one `result.json` per seed: the atomic, DONE-marked record the scheduler and the report rely on.
`yavih report build` indexes those records and prints comparability-checked tables. The results store (one
directory per model, lm-evaluation-harness style) is derived from them for downstream readers and is never read
to decide anything.

How the numbers are produced: [readouts](readouts.md). What `eval_hash`, `protocol_hash` and `probe_hash`
cover: [hashing](hashing.md). Validation waivers: [validation](validation.md).

## result.json

Path: `<run_dir>/probe/<eval_hash12>/seed<k>/result.json`, where a standard run directory is
`<runs root>/<benchmark>/<model>/<reading>/`. `result.json` is written atomically and schema-checked
(`readouts.base.check_result`); for `status` `ok` or `not_applicable` a `DONE` marker follows, carrying
`result_sha256`, `protocol_hash`, `probe_hash`, `eval_hash`, `model`, `reading`, `describe_sha256`,
`features_fingerprint`, `yavih_version` and `created`. A `failed` result never gets a DONE.

```jsonc
{
  "schema_version": 4,
  "status": "ok",                          // "ok" | "not_applicable" | "failed"
  "reason": null,                          // a string for not_applicable / failed
  "standard": true,                        // false for a non-pinned ProbeConfig
  "eval_hash": "…",                        // sha256(canon({protocol_hash, probe_hash})): the comparability key
  "benchmark": {"name": "ave", "protocol_version": 2, "readout": "seq", "readout_version": 2,
                "manifest_hash": "…", "protocol_hash": "…", "folder_source": {"probe": "…", "reader": null}},
  "model": "vjepa2-l",                     // describe()["model"]
  "reading": "av",                         // describe()["reading"]
  "bridge": {"name": "vjepa-av-ave", "describe": {…}, "describe_sha256": "…", "source_sha256": "…"},
  "features": {"fingerprint": "…", "dtype": "float16", "N": 1568, "D": 1024, "pos_mode": "per_sample",
               "bytes": 5.2e9, "backend": "ram", "canary_max_rel_l2": 3.1e-6, "batch_tolerance": 1e-05,
               "rules_checked": {"video": {"temporal": "uniform", "spatial": true}}},
  "probe": {"hash": "…", "config": {…}, "seed": 0, "selected_lr": 3e-4, "epochs_run": 23,
            "best_epoch": 13, "stop": "patience", "n_train": 2362, "n_val": 262},
  "metrics": {"seg_acc": 0.412, "clip_acc": 0.412, "seg_acc_boundary": 0.531, "n_boundary": 104, "n_eval": 316},
  "ci95": {"seg_acc": [0.36, 0.47], "seg_acc_boundary": [0.44, 0.63]},
  "floors": {"seg_acc_floor": 0.164, "seg_acc_boundary_floor": 0.499},
  "controls": {},
  "chance": {"seg_acc": 0.034483},
  "external": {},
  "coverage": {"train": {"present": 2624, "nominal": 2624, "ids_sha256": "…"},
               "eval":  {"present": 316,  "nominal": 316,  "ids_sha256": "…"},
               "floor": null},
  "not_applicable": null,
  "timing": {"validate_s": 12.1, "materialize_s": 40.2, "fit_s": 611.0, "score_s": 3.2, "wall_s": 670.0,
             "s_per_epoch": 26.5},
  "env": {"host": "gpu-node-02", "gpu": "NVIDIA A100-SXM4-80GB", "torch": "2.5.1+cu124", "python": "3.12.4",
          "yavih": "0.2.0", "slurm_job_id": "…"},
  "warnings": [],
  "waived": [],
  "created": "2026-10-07T14:00:00Z"
}
```

Values are illustrative.

| key | content |
|---|---|
| `schema_version` | `RESULT_SCHEMA_VERSION` (4) |
| `status`, `reason` | `reason` is null for `ok`, a string otherwise |
| `standard` | true iff the probe config's `probe_hash` equals the pinned one |
| `eval_hash` | full hex; must equal the hash of `benchmark.protocol_hash` and `probe.hash` |
| `benchmark` | exactly `name` (canonical), `protocol_version`, `readout`, `readout_version`, `manifest_hash`, `protocol_hash`, `folder_source` (`{probe, reader}` normalized-source digests of the folder's `probe.py` / `reader.py` at scoring: provenance, never a key; `null` in a schema-3 record read through [`report.migrate`](hashing.md#reportmigrate)) |
| `model`, `reading` | from the bridge's `describe()`; `model` matches `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`, `reading` `[A-Za-z0-9][A-Za-z0-9_-]{0,31}` |
| `bridge` | exactly `name`, `describe`, `describe_sha256` (current describe schema), `source_sha256`; recorded, never a key |
| `features` | exactly `fingerprint` (the split's, or a sha256 over the per-split ones), `dtype`, `N`, `D`, `pos_mode`, `bytes` (summed), `backend` (`ram` / `memmap`), `canary_max_rel_l2`, `batch_tolerance`, `rules_checked` (per stream: the temporal rule or `"custom"`, the spatial check). Null for a plan-time `not_applicable`; every key present with null geometry for a bridge-declared one |
| `probe` | exactly `hash`, `config` (`ProbeConfig.to_json()`), `seed`, `selected_lr`, `epochs_run`, `best_epoch`, `stop`, `n_train`, `n_val` (fit fields null for a non-ok status; under folds, means over folds) |
| `metrics`, `ci95`, `floors`, `controls` | `ok` only (see rules) |
| `chance` | `metrics.chance` |
| `external` | `{package: version}` of the packages whose code produced a reported number; `{}` when none |
| `coverage` | `train`, `eval`, `floor` (each `{present, nominal, ids_sha256}` or null), plus `folds: {fold: {…}}` under `splits.folds`. Null for a plan-time `not_applicable` |
| `not_applicable` | `{missing, reason, decided_by}` with `decided_by` ∈ `validate`, `plan`, `bridge`; null otherwise |
| `timing` | exactly `validate_s`, `materialize_s`, `fit_s`, `score_s`, `wall_s`, `s_per_epoch` |
| `env` | exactly `host`, `gpu`, `torch`, `python`, `yavih`, `slurm_job_id` |
| `warnings` | probe-time findings `[{code, message, where}]` |
| `waived` | validation waivers in force over the probe's splits `[{code, reason, source, count, splits}]`, merged per `(code, source)` with counts summed |
| `created` | UTC timestamp |
| `sample` | only on a result of a capped run (`plan build --sample N`): `{split: [lo, hi]}`, the rows each split's features cover (the `_meta.json` `sample` keys). Absent on every other result |

Rules:

- Metric blocks of a non-`ok` result are absent, never zero-filled.
- For `ok`, `metrics` holds exactly `metrics.list`, `floors` exactly `metrics.floors`, `controls` exactly
  `metrics.controls`, and `ci95` one `[lo, hi]` per non-null `metrics.ci` entry. Coverage and probe facts are
  separate blocks and never named in `metrics.list`.
- A metric value is `null` only when the folder's external scorer declares it not applicable to this
  configuration (it returns `None` for it; vggsounder's `*_only` metrics for the `av` reading, which has no
  exclusive label set). A registry metric never returns `null`: a missing value refuses the result.
- A benchmark may declare `metrics.floors: []`: its results then carry `floors: {}` and the report shows no floor
  and greys no row for it (vggsounder, whose numbers are all the authors' package's).
- `features.rules_checked` records a bridge's opt-out (`"custom"`); nothing flags it.
- `waived` is recorded and marked by the report, never part of a key, never a penalty.
- A result with `sample` never reaches the results store (the probe skips it whatever `--results` says), so
  `--merge-results` never sees it; the report shows it as a `partial` row ([below](#partial-rows)).

### Warnings

| code | when |
|---|---|
| `W_COVERAGE` | eval coverage below `min_eval_coverage`, accepted with `--allow-low-coverage` |
| `W_WEIGHTS_UNFINGERPRINTED` | `describe()["weights_sha256"]` is null |
| `W_NONDETERMINISTIC` | ops without a deterministic CUDA kernel ran during training (named in the message) |
| `W_SIZE` | the train bytes exceed the RAM budget and no scratch is configured: memmap in place |

### The result key

A result is keyed `(benchmark name after alias, eval_hash, model, reading, probe.seed)`. Nothing in `bridge`
(`describe()`, `describe_sha256`, weights fingerprint, args, dtype, batch size) is part of it: those are
recorded so a reader sees what produced the number, and the report uses `describe_sha256` only to tell
configurations of one key apart. Two numbers of one benchmark are comparable iff `eval_hash` and `standard`
agree (`yavih.report.comparable`; [hashing](hashing.md#eval_hash-and-comparability)).

## yavih report

```python
def report(globs, *, fmt="md", seeds="aggregate", allow_coverage_diff=False, join=None, detail=False,
           results_root=None, merge_results=False, suite=None) -> Report
```

CLI: `yavih report build <glob>... [--fmt md|csv|json] [--out PATH] [--seeds aggregate|list]
[--allow-coverage-diff] [--detail] [--join BENCH READING=RUN ...] [--results DIR] [--merge-results]
[--suite S]`.

### What it reads

- Every `result.json` at any depth under the globs that has a valid sibling `DONE`: a JSON object whose
  `result_sha256` matches the file and whose recorded hashes, model and reading equal the result's. Anything
  else, including a `failed` result, is skipped.
- The index `<runs root>/_index.jsonl`: one line per result, cached on `(path, mtime_ns, size)`, so a rerun costs one
  `stat` per result.
- Every record passes through `yavih.report.migrate` in memory: a schema newer (or older) than this yavih's is
  refused, and every record's `protocol_hash` is mapped to its benchmark's newest hash through the folders'
  changelog `supersedes` lists (and `eval_hash` recomputed) ([hashing](hashing.md#migration)). Files on disk are
  never rewritten.
- A result found under a `probe/_stale/` directory is indexed `moved_aside` and is never a row (below).

Outputs: the rendered table is printed and written to `<runs root>/_report/report.<fmt>` with a `DONE` marker;
`--out PATH` also writes it to `PATH` (a directory gets `report.<fmt>`). With `--json`, the report's warnings
are also the envelope's findings ([cli](cli.md)).

### Selection

Both views select results the same way:

- Row key `(benchmark after alias, eval_hash, model, reading)`; standard and non-standard results apart.
- Configuration: within a row key, results are grouped by `bridge.describe_sha256`. More than one group is
  `W_BRIDGE_CHANGED`; the group holding the newest `created` is shown, the others are listed under `--detail`
  with their `describe()` diff against the shown one, their seeds and headline values.
- Moved-aside outputs: a moved-aside result whose `describe_sha256` differs from every live configuration of its
  row key is an older configuration of that row (`W_BRIDGE_CHANGED` names it "only in outputs moved to
  probe/_stale/"; `--detail` marks `moved_aside: true`), listed after the live ones and never shown. A
  moved-aside result of a live configuration is ignored. A row key with moved-aside results only gets no row.
- Feature set: within the shown configuration, results are grouped by `features.fingerprint`; more than one is
  `W_FEATURES_DIFFER` and the newest is shown.
- Seeds: aggregated over one configuration and one feature set only, deduplicated on `probe.seed` (newest
  `created` wins), so `n_seeds` counts distinct seeds. `--seeds list` prints one row per seed instead.

### View 1: per-model tables

One `## <model>` table, one row per `(benchmark, reading)` and `metrics.list` metric with a value (headline
first, disclosures such as `n_eval` last):

| benchmark | reading | version | eval_hash | readout | metric | value | ci95 | floor |
|---|---|---|---|---|---|---|---|---|
| ave | av | 2 | 1a2b3c4d5e6f | seq | **seg_acc** | 0.412 | [0.36, 0.47] | 0.164 |

`version` is `protocol_version`; `eval_hash` its first 12 hex; `value` the mean over seeds (`± std` and
`(n_seeds k)` when several); `ci95` the interval of the lowest seed, or `—`; `floor` the value of the floor
whose `floor_of` names the metric, or `—`. A `not_applicable` benchmark prints one row with `metric = —` and the
reason; a non-standard probe is labelled `<benchmark> (nonstandard)`.

### View 2: comparability columns

One column per `(benchmark, eval_hash)` (header: benchmark, `protocol_version`, `eval_hash12`, standard), rows
by `(model, reading)`. Each metric prints `value (floor)` and its CI (the mean of the per-seed bounds),
`mean ± std` and `n_seeds`. Row flags:

| flag | meaning | effect |
|---|---|---|
| `below_floor` | the headline CI's lower bound (or, without a CI, the value) is at or below its floor | greyed (md `~~value~~`) |
| `floor_violated`, `audio_blind` | a voiding control fired | greyed |
| `low_coverage` | eval coverage below `min_eval_coverage` or `W_COVERAGE` recorded | marked |
| `population_differs` | eval `ids_sha256` differ within the column | marked; below `min_eval_coverage` refused (`E_POPULATION_DIFFERS`, exit 2) unless `--allow-coverage-diff` |
| `single_seed` | one seed aggregated | marked |
| `not_applicable`, `failed` | status | the reason (and `missing`) is printed |
| `missing` | the model has this reading elsewhere but not here | `MISSING` cells |

A reading is expected in a column only when the benchmark has every modality the reading's streams consume
(audio = `audio`; visual = `video`, `image`; the column's modalities from the packaged
`streams.*.kind`): audio-only benchmarks expect `{a}`-type readings, visual-only `{v}`, audio-visual all. A
stream that cannot be placed, or a benchmark that is not packaged, keeps every reading expected. Columns note
`population differs`, a superseded `protocol_version` (every version below the newest seen or packaged is
listed) and a non-standard config.

No row is greyed, flagged or footnoted because its bridge declared a `custom` rule, because its weights are
unfingerprinted (`W_WEIGHTS_UNFINGERPRINTED` is listed once among the report's warnings), or because of any
other bridge decision.

### Ties

Two values under one `eval_hash` that differ by no more than the metric's conformance tolerance
([conformance](conformance.md#tolerance)) are a tie, not an order. In a standard column, per metric with a
direction (`higher_is_better`; a disclosure such as `n_eval` has none), over the rows that are `ok`, full, at
coverage and not greyed: the values within the tolerance of the best one are tied when there are at least two of
them. md prefixes each tied value with `≈` and notes it under the column title (the column flag `ties`); json
cells carry `"tie": true` (every other ranked cell `"tie": false`); csv has a `tie` column (`true` / `false`, empty
where nothing is ranked). Tied rows are never ordered by value. When the benchmark's folder is resolvable here
(packaged with this yavih, project-local, on `$YAVIH_BENCH_PATH` or in a pack), its tolerance always applies: the
metric's `metrics.tolerance` entry if declared, else yavih's default `0.001`; a tolerance recorded by a registry
entry (`conformance.tolerance`, [registry](registry.md)) never overrides it. Only for a benchmark no folder here
resolves is the largest tolerance recorded by the column's registry entries used (else the default).

### The † waived mark

A row whose results record validation waivers (`result.json` `waived`) is marked, not greyed: md puts `†` after
the value (view 1) and after the model name (view 2) and lists under the table `† <benchmark|model> <reading>:
waived <CODE> (<reason>; <source>, <n> finding(s))`; json rows carry `waived`, csv a `waived` column
(`;`-joined codes). The row is not excluded, flagged or counted against.

### Partial rows

A result of a capped run (`plan build --sample N`, `result.json` `sample`) is scored on the first rows of each
split, so its numbers are not the benchmark's. Within a row key it never competes with, or is averaged with, the
full run's results: it is a row of its own, sorted after the full row, and marked in every format: md appends
`(partial)` to its values (view 1 and view 2) and view 2 lists the flag `partial`; json rows carry
`"partial": true` (view 2 rows also `sample`) and the flag; csv has a `partial` column (`true` / `false`) and the
flag. A partial row is never greyed for being partial and takes no part in the `population_differs` check.
`--detail` keys it `<model>:<reading>:partial`. Capped runs live under their own runs root (`<out>-sample<N>`), so
they reach a report only when its globs name that root. In `--suite` completeness a pair with partial results only
is `missing` with the note `partial only`.

### Formats and detail

- `md`: view 1, then view 2. A benchmark that declares `metrics.table` shows only those metrics by default:
  they are view 2's metric columns and view 1's metric rows; every other metric of `metrics.list` is in csv, json
  (`columns[].metrics`, the rows' `metrics`, view 1 rows with `in_table: false`) and in md under `--detail`.
  Without `metrics.table` every metric is shown (`columns[].table` = `columns[].metrics`).
- `csv`: view 2 in long format, one line per (row, metric) plus one per control, with `benchmark`, `eval_hash`,
  `protocol_hash`, `probe_hash`, `model`, `reading`, `describe_sha256`, `features_fingerprint`, `bridge_name`,
  `status`, `seed`, `n_seeds`, coverage, `metric`, `kind`, `value`, `std`, `floor`, `ci_lo`, `ci_hi`,
  `chance`, `tie` ([ties](#ties)), `partial` (`true` / `false`), `flag` (`;`-joined flags, e.g. `below_floor`)
  and `waived`.
- `json`: `lm_eval` (view 1), `columns`, `rows`, `warnings`, `superseded`, `detail`, `joins`, `merged`, `suite`, `paths`.
- csv carries view 2 only: the superseded, join, warning and suite completeness sections are in md and json.
- `--detail`: per row, the shown configuration's `describe()` and `describe_sha256`, `rules_checked`, the
  older configurations and feature sets with their `describe()` diff, seeds and headline values, and the codes
  of the shown results' recorded warnings; in md also the metrics outside `metrics.table` and each join's full
  output.

### Report warnings

| code | meaning |
|---|---|
| `W_BRIDGE_CHANGED` | a row key holds several bridge configurations; newest shown |
| `W_FEATURES_DIFFER` | the shown configuration holds several feature sets; newest shown, never averaged |
| `W_FOLDER_SOURCE_DIFFERS` | results in a column were scored by different versions of the folder's `probe.py` / `reader.py` (`benchmark.folder_source`); comparable by declaration, listed for diagnosis. A result without a recorded source (`null`: schema 3, a registry entry without one) is unknown and not compared |
| `W_POPULATION_DIFFERS` | eval populations differ within a column |
| `W_MISSING_READING` | a model lacks an expected reading in a column |
| `W_NONSTANDARD` | a column of non-pinned probe configs: comparable with nothing |
| `W_WEIGHTS_UNFINGERPRINTED` | listed once: the rows whose `weights_sha256` is null |

To show two configurations of one model side by side, give them two `model` names.

### Joins

`--join <benchmark> <reading>=RUN ...` (e.g. `--join vggsounder a=RUN v=RUN av=RUN`; a RUN is a run directory,
a `probe/<eval_hash12>/` directory or a seed directory; the lowest seed is used) calls the benchmark folder's
`probe.py` `join()` hook. The core refuses (exit 2) results of different benchmarks or `eval_hash`es
(`E_JOIN_MIXED`), different eval `ids_sha256` (`E_JOIN_POPULATION`), readings of more than one `model`
(`E_JOIN_MODEL`), a missing result or `eval_logits.npy` (`E_JOIN_MISSING`) and a non-`ok` status
(`E_JOIN_STATUS`); a benchmark without a `join()` hook is `E_JOIN_UNSUPPORTED` (exit 1); the folder's
external packages are checked as in a probe (exit 3). The folder adds its own refusals (vggsounder:
`E_JOIN_ROUTING` when a reading's bridge reads other streams than `a`/`v`/`av` imply, `E_JOIN_PROBE` for
different or non-standard probe configs). Different non-null `weights_sha256` among the readings are
`W_JOIN_WEIGHTS`, recorded, never a refusal. The joined numbers are printed and written to the model's results
directory under the key `<benchmark>:a+v+av`: the hook's whole output is stored. When that output carries
`table` (`{header, rows, notes}`, the folder's compact view), md prints that table and moves the rest under
`--detail`; json always has everything. vggsounder's join stores the package's `compute_all_results` output
under `scores` (`{set: {metric: value}}`: 30 metrics on each of `a`, `v`, `av`, `a only`, `v only`, plus `mu` on
`a`, `v`, `av`) and prints rows A / V / AV with mAP (`auc_pr`), HIT (`hit_rate@1`), F1 (`f1@1`) and mu, labelled
A, V and A∩V for the package keys `a`, `v` and `av`.

### Suite completeness

`--suite S` ([suites](suites.md)) appends a trailing `## Suite completeness: S` section to md and the `suite`
key to json (`{name, description, origin, path, results_root, rows, counts}`), without changing a column or row
of either view. It has one row per suite `(capability, benchmark, reading)` and one status column per model of
the report:

| status | meaning |
|---|---|
| `done` | a result of `(benchmark, model, reading)` is in the report (a note when none is `ok` or standard) |
| `missing` | none is |
| `pending` | the suite entry has no packaged folder |
| `join-missing` | a join row (`<a+v+av> (join)`) whose entry is neither computed by this invocation's `--join` nor in the model's results store; the command that computes it is listed under the table |

json rows are `{capability, benchmark, reading, model, status, required, note, hint}`. `yavih suite missing`
prints the same statuses for one model without building a report.

## Results store

### Root

Decided before any training (`yavih.readouts.base.resolve_results_root`): `probe run --results DIR` (a matrix
`results` key), else `YAVIH_RESULTS`, else config `[results] root` ([configuration](configuration.md)), else
the directory `results` next to the runs root of a standard run directory: the absolute run directory's last
three components must be `<benchmark or alias>/<model>/<reading>` of the feature set being probed, and the runs
root is the directory above them. Any other run directory is refused with `E_RESULTS_ROOT` (exit 1, nothing
trained). `--no-results` needs no root, and neither does a capped run's probe (`_meta.json` `sample`), which never
writes the store. `report build` uses `--results`, else the same configured root, else
`results` next to the runs root of the indexed results.

### Layout

```
<results_root>/<model>/
  results_<timestamp>.json                          one per probe invocation (one entry), per report --join
                                                    (one entry), or merged by report build --merge-results
  samples_<benchmark>_<reading>_<timestamp>.jsonl   with --log-samples
```

One directory per model name, every reading and benchmark of it inside. `<timestamp>` is UTC
`%Y-%m-%dT%H-%M-%S.%f`. Files are written tmp + `os.replace` after the unit's DONE markers. Entries are keyed
`<benchmark>:<reading>` (canonical benchmark name; a reading cannot contain `:` or `+`), a join's entry
`<benchmark>:a+v+av` (readings in routing order), an older entry kept by a merge
`<benchmark>:<reading>@<eval_hash12>`. Readers take `benchmark`, `reading` and `eval_hash` from
`configs[key]`, never by parsing the key.

### File schema

`RESULTS_FILE_VERSION = 2`:

```jsonc
{
  "results_version": 2,
  "results": {"ave:av": {"seg_acc": 0.412, "seg_acc_ci95": [0.36, 0.47], "seg_acc_floor": 0.164,
                         "clip_acc": 0.412,
                         "seg_acc_boundary": 0.531, "seg_acc_boundary_ci95": [0.44, 0.63],
                         "seg_acc_boundary_floor": 0.499, "n_boundary": 104, "n_eval": 316,
                         "status": "ok", "headline": "seg_acc", "n_seeds": 1, "coverage_eval": 1.0}},
  "configs": {"ave:av": {"benchmark": "ave", "reading": "av", "eval_hash": "…",
                         "readout": "seq", "readout_version": 2, "protocol_version": 2,
                         "protocol_hash": "…", "probe_hash": "…", "manifest_hash": "…",
                         "roles": {"train": {"splits": ["train"], "n": 2624}, "eval": {"splits": ["test"], "n": 316}},
                         "metrics": {"headline": "seg_acc", "list": […], "floors": […], "controls": []},
                         "input_contract": {…}, "output_contract": {…}, "external": [],
                         "probe": {"config": {…}, "seeds": [0], "standard": true},
                         "bridge": {"name": "vjepa-av-ave", "describe": {…}, "describe_sha256": "…", "source_sha256": "…"},
                         "describe_sha256": "…", "features_fingerprint": "…",
                         "unit_results": ["runs/ave/vjepa2-l/av/probe/<eval_hash12>/seed0/result.json"]}},
  "versions": {"ave:av": 2},
  "higher_is_better": {"ave:av": {"seg_acc": true, "clip_acc": true, "seg_acc_boundary": true}},
  "n_samples": {"ave:av": {"train": 2624, "eval": 316}},
  "config": {"model": "vjepa2-l", "bridge": {…}, "probe": {"config": {…}, "seeds": [0], "standard": true},
             "external": {}},
  "git_hash": "3f2a…",
  "yavih_version": "0.2.0",
  "date": "2026-10-08T14:00:00Z",
  "env": {"host": "gpu-node-02", "python": "3.12.4", "torch": "2.5.1+cu124", "cuda": "12.4",
          "gpu": "NVIDIA A100-SXM4-80GB", "slurm_job_id": "…"}
}
```

| key | content |
|---|---|
| `results[key]` | every `metrics.list` value under its name, `<m>_ci95` per `ci95` entry, `<m>_floor` for every metric with a floor (`floor_of`), every control under its name, `status`, `headline`, `n_seeds`, `coverage_eval` (`present / nominal` of eval). With several seeds a value is the mean, `<m>_std` the standard deviation (omitted for equal integer counts) and `<m>_ci95` the interval of the lowest seed. A non-`ok` entry holds only `status` and `reason`. `waived` (the codes) when waivers are recorded |
| `configs[key]` | `benchmark`, `reading`, `eval_hash` (the comparability key), `readout`, `readout_version`, `protocol_version`, `protocol_hash`, `probe_hash`, `manifest_hash`, `roles` (splits and present counts), `metrics`, `input_contract`, `output_contract`, `external` (the `benchmark.yaml` entries), `probe` (`config`, `seeds`, `standard`), `bridge`, `describe_sha256`, `features_fingerprint`, `unit_results`; `waived` `[{code, reason, source, count}]` when present. A join entry carries `joined: {reading: {describe_sha256, unit_results}}` and `warnings` instead of `bridge` |
| `versions[key]` | `protocol_version` (lm-eval's task version) |
| `higher_is_better[key]` | from `MetricSpec.higher_is_better`; metrics with `None` (disclosures, controls) omitted |
| `n_samples[key]` | present rows per role (`train`, `eval`, `floor`) |
| `config` | `model`; `bridge` and `probe` of a single-entry file (of a merged file only when every main entry agrees, else null); `external` = every package version any entry used; a merged file adds `merged_from` |
| `git_hash` | the running yavih's source commit (`yavih.buildinfo`), or null |
| `yavih_version`, `date`, `env` | writer version, UTC date, host / python / torch / cuda / gpu / SLURM job |

`configs[key].bridge` and `features_fingerprint` are per entry because one model directory holds several
readings, i.e. several bridges.

### Samples file

With `--log-samples`, one JSON line per scored row, seed and fold: `{"clip_id", "row", "seed", "role",
"fold"?, "prediction", "target"}` with `role` ∈ `eval`, `floor`:

| readout | `prediction` / `target` |
|---|---|
| clf single | class indices, plus `"correct"` |
| clf multi / multi_sets | `prediction` the top-10 `[index, score]` pairs; `target` the label indices (per set for multi_sets) |
| seq | `[T]` class indices each |
| grid | `prediction` that image's `{d1, absrel, rmse}`; `target` null |
| box | `[cx, cy, w, h]` each, plus `"iou"` |

### --registry

`report build --registry DIR` adds the accepted entries of a model registry ([registry](registry.md)) to the
comparison as rows: model `<org>/<model>`, flag `registry`, one record per recorded seed, under the same
comparability rule as every local result (same benchmark, same `eval_hash`, both standard). An entry whose
conformance did not pass is not a row. Differences within the metric's tolerance are ties ([ties](#ties): the
tolerance of the folder resolvable here, a recorded one only for a benchmark no folder resolves). An entry whose eval
coverage (`present / nominal`, else its recorded `fraction`) is below `min_eval_coverage` is flagged
`low_coverage` and takes no part in ties.

### --merge-results

`report build --merge-results` reads every `results_*.json` under `<results_root>/<model>/` for the models in
the report (a file of another `results_version` is skipped with a notice) and, for every
model with two or more files, writes one merged `results_<timestamp>.json`:

- the newest entry per `(benchmark, reading, eval_hash)` wins: one entry, so one configuration and one feature
  set, never averaged across them;
- entries of one `(benchmark, reading)` with different `eval_hash`es are all kept: the newest standard entry of
  the newest `protocol_version` under `<benchmark>:<reading>` (a non-standard entry holds it only when no
  standard one exists), the others under `<benchmark>:<reading>@<eval_hash12>`;
- every entry keeps its own provenance in `configs[key]`; file-level `config.probe` and `config.bridge` are those
  of the main-key entries when they agree, else null; `config.merged_from` lists the files whose entries were
  kept.
