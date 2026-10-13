# Validation

`yavih features validate` checks one split of extracted features (see [features](features.md)) before any
probe reads them: the shard set, the identity of the features, every tar member, the bridge's reports against
the rules it declared, batch invariance, and a few statistical sanity checks. It also decides, once and only here,
whether the features are eligible for the benchmark's readout. A split that passes gets a `_meta.json`; the probe
trusts that file and re-derives nothing.

## Running it

```
yavih features validate <split dir or features/ dir> [--verify-sha] [--no-write-meta] [--waive CODE=reason ...]
```

Given a `features/` directory, it validates every split subdirectory and prints one report per split. It exits 2
when any split has an error and 0 otherwise; a well-formed split that is not eligible for the readout exits 0. A
malformed `--waive` is a usage error (exit 1) raised before any split is read. Per split it prints the coverage and
shape, the shard table (`shards 31/32 done, 0 in progress, 1 missing: --shard 7/32`), the rules line, each
`N_RULE_CUSTOM` note, the eligibility line, and one line per waiver in force
(`waived E_DUPLICATE_CLIPS (describe, 3 finding(s)): <reason>`). The eligibility line reads, for example,
`eligible for box: no (grid coverage median 0.18 < 0.25 over the full frame) → probe will record not_applicable`.
`yavih probe run` runs the same validation as its prologue and accepts `--waive` too.

The library entry point:

```python
from yavih.io import validate_features

report = validate_features(features_dir, *, bench=None, verify_sha=False, sample_rows=512,
                           canary_rel_tol=None, write_meta=True, waive=None)
```

| argument | meaning |
|---|---|
| `features_dir` | a split directory, `<run_dir>/features/<split>/` |
| `bench` | the `Benchmark`; `None` loads `Benchmark.load(<sidecar benchmark.name>)`. When that fails, the hash checks compare sidecars with each other, stream kinds are inferred from the members, and eligibility is `null` (undecided, with the reason). |
| `verify_sha` | also hash every tar and compare with the sidecar (`E_TAR_SHA`); forces a full pass |
| `sample_rows` | size of the seeded sample used by the dead-row, duplicate and eligibility checks |
| `canary_rel_tol` | the batch-invariance tolerance; `None` uses the bridge's `describe()["batch_tolerance"]["rel_l2"]`, else `1e-5`; never above `1e-3` |
| `write_meta` | write `_meta.json` when the split passes |
| `waive` | an operator's waivers `{CODE: reason}` on top of `describe()["waive"]` (see [Waivers](#waivers)) |

`ValidationReport` holds `ok`, `errors`, `warnings`, `notes` (lists of `Finding(code, message, where)`), `split`,
`coverage`, `N`, `D`, `dtype`, `pos_mode`, `bytes`, `backend_hint` (`ram` or `memmap`), `eligibility`,
`missing_shards` (`["--shard 7/32", ...]`), `shards` (the shard table), `rules_checked`, `meta_path` and `waived`.
`raise_for_errors()` raises `ValidationError`; `to_json()` gives the `--json` payload.

Findings have three levels: `error` (the split is not `ok`), `warning`, and `note` (a recorded fact that is
neither). At most 20 findings per code are kept; the 21st becomes one "further findings suppressed" finding.

## What it reads

- The directory listing: `shard-*.json`, `shard-*.tar`, `NOT_APPLICABLE.json`, `_meta.json` and tmp files.
- Every sidecar, in full.
- Every tar, streamed once with `tarfile`. Validation does not go through `FeatureReader`: the reader trusts the
  sidecar offsets, and validation exists to check that trust. The per-member checks (`E_TIMES_OUT_OF_MEDIA`,
  `E_RULE_MISMATCH`) read only the small `.json` members and run on every member; `E_FINITE`, `W_FP16_RANGE` and
  `W_CONSTANT_CLIP` also cover every member.
- A seeded sample of present rows, `default_rng(sha256("<manifest_hash>:<split>:dup"))`, of size `sample_rows`.
  The same sample on every host, so the sampled verdicts (`E_DEAD_ROWS`, `E_DUPLICATE_CLIPS`, eligibility) are
  reproducible. `yavih features inspect` uses the same sample.
- The benchmark: hashes, `n_rows(split)`, stream kinds, the readout's
  `output_contract.requires`, `readout_params`, `protocol`, and the site's `data_status.json`.
- An existing `_meta.json`, for the cheap re-run and for carried-over operator waivers.

## Checks, in order

1. **Inventory.** A tmp file older than 36 h is `W_STALE_TMP`; a younger one counts as "in progress".
2. **Sidecars** are parsed: `E_FORMAT` for an unreadable or incomplete sidecar, a format other than 4, a
   malformed `shard` or `features` block, or a `shard.index` that differs from the file name. Sidecars naming
   several splits are `E_DESCRIBE_MISMATCH`.
3. **`NOT_APPLICABLE.json`**: alone, the split is reported complete and not applicable (nothing else is checked);
   next to shards, `E_NA_AND_SHARDS`.
4. **Shard set** (`E_SHARD_SET`): one `n`, no index missing or out of range, every sidecar with its tar and every
   tar with its sidecar. Missing indices are listed as `--shard i/n`.
5. **Identity of the benchmark**: `E_HASH_MANIFEST`, `E_HASH_INPUT`.
6. **Tiling** (`E_RANGE_TILING`), on a complete shard set only.
7. **Consistency of the feature set**: `E_DESCRIBE_MISMATCH`, then `W_SOURCE_SHA`.
8. **Waivers** are assembled (`E_WAIVE_INVALID`) and are in force for every later check.
9. `E_ARGS_UNCONSUMED`; `E_FORMAT` when `N` or `D` is not an integer of at least 1; `W_WEIGHTS_UNFINGERPRINTED`;
   `W_LOOSE_TOLERANCE`; `N_RULE_CUSTOM`; `W_SIDE_EFFECT`.
10. **Cheap re-run** on a finalized split (see below); when it applies, validation stops here.
11. **Streaming pass** over every tar: `E_TAR_SHA` (with `verify_sha`), `E_MEMBERS`, `E_ROWS`,
    `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`, `E_FINITE`, `E_POS_SHAPE`, `E_POS_RANGE`, and the canary pairs.
    Shards are scanned in parallel (below); the findings are reported in shard order.
12. `W_RULE_UNCHECKED`; `E_ROWS` for rows written in more than one shard; `E_POS_SHAPE` for a `_pos.npy` that
    differs across shards.
13. **Batch invariance**: `E_BATCH_INVARIANCE` or `N_CANARY_DRIFT`.
14. **Sample statistics**: `E_DEAD_ROWS`, `E_DUPLICATE_CLIPS`.
15. **Eligibility**: `N_ELIGIBILITY`.
16. `W_COVERAGE`, `W_MEDIA_FP`, `W_FP16_RANGE`, `W_CONSTANT_CLIP`, `W_SIZE`.
17. **`_meta.json`**: when the split is `ok` and `write_meta` is set, `finalize` writes it atomically, including
    the eligibility block whatever its verdict, and only when its content changed. When the split fails and an
    earlier `_meta.json` says `ok`, that file's `validation`, `eligibility` and `waived` blocks are replaced by the
    failing verdict, so the probe refuses the split.

### Parallel scan

The streaming pass scans several shards at once: `YAVIH_VALIDATE_WORKERS` threads, by default
`min(shards, CPUs, RAM budget / largest shard tar)` (CPUs as SLURM, the affinity mask or the cgroup allow; a shard
scan holds its tar's features in memory). `YAVIH_VALIDATE_WORKERS=1` scans one shard after the other. Each shard is
scanned on private state and the results are merged in shard order: findings are replayed in order through the
per-code cap and the waivers, and the order-sensitive sums (the float64 `sum` / `sumsq` over the sampled rows behind
`E_DEAD_ROWS`, the per-row eligibility measurements) run during the merge in scan order. The report and `_meta.json`
are therefore the same, byte for byte, whatever the number of workers.

### Cheap re-run

Validating a finalized split again is cheap. When the shard set is complete, steps 1 to 9 found no error,
`verify_sha` is off, and the existing `_meta.json` says `ok`, was decided for the benchmark's current readout, has a
`fingerprint` equal to the one recomputed from the sidecars, and lists every tar with its current size and sha256,
then the earlier verdict is reused: no tar is read and `_meta.json` is left untouched. The earlier warnings and
notes are carried over with the level the code table gives them now; a code the table no longer has is dropped.

## Eligibility

The readout's `output_contract.requires` (see [readouts](readouts.md) and
[benchmark-folder](benchmark-folder.md)) is evaluated on the features, over the seeded sample. The verdict is the
`N_ELIGIBILITY` note and the `eligibility` block of `_meta.json`. It never affects `ok`: the probe turns
`eligible: false` into a `not_applicable` result (exit 0), and refuses a split whose verdict is undecided.

| requirement | how it is measured | missing entry |
|---|---|---|
| a dense readout | `N > 1` (pooled features cannot serve it) | `tokens>1` |
| `requires.pos` axes | each axis (`t`, `y`, `x`) is non-negative for some token of the sample; never with `pos_mode none` | `pos.<axis>` |
| `requires.streams`, plus `readout_params.visual_stream` and `conditioning_stream` | the stream appears as a tag in the sampled positions (`describe()["streams"]` indices) | `stream:<name>` |
| `requires.grid` (`h`, `w`, `min_cell_coverage`) for grid and box readouts | per sampled row, the fraction of the `h x w` cells tiling the target region (`protocol.target_region`, else the full frame) that hold at least one token centre of the visual stream; eligible when the median is at least `min_cell_coverage` | `grid_cell_coverage` |
| `requires.t_bins` for seq readouts | per sampled row, the number of scored seconds holding at least one token with `t >= 0`, the token's second being `floor(t * duration_s / segment_seconds)`; eligible when the median is at least `t_bins` | `t_bins` |
| `requires.annotated_frame` (`stream`, `fact`, `tol_s`) for box readouts | per sampled row, the smallest distance in seconds between a timed token of `stream` and `protocol[fact]` (infinite when there is none); eligible when the median is at most `tol_s` | `annotated_frame` |

The scored seconds of a seq readout are `segment_offset` to `segment_offset + T - 1`, with `T` from
`protocol.segments_scored`, else `readout_params.T`, else `protocol.num_segments`.

The block records `readout`, `eligible` (`true`, `false`, or `null` when the benchmark could not be loaded),
`missing`, `reason`, `target_region`, `grid_cell_coverage` and `t_bins` (each `{median, min}` or `null`),
`annotated_frame_dt` (when required) and `stream_tags`.

## Codes

Every code `validate_features` can report. "waivable" says whether a [waiver](#waivers) can turn the error into
`W_WAIVED`.

| code | level | waivable | meaning |
|---|---|---|---|
| `E_FORMAT` | error | no | a sidecar is unreadable, not an object, lacks required keys, has a malformed `shard` or `features` block, a `shard.index` other than its file name, or a `format_version` other than 4 (`unsupported format_version N (this yavih reads 4); re-extract`); `NOT_APPLICABLE.json` is unreadable; `N` or `D` is not an integer of at least 1 |
| `E_NA_AND_SHARDS` | error | no | `NOT_APPLICABLE.json` exists together with shards |
| `E_SHARD_SET` | error | no | sidecars disagree on `n`; a shard index is missing or not below `n`; a sidecar without its tar or a tar without its sidecar; no sidecar at all |
| `W_STALE_TMP` | warning | no | a tmp file older than 36 h: a killed task's leftover; delete it or run its shard again |
| `E_RANGE_TILING` | error | no | on a complete shard set, the `shard.rows` ranges sorted by index do not tile `[0, n_rows)` of the split exactly (empty ranges allowed), or a range's start is not a multiple of `shard.batch_size` |
| `E_HASH_MANIFEST` | error | no | a sidecar's `benchmark.manifest_hash` differs from the benchmark's (without a benchmark: differs across sidecars) |
| `E_HASH_INPUT` | error | no | a sidecar's input hash (read through `identity_value`) differs from the benchmark's `input_hash` (without a benchmark: differs across sidecars). The message names the differing `input_components` and both yavih versions and commits. A difference on the scoring side is never a finding. |
| `E_DESCRIBE_MISMATCH` | error | no | the shards are not one feature set: `describe_subset(describe)`, `describe_sha256`, `model`, `reading`, `shard.batch_size`, `dtype`, `pos_mode`, `N`, `D` or `pos_sha256` differ across sidecars, or the sidecars name several splits; the stored `describe_sha256` is not the digest of the stored `describe`; `model` or `reading` differ from the `describe()` values. The message names which shards carry which digest; running `yavih extract` for the split moves the other configuration's shards to `_stale/`. |
| `E_ARGS_UNCONSUMED` | error | yes | a `bridge.args` key is not echoed in `describe()["args"]` |
| `E_TAR_SHA` | error | no | with `verify_sha` only: a tar's sha256 differs from its sidecar's |
| `E_MEMBERS` | error | no | the tar's size differs from the sidecar's `tar.bytes`; the tar is unreadable or truncated; a member name or type is unexpected; a `.json` is unparsable, lacks `clip_id`, `media` or a `used` entry for a stream in `describe()["input"]`, has an ill-formed `used` entry, or a `row` or `clip_id` that disagrees; a `.npy` header is not version 1.0, is Fortran-ordered, has an object dtype or the wrong payload size; a feature is not `(N, D)` in the declared dtype; sidecar offsets, `header_len` or `nbytes` disagree with the tar; members are not ascending; the member set differs from the sidecar's `members`; `rows.n_written` or `rows.n_missing` disagree; a canary lacks its original, its `.json` or its `.feat.npy` |
| `E_ROWS` | error | no | a row of a shard's range is neither written nor listed missing; a row is written twice (within or across shards), both written and missing, or outside its shard's range; a `missing` entry lies outside the range |
| `E_TIMES_OUT_OF_MEDIA` | error | yes | a reported video time outside `[0, media.duration_s]` (1 ms slack); an audio sample range outside `[0, media.n_samples]` or with `lo > hi`; the declared rule does not fit the clip's media; any reported time or sample range on a stream whose `media.has_track` is false (the valid report is `times: []` or `samples: []`) |
| `E_RULE_MISMATCH` | error | yes | the reported `used` differs from the declared temporal rule re-evaluated on the member's `media`, or a token's position lies outside what the bridge reported consuming (see [Rule checks](#rule-checks)) |
| `E_FINITE` | error | no | a feature value is NaN or infinite (checked on every member) |
| `E_POS_SHAPE` | error | no | positions are structurally wrong: missing for `pos_mode` `constant` or `per_sample`, present for another mode, `_pos.npy` not the first member, not `float32 [N, 4]`, non-finite, a stream column that is neither `-1` nor an integer in `[0, len(streams))`, `_pos.npy` differing across shards or from `features.pos_sha256` |
| `E_POS_RANGE` | error | yes | a coordinate `t`, `y` or `x` outside `[0, 1] ∪ {-1}` (for example letterbox padding tokens; the readouts carry such a value as it is and place no such token in a grid cell) |
| `E_BATCH_INVARIANCE` | error | yes | a canary pair's `rel_l2` is above the tolerance, or a canary reports a different `used` or different positions than its original: the features depend on batch composition |
| `E_DEAD_ROWS` | error | yes | token positions whose variance across the sampled clips is below `1e-12` times the mean token variance |
| `E_DUPLICATE_CLIPS` | error | yes | more than 1 % of the sampled clips have features byte-identical to another sampled clip with a different `media_key` (rows sharing a `media_key` read the same media and are exempt) |
| `W_WAIVED` | warning | no | a waived error: `"<CODE>: <original message> — waived: <reason>"`, with ` (--waive)` appended for an operator's waiver |
| `E_WAIVE_INVALID` | error | no | `describe()["waive"]` waives a code that is not waivable, a warning or note, or an unknown code, or gives no non-empty reason. On the command line and in a matrix entry the same mistake is a usage error (exit 1). |
| `N_ELIGIBILITY` | note | no | the eligibility verdict for the readout (see [Eligibility](#eligibility)) |
| `N_CANARY_DRIFT` | note | no | the worst canary `rel_l2` is above 0 and within the tolerance: drift the protocol accepts, recorded in `_meta.json` `canary.max_rel_l2`; no action needed |
| `N_RULE_CUSTOM` | note | no | a stream's declared temporal or spatial rule is `custom`: the bridge opted out of the rule vocabulary, with its description (see [Custom rules](#custom-rules)) |
| `W_LOOSE_TOLERANCE` | warning | no | the bridge declares `batch_tolerance.rel_l2` above `1e-5`; the declared reason is printed |
| `W_RULE_UNCHECKED` | warning | no | a declared checkable temporal rule could not be evaluated on `k` rows because a media fact it needs is null there (recorded as `rules_checked[stream].temporal_unchecked_rows`); only the out-of-media, range and position checks applied on those rows |
| `W_COVERAGE` | warning | no | fewer rows present than the split's nominal rows. The probe refuses an eval split below the benchmark's `min_eval_coverage` (`E_COVERAGE`) unless run with `--allow-low-coverage`. |
| `W_MEDIA_FP` | warning | no | a sidecar's `media_fingerprint` or `media_info_sha256` differs from this site's `data_status.json`: the features were extracted from differently staged media |
| `W_FP16_RANGE` | warning | no | float16 features whose largest absolute value exceeds `1e3`, or more than half of whose values are zero |
| `W_CONSTANT_CLIP` | warning | no | clips whose tokens are all identical (with `N > 1`) or all zero; rows sharing a `media_key` are counted once |
| `W_SIDE_EFFECT` | warning | no | a sidecar lists writes the bridge made outside the run directory during `encode` or `prepare` |
| `W_SIZE` | warning | no | the features exceed the probe's RAM budget: the probe will use the memmap backend; request scratch |
| `W_WEIGHTS_UNFINGERPRINTED` | warning | no | `describe()["weights_sha256"]` is null: the weights are recorded only by `weights_ref`; nothing is marked, forked or greyed |
| `W_SOURCE_SHA` | warning | no | the split's shards were written by bridge files with different `source_sha256` under one `describe_sha256` (an edit that left `describe()` unchanged may still change features) |

### Rule checks

`E_RULE_MISMATCH` re-evaluates, with the same `yavih.media.rules` helpers a bridge may use, what the bridge
declared in `describe()["input"]` (see [bridge](bridge.md)):

- **Temporal rule** (not `custom`): the reported times must match the rule evaluated on the member's `media`
  within 1 ms for `explicit` times and within `0.5 / fps + 1 ms` for decoded video (nearest frame). Audio sample ranges and
  padding must match within one sample. The count must match.
- **Spatial rule** (not `custom`): every token centre `(y, x)` of the stream must lie inside
  `crop_region(spatial, media.width, media.height)`, with `1e-6` slack.
- **Positions in the consumed span**, for every stream tagged in the positions whatever its rule: a token's
  `t * media.duration_s` (for `t >= 0`) must lie within the media span the bridge reported. For video
  that is `[min(times) - h, max(times) + h]`, `h` being half a frame (`0.5 / fps`, else
  `duration_s / (2 * n_frames)`) plus 1 ms. For audio it is
  `[min(lo) / sample_rate, max(hi) / sample_rate]` plus or minus 1 ms. Skipped when `duration_s` is null or `used`
  names no times or samples. Image tokens must carry `t = -1`.

A stream whose file holds no track (`media.has_track: false`) gets no rule, crop or position check: its only valid
report is an empty one (else `E_TIMES_OUT_OF_MEDIA`), and it does not count toward `W_RULE_UNCHECKED`.

## Codes of `yavih bridge check` and `yavih extract`

These codes are never reported by `validate_features`. `yavih bridge check` runs its smoke set through the same
per-member checks (`E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`, `E_POS_SHAPE`, `E_POS_RANGE`, `E_FINITE`, ...) plus
the checks below; `yavih extract` raises the run-time ones. Errors end the command with the exit code given.

| code | level | waivable | meaning |
|---|---|---|---|
| `E_RERUN_DRIFT` | error | yes | bridge check: encoding the same batches twice is not bit-identical (or, with `precision.deterministic: false`, differs beyond the tolerance); exit 2. A second bridge instance whose `describe()` differs is also `E_RERUN_DRIFT` and is never waivable. |
| `E_PREPARE_NONDETERMINISTIC` | error | no | bridge check: `Bridge.prepare` called twice on fresh copies of one smoke sample returned different payloads (prepare must be a pure, deterministic function of the sample); exit 2 |
| `E_CONSTANT_FEATURES` | error | yes | bridge check: every smoke clip encodes to the same features (the bridge ignores its input); exit 2 |
| `E_CONFIG_DRIFT` | error | no | extract with `--expect-profile`: the live `model`, `reading` or `describe_sha256` differs from the profile's; the message lists the `describe()` keys that differ; nothing written; exit 2 |
| `E_TEMPLATE_UNFILLED` | error | no | bridge check: `TODO` left in `model`, `weights_ref` or a stream's `norm.source`, or a null `norm`; exit 2 |
| `E_TRAIN_MODE` | error | no | a torch module is still in training mode after the bridge's `__init__`; exit 2 |
| `E_PRECISION_UNDECLARED` | error | no | a `precision` field other than `autocast` is null; exit 2 |
| `E_PRECISION_DRIFT` | error | no | extract: torch precision flags changed during `encode` from the enforced declaration; exit 2 |
| `E_PROFILE` | error | no | extract: the `--expect-profile` file is unreadable or lacks `model`, `reading` or `describe_sha256`; exit 1 |
| `E_USED` | error | no | `encode()` returned a missing or ill-formed `used`; exit 2 |
| `E_FEATURES` | error | no | `encode()` returned features or positions of the wrong shape for the declared `pos_mode`, or float16 values the writer would refuse; exit 2 |
| `E_NETWORK` | error | no | the bridge opened a network connection during `encode` or `prepare` (loopback only with `--allow-loopback`); exit 2 |
| `E_ORACLE` | error | no | bridge check refuses `yavih.testing`'s oracle bridge unless `YAVIH_TESTING=1`; exit 2 |
| `E_SMOKE` | error | no | bridge check: no present row to encode in the smoke set (data not ready); exit 3 |
| `E_SYNTHETIC` | error | no | `bridge check --synthetic`: the small stand-in benchmark cannot be built (exit 1, or 3 when a media writer is missing) |
| `E_LAYOUT` | error | no | extract: `--shard i/n` names another `n` than the split's frozen layout; run `plan build` again; exit 1 |
| `E_BRIDGE` | error | no | extract: the bridge raised inside `encode` or `prepare` (a bridge bug is not missing coverage; a `MediaUndecodable` from `prepare` naming another clip included), or raised `NotApplicable` after its first batch; exit 5 |
| `E_READER` | error | no | extract: the benchmark reader yielded rows out of order, duplicated, or not the shard's range; exit 5 |
| `E_READ_INTERNAL` | error | no | extract: reading the stored bytes raised unexpectedly; exit 5 |
| `E_WRITE` | error | no | extract: the writer could not serialize a row's provenance or features; exit 5 |
| `E_DEADLINE` | error | no | extract: `--deadline` reached; the shard is abandoned with nothing partial; exit 4 |
| `E_PREEMPTED` | error | no | extract: SIGTERM or SIGUSR1 received; the shard is abandoned after the current batch; exit 4 |
| `W_RERUN_NONDETERMINISTIC` | warning | no | bridge check: the rerun is within tolerance but not bit-identical, accepted because `precision.deterministic` is false |
| `W_NO_PREPROCESSING_SOURCE` | warning | no | bridge check: the bridge's docstring has no filled `Preprocessing source:` slot |
| `W_MISSING_ROWS` | warning | no | bridge check: some sampled rows had missing or undecodable media |
| `W_NO_CENSUS` | warning | no | plan build: a video stream guarantees `n_frames` and the site has no media-info census, so every extract unit probes each clip's header first (a demuxing pass per webm / mkv); run `yavih bench verify <b> --census` once. The plan is unchanged |

`bridge check` also reports `W_WEIGHTS_UNFINGERPRINTED`, `W_LOOSE_TOLERANCE`, `W_RULE_UNCHECKED`, `W_SIDE_EFFECT`,
`W_FP16_RANGE` and `N_RULE_CUSTOM` as in the table above, and records them in the profile. See [bridge](bridge.md)
for the passes and [execution](execution.md) for exit codes in plans.

## Waivers

A check that is wrong for a particular model is waived with a reason. It is never edited out of yavih and never
skipped silently. A waived error becomes the warning `W_WAIVED`, the split stays `ok`, and the waiver is recorded
wherever the features go.

**The bridge** declares waivers in `describe()["waive"] = {CODE: reason}`. `{"*": reason}` waives every waivable
check; an explicit code's reason beats `"*"`. The waivers are part of `describe()` and therefore of
`describe_sha256` (never of an input, protocol, probe or eval hash): changing them is a new configuration, and
`yavih extract` moves the finished shards aside. `bridge check` applies them to its own checks and lists them in
the profile's `waived`.

**An operator** waives checks on features already extracted, without changing any hash:

- `yavih features validate <dir> --waive CODE=reason` (repeatable; `all=reason` waives every waivable check),
- `yavih probe run ... --waive CODE=reason` (the validate prologue),
- the matrix entry key `waive: {CODE: reason}`, passed to the entry's probe units (see [execution](execution.md)).

These are recorded with `source: cli`. Later validations of the same configuration (same `describe_sha256`) carry
them over from `_meta.json`; a `--waive` for the same code replaces the carried reason. When the bridge and the
operator waive the same code, the bridge's waiver is the one recorded.

**What is recorded.** `_meta.json` `waived` lists every waiver in force as
`[{code, reason, source: "describe" | "cli", count}]`, `count` being the number of findings it turned into
`W_WAIVED` (a `"*"` waiver lists every waivable code, most with count 0). The record is copied into `result.json`
and the results store (see [results](results.md)). The report marks a result whose features had a waiver with
`†` next to its value and a footnote naming the code, reason, source and count. Like a `custom` rule, a waiver is
a legitimate, recorded decision: no row is greyed, excluded, flagged or penalized for it.

**Waivable codes** (`yavih.waivers.WAIVABLE`): `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`, `E_POS_RANGE`,
`E_BATCH_INVARIANCE`, `E_DEAD_ROWS`, `E_DUPLICATE_CLIPS`, `E_ARGS_UNCONSUMED` (validation and bridge check), and
`E_RERUN_DRIFT`, `E_CONSTANT_FEATURES` (bridge check). Each leaves the feature set well-formed and identifiable;
the waiver relaxes a check of the bridge's behaviour or of its declared rules.

**Never waivable** (`yavih.waivers.NOT_WAIVABLE`):

| codes | why |
|---|---|
| `E_FORMAT`, `E_SHARD_SET`, `E_RANGE_TILING`, `E_MEMBERS`, `E_ROWS`, `E_TAR_SHA`, `E_POS_SHAPE`, `E_FINITE`, `E_NA_AND_SHARDS` | structural: the shards, rows or arrays are malformed and no reader can trust them |
| `E_HASH_MANIFEST`, `E_HASH_INPUT`, `E_DESCRIBE_MISMATCH`, `E_CONFIG_DRIFT` | identity: the features are not this benchmark's input or not one consistent configuration |
| `E_TEMPLATE_UNFILLED` | the unfilled-template guard: fill the bridge from the model's own code instead |
| `E_WAIVE_INVALID` | a malformed waiver is fixed, not waived |

Any other error code (`E_NETWORK`, `E_TRAIN_MODE`, `E_USED`, ...) is not waivable either, and warnings and notes
never need a waiver. `batch_tolerance` is not a waiver: it is a declared tolerance, bounded by `1e-3`.

## Custom rules

A bridge whose sampling or crop does not fit the rule vocabulary declares `{"rule": "custom", "description": ...}`
as a stream's temporal or spatial rule (see [bridge](bridge.md)). Validation records this as the note
`N_RULE_CUSTOM` (`video: temporal rule custom — <description>; checked: out-of-media, range, positions`), never as
a warning: opting out is legitimate. `rules_checked` then records `"custom"` (temporal) or `false` (spatial), and
the rules line reads `audio custom (opted out: <description>)` where it would otherwise read, for example,
`rules checked: video uniform + spatial; audio span`.

What still runs on a custom stream: `E_TIMES_OUT_OF_MEDIA` on the reported times or sample ranges, `E_POS_RANGE`,
and the check that positions lie inside the media span the bridge reported consuming (`E_RULE_MISMATCH`).

## Thresholds

Constants of `yavih.io.validate`:

| constant | value | used by |
|---|---|---|
| `STALE_TMP_SECONDS` | 36 h | `W_STALE_TMP`; a younger tmp file counts as in progress |
| `DEAD_ROW_REL_VAR` | `1e-12` | `E_DEAD_ROWS`: variance relative to the mean token variance |
| `DUPLICATE_MAX_FRACTION` | `0.01` | `E_DUPLICATE_CLIPS`: fraction of sampled clips |
| `TIME_TOL_S` | `1e-3` s | `E_TIMES_OUT_OF_MEDIA` slack; `E_RULE_MISMATCH` time tolerance (added to the half-frame for decoded video) |
| `SPATIAL_SLACK` | `1e-6` | `E_RULE_MISMATCH`: token centres outside the crop region |
| `DEFAULT_BATCH_TOLERANCE` | `1e-5` | `E_BATCH_INVARIANCE` default tolerance; `W_LOOSE_TOLERANCE` above it |
| `MAX_BATCH_TOLERANCE` | `1e-3` | the loosest tolerance ever applied |
| `FP16_WARN_ABS` | `1e3` | `W_FP16_RANGE`: largest absolute value |
| `FP16_WARN_ZERO_FRACTION` | `0.5` | `W_FP16_RANGE`: fraction of zeros |

The canary `rel_l2` is `‖c − o‖₂ / max(‖o‖₂, 1e-12)` per clip. For float16 members it is measured net of storage
rounding: a per-value difference of at most one float16 ulp of the larger magnitude is not counted, because a
float32 drift of about `1e-6` already flips some values by one ulp once both copies are rounded to float16. Audio
rule checks allow one sample. The validation sample has 512 rows by default (`sample_rows`).
