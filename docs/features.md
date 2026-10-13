# Feature format

A bridge's features for one split of one benchmark are stored as a WebDataset: plain tar shards plus one JSON
sidecar per shard. The format is written by `yavih extract` (through `FeatureWriter`), checked by
`yavih features validate` (see [validation](validation.md)), and read by the probe (through `FeatureReader`).
It depends on nothing but numpy and the standard library's `tarfile`, so any tool that reads tars and `.npy`
files can read the features without yavih.

The current feature format is **4** (`yavih.versions.FEATURE_FORMAT_VERSION`). Writers always write format 4,
and validation, `FeatureReader`, `read_sidecar` and `read_meta` read format 4 only: any other `format_version` is
`E_FORMAT` ("unsupported format_version N (this yavih reads 4); re-extract"), and that split must be extracted
again.

## Directory layout

```
<run_dir>/features/<split>/
  shard-{i:06d}.tar         one per shard; streamed to a per-writer tmp, then os.replace'd into place
  shard-{i:06d}.json        the sidecar; written after the tar rename, so it is the shard's DONE marker
  _layout.json              the frozen shard layout of the split (written by the first extract unit)
  _meta.json                written only by finalize, when validation passes
  NOT_APPLICABLE.json       written instead of shards when the bridge does not apply to the benchmark
  shard-{i:06d}.claim       exists while a writer holds shard i
  _stale/<utc>/             shards, layouts and _meta.json of another configuration, moved aside (never deleted)
```

By convention `<run_dir>` is `runs/<benchmark>/<model>/<reading>/` (see [execution](execution.md)). Nothing
reads the benchmark, model or reading back from the path: they are recorded in every sidecar.

Every file is written to a tmp file and then renamed, so a killed task leaves only a tmp file. The tmp files of
tars, sidecars, layouts, claims and `NOT_APPLICABLE.json` are unique per writer: `<name>.<host>.<pid>.<thread>.tmp`. A shard that holds rows `[lo, hi)` with `hi == lo`
is legal (an empty shard): it has an empty tar and a sidecar with no members and null `N` / `D`.

`shard.n` (the shard count) is in every sidecar. Validation refuses a split whose sidecars name more than one
`n` (`E_SHARD_SET`).

## Tar members

Members are written in ascending row order. A member's basename is `{row:09d}`, so the stock WebDataset rule
"group by the part before the first dot" groups one row's members. Per row:

| member | content | present |
|---|---|---|
| `{row:09d}.json` | the row's provenance record (below), canonical JSON | always |
| `{row:09d}.feat.npy` | features, `[N, D]` in the declared dtype | always |
| `{row:09d}.pos.npy` | token positions, `float32 [N, 4]` | `pos_mode == "per_sample"` |
| `_pos.npy` | token positions, `float32 [N, 4]`, shared by every row | `pos_mode == "constant"`; the first member of every tar, identical across shards |
| `c{row:09d}.json`, `c{row:09d}.feat.npy` (and `.pos.npy`) | canary: `row` encoded again in another batch composition | when canaries were requested; after all regular members |

The `.npy` members are uncompressed, C-order, with a version 1.0 header
(`np.lib.format.write_array(..., allow_pickle=False)`). Tars are USTAR with deterministic headers: every
member has `mtime = 0`, `uid = gid = 0`, empty user and group names, and mode `0644`.

`N >= 1` and `D >= 1` are fixed for a feature set: every row of every split of one run directory has the
same shape. `dtype` is `float16` or `float32`. `pos_mode` is `none`, `constant` or `per_sample`.

### The per-row `.json` member

```json
{"row": 1856, "clip_id": "abc123_000010_000020", "media_key": "abc123",
 "media": {"video": {"duration_s": 10.0, "fps": 29.97, "width": 340, "height": 256, "n_frames": 300, "...": "..."}},
 "used": {"video": {"times": [0.0, 0.625, 1.25, "..."]}}}
```

| key | meaning |
|---|---|
| `row`, `clip_id` | the manifest row and its clip id; `row` equals the member name's row |
| `media_key` | the manifest's media key (rows that share it read the same media), or null |
| `media` | `{stream: MediaInfo}`: the media facts of each stream (duration, rate, frame size, track presence, ...), copied by `run_extract` from the sample's `RawStream.info`. The bridge cannot alter it. |
| `used` | `{stream: {...}}`: the bridge's report of what it consumed, one entry per stream in `describe()["input"]` (video: `times`; audio: `samples` and `pad`) |
| `canary_of` | only on canary members: the row the canary re-encodes |

Each stream's `media.duration_s` is also copied into the sidecar's `members[].duration_s`, so a readout can turn
positions into seconds without opening any member. See [bridge](bridge.md) for `MediaInfo`, the `used` entries
and the declared rules validation re-checks against them.

### Token positions

Column 0 of a position row is the stream: an index into `describe()["streams"]` stored as an exact float, or
`-1`. Columns 1 to 3 are media coordinates:

- `t` in `[0, 1]`: the token's temporal centre as a fraction of its stream's `media.duration_s` (seconds are
  `t * duration_s`). Image streams, and tokens centred in padding the bridge added beyond the media, carry
  `t = -1`.
- `y, x` in `[0, 1]`: the token's spatial centre as a fraction of the original frame's height and width
  (`media.height`, `media.width`), before any resize or crop by the bridge. `-1` when the token has no spatial position.

A row whose four columns are all `-1` is a global token. `constant` positions are legal only when one
`[N, 4]` array is correct for every row (equal durations and frame sizes, or no `t`, `y` or `x` at all);
otherwise the bridge declares `per_sample`. Details are in [bridge](bridge.md).

## Sidecar

`shard-{i:06d}.json` describes one shard completely. Its existence means the shard is finished. The example is
shard 3 of 32 of a 20 000-row split at batch size 32 (rows `[1856, 2496)`):

```json
{
 "format_version": 4,
 "yavih_version": "0.2.0",
 "benchmark": {"name": "k400", "protocol_version": 2, "manifest_hash": "…", "input_hash": "…",
               "input_components": {"name": "…", "manifest": "…", "splits": "…", "streams": "…", "…": "…"},
               "protocol_hash": "…", "scoring_hash": "…",
               "media_fingerprint": "…", "media_info_sha256": "…", "data_roots": {"media_root": "/data/kinetics400"}},
 "split": "train",
 "model": "vjepa2-l", "reading": "v",
 "shard": {"index": 3, "n": 32, "rows": [1856, 2496], "batch_size": 32},
 "rows": {"n_range": 640, "n_written": 637, "n_missing": 3, "n_canary": 8},
 "missing": [{"row": 1860, "clip_id": "…", "reason": "MediaMissing: video"}],
 "features": {"dtype": "float16", "N": 1568, "D": 1024, "pos_mode": "per_sample", "pos_sha256": null},
 "bridge": {"name": "VJepaVideoK400", "describe": {"…": "…"}, "describe_sha256": "…", "describe_schema": "yavih-describe-2",
            "source_sha256": "…", "source_path": "bridges/vjepa_video_k400.py", "args": {}},
 "tar": {"file": "shard-000003.tar", "bytes": 2066400000, "sha256": "…"},
 "members": [{"row": 1856, "clip_id": "…", "media_key": "…", "offset_data": 1536, "header_len": 128, "nbytes": 3211264,
              "duration_s": {"video": 10.0}, "pos_offset_data": 3214336, "pos_header_len": 128, "pos_nbytes": 25088}, "…"],
 "canary": {"max_abs_diff": 0.0, "max_rel_l2": 0.0, "n": 8, "tolerance": 1e-05},
 "side_effects": {"writes_outside_run_dir": []},
 "env": {"host": "gpu-node-01", "python": "3.12.4", "torch": "2.5.1", "cuda": "12.4", "gpu": "NVIDIA A100-SXM4-80GB",
         "slurm_job_id": "1234567_3", "decode_workers": 7, "read_ahead": 2,
         "precision_live": {"matmul_precision": "highest", "allow_tf32": false, "deterministic": true}},
 "timing": {"started": "2026-10-07T12:00:00Z", "finished": "…", "wall_s": 580.2, "clips_per_s": 1.11,
            "read_s": 41.3, "prepare_s": 0.0, "encode_s": 532.9},
 "canary_members": [{"row": 1900, "…": "…"}],
 "yavih": {"version": "0.2.0", "commit": "…40 hex…"},
 "claim": {"file": "shard-000003.claim", "token": "…", "lease_s": 900.0, "took_over": null}
}
```

| key | meaning |
|---|---|
| `benchmark.manifest_hash`, `benchmark.input_hash` | the shard's benchmark identity: the manifest and the input side of the protocol. A scoring edit (a metric, `probe.py`, a readout version) never stales a shard. See [hashing](hashing.md). |
| `benchmark.input_components` | `{component: sha256(canon(value))}` behind `input_hash`, so `E_HASH_INPUT` can name the component that differs |
| `benchmark.protocol_hash`, `scoring_hash` | what the benchmark was when the shard was written (provenance only) |
| `benchmark.media_fingerprint`, `media_info_sha256`, `data_roots` | copied from the site's `data_status.json`: provenance, never hashed; compared by `W_MEDIA_FP` |
| `model`, `reading` | copied from `describe()` |
| `shard.rows` | half-open `[lo, hi)`; `hi == lo` for an empty shard |
| `shard.batch_size` | the batch alignment of the split's frozen layout |
| `missing` | rows not written, with the exception class and message (see [Missing rows](#missing-rows)) |
| `features.pos_sha256` | sha256 of the serialized `_pos.npy` member (`constant` only), else null |
| `bridge.describe_sha256` | the configuration digest every shard of the feature set shares; resumption compares it. `describe_schema` names the subset it covers. See [hashing](hashing.md#describe_sha256). |
| `bridge.source_sha256`, `source_path` | the bridge file that wrote the shard (`W_SOURCE_SHA` compares it across shards) |
| `bridge.args` | the `--arg` values; each key must be echoed in `describe()["args"]` (`E_ARGS_UNCONSUMED`) |
| `tar` | file name, byte count and sha256 of the tar, computed while streaming |
| `members[]` | one entry per regular row: `offset_data` is the tar data offset of the `.feat.npy` member, `header_len` its `.npy` header length, `nbytes` the array's byte count. A reader maps the array at `offset_data + header_len` without extracting. `duration_s` is each stream's `media.duration_s` (null for images). Under `per_sample` the `pos_*` fields locate the `.pos.npy` member the same way. |
| `canary_members[]` | the same entries for the canary members |
| `canary` | the largest `max_abs_diff` and `max_rel_l2` over the canary pairs (`max_rel_l2` is net of float16 rounding, see [validation](validation.md#thresholds)), `n`, the `tolerance` (the bridge's declared or default batch tolerance), and `mismatch` (rows whose canary reported different `used` or positions; only when non-empty) |
| `side_effects` | files the bridge wrote outside the run directory during `encode` (`W_SIDE_EFFECT`); with `--allow-loopback`, also the local connections it made |
| `env`, `timing` | where and how fast the shard was written. `read_s` is the read-ahead's; `prepare_s` the time spent in `Bridge.prepare`, summed over samples across the prepare threads (0 for a bridge without `prepare`); `encode_s` includes whatever decoding the bridge does inside `encode`. |
| `yavih` | the writer's `{version, commit}`; `E_HASH_INPUT` names it next to the reader's |
| `claim` | the claim the writer held (see [Claims](#claims)) |

There is no `decoders` block: a bridge decodes its own media and declares its decoders in
`describe()["input"][stream]["decoder"]`, which is recorded and kept outside `describe_sha256`.

## `_layout.json`

The first extract unit that writes into a split freezes the split's shard layout. Every later unit, and every
later `plan build`, follows it:

```json
{
 "layout_version": 1,
 "n": 32, "batch_size": 32, "n_rows": 20000,
 "ranges": [[0, 608], [608, 1248], "…"],
 "benchmark": {"manifest_hash": "…", "input_hash": "…"},
 "model": "vjepa2-l", "reading": "v",
 "bridge": {"describe_sha256": "…", "describe_schema": "yavih-describe-2"},
 "created": "2026-10-07T11:58:00Z", "host": "gpu-node-01", "inferred_from_shards": false
}
```

`ranges` is `plan_shards(n_rows, batch_size, n)`: contiguous, batch-aligned ranges that tile `[0, n_rows)`
(see [execution](execution.md)). The layout belongs to one configuration: manifest hash, input hash, model,
reading and `describe_sha256`. The shard count and the batch size are not part of the configuration's identity,
so running `bridge check` again on another GPU, or planning with another `--shard-minutes`, never stales
finished shards.

- A split with finished shards but no `_layout.json` gets one inferred from them: the `(n, batch_size)` most of
  the configuration's shards carry (ties go to the larger `n`), with `inferred_from_shards: true`.
- The file is created atomically and only if absent. When two first writers race, one wins and the other reads
  the winner's layout.
- A unit whose `--shard i/n` names another `n` than the frozen layout is refused with `E_LAYOUT` (exit 1)
  before the model is loaded. The plan is older than the layout: run `yavih plan build` again, or move the split
  directory aside to re-shard it.
- A `_layout.json` of another configuration is moved to `_stale/` with that configuration's shards.
- A capped run (`yavih extract --rows lo:hi`, planned by `plan build --sample N`) freezes `"n": 1`,
  `"ranges": [[lo, hi]]`, `"n_rows": hi - lo` and the extra key `"sample": [lo, hi]`; a full split's layout has
  no `sample` key. `sample` is part of the identity: a full-split unit in a sampled split, a sample in a full split
  or a sample of another range is refused with `E_LAYOUT` (exit 1) before anything moves, and a sampled layout is
  never inferred from shards. `plan build --sample N` puts capped runs under their own runs root, so the refusal
  only meets a hand-run `extract`.

## `_meta.json`

`_meta.json` is derived data, written atomically by `yavih.io.finalize` and nothing else. `validate_features`
writes it when a split passes (`ok`) and `write_meta` is true. The probe reads `eligibility` and `validation.ok`
from it and never re-derives them.

```json
{
 "format_version": 4,
 "yavih": {"writer_versions": ["0.2.0"], "writer_commits": ["…"], "finalizer_version": "0.2.0", "finalizer_commit": "…"},
 "finalized": "2026-10-07T13:00:00Z",
 "benchmark": {"name": "k400", "manifest_hash": "…", "input_hash": "…", "…": "…"},
 "split": "train",
 "model": "vjepa2-l", "reading": "v",
 "bridge": {"name": "VJepaVideoK400", "describe": {"…": "…"}, "describe_sha256": "…", "describe_schema": "yavih-describe-2",
            "source_sha256": "…"},
 "features": {"dtype": "float16", "N": 1568, "D": 1024, "pos_mode": "per_sample", "pos_sha256": null, "bytes": 64225280000},
 "shards": [{"file": "shard-000000.tar", "rows": [0, 608], "n_written": 608, "bytes": 1952100000, "sha256": "…"}, "…"],
 "coverage": {"nominal": 20000, "present": 19987, "missing": 13, "ids_sha256": "…"},
 "canary": {"max_rel_l2": 3.1e-06, "n": 256, "tolerance": 1e-05},
 "rules_checked": {"video": {"temporal": "uniform", "spatial": true}},
 "dead_rows": [],
 "waived": [],
 "fingerprint": "…",
 "eligibility": {"readout": "clf", "eligible": true, "missing": [], "reason": null, "target_region": null,
                 "grid_cell_coverage": null, "t_bins": null, "stream_tags": ["video"]},
 "validation": {"ok": true, "errors": [], "warnings": [], "notes": [], "validated": "…", "yavih_version": "0.2.0"}
}
```

| key | meaning |
|---|---|
| `yavih` | `writer_versions`: the shards' `yavih_version` values; `writer_commits`: the sorted set of the shards' `yavih.commit`; the finalizer's version and commit |
| `benchmark` | the first shard's `benchmark` block, verbatim |
| `features.bytes` | bytes of all present features and positions |
| `shards` | one entry per shard in index order |
| `coverage` | `nominal` rows of the split (`hi - lo` for a capped run), `present` rows written, `missing`, and `ids_sha256 = sha256("\n".join(sorted present clip_ids))` |
| `canary` | the worst canary `max_rel_l2` over the split, the number of canary pairs, and the tolerance it was judged against |
| `rules_checked` | per stream, the temporal rule validation re-evaluated (`"custom"`: the bridge opted out) and whether the spatial rule was checked (`false`: custom; `null`: the stream kind has no spatial rule). `temporal_unchecked_rows: k` is added when the declared rule could not be evaluated on `k` rows. |
| `dead_rows` | token positions found constant across the validation sample (only non-empty when `E_DEAD_ROWS` was waived) |
| `waived` | every waiver in force: `[{code, reason, source: "describe" or "cli", count}]` (see [validation](validation.md#waivers)) |
| `fingerprint` | the `features_fingerprint` over format, dtype, `N`, `D`, `pos_mode`, `pos_sha256`, the shards' `[lo, hi, tar sha256]` in row order and the present count: integrity and staleness, never a row key. See [hashing](hashing.md#features_fingerprint). |
| `eligibility` | the readout's eligibility verdict, decided by validation and nowhere else (see [validation](validation.md#eligibility)) |
| `validation` | the verdict and the findings |
| `sample` | only for a capped run (the split's `_layout.json` has `sample`): `[lo, hi]`, the rows extracted. The probe reads it: such a result records `sample`, writes no results-store file and is a `partial` report row ([results](results.md#partial-rows)) |

`_meta.json` is rewritten only when its content changes. `finalized`, `validation.validated` and
`yavih.finalizer_commit` are ignored for that comparison, so validating an unchanged split again leaves the file
alone.

When validation fails and an earlier `_meta.json` with `validation.ok: true` exists, its `validation`,
`eligibility` and `waived` blocks are replaced by the failing verdict. The probe then refuses the split.

`FeatureReader` cross-checks `_meta.json` against the sidecars next to it. If a sidecar names another `model`,
`reading` or `describe_sha256`, or the fingerprint recomputed from the sidecars differs, it raises `FormatError`:
the shards changed after validation, so run `yavih features validate` again.

## Missing rows

A row whose media is missing or unreadable (in the reader or the read-ahead), or which the bridge reports as
undecodable (`MediaUndecodable`, or a `MediaMissing` raised by the bridge's own read), is listed in the sidecar's
`missing` list as `{row, clip_id, reason}`, where `reason` is `"<ExceptionClass>: <message>"`. The row is not
written. The bridge is never called with a row whose media could not be read. When the bridge reports rows of a
batch undecodable, those rows are dropped and the rest of the batch is encoded again without them, which gives the
same features because bridges must be batch-invariant.

Any other exception raised by the bridge aborts the shard with `E_BRIDGE` (exit 5). A bridge bug does not count as
missing coverage.

Every row of a shard's range is either written or listed in `missing`, never both. `FeatureWriter.close` refuses to
write a sidecar otherwise. Missing rows lower the split's coverage: validation warns with `W_COVERAGE`, and the
probe refuses an eval split below the benchmark's `min_eval_coverage` unless it is given `--allow-low-coverage`
(see [readouts](readouts.md)).

## `NOT_APPLICABLE.json`

When the bridge raises `NotApplicable` (from `__init__` or its first `encode`), `run_extract` writes this file
instead of shards and exits 0:

```json
{"status": "not_applicable", "reason": "audio-only bridge: benchmark has no audio stream",
 "benchmark": "k400", "split": "train", "model": "beats-iter3", "reading": "a", "describe_sha256": "…",
 "bridge": {"…": "the describe() record"}, "created": "2026-10-07T12:00:00Z"}
```

When `__init__` raised before the bridge could describe itself, `describe_sha256` and `bridge` are null or empty,
and `model` and `reading` come from the planned profile (or are null).

Before writing it, `run_extract` moves to one `_stale/<utc>/` every shard whose sidecar names another
configuration (another manifest or input hash, `model`, `reading` or `describe_sha256`), and any `_meta.json` of
the split: validation writes `_meta.json` only next to shards, so one next to `NOT_APPLICABLE.json` describes an
earlier configuration. It also deletes the split's `_layout.json`. Shards of the same configuration stay; when
`__init__` raised before `describe()`, the profile's `describe_sha256` stands in for the unknown live one.
Validation then reports `E_NA_AND_SHARDS`. Conversely, an extract unit that writes shards moves an existing
`NOT_APPLICABLE.json` aside first.

Validation treats a split holding only `NOT_APPLICABLE.json` as complete (eligibility `false`, reason
`not_applicable: <reason>`). The probe writes `result.json` with `status: not_applicable` (see
[results](results.md)) and never reads a `_meta.json` next to `NOT_APPLICABLE.json`.

## Resumption and idempotency

Running a unit `--shard i/n` again is always safe. In order, it:

1. With `--expect-profile` (always passed by `plan`), before loading the model: reads the frozen layout
   (`E_LAYOUT` on another `n`) and runs the skip check below read-only, with the profile's `model`, `reading` and
   `describe_sha256`. A finished shard costs no model load. Nothing is moved yet.
2. Loads the bridge and validates `describe()`. With `--expect-profile`, a live `model`, `reading` or
   `describe_sha256` that differs from the profile's exits 2 with `E_CONFIG_DRIFT`, listing the `describe()` keys
   that differ. Nothing is written. Run `bridge check` and `plan build` again.
3. Sweeps the split (`yavih.io.sidecar.retire_foreign`): every finished shard whose sidecar differs from the
   current run on any resumption key (below) is moved to `_stale/<utc>/`, sidecar first, with its tar. That
   configuration's `_layout.json` and a `_meta.json` of another configuration, or of a split that just lost
   shards, go with them.
4. Freezes or reads the layout, then sweeps again: a shard of the current configuration whose `shard.n` or
   `shard.rows` does not fit the frozen layout (written under an earlier plan) is moved the same way.
5. Claims the shard (see [Claims](#claims)), then deletes the shard's tmp files.
6. Skip check: when `shard-{i}.json` and its tar exist, the sidecar matches the current run on every
   resumption key, and its `shard.n` and `shard.rows` match the frozen layout, the unit exits 0 (skipped).
   A sidecar or a lone tar that does not match is moved to `_stale/<utc>/` and the shard is written again.

The resumption keys (`RESUME_MATCH_KEYS`) are `benchmark.manifest_hash`, `benchmark.input_hash`, `model`,
`reading`, `bridge.describe_sha256`, `features.dtype` and `features.pos_mode`. A shard written under another
bridge configuration (another tap, rule, precision or dtype under the same model name) therefore never mixes with
the current one. Nothing is ever deleted: foreign files are moved to `_stale/`.

In the sweeps, a tar without a sidecar and other units' tmp files are left alone, because a sibling unit may be
between its tar rename and its sidecar write. Sibling array tasks sweeping the same files concurrently skip what
another task moved first.

A killed task leaves only a tmp file. With `--deadline`, `run_extract` checks after every batch and, when the next
batch would not finish in time, abandons the shard without writing (`E_DEADLINE`, exit 4). SIGTERM and SIGUSR1
set the deadline to now: the current batch finishes, the shard is abandoned, exit 4 (`E_PREEMPTED`). Resuming
means running the missing shards again; `yavih features validate` lists them as `--shard i/n` arguments. See
[execution](execution.md) for deadlines and SLURM requeueing.

### Identity comparisons

Every identity comparison of a stored record (the skip check, the sweeps, the layout match, validation, the probe
and the plan's DONE check) reads values through `yavih.io.sidecar.identity_value(record, key)`. It applies two
rule to sidecars, `_meta.json` and `_layout.json` alike: `benchmark.input_hash` is
`hash_history.effective_input_hash(record["benchmark"])`, the recorded `input_hash` followed through the folder
changelog's `supersedes_input` (an earlier input hash whose feature sets stay valid counts as the current one);
a record without an `input_hash` yields null, which never matches (extract again). `bridge.describe_sha256` is
compared as stored. See [hashing](hashing.md#migration).

### Claims

Exactly one writer works on a shard at a time (`yavih.io.claim`). A writer:

1. creates `shard-{i:06d}.claim` only if it is absent (a hard link of a complete file, falling back to
   `O_CREAT | O_EXCL`), holding `{token, host, pid, started, lease_s}`. A heartbeat thread refreshes its mtime
   every `lease_s / 4`.
2. treats an existing claim as live unless its writer is provably gone: same host with a dead pid, or no
   heartbeat for `lease_s` (default 900 s, `YAVIH_CLAIM_LEASE_S`). A live claim refuses the unit with
   `E_SHARD_CLAIMED` (exit 3: the shard is being written elsewhere, resubmit later). A stale claim is taken over
   by renaming it aside; the takeover is recorded in the sidecar's `claim.took_over`.
3. streams the tar to its own tmp file and checks it still holds the claim before the tar rename and again
   before writing the sidecar. A claim taken over in the meantime is `E_CLAIM_LOST` (exit 4): nothing is
   published. After the rename it also checks the tar's size against the bytes it wrote.
4. removes the claim when the shard is done or abandoned.

Claims use only files, renames and mtimes (no `flock`), so they work on NFS and Lustre. Keep host clocks within
minutes of each other (NTP); a clock skewed by more than the lease defeats staleness detection.

## Size guidance

```
bytes_per_row = N * D * itemsize  (+ N * 16 with per_sample positions)
n_shards      = max(ceil(rows / rows_per_shard_time), ceil(rows * bytes_per_row / 2 GiB))
rows_per_shard_time = clips_per_s * 60 * shard_minutes      (clips_per_s from the bridge check profile)
```

`plan build` chooses `n_shards` this way when it freezes a new layout, so no shard exceeds 2 GiB; an empty split
still gets one (empty) shard. Per-sample positions cost `16 / (D * itemsize)` of the features, about 1 % at
`D = 768` in float16. Media coordinates make `per_sample` the common case for token bridges on media whose
durations or frame sizes vary.

Example totals in float16:

| split | rows x N x D | size |
|---|---|---|
| a 20 000-clip video train split, 1 568 x 1 024 tokens | 20 000 x 1 568 x 1 024 | about 64 GB |
| a 50 000-image train split, 196 x 768 tokens | 50 000 x 196 x 768 | about 15 GB |
| a 12 008-clip audio-visual eval split, 1 968 x 768 tokens | 12 008 x 1 968 x 768 | about 36 GB |
| any pooled bridge (`N = 1`) | | under 100 MB |

When the features of a split exceed the probe's RAM budget, validation warns `W_SIZE` and the probe reads through
memmaps; stage the features to local scratch with `yavih features stage` (see [execution](execution.md)).

## `FeatureWriter`

```python
from yavih.io import FeatureWriter

class FeatureWriter:
    def __init__(self, out_dir, split: str, *, shard: tuple[int, int], rows: tuple[int, int],
                 bench: Benchmark, bridge: Bridge, describe: dict, source_sha256: str,
                 claim: Claim | None = None, batch_size: int | None = None) -> None
    def write(self, rows: list[int], clip_ids: list[str], feats, pos=None,
              meta: list[dict] | None = None, canary: bool = False) -> None
    def record_missing(self, row: int, clip_id: str, reason: str) -> None
    def close(self, *, timing: dict, env: dict, canary_stats: dict, side_effects: dict | None = None) -> Path
    def abort(self) -> None
    # context manager: leaving the block without close(), or with an exception, calls abort()
```

`out_dir` is the run directory (or the split directory itself); `shard = (i, n)`; `rows = (lo, hi)` half-open.
`model` and `reading` are taken from `describe`, and `describe_sha256` is computed from it. `claim` is the shard's
`Claim` (verified before publishing; `None` for tests and single-process use). `batch_size` is the frozen layout's
batch alignment recorded in `shard.batch_size` (default: the bridge's).

`write` takes one batch, `feats` as `[B, N, D]` (an ndarray, or a tensor converted with
`.detach().float().cpu().numpy()`), and `meta[i] = {"media_key", "media", "used"}` per row. It:

- requires `len(rows) == len(clip_ids) == feats.shape[0]` (and `len(meta)`);
- fixes `(N, D)` from the first call and raises `FeatureShapeError(clip_id, got, expected)` on any other shape;
- requires regular rows strictly ascending, inside `[lo, hi)`, not yet written or missing; canaries only after
  all regular rows, and only for rows already written;
- checks positions against `pos_mode`: none for `none`; for `constant`, `[N, 4]` (or a per-sample array whose
  rows are all equal) on the first call, identical afterwards; for `per_sample`, `[B, N, 4]`;
- raises `FeatureMetaError(clip_id, ...)` when a row's `meta` lacks `media` or a `used` entry for a stream in
  `describe()["input"]`;
- raises `FeatureValueError(clip_id, ...)` on non-finite values, and for float16 when the largest absolute value
  exceeds `1e4` (the message says to declare float32), then casts to the declared dtype;
- appends the `.json`, `.feat.npy` and (per-sample) `.pos.npy` members, updating the tar sha256 as it streams.

`close` refuses (and aborts) unless every row of the range is written or missing. It then closes the tar, verifies
the claim, renames the tmp tar into place, checks the tar's size, verifies the claim again and writes the sidecar
atomically; it returns the sidecar path. `abort` removes the tmp tar; nothing partial is left behind.

## `FeatureReader`

```python
from yavih.io import FeatureReader

class FeatureReader:
    def __init__(self, features_dir, *, require_meta: bool = True) -> None   # a split directory
    meta: dict; N: int; D: int; dtype: np.dtype; n_rows: int; pos_mode: str
    present_rows: np.ndarray            # int64 manifest rows present, ascending
    def iter(self, *, with_pos: bool = False) -> Iterator[tuple[int, np.ndarray, np.ndarray | None]]
    def rows(self, idx) -> np.ndarray   # [len(idx), N, D], stored dtype, gathered by manifest row
    def pos(self, row: int | None = None) -> np.ndarray | None
    def durations(self, idx, stream: str) -> np.ndarray   # float64 media.duration_s per row; NaN when null
    def matrix(self, *, ram_budget_bytes: int | None = None) -> RowView
    def clip_ids(self) -> list[str]
    def media_keys(self) -> list[str | None]
    def canaries(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]   # (row, canary, original)
    def fingerprint(self) -> str
    def close(self) -> None             # also a context manager
```

- With `require_meta=True` (the default) a split without `_meta.json` is refused with `FormatError`: run
  `yavih features validate` first. `require_meta=False` reads raw shards.
- `rows(idx)` memory-maps each tar once (read-only) and returns the arrays at the sidecars' offsets without
  extracting. A row that is not present raises `KeyError`.
- `iter()` streams the tars with `tarfile`, independently of the sidecar offsets, in row order.
- `pos()` is `None` for `pos_mode none`, the shared `[N, 4]` for `constant`, and for `per_sample` the row's
  `[N, 4]` (or `[n_present, N, 4]` when `row` is `None`).
- `matrix()` returns a `RowView` backed by one contiguous RAM array when the features fit the budget, else by
  memmaps. Values and row order are identical in both backends. A `RowView` supports `len`,
  `view[i]` and `view[array]` (indices are positions into `present_rows`, not manifest rows), `.shape`,
  `.present_rows` and `.backend` (`"ram"` or `"memmap"`). Stored float16 stays float16 in RAM; the probe upcasts
  per mini-batch. The RAM array is read once per reader and cached (read-only; `view[array]` returns a copy), so
  repeated `matrix()` calls share it; a budget that selects memmaps drops it.
- The default budget is `YAVIH_RAM_BUDGET_GB` (or `[resources] ram_budget_gb`), else
  `min(16 GiB, 0.45 x the job's memory limit)` (SLURM's or the cgroup's), else 16 GiB. See
  [configuration](configuration.md).

## Other tools

| command | what it does |
|---|---|
| `yavih features validate <dir>` | checks a split (or every split under `features/`) and writes `_meta.json`; see [validation](validation.md) |
| `yavih features inspect <dir>` | diagnostics over the validation sample: per-dimension std histogram, near-constant dimensions, mean absolute value, position summary, dead token positions. It decides nothing. |
| `yavih features diff <a> <b>` | compares the rows present in two split directories: `rows_compared`, `max_abs`, `max_rel_l2` and a verdict (`identical`, `within_tol`, `differs`, `incompatible`, `empty`) |
| `yavih features stage <features/> <scratch>` | copies tars, sidecars, `_meta.json` and `NOT_APPLICABLE.json` (never tmp files) to local scratch for `probe run --scratch`, after checking free space (exit 3 when short) |
