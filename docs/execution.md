# Execution

How a set of `(bridge, benchmark)` pairs becomes features and results: the run is described in a
`matrix.yaml`, `yavih plan build` turns it into unit records (`plan.jsonl`), and every unit is run by
`yavih exec`, on this machine (`yavih plan run --local`) or as one task of a scheduler array that you
write. yavih is scheduler-agnostic: it never writes or submits batch scripts; it gives every unit
its resource needs, a deadline-aware runner, resumable outputs and exit codes a driver loop can act
on.

```
matrix.yaml ──plan build──▶ plan.jsonl ──exec / plan run --local──▶ features/, probe/, _report/
                               ▲                                          │
                               └──── plan build --pending, plan status ◀──┘ (DONE markers, logs/last.json)
```

## Sharding

A split is extracted in shards; one shard is one extract unit, scheduled independently.

### `plan_shards`

```python
yavih.io.shards.plan_shards(n_rows: int, batch_size: int, n_shards: int) -> list[tuple[int, int]]
```

returns `n_shards` half-open row ranges `[lo, hi)` that are ascending, disjoint and tile
`[0, n_rows)`:

- every `lo` is a multiple of `batch_size`: global batch `j` is always rows `[j·B, (j+1)·B)`, so any
  shard count replays exactly the same batches and an encoder that is not batch-invariant still
  sees identical inputs;
- batches are dealt as evenly as possible: shard `i` starts at batch `(i · n_batches) // n_shards`;
- the single short final batch belongs to the last non-empty shard;
- when `n_shards > n_batches` some ranges are empty (`hi == lo`); an empty shard still gets a sidecar,
  with no tar members.

| call | ranges |
|---|---|
| `plan_shards(1000, 32, 4)` | `(0, 256) (256, 512) (512, 768) (768, 1000)` |
| `plan_shards(100, 8, 3)` | `(0, 32) (32, 64) (64, 100)` |
| `plan_shards(10, 4, 5)` | `(0, 0) (0, 4) (4, 4) (4, 8) (8, 10)` |

The benchmark reader `reader(split, shard=(i, n), batch_size=B)` yields the rows of range `i` in
ascending order; the same `(manifest, split, shard, batch_size)` gives identical rows and metadata on
every host, and a missing media file never shifts rows (it becomes a missing row). Ranges are
contiguous on purpose: every shard is one independently scheduled task, so the scheduler absorbs a
heavy shard, and contiguity keeps tar-index page-cache locality. Benchmarks whose manifest is grouped
(folds, classes) can make some shards heavier than the mean throughput predicts; the 1.5× margin in
the time estimate and `plan calibrate` absorb that.

### Shard count

Before a split's layout is frozen, `plan build` chooses `n` per split as

```
n = max(1, ceil(rows / (clips_per_s · 60 · shard_minutes)), ceil(rows · bytes_per_row / 2 GiB),
           ceil(rows / max_rows_per_shard))
```

with `clips_per_s` from the bridge's profile, `shard_minutes` from the matrix (or `--shard-minutes`),
`bytes_per_row = N · D · itemsize` (+ `16 · N` for `per_sample` positions) and `max_rows_per_shard`
from the matrix entry, else the matrix (default 10 000: it bounds what one unit holds and what a
preempted unit loses, whatever the measured speed). An empty split still gets one (empty) shard. A
matrix entry's `shards` key pins `n` instead; a frozen layout wins over both.

### The frozen layout (`_layout.json`)

The first extract unit that writes into `features/<split>/` freezes the split's layout in
`features/<split>/_layout.json`: `n`, the batch alignment (`batch_size`), `n_rows`, every shard's
`[lo, hi)` range, and the configuration it belongs to (`benchmark.manifest_hash`,
`benchmark.input_hash`, `model`, `reading`, `bridge.describe_sha256`), with `created` and `host`.
The file is written create-if-absent (a hard link of a complete temporary file, else `O_EXCL`), so of
two first writers on two hosts one wins and the other reads the winner's layout.

From then on:

- `plan build` plans exactly that layout, whatever the profile's `clips_per_s` (another GPU type) or
  `shard_minutes` (another partition) would give, so a re-run `bridge check` or two clusters sharing
  one run directory never invalidate finished shards;
- `yavih extract` reads its row range from the file; a `--shard i/m` with another `m` is refused with
  `E_LAYOUT` (exit 1: the unit comes from a plan built before the freeze; re-run `plan build`);
- a matrix `shards` pin that disagrees with the frozen `n` is refused by `plan build` (exit 1);
  re-sharding a split means moving its directory aside first;
- when the layout's alignment differs from the bridge's `batch_size`, the reader is called with
  `align=` the layout's alignment.

A split without `_layout.json` gets the layout most of its finished shards of the
current configuration carry (ties: the larger `n`), frozen on first use. Only another configuration
(another model, reading, `describe_sha256` or benchmark input side) owns another layout: its shards
and its `_layout.json` are moved to `_stale/<utc>/` together before the new configuration writes.
A capped run (`extract --rows lo:hi`) freezes one shard over `[lo, hi)` with `sample: [lo, hi]`; the sample is
part of the layout's identity, so a full-split unit in a sampled split directory, a sample in a full one, or a
sample of another range is refused with `E_LAYOUT` before anything moves ([capped runs](#capped-runs-plan-build---sample-n)).
The shard files themselves are described in [features](features.md).

### One writer per shard: claims

Two tasks can land on one shard (a requeue while the first task still runs, a pod restart, a stale
`running` record resubmitted). Before writing, an extract unit takes the shard's claim
`features/<split>/shard-{i:06d}.claim`:

1. the claim is created atomically and holds `{token, host, pid, started, lease_s}`; a heartbeat
   thread refreshes its mtime every `lease_s / 4`;
2. an existing claim is live unless its writer is provably gone (same host and a dead pid, or no
   heartbeat for `lease_s`, default 900 s, `YAVIH_CLAIM_LEASE_S`). A live claim → `E_SHARD_CLAIMED`,
   exit 3 (resubmit later). A stale claim is taken over by an atomic rename;
3. the tar is written to a per-writer temporary file, and the writer checks it still holds the claim
   before the rename into place and before the sidecar; if another writer took over meanwhile →
   `E_CLAIM_LOST`, exit 4, nothing published;
4. the claim is removed when the shard is done or abandoned.

Only files, renames and mtimes are used (no `flock`), so claims work on NFS and Lustre. Keep host
clocks within minutes of each other (NTP), or raise the lease.

## Byte read-ahead and decode workers

Decoding belongs to the bridge ([bridge](bridge.md)); yavih hides storage latency and, for a bridge
with `prepare`, decoding time. While the bridge encodes batch `b`, a torch-free thread pool of
`--decode-workers` threads reads the stored bytes of every stream and frame of batches
`b+1 … b+read_ahead` (`--read-ahead`, default 2) and resolves their media facts. A read failure
(absent file, short read, a header that cannot supply a guaranteed fact) turns that row into a missing
row before the bridge sees it.

A bridge that defines `Bridge.prepare` gets a second pool of `--decode-workers` threads
(`yavih-prepare-*`): every Sample of a batch the read-ahead delivered is passed to `prepare`, up to
`read_ahead + 1` batches ahead of `encode`, so decoding and preprocessing overlap the forward pass. A
bridge without `prepare` decodes inside `encode` and gets no such pool. `run_extract` drops a batch's
stored bytes once its rows are written, and right after `prepare` unless the bridge sets
`keep_bytes = True` ([bridge](bridge.md#prepare)), so a shard holds at most the batches in flight.
The `yavih.media.decode` helpers decode from those bytes in memory, never reading the store twice.

The same count is the bridge's CPU budget: `Bridge.decode_workers` is set before the first `encode`;
it is the number of prepare threads, and a bridge that decodes inside `encode` may parallelize its own
decoding with it. The default is

- `YAVIH_DECODE_WORKERS` / `[resources] decode_workers` when set;
- else `CPUs − 1` when the CPUs are an allocation (`YAVIH_CPUS` / `[resources] cpus`, SLURM, PBS,
  an affinity mask or a cgroup quota), `CPUs // 2` on a whole machine;
- capped at 16 per task, at least 1.

A planned extract unit passes `--decode-workers <cpus − 1>` with the matrix entry's `cpus`; `exec`
rewrites it to `N − 1` under `--cpus N`, a slot's share, or when the process may use fewer CPUs than
planned. The count is recorded in the sidecar (`env.decode_workers`, `env.read_ahead`,
`timing.read_s`, `timing.prepare_s`), and `bridge check` measures `clips_per_s` at the same default, so pass
`bridge check --decode-workers <cpus − 1>` when the profile is measured elsewhere. Any comparison of
extraction speed must use equal worker counts.

## `yavih plan init`

`yavih plan init --suite S --model M [--launcher PYTHON|NAME] [--out matrix.yaml] [--bridges-dir DIR] [--force]`
writes the starting point for running one model on a [suite](suites.md). In a project
([project layout](project-layout.md)) the matrix goes to `yavih-bench/matrices/<M>.yaml` and the bridges to
`yavih-bench/bridges/`; without one, to `./matrix.yaml` and `./bridges/`:

- one bridge file per packaged suite benchmark, `<bridges-dir>/<M>_<benchmark>.py`: a copy of the folder's
  `bridge_template.py` with the model name filled in (`yavih bridge new`). A template that takes the reading as an
  argument (`vggsounder`, `audioset20k`, `avsync`) serves every reading from that one file. An existing bridge
  file is kept, never overwritten;
- the matrix in the layout [below](#matrixyaml): `defaults.launcher` = `--launcher` (a python path or a
  `[launchers]` name), or `null` when omitted so the configured launcher applies on whichever site runs the units
  (`W_LAUNCHER_UNSET` when no default `launcher` is configured either, at the top level or in any `[site.<id>]`
  table; a `[launchers]` name alone does not count, since `null` never resolves to one); `out: null` (the
  configured runs root); and one
  entry `<M>-<benchmark>-<reading>` per suite `(benchmark, reading)` with `script`, `benchmark`, `args: {reading:
  <r>}` for a multi-reading template, `profile: <matrix stem>.profiles/<M>-<benchmark>-<reading>.profile.json`
  (profiles belong to matrix entries and sit next to their matrix) and a commented `waive:` line. A join entry adds
  a comment with its `report build --join` command; pending suite entries are listed as comments. An existing
  matrix is refused (`E_EXISTS`, exit 1) unless `--force`.

The output lists what to run next: one `yavih bridge check <script> --bench <benchmark> [--arg reading=<r>] --out
<profile>` per entry (after the bridge's TODOs are filled, on the staged data), then `yavih plan build <matrix>
--out <runs root>/_plans/<M>.jsonl` (`plan` in `--json`). A plan file is run state, so it sits under the runs
root (by default `yavih-bench/runs`, which the project's `.gitignore` covers), never among the committed files; the
plan stores each unit's `cwd` relative to itself, so it runs from there. Until every profile exists `plan build` refuses the matrix
(`E_PROFILE`).

In the commands below, `matrix.yaml` stands for the matrix (`yavih-bench/matrices/<M>.yaml` in a project) and
`plan.jsonl` / `pending.<kind>.jsonl` for plan files under `<runs root>/_plans/` (`yavih config get runs.root`).

## `matrix.yaml`

```yaml
schema_version: 1
out: null                           # runs root (a path is relative to this file); null = the configured runs root
results: null                       # results-store root for the probe units; null = the usual resolution
seeds: [0]                          # one probe unit per seed
shard_minutes: 30                   # target wall per extract unit (--shard-minutes overrides)
max_rows_per_shard: 10000           # rows cap of the shard-count estimate
canaries: 8                         # canary rows re-encoded per shard
defaults:                           # merged into every entry
  launcher: "/envs/vjepa/bin/python"  # or a [launchers] name, or null = the configured launcher of the site
  env: {HF_HUB_OFFLINE: "1", OMP_NUM_THREADS: "8"}
  cpus: 8                           # decode workers = cpus − 1
  args: {}
  allow_loopback: false
bridges:                            # one entry per (bridge, benchmark)
  vjepa-video-k400:
    script: bridges/vjepa_video_k400.py
    benchmark: k400                 # one name or alias
    args: {layer: 12}
    profile: matrix.profiles/vjepa-video-k400.profile.json   # from `yavih bridge check`; required
    # shards: {train: 64}           # pin n for a split with no frozen layout (an int = every split)
    # max_rows_per_shard: 5000      # this entry's rows cap (default: the matrix's)
    # allow_loopback: true          # extract with --allow-loopback (a local inference server)
    # waive: {E_DUPLICATE_CLIPS: "digitally silent audio"}   # operator waivers for the probe's validate
```

| key | rule |
|---|---|
| `schema_version` | must be 1 |
| `out` | non-empty path; default the configured runs root ([configuration](configuration.md)) |
| `results` | `null` or a non-empty path; when set, every probe unit gets `--results <path>` |
| `seeds` | non-empty list of distinct ints (default `[0]`) |
| `shard_minutes` | int ≥ 1 (default 30) |
| `max_rows_per_shard` | int ≥ 1 (default 10 000); the rows cap of the shard-count estimate |
| `canaries` | int ≥ 0 (default 8) |
| `defaults` | `launcher`, `env`, `cpus`, `args`, `allow_loopback` |
| `bridges.<key>` | key matches `[A-Za-z0-9_.-]+`; `script`, `benchmark`, `profile` required; also `args`, `launcher`, `env`, `cpus`, `shards`, `max_rows_per_shard`, `allow_loopback`, `waive` |

Unknown keys are an error (`E_MATRIX`, exit 1). Entry values override `defaults` (`env` and `args`
are merged key by key). `env` values are coerced to strings. Relative `script`, `profile`, `out` and
`results` paths are relative to the matrix file's directory.

`launcher` is how the model environment's Python is started. `yavih exec` resolves it when the unit
runs (`yavih.plan.exec.resolve_launcher`): a value with a path separator or whitespace is the command
itself; a bare name is the `[launchers] <name>` entry of the active configuration (per site; [configuration](configuration.md#sites)),
else the name when it is a command on PATH; a name that is neither records the unit `misconfigured` (exit 1) before
anything starts, with a `launcher: …` message naming the known entries, shown by `plan status` and kept in the
unit's `<out>/logs/last.<id>.json`. `null` (or no `launcher` at all) is the configured default `launcher`, else
the interpreter running `exec` (`sys.executable`, never whatever `python` is on `PATH`). If the basename of the command's
last token matches `python*`, it replaces `cmd[0]` (`/envs/vjepa/bin/python`, `conda run -n vjepa python`,
`apptainer exec --nv --bind /e model.sif python`); otherwise it is a prefix to the whole command.
`script` may also be `module:Class` for a bridge installed in the model environment
([bridge](bridge.md)). Probe and report units have no launcher: they run inside the process running
`yavih exec`.

`waive: {CODE: reason}` gives every probe unit of the entry `--waive CODE=reason` for its validate
prologue ([validation](validation.md)); a non-waivable code or a missing reason is `E_MATRIX`. The
waiver is recorded in the unit's `cmd` but changes no unit id or hash.

`plan build` refuses (exit 1, `E_PROFILE`) an entry whose profile names another benchmark, records
other `args` than the entry's resolved `args`, fails the describe schema, lacks passed hard
invariance checks (a check the bridge's own `describe()["waive"]` waives may exceed the tolerance),
lacks `model` / `reading`, or disagrees with its own `describe` (`describe_sha256`, `model`,
`reading`, `dtype`, `pos_mode`, `streams`). `model` and `reading` are taken from the profile, never
computed. Two entries resolving to the same `(benchmark, model, reading)`, compared ignoring case,
would share one run directory and are refused (`E_PLAN`); a layer sweep is several model names.

## Unit records (`plan.jsonl`)

One JSON object per line, sorted keys:

```json
{"schema_version": 3, "id": "9f2a1c0b7e3d", "kind": "extract",
 "benchmark": "k400", "bridge": "vjepa-video-k400", "model": "vjepa2-l", "reading": "v",
 "split": "train", "shard": [3, 32], "seed": null,
 "protocol_hash": "…", "input_hash": "…", "probe_hash": null, "eval_hash": null, "describe_sha256": "…",
 "depends_on": [],
 "needs": {"gpu": 1, "cpus": 8, "mem_gb": 48, "scratch_gb": 0, "minutes": 15},
 "cmd": ["python", "-m", "yavih.extract", "--bench", "k400", "--bridge", "bridges/vjepa_video_k400.py",
         "--split", "train", "--shard", "3/32", "--out", "<runs root>/k400/vjepa2-l/v", "--arg", "layer=12",
         "--canaries", "8", "--decode-workers", "7", "--expect-profile", "matrix.profiles/vjepa-video-k400.profile.json"],
 "launcher": "/envs/vjepa/bin/python", "env": {"HF_HUB_OFFLINE": "1"}, "cwd": ".",
 "out": "<runs root>/k400/vjepa2-l/v/features/train",
 "done": "<runs root>/k400/vjepa2-l/v/features/train/shard-000003.json",
 "inputs_hash": "…", "expected_status": "ok",
 "profile": "matrix.profiles/vjepa-video-k400.profile.json", "not_applicable": null}
```

| kind | one per | `cmd` | `out` | `depends_on` |
|---|---|---|---|---|
| `extract` | entry × split of the benchmark's `train`, `eval`, `floor` and `folds` roles × shard | `python -m yavih.extract …` | `<run_dir>/features/<split>` | none |
| `probe` | entry × seed | `python -m yavih.cli probe run <run_dir> --seeds <k> [--results R] [--waive …]` | `<run_dir>` | every extract unit of the entry |
| `report` | plan | `python -m yavih.cli report build '<out>/*' --fmt md --out <out>/_report/report.md` | `<out>/_report` | every probe unit |

- The run directory is `<out>/<benchmark>/<model>/<reading>/` with the canonical benchmark name.
- `id`, `inputs_hash` and the hashes are defined in [hashing](hashing.md). An extract unit's id,
  `inputs_hash` and DONE check key on the benchmark's `input_hash`, so a scoring-side edit of a
  benchmark re-plans its probes and no extraction. `inputs_hash` covers only what a unit computes:
  host paths, the launcher, `env`, `cwd`, worker counts, `--results`, `--waive` and `needs` are left
  out, so moving the project or recalibrating never makes a finished unit stale.
- A probe unit carries `protocol_hash`, `probe_hash` and `eval_hash`; its `done` is
  `<run_dir>/probe/<eval_hash12>/seed<k>/DONE`.
- `expected_status: not_applicable` when the profile already shows the benchmark cannot be served (a
  dense readout against `N == 1` or `pos_mode: none`; a required stream the bridge does not produce;
  `not_applicable` holds `{missing, reason, decided_by}`). No extract units are planned; the probe
  unit writes the `not_applicable` result directly.
- Probe units are not split per fold or control arm: where the probe dominates, the feature sets are
  small enough that the serial loop is cheaper than scheduling overhead.
- Units are written topologically (extract, probe, report) and, within a kind, grouped by
  `needs.gpu` (GPU units first) and a minutes bucket (15, 30, 60, 120, 240, 480, 720, 1440), keeping
  matrix, split and shard order inside a group, so a plan is deterministic.
- `write_plan` stores `cwd` relative to the plan file and `read_plan` anchors it there again (or at
  `--root DIR`), so a plan survives the project being mounted elsewhere; `out` and `done` (under the
  runs root) are stored absolute and are not re-anchored.
- A record of an older plan schema is refused (`E_PLAN`: rebuild the plan with `plan build`;
  finished shards are kept because they are checked by input hash); a newer one asks for a newer
  yavih.

### `yavih plan build`

```bash
yavih plan build matrix.yaml --out plan.jsonl                       # every unit
yavih plan build matrix.yaml --pending --kind probe --out pending.probe.jsonl
yavih plan build matrix.yaml --pending --kind extract --format count   # prints the number
```

`--pending` keeps only units whose DONE marker is missing or does not match, or whose last record's
`inputs_hash` differs from the unit's. `--kind` filters after that. With `--out`, `--format count
--json` adds the file's `sha256` for `exec --plan-sha`.

`--sample N` plans a capped run ([below](#capped-runs-plan-build---sample-n)).

`W_NO_CENSUS` (stderr and the `--json` envelope's `warnings`, once per benchmark) means a video stream
guarantees `n_frames` and this site has no media-info census: every extract unit then probes each
clip's header before decoding it, a full demuxing pass for webm / mkv. Run `yavih bench verify <b>
--census` once; the warning never changes a unit.

### Capped runs: `plan build --sample N`

`yavih plan build matrix.yaml --sample N --out plan.jsonl` plans the whole matrix on the first `N` rows of every
role split, to try the chain (a new model, a new benchmark folder, a cluster) end to end before the full run:

- the rows are the first `min(N, rows)` of each split in manifest order. The sample is not class-balanced or
  shuffled: on a manifest sorted by class it may hold one class, so its numbers say that the chain runs, not how
  good the model is;
- one extract unit per split, `yavih extract … --rows 0:k` (no `--shard`), written as shard `0/1` of a layout
  frozen with `sample: [0, k]`; `validate` measures coverage against `k` and records `sample` in `_meta.json`;
- the probe units carry `--no-results`, and the probe writes no results-store file for a sampled feature set
  whatever its flags; `result.json` records `sample: {split: [lo, hi]}` and every report row from it is marked
  `partial` ([results](results.md#partial-rows));
- everything lives under the separate runs root `<out>-sample<N>` (`runs` → `runs-sample4`), with its own
  `_report/`: a capped run never shares a run directory, a DONE marker, an index or a report with the full run,
  and the full plan of the same matrix later starts from nothing;
- unit ids are derived from the full run's ids and `N`: deterministic, and never equal to a full-run id.

### Resource estimates (`needs`)

`needs = {gpu, cpus, mem_gb, scratch_gb, minutes}` is what one unit asks for:

| kind | `gpu` | `cpus` | `mem_gb` | `scratch_gb` | `minutes` |
|---|---|---|---|---|---|
| extract | 1 (0 for a CPU-only profile) | entry `cpus` | `16 + 4 · cpus` | 0 | `ceil(1.5 · rows_in_shard / clips_per_s / 60)`, at least 1 |
| probe | as extract | entry `cpus` | `ceil(max(16, 1.3 · bytes_train / 2³⁰ + 8))` | `ceil(1.1 · bytes_train / 2³⁰)` when `bytes_train` exceeds the RAM budget, else 0 | `probe_minutes_ref · clamp(N · D / (16 · 768), 1, 24) · fits`, at least 30 |
| probe (planned `not_applicable`) | 0 | 1 | 4 | 0 | 5 |
| report | 0 | 2 | 8 | 0 | 30 |

`clips_per_s`, `N`, `D` and `peak_gpu_gb` come from the profile (a CPU-only profile has
`peak_gpu_gb == 0`; an unknown peak keeps `gpu: 1`). `bytes_train` is the row count of the train and
fold splits times `bytes_per_row`; the RAM budget is the planning host's ([configuration](configuration.md)).
`probe_minutes_ref` is the benchmark's `budget.probe_minutes_ref` for its readout, and `fits` is 2 when
the readout trains a shuffled control as a second full fit, else 1. Both `minutes` formulas are
multiplied by the calibration multipliers and per-unit factors of `plan calibrate` (below) before
rounding up.

### `yavih plan calibrate`

`yavih plan calibrate <runs_root>` scans the runs root (skipping `_stale/`) for unit records, probe
`probe_fit.json` files and finished `result.json` timings, and writes `<runs_root>/_calibration.json`,
which the next `plan build` of a matrix whose `out` is that root applies:

- **multipliers** `extract_minutes` and `probe_minutes`: the 90th percentile of observed wall /
  planned minutes over finished units of the kind, compounded with the previous value and clamped to
  `[0.5, 8]`;
- **per-unit bumps**: `minutes × 1.5` for a unit whose last attempt was stopped by its own deadline
  while running (`status: deadline`, `wall_s > 0`, not `refused_at_start`), once per new attempt, so
  calibrating twice does not double it. A start refusal or a preemption says nothing about a unit's
  length and is never bumped. This keeps a heavy shard from hitting the same wall twice.

## `yavih exec`

```
yavih exec plan.jsonl (--index N | --id X | --range a:b) [--deps plan.jsonl]
           [--deadline D] [--deadline-margin 900] [--slots K] [--root DIR] [--cpus N] [--plan-sha HEX]
```

For one unit, `exec`:

1. reads the plan (checks the schema; `--root DIR` replaces every unit's `cwd`, the only relative
   path a plan stores, with the directory the project is mounted at on this machine) and, with `--plan-sha HEX`, refuses (exit 1) a plan
   file whose sha256 does not start with `HEX` (8 to 64 hex digits);
2. checks dependencies: every `depends_on` unit must have a matching DONE marker, else exit 3
   (`not_ready`, nothing written). `--deps plan.jsonl` names the full plan used only to look up
   dependencies: a `--pending --kind probe` file holds no extract units. The selection still comes
   from the first file;
3. checks its own DONE marker: when it matches, exit 0 with `status: skipped`;
4. refuses to start when less than `max(10 min, 0.25 · needs.minutes)` remains before the deadline:
   exit 4, `status: deadline`, `refused_at_start: true`;
5. writes the provisional record `<out>/logs/last.<id>.json` with `status: running`, `host`, `pid`,
   `started_epoch` and `deadline`, so `plan status` can tell a running unit from one that died
   without a final record;
6. runs the unit:
   - **extract**: the launcher applied to `cmd`, started in `cwd` with
     `os.environ | unit.env | {YAVIH_UNIT_ID, YAVIH_DEADLINE}` (plus `YAVIH_CPUS` when the CPU budget
     differs from the plan, and the slot's GPU mask). Output is teed to stderr and to
     `<out>/logs/<id>.<job>.<utc>.log` (`<job>` = `<array job>_<task>` in a SLURM array, the job id in
     a job, else the pid). SIGTERM, SIGINT and SIGUSR1 are forwarded to the child;
   - **probe** and **report**: in this process (so the environment running `exec` needs
     `yavih[probe]`). The same signals set the probe's stop flag; the unit log holds a header and the
     full traceback of any failure, and a failed unit's message ends with `(traceback: <log>)`. Before
     a probe with `needs.scratch_gb > 0` the scratch root (`YAVIH_SCRATCH` → `[scratch] root` →
     `SLURM_TMPDIR`) is checked for space (exit 3 when short) and stale `<unit_id>.<host>.<pid>`
     scratch directories (dead pid, or older than 36 h) are swept. Without a scratch root the probe
     reads the features in place (memmap backend, `W_SIZE`), as `probe run` does;
7. maps the outcome to a status and an exit code, and writes the final record to
   `<out>/logs/last.json` and `<out>/logs/last.<id>.json`; it prints
   `[yavih] unit <id> <kind> <status> wall_s=… exit=N` on stderr.

| status | exit | when |
|---|---|---|
| `done` | 0 | the unit wrote a matching DONE marker |
| `skipped` | 0 | the DONE marker already matched |
| `not_applicable` | 0 | `NOT_APPLICABLE.json` or a `not_applicable` result |
| `misconfigured` | 1 | the child (or in-process unit) exited 1: a usage or configuration error no resubmission heals (`E_LAYOUT`, a bad `--deadline`); also a bare launcher name that is neither a `[launchers]` entry nor a command on `PATH`, recorded before the unit starts |
| `not_ready` | 3 | a dependency is not done, data or scratch is missing, the shard is claimed |
| `deadline` | 4 | the unit's deadline stopped it, or it was refused at start; an in-process probe stopped by a forwarded signal is recorded here too |
| `preempted` | 4 | a signal stopped it: a child killed by a signal (negative return code or 128 + n), a child that exited 4 after a forwarded signal, an in-process unit that failed or returned without DONE after a signal |
| `failed` | the child's code (2 or 5), else 5 | anything else, including a child that exited 0 without a matching DONE |

`yavih exec` imports only the standard library for extract units, so it can run from a small
environment while the launcher starts the model environment.

### Deadlines and margins

`--deadline` takes the forms listed in [cli](cli.md#--deadline). `--deadline slurm` is
`SLURM_JOB_END_TIME − --deadline-margin` (default 900 s); inside a SLURM job without a usable
`SLURM_JOB_END_TIME` it is `E_DEADLINE_SPEC` (exit 1), never "now"; outside SLURM it means no
deadline. `exec` exports the resolved epoch as `YAVIH_DEADLINE`, which `extract` and `probe run`
inherit:

- an extract checks after every batch whether the next batch (estimated from the batches so far)
  would overrun, and abandons the shard (exit 4, `E_DEADLINE`) with only its temporary file, which is
  removed; SIGTERM and SIGUSR1 do the same after the current batch (`E_PREEMPTED`);
- a probe stops between epochs with `probe_ckpt.pt` written and resumes from it on the next run.

The 900 s default is larger than the 600 s signal lead of the SLURM example below, so a unit stops on
its own clock, with a checkpoint, before the scheduler's TERM arrives.

### Ranges and slots

- `--range a:b` runs units `a … b−1` in plan order in this process (`a:` to the end, `:b` from 0).
- `--slots K --index i` runs units `K·i … K·i+K−1` concurrently; `--slots K --range a:b` runs the
  range at most K at a time. Every slot is a separate `yavih exec --id` process (probe and report slots
  too), so devices, torch globals and stop flags never cross between units.
- Slot `g` gets the `g`-th entry of the inherited `CUDA_VISIBLE_DEVICES` (indices or GPU UUIDs);
  `0 … K−1` when nothing is inherited; every slot gets `""` when the inherited value is empty; fewer
  entries than slots is `E_EXEC` (exit 1). The mask is passed as `YAVIH_SLOT_GPU` and applied after the
  unit's own `env`, so a matrix `env: {CUDA_VISIBLE_DEVICES: …}` cannot put two slots on one GPU.
- Each slot runs with `OMP_NUM_THREADS=1` and `--cpus N` CPUs when given, else the largest
  `needs.cpus` among the selected units divided by K (decode workers = that − 1). To pack K units on a
  whole node, set the matrix `cpus` to the node's share for all K slots.
- With several units the result is `{units: [record…], status, exit_code, id, not_started}` with the
  worst exit code; a `deadline` or `preempted` result, or a signal, stops further starts.
- `--cpus N` without slots sets the unit's CPU budget; without it the planned `needs.cpus` is used,
  clamped to the CPUs this process may use (affinity mask, cgroup quota).

## DONE markers and `last.json`

| kind | marker | written when |
|---|---|---|
| extract | `features/<split>/shard-{i:06d}.json` (the sidecar), or `features/<split>/NOT_APPLICABLE.json` | the tar is renamed into place and the sidecar is complete |
| probe | `probe/<eval_hash12>/seed<k>/DONE` = `{result_sha256, protocol_hash, probe_hash, eval_hash, model, reading, describe_sha256, features_fingerprint, yavih_version, created}` | `result.json` passed its schema check with status `ok` or `not_applicable` |
| report | `<out>/_report/DONE` | `report.md` and `report.json` are written |

A marker counts only when its recorded values equal the unit's (else the unit runs and the old output
moves to `_stale/<utc>/`):

- extract: the sidecar's `benchmark.input_hash`, `model`, `reading`, `bridge.describe_sha256` (the
  profile's), `shard.index` and `shard.n`; or a `NOT_APPLICABLE.json` with the same `model`,
  `reading` and `describe_sha256`;
- probe, before it runs (`plan --pending`, `exec`): `protocol_hash`, `probe_hash`, `model`, `reading`
  and `describe_sha256`, so a reconfigured bridge is re-extracted and re-probed;
- probe, inside the probe after its validate prologue: the same plus `features_fingerprint`
  recomputed from the sidecars on disk, so features that changed under an unchanged configuration
  (a deleted and re-extracted shard) are re-probed; an unchanged set is `skipped` without training.

A `failed` unit never writes a marker. Results-store files ([results](results.md)) are written after
the probe's DONE and are never read to decide whether a unit is done.

The record `<out>/logs/last.json` (shared by the units of one `out`, the last one wins) and
`<out>/logs/last.<id>.json` (per unit, which `plan status` prefers) hold `{id, kind, status, exit_code,
wall_s, host, gpu, slurm_job_id, inputs_hash, inputs_scheme, needs, started, ended, log, message,
history: {attempts, status, streak}, launcher, cwd, env}` and `refused_at_start` for a start
refusal. `streak` counts consecutive attempts that ended with the same non-zero status.

## `yavih plan status`

`yavih plan status plan.jsonl [--root DIR]` decides each unit's state from its outputs first (a
matching DONE marker → `done` / `not_applicable`) and from its last record otherwise:

| state | from the record |
|---|---|
| `pending` | no record; or a `done` / `skipped` record whose marker is gone; or a `running` record whose process cannot be alive (listed under `stale_running`) |
| `running` | a `running` record whose pid is alive on this host, or, from another host, started less than `2 × needs.minutes + 30 min` ago and not past its deadline + 30 min |
| `preempted` | exit 4 (`deadline` or `preempted`): resubmittable, never counted as failed |
| `not_ready` | exit 3 |
| `misconfigured` | exit 1 |
| `failed` | exit 2 or 5 |

`repeated` lists units whose last three or more attempts ended with the same exit 3 or 4: resubmitting
them again will not help, and their message says what to fix. Text mode prints the totals, one line
per kind and the units that need attention with the last line of their message; `--json` lists every
unit (`{done, pending, running, not_ready, preempted, misconfigured, failed, not_applicable, total,
by_kind, stale_running, repeated, units}`).

## Local runs

`yavih plan run plan.jsonl --local` runs every unit that is not done yet on this machine — a
workstation, a GPU box, a container, a Kubernetes pod or an interactive allocation — through the same
`exec` steps (checks, records, logs):

```bash
yavih plan build matrix.yaml --out plan.jsonl
yavih plan run plan.jsonl --local                    # one unit at a time, dependencies first
yavih plan run plan.jsonl --local --slots 2          # two at a time, slot g on the g-th visible GPU
yavih plan run plan.jsonl --local --deadline +8h     # stop starting units 8 h from now
yavih plan status plan.jsonl
```

- A unit starts once all its `depends_on` units are done (extract → probe → report).
- With `--slots K` the units run as `yavih exec --id` processes, K at a time, each with
  `--cpus` or the usable CPUs divided by K.
- A unit that ends `failed`, `misconfigured`, `not_ready`, `deadline` or `preempted` is not retried in
  that run, and its dependents are reported `blocked`; a deadline or preemption, or a signal, stops
  new starts. Fix the cause and run `plan run` again: finished units are skipped.
- The result is `{units, counts, blocked, not_started, already_done, total, status, exit_code}`;
  exit 0 when everything is done, else the worst unit's code, and 3 when units were only blocked or
  not started.

Lower-level pieces for your own runner: `yavih exec plan.jsonl --range 0:40` (in order, one process),
`yavih exec plan.jsonl --index 7`, or GNU parallel with one GPU per job slot. In a container, mount the
runs root at the same absolute path (`-v "$R:$R"`): the plan stores only each unit's `cwd` relative to itself, while
`out`, `done` and the children's `--out` are absolute paths under the runs root. Mount the project at its path as
well, or pass `--root /workspace` to re-anchor `cwd` (and only `cwd`) to where the project is mounted; otherwise
rebuild the plan inside the container. CPU quotas are read from the cgroup. A Kubernetes Indexed Job runs
`yavih exec $R/_plans/<model>.jsonl --index $JOB_COMPLETION_INDEX --deadline +<seconds>`.

## Running under SLURM

yavih does not know SLURM beyond reading `SLURM_*` variables: you write the batch script. The unit
of scheduling is one line of a plan file, run by one array task:

```bash
#!/bin/bash
#SBATCH --job-name=yavih-extract
#SBATCH --array=0-31%8                 # 0 … COUNT−1 of the pending file, %8 = at most 8 at once
#SBATCH --gres=gpu:1                   # max needs.gpu of the pending file
#SBATCH --cpus-per-task=8              # = matrix cpus (decode workers 7)
#SBATCH --mem=48G                      # ≥ max needs.mem_gb
#SBATCH --time=00:50:00                # ≥ max needs.minutes + 20 min
#SBATCH --signal=B:TERM@600            # TERM 600 s before the wall (exec stops on its own 900 s before)
#SBATCH --requeue                      # preemption and node failure
#SBATCH --output=yavih-bench/logs/%x_%A_%a.log   # ignored by git; mkdir -p yavih-bench/logs before sbatch
set -euo pipefail
cd /path/to/project                    # the repository root (yavih-bench/ below it; unit paths come from the plan)
PLAN="${1:?pending file}"              # <runs root>/_plans/pending.extract.jsonl
DEPS="${2:?full plan}"                 # <runs root>/_plans/<M>.jsonl
# `exec` replaces the shell, so SLURM's TERM reaches yavih (bash would not forward it)
exec yavih exec "$PLAN" --deps "$DEPS" --index "$SLURM_ARRAY_TASK_ID" --deadline slurm
```

and a driver loop that resubmits until nothing is pending:

```bash
M=yavih-bench/matrices/mymodel.yaml
R=$(yavih config get runs.root); P=$R/_plans    # plan files are run state: under the runs root
mkdir -p yavih-bench/logs                       # sbatch --output lands there
yavih plan build "$M" --out "$P/mymodel.jsonl"
for KIND in extract probe report; do
  while :; do
    N=$(yavih plan build "$M" --pending --kind "$KIND" --out "$P/pending.$KIND.jsonl" --format count)
    [ "$N" -eq 0 ] && break
    sbatch --wait --array="0-$((N-1))%8" "$KIND.sbatch" "$P/pending.$KIND.jsonl" "$P/mymodel.jsonl" || true
    yavih plan status "$P/mymodel.jsonl"   # stop here on failed / misconfigured / repeated units
    yavih plan calibrate "$R"              # optional: raise minutes of units their deadline stopped
  done
done
yavih report build "$R/*"
```

Rules the script must follow:

- **Request what the units need.** Build one pending file per kind and take the maxima of `needs`
  over it for `--gres`, `--mem` and `--time`; `--cpus-per-task` equals the matrix `cpus`.
- **Pass `--deps plan.jsonl`.** A pending file of probe units holds no extract units; without the
  full plan every probe exits 3.
- **Let TERM reach `yavih exec`.** Start it with the shell's `exec`, or run it in the background and
  forward TERM yourself; a non-interactive bash neither forwards TERM to a foreground child nor runs a
  trap until that child exits.
- **Use `--deadline slurm` only where the cluster exports `SLURM_JOB_END_TIME`**; elsewhere pass
  `--deadline +<seconds>` (the job's `--time` minus a margin).
- **Resubmit by exit code.** Exit 3 (`not_ready`) and 4 (`deadline`, `preempted`) are resubmitted by
  the next `--pending` iteration; `--requeue` covers only preemption and node failure, not a time
  limit. Exit 1, 2 and 5, and `repeated` units, need a fix first. Chain kinds with
  `--dependency=afterany:<jobid>`, not `afterok`: a preempted task exits 4 and the probe's own
  dependency check is what matters.
- **Guard `--index` against a rebuilt pending file** with `--plan-sha <sha256 prefix>` (from
  `plan build … --format count --json`).
- **A shard longer than the partition's maximum wall never finishes**: extraction resumes per shard,
  not inside one. Move the split directory aside and re-plan with a smaller `shard_minutes` or a
  `shards` pin.

Packing several units on one node uses `--slots K` (one array task runs K units, `ceil(COUNT / K)`
tasks). Site-specific templates (whole-node packing, containers as launchers, offline nodes) and a
pre-submit checklist ship with the agent skill: `yavih skill show --ref slurm`.
