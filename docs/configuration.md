# Configuration

yavih reads its site configuration once per process into one typed `Settings` object
(`yavih.config`). Configuration holds **site knobs only**: where data, runs, results and scratch
live, and how many CPUs and how much RAM a process may use. Nothing here changes a number in a
result: comparability knobs (`ProbeConfig`, readout parameters, metric pins) are fixed in code and
in `benchmark.yaml` and are never configurable (see [readouts](readouts.md) and
[hashing](hashing.md)).

Every resolved value keeps its source (`cli`, `env:NAME`, `file:PATH:LINE`, `yaml default …`,
`default`), and `yavih config show` prints them.

## Layers and precedence

Layers, lowest to highest precedence. A later layer overrides an earlier one key by key.

| # | layer | where |
|---|---|---|
| 1 | built-in defaults and `benchmark.yaml` root defaults | `${data.root}`, `${datasets.<name>}` placeholders (below) |
| 2 | user file | `$XDG_CONFIG_HOME/yavih/config.toml`, else `~/.config/yavih/config.toml` |
| 3 | the active `[site.<id>]` table of the user file | see [Sites](#sites) |
| 4 | project file | `--config FILE`, else `$YAVIH_CONFIG`, else the nearest `yavih.toml` found walking up from the current directory |
| 5 | the active `[site.<id>]` table of the project file | |
| 6 | environment variables | `YAVIH_*` ([Environment variables](#environment-variables)) |
| 7 | CLI flags | `--config`, `--runs-root`, `--results`, `--data`, `--lenient` |

Programmatic overrides (`Benchmark.load(..., data={...})`) sit above all layers for the roots they
name.

The CLI flags are accepted before or after the subcommand (`yavih --runs-root /x plan build m.yaml`
and `yavih plan build m.yaml --runs-root /x` are the same). A command exports them to its own
environment (`YAVIH_CONFIG`, `YAVIH_RUNS`, `YAVIH_RESULTS`, `YAVIH_DATA__…`, `YAVIH_LENIENT`), so the
children it starts (`yavih exec` → `python -m yavih.extract`) resolve the same values.

| flag | sets |
|---|---|
| `--config FILE` | the project file (beats `$YAVIH_CONFIG` and `yavih.toml` discovery); a missing file is an error |
| `--runs-root DIR` | `[runs] root` |
| `--results DIR` | `[results] root` (`probe run --results` and `report build --results` mean the same root) |
| `--data BENCH.ROOT=PATH` | one data root of one benchmark (name or alias); repeatable |
| `--data root=PATH` | `[data] root` |
| `--data datasets.NAME=PATH` | `[datasets] NAME` |
| `--lenient` | schema errors become warnings ([Strict schema](#strict-schema-and-lenient-mode)) |

A relative path given on the command line is made absolute against the current directory.

## Files

Both files are TOML with the same schema, and every key is optional. The user file is the place for
per-person, per-cluster values; a project `yavih.toml` (committed next to a `matrix.yaml`, say)
layers over it. When the project file is the user file itself, it is read once.

```toml
# ~/.config/yavih/config.toml (user) or ./yavih.toml (project)
[data]
root = "/data/datasets"                 # ${data.root}
[data.in1k]                             # one table per benchmark (name or alias); keys = its declared roots
media_root = "$GROUP/imagenet2012"      # ~ and $VAR expand; a relative path is relative to THIS file
[datasets]
vggsound = "/data/vggsound"             # ${datasets.vggsound}: shared by several benchmarks
[runs]
root = "runs"                           # relative to this file
[results]
root = "/data/yavih-results"            # default: <runs root>/../results
[scratch]
root = "/tmp/yavih"                     # staged probe shards; default SLURM_TMPDIR; never TMPDIR
[home]
root = "~/.yavih"
[resources]
ram_budget_gb = 24
cpus = 8
decode_workers = 7
[site.cluster_a]                        # active when YAVIH_SITE=cluster_a or a hosts glob matches
hosts = ["gpu*", "*.cluster-a.example"]
[site.cluster_a.data]
root = "/project/datasets"
```

## Tables and keys

| table | key | meaning | default |
|---|---|---|---|
| `[home]` | `root` | yavih home: fetched benchmark copies (`benchmarks/`), `data_status/`, `media_info/` | `~/.yavih` |
| `[runs]` | `root` | runs root: run directories `<runs root>/<benchmark>/<model>/<reading>/` | `./runs` (relative to the current directory) |
| `[results]` | `root` | results-store root ([results](results.md)) | `results/` next to the runs root (`<runs root>/../results`) |
| `[scratch]` | `root` | where a probe stages shards that exceed the RAM budget | `$SLURM_TMPDIR`; else none (the probe reads features in place) |
| `[data]` | `root` | the site data root, `${data.root}` | none |
| `[data.<bench>]` | `<root>` | one declared data root of benchmark `<bench>` (name or alias) | the root's `default` in `benchmark.yaml` |
| `[datasets]` | `<name>` | a dataset shared by several benchmarks, `${datasets.<name>}` | `${data.root}/<name>` |
| `[resources]` | `ram_budget_gb` | RAM budget of a probe / validate, GiB (float, `>= 0`) | `min(16, 0.45 × memory limit)`, else 16 |
| `[resources]` | `cpus` | CPUs this process may use (int, `> 0`) | scheduler / affinity / cgroup, else `os.cpu_count()` |
| `[resources]` | `decode_workers` | default decode workers (int, `> 0`) | `cpus − 1` for an allocation, `cpus // 2` on a whole machine |
| `[site.<id>]` | `hosts` + any table above | per-cluster values | see [Sites](#sites) |

Notes:

- **Scratch is never `TMPDIR`.** The resolution is `YAVIH_SCRATCH` / `[scratch] root` →
  `$SLURM_TMPDIR` → none. With none, a probe whose training set exceeds the RAM budget reads the
  features in place through the memmap backend and records `W_SIZE`; it does not fail.
- **Resources** are read in this order. CPUs: `YAVIH_CPUS` / `[resources] cpus` →
  `SLURM_CPUS_PER_TASK` → PBS `NCPUS` / `PBS_NUM_PPN` → the affinity mask and the cgroup CPU quota
  (`cpu.max` v2, `cpu.cfs_quota_us` v1) → `os.cpu_count()`. Memory limit: `SLURM_MEM_PER_NODE` →
  `SLURM_MEM_PER_CPU × CPUs` → the cgroup memory limit (never the node's physical RAM). Decode
  workers are capped at 16 per task and are at least 1. A malformed resource value is always an
  error, even in lenient mode. See [execution](execution.md#byte-read-ahead-and-decode-workers).
- `[data]` takes only `root` and `[data.<bench>]` sub-tables. `[datasets]` names must match
  `[a-z][a-z0-9_]*`.

### Values and paths

- `~`, `$VAR` and `${VAR}` are expanded in every path value. An unset `$VAR` is a schema error.
- A relative path in a file is relative to **that file's directory**. A relative path in an
  environment variable stays relative to the current directory.
- `${data.root}` and `${datasets.<name>}` are substituted in `benchmark.yaml` defaults and config
  values alike. `[data] root` itself cannot use a placeholder; a `[datasets]` entry may use
  `${data.root}` but not another `${datasets.*}`.

## Data roots

Every benchmark declares its data roots in `benchmark.yaml` under `data.roots`, each with a `default`
and a description ([benchmark-folder](benchmark-folder.md)). One root resolves, highest first:

1. `Benchmark.load(data={...})` (programmatic);
2. `--data <bench>.<root>=PATH`;
3. `YAVIH_DATA__<BENCH>__<ROOT>` (upper case, double underscores: `YAVIH_DATA__IN1K__MEDIA_ROOT`);
4. the project file `[data.<bench>] <root>` (its active site table first);
5. the user file `[data.<bench>] <root>` (likewise);
6. the yaml default with `${data.root}` / `${datasets.<name>}` substituted.

`${datasets.<name>}` is `[datasets] <name>` (or `YAVIH_DATASETS__<NAME>`), else `${data.root}/<name>`.
A root that does not resolve (a placeholder with nothing configured, no default) is an error that names
the config key, the environment variable and the `--data` spelling that would set it. A root marked
`prepare_only: true` is read only by the folder's `prepare/` steps, so it never blocks loading the
benchmark when unresolved. Paths are not checked for existence here: `yavih bench verify` and
`yavih doctor --bench B` do that.

### Example

```toml
# ~/.config/yavih/config.toml
[data]
root = "/data/me/datasets"
[data.in1k]
media_root = "/shared/mirrors/imagenet2012"
[runs]
root = "/scratch/me/yavih-runs"
[results]
root = "/scratch/me/yavih-results"   # optional; default <runs root>/../results = /scratch/me/results
[datasets]
vggsound = "/data/me/datasets/vggsound"
```

With the packaged defaults this resolves to:

| root | yaml default | resolved | source |
|---|---|---|---|
| `esc50.media_root` | `${data.root}/esc50` | `/data/me/datasets/esc50` | `yaml default ${data.root}/esc50` |
| `in1k.media_root` | `${data.root}/imagenet` | `/shared/mirrors/imagenet2012` | `file:…/config.toml:5` |
| `vggsounder.media_root` | `${datasets.vggsound}` | `/data/me/datasets/vggsound` | `yaml default ${datasets.vggsound}` |

`export YAVIH_DATA__ESC50__MEDIA_ROOT=/scratch/esc50` would override the first row for one shell;
`yavih --data esc50.media_root=/scratch/esc50 …` for one command.

Per-site caches under the home directory (`data_status/<bench>-<digest>.json`) are keyed by a digest
of the resolved roots, so two sites sharing one home directory keep their own presence and
verification stamps.

## Sites

One shared file can hold the values of several clusters in `[site.<id>]` tables. A site table takes
`hosts` (a hostname glob or a list of globs) and any of the tables above except `[site]`.

- The active site is `$YAVIH_SITE`, else the first `[site.<id>]` (user file first, then project file)
  whose `hosts` globs match `socket.gethostname()` (shell-style `fnmatch`).
- When the host matches several sites, the first wins with the warning `W_CONFIG_SITE`; set
  `YAVIH_SITE` to choose. `YAVIH_SITE` naming no `[site.<id>]` table is a schema error.
- The active site table of a file overrides that file's top-level values; it does not override a
  higher layer (a project file's top level beats the user file's site table).

## Environment variables

Every `YAVIH_*` variable yavih reads or sets is listed below; any other `YAVIH_*` name is a schema
error (a typo or a retired variable never goes unnoticed). Two families are patterns rather than
fixed names: `YAVIH_DATA__<BENCH>__<ROOT>` (one data root) and `YAVIH_DATASETS__<NAME>` (one
`[datasets]` entry). An empty value is ignored.

<!-- yavih:docs-table env begin (generated from yavih.config.ENV_VARS: python tests/docs_tables.py --write) -->
| variable | meaning |
|---|---|
| `YAVIH_HOME` | yavih home (fetched benchmarks, data_status/, media_info/); config [home] root; default ~/.yavih |
| `YAVIH_RUNS` | runs root; config [runs] root; default ./runs |
| `YAVIH_RESULTS` | results-store root; config [results] root; default <runs root>/../results |
| `YAVIH_SCRATCH` | scratch for staged shards; config [scratch] root; else SLURM_TMPDIR; never TMPDIR |
| `YAVIH_CONFIG` | the project config file (beats ./yavih.toml discovery); same as --config |
| `YAVIH_SITE` | the active [site.<id>] table (else the first whose hosts glob matches the hostname) |
| `YAVIH_LENIENT` | 1 = unknown config keys / YAVIH_* variables / stale benchmark copies warn instead of failing |
| `YAVIH_BENCH_PATH` | colon-separated directories of out-of-tree benchmark folders (after $YAVIH_HOME/benchmarks and the packaged ones) |
| `YAVIH_DATA__ROOT` | the site data root (${data.root}); config [data] root |
| `YAVIH_RAM_BUDGET_GB` | RAM budget of a probe / validate in GiB; config [resources] ram_budget_gb |
| `YAVIH_CPUS` | CPUs this process may use (set per slot by `yavih exec --slots`); config [resources] cpus |
| `YAVIH_DECODE_WORKERS` | default decode workers; config [resources] decode_workers |
| `YAVIH_JSON` | 1 = every command emits one JSON envelope |
| `YAVIH_TESTING` | 1 = non-pinned ProbeConfig allowed (results get standard: false) |
| `YAVIH_UNIT_ID` | set by `yavih exec` for the child: the plan unit id |
| `YAVIH_DEADLINE` | the deadline as an epoch (set by `yavih exec` for the child; inherited by extract / probe run) |
| `YAVIH_PROBE_CKPT_EVERY_S` | seconds between probe checkpoints (default 600) |
| `YAVIH_SLOT_GPU` | set by `yavih exec --slots` for each slot: its device mask, applied after the unit's env |
| `YAVIH_CLAIM_LEASE_S` | shard-claim heartbeat lease in seconds (default 900); a staler claim is taken over |
| `YAVIH_BENCH` | set by `yavih bench prepare <b> <step>` for the step it runs: the benchmark name (FOLDER = its folder) |
<!-- yavih:docs-table env end -->

Variables yavih reads from the scheduler or the host (not `YAVIH_*`, never an error):
`XDG_CONFIG_HOME`, `SLURM_TMPDIR`, `SLURM_CPUS_PER_TASK`, `SLURM_MEM_PER_NODE`, `SLURM_MEM_PER_CPU`,
`SLURM_JOB_END_TIME`, `SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `NCPUS`,
`PBS_NUM_PPN`, `CUDA_VISIBLE_DEVICES`.

## Strict schema and lenient mode

The schema is strict. Each of these is an error, and the command exits 1 (`E_USAGE`) with the full
list of problems and the hint to run `yavih config show`:

- an unknown table or key (with a "did you mean" suggestion), or a value of the wrong type;
- a near miss of a known benchmark name (`[data.ESC50]`, `YAVIH_DATA__ESC_50__MEDIA_ROOT`);
- a root that a known benchmark does not declare;
- an unknown `YAVIH_*` variable or a malformed `YAVIH_DATA__…` name;
- an unset `$VAR` inside a value;
- `YAVIH_SITE` naming no site table.

`--lenient` or `YAVIH_LENIENT=1` turns these into warnings (`W_CONFIG_UNKNOWN`, `W_CONFIG_TYPE`,
`W_CONFIG_UNKNOWN_BENCH`, `W_CONFIG_UNKNOWN_ROOT`, `W_CONFIG_ENV`, `W_CONFIG_UNSET_VAR`,
`W_CONFIG_SITE`, each suffixed "ignored: lenient"), and also downgrades a stale fetched benchmark copy
(`E_HOME_BENCH_STALE`) to a warning. It exists for a home directory shared by installs of different
yavih versions, where a key a newer version knows must not break an older one. Lenient mode does not
excuse a malformed TOML file, a missing `--config` file or a malformed `[resources]` value.

A `[data.<name>]` table naming no known benchmark at all is only the warning
`W_CONFIG_UNKNOWN_BENCH`: a benchmark folder loaded by path may carry any name, and its keys are
checked when that folder is loaded.

## Inspecting the configuration

`yavih config show [--bench B]` prints the files read (layer, path, how each was found, whether it
exists), the active site and why, and every value with its source. With `--bench B` it also resolves
that benchmark's roots and shows, per root, the value, the source, whether it exists, whether it is
`prepare_only`, and the config key, environment variable and `--data` spelling that would set it.
Under `--json`, `data` is `{files, site, site_source, lenient, values: {key: {value, source}}, warnings,
bench?: {name, folder, roots: {root: {value, source, exists, prepare_only, default, env_var,
config_key, error}}}}`.

`yavih config get KEY` prints one resolved value and nothing else, for shell scripts (a benchmark's
`prepare/*.sh`):

```bash
yavih config get runs.root            # also results.root, scratch.root, home.root, data.root
yavih config get datasets.vggsound    # [datasets] vggsound, else ${data.root}/vggsound
yavih config get resources.decode_workers
MEDIA=$(yavih config get esc50.media_root)   # <benchmark>.<root>, name or alias
```

A key that does not resolve exits 1.

`yavih doctor [--bench B]` embeds the same `config show` block and never aborts on configuration: an
invalid configuration is the warning `W_CONFIG` (doctor then shows the lenient view), an unresolvable
or missing data root is `W_DATA_ROOT` naming its config key and environment variable, and a missing
`prepare_only` root is the note `N_DATA_ROOT_PREPARE`. It also reports the effective resources
(`decode_workers_default`), the home, runs and results roots, the GPUs, and, with `--bench`, one row
read from the benchmark's media ([cli](cli.md#environment-and-configuration)).
