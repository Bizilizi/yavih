# Command line

`yavih` is a thin argparse layer over the Python API: it parses the arguments, calls one library
function, wraps the result in an envelope and maps failures to an exit code. `python -m yavih …` is
the same program, and `python -m yavih.extract …` is the stand-alone form of `yavih extract`.

```
yavih [--json] [--version] [--config FILE] [--runs-root DIR] [--results DIR]
      [--data BENCH.ROOT=PATH]... [--lenient] <command> [<subcommand>] ...
```

`yavih <command> --help` lists the flags of every command; this page groups them and says what each
one is for. The agent-facing summary of the same material ships inside the package:
`yavih skill show --ref cli`.

## Global options

These are accepted by every command, before or after the subcommand.

| option | meaning |
|---|---|
| `--json` | emit exactly one JSON envelope on stdout ([below](#the---json-envelope)); also `YAVIH_JSON=1` |
| `--version` | print the version and exit 0 |
| `--config FILE` | the project config file (beats `$YAVIH_CONFIG` and the discovery of the nearest `yavih-bench/yavih.toml`, else `yavih.toml`, up to the repository root) |
| `--runs-root DIR` | runs root (beats `YAVIH_RUNS` and `[runs] root`) |
| `--results DIR` | results-store root (beats `YAVIH_RESULTS` and `[results] root`) |
| `--data BENCH.ROOT=PATH` | one data root override, repeatable; also `root=PATH` and `datasets.NAME=PATH` |
| `--lenient` | unknown config keys, unknown `YAVIH_*` variables and stale benchmark copies warn instead of failing |

The configuration options are described in [configuration](configuration.md); the command exports
them to the environment of every child process it starts.

## Environment and configuration

| command | purpose | important flags |
|---|---|---|
| `yavih init` | write the project folder `yavih-bench/` (`yavih.toml`, `README.md`, `.gitignore` with `runs/`, `logs/`, `__pycache__/`, `bridges/`, `matrices/`, `suites/`, `benchmarks/`, `tools/`); runs and results default to `runs/` and `results/` inside it, the user config is never written ([project layout](project-layout.md)) | `--dir D` (default `.`); `--site ID [--hosts GLOB]... [--launcher CMD]` writes a real site table; `--force` refreshes README.md, .gitignore and the directories (`yavih.toml` is never overwritten); refuses a folder another project file already governs (`E_PROJECT_EXISTS`) unless that project lists it in `[project] subprojects` |
| `yavih doctor` | report the environment: installed extras, torch, decoders, GPUs, the configuration with sources, the home / runs / results roots, the default decode-worker count, installed skill copies, the project folder (version requirement, nested project files, stray matrices, a project-default runs root over 50 GB: `W_PROJECT_VERSION`, `W_PROJECT_NESTED`, `W_PROJECT_STRAY`, `W_RUNS_SIZE`); a configuration that does not load even leniently is `W_CONFIG` with `config_error: {message, project_file}` and the text names that file (no `yavih init` hint for a project whose file is malformed) | `--bench B` also resolves B's data roots, reads one stored row per stream (`W_MEDIA_READ`), checks the header backends (`W_HEADER_BACKEND`), the benchmark's external packages and its pinned tools (`data.tools:` per tool, `W_TOOL_PIN` when one is missing or off its pin); `--selftest` runs the reference bridge on 32 ESC-50 clips |
| `yavih config show` | every configuration value with its source, the files read, the active site and the project folder | `--bench B` adds B's data roots, whether each exists, and the key / env var that sets it |
| `yavih config get` | print one resolved value for shell scripts | `KEY` = `runs.root`, `results.root`, `scratch.root`, `home.root`, `data.root`, `datasets.<name>`, `resources.<key>` or `<benchmark>.<root>`; exit 1 when it does not resolve |
| `yavih skill install` | install the packaged agent skill into a project | `--target claude\|codex\|cursor\|agents-md` (default `claude`); `--path P` (default per target: `.claude/skills/yavih`, `.codex/skills/yavih`, `.cursor/rules/yavih.mdc`, `AGENTS.md`) |
| `yavih skill show` | print the skill's `SKILL.md` | `--all` appends every reference; `--ref NAME` prints one reference (`bridge`, `cli`, `data`, `readouts`, `slurm`) |

`doctor` never aborts on configuration: an invalid configuration is the warning `W_CONFIG`, an
unresolvable or missing data root `W_DATA_ROOT`, an installed skill copy that differs from the
packaged one `W_SKILL_STALE`. Run it on a compute node of the cluster that will run the units.

## Benchmarks

| command | purpose | important flags |
|---|---|---|
| `yavih bench list` | every discoverable benchmark: aliases, readout, `capability` tags, clips per split, media size, source kind, local status (`missing`, `partial NN%`, `verified`) | `--fmt text\|md` (`md`: the table the docs embed, no data scan); `--suite S` only the suite's benchmarks, with their suite group (`suite_capability`) and `readings` |
| `yavih bench info` | a benchmark's README plus its `capability` tags, hashes, roles, floors, metrics, source, budget, external packages and hash history | `<name>`; `--readme-only`; `--setup` prints `setup.md` instead |
| `yavih bench path` | print the benchmark folder, or the absolute path of one shipped file | `<name> [<what>]` (`what` as for `bench show`) |
| `yavih bench files` | every shipped file of a folder with absolute path, kind (`readme`, `setup`, `yaml`, `template`, `probe`, `manifest`, `classes`, `conformance`, `prepare`, `test`, `other`) and size | `<name>` |
| `yavih bench show` | print one shipped file | `<name> <what>`: `readme`, `setup`, `yaml`, `template`, `probe`, `classes`, `conformance` (`conformance/expected.json`), `prepare/<file>` or a path inside the folder |
| `yavih bench prepare` | list the numbered `prepare/NN_*.{sh,py}` steps, or run one | `<name> [<step> [-- args…]]`; the step runs with `FOLDER` and `YAVIH_BENCH` set and the command exits with the step's own code |
| `yavih bench fetch` | build or download the manifest into `$YAVIH_HOME` and verify its sha256 | `--refresh` overwrites locally edited files of the home copy with the packaged ones |
| `yavih bench verify` | presence scan, byte reads of a seeded sample, an optional decode health check, derived / target sha256s, curation sums; writes `data_status.json` | `--census` probes every present file and writes the media-info table; `--decode-sample N` (32); `--fingerprint-sample N` (256) |
| `yavih bench lint` | check folders against the folder contract | `<name\|path>...`; `--all` every discoverable folder; `--skip CHECK` (repeatable); errors exit 2, `TODO` markers are warnings |
| `yavih bench conformance` | check (or record) a folder's conformance fixture: the numbers its scoring path produces on a generated population, at its `protocol_version` ([conformance](conformance.md)) | `<name\|path>...`; `--all`; `--record` writes `conformance/expected.json`; `--same-version --reason TEXT` overwrites differing values at the same version (the reason is stored only when values did move); `--force` overwrites without comparing; `--same-version`, `--reason` and `--force` need `--record`, `--reason` needs `--same-version` (`E_USAGE`); a drift, stale or missing fixture exits 2, an absent external package is a warning |
| `yavih bench hash` | a folder's `protocol_hash`, `input_hash`, `scoring_hash`, `manifest_hash` and `folder_source` | `<name\|path>`; `--explain` prints the canonical JSON and digest of every hashed component (recompute the hashes with stdlib only) |
| `yavih bench new` | scaffold a folder that passes `bench lint`, with `TODO` where facts are needed; reports how its conformance fixture was made (`recorded`, or `copied from <like>` when the copied protocol's package is absent here) | `<name>`; `--readout R`; `--like B` (the folder copied); `--dir D` (default: the project's `benchmarks/`, else the first `$YAVIH_BENCH_PATH` entry, else `.`) |
| `yavih bench trial` | run a folder's whole pipeline in process on a tiny synthetic stand-in: conformance fixture, stand-in, oracle bridge, `bridge check`, extract, validate, probe, report, headline against its floor; per-stage status | `<name\|folder> --synthetic` (required); `--readout-fast` (the test-only fast `ProbeConfig`; the headline must then clear its floor); `--keep DIR` (work in a new or empty `DIR` and keep it; `paths` names the stand-in, bridge, profile, run directory and report) |

The folder contract, `benchmark.yaml` and the lint checks are described in
[benchmark-folder](benchmark-folder.md); writing a new folder and opening its pull request in
[contributing a benchmark](contributing-a-benchmark.md).

### `bench trial --synthetic`

`yavih bench trial <name|folder> --synthetic` runs the chain a real run takes, in process, on a stand-in of the
folder (`yavih.testing.make_synthetic_like`: its name, readout, stream names and kinds, guaranteed media facts and
label facts over tiny generated media), with the oracle test bridge, whose features encode the label:
`stand_in`, `bridge`, `bridge_check`, `extract` (every role split in two shards), `validate`, `probe` (no
results-store file), `report`, `headline`. `data.stages` holds one record per stage (`ok`, `warning`, `failed`,
`skipped` after a failure; `seconds`; `detail`; `error`), `data.headline` the headline metric, its CI and floor and
`clears_floor`. Under `--readout-fast` the oracle must clear the floor; without it a miss is the warning
`W_TRIAL_HEADLINE`, since the pinned schedule may underfit the stand-in's few dozen rows.

The trial never reads or writes the user's runs root, results store, `$YAVIH_HOME` or configuration: while it
runs, every `YAVIH_*` variable is replaced (home, runs and results under the work directory, no config file,
`YAVIH_TESTING=1`), the `--config` / `--runs-root` / `--results` / `--data` layer is cleared, and both are restored
afterwards, also on error. The work directory is temporary unless `--keep DIR`.

What it does not cover: the `conformance` stage runs the folder's real scoring path (its readout, metrics,
`probe.py` hooks and pinned `external` package) on the fixture's generated population, but every later stage
runs on the stand-in, which never reads the real data; it is single-label, so the `probe` stage of a `multi` or
`multi_sets` folder trials as single-label `acc`; its bridge is the oracle, not yours; speed and numbers on the
stand-in carry over to nothing. The trial on real data is the fresh-agent [trial run](trial-run.md); without
`--synthetic` the command is a usage error (exit 1) that points there.

| code | exit | meaning |
|---|---|---|
| `E_TRIAL` | 2 | a stage failed a contract (the oracle below its floor under `--readout-fast`, a failed validation or probe); `errors[0].where` names the stage. A stage that was not ready (a decoder missing) exits 3 with its own code |
| `E_TRIAL_NOT_READY` | 3 | no stand-in can be built: a readout other than `clf`, `seq`, `grid`, `box`, a stream kind other than video, audio, image, or no media writer (PyAV, soundfile, pillow) |
| `W_TRIAL_HEADLINE` | 0 | without `--readout-fast`, the oracle's headline did not clear its floor |

## Suites

A suite is a named set of benchmarks and readings grouped by capability; `minimal` and `default` ship with
yavih ([suites](suites.md)).

| command | purpose | important flags |
|---|---|---|
| `yavih suite list` | every suite with its `origin` (`shipped`, `project`; a project suite of a shipped name shadows it) and description | an invalid project file is listed with its error (`W_SUITE_INVALID`) |
| `yavih suite show` | a suite's capabilities, benchmarks, readings and joins | `<name\|file.yaml>` |
| `yavih suite new` | copy a suite into the project suites directory to edit | `<name>`; `--from S` (default `default`); `--out DIR\|FILE`; `--force` overwrites |
| `yavih suite missing` | per suite `(benchmark, reading)` of one model `done`, `missing` or `pending`; per join `done`, `join-missing` (with the `report build --join` command) or `join-unknown`; exit 0 | `<name> [<glob>...] --model M` (globs before the options; default: the runs root); `--results DIR` (where joins are found) |

An unknown suite or a file that does not validate is `E_SUITE` (exit 1).

## Bridges

| command | purpose | important flags |
|---|---|---|
| `yavih bridge new` | copy a benchmark's `bridge_template.py` to the project's `bridges/<model>_<bench>.py` (else `./bridges/`) with the model name (and reading) filled in | `<bench> --model M`; `--reading a\|v\|av`; `--out DIR\|FILE`; `--force` overwrites |
| `yavih bridge check` | smoke-test a bridge: describe schema, declared rules, rerun / batch / order invariance, throughput; writes `profile.json` | `<bridge.py> --bench B`; `--n 32`; `--arg k=v` (repeatable); `--decode-workers N`; `--out profile.json`; `--synthetic` (a tiny generated stand-in, no data, no profile); `--allow-loopback` |

A failed hard check, an unfilled template (`E_TEMPLATE_UNFILLED`) or features equal for every smoke
clip (`E_CONSTANT_FEATURES`) exit 2. The text output names the run directory the plan will use,
`<runs root>/<bench>/<model>/<reading>/`. See [bridge](bridge.md).

## Extraction and features

| command | purpose | important flags |
|---|---|---|
| `yavih extract` | extract one shard into `<out>/features/<split>/` | `--bench B --bridge P --split S --shard i/n --out RUN_DIR`, or `--rows lo:hi` instead of `--shard` (a capped run over manifest rows `[lo, hi)` as the one shard `0/1`; `plan build --sample`); `--arg k=v`; `--canaries 8`; `--decode-workers N`; `--read-ahead 2`; `--deadline …`; `--expect-profile P` (skip finished shards before loading the model; `E_CONFIG_DRIFT` on drift); `--allow-loopback` |
| `yavih features validate` | validate feature sets and write `_meta.json` | `<features_dir or features/>`; `--verify-sha`; `--no-write-meta`; `--waive CODE=reason` (repeatable, `all=reason`) |
| `yavih features inspect` | per-dimension statistics, near-constant dims, dead rows, position and `used` summaries | `<features_dir>` |
| `yavih features diff` | compare two feature sets row by row | `<a> <b>`; reports `rows_compared`, `max_abs`, `max_rel_l2`, `verdict` |
| `yavih features stage` | copy shards, sidecars and `_meta.json` to scratch and print the staged path for `probe run --scratch` | `<features/> <scratch>` |

`yavih extract` is normally started by `yavih exec` from a plan ([execution](execution.md)). The
feature format is in [features](features.md), the validation codes and waivers in
[validation](validation.md).

## Probing and reporting

| command | purpose | important flags |
|---|---|---|
| `yavih probe run` | train and score the benchmark's standardized readout on a run directory; writes `result.json`, `DONE` and the results-store file | `<run_dir>`; `--seeds 0,1,2` (default `0`); `--deadline …`; `--scratch DIR`; `--allow-low-coverage`; `--loader-workers N` (default 1: a background thread gathers the next two training batches; `0` = inline; same numbers); `--bench REF` (when `_meta.json`'s name does not resolve); `--no-validate`; `--waive CODE=reason`; `--log-samples`; `--out DIR` (probe a read-only feature set); `--device cpu\|cuda\|cuda:N`; `--results DIR` or `--no-results` |
| `yavih report build` | comparability-checked tables with floors | `<glob>...`; `--fmt md\|csv\|json`; `--out PATH`; `--seeds aggregate\|list`; `--allow-coverage-diff`; `--detail`; `--join BENCH READING=RUN ...`; `--results DIR`; `--merge-results`; `--suite S` (a completeness appendix per model, md and json); `--registry DIR` (a model registry's accepted entries join the table as rows, flag `registry`, model `<org>/<model>`) |
| `yavih registry export` | registry entries from a results store ([registry](registry.md)) | `<results_root\|results_file> --org ORG --out DIR`; `--model NAME` (repeatable); `--include-nonstandard`; writes `entries/<benchmark>/<eval_hash12>/<org>/<model>/<reading>.json` and `protocols/<benchmark>/<protocol_hash12>.json`; an entry whose conformance did not pass here is written, listed under `unverified` and warned about (`W_REGISTRY_UNVERIFIED`); `check` refuses it |
| `yavih registry check` | the registry repository's CI check | `DIR`; every entry parses, is filed under its id, carries consistent hashes, a protocol payload that recomputes them, a passed conformance and a tolerance; non-standard or low-coverage entries are warnings; errors exit 2 |

`probe run` first validates every split (skip with `--no-validate`). Its `<run_dir>` is
`<runs root>/<bench>/<model>/<reading>`; any other layout needs `--results` or a configured results
root, else `E_RESULTS_ROOT` (exit 1). See [readouts](readouts.md) and [results](results.md).

## Plans and execution

| command | purpose | important flags |
|---|---|---|
| `yavih plan init` | write a matrix skeleton (one entry per suite `(benchmark, reading)`) and one bridge stub per benchmark for one model, into the project's `matrices/<M>.yaml` and `bridges/` (else `./matrix.yaml`, `./bridges/`); profiles go to `<matrix stem>.profiles/`; prints the `bridge check` command of every entry, then `plan build <matrix> --out <runs root>/_plans/<M>.jsonl` | `--suite S --model M`; `--launcher PYTHON\|NAME` (default: the configured launcher at execution; `W_LAUNCHER_UNSET` when none); `--out FILE`; `--bridges-dir DIR` (relative to the matrix); `--force` overwrites the matrix (bridge files are always kept) |
| `yavih plan build` | turn a `matrix.yaml` into unit records (`plan.jsonl`) | `<matrix.yaml>`; `--out plan.jsonl` (else JSONL on stdout); `--pending` (only units whose DONE is missing or stale); `--kind extract\|probe\|report`; `--shard-minutes M`; `--format jsonl\|count`; `--sample N` (a capped trial run, below) |
| `yavih plan status` | count units by status and list those that need attention | `<plan.jsonl>`; `--root DIR` |
| `yavih plan run` | run every pending unit of a plan on this machine, dependencies first | `<plan.jsonl> --local` (required); `--slots K` (default 1); `--deadline …`; `--deadline-margin 900`; `--root DIR`; `--cpus N`; `--deps plan.jsonl` |
| `yavih plan calibrate` | refine resource multipliers from finished units and bump units their deadline stopped; writes `<runs root>/_calibration.json` | `<runs_root>` |
| `yavih exec` | run one plan unit (or a range, or K units at once) with its dependency, DONE and deadline checks | `<plan.jsonl>`; exactly one of `--index N`, `--id X`, `--range a:b`; `--deps plan.jsonl`; `--deadline …`; `--deadline-margin 900`; `--slots K`; `--root DIR`; `--cpus N`; `--plan-sha HEX` |

`plan build --format count` prints the number of units; with `--out` and `--json`, `data` also holds
the file's `sha256` (what `exec --plan-sha` checks). `plan build --sample N` plans a capped run of the
matrix: the first `N` rows of every split in manifest order (not class-balanced), one `extract --rows 0:k` unit
per split, probe units with `--no-results`, all under the separate runs root `<out>-sample<N>`; its results are
never stored and every report row from them is marked `partial`
([execution](execution.md#capped-runs-plan-build---sample-n)). `plan run` without `--local` is a usage error:
yavih never writes or submits scheduler scripts. Everything about plans, units, deadlines, slots and
SLURM wrapping is in [execution](execution.md).

### `--deadline`

`extract`, `probe run`, `exec` and `plan run` share one parser (`yavih.deadline`):

| form | meaning |
|---|---|
| `none`, `off` | no deadline |
| `+3600`, `+90m`, `2h`, `1h30m`, `45s`, `1d` | that long from now (a bare `+N` is seconds) |
| `1760000000` | an epoch in seconds; a bare number before 2001 is refused as a duration typo |
| `2026-10-10T18:00:00Z` | an ISO-8601 time (no offset = UTC) |
| `slurm` | `SLURM_JOB_END_TIME − margin` (`--deadline-margin`, default 900 s); outside SLURM no deadline; inside a job without a usable `SLURM_JOB_END_TIME` the error `E_DEADLINE_SPEC` (exit 1) |

Without `--deadline`, `extract` and `probe run` inherit `$YAVIH_DEADLINE` (what `exec` sets for its
children); `exec` and `plan run` have no deadline.

## The `--json` envelope

With `--json` (or `YAVIH_JSON=1`) stdout carries exactly one JSON object, written with sorted keys
when the command ends; logs and `[yavih] warning …` lines go to stderr.

```json
{"status": "ok", "data": {}, "errors": [], "warnings": [], "paths": {}}
```

| key | type | content |
|---|---|---|
| `status` | string | `ok`, `error`, `skipped` or `not_applicable` |
| `data` | object or array | the command's result (Paths as strings, NaN and infinities as `null`) |
| `errors` | array of Finding | why the command failed; empty on success |
| `warnings` | array of Finding | warnings of the command, plus every configuration warning emitted while it ran |
| `paths` | object | `{name: path}` of files written or found (`profile`, `plan`, `folder`, `staged`, `features`, `results_file`, `samples_files`, `script`, …); a value may be a list of paths |

A Finding is `{"code": "E_…|W_…|N_…", "message": str, "where": str | null}`. Parse `status` first;
`errors[0].code` names the rule that failed.

`status` is `skipped` for a unit or shard that was already done (`exec`, `extract`),
`not_applicable` when the bridge or the planner declared the benchmark not applicable (`exec`,
`extract`, `probe run`), `error` whenever the exit code is non-zero.

Stdout stays one object even when a bridge or a library prints: while a `--json` command runs,
Python's `sys.stdout` is redirected to stderr and, when stdout and stderr are real descriptors 1 and
2, fd 1 is duplicated onto fd 2 for the duration, which also covers C-level writes and child
processes. A usage error under `--json` is still one envelope (`E_USAGE`, exit 1).

Without `--json`, the human rendering goes to stdout, every warning and error to stderr as
`[yavih] warning CODE: message (where)` / `[yavih] error CODE: message (where)`, and a non-`ok`
status ends with `[yavih] status <status> (exit N)` on stderr.

## Exit codes

All commands use the same codes; a scheduler or driver loop can act on them without parsing output.

| code | meaning | action |
|---|---|---|
| 0 | done, skipped, or `not_applicable` recorded; also `--help` and `--version` | none |
| 1 | usage or configuration error: bad arguments, an invalid config, unknown `--arg`, bad `matrix.yaml`, profile or suite file, bad `--deadline`, a unit from a plan older than the split's frozen layout (`E_LAYOUT`); `exec` status `misconfigured` | fix, then resubmit |
| 2 | contract violation: validation errors, hash mismatch, describe schema, a failed invariance check, lint errors | fix the bridge or the data |
| 3 | not ready: data missing, a dependency's DONE missing, decoder or external package missing, scratch too small, the shard claimed by another live task (`E_SHARD_CLAIMED`) | resubmit later |
| 4 | deadline reached, SIGTERM / SIGUSR1 / SIGINT received, or the shard's claim taken over while writing (`E_CLAIM_LOST`); nothing partial left behind; resumable | resubmit |
| 5 | internal or unexpected failure (also an exception raised inside a bridge's `encode`) | inspect the log |

Details that follow from the code:

- An argparse error (unknown flag, missing argument) exits 1, not argparse's own 2.
- `yavih bench prepare <name> <step>` exits with the step's own code (`E_PREPARE_FAILED`).
- `yavih exec` passes a child's exit code through, except that a child killed by a signal (negative
  return code, or 128 + n) is exit 4 `preempted` and a child exit 1 is recorded `misconfigured`. For
  several units (`--range`, `--slots`) the worst code wins; for `plan run --local` too, and units that
  could not start because a dependency did not finish make it at least 3.
- `plan status` maps the last record of each unit: exit 4 → `preempted`, 3 → `not_ready`, 1 →
  `misconfigured`, 2 and 5 → `failed`.

## Error-code conventions

Every finding carries a code with a prefix that says how serious it is:

| prefix | meaning | effect |
|---|---|---|
| `E_` | error | the command fails with the exit code of that error |
| `W_` | warning | reported (stderr, envelope `warnings`, often recorded in `_meta.json` or `result.json`); the command succeeds |
| `N_` | note | information that needs no action (`N_RULE_CUSTOM`, `N_ELIGIBILITY`, `N_DATA_ROOT_PREPARE`) |

How a failure becomes a code:

- A library exception that carries a `code` and an `exit_code` (`ExtractError`, `ProbeRefused`,
  `MatrixError`, `ProfileError`, `PlanError`, `ExecError`, `LayoutError`, `ClaimHeld`, `ClaimLost`,
  `DeadlineError`, …) is reported with exactly those.
- Otherwise the exception class decides: configuration, schema and unknown-benchmark errors →
  `E_USAGE` (1); validation, manifest, describe and format errors → `E_CONTRACT` (2); missing data,
  media or decoders → `E_NOT_READY` (3); a timeout → `E_DEADLINE` (4); an interrupt → `E_INTERRUPTED`
  (4); an unimplemented stub → `E_NOT_IMPLEMENTED` (5); anything else → `E_INTERNAL` (5, with the
  traceback on stderr in text mode).
- `yavih exec` and `yavih plan run` report a unit that did not finish as `E_UNIT_<STATUS>`
  (`E_UNIT_NOT_READY`, `E_UNIT_DEADLINE`, `E_UNIT_PREEMPTED`, `E_UNIT_MISCONFIGURED`,
  `E_UNIT_FAILED`); the unit's own record holds the underlying message.
- A check that is wrong for a model is waived with a reason, never edited out; a waived error becomes
  the warning `W_WAIVED` ([validation](validation.md)).

Codes of plans and execution:

| code | exit | meaning |
|---|---|---|
| `E_SUITE` | 1 | an unknown suite, or a suite file that does not parse or validate ([suites](suites.md)) |
| `E_EXISTS` | 1 | `init` / `plan init` / `suite new` / `bridge new`: the destination exists (pass `--force`) |
| `E_PROJECT_EXISTS` | 1 | `init`: another project file already governs this directory (one project per repository; `[project] subprojects` for a nested one) |
| `E_MATRIX` | 1 | malformed `matrix.yaml` (unknown key, wrong type, missing entry key) |
| `E_PROFILE` | 1 | a profile missing, malformed, disagreeing with its entry or without passed hard checks |
| `E_PLAN` | 1 | a plan cannot be built or read (two entries on one run directory, a frozen layout contradicting a `shards` pin, an older or newer plan schema, duplicate ids) |
| `E_EXEC` | 1 | bad unit selection: unknown `--id`, `--index` out of range, a bad `--range`, `--slots` with `--id` or more slots than inherited GPUs, a `--plan-sha` mismatch |
| `E_DEADLINE_SPEC` | 1 | an unusable `--deadline` |
| `E_LAYOUT` | 1 | the unit's `--shard i/n` disagrees with the split's frozen layout: re-run `plan build` |
| `E_WAIVE_INVALID` | 1 | a command-line or matrix waiver of a non-waivable code, or without a reason |
| `E_SHARD_CLAIMED` | 3 | another live task is writing this shard |
| `E_CLAIM_LOST` | 4 | this writer's claim was taken over; nothing was published |
| `E_DEADLINE`, `E_PREEMPTED` | 4 | an extract stopped at its deadline or on SIGTERM / SIGUSR1 after the current batch |

Bridge, extraction and `bridge check` codes are listed in [bridge](bridge.md), validation codes in
[validation](validation.md), probe and report codes in [readouts](readouts.md) and
[results](results.md). `yavih skill show --ref cli` carries the complete list in one place.
