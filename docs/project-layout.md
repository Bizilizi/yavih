# Project layout

A repository that evaluates models with yavih keeps everything yavih-related in one folder,
`yavih-bench/`, at the repository root. `yavih init` writes it, every command finds it from any
directory below the root, and `yavih doctor` checks it. It holds text only: configuration, bridges,
matrices, profiles, suites, project-local benchmark folders and the project's own scripts, plus the
two roots yavih writes by default: `runs/` (features, plan state; ignored by git) and `results/`
(small results files, committed). Scratch is a configured root ([configuration](configuration.md)).

## The folder

```
yavih-bench/
  yavih.toml      the project configuration: [project], data roots, one [site.<id>] per cluster
  README.md       what lives here and where runs go (written by init)
  .gitignore      runs/, logs/, __pycache__/ (results/ is committed)
  bridges/        one file per (model, benchmark) plus the shared <model>_common.py
  matrices/       one matrix per model, <model>.yaml, with its profiles in <model>.profiles/
  suites/         project suites (yavih suite new)
  benchmarks/     project-local benchmark folders (yavih bench new), resolved by name
  tools/          the project's own glue: drivers, matrix generators, sync scripts; yavih never reads it
  runs/           the default runs root: <benchmark>/<model>/<reading>/ run directories, _plans/ (ignored by git)
  results/        the default results root: <model>/results_<timestamp>.json (committed)
```

| what | default | command that writes there |
|---|---|---|
| project configuration | `yavih-bench/yavih.toml` | `yavih init` |
| bridges | `bridges/<model>_<bench>.py` | `yavih bridge new`, `yavih plan init` |
| matrices | `matrices/<model>.yaml` | `yavih plan init` |
| profiles | `matrices/<model>.profiles/<entry>.profile.json` | `yavih bridge check --out` (the path `plan init` prints) |
| suites | `suites/<name>.yaml` | `yavih suite new` |
| benchmark folders | `benchmarks/<name>/` | `yavih bench new` |
| runs | `runs/<benchmark>/<model>/<reading>/`, plan files `runs/_plans/<model>.jsonl` | `yavih exec`, `yavih plan build --out` |
| results | `results/<model>/results_<timestamp>.json` | the probe units, `yavih report build` |

A profile belongs to a matrix entry (one model, benchmark and reading), not to a bridge file, so it
sits next to its matrix. Profiles are small JSON files and are committed with the matrix.

## Runs and results

With a project and no `[runs] root`, `YAVIH_RUNS` or `--runs-root`, the runs root is
`yavih-bench/runs` (absolute: a matrix `out: null` and `yavih config get runs.root` name the same
directory, and `config show` reports the source `default (project)`); the results root is likewise
`yavih-bench/results` unless `[results] root`, `YAVIH_RESULTS` or `--results` says otherwise. Outside
any project the fallbacks are `runs` and `results` under the current directory, made absolute
(`default (./runs)`).

The two differ in what they hold. `runs/` holds features, probe state and plan files: large, and
ignored by the `.gitignore` that `yavih init` writes (`runs/`, `logs/`, `__pycache__/`). `results/`
holds `results/<model>/results_<timestamp>.json`: small text files that are meant to be committed
with the project, so the results of every cluster and every person end up in one place.
`yavih doctor` never reports what lies under either root as a stray.

For a large suite put the runs on scratch: set `[runs] root` in the user file
(`~/.config/yavih/config.toml`; it stays personal and never gets committed) or in a `[site.<id>]`
table of `yavih.toml` (one path per cluster, shared by everyone on it), then `mv yavih-bench/runs
<that path>` and rebuild the plans (`yavih plan build … --out <new root>/_plans/<model>.jsonl`).
Results stay in `yavih-bench/results` unless `[results] root` moves them too. `yavih doctor` warns
`W_RUNS_SIZE` when the project-default runs root has grown past 50 GB and repeats these steps.

## One project per repository

Discovery takes the nearest `yavih-bench/yavih.toml` (else a bare `yavih.toml`) walking up from the
current directory and **stops at the repository root** (a `.git` directory, or the `.git` file of a
linked worktree; a submodule checkout is walked through to its superproject): a second project file
deeper in the same repository
would silently take over for commands run below it, which is how setups fragment into one copy per
experiment and per cluster. `yavih init` refuses to write a folder already governed by another
project file (`E_PROJECT_EXISTS`) unless that project lists the folder (or one containing it) in
`[project] subprojects`, and `yavih doctor` reports:

| finding | meaning |
|---|---|
| `W_PROJECT_NESTED` | another `yavih.toml` inside the repository that `[project] subprojects` does not list |
| `W_PROJECT_STRAY` | a `matrix*.yaml` outside the project folder and its relocated directories |
| `W_PROJECT_VERSION` | the installed yavih is outside `[project] yavih`, or the requirement cannot be read |
| `W_RUNS_SIZE` | the project-default runs root `yavih-bench/runs` holds more than 50 GB ([runs and results](#runs-and-results)) |
| `N_PROJECT` (note) | no project folder (`yavih init`), a development build, or a scan that hit its directory cap |

The scan never enters dot-directories, virtual environments, `__pycache__`, `build`, `dist`,
`node_modules`, the runs and results roots in effect (configured, or the project defaults), or more
than eight levels deep.

Different clusters are **sites, not folders**: one `[site.<id>]` table per cluster in the one
`yavih.toml`, with that cluster's data roots and launchers. One matrix then runs on every cluster,
because its entries name a launcher by name (or leave it to the site's default) and the site table
supplies the command. A monorepo of unrelated projects lists the others in `[project] subprojects`
of the root project (paths relative to the root project file, e.g. `subprojects = ["../services/asr"]`);
`yavih init` then writes each one's own folder, discovery below it finds that folder first, and
`doctor` reports neither the subproject from the root nor the root (its folder and its other
subprojects) from the subproject.

## The project file

```toml
launcher = "/envs/mymodel/bin/python"      # the default launcher (a top-level key: before any table header)

[project]
yavih = ">=1,<2"              # the yavih this project expects; doctor warns (never fails) on a mismatch
# bridges = "bridges"         # relocate a directory (relative to this file); also matrices, suites, benchmarks
# subprojects = []            # nested project folders that are not strays

[data]
root = "/data/datasets"       # ${data.root}; per benchmark: [data.<bench>] <root> = "..."

[launchers]
vjepa = "/envs/vjepa/bin/python"           # a matrix entry says `launcher: vjepa`

[site.cluster_a]
hosts = ["gpu*", "*.cluster-a.example"]
launcher = "/cluster-a/envs/mymodel/bin/python"
[site.cluster_a.launchers]
vjepa = "apptainer exec --nv /cluster-a/vjepa.sif python"
[site.cluster_a.data]
root = "/cluster-a/datasets"
```

The project file carries project facts only: the version requirement, data roots and model
environments. Runs and results default to the folder's own `runs/` and `results/`
([above](#runs-and-results)); a personal runs or scratch root belongs in the user file
`~/.config/yavih/config.toml` (holding overrides only; `yavih init` never writes it), so the project
file can be committed as it is. The version requirement is advisory: a development build (`.devN`, a dirty tree,
a checkout without tags) is reported as a note, never refused, and compatibility of features and
results is decided by the [version constants](hashing.md#version-constants), not by the package
version. Pin the install itself with the project's own requirements file.

## What may live elsewhere

The folder is the default, not a fence. Three things are fine outside it and raise no finding:

- **a bridge shipped inside a model package**: the matrix entry says `script: mymodel.yavih:Bridge`
  and the package is installed in the model environment ([execution](execution.md#matrixyaml));
- **a benchmark folder that is the repository's deliverable**, kept visible at the top level and
  declared by `[project] benchmarks = "../benchmarks"` (or placed on `$YAVIH_BENCH_PATH`);
- **job scripts kept with the rest of the project's infrastructure**: `tools/` is the sanctioned
  home for drivers and generators, but yavih reads nothing there, so calling existing scripts from
  it is the same to yavih.

A bare `yavih.toml` at the repository root is also accepted (the folder is tried first). What
`doctor` warns about is a second project file or a matrix file outside the folder, because those are
what split a setup.

## `yavih init`

`yavih init [--dir D] [--site ID [--hosts GLOB]... [--launcher CMD]] [--force]`
writes `D/yavih-bench/` (`D` default: the current directory): `yavih.toml` (never overwritten),
`README.md`, `.gitignore` and the five directories with a `.gitkeep` each; it never writes the user
config. Its text names the runs and results roots in effect (the folder's `runs/` and `results/` by
default). `--site` writes a real `[site.ID]` table for this cluster
(`--hosts` default: this host's name) with `--launcher` inside it; `--launcher` alone writes a
top-level `launcher`; otherwise the file carries a commented example. `--force`
refreshes README.md, .gitignore and the directories of an existing folder. The text ends with the next
commands: edit the file, `yavih doctor`, `yavih plan init --suite default --model <name>`.
