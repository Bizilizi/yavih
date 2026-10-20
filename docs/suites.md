# Suites

A suite is a named set of benchmarks and readings, grouped by capability. "Benchmark my model" means: run a
suite. yavih ships two, `minimal` and `default`; a project can add its own or replace a shipped one.

```
yavih suite list                                   # every suite and where it comes from
yavih suite show default                           # its capabilities, benchmarks, readings and joins
yavih plan init --suite default --model M --launcher /envs/m/bin/python   # matrix.yaml + bridge stubs
yavih suite missing default --model M              # what M still lacks
yavih report build runs/ --suite default           # the report with a completeness appendix
```

A suite is not a profile: a profile is the `profile.json` that `yavih bridge check --out` writes for one bridge
on one benchmark ([bridge](bridge.md)); `plan build` needs one per matrix entry.

## Schema

```yaml
schema_version: 1
name: default
description: "Default set: audio, visual, audio-visual, temporal and spatial capabilities on packaged benchmarks."
groups:
  # audio-visual: every reading of one model; VGGSounder also scores them together (join)
  - capability: audio-visual
    entries:
      - {benchmark: vggsounder, readings: [a, v, av], join: true}
      - {benchmark: audioset20k, readings: [a, v, av]}
```

| key | rule |
|---|---|
| `schema_version` | must be 1 |
| `name` | `[a-z0-9][a-z0-9_-]*`, at most 64 characters; equals the file name (`<name>.yaml`) when loaded by name |
| `description` | one non-empty line |
| `groups` | non-empty list of `{capability, entries}`; each capability at most once |
| `groups[].capability` | `audio`, `visual`, `audio-visual`, `temporal`, `spatial` or `retrieval` (`yavih.suites.CAPABILITIES`, the vocabulary of `benchmark.yaml` `capability` too) |
| `groups[].entries[].benchmark` | a benchmark name or alias (`yavih bench list`); an alias is stored as the folder's name |
| `groups[].entries[].readings` | non-empty list of distinct reading tags; each one a reading the folder's `bridge_template.py` declares (a literal `"reading"`, or the `choices` of its `reading` argument) |
| `groups[].entries[].join` | `false` (default) or `true`: the folder's `probe.py` defines `join()`, and the readings are scored together by `yavih report build --join <benchmark> <reading>=RUN ...` ([results](results.md#joins)); needs at least two readings |
| `groups[].entries[].required` | `true` (default) or `false`: an optional entry is marked `(optional)` in the completeness views |
| `groups[].entries[].status` | `packaged` (default: the benchmark resolves to a folder) or `pending` (no folder yet: listed, never planned or checked) |

Unknown keys at any level, a `(benchmark, reading)` listed twice, and every rule above are refused with
`E_SUITE` (exit 1). A join's results-store key is `<benchmark>:<readings joined by +>` (`vggsounder:a+v+av`).

The capability vocabulary is shared: every benchmark folder tags itself in its `benchmark.yaml` with
`capability: [...]` from the same `yavih.suites.CAPABILITIES` ([benchmark folder](benchmark-folder.md#capability)),
and the benchmarks index and `yavih bench list` group by those tags. A suite still chooses the group an entry sits
in; `yavih bench lint` warns (`W_SUITE_CAPABILITY`) when a shipped suite lists a benchmark under a capability the
folder does not carry.

## Where suites come from

`yavih suite show <name>` and every `--suite <name>` resolve in this order:

1. an existing `.yaml` / `.yml` path (any argument with a `/` or a yaml suffix is a path);
2. `<project suites dir>/<name>.yaml`, where the project suites directory is `$YAVIH_SUITES` when set, else
   `suites/` next to the project config file (`--config`, `$YAVIH_CONFIG` or the `yavih.toml` found walking up
   from the current directory; [configuration](configuration.md)), else `./suites`;
3. the shipped `yavih/suites/<name>.yaml`.

A project suite of a shipped name shadows it (`yavih suite list` marks it `project*`). `yavih suite new <name>
[--from default] [--out DIR|FILE] [--force]` copies a suite, comments included, into the project suites directory
with its `name` replaced; an existing file is refused (`E_EXISTS`) unless `--force`. Suites are files only: there
is no configuration table for them.

## Shipped suites

### `minimal`

The smallest set that exercises the three readings: one benchmark per modality.

| capability | benchmark | readings | why |
|---|---|---|---|
| audio | `esc50` | a | small, five folds, a standard audio-classification reference |
| visual | `ssv2` | v | motion-centric: appearance alone does not solve it |
| audio-visual | `vggsounder` | a, v, av + join | the three readings of one model on one population, scored together by the authors' scorer |

### `default`

What "benchmark my model" means when nothing else is said.

| capability | benchmark | readings | why |
|---|---|---|---|
| audio | `esc50`, `fsd50k` | a | single-label environmental sounds and multi-label sound events |
| visual | `ssv2`, `k400`, `in1k` | v | motion-centric video, appearance-centric video, still images |
| audio-visual | `vggsounder` | a, v, av + join | per-modality labels; the join reports what the model hears, sees and confuses |
| audio-visual | `audioset20k` | a, v, av | large-vocabulary multi-label tagging from either or both streams |
| temporal | `avsync` | av | synchronisation offset: needs both streams aligned in time |
| spatial | `vggss_box` | av | where the sounding object is in the frame |

Kinetics-400, Something-Something-v2, ESC-50 and VGGSound (here through VGGSounder, its re-annotated test set)
are the benchmarks the literature keeps reporting as continuity anchors, so numbers stay comparable with
published work. The same literature names AVATAR as the successor of VGG-SS-style sound-source localisation and
AV-SyncBench as the successor of VGGSound-Sparse-style synchronisation; neither has a packaged folder yet, so the
`default` suite keeps `vggss_box` and `avsync`.

### `extensive`

Not defined yet. Candidates, in order:

1. VGGSound retrieval
2. SSv2 Temporal-50 / SSv2-Template subsets
3. the UTD-debiased Kinetics-400 split
4. AV-SyncBench
5. AVATAR
6. UnAV-100
7. RTime
8. Diving48

Each needs a packaged benchmark folder first ([contributing a benchmark](contributing-a-benchmark.md)); until then a project
suite can list one as `status: pending`.

## Completeness

`yavih suite missing <name> [<glob>...] --model M [--results DIR]` (globs default to the runs root; give the
globs before the options) prints, for model `M`, one status per suite `(benchmark, reading)` and one per join:

| status | meaning |
|---|---|
| `done` | a DONE-marked `result.json` of `(benchmark after alias, M, reading)` exists under the globs; a note says when none is `ok` or none is standard |
| `missing` | no such result |
| `pending` | the entry has no packaged folder |
| `join-missing` | the join entry is not in M's results store; the row carries the `yavih report build ... --join ...` command that computes it from the standard run layout |
| `join-unknown` | no results store could be resolved to look in |

It exits 0 either way. `yavih report build <glob>... --suite <name>` appends the same statuses for every model
in the report as a trailing `## Suite completeness: <name>` section (md) and the `suite` key (json); a join
computed by the same invocation (`--join`) counts as done ([results](results.md#suite-completeness)).
`yavih bench list --suite <name>` lists the suite's packaged benchmarks with their suite group (`suite_capability`
in json), readings and the folder's own `capability` tags.

`yavih plan init --suite <name> --model M --launcher PYTHON` writes the matrix of a suite for one model
([execution](execution.md#yavih-plan-init)).

## Codes

| code | exit | meaning |
|---|---|---|
| `E_SUITE` | 1 | an unknown suite, or a suite file that does not parse or validate |
| `E_EXISTS` | 1 | `suite new` / `plan init`: the destination exists (pass `--force`) |
| `E_ARG` | 1 | `plan init`: a `--model` that is not a valid model name, an empty `--launcher` |
| `W_SUITE_INVALID` | 0 | `suite list`: a suite file that does not validate (listed with its error) |
