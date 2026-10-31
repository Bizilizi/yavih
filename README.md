<p align="center">
  <img src="https://raw.githubusercontent.com/Bizilizi/yavih/main/assets/header.png" alt="yavih — frozen-feature benchmark harness for video and audio models" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/yavih/"><img alt="PyPI" src="https://img.shields.io/pypi/v/yavih?color=22d3ee"></a>
  <a href="https://github.com/Bizilizi/yavih/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Bizilizi/yavih/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/yavih/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/yavih?logo=python&logoColor=white"></a>
  <a href="https://github.com/Bizilizi/yavih/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-a78bfa"></a>
</p>

<h3 align="center">A growing library of video, audio and audio-visual benchmarks.<br>One harness, one protocol — and your coding agent does the wiring.</h3>

**yavih** (Yet Another Video Harness) evaluates frozen video, audio and audio-visual encoders. You bring a model;
yavih brings the benchmarks, the data recipes, the probes and the guarantees that your numbers mean the same thing
as everyone else's.

<p align="center"><img src="https://raw.githubusercontent.com/Bizilizi/yavih/main/assets/readme/agent.gif" alt="An agent benchmarks a model with yavih from one sentence" width="100%"></p>

## Get started in one sentence

```bash
pip install yavih          # into the Python environment that runs your model
yavih skill install        # teaches your coding agent (Claude Code, Codex, Cursor, …) how to use yavih
```

Then tell your agent:

> **Benchmark my model with yavih.**

That's it. The agent picks the default suite, prepares each benchmark's data, writes the small adapter that feeds
your model, checks it, extracts features, trains the standard probes and hands you a report. What it cannot do on
its own — a licence you have to accept, a download only a person can make — it lists for you at the end, while the
rest runs.

## Why yavih

<table>
<tr>
<td width="50%" valign="top">

### 🗂 Many benchmarks, one harness
Classification, temporal, spatial, sync and dense tasks across audio, video and audio-visual data, all behind the
same commands. Pick a ready-made suite or define your own.

</td>
<td width="50%" valign="top">

### 🤖 One skill — your agent does the rest
yavih is built to be driven by a coding agent. Every benchmark explains itself to the agent: where the data comes
from, what the model receives, what it must return, and how to check each step.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔒 Same protocol, same data — provably
Every benchmark is frozen: the clips, the labels, the probe and the metric. A short hash names that exact protocol,
and two numbers are compared only when their hashes match. No silent drift, no "our split".

</td>
<td width="50%" valign="top">

### ➕ Add your own benchmark
Ask your agent to *add &lt;your dataset&gt; to yavih*. It follows the contributor guide, builds the folder, runs a
trial and opens the pull request — and everyone can run your benchmark the same way.

</td>
</tr>
</table>

## A growing library

You rarely choose benchmarks one by one. A **suite** is a named set: `default` is what "benchmark my model" runs,
`minimal` is a quick smoke pass, and `yavih suite new` makes your own.

```bash
yavih suite list               # default, minimal, and any suite you define
yavih bench list               # every benchmark: task, size, how to get the data, what's ready on your machine
```

<p align="center"><img src="https://raw.githubusercontent.com/Bizilizi/yavih/main/assets/readme/library.gif" alt="The benchmark library grouped by capability" width="100%"></p>

<!-- yavih:bench-table capabilities begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
| capability | benchmarks | examples |
|---|---|---|
| `audio` | 3 | [audiocaps_t2a](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/audiocaps_t2a/README.md), [esc50](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/esc50/README.md), [fsd50k](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/fsd50k/README.md) |
| `visual` | 4 | [in1k](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/in1k/README.md), [k400](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/k400/README.md), [nyuv2_depth](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/nyuv2_depth/README.md), [ssv2](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/ssv2/README.md) |
| `audio-visual` | 6 | [audioset20k](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/audioset20k/README.md), [ave](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/ave/README.md), [avsync](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/avsync/README.md), [vggsounder](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/vggsounder/README.md) + 2 more |
| `temporal` | 2 | [ave](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/ave/README.md), [avsync](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/avsync/README.md) |
| `spatial` | 2 | [nyuv2_depth](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/nyuv2_depth/README.md), [vggss_box](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/vggss_box/README.md) |
| `generation` | 2 | [audiocaps_t2a](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/audiocaps_t2a/README.md), [vggsounder_v2a](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/vggsounder_v2a/README.md) |

[All 13 benchmarks → the index](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/README.md)
<!-- yavih:bench-table capabilities end -->

Each benchmark is one self-contained folder with its own page: the paper, how to get the data, and exactly how yavih
runs it.

## Numbers you can trust

A benchmark result is only useful if someone else can get the same number the same way. yavih freezes everything
that decides what a number means and leaves everything else to you:

- **Frozen by yavih:** which clips, which labels, which raw files your model receives, which probe is trained on top,
  how it is trained, and which metric is reported — with a floor (what a model that ignores its input would score)
  next to every result.
- **Yours to choose:** the model, how it reads the media (frames, crops, audio front end), its precision and batch
  size. All of it is recorded with every number, none of it changes what the number is compared against.

That frozen protocol is summarised in one short hash, the `eval_hash`. Same hash ⇔ same data, same protocol, same
probe. The report only ever puts numbers with the same hash in one column; change a clip or a metric and you get a
new hash, never a quietly different number.

<p align="center"><img src="https://raw.githubusercontent.com/Bizilizi/yavih/main/assets/readme/hash.gif" alt="Matching hashes make results comparable; a changed protocol gets its own hash" width="100%"></p>

## Add your benchmark

Tell your agent:

> **Add Diving48 to yavih.**

It follows [contributing a benchmark](https://github.com/Bizilizi/yavih/blob/main/docs/contributing-a-benchmark.md)
step by step — scaffold the folder, freeze the manifest from the public release, write the data recipe and the
tests, render the header — then runs a synthetic trial and a fresh-agent
[trial run](https://github.com/Bizilizi/yavih/blob/main/docs/trial-run.md) before opening the pull request.

<p align="center"><img src="https://raw.githubusercontent.com/Bizilizi/yavih/main/assets/readme/contribute.gif" alt="An agent adds a new benchmark folder and opens a pull request" width="100%"></p>

## How it works

```
prepare ──► extract ──► validate ──► probe ──► report
 data         your model   yavih checks   standard     comparable
 recipe       (a bridge)   the features   probe        tables
```

Your model connects through a **bridge**: a small Python file, written by your agent from a template, that turns a
benchmark's raw files into features. It runs in your model's own environment — yavih installs without touching your
torch. yavih checks the features, trains the same probe every model gets, and reports with floors and confidence
intervals. Everything runs on a laptop, a workstation or as job arrays on any cluster.

<details>
<summary><b>Learn more</b></summary>

- [Quickstart by hand](https://github.com/Bizilizi/yavih/blob/main/docs/quickstart.md) — every step without an agent, with a complete bridge.
- [Design](https://github.com/Bizilizi/yavih/blob/main/docs/design.md) — why yavih works the way it does.
- [The bridge contract](https://github.com/Bizilizi/yavih/blob/main/docs/bridge.md), [validation](https://github.com/Bizilizi/yavih/blob/main/docs/validation.md), [readouts](https://github.com/Bizilizi/yavih/blob/main/docs/readouts.md), [hashing](https://github.com/Bizilizi/yavih/blob/main/docs/hashing.md), [the CLI](https://github.com/Bizilizi/yavih/blob/main/docs/cli.md) — the reference.
- [The agent skill](https://github.com/Bizilizi/yavih/blob/main/src/yavih/skill/yavih/SKILL.md) — what your agent reads.
- [Developing yavih](https://github.com/Bizilizi/yavih/blob/main/docs/development.md) and the [releases](https://github.com/Bizilizi/yavih/releases).

</details>

## License

yavih is released under the [MIT License](https://github.com/Bizilizi/yavih/blob/main/LICENSE). Benchmark datasets
are not part of yavih and keep their own licences and terms of use (each benchmark's page says where they come from).
