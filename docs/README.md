# yavih documentation

Repository documentation for yavih, a frozen-feature benchmark harness for video, audio and audio-visual
encoders. Start with the project [README](../README.md); [Quickstart by hand](quickstart.md) walks through every step without an agent. The agent-facing
documentation ships with the package as a skill (`yavih skill show`, source in
[`src/yavih/skill/yavih/`](../src/yavih/skill/yavih/SKILL.md)).

## Concepts

- [Quickstart by hand](quickstart.md): install, prepare a benchmark, write a bridge, extract, validate, probe and report, step by step, with what yavih enforces along the way.
- [Design](design.md): what yavih is and is not, and the reasons behind its rules.

## Reference

Normative descriptions of current behaviour.

| page | covers |
|---|---|
| [Benchmark folder](benchmark-folder.md) | the folder contract, the `benchmark.yaml` schema, manifest and classes, the `Benchmark` API and resolution order, README.md / setup.md contracts, what `yavih bench lint` checks |
| [Bridge](bridge.md) | what a bridge receives (`Sample`, `RawStream`, media facts, canaries), the `Bridge` API, `describe()` keys and declared rules, the `encode()` contract, positions, `used`, guardrails, `yavih bridge check` |
| [Features](features.md) | the feature WebDataset format: shards, members, sidecars, `_layout.json`, `_meta.json`, missing rows, `NOT_APPLICABLE.json`, resumption and claims, `FeatureWriter` / `FeatureReader` |
| [Validation](validation.md) | `yavih features validate`: every check and code with its level, eligibility, waivers and custom rules |
| [Readouts](readouts.md) | the metric registry, common probe rules, `ProbeConfig`, the `clf` / `seq` / `grid` / `box` readouts, floors, controls, intervals |
| [Results](results.md) | `result.json`, `yavih report`, comparability in reports, the results store |
| [Hashing](hashing.md) | version constants, canonicalization, input / scoring / protocol / describe / probe / eval hashes, unit ids, migration through folder changelogs |
| [Configuration](configuration.md) | config layers and tables, data roots, sites, environment variables |
| [CLI](cli.md) | every command and flag, the `--json` envelope, exit codes |
| [Suites](suites.md) | named benchmark sets (`minimal`, `default`), the suite file schema, project suites, completeness (`suite missing`, `report build --suite`) |
| [Execution](execution.md) | sharding, read-ahead, `plan init`, `matrix.yaml`, plan units, `yavih exec`, deadlines, slots, DONE markers, local runs, wrapping in SLURM |

## Developing yavih

- [Development](development.md): repository layout and module map, invariants, testing, conventions.
- [Contributing a benchmark](contributing-a-benchmark.md): the ordered steps from a fork to a merged benchmark
  pull request: scaffold, protocol, freeze, prepare, docs, bridge template, hashes, tests, header, trials, CI.
- [Trial run](trial-run.md): the required pre-PR run of a new benchmark by a fresh agent with only the wheel and
  the skill: protocol, worker prompt, report template, pass criteria.
- [Adding a readout](adding-a-readout.md): the readout registry, drivers, metrics, entry points.
- [Releasing](releasing.md): versions from git tags, CI, the release workflow, PyPI trusted publishing.
