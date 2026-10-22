# Hashing, identity and comparability

This page is the normative reference for every digest yavih computes, the version constants that tag them, and
how a benchmark folder records its hashes so that older features and results stay usable. The implementation is
`yavih.hashing` (stdlib only), `yavih.versions`, `yavih.hash_history` and `yavih.report.migrate`.

## The principle

A comparability key covers only what yavih controls: the benchmark folder (input side and scoring side) and the
probe. It never covers the model. What the bridge decides (its configuration, dtype, batch size, weights) and what
the site has (media copies, presence, decoder builds) are recorded beside every number and checked for
consistency, but they are never folded into a comparability key, a path or a result key.

The model side enters keys only as two names the user chooses, `model` and `reading` (see
[model and reading](#model-and-reading)).

**Identity is the declaration, not the code** (`yavih-protocol-5`). `protocol_hash` is computed from what the
benchmark folder declares: the frozen manifest and classes by content, typed views of the `benchmark.yaml` blocks,
and declared versions (`protocol_version`, `READOUT_VERSION`). No source file enters it. The folder's `probe.py`
and `reader.py`, and yavih's own readouts and metrics, may be refactored, re-implemented in another framework or
ported by a third party without moving the hash; what pins their behaviour is `protocol_version` together with the
folder's [conformance fixture](conformance.md), which fails until the version is bumped when the numbers move. A
score recorded next to a hash is therefore comparable with every other score recorded next to that hash, whichever
implementation produced it. The source digests are still computed and recorded as provenance
(`Benchmark.folder_source`, [below](#folder-source-provenance)) so that two numbers under one hash can be traced to
the code that produced them.

Every digest by role:

| digest | inputs | controlled by | role |
|---|---|---|---|
| `manifest_hash` | `manifest.csv` (decompressed) + `classes.csv` bytes | benchmark | part of `input_hash`; seeds canary, duplicate-sample and verify-sample draws; keys the site census file |
| `input_hash` | the [input components](#input-side) | benchmark | feature-set identity: resume, `_layout.json`, validate (`E_HASH_INPUT`), the probe, extract unit ids, DONE checks; never comparability on its own |
| `scoring_hash` | the [scoring components](#scoring-side) | benchmark (+ `READOUT_VERSION`) | part of `protocol_hash`; recorded in sidecars as provenance |
| `protocol_hash` | `{input_hash, scoring_hash}` | benchmark (+ `READOUT_VERSION`) | comparability (via `eval_hash`); probe and report unit ids |
| `probe_hash` | `ProbeConfig`: head and training | probe (task sizes from the benchmark) | comparability (via `eval_hash`) |
| `eval_hash` | `{protocol_hash, probe_hash}` | benchmark + probe | **the** comparability key: report columns, probe directory, results `configs` |
| `describe_sha256` | `describe_subset(describe())` | bridge | consistency and provenance; never a key |
| `features_fingerprint` | shard sha256s, `N`, `D`, dtype, pos, `n_present` | bridge output (+ site presence) | integrity and staleness (probe DONE gate); never a row key |
| tar `sha256`, `pos_sha256`, `result_sha256` | file bytes | outputs | integrity |
| `source_sha256` (bridge) | bridge file bytes | bridge | provenance (`W_SOURCE_SHA`) |
| `folder_source` | `{probe, reader}`: normalized-source digests of the folder's `probe.py` / `reader.py` | benchmark folder | provenance in sidecars, `result.json` and results files; `W_FOLDER_SOURCE_DIFFERS` in the report; never identity |
| `weights_sha256` | state dict | bridge | provenance (`W_WEIGHTS_UNFINGERPRINTED` when null; `W_JOIN_WEIGHTS`) |
| `coverage.*.ids_sha256` | present clip ids | site | recorded; "population differs" in the report, `E_JOIN_POPULATION` |
| `media_fingerprint`, `media_info_sha256` | staged media, census | site | provenance (`W_MEDIA_FP`) |
| unit id, `inputs_hash` | plan record | plan (+ site, + bridge for staleness) | scheduling and staleness; never comparability |
| `RawStream.path` handle, canary/duplicate draws (`rng_seed`) | manifest hash, split, row, shard index | yavih | opacity and deterministic selection; never identity |

## Version constants

Every version constant lives in one stdlib-only, torch-free module, `yavih.versions`, so that the benchmark hashes
can be computed identically in the model env and in the probe env. The modules named below re-export them; nothing
else defines a version, except that each readout declares its own version, which `yavih.versions` reads through
the readout registry.

| constant | defined in / re-exported by | current value | bump when |
|---|---|---|---|
| `FEATURE_FORMAT_VERSION` | `yavih.versions` / `yavih.io.format` | `4` | the tar layout, member names, sidecar, `_meta.json` or `NOT_APPLICABLE.json` schema changes ([features](features.md)) |
| `BENCHMARK_SCHEMA_VERSION` | `yavih.versions` / `yavih.benchmark` | `2` | the `benchmark.yaml` schema changes incompatibly ([benchmark folder](benchmark-folder.md)); a purely additive optional key does not bump it |
| `RESULT_SCHEMA_VERSION` | `yavih.versions` / `yavih.readouts.base` | `4` | the `result.json` schema changes ([results](results.md)); register an upgrade step in `report.migrate` |
| `RESULTS_FILE_VERSION` | `yavih.versions` / `yavih.report.store` | `2` | the results-store file schema changes ([results](results.md)) |
| `PLAN_SCHEMA_VERSION` | `yavih.versions` / `yavih.plan.units` | `3` | the unit record in `plan.jsonl` changes ([execution](execution.md)); an older record is refused with "rebuild the plan" (finished shards are kept) |
| `READOUT_VERSION[readout]` | each readout's own module (`VERSION`), read through the readout registry | `clf: 1`, `seq: 2`, `grid: 2`, `box: 2` | any change to that readout's head, loss, floor, metric or protocol, including a change to a `yavih.metrics` function it lists ([readouts](readouts.md)) |
| `PROBE_SCHEMA` | `yavih.versions` / `yavih.probes.config` | `"yavih-probe-1"` | the `probe_hash` payload shape changes |
| `PROTOCOL_SCHEMA` | `yavih.versions` / `yavih.hashing` | `"yavih-protocol-5"` | the input / scoring / protocol payload shape or a typed view changes, or the shared reader `yavih.sample.iter_samples` changes what is handed over (its semantics are part of every folder's input side, its bytes are not hashed) |
| `DESCRIBE_SCHEMA` | `yavih.versions` / `yavih.hashing` | `"yavih-describe-2"` | the `describe()` subset that `describe_sha256` covers changes; `describe_subset` asserts it implements this schema, so both change together |
| `protocol_version` | each folder's `benchmark.yaml` | per folder (`yavih bench info <name>`); the folder is authoritative | see [protocol_version bump or a moved hash](#protocol_version-bump-or-a-moved-hash) |
| `yavih.__version__` | package | PEP 440 from the git tag `vX.Y.Z` (setuptools-scm writes `yavih/_version.py`; else the distribution metadata; else `0.0.0+unknown`) | set by the release tag ([releasing](releasing.md)); provenance only, never hashed |
| `GIT_HASH` | `yavih.versions`, resolved lazily by `yavih.buildinfo.commit()` | 40-hex commit (`+dirty` when tracked files are modified) or `null` | never bumped; provenance only, never hashed |

`GIT_HASH` is `git rev-parse HEAD` of the checkout `src/yavih` runs from; an installed wheel reads the commit
setuptools-scm recorded in `yavih/_version.py` (absent for a wheel built from an sdist or without git metadata);
otherwise `null`. Importing `yavih.versions` never spawns `git`. `versions.writer_stamp()` returns
`{version, commit}` and is recorded by the feature writers (sidecars, `_meta.json`), so an `E_HASH_INPUT`
message can name both the writing and the reading yavih; results files record the yavih version and `GIT_HASH` too.

Reading rules:

- A reader refuses a `format_version`, `schema_version` or `results_version` greater than it knows.
- `features validate` and the probe read feature format 4 only; any other format is `E_FORMAT` ("re-extract").
- `report` reads `result.json` records of the current schema; an older schema is refused unless `report.migrate`
  registers an upgrade step for it ([migration](#migration)). `--merge-results` skips a results file of another
  `results_version` with a notice.

## Canonicalization

`canon(x)` is the only serialization any digest uses:

```python
json.dumps(normalize(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
```

`normalize` maps tuples to lists, numpy scalars to Python scalars (duck-typed, so `yavih.hashing` never imports
numpy), keeps bools as bools and floats as floats (serialized with `repr`, the shortest round-trip form, so `1.0`
never collapses into `1`). Dict keys must be `str`. Anything else, bytes included, raises `TypeError`; there is
never a `default=str`. NaN and infinities are refused. The hashed bytes are the UTF-8 encoding of the text.

Helpers: `sha256_canon(x) = sha256(canon(x))` as lower-case hex, `hash12(d) = d[:12]` (display and paths),
`rng_seed(text)` = the big-endian integer of `sha256(text)`, used to seed `numpy.random.default_rng` for every
deterministic draw.

### Folder source provenance
A folder's `reader.py` and `probe.py` never enter a hash. Each is digested as provenance:
```
source_sha256(text) = sha256(normalized_source(text))
Benchmark.folder_source = {"probe": source_sha256(probe.py) | null, "reader": source_sha256(reader.py) | null}
```
`normalized_source` parses the file with `ast.parse`, removes every bare string statement (module, class and
function docstrings, attribute docstrings, any other string used as a statement), and serializes the tree as
canonical JSON node by node, `[NodeType, [field, value], ...]`, without source positions, without the `kind` and
`type_comment` fields and without empty or absent fields. It does not use `ast.dump`, whose output differs between
Python versions; the digest is the same on every supported interpreter (`tests/test_hashing.py` pins one). A
comment, docstring or formatting edit leaves it unchanged; any edit to what the code does changes it.

`folder_source` is written into every sidecar's `benchmark` block, into `result.json` (`benchmark.folder_source`)
and into the results file `configs`. The report raises `W_FOLDER_SOURCE_DIFFERS` when results in one column were
scored by different folder code: they are comparable by declaration (the conformance fixture holds every version
within tolerance) and the warning exists for diagnosis. A file that does not parse raises `SyntaxError` at load.

## manifest_hash

```
manifest_hash = sha256(b"manifest:" + sha256(manifest).hex + b"\nclasses:" + (sha256(classes).hex | b"none"))
```

over the decompressed `manifest.csv` bytes and the `classes.csv` bytes (`none` when the folder has no classes file).
`Benchmark.load` first checks both files against `manifest.sha256` / `manifest.classes_sha256` in `benchmark.yaml`
(`ManifestError` on a mismatch). See [benchmark folder](benchmark-folder.md) for the manifest itself.

## protocol_hash = H(input_hash, scoring_hash)

```
component_digests(c) = {name: sha256(canon(value)) for name, value in c.items()}
input_hash    = sha256(canon({"schema": PROTOCOL_SCHEMA, "part": "input",   "components": component_digests(INPUT)}))
scoring_hash  = sha256(canon({"schema": PROTOCOL_SCHEMA, "part": "scoring", "components": component_digests(SCORING)}))
protocol_hash = sha256(canon({"schema": PROTOCOL_SCHEMA, "input_hash": input_hash, "scoring_hash": scoring_hash}))
```

`Benchmark.load` computes all three from the raw parsed yaml blocks (`spec.raw`), the manifest digest and
`yavih.versions` only, so they are identical in every env and reproducible outside yavih. `Benchmark.hashes`
(`hashing.ProtocolHashes`) carries the three digests, both `{component: digest}` maps and both `{component: value}`
maps (the exact hashed payloads; `yavih bench hash <name> --explain` prints their canonical text); the input digest
map is written into every sidecar so `E_HASH_INPUT` names the component that differs.

Every yaml block enters through a **typed view**: values only, no prose and no display setting, so a wording edit
never forks a hash. `bench lint` check `hashed` (`E_HASHED_PROSE`) refuses a sentence inside any hashed view.

### Input side

What decides the features a bridge sees and must produce, and everything `features validate` reads from the
benchmark (`hashing.input_components`):

| component | value |
|---|---|
| `name` | `benchmark.yaml[name]` |
| `manifest_hash` | as above: rows, media keys and labels (whole file, conservatively) |
| `manifest` | `{"key": manifest.key, "columns": manifest.columns}`: which rows and media a Sample carries |
| `splits` | `benchmark.yaml[splits]`: split values and roles |
| `streams` | `benchmark.yaml[streams]`: handed-over streams, layouts, layout params (root names, never resolved paths) |
| `input_contract` | `hashing.input_contract_view`: `{info, facts, materialize}` (`handed_over`, one sentence per stream, is documentation) |
| `output_contract` | `benchmark.yaml[output_contract]`: what the features must carry |
| `readout` | the readout name |
| `readout_inputs` | `hashing.input_readout_view`: `{"params": ..., "protocol": ...}` (below) |

`readout_inputs` holds the keys that are present among:

- `params`: `readout_params` restricted to `T`, `visual_stream`, `conditioning_stream` and the readout's
  `Readout.input_keys`;
- `protocol`: `protocol` restricted to `views`, `segments_scored`, `num_segments`, `segment_offset`,
  `segment_seconds`, `target_region`, every name in `input_contract.facts`, the fact named by
  `output_contract.requires.annotated_frame.fact`, and the readout's `Readout.input_keys`.

A readout whose output checks read further `protocol` / `readout_params` keys declares them in `Readout.input_keys`
([adding a readout](adding-a-readout.md)); they then enter the input side.

### Scoring side

How features become numbers; none of it reaches the bridge (`hashing.scoring_components`):

| component | value |
|---|---|
| `protocol_version` | `benchmark.yaml[protocol_version]`: the declared version of the folder's scoring behaviour (code, yavih's or the folder's, is pinned by it, never hashed) |
| `protocol` | `hashing.protocol_view`: every `protocol` key as written, `train_draw` restricted to `{cap, seed, balance}` (the draw's description lives in `manifest.upstream.draw`) |
| `readout_params` | the whole `readout_params` block |
| `metrics` | `hashing.metrics_view`: `{headline, list, floors, controls, chance, ci, bootstrap, aggregation}`, absent ones as `null`; `table` (report columns) and `tolerance` (the conformance bound) stay out |
| `external` | `hashing.external_view`: `[{name, version_pin}, …]` sorted by name (`pip` is an install instruction, `used_for` prose) |
| `targets` | `[{split, kind, file, sha256, index_column}, ...]` sorted by split (`index_column` null when absent), or `null` |
| `label` | `manifest.label` as written |
| `readout` | `{"name": readout, "version": READOUT_VERSION[readout]}` (an unknown readout raises `ValueError`) |

### What moves which hash

- Feature sets are checked against `input_hash` only. Bumping `protocol_version`, a metric's `version_pin` or a
  `READOUT_VERSION`, changing a metric, a target file or a scoring-only protocol value moves `scoring_hash` and
  `protocol_hash`, hence `eval_hash`: it re-probes and never re-extracts.
- Editing `probe.py` or `reader.py` moves no hash: it moves `folder_source` (provenance) and, if the numbers moved,
  fails the conformance fixture until `protocol_version` is bumped.
- A `protocol` / `readout_params` key that is also in `readout_inputs` moves both sides.
- `protocol_version` is scoring-side: a version bump that keeps the inputs keeps the features.
- `manifest_hash` is whole and input-side: labels share the file, and a manifest change is always a protocol event.
- Over-inclusion is allowed (a stream's layout is hashed although a tar copy of the same bytes would score the
  same); omitting a number-changing benchmark input is not.

Deliberately not hashed: `title`, `aliases` (one protocol, several names), `capability` and `header` (how the
benchmark is grouped and presented), `changelog` (it records the hashes),
`source` (provenance; the staged copy is checked by `bench verify` and `W_MEDIA_FP`), `budget`, `data` (site roots,
tar-index paths, derived-file listings), `obtain`, `manifest.file`, `manifest.sha256`, `classes`,
`classes_sha256` (their content is `manifest_hash`), `manifest.upstream` (including `draw`), the `source` / `root`
of a target, `input_contract.handed_over`, `external[].pip` / `used_for`, `metrics.table`, `metrics.tolerance`,
`protocol.train_draw` beyond `cap` / `seed` / `balance`, and every source file (`probe.py`, `reader.py`, yavih).
Covered by declared versions instead of bytes: the folder's `probe.py` / `reader.py` (`protocol_version`, held by
the [conformance fixture](conformance.md)), the shared reader `yavih.sample.iter_samples` (`PROTOCOL_SCHEMA` bump)
and the readouts with the `yavih.metrics` functions they list (`READOUT_VERSION` bump, held by every folder's
fixture on that readout).

## describe_sha256

```
describe_sha256 = sha256(canon(describe_subset(describe())))        # DESCRIBE_SCHEMA "yavih-describe-2"
```

`describe_subset` drops the top-level keys `DESCRIBE_SHA_EXCLUDED = {"weights_ref", "batch_tolerance",
"batch_size"}` and the `decoder` block of every `input.<stream>` entry (`DESCRIBE_PROVENANCE_INPUT_KEYS`). A path,
a declared tolerance, a batch size (batch invariance is the canaries' job) and a host decoding-library build are
recorded in every sidecar and never compared. A waiver the bridge declares in `describe()` is covered: which checks
a bridge waives is its decision, so changing it is another configuration. An operator's `--waive` or a matrix
`waive` is not part of `describe()` and changes no digest. See [bridge](bridge.md) for the `describe()` keys.

`describe_sha256` is a consistency and provenance digest of the bridge's configuration, never identity, never
comparability, never part of a path or a key:

1. one feature set has one `describe_sha256`: every sidecar of a split carries the same one and the same subset
   (`E_DESCRIBE_MISMATCH`, [validation](validation.md));
2. resuming an extraction compares it; a shard written under another one is moved to `_stale/<utc>/` and
   re-extracted ([features](features.md));
3. probe DONE markers and `inputs_hash` carry it, with `features_fingerprint`, so changed features are re-probed;
4. `run_extract --expect-profile` refuses a live configuration that drifted from the `bridge check` profile
   (`E_CONFIG_DRIFT`);
5. the report tells configurations of one row key apart (`W_BRIDGE_CHANGED`, newest shown).

A stored digest is compared as stored: a record written under another `DESCRIBE_SCHEMA` compares as a mismatch
(another configuration), and the feature set is re-extracted on resume.

### model and reading

`model` and `reading` are names, not digests. They are copied from `describe()` into the profile, sidecars,
`_meta.json`, `NOT_APPLICABLE.json`, `result.json`, DONE markers, unit records and results files, and are checked
against their charsets: `model` `[A-Za-z0-9_.-]{1,64}`, `reading` `[A-Za-z0-9_-]{1,32}`, each starting with a
letter or digit (never `.`, `..`, `_stale` or a hidden name). They are the only model-side part of any key: run
directory `runs/<benchmark>/<model>/<reading>/`, results directory `<results_root>/<model>/`, report row
`(benchmark, eval_hash, model, reading)`. `plan` takes them from `profile.describe` and never computes them. A null
`weights_sha256` changes no key; it is the warning `W_WEIGHTS_UNFINGERPRINTED`.

## probe_hash

```
probe_hash = sha256(canon({"schema": PROBE_SCHEMA, "readout": cfg.readout, "head": cfg.head,
                           "train": {lr, lr_grid, weight_decay, max_epochs, batch_size, patience, warmup_fraction,
                                     schedule, val_fraction, loss, selection, max_tokens_per_forward, precision}}))
```

`hashing.probe_hash` takes a `ProbeConfig` ([readouts](readouts.md)), any object with those attributes, or a dict;
it never imports `yavih.probes`, and a config missing a field raises `KeyError`. The `train` field list is explicit
(`PROBE_TRAIN_FIELDS`), so adding a field to `ProbeConfig` does not move any digest. The seed is not included: it is
a replicate axis of the result key. The payload is the probe's pinned head and training protocol plus the task
sizes it takes from the benchmark (label kind, class count, and the readout's `task_sizes`, e.g. `T`, grid, target
size); it holds nothing from the bridge (no `N`, `D` or dtype) and nothing from the site.

## features_fingerprint

```
features_fingerprint = sha256(canon({"format_version", "dtype", "N", "D", "pos_mode", "pos_sha256",
                                     "shards": [[lo, hi, sha256], ...], "n_present"}))
```

`shards` is in row order; the caller sorts by `lo` and the function does not reorder, so a wrong order is a
different fingerprint. It is an integrity and staleness fingerprint of one extracted feature set: `_meta.json`
carries it ([features](features.md)), the probe's DONE marker records it and the probe re-runs when it changed,
results files record it as provenance. It never groups or splits report rows.

## Unit ids and inputs_hash

```
unit_id = sha256(canon({"kind", "benchmark", "model", "reading", "split", "shard": [i, n] | null,
                        "protocol_hash" | null, "input_hash" | null, "probe_hash" | null, "seed" | null,
                        "plan_schema": PLAN_SCHEMA_VERSION}))[:12]
```

An extract unit passes `input_hash` and `protocol_hash: null` (a scoring edit re-plans no extraction); probe and
report units pass `protocol_hash` and `input_hash: null`. The unit id is a scheduling handle: `[i, n]` follows from
the profile's throughput and feature size, which is irrelevant to comparability.

```
inputs_hash = sha256(canon({"cmd": identity_cmd(cmd), "protocol_hash", "input_hash", "probe_hash",
                            "model", "reading", "describe_sha256"}))
```

with the same `input_hash` / `protocol_hash` rule per kind. It is what the unit computes, the bridge's
configuration included: staleness, not comparability. `identity_cmd` (`yavih.plan.units`) drops the interpreter,
the values of `--out`, `--bridge`, `--expect-profile`, `--results` and `--decode-workers`, and the run-directory
positional of `probe run` / `report build`. The launcher, `env`, `cwd`, worker counts and `needs` are logged in
`logs/last.json` and never hashed, so a plan built on the host is not stale in a container that mounts the project
elsewhere. `plan --pending` treats a done unit whose `logs/last.json` `inputs_hash` differs as stale and re-emits
it. See
[execution](execution.md).

## eval_hash and comparability

```
eval_hash = sha256(canon({"protocol_hash": protocol_hash, "probe_hash": probe_hash}))
```

`hashing.eval_hash` has no schema tag of its own (each component carries one) and refuses an empty component. It
is the comparability key: everything that decides what a number means, the benchmark and the probe, and nothing
else. Reports show it (12 hex) next to `protocol_version`; it names the probe directory
`runs/<benchmark>/<model>/<reading>/probe/<eval_hash12>/` and is recorded in `result.json`, DONE markers, probe
unit records and results files ([results](results.md)).

Two results are **comparable** iff they have the same `benchmark.name` after alias resolution, the same
`eval_hash`, and both are `standard` (`report.compare.comparable`). Equal `eval_hash` implies equal
`protocol_hash` and `probe_hash`, hence equal `input_hash` and `scoring_hash`, the same `external` versions, the
same declared `protocol_version`, head and training. The code that scored them may differ (`folder_source`, the
yavih version): results under one hash agree within the folder's conformance tolerance by construction. Bridge
configuration, dtype, batch size, weights and site never make two results incomparable and never make them
comparable.

Report columns are keyed `(benchmark, eval_hash)` and rows `(benchmark, eval_hash, model, reading)`:

- several `bridge.describe_sha256` values under one row key (the same model and reading under another
  configuration) raise `W_BRIDGE_CHANGED`; the newest configuration is shown, older ones under `--detail`;
- within the shown configuration, results of different `features_fingerprint`s are never averaged: the newest
  feature set is shown, `W_FEATURES_DIFFER`;
- a differing `coverage.eval.ids_sha256` within a column is marked "population differs";
- differing `benchmark.folder_source` within a column raises `W_FOLDER_SOURCE_DIFFERS` (comparable; for diagnosis).

## Migration

Hashes move whenever a hashed input moves, including a `PROTOCOL_SCHEMA` bump that changes nothing numerically.
Features and results written under an earlier hash stay usable when the folder records that the move changes
nothing (`supersedes` / `supersedes_input` in its changelog); there is no central table. A packaged folder at 1.0
carries `supersedes: []`: the lists fill only when a later release moves a hash.

### Recorded hashes in the changelog

Each `benchmark.yaml` `changelog` entry ([benchmark folder](benchmark-folder.md)) may carry four hash keys, each a
64-hex digest. They sit outside every hashed payload (no input or scoring component reads `changelog`), so a folder
recording its own digest is not circular.

| key | meaning |
|---|---|
| `protocol_hash` | the protocol hash this entry's version computes |
| `input_hash` | the input hash this entry's version computes |
| `supersedes` | earlier protocol hashes of the same benchmark: their results are this entry's results (`report.migrate` maps them to this entry's `protocol_hash`) |
| `supersedes_input` | earlier input hashes whose feature sets stay valid (the bridge is handed exactly the same thing and must produce the same output) |

```yaml
changelog:
  - version: 1
    date: "2026-01-15"
    reason: first release
    invalidates: false
    protocol_hash: bbbbbbbbbbbb…  # what the folder computes now (64 hex)
    input_hash: cccccccccccc…
    supersedes: [aaaaaaaaaaaa…]   # earlier protocol hashes, same numbers ([] in every packaged folder at 1.0)
    supersedes_input: [dddddddddddd…]  # earlier input hashes, same features
```

**The newest changelog entry records the hashes the folder computes.** `bench lint` check `hashes` fails with
`E_CHANGELOG_HASH` until it does (also `hash_history.check_recorded`, and `tests/test_hashing.py` for every packaged
folder). The message names the exact lines to write:

- both keys missing: add `protocol_hash` and `input_hash` with the computed values;
- `protocol_hash` differs: the conformance fixture decides (it is keyed on `protocol_version`, not on the hash).
  If it still passes, set the new value and append the old one to `supersedes`; if it fails, bump
  `protocol_version`, add a new entry (`invalidates: true`) recording both hashes and re-record the fixture;
- `input_hash` differs: set the new value; append the old one to `supersedes_input` only if the bridge is still
  handed exactly the same thing (otherwise features must be re-extracted);
- an entry never lists its own digest under `supersedes` / `supersedes_input`.

The newest entry's `version` must also equal `protocol_version` (a schema error at load), and entries are newest
last with strictly increasing versions. `bench new` writes the hashes for a scaffold; see
[contributing a benchmark](contributing-a-benchmark.md).

### yavih.hash_history

`hash_history.history()` merges the changelogs of every folder yavih can resolve (packaged,
`$YAVIH_HOME/benchmarks`, `$YAVIH_BENCH_PATH`, `yavih.benchmarks` entry-point packs) plus every folder a
`Benchmark` was loaded from in this process (`Benchmark.load` calls `register`; a loaded folder's history wins over a
scanned one of the same name), so an out-of-tree benchmark's history travels with it. The yaml is read without full
schema validation: a malformed digest is skipped and `bench lint` reports it. Both lists are followed transitively;
a cycle stops where it closes.

- `upgrade_protocol_hash(ph)`: the newest protocol hash that supersedes `ph`, else `ph` unchanged.
- `effective_input_hash(block)`: the input hash a feature set's `benchmark` block (sidecar, `_meta.json`,
  `_layout.json`) stands for: its `input_hash` followed through `supersedes_input`; `None` without one. Sidecar
  resumption, `features validate` and the probe compare this value with the loaded benchmark's `input_hash`
  (`E_HASH_INPUT` on a mismatch, with the differing components and both yavih versions).

### report.migrate

`report.migrate.migrate(result)` returns a current-schema copy of a `result.json` record, in memory only; the file
on disk is never rewritten. A record of the current `RESULT_SCHEMA_VERSION` is copied unchanged; a newer one raises
`MigrationError` ("newer than this yavih"), and so does an older one for which no upgrade step is registered
("older than this yavih"). A future schema bump registers one step `_UPGRADES[old] = fn` that turns a record of
schema `old` into one of `old + 1`; at 1.0 `_UPGRADES` is empty.

Then, for every record, `benchmark.protocol_hash` is mapped to the newest hash of its benchmark through the
changelog `supersedes` lists, and when it moved, `eval_hash` is recomputed. For example, a folder whose newest
entry records `protocol_hash: bbbbbbbbbbbb…` with `supersedes: [aaaaaaaaaaaa…]` files a result scored under
`aaaaaaaaaaaa…` in the `bbbbbbbbbbbb…` column. Any hash no changelog supersedes is kept and stays its own column.

`report.migrate.PROTOCOL_HASH_UPGRADES` is a read-only `{old: newest}` view over the folders' `supersedes` lists,
not a table.

### protocol_version bump or a moved hash

Bump `protocol_version` (and add a changelog entry with `invalidates: true`) for any change that can change
numbers: to what the hashes cover (`protocol`, `splits`, `streams`, `input_contract`, `output_contract`, `external`,
`metrics`, `readout_params`, `targets`, `manifest.key` / `label` / `columns`, or the frozen manifest) and to the
code behind them (`probe.py`, `reader.py`, a readout or metric of yavih). The conformance fixture
([conformance](conformance.md)) is the detector: a code change that moves the fixture's numbers beyond tolerance
is `E_CONFORMANCE_DRIFT` until the version is bumped and the fixture re-recorded.

Do not bump it when the hash moves but numbers cannot change; record the new hash on the newest entry and list the
old one under `supersedes` (and `supersedes_input` when the input side moved but hands over the same thing):

- a `PROTOCOL_SCHEMA` bump alone;
- a typed-view change or a `READOUT_VERSION` bump that leaves this folder's fixture passing;
- a value moved between hashed keys with identical meaning.
A `probe.py` or `reader.py` edit of any kind moves no hash and needs no changelog entry unless the fixture fails.

A `READOUT_VERSION` bump moves `scoring_hash` of every folder using that readout; whether it changes a folder's
numbers decides between the two paths above.

### Training order

The train permutation (the order in which each epoch visits the rows, drawn from the seed) is part of the numbers
but not of `probe_hash`; changing it requires a `PROBE_SCHEMA` bump, so a stored `result.json` and a new one at
equal `(probe_hash, seed)` never differ. The trainer's loader thread, its prefetch depth and the feature backend
(RAM or memmap) leave the order unchanged.
