# The model registry

A registry is a collection of recorded scores, each next to the hash it was produced under, so that models can be
compared without rerunning anything: two entries with the same `benchmark` and `eval_hash` are comparable, whoever
ran them and whichever implementation scored them, because the hash covers the declaration
([hashing](hashing.md)) and the folder's conformance fixture holds every implementation to it
([conformance](conformance.md)). This page is the normative reference for the entry contract, the registry layout,
`yavih registry export` / `check`, and `report build --registry`. The implementation is `yavih.registry`.

## Layout

A registry is a text-only directory, its own git repository: never the yavih repository (which ships protocols,
not scores) and never a project folder. Entries are grouped the way the report builds columns, one directory per
comparable column:

```
<registry>/
  entries/<benchmark>/<eval_hash12>/<org>/<model>/<reading>.json   one entry = one comparable score (~2 KB)
  protocols/<benchmark>/<protocol_hash12>.json                      the hashed payload + the conformance fixture
```

`<org>` is the submitter's namespace (`[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`, like a model name), so user-chosen model
names cannot collide across submitters: a registry row is `<org>/<model>`. A thousand models over ten benchmarks
and three readings is on the order of 60 MB of JSON.

## An entry

`REGISTRY_ENTRY_VERSION = 1`. Written by `yavih registry export` from a results-store file
([results](results.md#results-store)); nothing is typed by hand but `--org`.

| block | fields | role |
|---|---|---|
| `registry_version`, `id` | `id` = `<benchmark>/<eval_hash12>/<org>/<model>/<reading>` | the file's identity; `check` refuses a file not filed under it |
| `identity` | `benchmark` (canonical), `reading`, `readout`, `readout_version`, `protocol_version`, `protocol_hash`, `input_hash`, `scoring_hash`, `manifest_hash`, `probe_hash`, `eval_hash`, `standard` | the comparability key and its parts (`input_hash` / `scoring_hash` are null when the exporting yavih could not resolve the folder at that hash) |
| `model` | `name` = `<org>/<model>`, `org`, `model`, `bridge`, `weights_ref`, `weights_sha256`, `describe_sha256`, `source_sha256`, `describe` | who was measured; recorded, never a key |
| `score` | `headline`, `metrics`, `ci95`, `std`, `floors` (by metric), `floor_names`, `controls`, `n_seeds`, `seeds: [{seed, metrics, floors, controls, ci95, created}]`, `higher_is_better`, `waived` | the numbers; per-seed values when the results file's `unit_results` were readable at export |
| `population` | `coverage` per role (`present`, `nominal`, `ids_sha256`; or the results file's fraction), `features_fingerprint` | what was scored; gates comparability (`W_REGISTRY_COVERAGE`) |
| `conformance` | `passed`, `skipped`, `protocol_version`, `compared`, `tolerance`, `expected_sha256`, `checked_with`, `reason` | the exporting yavih ran the folder's fixture at the entry's hash; `passed: true` is required for acceptance |
| `provenance` | `yavih_version`, `commit`, `date`, `folder_source`, `external`, `probe_config`, `results_file`, `results_file_sha256`, `unit_results`, `env`, `exported_with`, `replaces` | how to diagnose a disagreement; never compared for identity |

A resubmission replaces the file and records the previous file's sha256 under `provenance.replaces`.

A protocol payload (`protocols/<benchmark>/<protocol_hash12>.json`) holds the folder's `bench hash --explain`
payload (the canonical text and digest of every hashed component: anyone recomputes the hash from it with sha256
alone), its `conformance/expected.json`, its metric list and tolerance, `min_eval_coverage`, `folder_source` and
the earlier hashes the folder `supersedes`. It is written once per `(benchmark, protocol_hash)` by the first export
that resolves the folder at that hash.

## Rules

- **Comparable** = the report's rule: same benchmark after alias, same `eval_hash`, both `standard`
  (`yavih.report.compare.comparable`). Non-standard entries are stored, shown as `(nonstandard)`, never ranked.
- **Coverage**: an entry whose eval coverage is below the folder's `splits.min_eval_coverage` is shown as partial,
  never ranked (`W_REGISTRY_COVERAGE`).
- **Ties**: differences within a metric's recorded conformance tolerance are ties; a table must not order them.
  `report build` marks them (`≈` in md, `tie` in json and csv) and never orders tied rows by value
  ([results: ties](results.md#ties)).
- **Provenance differs, comparability does not**: entries under one `eval_hash` with different `folder_source` or
  yavih versions are comparable by construction (the fixture held both) and flagged for diagnosis
  (`W_FOLDER_SOURCE_DIFFERS` in the report).
- **Acceptance** (`yavih registry check`, the registry repository's CI): every entry parses
  (`E_REGISTRY_SCHEMA`), is filed under its id (`E_REGISTRY_ID`), has `eval_hash = H(protocol_hash, probe_hash)`
  (`E_REGISTRY_HASH`), a protocol payload that recomputes `protocol_hash` and records a tolerance
  (`E_REGISTRY_PROTOCOL`, `E_REGISTRY_SCHEMA`), and `conformance.passed: true` (`E_REGISTRY_CONFORMANCE`: a
  skipped or unverified conformance is refused; export from a yavih that has the folder and its pinned packages).
- **Filing under the original hash**: entries are never rewritten when a folder's hash moves. When a table is
  built, `report.migrate` follows the folders' `supersedes` chains and puts an entry recorded under a superseded
  hash into the current column; the `protocols/` payloads keep each chain readable without a yavih checkout.

## Commands

```
yavih registry export <results_root|results_file> --org ORG --out DIR [--model NAME]... [--include-nonstandard]
yavih registry check DIR
yavih report build 'runs/*' --registry DIR
```

`export` reads every `results_*.json` under `<results_root>/<model>/` (or one file), registers every ok,
single-reading, standard entry (joins and non-ok entries are skipped and listed; `--include-nonstandard` registers
non-standard probes as such), runs the folder's conformance fixture when the folder resolves here at the entry's
hash, and writes the protocol payload on first use. An entry whose conformance did not pass here is written with
`W_REGISTRY_UNVERIFIED`; `check` refuses it until a yavih with the folder and its packages re-exports it.

`report build --registry DIR` adds the accepted entries to the comparison as rows (model `<org>/<model>`, flag
`registry`, one record per recorded seed) under the same comparability rule as every local result, so "compare
my model against the registry" needs no rerun of anything ([results](results.md#--registry)).

## Submitting

1. Score the model with yavih (`yavih plan ...`, `yavih report build --merge-results`), which leaves
   `<results_root>/<model>/results_<timestamp>.json`.
2. `yavih registry export <results_root> --org <your org> --out <registry checkout>`.
3. `yavih registry check <registry checkout>`; fix what it names.
4. Open a pull request on the registry repository; its CI runs the same check.

Open points, decided per registry: whether `protocols/` payloads are copied (as here, self-contained) or resolved
from the yavih release that computed them, and who the submitter is (proposed: the pull request author, not a
field of the entry).
