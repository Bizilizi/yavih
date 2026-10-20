# Conformance fixtures

A benchmark's hashes cover its declaration, never its code ([hashing](hashing.md)). This page is the normative
reference for what holds the code to the declaration instead: the per-folder **conformance fixture**
(`conformance/expected.json`), the generated population it is computed on, the per-metric tolerance, the readout
bands, and how a third party claims a protocol hash from outside yavih. The implementation is `yavih.conformance`
(numpy + yavih core, importable without torch), the command `yavih bench conformance`, the lint check `conformance`
and the first stage of `yavih bench trial --synthetic`.

## The principle

`protocol_hash` is computed from the frozen manifest, typed views of `benchmark.yaml` and declared versions
(`protocol_version`, `READOUT_VERSION`). A folder's `probe.py` and `reader.py`, and yavih's own readouts and
metrics, may be refactored, re-implemented in another framework or ported by a third party without moving it. Two
scores recorded under one hash are therefore comparable only if every implementation that claims the hash produces
the same numbers. The fixture is the operational meaning of "the same": the numbers the folder's whole scoring path
produces on a fixed population, recorded at the folder's `protocol_version` and compared within a per-metric
tolerance. A change that moves them is `E_CONFORMANCE_DRIFT` until the version is bumped and the fixture
re-recorded; a change that leaves them is a refactor and needs nothing.

The fixture checks numbers, not code. It never asserts correctness (the folder tests and `tests/test_metrics_*.py`
do that); it asserts that nothing moved since the version was declared.

## What is checked

Every packaged folder carries one fixture, whatever its readout. At check time `yavih.conformance.compute` runs:

1. the readout's target construction on real manifest rows (`clf` labels and sets, `seq` segments, `box` cells;
   `grid` targets are fabricated, since dense targets live in site files);
2. the floor outputs the readout defines (`Readout.floor_outputs`) from the drawn train rows;
3. `yavih.readouts.base.evaluate_metrics` with the folder's resolved metric specs: the metrics registry with the
   folder's `protocol` values, the folder's `probe.py` `METRICS` and `score` hook, the pinned `external` package
   where one exists, the floor role (avsync), the folds (esc50), the shuffled control (vggss_box);
4. for a reading-routed `score` hook (one that takes `reading=`), once per reading in the hook's `READINGS` tuple.

`metrics`, `floors` and `controls` are compared; `ci95` is recorded, never compared (an interval depends on the
bootstrap draw, not on the protocol).

## The population

No input is stored. Everything is generated at check time by **sha256 in counter mode** (`conformance.stream`),
never by a numpy distribution method (NEP 19 leaves those free to change between releases), so the population is
the same on every platform, Python and numpy:

- rows: the `n` rows of each role's split(s) with the smallest `sha256(f"{manifest_hash}:conformance:{split}:{role}:{row}")`,
  a uniform draw over the folder's real manifest, so real label kinds, empty labels and exclusive sets appear at
  their own rate (`conformance.select_rows`);
- outputs: fabricated in the readout's raw output shape from the same stream — logits `[n, C]` (clf) or
  `[n, T, C]` (seq), depths in `(0, max_depth]` `[n, S, S]` (grid, with ~5 % holes in the targets), boxes
  `(cx, cy, w, h)` `[n, 4]` (box, plus a shuffled-control arm and stable derangements).

| readout | eval | train | other |
|---|---|---|---|
| `clf` | 256 | 256 | floor role 64; under folds, 64 per fold (eval and train alike) |
| `seq` | 64 | 64 | |
| `grid` | 16 maps at `protocol.target_size` | 16 | |
| `box` | 128 | 128 | shuffled arm 128 |

A split smaller than the population is used whole. The sizes keep `1 / n` above the default tolerance, so one row
scored differently still fails while an interpolation-level difference passes. In memory a check is at most a few
tens of MB (the depth maps); on disk only `expected.json` exists. `GENERATOR_VERSION` tags the draw and the
fabrication; bumping it makes every fixture stale (`E_CONFORMANCE_STALE`) and every folder re-records at its
current version.

## Tolerance

Absolute, per metric, on the metric's reported scale. Default `conformance.DEFAULT_TOLERANCE` = `0.001` (a tenth
of a point on a fraction); a folder overrides it per metric:

```yaml
metrics:
  tolerance: {mae_seconds: 0.01}   # optional; names from list, floors or controls; >= 0
```

`tolerance` is not hashed: it changes no number, it bounds what "same protocol" accepts. It is recorded in
`expected.json` and belongs in every registry entry; two scores under one hash that differ by less than it are
ties, not an order. Values compare as `|got - expected| <= tolerance`; `null` (a metric not applicable to a
reading), booleans and lists compare exactly element-wise, NaN equals NaN.

## expected.json

```json
{
 "schema": "yavih-conformance-1",
 "benchmark": "esc50",
 "protocol_version": 1,
 "generator": 1,
 "population": {"eval": 320, "train": 256, "floor": 0, "folds": 5},
 "tolerance": {"acc_mean": 0.001, "...": 0.001},
 "values": {"*": {"metrics": {"acc_mean": 0.0156, "...": 0.0}, "floors": {"acc_floor": 0.0156}, "controls": {}, "ci95": {"...": [0.0, 0.03]}}},
 "recorded_with": {"yavih_version": "…", "commit": "…", "protocol_hash": "…", "scoring_hash": "…",
                   "folder_source": {"probe": null, "reader": null}, "externals": {}}
}
```

`values` is keyed by reading for a reading-routed scorer (`"a"`, `"v"`, `"av"` for vggsounder), else `"*"`.
`recorded_with` is provenance: which yavih, commit, hashes, folder source and external package versions produced
the record. A `same_version_reason` is stored when the record was overwritten at the same version.

## Commands and codes

```
yavih bench conformance <name|path>... [--all]                     check
yavih bench conformance <name> --record                            write expected.json
yavih bench conformance <name> --record --same-version --reason T  overwrite differing values at the same version
yavih bench conformance <name> --record --force                    overwrite without comparing
```

| code | level | meaning |
|---|---|---|
| `E_CONFORMANCE_MISSING` | error | no `conformance/expected.json`; record one |
| `E_CONFORMANCE_STALE` | error | recorded at another `protocol_version` or generator; re-record |
| `E_CONFORMANCE_DRIFT` | error | a value outside its tolerance at the same version; the message names the first six |
| `E_CONFORMANCE_SCHEMA` | error | the file does not parse, is another schema, or records another benchmark |
| `E_CONFORMANCE_UNSUPPORTED` | error | a readout without a population, or a scoring refusal unrelated to an external package |
| `W_CONFORMANCE_SKIPPED` | warning | a pinned external package is absent (or the installed build fails the folder's own guard); nothing compared |

`bench lint` check `conformance` reports the same codes; the repository test
`tests/test_conformance.py::test_packaged_folder_fixture_is_current` checks every packaged folder and skips (with
the reason) where the external package is absent. CI installs the pinned packages it can.

### Which path after a change

The fixture is keyed on `protocol_version`, not on the hash, so it answers the question a moved hash raises:

- the hash moved (a schema bump, a readout version bump, a value moved between hashed keys) and the fixture still
  passes: record the new hash on the newest changelog entry and list the old one under `supersedes`
  ([hashing: migration](hashing.md#migration)); nothing to re-record;
- the fixture fails: bump `protocol_version` (a new changelog entry with `invalidates: true`), then `--record`;
- the fixture fails but the protocol did not change (a bug fixed in `probe.py`, a wrong fixture): `--record
  --same-version --reason TEXT`, which stores the reason; results under that version were produced by code that
  gave other numbers, and the registry's `folder_source` provenance tells them apart.

A `READOUT_VERSION` bump moves the hash of every folder on that readout; each one runs its fixture and takes one of
the paths above (`yavih bench conformance --all`, then `--all --record` for the folders whose numbers moved).

## Readout bands

Probe training is not bit-reproducible across torch builds, so the readouts are held to a band, not a point. Each
`yavih/readouts/<name>.py` declares next to its `VERSION`:

```python
CONFORMANCE = {"oracle_min": 0.9, "random_max": 0.5}
```

the synthetic end-to-end run (`tests/test_e2e_synthetic.py`) asserts the oracle bridge's headline at or above
`oracle_min` and the random bridge's at or below `random_max`. A trainer or readout change that leaves the band
bumps `VERSION`, which moves `scoring_hash` of every folder on that readout; those folders' fixtures then say
whether their scoring numbers moved too.

## Claiming a protocol hash from outside yavih

An implementation that is not yavih (another framework, a port) claims a benchmark's `protocol_hash` by
reproducing two things:

1. the hash itself: `yavih bench hash <name> --explain` prints the canonical JSON and digest of every hashed
   component and the combination formula ([hashing](hashing.md#protocol_hash--hinput_hash-scoring_hash)); the
   payload is plain JSON and sha256, recomputable with any language's standard library;
2. the fixture: generate the population exactly as `yavih.conformance` does (the sha256 stream and the row draw
   are specified above and pinned by `tests/test_conformance.py`), score it, and match `expected.json` within the
   recorded tolerance.

A registry entry carries the conformance result it was produced under; an implementation that matches the fixture
is, for the registry, the same protocol.
