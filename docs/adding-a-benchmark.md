# Adding a benchmark

A benchmark is one folder. This guide walks through creating one, freezing its manifest, and getting it
past `yavih bench lint` and its tests. The contract the folder must satisfy is the
[benchmark folder](benchmark-folder.md) reference; the reasons behind it are in [design](design.md).

A folder can live in three places, and the workflow is the same for all of them:

- **packaged**: `src/yavih/benchmarks/<name>/`, shipped in the wheel (a pull request to yavih);
- **out of tree**: any directory listed in `$YAVIH_BENCH_PATH` (or `$YAVIH_HOME/benchmarks/<name>`);
- **a benchmark pack**: a pip-installed package that registers its folders under the entry-point group
  `yavih.benchmarks`.

An out-of-tree folder never shadows a packaged name.

## 1. Scaffold

```bash
yavih bench new mybench --readout clf              # copies the default folder for the readout
yavih bench new mybench --like vggsounder --dir D  # or start from a specific folder, in D/mybench
```

`--like` picks the folder whose machinery is copied (frozen manifest, classes, an optional `reader.py` and
`probe.py`, the bridge template); the defaults are `esc50` (clf), `ave` (seq), `nyuv2_depth` (grid) and
`vggss_box` (box). `benchmark.yaml` is renamed and reset to `protocol_version: 1`; `README.md`,
`setup.md`, `make_manifest.py`, `prepare/` and `tests/test_folder.py` are written from the folder contract, and a
`header:` block (for the README header image) is added to the yaml.
Every fact you must supply is marked `TODO`. The scaffold loads and lints clean except for `TODO` warnings:
it is a starting point, not a protocol.

For a packaged benchmark, scaffold into `src/yavih/benchmarks/` (`--dir src/yavih/benchmarks`).

## 2. Write the protocol in `benchmark.yaml`

Decide, in this order (each block is described in the [schema](benchmark-folder.md)):

1. **Source and obtainability**: `source` (kind, URLs, a pinned `revision`, licence) and `obtain` (how an
   agent gets the data, and whether a human must act).
2. **Streams and the input contract**: which stored file each stream reads (`streams`), what is handed
   over (`input_contract.handed_over`, one sentence per stream), which media facts are guaranteed
   (`input_contract.info`) and which label facts are handed over (`input_contract.facts`). Hand over the
   stored file; transform it only when the label requires it.
3. **Labels, splits and the readout**: `manifest.label`, `splits` roles (`train`, `eval`, `floor`, `folds`,
   `unused`), `readout`, `readout_params`, `targets` (grid only).
4. **The output contract**: `requires` (position axes, stream tags, grid coverage or scored seconds) and
   `target_frame`.
5. **Metrics**: headline, list, floors, controls, chance, intervals. Every name resolves in the metric
   registry or in the folder's `probe.py` `METRICS`; every per-benchmark `protocol` key must be consumed by a
   metric, the readout, `probe.py` (`PROTOCOL_KEYS`) or `input_contract.facts`.
6. **Authors' code**: when the authors ship a scorer, list it under `external` (import name, exact pip
   command, version pin, what it is used for) and call it from `probe.py` through
   `yavih.external.require`. It is never added to `pyproject.toml`.

## 3. Freeze the manifest

`make_manifest.py` re-derives `manifest.csv.gz` and `classes.csv` from the benchmark's public upstream files
(official CSVs, split lists, devkits, annotations) and, where needed, the staged media, byte for byte: fixed column order, fixed row order, a gzip header without file name or mtime. Use
the helpers in `yavih.freeze` (`gzip_bytes`, `csv_bytes`, `write_if_changed`, `class_balanced_subsample`
for a seeded, class-balanced train draw). The script prints the `manifest.upstream` block (upstream file
hashes and the curation counts); paste it into `benchmark.yaml`.

Rules that keep a freeze honest:

- Cap the train population in the manifest itself (a frozen draw), never at probe time; never cap eval.
- Record every exclusion under `curation.excluded` with its reason; the sums are re-checked
  (`listed + excluded == total_upstream`).
- Freeze only from public upstream files, so anyone can re-run the freeze; name them in the docstring with their
  commits and hashes.

Then run the folder's prepare steps on staged data and verify:

```bash
yavih bench prepare mybench            # list the numbered steps in prepare/
yavih bench prepare mybench 01         # run one (FOLDER and YAVIH_BENCH are set for it)
yavih bench verify mybench --census    # manifest hash, presence per split, a read sample, the media-info table
```

## 4. Record the hashes

The newest `changelog` entry must record the `protocol_hash` and `input_hash` the folder computes:

```bash
yavih bench lint mybench               # E_CHANGELOG_HASH prints the exact lines to write
```

While the folder is new, keep one changelog entry and update its hashes as you edit. Once results exist,
every later edit that moves a hash needs a decision (see [hashing](hashing.md#migration)):

- the numbers can change → add an entry with `protocol_version + 1` (and `invalidates: true` when old
  numbers are void);
- the meaning is unchanged (a refactor, a payload-schema change) → keep the version, record the new hashes
  and list the old `protocol_hash` under `supersedes` (and the old `input_hash` under `supersedes_input`
  when existing feature sets stay valid).

Comments and docstrings in `probe.py` / `reader.py` never move a hash; code and string literals do.

## 5. Write the docs

- `README.md` is for people: the header image, a links line (paper, homepage, dataset, setup.md),
  `## Description` (the task, data and splits), `## Citation` (BibTeX) and `## In yavih` (the protocol, what the
  bridge gets and returns, variants and aliases, published reference numbers with their citation, and how the
  manifest is re-frozen). No changelog or checklist: the yaml `changelog` is the version record.
- `setup.md` is for the agent that sets the benchmark up, and only that: how to obtain the data, independent
  dependencies (the `external[].pip` lines verbatim, or "None"), the input and output contracts ending with a copy
  of the yaml blocks, prepare and verify, gotchas.
- Fill the `header:` block of `benchmark.yaml` (display name, paper title, "authors · venue year", an abstract of at
  most 60 words) and, in a source checkout with Chrome or Chromium, render the image with
  `python -m yavih.headers mybench` (it writes `assets/header.png`, which the README's first line shows).

Both files are shipped. Link only what a pip install can open: a GitHub URL, a public dataset page, a
`yavih ...` command. Maintainer notes may live in the folder's `notes/` (never shipped, never linked).

## 6. Bridge template and folder tests

- `bridge_template.py` is what `yavih bridge new mybench --model NAME` copies for users. It declares a
  literal `describe()["model"]` placeholder and a `reading`, and shows the declared rules and `used`
  reporting the benchmark's streams need.
- `tests/test_folder.py` pins the folder: manifest hashes, curation sums, split assignment, docs against the
  yaml, metric names, lint clean (helpers in `yavih.testing.folder`). Tests that need staged media skip when
  no data root is configured.

## 7. Lint, tables and tests

```bash
yavih bench lint mybench                         # errors exit 2; TODO markers are warnings
python -m yavih.doctables --write                # packaged folders: regenerate the benchmark tables
python -m pytest src/yavih/benchmarks/mybench -q -p no:cacheprovider -o addopts="--import-mode=importlib"
python -m pytest tests -q -p no:cacheprovider -o addopts="--import-mode=importlib"
```

For a packaged folder, `tests/test_hashing.py` also checks the changelog hashes of every packaged
benchmark, and the repo `README.md`, `src/yavih/benchmarks/README.md` and the skill's
`references/data.md` get their rows from `doctables`; never edit those tables by hand.

## 8. Smoke it end to end

Before claiming the folder works, connect a real or reference bridge:

```bash
yavih bridge new mybench --model mymodel
yavih bridge check bridges/mymodel_mybench.py --bench mybench --synthetic   # no data needed
yavih bridge check bridges/mymodel_mybench.py --bench mybench --out mybench.profile.json
```

and run one extract → validate → probe → report cycle on a small model.
