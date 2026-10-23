# Contributing a benchmark

The steps below take a benchmark from nothing to a pull request against yavih. Follow them in order: each step
names the commands to run and ends with a **Done when:** line, and the next step assumes it holds. The contract a
folder must satisfy is the [benchmark folder](benchmark-folder.md) reference; this page links to it instead of
repeating it, and spells out only what the code checks without a reference page saying so. The reasons behind
the rules are in [design](design.md).

Throughout, `<name>` is the new benchmark's name (`[a-z][a-z0-9_]*`, the folder name, the `name:` in
`benchmark.yaml`) and `F=src/yavih/benchmarks/<name>` is its folder.

## 0. Fork, clone, install

```bash
gh repo fork Bizilizi/yavih --clone && cd yavih     # or fork on GitHub, then git clone your fork
git switch -c bench/<name>
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # or your CUDA build
pip install "setuptools>=77" "setuptools-scm>=10" wheel
pip install -e ".[media,dev]"
```

The version comes from git tags, so clone with history (no `--depth`). The `dev` extra brings pytest, ruff and the
`media` and `probe` extras ([development](development.md#setting-up)).

**Done when:** `yavih --version` prints a version and `yavih bench lint --all` exits 0 on the untouched checkout.

## 1. Scaffold

```bash
yavih bench new <name> --readout clf --dir src/yavih/benchmarks                # copies the readout's default folder
yavih bench new <name> --like vggsounder --dir src/yavih/benchmarks            # or the folder closest to yours
```

`--like` picks the folder whose machinery is copied; without it the readout's default is used (`clf` esc50, `seq`
ave, `grid` nyuv2_depth, `box` vggss_box). Pick the folder whose streams and readings are closest to the new
benchmark: `vggsounder` or `audioset20k` for an audio-visual one whose bridge serves `a`, `v` and `av`, `k400`
for video only.

What the scaffold holds, and what has to be replaced:

| file | where it comes from | replace with |
|---|---|---|
| `manifest.csv.gz`, `classes.csv` | copied from `--like` (a loadable folder needs them) | your frozen manifest (step 3) |
| `bridge_template.py` | copied from `--like` | your benchmark's template (step 6) |
| `reader.py`, `probe.py` | copied from `--like` when it has them, with a `# TODO: copied verbatim from <like> by yavih bench new` banner as the first lines | your own, or delete them (steps 3 and 7) |
| `benchmark.yaml` | `--like`'s yaml renamed: `protocol_version: 1`, one `changelog` entry with the hashes the scaffold computes, `aliases: []`, a `header:` block and `obtain.how` with `TODO`, `--like`'s `capability` under a `TODO` comment | your protocol (step 2) |
| `README.md`, `setup.md` | written from the folder contract, `TODO` in every section | your docs (step 5) |
| `make_manifest.py`, `prepare/01_obtain.sh` | stubs that exit 2 | your freeze and prepare steps (steps 3 and 4) |
| `tests/test_folder.py` | generated: the shared checks of `yavih.testing.folder`, `--like`'s values pinned, two tests that skip with a `TODO:` reason | your folder tests (step 9) |

Never copied: `--like`'s `curation.json`, `notes/`, `tests/`, `prepare/` and `assets/`. Until step 3, every number
in the folder is `--like`'s.

**Done when:** `yavih bench lint <name>` exits 0 with only `W_TODO` warnings, and
`python -m pytest $F -q -p no:cacheprovider -o addopts="--import-mode=importlib"` passes (two tests skip).

## 2. Write the protocol in `benchmark.yaml`

Every key is described in the [schema](benchmark-folder.md#benchmarkyaml). Decide, in this order:

1. **Source and obtainability.** `source` (kind, URLs, a pinned `revision`, licence) and `obtain: {tier, how}`:
   `tier` is `auto` (fetchable unattended), `manual-download` (a person downloads it from an official portal),
   `needs-licence` (a human signs a licence), `needs-data-owner` (only its holder has it) or `site-only`; `how` says
   in one or two sentences who acts and how. Only public upstream files enter the protocol: the manifest is frozen
   from the official annotation files anyone can download (step 3), each pinned by sha256.
2. **Data roots.** `data.roots.<root>` with a `default` (`"${data.root}/<name>"`) and a `description` of the
   staged layout; scripts and tests resolve roots through it ([configuration](configuration.md)).
3. **Streams and the input contract.** Which stored file each stream reads (`streams`), what is handed over
   (`input_contract.handed_over`, one sentence per stream), the media facts guaranteed (`input_contract.info`) and
   the label facts (`input_contract.facts`). Hand over the stored file; transform it only when the label
   requires it.
4. **Labels, splits and the readout.** `manifest.label`, `splits` roles (`train`, `eval`, `floor`, `folds`,
   `unused`), `readout`, `readout_params`, `targets` (grid only).
5. **The output contract.** `requires` (position axes, stream tags, grid coverage or scored seconds) and
   `target_frame`.
6. **Metrics.** Headline, list, floors, controls, chance, intervals. Every name resolves in the metric registry or
   the folder's `probe.py` `METRICS`; every `protocol` key is consumed by a metric, the readout, `probe.py`
   (`PROTOCOL_KEYS`) or `input_contract.facts`.
7. **Authors' code.** A scorer the authors ship is listed under `external` (import name, the exact
   `pip install …` line, version pin, what it is used for) and called from `probe.py` (step 7). It is never added
   to `pyproject.toml`.
8. **The header.** `header: {name, title, venue, abstract}`: the display name, the paper title,
   `"<authors> · <venue> <year>"` and a neutral abstract of at most 60 words. The README header image is rendered
   from it (step 11).
9. **Capability.** `capability: [...]`: what the benchmark measures, one or more of `audio`, `visual`,
   `audio-visual`, `temporal`, `spatial`, `retrieval` (`yavih.suites.CAPABILITIES`, the vocabulary suites group
   by). The benchmarks index and `yavih bench list` group by it; tag every capability a suite might list the
   benchmark under (a synchronisation task is `[audio-visual, temporal]`). The scaffold keeps the `--like`
   folder's tags under a `TODO` comment. Not hashed.
10. **Budgets.** `budget.clips` equals the frozen rows per split (checked); `media_gb` is measured on the staged
   media; `extract_min_per_1k_clips_ref` and `probe_minutes_ref` (one entry for the readout) are measured, never
   guessed. Write first values from your own bridge check (`clips_per_s`) and probe wall time; the trial run
   (step 13) checks them.

**Done when:** `yavih bench info <name>` loads the folder, and `yavih bench lint <name>` reports no `E_SCHEMA`,
`E_OBTAIN`, `E_HEADER`, `E_CAPABILITY` or `E_EXTERNAL`.

## 3. Freeze the manifest: `make_manifest.py`

`make_manifest.py` re-derives `manifest.csv.gz` and `classes.csv` from the public upstream files, byte for byte.
Follow the conventions of `src/yavih/benchmarks/esc50/make_manifest.py`:

- **Docstring.** Names every input (upstream file, commit, sha256), the outputs, the curation applied, and a
  `Usage::` block with the commands below.
- **Pinned inputs.** The upstream sha256s and commit are module constants; a file with another sha256 is refused
  (`SystemExit` naming both digests).
- **Roots.** `--media-root` defaults to `config.folder_root(HERE, "media_root", args.media_root)`
  (`from yavih import config`; `HERE = Path(__file__).resolve().parent`): the folder's declared root resolved
  through the configuration layers without loading the manifest being rewritten. Catch `config.ConfigError` and
  exit with "pass --media-root or configure …".
- **`--no-media --metadata FILE`** re-freezes from the public metadata alone (no presence or header checks); this
  is what the folder test runs where no media is staged.
- **`--check`** re-derives and compares with the committed `manifest.csv.gz` (decompressed) and `classes.csv`,
  writes nothing, and exits 1 when either differs. `--out DIR` (default `HERE`) writes elsewhere.
- **Bytes.** Rows in a fixed column and row order through `yavih.freeze`: `csv_bytes`, `gzip_bytes` (no file name,
  `mtime=0`), `write_if_changed`; a train cap is a seeded draw with `class_balanced_subsample(labels, max_clips,
  seed)`.
- **Output.** The `manifest.upstream` block (`source`, `sha256`, `rows`, `upstream_files`, `curation`) and the
  manifest digests are printed as JSON on stdout; paste them into `benchmark.yaml` (`manifest.sha256` and
  `manifest.classes_sha256` are over the decompressed bytes).

Rules that keep a freeze honest: cap the train population in the manifest (a frozen draw), never at probe time,
and never cap eval; record every exclusion under `curation.excluded` with its reason, so that `listed + excluded ==
total_upstream`; freeze only from files anyone can download again.

```bash
python $F/make_manifest.py --media-root /path/to/staged/<name>       # write the manifest, print the upstream block
python $F/make_manifest.py --check                                    # re-derive and compare, write nothing
python $F/make_manifest.py --no-media --metadata /path/to/annotations.csv --check
```

**Done when:** `--check` exits 0, a second run without it reports both files unchanged, and `yavih bench lint
<name>` reports no `E_BUDGET` or `E_CURATION`.

## 4. Prepare steps

Replace `prepare/01_obtain.sh` with the numbered steps the data needs ([prepare/ steps](benchmark-folder.md#prepare-steps)):
download with expected sha256s, verify, build indexes, materialize only what a label requires, re-freeze. A step
resolves its roots with `ROOT="${ROOT:-$(yavih config get <name>.media_root)}"`, never a site path, and runs as
`yavih bench prepare <name> <step>` with `FOLDER` and `YAVIH_BENCH` set.

```bash
yavih bench prepare <name>                 # lists the steps
yavih bench prepare <name> 01              # runs one
yavih bench verify <name> --census         # manifest hash, presence per split, a read sample, the media-info table
```

**Done when:** every step runs on a clean staging directory, and `yavih bench verify <name> --census` reports eval
coverage at or above `splits.min_eval_coverage`, `labels_ok`, a successful read sample and the media-info table
written.

## 5. `setup.md` and `README.md`

Write both against their contracts: [README.md](benchmark-folder.md#readmemd-contract) (for people: header image,
links line, `## Description`, `## Citation`, `## In yavih`) and [setup.md](benchmark-folder.md#setupmd-contract)
(for the agent that sets the benchmark up: the six sections in order, ending the contract sections with a copy of
the yaml blocks). Both are shipped in the wheel: link only what a pip install can open (a public dataset page, a
`https://github.com/Bizilizi/yavih/blob/main/docs/<page>.md` URL, a `yavih …` command). Maintainer notes may live
in the folder's `notes/`, which is never shipped and never linked.

**Done when:** `yavih bench lint <name>` reports no `E_README`, `E_SETUP`, `E_CITATION` or `E_NOTES_LINK`.

## 6. `bridge_template.py`

The template is what `yavih bridge new <name> --model M` copies for users, and what an agent fills with its model's
code. It carries the benchmark's streams, the `input` rules to declare, `used` reporting and positions in media
coordinates ([bridge](bridge.md)). These parts are read by code, so write them exactly as shown:

- **The model literal.** `describe()` holds the line `"model": "TODO-model",` on its own line, indented, with one
  space after the colon and a trailing comma (`bench lint` check `template` matches
  `^\s+"model": "[A-Za-z0-9_.-]{1,64}",`). The optional file label is a class attribute `name = "TODO-model-<name>"`.
  `yavih bridge new … --model M` replaces `"TODO-model"` in both with `M` and reports what it filled (`filled`:
  `describe()['model']`, `Bridge.name`).
- **The reading.** Either a literal `"reading": "v",` in `describe()`, or, for a template that serves several
  readings, an `ARGS` entry `"reading": ArgSpec(type=str, default="av", choices=("a", "v", "av"), help=…),` and
  `"reading": self.reading` in `describe()`. Every value is in `a`, `v`, `av`. `yavih.lint.template_readings` reads
  these values; suites, `plan init` and the header chips use them, and `bridge new --reading R` fills the literal
  or the `ArgSpec` default.
- **The preprocessing source slot.** A docstring line starting with `Preprocessing source` followed by an indented
  block whose first line starts with `TODO` (or an inline `Preprocessing source: TODO <repo>@<commit>
  <file>::<function>`). `bridge check` warns `W_NO_PREPROCESSING_SOURCE` until the user pastes the model's own
  code there ([bridge](bridge.md)).
- **Unfilled values.** `weights_ref`, `norm.source` carry `TODO` and the `norm` constants are `None`, so `bridge
  check` refuses an unfilled copy with `E_TEMPLATE_UNFILLED`.
- **Decode in `prepare()`.** The template decodes in [`prepare()`](bridge.md#prepare) and returns a dict (decoded
  frames or waveform, their times, the `used` entry); `encode()` reads `sample.prepared` and the stream `info`,
  never `bytes()` or `open()` (the bytes are released after `prepare`).

**Done when:** `yavih bench lint <name>` reports no `E_TEMPLATE`; `yavih bridge new <name> --model toy --out
/tmp/b/` writes a copy whose `filled` lists `describe()['model']`; `yavih bridge check /tmp/b/toy_<name>.py --bench
<name> --synthetic` refuses it with `E_TEMPLATE_UNFILLED` and nothing else.

## 7. `probe.py` (only when needed)

Write a `probe.py` only when the registry lacks what the protocol needs: a metric (`METRICS`), protocol keys it
consumes (`PROTOCOL_KEYS`), target construction (`targets`), the authors' scorer (`score`) or a cross-reading join
(`join`) ([probe.py hooks](benchmark-folder.md#probepy-hooks); new readouts: [adding a readout](adding-a-readout.md)).
Module scope imports only the standard library, numpy and yavih core; the authors' package is imported at call time
through `yavih.external.require(name, pip, version_pin)` with the matching `external` entry. A `probe.py` copied by
the scaffold is deleted when the folder needs none.

**Done when:** `yavih bench lint <name>` reports no `E_IMPORT`, and `test_metric_names_resolve` passes.

## 8. Record the hashes and the conformance fixture

A new folder has one `changelog` entry, `version: 1`, with `protocol_version: 1`. Its `protocol_hash` and
`input_hash` must be the ones the folder computes; `yavih bench lint <name>` (`E_CHANGELOG_HASH`) prints the exact
lines to write. Until the pull request is merged, keep that one entry and update its two hashes after every edit
that moves them; nothing goes under `supersedes`, since no result was ever filed under the earlier digests. Every
later change to a merged folder follows [hashing: migration](hashing.md#migration).

The hashes cover the declaration, never the code: what pins the folder's scoring behaviour (its `probe.py`, the
metrics and readout it uses) to `protocol_version: 1` is the conformance fixture
([conformance](conformance.md)). Record it once the protocol, `probe.py` and `external` entries are final:

```bash
yavih bench conformance <name> --record
```

writes `conformance/expected.json` (the scoring path's numbers on a generated population; a few KB, no inputs
stored). Re-record after every later edit that moves them, as long as the folder is unmerged: while the changelog
has its one entry, `--record` overwrites the record without comparing, including the fixture the scaffold recorded
for the copied protocol. Once the folder has a second changelog entry, a moved number means a `protocol_version`
bump (or `--same-version --reason TEXT`), and `--record` refuses to overwrite differing values otherwise.

**Done when:** `yavih bench lint <name>` reports no `E_CHANGELOG_HASH` and no `E_CONFORMANCE_*`, and
`python -m pytest tests/test_hashing.py tests/test_conformance.py -q -p no:cacheprovider -o addopts="--import-mode=importlib"` passes.

## 9. Folder tests

Make `tests/test_folder.py` this folder's: replace the pinned `--like` values (streams, clips per split, readings,
output contract, external packages), add the folder's own facts (columns, classes, curation sums, README.md and
setup.md facts), and write the two tests the scaffold skips:

- `test_make_manifest_check`: run `make_manifest.main(["--no-media", "--metadata", <file>, "--check"])` on the public
  metadata (or on a copy rebuilt from the committed manifest, as esc50's `test_make_manifest_check_from_metadata`
  does) and assert exit 0 and the printed `upstream` block equal to the yaml's;
- `test_reader_smoke_synthetic_raw`: write synthetic media for a few manifest rows into a temporary root and assert
  what `Benchmark.from_dir(HERE, data={…}).reader(split, rows=…)` hands over: streams, `info` keys, `meta`, canaries.

Tests that need staged media skip when no data root is configured (`yavih.testing.folder.site_root`).

**Done when:** the folder's tests pass with no `TODO:` skip, and `yavih bench lint <name>` reports no `W_TODO`.

## 10. Generated tables, lint, tests

```bash
python -m yavih.doctables --write            # the benchmark tables in README.md, docs/ and the skill references
yavih bench lint <name>
python -m pytest tests src/yavih/benchmarks -q -p no:cacheprovider -o addopts="--import-mode=importlib"
ruff check src tests
```

The benchmark lists in the repository are generated from the folders; never edit a row between
`yavih:bench-table` markers by hand.

**Done when:** `python -m yavih.doctables --check` exits 0, `yavih bench lint <name>` reports nothing, the full
test suite passes and ruff is clean.

## 11. Header image

With Chrome or Chromium installed (`--chrome PATH` or `$CHROME` when it is not on the path):

```bash
python -m yavih.headers <name>               # writes assets/header.png and assets/header.src.sha256
```

Without one, write the sha file only and let the pull request render the image:

```bash
python -m yavih.headers <name> --html-only   # writes assets/header.src.sha256 (and assets/header.html: do not commit it)
```

then, once the pull request is open, download the `headers` artifact of its `headers` workflow run, copy
`src/yavih/benchmarks/<name>/assets/header.png` from it into the folder and commit it
([header image](benchmark-folder.md#header-image)).

**Done when:** `assets/header.png` and `assets/header.src.sha256` are committed together and `yavih bench lint
<name>` reports no `W_HEADER_STALE`.

## 12. Synthetic trial

```bash
yavih bench trial <name> --synthetic --readout-fast
```

runs the whole chain (conformance fixture, stand-in, oracle bridge, `bridge check`, extract, validate, probe, report)
in process on a generated stand-in of the folder ([CLI](cli.md#bench-trial---synthetic)). Beyond the conformance
stage (which runs the real folder's scoring path, `probe.py` and `external` included), it does not exercise the
real data; step 13 does.

**Done when:** it exits 0, every stage in `data.stages` is `ok`, and the headline clears its floor.

## 13. Trial run with a fresh agent (required)

A fresh agent with only the built wheel and the shipped skill sets the benchmark up and runs a toy model through
it, on a small sample of the real data. The protocol, the worker prompt, the report template and the pass
criteria are in [trial run](trial-run.md).

**Done when:** the report meets every reviewer criterion of [trial run](trial-run.md#reviewer-criteria), the
friction it found is fixed in the branch (or explained), and the budgets are within 2× of the measured values or
updated.

## 14. The pull request

```bash
git push -u origin bench/<name>
gh pr create --repo Bizilizi/yavih --title "Add <name> benchmark" --label benchmark --body-file pr-body.md
```

The body follows this template. It has no checklist: what it states is checked by CI and by the trial report.

````markdown
## Summary

<name>: <the task in one sentence>. Readout `<readout>`, headline `<metric>`, readings <a / v / av>.
Splits and frozen rows: <train n / eval n>. Scaffolded from `<like>`.

## Source and licence

<upstream URLs, the pinned revision, the licence, which upstream files the manifest is frozen from (with sha256)>

## Obtain tier

`<tier>`: <who must act and how, as in `obtain.how`>

## Budgets

| key | value | measured on |
|---|---|---|
| `media_gb` | <x> | <staged copy> |
| `extract_min_per_1k_clips_ref` | <x> | <model, GPU, decode workers> |
| `probe_minutes_ref` | <readout: x> | <GPU> |

## Trial report

<the trial-run report, pasted verbatim>

## Header contact sheet

<the rendered header (assets/header.png), or a link to the `headers` artifact of this pull request>
````

**Done when:** the pull request is open with that title, the `benchmark` label and the filled body.

## 15. What CI runs

| workflow | check | what it runs |
|---|---|---|
| `ci.yml` | ruff | `ruff check src tests` on Python 3.12 |
| `ci.yml` | pytest (Python 3.10 to 3.13) | `pytest tests src/yavih/benchmarks` with CPU torch, ffmpeg and `.[media,dev]` and no benchmark data (tests that need staged media skip): every folder's lint and tests, the recorded hashes, the generated tables, the docs links |
| `ci.yml` | build and smoke-test the wheel | builds the sdist and wheel, checks that no `notes/`, `assets/` or `docs/` entered the wheel, installs it into a clean venv without torch and runs `yavih bench lint --all` and `yavih bench files esc50` there |
| `headers.yml` | render the header images | on pull requests touching `src/yavih/benchmarks/**`: renders every header and a contact sheet into the `headers` artifact, and fails when a committed `assets/header.src.sha256` is not current |

**Done when:** every check of the pull request is green.

## Out-of-tree folders

A folder can also live outside the repository: in a directory on `$YAVIH_BENCH_PATH` (or
`$YAVIH_HOME/benchmarks/<name>`), or in a pip-installed benchmark pack that registers its folders under the
entry-point group `yavih.benchmarks`. An out-of-tree folder never shadows a packaged name. It follows the same
steps without the pull request: scaffold with `--dir` pointing at that directory, skip `doctables` (the generated
tables list packaged folders only) and steps 14 and 15, and in the trial run (step 13) hand the worker the folder on
`$YAVIH_BENCH_PATH`, or the pack's wheel next to yavih's.
