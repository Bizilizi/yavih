# Developing yavih

A guide for people who continue to develop yavih itself: where things live, which invariants the code
keeps, how to test, and the conventions every module follows. The rationale behind the invariants is in
[design](design.md); task-specific guides are [adding a benchmark](adding-a-benchmark.md),
[adding a readout](adding-a-readout.md) and [releasing](releasing.md).

## Setting up

```bash
git clone https://github.com/Bizilizi/yavih && cd yavih
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
.venv/bin/pip install "setuptools>=77" "setuptools-scm>=10" wheel
.venv/bin/pip install -e ".[media,dev]"
sudo apt-get install ffmpeg                                                      # the ffprobe fallback tests
```

The `dev` extra brings pytest, ruff, scikit-learn (tests pin the numpy metrics against it) and the
`media` and `probe` extras. The version comes from git tags (see [releasing](releasing.md)), so clone with
history; an editable install writes `src/yavih/_version.py`, which is never committed.

## Repository layout

```
src/yavih/            the package (everything below ships in the wheel)
tests/                repo tests: contract, integration, golden, docs and import-boundary tests
docs/                 this documentation (not shipped in the wheel)
assets/               header.html → header.png (the README header), benchmark-header/ (the folder header template)
.github/workflows/    ci.yml (ruff, pytest on 3.10–3.13, wheel smoke test), release.yml (PyPI + GitHub Release)
pyproject.toml        metadata, extras, package data, setuptools-scm, pytest and ruff settings
```

### Module map

| module | responsibility |
|---|---|
| `__init__.py` | public API: `Benchmark`, `BenchmarkView`, `Sample`, `RawStream`, `Bridge`, `FeatureWriter`, `FeatureReader`, `validate_features`, `NotApplicable`, `MediaUndecodable`, `grid_pos`, `__version__`; `probes`, `readouts`, `report`, `plan` load lazily |
| `__main__.py` | `python -m yavih` = the `yavih` command |
| `versions.py` | every version constant (formats, schemas, readout versions, `GIT_HASH`, `writer_stamp()`); stdlib only |
| `buildinfo.py` | which source commit is running (checkout `git rev-parse` or the wheel's recorded commit) |
| `config.py` | layered site configuration (`Settings`, data roots, runs/results/scratch roots, `ENV_VARS`) |
| `_toml.py` | a minimal TOML reader for Python 3.10 without `tomli` |
| `resources.py` | CPU, memory, RAM budget and decode-worker defaults from env, SLURM, PBS, cgroups |
| `deadline.py` | the one `--deadline` parser shared by `exec`, `extract` and `probe run` |
| `hashing.py` | canonical JSON, normalized source, manifest/input/scoring/protocol/describe/probe/eval/unit hashes |
| `hash_history.py` | the folders' changelog hashes merged (`supersedes`), used by validation, migration and lint |
| `benchmark.py` | `Benchmark`: `benchmark.yaml` schema, manifest and classes, roles, targets, contracts, metric resolution, data status, folder resolution |
| `sample.py` | `RawStream`, `Sample` and the generic reader `iter_samples` (stored bytes + facts, canaries) |
| `media/` | `rules.py` (evaluate declared rules, numpy only), `tarindex.py` (read tar members in place), `info.py` (header probes, census), `decode.py` and `ffmpeg.py` (optional decode helpers); no protocol semantics |
| `external.py` | `require()`: assert-import and version check of benchmark authors' packages |
| `bridge.py` | `Bridge` base class, `BenchmarkView`, `describe()` validation, `fingerprint_state_dict`, `grid_pos` |
| `bridges/examples/` | `hf_reference.py`, the one shipped reference bridge (`doctor --selftest`) |
| `extract.py` | `yavih extract` and `yavih bridge check`: reader → read-ahead → bridge → `used` check → writer; invariance checks, audit hook, precision, deadline |
| `waivers.py` | waivable and never-waivable codes, `describe()["waive"]` and `--waive` parsing |
| `io/` | feature format: `format`, `writer`, `reader`, `sidecar`, `shards`, `layout`, `claim`, `validate`, `finalize`, `inspect`, `diff`, `stage` |
| `metrics/` | metric registry (lazy functions, sklearn-free import) and the classification, segment, depth, box metrics and intervals |
| `probes/` | torch heads (attentive, dense), losses, trainer, pinned `ProbeConfig` (torch-free), floors, checkpoints |
| `readouts/` | readout registry and the `clf`/`seq`/`grid`/`box` drivers; `base.py` runs a probe and writes `result.json` and the results store |
| `report/` | results index, comparability, tables, in-memory migration of older results, results store merge, cross-reading joins |
| `plan/` | `matrix.yaml`, unit records, resource estimates, `exec`, status, calibrate, `plan run --local` |
| `lint.py` | `yavih bench lint`: the benchmark-folder contract as a tool |
| `scaffold.py` | `yavih bench new`: scaffold a folder that passes lint |
| `freeze.py` | shared helpers of the folders' `make_manifest.py` (byte-identical re-freeze, seeded train draw) |
| `doctables.py` | the benchmark lists generated from the folders into README.md, `docs/benchmark-folder.md`, the index and the skill's `references/data.md` and `references/readouts.md` |
| `headers.py` | `python -m yavih.headers`: renders the folders' README header images and the repository's `assets/header.png` and records each folder render's HTML sha256 (a maintainer tool; the images are not shipped) |
| `shipped.py` | one command surface for shipped files (`bench files/show/prepare`, `bridge new`, `skill show`) |
| `cli.py` | argparse mirror of the API, the `--json` envelope, exit-code mapping |
| `testing/` | shipped fixtures: synthetic benchmarks, `random_bridge`, `oracle_bridge`; `folder.py` helps folder tests |
| `benchmarks/<name>/` | one folder per packaged benchmark (see [benchmark folder](benchmark-folder.md)); `benchmarks/README.md` is the generated index |
| `skill/yavih/` | the agent skill: `SKILL.md` and `references/{bridge,cli,data,readouts,slurm}.md` |

## Invariants

These hold across the code base; tests enforce most of them.

### The core is torch-free

The core installs into a model's environment with `pip install --no-deps yavih` and must never touch its
torch. Module-scope imports are restricted:

| allowed at module scope | where |
|---|---|
| stdlib only | `versions.py`, `buildinfo.py`, `config.py`, `resources.py`, `deadline.py`, `hashing.py`, `hash_history.py`, `external.py`, `readouts/registry.py`, `plan/`, `report/` |
| numpy + pyyaml + stdlib | the rest of the core: `benchmark.py`, `sample.py`, `bridge.py`, `extract.py`, `io/`, `media/` (decoders inside functions), `metrics/` (scikit-learn inside functions), `cli.py` |
| torch | `probes/` (not `probes/config.py`), `readouts/` drivers (their `READOUT` descriptors must import without torch), `bridges/examples/`, `testing/__init__.py` |

A folder's `probe.py` and `reader.py` import only stdlib, numpy and yavih core at module scope; torch and
authors' packages are imported inside functions, the latter only through `yavih.external.require`.
`tests/test_import_boundary.py` imports every core module (discovered from the tree, so new files are
covered) in a subprocess where `import torch` fails; `tests/test_extensibility.py` checks that the readout
registry and `Benchmark.load(...).protocol_hash` work without torch.

### The folder is the source of truth

Everything about a benchmark lives in its folder. Nothing outside it repeats a folder fact by hand: the
benchmark lists in `README.md`, `docs/benchmark-folder.md`, `src/yavih/benchmarks/README.md` and the skill
references are generated by `python -m yavih.doctables --write` (checked by a test), and `yavih bench lint` checks
every folder against the contract. A change to a benchmark is a change to its folder only.

### Hashes cover only what yavih controls

Any change that can change a number must move a hash, and nothing else may. In practice:

- Editing a hashed part of a folder (see [hashing](hashing.md)) moves its `input_hash` or `scoring_hash`.
  The newest `changelog` entry in `benchmark.yaml` must record the hashes the folder computes;
  `yavih bench lint` (`E_CHANGELOG_HASH`) prints the exact lines to write. Bump `protocol_version` when
  numbers can change; otherwise list the old hash under `supersedes` (and `supersedes_input` when feature
  sets stay valid).
- `probe.py` and `reader.py` are hashed as normalized syntax trees: comments and docstrings are free to
  edit, but **string literals and code are hashed** (an error message in a folder's `probe.py` is part of the
  scoring hash).
- Changing a payload shape bumps the matching schema constant in `versions.py`; changing a readout's head,
  loss, floor or metric bumps its `VERSION`.
- `tests/test_hashing.py` pins the hash payloads and formulas and checks that every packaged folder's
  changelog records the hashes it computes. A failure means you changed what a number means: either revert
  or record it in the folder changelog.

### Shipped files cite only what a pip install can open

Everything under `src/yavih/` ships in the wheel; `docs/` and maintainer notes do not. Shipped text (code
comments included) never refers to `docs/...` by relative path; it names a `yavih ... --help` command or
uses the full GitHub URL `https://github.com/Bizilizi/yavih/blob/main/docs/<page>.md`. Folder files are
checked by `yavih bench lint` (`E_CITATION`, `E_NOTES_LINK`), the skill by `tests/test_skill.py`.
Agents reach shipped files only through the CLI (`yavih bench files|show|prepare`, `yavih bridge new`,
`yavih skill show`), which answers identically from a wheel and a checkout.

Shipped and repository text describes yavih as it is: no development history, review notes or names of
internal projects, and no site specifics (cluster names, accounts, mount points, home directories; examples use
placeholders such as `<partition>` and `/path/to/...`). `tests/test_no_internal_refs.py` enforces this with a
repository-wide `git grep`.

### Atomic, idempotent, recorded

Every write goes through a temporary file and `os.replace` on the same filesystem. Units are idempotent:
a DONE marker carries the hashes it was produced under, a mismatching marker counts as absent, and stale
output is moved to `_stale/`, never deleted. Every artefact records the yavih version and source commit
that wrote it (`versions.writer_stamp()`).

## Testing

```bash
.venv/bin/ruff check src tests
.venv/bin/python -m pytest tests src/yavih/benchmarks -q -p no:cacheprovider -o addopts="--import-mode=importlib"
```

The second line is what CI runs (Python 3.10–3.13, CPU torch, ffmpeg installed, no benchmark data). It
covers the repo tests and every folder's `tests/test_folder.py`; all benchmark folders share the test file
basename, hence importlib import mode.

**Isolation.** `tests/conftest.py` sets `YAVIH_TESTING=1`, points `YAVIH_HOME`, `YAVIH_CONFIG` and
`XDG_CONFIG_HOME` at temporary locations and removes every other `YAVIH_*` variable, so no test reads a
developer's real configuration or site caches. To reproduce CI's environment exactly, also run the suite
with an empty home and no `YAVIH_*` variables:

```bash
env -i PATH="$PATH" HOME="$(mktemp -d)" XDG_CONFIG_HOME="$(mktemp -d)" \
  .venv/bin/python -m pytest tests src/yavih/benchmarks -q -p no:cacheprovider -o addopts="--import-mode=importlib"
```

Tests that need staged benchmark media (real-media reader smoke tests) skip when no data
root is configured.

**Kinds of tests.**

| kind | where | what |
|---|---|---|
| contract tests | `tests/test_<module>*.py` | each module against its documented behaviour, on synthetic data |
| end to end | `tests/test_e2e_synthetic.py`, `test_readouts_integration.py` | extract → validate → probe → report on `yavih.testing` synthetic benchmarks; the oracle bridge clears every floor, the random bridge does not |
| folder tests | `src/yavih/benchmarks/<name>/tests/test_folder.py` | manifest hashes and curation sums, split assignment, docs pinned to the yaml, lint clean, metric names resolve; real-media smoke when data is present |
| golden tests | `tests/test_hashing.py`, `tests/data/avsync_metrics_golden.json`, `tests/test_freeze.py` | pinned hashes and payloads, metric values, the seeded train draw |
| boundary tests | `tests/test_import_boundary.py`, `tests/test_extensibility.py` | torch-free core, registry and entry points |
| docs tests | `tests/test_docs.py`, `tests/test_skill.py`, `tests/test_packaged_folders.py` | docs tables match code constants, every relative link resolves, the skill's commands parse, shipped docs cite nothing unopenable |
| independence guard | `tests/test_no_internal_refs.py` | a `git grep` over every tracked file: no internal project names, development-diary wording or site specifics (cluster names, mount points, home directories); the only allowed hits are listed in the test |

Tests are fast (seconds), use synthetic data, need no network and no GPU (GPU-only paths are skipped).
Shared fixtures: `synthetic_benchmark`, `write_synthetic_benchmark`, `yavih_home`, `runs_dir`
(`tests/conftest.py`) and `yavih.testing.make_synthetic_benchmark(tmp, readout, media="tiny")` for every
readout with decodable media.

## Conventions

### Errors, codes and exit codes

- Findings and refusals carry a stable code: `E_*` errors, `W_*` warnings, `N_*` notes. Validation codes
  and their levels are listed in `yavih.io.validate.CODES` and documented in [validation](validation.md); a
  test keeps the two in sync. Add a new code to the table and the docs together.
- An exception that the CLI reports carries `code` and `exit_code` attributes (`ExtractError`,
  `ProbeRefused`, `CliError`); otherwise `cli.exit_code_for` maps the exception class.
- Exit codes are part of the interface: `0` done / skipped / `not_applicable`, `1` usage or
  configuration, `2` contract violation to fix, `3` not ready (data, dependency or decoder missing;
  resubmit later), `4` deadline or preemption (nothing partial written; resubmit), `5` internal failure.
- Messages say what is wrong and what to do (the exact command or line to write where possible). Never
  turn an error into a warning to make a test pass; unknown keys and unconsumed arguments are errors.

### The `--json` envelope

Every command handler returns a `cli.Envelope`; with `--json` (or `YAVIH_JSON=1`) `main` prints exactly one
JSON object `{status, data, errors, warnings, paths}` on stdout, where `status` is one of `ok`, `error`,
`skipped`, `not_applicable` and each finding is `{code, message, where}`. Human output goes to stdout
without `--json`; progress goes to stderr. See [cli](cli.md).

### Style

- Python 3.10+, type hints, frozen dataclasses for records; `ruff` with the rules in `pyproject.toml`
  (line length 100 is advisory: `E501` is ignored).
- Docstrings and comments explain invariants and reasons, not what the next line does. Refer to concepts
  by name (`input_hash`, "the frozen shard layout"), not by document section numbers.
- Public names and on-disk keys are exact and stable; renaming one is a schema change.
- No new runtime dependency in the core. An optional dependency goes into an extra and is imported inside
  the function that needs it. A benchmark authors' package is never a dependency of any kind.

## Documentation

| what | where | maintained |
|---|---|---|
| philosophy and rationale | `docs/design.md` | by hand |
| normative reference | `docs/*.md` (see [the index](README.md)) | by hand; tables of codes, env vars and commands are checked against the code by `tests/test_docs.py` |
| per-benchmark docs | each folder's `README.md`, `setup.md` | by hand; pinned to `benchmark.yaml` by folder tests and `bench lint` |
| benchmark lists | `README.md` (table, controls), `docs/benchmark-folder.md` (packaged benchmarks), `benchmarks/README.md` (index, setup links), `references/data.md` (obtainability, `prepare_only` roots), `references/readouts.md` (benchmarks per readout) | generated between `yavih:bench-table` markers: `python -m yavih.doctables --write` (`--check` in the tests) |
| header images | `assets/header.png`, each folder's `assets/header.png` + `assets/header.src.sha256` | rendered: `python -m yavih.headers` (below); checked by the `headers` workflow |
| agent-facing docs | `src/yavih/skill/yavih/` | by hand; shipped, so links go to GitHub URLs |
| user-facing overview | `README.md` | by hand (do not regenerate it) |

When behaviour changes, update the reference page in the same commit. Release notes are generated by
GitHub from merged pull requests; there is no changelog file to edit.

### Header images

Each folder's README opens with `assets/header.png`, rendered from the folder's `benchmark.yaml` `header:` block
by `python -m yavih.headers <name>` (`--all` for every packaged folder; `--contact-sheet PATH` tiles them for a
review). `python -m yavih.headers --main` re-renders the repository's own `assets/header.png` from
`assets/header.html`. Rendering needs a headless Chrome or Chromium (`--chrome PATH` or `$CHROME`); the images are
repository files and are not shipped in the wheel.

Each folder render also writes `assets/header.src.sha256`, the sha256 of the HTML the image was rendered from;
commit it with the image. `bench lint` warns `W_HEADER_STALE` when a folder's image has no or another sha (a changed
`header:` block, chip or template), and the `headers` workflow (`.github/workflows/headers.yml`) fails a pull request
whose committed sha files are not current. Without Chrome, take the PNG from the workflow's `headers` artifact and
write the sha file with `python -m yavih.headers <name> --html-only` (the `assets/header.html` it also writes is not
committed).
