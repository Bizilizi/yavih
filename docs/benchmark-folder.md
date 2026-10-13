# Benchmark folder

A benchmark is a folder, and the folder is the only source of truth for its protocol: the rows, splits, contracts,
metrics, floors, controls, curation counts and changelog all live in it. This page is the normative reference for
what a folder holds, the `benchmark.yaml` schema, what `Benchmark.load` checks, the manifest and classes files, the
`Benchmark` API and how a name resolves to a folder, the README.md / setup.md contracts, and what `yavih bench lint`
checks.

Related pages: [adding-a-benchmark](adding-a-benchmark.md) (the how-to workflow), [bridge](bridge.md) (what a
`Sample` / `RawStream` hands over and the media facts), [hashing](hashing.md) (which keys enter `input_hash` and
`scoring_hash`), [readouts](readouts.md) (readout parameters and metrics), [validation](validation.md) (how
`output_contract` is checked), [configuration](configuration.md) (data roots).

## Folder layout

```
<name>/
  README.md             for people: header image, links, description, citation, the protocol in yavih
  setup.md              for the agent: obtain, dependencies, contracts, prepare & verify, gotchas
  benchmark.yaml        the machine-readable protocol
  manifest.csv.gz       the frozen rows (sha256 in benchmark.yaml)
  classes.csv           the label vocabulary, iff manifest.classes is set
  bridge_template.py    a Bridge skeleton to copy
  make_manifest.py      upstream metadata -> manifest.csv.gz + classes.csv, byte-identical re-freeze
  prepare/              numbered steps: 01_download.sh, 02_verify.sh, 03_materialize.py, ...
  probe.py              optional: folder metrics, targets, the authors' scorer
  reader.py             optional: replaces the generic reader
  tests/test_folder.py  the folder's own facts
  assets/header.png     the README header image (python -m yavih.headers) + header.src.sha256; never shipped
  notes/                optional maintainer notes; never shipped
```

| file | required | role |
|---|---|---|
| `README.md` | yes | Header image, links, description, citation, the protocol in yavih ([contract](#readmemd-contract)). |
| `setup.md` | yes | The agent's first read ([contract](#setupmd-contract)). |
| `benchmark.yaml` | yes | The protocol ([schema](#benchmarkyaml)). |
| `manifest.csv.gz` | yes | Frozen rows; `manifest.sha256` is over the decompressed bytes. |
| `classes.csv` | iff `manifest.classes` | `index,name[,mid]`. |
| `bridge_template.py` | yes | The benchmark's streams, a `describe()` with a literal `"model"` and a `reading` in `a`/`v`/`av`, the `input` block to fill, `used` reporting, `TODO` markers and a "Preprocessing source" docstring slot for the model's own code. |
| `make_manifest.py` | yes | Re-freezes `manifest.csv.gz` / `classes.csv` byte-identically (through `yavih.freeze`: `gzip_bytes`, `csv_bytes`, `write_if_changed`, `class_balanced_subsample`) and writes the `manifest.upstream.curation` counts it applied. |
| `prepare/` | yes, non-empty | Prepare-time steps with expected sha256s: download, verify, index builders, materialization, target derivation. |
| `probe.py` | no | [Hooks](#probepy-hooks) for the scoring side. |
| `reader.py` | no | [Folder reader](#readerpy) for what the yaml cannot express. |
| `tests/test_folder.py` | yes | Pins the folder's own facts (row counts, hashes, curation sums, reader smoke on synthetic media, docs). |
| `assets/header.png` | no | The README header image, rendered from the `header:` block, and `assets/header.src.sha256`, the sha256 of the HTML it was rendered from ([header image](#header-image)); excluded from the wheel. |
| `notes/` | no | Maintainer notes (freeze decisions, provenance); excluded from the wheel. |

A folder is data plus scripts, never a Python package: there is no `__init__.py` in it or in `tests/`. `probe.py`,
`reader.py`, `make_manifest.py` and `bridge_template.py` are loaded by path. pytest runs with
`--import-mode=importlib`, so every folder's `tests/test_folder.py` is collected under a unique name. Run one folder's
tests with `python -m pytest "$(yavih bench path <name>)/tests" -q -p no:cacheprovider`.

The wheel ships the folder, `prepare/` and `tests/`, minus `notes/`, `assets/`, bytecode and hidden files. No shipped text file
of a folder (`*.md`, `*.yaml`, `*.yml`, `*.py`, `*.sh`, `*.txt` in the folder, `prepare/` or `tests/`) may name a
`notes/` path (`E_NOTES_LINK`). `yavih bench files <name>` lists the shipped files and `yavih bench show <name>
<what>` prints one (`readme`, `setup`, `yaml`, `template`, `probe`, `classes`, or a path relative to the folder).

### prepare/ steps

A numbered step is `prepare/<NN>_<name>.sh` or `.py`. `yavih bench prepare <name>` lists them and `yavih bench
prepare <name> <step> [-- args]` runs one (by number, name or file name) with `bash` or the current Python, in the
current directory, with `FOLDER=<folder>` and `YAVIH_BENCH=<name>` set. Helper scripts without a number (for example
`prepare/derive_targets.py`) may sit next to the steps. A step resolves data roots through `yavih config get
<name>.<root>`, never a hard-coded site path.

### probe.py hooks

All hooks are optional. Module scope imports only the standard library, numpy and yavih core (`yavih bench lint`
imports it with torch and the folder's `external` packages blocked). It reuses the shared heads, trainer and
`ProbeConfig`; it never defines a head, loss or training loop. Its normalized source (comments and docstrings
excluded) enters `scoring_hash`, so editing it never invalidates extracted features.

```python
METRICS: dict[str, MetricSpec]      # folder metrics; may replace a registry metric only with the same readouts/shape/kind
PROTOCOL_KEYS: frozenset[str]       # protocol keys this file consumes (must be present in `protocol`)
def targets(bench, split: str, rows: np.ndarray) -> np.ndarray            # replaces the readout's target construction
def score(bench, split: str, rows: np.ndarray, outputs, targets) -> dict  # the external scorer for its METRICS
def join(bench, results: dict) -> dict                                    # cross-reading join for `report --join`
```

Every call into an original-author package goes through `yavih.external.require(name, pip, version_pin)` with the
matching `benchmark.yaml` `external` entry. A `probe.py` that re-implements the authors' code instead (because it is
not importable) says so in README "## In yavih" and setup.md "## Dependencies".

### reader.py

Without a `reader.py`, `Benchmark.reader` iterates through the generic `yavih.sample.iter_samples`, which builds each
`Sample` from the yaml (see [bridge](bridge.md)). A folder may ship a `reader.py` exporting

```python
def iter_samples(bench, split, rows, batch_size, canaries, canary_seed, *, shard=None, roots=None) -> Iterator[Sample]
```

It replaces the generic reader, imports only yavih core, must accept `shard=(i, n)` (per-shard canaries; `bench
lint` check `reader`), and its normalized source enters `input_hash`. No packaged folder ships one.

## benchmark.yaml

`yavih.benchmark.load_yaml(path)` parses the file into a frozen `BenchmarkSpec`. Unknown keys at any level are an
error (`SchemaError`). Keys are required unless the comment says optional. Which blocks are hashed, and on which side,
is in [hashing](hashing.md); in short, `aliases`, `title`, `header`, `changelog`, `source`, `obtain`, `budget`,
`data` and `manifest.upstream` are never hashed.

```yaml
schema_version: 2                  # int, == yavih.versions.BENCHMARK_SCHEMA_VERSION
name: k400                         # [a-z0-9_]+, == the folder's directory name
aliases: []                        # list of [a-z0-9_]+; Benchmark.load(alias) resolves here; not hashed
title: Kinetics-400 action recognition   # one line; bench list / bench info and the benchmarks index
header:                            # optional for the loader, required by bench lint; not hashed (README header image)
  name: Kinetics-400                # the display name
  title: The Kinetics Human Action Video Dataset   # the paper title
  venue: "Kay et al. · arXiv 2017"  # "authors · venue year"
  abstract: "..."                   # two or three neutral sentences on the data and the task, at most 60 words
readout: clf                       # a registered readout: clf | seq | grid | box (yavih.readouts.registry)
protocol_version: 1                # int >= 1, == the newest changelog entry's version
changelog:                         # non-empty, newest last, strictly increasing versions; the version record
  - version: 1
    date: "2026-10-12"
    reason: "..."
    invalidates: true              # true when a number under the previous version is void under this one
    protocol_hash: "<64 hex>"      # optional per entry; required on the newest by bench lint (see "Changelog hash keys")
    input_hash: "<64 hex>"         # optional per entry; required on the newest by bench lint
    supersedes: []                 # optional: earlier protocol hashes of this version
    supersedes_input: []           # optional: earlier input hashes whose feature sets stay valid

source:                            # provenance; not hashed
  kind: torrent                    # public_url | torrent | manual_portal | private_mirror | derived
  urls: ["https://..."]            # list[str]; may be empty
  licence: "..."
  revision: null                   # optional str, non-empty: the pinned upstream release; setup.md "## Obtain the data" quotes it
  bytes: 157300000000              # optional int: the full upstream download
  sha256: {}                       # optional: upstream file name -> sha256
  notes: "..."                     # optional: one paragraph

obtain:                            # optional for the loader, required by bench lint; not hashed
  tier: auto                       # auto | manual-download | needs-licence | needs-data-owner | site-only
  how: "one line: what a user does to get the data"

budget:                            # used by bench list/info and plan; not hashed
  clips: {train: 20000, val: 13492}      # split value -> rows; bench lint: equal to the frozen rows of every read split
  media_gb: 20.2                         # media the frozen manifest touches once staged
  extract_min_per_1k_clips_ref: 1.2      # reference GPU minutes per 1000 clips
  probe_minutes_ref: {clf: 20}           # readout -> minutes; must hold an entry for `readout`

manifest:
  file: manifest.csv.gz            # gzip of a UTF-8 CSV
  sha256: "<64 hex>"               # over the DECOMPRESSED bytes
  classes: classes.csv             # str | null; required for single/multi/multi_sets/segments, null for box/none
  classes_sha256: "<64 hex>"       # set iff classes is set
  key: clip_id                     # the row key column (unique)
  label:
    column: label                  # str | null; null iff kind none; for multi_sets the TRAIN label column (or use train_set)
    kind: single                   # single | multi | multi_sets | box | segments | none; must be one the readout scores
    sep: ";"                       # optional, default ";": separator inside multi-label cells
    class_key: name                # optional, default name: name | mid | index (which classes.csv column cells match)
    sets: null                     # multi_sets only: {set_name: [columns]} unioned per row; set names [a-z0-9_]+
    train_set: null                # multi_sets only: the set holding TRAIN targets; null = `column` holds them
    allow_unknown: false           # optional: drop label tokens not in classes.csv (counted) instead of failing
    empty_ok: false                # optional: an empty label cell means "no labels" instead of failing
  columns:                         # every other manifest column, typed; key, split and label columns are implicit
    rel_media: {type: str, role: media, stream: video}   # type str|int|float; role media|extra|label_aux
    offset_bin: {type: int, role: extra}                 # `stream` (a name or a list) only with role media
  upstream:                        # optional for the loader; bench lint requires `curation`; not hashed
    source: "..."                  # the file the rows were frozen from
    sha256: "..."
    rows: 254110
    note: "..."                    # optional
    upstream_files: {}             # optional: name -> sha256 of every file make_manifest.py read
    curation:
      total_upstream: 254110
      per_split: {train: {listed: 20000, staged: 20000, missing: 0}, val: {listed: 13492, staged: 13492, missing: 0}}
      excluded: {}                 # reason -> count
      short_clips: 0               # optional: media shorter than protocol.clip_seconds at freeze time (a fact)
      leakage_audit: null          # optional: {against: "<what>", excluded: n}
      # invariant: sum(per_split.*.listed) + sum(excluded.*) == total_upstream

splits:                            # every manifest split value in exactly one role
  train: [train]                   # rows the probe trains on (val is carved from these)
  eval: [val]                      # scored, never trained on
  floor: []                        # optional: scored with the same probe as a validity floor
  folds: null                      # optional: cross-validation values (>= 2), each held out once; train and eval then empty
  unused: []                       # optional: present in the manifest, never read
  min_eval_coverage: 0.95          # optional, in (0, 1]: the probe refuses below it unless --allow-low-coverage

streams:                           # name ([a-z][a-z0-9_]*) -> stream; names are Sample.streams keys
  video:
    kind: video                    # video | audio | image
    media_column: rel_media        # str or {split value | role: str}; a manifest column with {role: media, stream: <this>}
    root: media_root               # str or per split; a key of data.roots
    layout: dir                    # optional, default dir: dir | tar_index (per split allowed)
    layout_params: {}              # optional; per layout, may be per split
    required: true                 # optional: false = an empty or missing media cell does not make the row missing
  # Two streams may read the same media cell (one mp4 handed over as `video` and `audio`).

targets: null                      # present iff the readout takes labels from targets (grid); then label.kind is none
  # train: {kind: dense_depth, source: derived_file, root: derived_root, file: train_depth_224.npy, sha256: "<64 hex>",
  #         index_column: null}    # one entry per read split; index_column maps manifest rows to memmap rows

data:                              # not hashed; resolved by yavih.config (see configuration.md)
  roots:                           # name ([a-z][a-z0-9_]*) -> root
    media_root:
      default: "${data.root}/kinetics400"   # ${data.root}: the site root; ${datasets.<name>}: a shared dataset
      description: "{train,val}/<class>/*.mp4"   # optional
      prepare_only: false          # optional: read only by prepare/ and make_manifest.py, never by the reader
  tar_index: null                  # path of index.json under a root; required iff a stream uses a tar_index layout
  derived: {}                      # optional: file -> sha256 of what prepare/ produces

input_contract:                    # what the reader hands over; setup.md "## Input contract" ends with a copy
  handed_over:                     # exactly the stream names, one non-empty sentence each
    video: "the Kinetics-400 mp4 exactly as stored; nothing decoded, cut or resized"
  info:                            # exactly the stream names: RawStream.info keys guaranteed non-null on every row
    video: [duration_s, fps, width, height, container, bytes]   # from the media-fact vocabulary of the stream's kind
  facts: []                        # top-level protocol keys copied into every Sample.meta["facts"] (label facts, never a label)
  materialize: null                # null, or the folder's pinned prepare-time transform the label requires:
  # materialize:
  #   kind: av_offset_mux          # [a-z0-9_]+: the folder recipe's name; its prepare/ script interprets the other keys
  #   offset_column: offset_seconds        # a key ending in _column names a declared manifest column
  #   source_columns: {video: rel_video, audio: rel_audio}   # ending in _columns: a mapping or list of declared columns
  #   pivot: 1.0                   # any other recipe key: JSON values
  #   tool: {name: ffmpeg, version_pin: "...", video_args: [...], audio_args: [...],
  #          pins: {libx264: {value: "...", probe: ["{tool}", ...], match: "<regex>"}}}   # every argument pinned
  #   no output sha256: re-encoded bytes differ between hosts, so the recipe is the pin (`outputs` is refused);
  #   a per-row record of a site's own files stays at that site, never shipped

output_contract:                   # checked by validate (validation.md); setup.md "## Output contract" ends with a copy
  shape: fixed                     # pinned: one (N, D) for every row of a feature set
  coordinates: media               # pinned: pos (t, y, x) in media coordinates
  target_frame: null               # "full_frame" iff the readout is grid or box, else null
  requires:
    pos: []                        # subset of [t, y, x]
    streams: []                    # declared stream names that must appear as tags among tokens with pos
    grid: null                     # grid/box: {h, w, min_cell_coverage in (0, 1]}; required by those readouts
    t_bins: null                   # seq: scored seconds that must hold >= 1 token; required by seq
    annotated_frame: null          # optional: {fact, stream, tol_s}; fact in input_contract.facts and a number in
                                   #   protocol; stream a video stream; tol_s > 0

external: []                       # original-author packages probe.py calls; never yavih dependencies
  # - name: vggsounder             # import name, unique
  #   pip: "pip install --no-deps vggsounder==1.0"   # a full `pip install ...` command; setup.md repeats it verbatim
  #   version_pin: "1.0"           # importlib.metadata.version(name) must equal it at call time
  #   used_for: "..."

protocol:                          # label facts only, JSON values (finite numbers); constrains nothing the bridge does
  clip_seconds: 10.0               # number > 0 | null: nominal media duration (a fact; never used to cut or pad)
  train_draw:                      # null when uncapped
    cap: 20000
    seed: 0
    balance: class                 # class | first_label | session | null
    drawn_from: "..."
  views: none                      # none | rows: whether several rows derive from one source clip
  # Every other key is a per-benchmark label fact and must be consumed: named by a resolved metric, by the
  # readout, by probe.py PROTOCOL_KEYS, or by input_contract.facts. target_region, when present, is
  # [y0, x0, y1, x1] in full-frame fractions with 0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1.

metrics:                           # names: yavih.metrics registry overlaid by probe.py METRICS (readouts.md)
  headline: acc                    # in `list`; the metric the report sorts on
  list: [acc, n_eval]              # non-empty; kinds metric | disclosure; result.json[metrics] holds exactly these
  floors: [acc_floor]              # kind floor
  controls: []                     # kind control
  chance: {acc: 0.0025}            # keys in `list`
  ci: {acc: binomial}              # keys in `list`: binomial | bootstrap_clip | pooled_binomial | null, admissible for the metric
  bootstrap: {samples: 2000, seed: 0}    # optional; required when a ci is bootstrap_clip
  aggregation: null                # required iff splits.folds: {over: folds, report: [mean, pstd, per_fold], ci?: <kind>}

readout_params: {}                 # keys exactly the readout's: clf {}; seq {T}; grid {h, w, out, visual_stream};
                                   #   box {grid, visual_stream, conditioning_stream, shuffled_control, shuffle_seed}
```

### What Benchmark.load validates

`load_yaml` checks everything that can be checked from the yaml and the folder's `probe.py`; `Benchmark.from_dir`
then checks the folder files. Neither touches media or derived files; `yavih bench verify` does.

From the yaml (`SchemaError`, exit 1):

- the schema: unknown and missing keys at every level, types, enums, 64-hex digests, `schema_version`;
- `name` and `aliases` match `[a-z0-9_]+`, no alias equals the name; `readout` is registered;
- `changelog`: strictly increasing versions, newest last, newest `version == protocol_version`;
- `source.revision` non-empty when present; `obtain` (when present) has a known tier and a non-empty `how`;
- `budget.clips` keys are declared split values, `budget.probe_minutes_ref` has the benchmark's readout;
- `manifest`: `classes` / `classes_sha256` both set or both null and present exactly for the label kinds with
  classes; the label rules of the schema above; media columns name declared streams; the key, `split` and label
  columns are not redeclared; `segments` needs `event_start` / `event_end` as `{type: int, role: label_aux}`;
- `splits`: every value assigned once; `folds` (at least two) excludes `train` / `eval`, otherwise both are non-empty;
- `streams`: for every read split, `media_column` resolves to a `manifest.columns` entry with `role: media` naming
  this stream, `root` is in `data.roots`, the layout is known and its `layout_params` valid;
  `data.tar_index` set iff a `tar_index` layout is used;
- `targets` present iff the readout takes labels from targets (then `label.kind: none`), one per read split;
- `manifest.label.kind` is one the readout scores;
- `protocol`: JSON values only; the fixed keys present; `target_region` valid; every per-benchmark key consumed
  (an unknown key is refused) and every key a metric or `probe.py` declares present;
- `input_contract`: `handed_over` and `info` keyed by exactly the stream names, `info` keys in the media-fact
  vocabulary of the stream's kind ([bridge](bridge.md)), `facts` are top-level protocol keys, the `materialize`
  envelope;
- `output_contract`: `shape: fixed`, `coordinates: media`, `target_frame: full_frame` iff the readout is `grid` or
  `box`, the `requires` rules the readout declares, `annotated_frame` cross-references;
- `external`: four non-empty strings per entry, unique names, a non-empty list requires `probe.py`, and every metric
  computed by an external package names a declared entry;
- `metrics`: every name resolves (registry or `probe.py` `METRICS`) with the right kind and is allowed for the
  readout; no name in two lists; `headline`, `chance` and `ci` keys in `list`; ci kinds admissible; `bootstrap` and
  `aggregation` rules;
- `readout_params`: exactly the readout's keys, `*_stream` values are declared streams, and the readout's own
  cross-checks;
- `probe.py` imports, `METRICS` is a `dict[str, MetricSpec]` and `PROTOCOL_KEYS` a `frozenset[str]`.

From the folder (`UnknownBenchmark` / `SchemaError`, exit 1; `ManifestError`, exit 2):

- the folder holds `benchmark.yaml` and `name` equals the directory name;
- a `$YAVIH_HOME/benchmarks` copy of a packaged folder is current (below);
- `manifest.csv.gz` exists, decompresses, and its decompressed sha256 equals `manifest.sha256`; `classes.csv`
  exists, matches `classes_sha256` and follows its rules;
- the manifest header is exactly key, `split`, label columns and declared columns (no duplicate, nothing undeclared);
  every row has the header's width; keys are unique and match the charset; every split value has a role; every read
  split has rows; required media cells are non-empty, match the charset, are relative and contain no `..`; typed
  cells parse; every label cell parses against `classes.csv` (unknown tokens only with `allow_unknown`, empty cells
  only with `empty_ok`); `single` / `segments` cells hold exactly one class; boxes are `x1|y1|x2|y2`; segments need
  `0 <= event_start < event_end`;
- `protocol.background_index`, when present, indexes `classes.csv`.

Load then computes `manifest_hash`, `input_hash`, `scoring_hash` and `protocol_hash` from the yaml, the manifest,
`reader.py`, `probe.py` and `yavih.versions` only (identical in the model env and the probe env), and registers the
folder's changelog hashes with `yavih.hash_history`.

### Changelog hash keys

Each `changelog` entry may carry four optional hash keys, each a 64-hex digest. The changelog is outside every hashed
payload, so a folder recording its own digests is not circular.

| key | meaning |
|---|---|
| `protocol_hash` | The `protocol_hash` this version computes. Required on the newest entry. |
| `input_hash` | The `input_hash` this version computes. Required on the newest entry. |
| `supersedes` | Earlier protocol hashes that denote this same version (a hash-schema bump, a refactor). Their results are this version's: `report.migrate` maps them to this entry's `protocol_hash`. |
| `supersedes_input` | Earlier input hashes whose feature sets stay valid (an input-side edit that hands the bridge exactly the same thing). |

`yavih bench lint` (check `hashes`, code `E_CHANGELOG_HASH`) requires the newest entry to record the computed
`protocol_hash` and `input_hash`, and forbids an entry listing its own digest under `supersedes` /
`supersedes_input`; its message names the exact lines to write and whether to bump `protocol_version` or append to
`supersedes`. `yavih bench new` writes both hashes for the scaffold.

`yavih.hash_history` merges the changelogs of every resolvable folder (packaged, `$YAVIH_HOME/benchmarks`,
`$YAVIH_BENCH_PATH`, entry-point packs) and of every folder a `Benchmark` was loaded from in the process, and follows
`supersedes` / `supersedes_input` transitively. There is no central table. How the hashes are computed and how
migration works: [hashing](hashing.md).

## Manifest and classes

### manifest.csv

`manifest.csv.gz` is a gzip of a UTF-8, RFC 4180 CSV with a header. `make_manifest.py` writes it in a fixed column
and row order through `yavih.freeze` (no file name, `mtime=0`, level 9, `\n` line ends), so a re-freeze is
byte-identical.

- The key column (`manifest.key`, `clip_id` in every packaged folder) is unique, charset `[A-Za-z0-9_.#:-]`, never `/`.
- `split` (str) is required; every value is assigned to one role in `splits`.
- Media columns (`role: media`) use charset `[A-Za-z0-9_.#:/-]` and hold relative POSIX paths or tar-index keys.
- Multi-label cells join tokens with `manifest.label.sep` (CSV-quoted as needed).
- Box cells are `x1|y1|x2|y2` in fractions of the full original frame.
- Segment rows carry integer `event_start`, `event_end` seconds.

`Benchmark.manifest(split=None)` returns a `Manifest` in file order filtered to the split: `row` (0-based within the
split), `clip_id`, `splits`, `label_ids` (`int | None` for single/segments, `list[int]` for multi and multi_sets
train targets, `None` for box/none), `label_sets` (multi_sets), `media` (`{stream: cell}`), `media_key`, `extra`
(every declared column, typed), `boxes` (`float32 [n, 4]`), `segments` (`int64 [n, 2]`) and `file_row`. `media_key`
is equal for rows with identical media cells; it feeds only `E_DUPLICATE_CLIPS` and the member record. The bridge
never sees labels, the manifest or the media key.

Labels a CSV cell cannot hold (dense depth) come from `Benchmark.targets(split) -> TargetReader` (numpy only): it
memmaps `targets[split].file` under its root, verifies the sha256 once (cached in the data status file), and exposes
`rows(idx)`, `shape`, `dtype` in manifest row order (through `index_column` when set). It raises `NoTargets` for
readouts whose labels are manifest cells and `DataNotReady` when the file is missing.

### classes.csv

`index,name[,mid]` with a header; `index` contiguous `0..n-1`, `name` non-empty and unique, `mid` non-empty and
unique when present, at least one class. The `index` column is the only logit vocabulary.

### manifest_hash

```
manifest_hash = sha256(b"manifest:" + sha256(manifest.csv).hex + b"\nclasses:" + (sha256(classes.csv).hex | "none"))
```

over the decompressed manifest bytes. It enters `input_hash` together with `manifest.key` and `manifest.columns`;
`manifest.label` enters `scoring_hash` ([hashing](hashing.md)).

## Benchmark API

```python
class Benchmark:
    @classmethod
    def load(cls, ref: str, *, data: dict | None = None, fetch: bool = False) -> Benchmark
    @classmethod
    def from_dir(cls, path, *, data: dict | None = None) -> Benchmark
    name: str; spec: BenchmarkSpec; folder: Path
    manifest_hash: str; input_hash: str; scoring_hash: str; protocol_hash: str; hashes: ProtocolHashes
    def manifest(self, split: str | None = None) -> Manifest
    def n_rows(self, split: str) -> int
    def classes(self) -> list[dict]; num_classes: int
    def roles(self) -> dict[str, list[str]]      # {"train": [...], "eval": [...], "floor": [...], "folds": [...]}
    def reader(self, split, *, shard=None, batch_size=1, rows=None, canaries=0, canary_seed=0, align=None) -> Iterator[Sample]
    def sample(self, split: str, row: int) -> Sample
    def targets(self, split: str) -> TargetReader
    def resolved_roots(self) -> dict[str, str]
    def data_status(self, *, refresh: bool = False) -> DataStatus
    def media_info(self) -> MediaInfoTable | None
    def view(self) -> BenchmarkView
    def probe_module(self)                       # the folder's probe.py as a module, or None
```

- `data` overrides data roots for this instance ([configuration](configuration.md)); `fetch=True` first syncs the
  folder into `$YAVIH_HOME/benchmarks/<name>` (`yavih bench fetch`).
- `reader` yields the `Sample`s of a row range or of shard `(i, n)`, then the shard's canaries; it refuses `unused`
  splits. It delegates to `reader.py` when present, else to `yavih.sample.iter_samples`. Sharding and canaries:
  [execution](execution.md) and [bridge](bridge.md).
- `data_status` returns presence per split, `media_fingerprint`, the resolved roots, the last `bench verify` stamp
  and the census sha256, cached in `$YAVIH_HOME/data_status/<name>-<roots digest>.json` and reused while
  `manifest_hash` and the roots are unchanged.
- `media_info` is the site census `$YAVIH_HOME/media_info/<name>-<manifest_hash12>.jsonl.gz`, or `None` when absent.
- `view` is what a bridge receives: name, aliases, readout, `protocol_version`, `protocol_hash`, `input_hash`,
  `{stream: kind}`, both contracts and the facts; no manifest, labels, splits or roots.

### Resolution order

`Benchmark.load(ref)` (and every `yavih bench` command) resolves `ref` to a folder in this order; the first match wins:

1. an existing directory path holding `benchmark.yaml` (a `ref` containing `/` is only tried as a path);
2. `$YAVIH_HOME/benchmarks/<ref>`;
3. the packaged folder `yavih/benchmarks/<ref>` (found through `importlib.resources`);
4. the directories of `$YAVIH_BENCH_PATH` (colon-separated, first wins; each is a benchmark folder or holds
   benchmark folders);
5. pip-installed benchmark packs: entry-point group `yavih.benchmarks`, value `"pkg.module"`, whose package directory
   is a benchmark folder or holds benchmark folders (read as data, never imported);
6. an alias declared by any of the folders above.

A folder from `$YAVIH_BENCH_PATH` or a pack never shadows a packaged name. A `$YAVIH_HOME/benchmarks` copy of a
packaged folder carries a `.yavih_fetch.json` stamp; a copy whose source files changed since it was fetched is
refused (`E_HOME_BENCH_STALE`, a warning under `YAVIH_LENIENT=1`) and an unstamped copy warns
(`W_HOME_BENCH_UNSTAMPED`); `yavih bench fetch <name> --refresh` re-syncs it.

## README.md contract

For people, and rendered on GitHub. No H1 (the header image carries the name); in this order:

1. the header image, alone on the first line: `![<header.name>](assets/header.png)`;
2. the links line: `[Paper](…) · [Homepage](…) · [Dataset](…) · [setup.md](setup.md)`, whichever exist;
3. exactly these `##` sections:

| heading | content |
|---|---|
| `## Description` | The task, the data and the official splits, as the benchmark's authors define them. |
| `## Citation` | BibTeX of the paper(s) in a fenced `bibtex` block, and of the annotation source where it differs. |
| `## In yavih` | The protocol: readout, splits and roles with counts, the headline metric (named as in `metrics.headline`) with chance, floors and controls, the frozen train draw, `protocol_version` and its number; what the bridge gets and returns in two or three lines with a pointer to setup.md. Then variants and aliases (and what is not this benchmark), reference numbers only when they are published (with the citation), and the notes for re-freezing the manifest (`make_manifest.py`, its public inputs, the folder tests). |

There is no changelog, history or checklist section: the `benchmark.yaml` `changelog` is the version record. A
`protocol_hash` printed in README.md must be a prefix of the current one.

### Header image

`assets/header.png` is a 1600 x 500 (CSS px) banner at device scale 2, rendered by `python -m yavih.headers <name>`
(`--all` for every packaged folder) from the `header:` block and chips derived from the yaml: the stream kinds, the
readout (multi-label when the labels are), the clip and class counts, the folds, the readings of the bridge template
or the train/eval sizes, and the headline metric. The template is `assets/benchmark-header/` in the source
repository; rendering needs a headless Chrome or Chromium (`--chrome PATH` or `$CHROME`). The image is a maintainer
artefact for the repository's README pages: the wheel does not ship it and `bench lint` does not require it.

Every render also writes `assets/header.src.sha256`: the sha256 of the filled HTML the image was rendered from. The
image and the sha file are committed together. `bench lint` warns `W_HEADER_STALE` when an image exists and its sha
file is missing or differs from the current HTML (an edited `header:` block, a chip that changed with the yaml or the
bridge template's readings, an edited template). Pixels differ between machines, the HTML does not, so the HTML is what
is compared. `--html-only` writes `assets/header.html` (not committed) and the sha file without rendering. On a pull
request that touches a folder, the `headers` workflow renders every header and a contact sheet as the `headers`
artifact, and fails when a committed sha file is not current.

## setup.md contract

The agent's first read, and only about setting the benchmark up. H1 `# <name> — setup` (em dash), then exactly these
`##` sections, in this order:

| heading | content |
|---|---|
| `## Obtain the data` | Concrete commands with sizes and expected sha256 per file; disk budget; what is not downloadable and where to request it. Numbered `prepare/` steps run as `yavih bench prepare <name> <step>`. Contains `source.revision` verbatim when set. |
| `## Dependencies` | Every `external[].pip` line verbatim, with the assert-import it must satisfy (`yavih doctor --bench <name>`); prepare-time tools with versions; the word "None" when `external` is empty. |
| `## Input contract` | Per stream: what is handed over, the guaranteed info keys, the facts, any materialization; what the bridge decides itself and the media's pitfalls. Ends with a fenced `yaml` block equal to `input_contract`. |
| `## Output contract` | `[N, D]` fixed; pos requirements in media coordinates and the eligibility numbers; the target coordinate system; what validate checks. Ends with a fenced `yaml` block equal to `output_contract`. |
| `## Prepare & verify` | The prepare and verify commands and what success looks like (the expected `yavih bench verify <name>` result). |
| `## Gotchas` | Only what affects the setup or writing the bridge: mistakes that pass validation on this benchmark. |

The yaml copies may keep the top-level key (`input_contract: ...`) or not; they are compared parsed. Neither
README.md nor setup.md has a `Changelog`, `History`, `Checklist` or `Maintainers` heading. Shipped documents
(setup.md, README.md, `bridge_template.py`) cite nothing a pip-installed user cannot open: no repository
documentation section, `notes/` path or source-tree path; a `https://github.com/Bizilizi/yavih` URL is the
exception. Commands use `yavih bench prepare` / `yavih bench show`; `FOLDER=$(yavih bench path <name>)` remains for
maintainer commands.

## The benchmarks index

`src/yavih/benchmarks/README.md` lists the packaged folders. Its table sits between
`<!-- yavih:bench-table index begin ... -->` and `<!-- yavih:bench-table index end -->` markers and is generated from
each folder's `benchmark.yaml` by `python -m yavih.doctables --write`. The same command regenerates every other list
of the packaged benchmarks, each between `<!-- yavih:bench-table <kind> begin ... -->` / `end` markers: the repo
README's table (`readme`) and controls bullet (`controls`), the index's `setup.md` links (`setup_links`), the
[packaged benchmarks](#packaged-benchmarks) table below (`folder`), and in the skill `references/data.md`'s
obtainability table (`data`) and `prepare_only` roots (`prepare_roots`) and `references/readouts.md`'s benchmarks per
readout (`readout_groups`). `--check` exits 1 when a block is stale:

```
| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [esc50](esc50/README.md) | ESC-50 environmental sound classification (...) | clf | audio | public (obtain: auto) |
```

One row per packaged folder, sorted by name; the first cell links `<name>/README.md`; the description names every
alias in backticks; `Readout` is the folder's `readout`; `Media` names what is handed over; `Source` is
`source.kind` in words (public, torrent, manual portal, private mirror, derived). Below the table: the folder layout
in one paragraph and a link to each folder's `setup.md`.

## Tests that pin the documentation

- `tests/test_packaged_folders.py` runs every `bench lint` check on every packaged folder (one test per folder and
  check, so a new folder is covered without editing the test), and checks that aliases are unique and resolve with
  the same `protocol_hash`, that the index has one correct row per packaged folder and its links resolve, that the
  generated tables are current, and that the lint catches what it claims on a mutated copy of `esc50`.
- Each folder's `tests/test_folder.py` pins that folder's facts: the yaml parses and loads, the sha256s match and the
  gzip header is reproducible, splits are assigned once, classes are contiguous, the curation sums hold, a reader
  smoke on synthetic media, and the README / setup.md sections (helpers in `yavih.testing.folder`, including
  `assert_lint_clean`).
- `tests/test_extensibility.py` checks that a `yavih bench new` scaffold of every readout loads and passes the lint
  with only `W_TODO` warnings, that its generated `tests/test_folder.py` passes (the tests that need the benchmark's
  own data skip with a `TODO:` reason), and covers out-of-tree folders (`$YAVIH_BENCH_PATH`, entry-point packs).

## yavih bench lint

`yavih bench lint <name|path>... [--all] [--skip CHECK] [--json]` runs the folder contract (`yavih.lint`) on packaged
or out-of-tree folders. Errors exit 2 (a contract violation); warnings do not fail. Each check has an id
(`yavih.lint.CHECKS`, in run order) and can be skipped; a check that crashes is itself an error (`E_LINT_CRASH`).

| check | codes | what it checks |
|---|---|---|
| `yaml` | `E_YAML`, `E_SCHEMA`, `E_NAME`, `E_LAYOUT` | `benchmark.yaml` exists and parses; `Benchmark.load` succeeds; `name` equals the folder name. |
| `hashes` | `E_CHANGELOG_HASH` | The newest changelog entry records the computed `protocol_hash` and `input_hash` and does not supersede its own digests. |
| `layout` | `E_LAYOUT` | `README.md`, `setup.md`, `benchmark.yaml`, `manifest.csv.gz`, `bridge_template.py`, `make_manifest.py`, `tests/test_folder.py` exist; `prepare/` is non-empty; `classes.csv` exists iff `manifest.classes`; no `__init__.py` in the folder or `tests/`. |
| `reader` | `E_READER` | An optional `reader.py` loads and its `iter_samples` accepts `shard=`. |
| `budget` | `E_MANIFEST`, `E_BUDGET`, `E_CURATION` | The manifest reads; `budget.clips` equals the frozen rows per read split; `manifest.upstream.curation` exists, its `per_split.*.listed` equal the frozen rows, and listed plus excluded equals `total_upstream`. |
| `header` | `E_HEADER`, `W_HEADER_STALE` (warning) | `benchmark.yaml` has a `header: {name, title, venue, abstract}` block (its content is checked by the loader: non-empty strings, an abstract of at most 60 words). When `assets/header.png` exists and the header template is at hand (a source checkout; skipped for an installed wheel), `assets/header.src.sha256` equals the sha256 of the current header HTML; otherwise the image predates its yaml, chips or template (`W_HEADER_STALE`). |
| `readme` | `E_README`, `E_README_HASH` | The [README.md contract](#readmemd-contract). |
| `setup` | `E_SETUP` | The [setup.md contract](#setupmd-contract). |
| `external` | `E_EXTERNAL` | Every `external[].pip` starts with `pip install `. |
| `obtain` | `E_OBTAIN` | `obtain` is exactly `{tier, how}`, a known tier and a non-empty `how`. |
| `template` | `E_TEMPLATE` | `bridge_template.py` has a literal `describe()["model"]` and a `reading` within `a`/`v`/`av`. |
| `citations` | `E_NOTES_LINK`, `E_CITATION` | No shipped text file names a `notes/` path; setup.md, README.md and `bridge_template.py` cite nothing a pip-installed user cannot open. |
| `imports` | `E_IMPORT` | `reader.py` / `probe.py` import in a fresh interpreter with torch and the `external` packages blocked. |
| `todo` | `W_TODO` (warning) | Unfilled `TODO` markers in `benchmark.yaml`, README.md, setup.md, `reader.py`, `probe.py`, `make_manifest.py`, `classes.csv`, and `TODO:` markers in `tests/*.py` and `prepare/*` (a folder test may name the bridge template's own `TODO` slots without the colon). |

A path without `benchmark.yaml` yields `E_LAYOUT`; an unknown name or check is a usage error (exit 1).

## Packaged benchmarks

The full index with descriptions, media and sources is [src/yavih/benchmarks/README.md](../src/yavih/benchmarks/README.md);
`yavih bench list` shows the same set with sizes and site status.

<!-- yavih:bench-table folder begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
| benchmark | readout | label kind | streams | headline | probe.py |
|---|---|---|---|---|---|
| audioset20k | clf | multi | video, audio | `map` |  |
| ave | seq | segments | video, audio | `seg_acc` |  |
| avsync | clf | single | video, audio (materialized) | `acc` | yes |
| esc50 | clf (5 folds) | single | audio | `acc_mean` |  |
| fsd50k | clf | multi | audio | `map` |  |
| in1k (aliases `imagenet1k`, `imagenet`) | clf | single | image | `acc` |  |
| k400 | clf | single | video | `acc` |  |
| nyuv2_depth | grid | none (targets) | image | `d1` | yes, `external` |
| ssv2 | clf | single | video | `acc` |  |
| vggsounder | clf | multi_sets | video, audio | `auc_pr_all` | yes, `external` |
| vggss_box | box | box | video, audio | `iou_mean` |  |
<!-- yavih:bench-table folder end -->
