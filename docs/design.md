# Design

Why yavih is built the way it is. This page states the philosophy and the reasoning behind it; the
[reference pages](README.md#reference) say exactly what the code does.

## What yavih is

yavih is a frozen-feature benchmark harness for video, audio and audio-visual encoders. It owns the
*protocol* side of an evaluation: which clips, which stored files, which labels, which readout, which
metric, which floor, which hash. It does not own the model side, not even which seconds or which pixels
the model looks at.

A model is connected by a **bridge**: a small feature extractor, written per (model, benchmark), that runs
in the model's own Python environment and writes features in yavih's format. yavih then validates those
features, trains the same standardized probe every model gets, scores it next to a label-only floor, and
reports only numbers that are comparable.

```
prepare ──► extract ──► validate ──► probe ──► report
 agent       bridge       yavih        yavih     yavih
 setup.md    model env    features vs  fixed     tables, floors,
                          manifest     readout   eval_hash
```

| stage | who | output |
|---|---|---|
| prepare | the agent, following the folder's `setup.md`; `yavih bench verify --census` | media on disk, verified manifest, coverage, the site's media-info table |
| extract | `yavih extract` in the model env: reader (stored files + facts) → bridge → `FeatureWriter` | `features/<split>/shard-*.tar` + one sidecar per shard |
| validate | `yavih features validate` (also the probe's prologue) | pass/fail findings, `_meta.json`, coverage, the readout's eligibility |
| probe | `yavih probe run` | `result.json` per seed: metrics, floors, controls, intervals, hashes |
| report | `yavih report build` | lm-evaluation-harness style tables, comparability-checked |

## What yavih is not

- **Not a model zoo.** The package contains no model code. Everything model-specific lives in a bridge
  the user (usually a coding agent) writes outside the package.
- **Not a preprocessing library.** yavih never decodes, resamples, crops or normalizes media for a model.
  Optional decode helpers exist for convenience and carry no protocol meaning.
- **Not a scheduler.** yavih cuts work into units and runs one unit at a time; it never writes an sbatch
  script and has no SLURM mode.
- **Not a dataset mirror.** yavih ships frozen manifests and the recipe to obtain the data from its public
  source. Dataset packers and curation pipelines that produce a published dataset are not part of yavih;
  a folder links to the published copy and pins its revision.

## The benchmark controls the input, the bridge controls the output

Every evaluation harness has to answer "who decides how a clip becomes model input?". yavih's answer is
a clean split:

- The **benchmark** decides which stored file belongs to which row, which facts come with it, how labels
  are read and how a number is computed.
- The **bridge** decides everything the model needs: which seconds and frames, at what rate, which crop
  and resolution, the audio front end, normalization, batching, the layer it taps, pooling, precision and
  storage dtype.

Harnesses that let the benchmark pre-cut windows or pre-resize frames bake one model family's
preprocessing into the protocol, and every other model is evaluated on someone else's input. Splitting
responsibility this way means the benchmark is fair to any model, and any model's preprocessing is
expressible.

### Raw input, defined per benchmark

A benchmark folder hands the bridge the rawest data it can: the original stored file (mp4, wav, jpg,
png), byte for byte, in batches. Nothing is decoded, trimmed, cropped, resized, selected, padded or
normalized, and no statistic of any kind is computed. Facts read from the file header (duration, frame
rate, sample rate, frame size) travel with it.

A benchmark may transform its input **only when the label requires it**, never to constrain the model.
avsync is the one example: its label is an audio-video offset, so the audio is shifted against the video
once at prepare time with a pinned recipe, and the result is still an mp4. What "raw" means for each
benchmark is written in its `setup.md` and in `benchmark.yaml` (`input_contract.handed_over`).

The bridge never sees labels, the manifest, the split or a file name that could leak a label: it receives
a `BenchmarkView` and `Sample` objects whose `path` is an opaque handle, and `meta` carries only the label
*facts* the folder chose to hand over (for example "the box is annotated at 5.0 s"). Labels reach the
probe by row index, never through the bridge.

### Two contracts per benchmark

Each folder states two contracts, both in `benchmark.yaml` and copied into `setup.md`:

- the **input contract**: exactly what the bridge receives per stream, which media facts are guaranteed
  non-null on every row, and which label facts are handed over;
- the **output contract**: what the probe expects back, namely one `[N, D]` token array per row with the
  same `N` and `D` for the whole feature set and, for localized labels, token positions in **media
  coordinates**.

Media coordinates make dense readouts model-agnostic. Each token carries `(stream, t, y, x)`: `t` is its
temporal centre as a fraction of that stream's duration, `y, x` its spatial centre as fractions of the
*original* frame, before any crop or resize the bridge applied, and `-1` means "no coordinate on this
axis". Spatial targets (boxes, depth maps) are stored in the same full-frame coordinates. Any crop,
resolution or frame count therefore works with the same readout; a bridge that cropped simply leaves the
cells it did not see empty.

The output contract is machine-checked by validation and by the readout's eligibility test, never only
prose.

## Bridge decisions are declared, recorded, and never identity

Every model-side decision is declared in the bridge's `describe()`: the model name and reading, a weights
fingerprint, per stream the temporal rule (frame count, span, rate), the spatial rule (resize, crop) or the
audio front end, the normalization constants and where they come from, the decoder, the tap, pooling,
precision and storage dtype, and any arguments.

Two reasons drive this:

1. **A declaration that cannot drift from what ran.** The bridge reports per clip what it actually
   consumed (`used`: frame times, sample ranges). Validation checks `used` against the declared rule and
   the media's duration, so a stale declaration is caught instead of silently recorded.
2. **Everything that produced a number travels with it.** `describe()` is stored in every sidecar and
   every `result.json`; a reader always sees how a number was made.

What `describe()` never does is decide whether two numbers compare. That is the job of the hashes (see
below). A model is a **name the user chooses** (`describe()["model"]`), read in a declared **reading**
(`a`, `v`, `av`, ...). A layer sweep or two checkpoints shown side by side are several names; the same name
reused with another configuration is one row, newest shown and flagged (`W_BRIDGE_CHANGED`).

A digest of the configuration decisions, `describe_sha256`, keeps one feature set to one configuration and
detects staleness: a resumed shard written under another configuration is set aside and re-extracted, and
extraction refuses a bridge that drifted from the profile `bridge check` recorded. Provenance (a weights
path, the batch size, the decoder library build) is outside that digest.

## Instance-wise extraction

A feature must depend on its own clip and nothing else. Batch statistics, padding to the batch maximum,
caches keyed on a clip, test-time augmentation or dropout all make a number depend on what else happened to
be in the batch or on the run. The guardrails for bridge authors, enforced where yavih can and advised
otherwise:

- no batch or dataset statistics; pad to a fixed length, never to the batch maximum;
- normalization constants only from the model's own pretraining recipe;
- one deterministic view: no random crops, jitter, dropout, test-time augmentation or multi-view averaging;
- no cross-sample state: no cache keyed on clip id or path, no online adaptation, no reading another row;
- media only, never manifest, labels or split;
- frozen weights under `inference_mode`, an explicit precision block;
- fixed `N` and `D` for every sample, positions in media coordinates;
- weights loaded before the first sample, no network and no writes outside the run directory during
  extraction;
- the model's own preprocessing source pasted into the bridge and matched by `describe()` line by line.

`yavih bridge check` makes the core of this a hard gate on a 32-clip smoke set: a rerun must be
bit-identical, and each clip encoded alone (on a fresh instance, with row, clip id and split scrambled and
fresh path handles) or in a shuffled batch must match its batched features within `1e-5`. During
extraction, canaries re-encode rows in a different batch composition so validation can catch the same
failure on the real data. The SKILL.md guardrail table names the function that checks each rule, or says
"advisory".

## The authors' implementation, through an assert-import

When a benchmark's authors ship an official loader, scorer or metric (VGGSounder's `vggsounder` package,
Probe3D's depth metrics, the VGG-SS evaluation code), yavih calls it instead of re-implementing it, to stay
faithful to the published numbers. Such code is **never a yavih dependency**: there is no extra for it. The
folder's `setup.md` names the exact install command and version, `benchmark.yaml` lists it under
`external`, the folder's `probe.py` reaches it only through `yavih.external.require(name, pip, version_pin)`
(an assert-import that names the install line), and the probe records the version used in the result.

This keeps the core installable into any model environment while letting each benchmark be as faithful
as its authors' code allows.

## One bridge per (model, benchmark), no modality

A bridge is written per (model, benchmark) pair. yavih has no notion of "modality": an audio-only, a
video-only and an audio-visual reading of the same model are three bridges that share the model's name
and declare three readings. A future modality needs no change to the harness: it is another stream kind
the reader hands over and another reading a bridge declares.

Per-benchmark bridges are deliberate. Benchmarks differ in what is stored (an mp4 with both tracks, a wav,
a jpg) and in what the output must carry (pooled tokens, timed tokens, tokens over the full frame), so a
universal bridge would hide per-benchmark decisions behind flags. A model's bridges may share code through
a `<model>_common.py` next to them.

## Bridges run in the model environment; the core is torch-free

The hand-off between model and harness is a WebDataset of features on disk: stock tar shards keyed by
manifest row, `.npy` members memmapped in place by the probe, one sidecar per shard that doubles as its
DONE marker. Nothing else crosses the boundary.

This lets the bridge run in whatever environment the model needs (its own torch, CUDA, containers), and
the probe run anywhere, later. To make that possible the yavih core (benchmark loading, the reader,
`Sample`, `FeatureWriter`, hashing, rule checks, every version constant) depends on numpy and PyYAML only
and installs with `pip install --no-deps yavih` without touching the model's torch. Decoders and torch are
optional extras (`yavih[media]`, `yavih[probe]`). Hashes are computable in the model env and in the probe
env alike, because the core that computes them is the same stdlib-and-numpy code.

## Standardized readouts with floors and controls

Every model is read out the same way: fixed heads, fixed and pinned training hyperparameters, fixed
roles. There are four readouts:

| readout | what it scores | floor / control |
|---|---|---|
| `clf` | classification (single-label, multi-label, label sets, cross-validation folds) with an attentive probe | majority class or class prior; avsync's stationary arm (`floor_violated`) |
| `seq` | per-second labels, with tokens placed by absolute second (`pos.t`) | constant segment label |
| `grid` | dense maps (depth) over the full-frame target region (`pos.y, pos.x`) | per-pixel mean train target |
| `box` | an audio-conditioned box in the full frame | mean train box; shuffled-audio control (`audio_blind`) |

Dense heads are position-aware on purpose: a permutation-invariant query head can learn the dataset prior
and report it as localization. Every readout also computes a **label-only floor** from the train targets,
and benchmarks that need it declare **validity controls**. A number is reported next to its floor or not
at all, and the report greys out rows that do not clear the floor or trip a control. A non-pinned probe
configuration is accepted only in testing mode and marks its result `standard: false`.

Train and eval roles come from `benchmark.yaml`, never from the user: validation is carved from train only,
floors use train targets only, eval splits are never capped, and the probe refuses train ∩ eval ≠ ∅.
The train population is a frozen draw committed in the manifest, so extraction never touches rows the
probe would discard and the trained population is certified by the manifest hash. Every metric carries
`n` and a 95 % interval.

## Hashes cover only what yavih controls

What a number means is decided by the benchmark and by the probe. The hashes cover exactly that:

- `input_hash`: what the bridge is handed and what its features must satisfy (manifest and classes, which
  rows and media a sample carries, splits, streams, both contracts, the readout's shape requirements, an
  optional `reader.py`);
- `scoring_hash`: how features become numbers (protocol facts, metrics, label reading, targets, pinned
  authors' packages, the readout version, the folder's `probe.py`);
- `protocol_hash = H(input_hash, scoring_hash)`; `probe_hash` for the head and training protocol;
- `eval_hash = H(protocol_hash, probe_hash)`: **the** comparability key, printed in every report the way
  lm-evaluation-harness prints a task version.

Two numbers of one benchmark compare iff their `eval_hash` agrees and both come from the pinned (standard)
probe configuration. Features are checked against the input
side only, so a metric edit re-probes and never re-extracts. No source file is hashed: identity is the
declaration (manifest content, typed yaml views, declared versions), and the code behind a declared
`protocol_version`, the folder's or yavih's, is held to it by a per-folder conformance fixture. A refactor, a
port to another framework or a third party's re-implementation that stays within tolerance keeps the hash, which
is what lets a registry of recorded scores compare models that were never run by the same code.

What the bridge decides and what the site has (paths, hosts, staged media, coverage) is recorded beside
every number and checked for consistency, but never enters a comparability key, a path or a result key.
Results are filed and compared as `(benchmark, eval_hash, model, reading)`.

When a hash moves without a change in meaning (a payload-schema change, a refactor), the folder's
changelog records the new hash and lists the old one under `supersedes`, so old results land in the same
column and old feature sets stay usable. There is no central table: each folder carries its own history.
See [hashing](hashing.md).

## Validation is help, not a gate

Validation exists to catch the mistakes that produce plausible-but-wrong numbers: a feature set that
disagrees with the manifest, a declaration that drifted from what ran, a batch-dependent bridge, a dead
token, a duplicated clip. It is not there to force every model into yavih's vocabulary.

Two legitimate opt-outs keep it that way:

- **Custom rules.** Where the declared-rule vocabulary does not fit a model's sampling, the bridge
  declares `{"rule": "custom", "description": "<one line>"}`. Validation records a note
  (`N_RULE_CUSTOM`), still checks that nothing was read outside the media, and never penalizes it.
- **Waivers.** When a check is wrong for a model (padding tokens outside `[0, 1]`, digitally silent audio
  that encodes identically), the bridge declares `describe()["waive"] = {CODE: reason}` or the operator
  passes `--waive CODE=reason`. The error becomes `W_WAIVED`, recorded with its reason and marked `†` in
  the report, never greyed. Structural checks (the shards and arrays are malformed) and identity checks
  (the features are not this benchmark's or not one configuration) are never waivable.

The rule for everyone is the same: if a check is wrong for your model, waive it with a reason; do not
edit yavih. Opt-outs are never penalized, but they are always recorded. Unknown protocol keys and
unconsumed bridge arguments, by contrast, are errors: a silent no-op is how a requested setting goes
unapplied.

## Record what ran, not what was requested

Every artefact carries the hashes it was produced under and the yavih version and commit that wrote it.
Every write goes through a temporary file and an atomic rename on the same filesystem; a DONE marker
whose hashes mismatch is treated as absent and the stale output is moved aside, never deleted. A
permanent incompatibility (pooled features on a dense readout, a model that cannot read a stream) is a
recorded `not_applicable` with exit 0, never a resubmitted failure and never a number.

## Primitives and a CLI, scheduler-agnostic

The Python API is a set of idempotent, on-disk, content-addressed functions; the CLI is a thin mirror of
them, and every command takes `--json` and prints one envelope `{status, data, errors, warnings, paths}`.
Exit codes carry the decision a driver needs: `0` done, `1` usage, `2` a contract violation to fix, `3`
not ready (resubmit later), `4` deadline or preemption (resubmit), `5` internal failure.

`yavih plan` decomposes a bridges × benchmarks matrix into independent units (extract shards, probes, a
report) with content-addressed ids, declared dependencies and resource hints taken from the `bridge
check` profile. `yavih exec` runs one unit. Any scheduler can drive that: a SLURM array, GNU parallel,
Kubernetes indexed jobs, or `yavih plan run --local` on one machine. yavih never writes an sbatch script;
cluster templates live in the skill as examples. Shards are small (about 30 minutes each) so a lost task
costs one shard, and the shard layout of a split is frozen once the first shard is written.

## One folder per benchmark, the single source of truth

Each benchmark is a self-contained folder shipped inside the package: `benchmark.yaml` (authoritative,
machine-readable), the frozen `manifest.csv.gz` and `classes.csv`, `README.md` for people
(lm-evaluation-harness task-README form: paper, citation, protocol in brief, variants, expected numbers,
changelog, checklist), `setup.md` for the agent that sets it up (obtain the data, independent
dependencies, input and output contracts, prepare and verify, gotchas), numbered `prepare/` steps, a
`make_manifest.py` that re-derives the frozen manifest byte for byte, a `bridge_template.py`, folder tests
and an optional `probe.py`. Every table that lists benchmarks is generated from the folders, and
`yavih bench lint` checks any folder against the contract, so nothing outside the folder repeats its facts.
Out-of-tree folders load the same way from `$YAVIH_BENCH_PATH` or a pip-installed pack.

## Agent-native

yavih is built so that a coding agent, not a human, connects a model to it. The package ships one skill
(`yavih skill install`): `SKILL.md` is the journey from install to report with the exact command and a stop
condition per step, plus the guardrail table and what to do when blocked; `references/` hold the bridge
contract, data notes, readouts, cluster templates and the CLI. Each folder's `setup.md` is written for the
same reader: an agent that has to obtain a dataset it has never seen and connect a model to it. What it
cannot obtain on its own (files a person must download by hand, a licence a human must accept, data only its owner has) it records as not ready
and skips, ending with a to-do list for the user.

Every shipped file (setup.md, README.md, `benchmark.yaml`, bridge templates, prepare steps, the skill
references) is reached only through the CLI (`yavih bench files|show|prepare`, `yavih bridge new`,
`yavih skill show`), which answers identically from a wheel and a checkout. An agent never has to search
site-packages, and shipped files never cite documents a pip install cannot open.

## Results stored the lm-evaluation-harness way

Under a results root there is one directory per model name, holding `results_<timestamp>.json` with every
benchmark × reading scored in that invocation, keyed `<benchmark>:<reading>` (metrics, configs with the
`eval_hash`, the bridge's `describe()` and the probe configuration, versions, `higher_is_better`, the
source commit), and optional per-sample logs. The per-unit `result.json` under `runs/` stays the
scheduler's atomic, DONE-marked record. `yavih report` prints the familiar table (benchmark, reading,
version, `eval_hash`, readout, metric, value ± CI, floor) and never puts non-comparable rows in one
column. For an audio-visual model all readings are written before any is presented, and a joint number is
never presented alone.
