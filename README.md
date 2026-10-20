<p align="center">
  <img src="https://raw.githubusercontent.com/Bizilizi/yavih/main/assets/header.png" alt="yavih — frozen-feature benchmark harness for video and audio models" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/yavih/"><img alt="PyPI" src="https://img.shields.io/pypi/v/yavih?color=22d3ee"></a>
  <a href="https://github.com/Bizilizi/yavih/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Bizilizi/yavih/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/yavih/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/yavih?logo=python&logoColor=white"></a>
  <a href="https://github.com/Bizilizi/yavih/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-a78bfa"></a>
</p>

**yavih** (Yet Another Video Harness) evaluates frozen video, audio and audio-visual encoders on a fixed suite of benchmarks with standardized probes — and it is built so that a coding agent, not a human, connects your model to it.

yavih owns the *protocol*: which clips, which stored files, which labels, which readout, which metric, which floor, which hash. You (or your agent) own the *model*: a small **bridge** that turns a benchmark's raw files into tokens, run inside your model's own Python environment. The two halves meet on disk, in a WebDataset of features. yavih then trains the same probe every model gets, scores it against a floor, and reports only numbers that are comparable.

## Why

Frozen-feature evaluation is simple in principle and fragile in practice. Harnesses grow model-specific adapters, preprocessing drifts between benchmarks, a probe quietly trains on an eval split, a position-blind head learns the dataset prior and reports it as localization, and a renamed checkpoint reuses stale features. yavih is designed around the incidents that produce plausible-but-wrong numbers:

- **Benchmark states two contracts, the bridge complies.** The reader hands over raw files plus facts; frames, window, crop and front end are the bridge's decisions, declared in `describe()` and checked against what it reports it consumed.
- **One bridge per (model, benchmark)**, a plain feature extractor. yavih has no notion of "modality": audio-only, video-only and audio-visual readings of the same model are three bridges, so any future modality fits without changing the harness.
- **Bridges run in the model environment.** `yavih` core is torch-free (numpy + PyYAML) and installs with `--no-deps`; the hand-off is a documented tar format. Probing runs anywhere, later.
- **Standardized readouts with mandatory floors.** Attentive probe for classification; position-aware dense heads for segments, depth grids and boxes. Every `result.json` carries the label-only floor and validity controls next to the metric.
- **Hash only what yavih controls.** Comparability is the benchmark (`protocol_hash`, manifest included; features are checked against its input side only, so a metric or `probe.py` edit re-probes without re-extracting) and the probe (`probe_hash`), folded into one `eval_hash` that the report prints next to the protocol version, as lm-evaluation-harness prints a task version; two numbers compare iff it is equal, and the report never puts other rows in one column. Everything the bridge decides is recorded with the number (`describe()`, its `describe_sha256`) and never compared.
- **Primitives first, scheduler-agnostic.** Idempotent, content-addressed units with resource hints run as SLURM arrays on any cluster; yavih never writes an sbatch script.

## Philosophy

- **Raw input, defined per benchmark.** The benchmark hands your bridge the original media file — mp4, wav, jpg — in batches. No decoding, trimming, cropping, resizing, frame selection or padding, and no statistics. A benchmark may transform the input only when its *label* requires it (avsync shifts audio against video; the result is still an mp4), never to constrain the model. Each folder's `setup.md` says exactly what "raw" means there.
- **Two contracts, the bridge complies.** Every benchmark states an *input contract* (what the bridge receives, plus facts it may need, e.g. "the box is annotated at 5.0 s") and an *output contract* (what the probe expects: `[N, D]` tokens, and for localized labels token positions in media coordinates — time as a fraction of the clip, space as a fraction of the original frame). Output contracts are machine-checked.
- **Everything else is the bridge's decision** — frames, fps, window, crop, resolution, audio front end, normalization, tap, pooling — and every decision is declared in `describe()`, recorded with every result and never part of what makes results comparable. Where the declared-rule vocabulary does not fit, a `custom` rule with a one-line description is a legitimate opt-out (a note, never penalized). The model is a name you choose (`describe()["model"]`), read in a declared `reading` (`a`, `v`, `av`). Bridges extract **instance-wise**: no batch or dataset statistics, no cross-sample state, deterministic single view, frozen weights.
- **The authors' implementation when one exists.** Official loaders, scorers and metrics (VGGSounder's package, Probe3D depth metrics, VGG-SS evaluation) are called, not re-implemented. They are never yavih dependencies: the agent installs them while setting the benchmark up, scripts assert the import with an install hint, and the result records the version used.
- **Results stored like lm-evaluation-harness:** one directory per model name, `results_<timestamp>.json` with every benchmark and reading scored (keys `<benchmark>:<reading>`), optional per-sample logs, and a `benchmark / reading / version / eval_hash / metric / value ± CI / floor` table.

## How it works

```
prepare ──► extract ──► validate ──► probe ──► report
 agent        bridge       yavih       yavih     yavih
 (setup.md)   (model env)  (features   (fixed    (tables,
                            vs manifest) readout   floors,
                                        + floor)  hashes)
```

| stage | who | input | output |
|---|---|---|---|
| `prepare` | agent, following the benchmark `setup.md`; `yavih bench verify --census` | raw dataset | media on disk, verified manifest, coverage, media-info table |
| `extract` | `yavih extract` in the model env: reader (raw files + facts) → your bridge → `FeatureWriter` | media + manifest | `features/<split>/shard-*.tar` + per-shard sidecar (the DONE marker) |
| `validate` | `yavih features validate` (also the probe prologue) | shards + manifest | pass/fail, `_meta.json`, coverage, readout eligibility |
| `probe` | `yavih probe run` | `_meta.json` + shards + manifest | `result.json` per seed: metrics, floors, controls, CIs, hashes |
| `report` | `yavih report build` | many `result.json` | md / csv / json tables, comparability-checked |

## Benchmarks

Each benchmark ships as a self-contained folder (`src/yavih/benchmarks/<name>/`), the single source of truth for its protocol: a `README.md` for people (a header image, links, then description, citation and the protocol in yavih), a `setup.md` for the agent (obtain the data, dependencies to install independently, input and output contracts, prepare & verify, gotchas), a machine-readable `benchmark.yaml`, the frozen manifest, a reader, an optional `probe.py` (folder-specific metrics and the authors' scorer) and a bridge template. The index is [`src/yavih/benchmarks/README.md`](https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks/README.md); `yavih bench lint <name|path>` checks a folder against the contract and `yavih bench new <name> --like esc50` scaffolds one. Out-of-tree folders load from `$YAVIH_BENCH_PATH` or a pip-installed pack (entry-point group `yavih.benchmarks`). The table below is generated from the packaged folders:

<!-- yavih:bench-table readme begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
| benchmark | task | readout | headline | source | frozen rows |
|---|---|---|---|---|---|
| `audioset20k` | AudioSet-20K (balanced) 527-class multi-label audio-visual tagging, published raw clips (macro mAP) | `clf` | `map` | public | 17 726 train / 16 223 eval |
| `ave` | AVE audio-visual event localization (28 events + background, per-second labels) | `seq` | `seg_acc` | public | 2 624 train / 316 test |
| `avsync` | AV synchronization-offset classification (9 offsets, transient VGGSound clips, stationary floor) | `clf` | `acc` | derived | 24 000 train / 3 000 test / 5 997 stationary |
| `esc50` | ESC-50 environmental sound classification (50 classes, official 5-fold CV, mean accuracy) | `clf` | `acc_mean` | public | 2 000 in folds 1–5 (400 each) |
| `fsd50k` | FSD50K 200-class multi-label audio tagging (Freesound clips, macro mAP) | `clf` | `map` | public | 20 000 train / 10 231 eval |
| `in1k` | ImageNet-1K image classification (frozen 50/class train draw, full 50k val); aliases `imagenet1k`, `imagenet` | `clf` | `acc` | manual portal | 50 000 train / 50 000 val |
| `k400` | Kinetics-400 action recognition (Academic-Torrents _256 release, 400 classes, video-only probe) | `clf` | `acc` | torrent | 20 000 train / 13 492 val |
| `nyuv2_depth` | NYUv2 dense depth (12 000 GeoNet train frames, 654 labelled test images, 14x14 grid) | `grid` | `d1` | manual portal | 12 000 train / 654 eval |
| `ssv2` | Something-Something-v2 temporal action classification (frozen 20k class-balanced train draw, full 24 777 val) | `clf` | `acc` | manual portal | 20 000 train / 24 777 val |
| `vggsounder` | VGGSounder audio / visual / audio-visual multilabel probe (VGGSound-train probe, 309 classes) | `clf` | `auc_pr` | public | 20 000 train / 12 008 test |
| `vggss_box` | VGG-SS audio-conditioned single-box regression | `box` | `iou_mean` | public | 3 700 train / 914 eval |
<!-- yavih:bench-table readme end -->

`yavih bench list` shows what ships and what is present at your site; `yavih bench info <name>` prints the README with live coverage (`--setup` prints the setup.md). Every shipped protocol records its version history in `benchmark.yaml` `changelog`, and its README "## In yavih" says where it deliberately differs from the original paper's protocol.

## Quickstart

### 1. Install into the model's environment

```bash
# inside the environment that runs your model; --no-deps keeps its torch untouched
pip install --no-deps yavih
python -c "import numpy, yaml"   # --no-deps assumes numpy>=1.24 and pyyaml>=6 are already there; else: pip install numpy pyyaml
yavih --version
yavih doctor --bench vggsounder  # (or python -m yavih …) import layers, media read per stream, data roots, external packages, GPU
```

In a fresh environment, `pip install yavih` brings numpy and PyYAML along. The development version installs from
GitHub: `pip install --no-deps "git+https://github.com/Bizilizi/yavih"`. Offline compute nodes: download the wheel
once where there is network, then install the file.

```bash
pip download --no-deps -d dist yavih          # -> dist/yavih-<version>-py3-none-any.whl
pip install --no-deps dist/yavih-*.whl        # on the node, in each env
```

The model env needs no decoder from yavih: your bridge decodes with the model's own stack, or with the optional helpers of the `[media]` extra (`pip install av soundfile pillow`). yavih itself only reads each file's header facts (duration, frame rate, sample rate, size): from the site census (`yavih bench verify <b> --census`) when present, else with PyAV / soundfile / pillow, else with the `ffprobe` CLI; `yavih doctor --bench <b>` warns (`W_HEADER_BACKEND`) and `extract` refuses at start (exit 3) when none is available. Probing needs torch in whatever environment runs the probes — it can be a different one: `pip install "yavih[probe]"` (or the wheel plus `pip install torch`), plus the benchmark authors' packages its setup.md lists (never yavih extras).

### 2. Discover and prepare a benchmark

```bash
yavih bench list
yavih bench info vggsounder --setup   # the folder setup.md: obtain, dependencies, input/output contracts, verify, gotchas
# ...follow setup.md "Obtain the data" and "Prepare & verify", then:
yavih bench verify vggsounder --census   # manifest hash, presence per split, read sample, media-info table, coverage
```

Data roots are resolved from `~/.config/yavih/config.toml` (`[data] root = ...`, per-benchmark overrides under `[data.<name>]`, shared trees under `[datasets]`), the project file `yavih-bench/yavih.toml` found upward from the current directory (or `--config FILE`), per-cluster `[site.<id>]` tables, `YAVIH_DATA__<name>__<root>` environment variables, and `--data <name>.<root>=PATH`; `yavih config show --bench <name>` prints every value with where it came from (skill reference `cli.md`, "Configuration").

A repository keeps everything yavih-related in one folder, `yavih-bench/` (`yavih init` writes it): the project configuration with one `[site.<id>]` per cluster, `bridges/`, `matrices/` with their profiles, `suites/`, project-local `benchmarks/` and the project's own `tools/`. One per repository, found from any directory below the root; `yavih doctor` warns about a second project file or a stray matrix ([project layout](https://github.com/Bizilizi/yavih/blob/main/docs/project-layout.md)).

### 3. Write a bridge

Copy the benchmark's template and fill it in (shipped files are reached through `yavih bench files|show|prepare`,
`yavih bridge new` and `yavih skill show`, the same from a wheel and a checkout):

```bash
yavih bench show vggsounder setup                          # the contract (= yavih bench info vggsounder --setup)
yavih bridge new vggsounder --model mymodel                # → bridges/mymodel_vggsounder.py, model name filled in
```

```python
import numpy as np, torch, yavih
from yavih import grid_pos
from yavih.bridge import ArgSpec, fingerprint_state_dict
from yavih.media.decode import decoder_info, frames_at
from yavih.media.rules import crop_region, rule_times

TEMPORAL = {"rule": "uniform", "n": 16, "span": "full"}          # which frames: YOUR decision, declared
SPATIAL = {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}

class Bridge(yavih.Bridge):
    """My model, video reading.

    Preprocessing source:
        TODO <repo>@<commit> <file>::<function>, then the model's own eval transform, verbatim, as this indented
        block; describe() must match it line by line (bridge check warns W_NO_PREPROCESSING_SOURCE while a line
        here starts with TODO or the block holds only <...> placeholders)
    """
    name = "mymodel-video"
    batch_size = 8
    dtype = "float16"
    pos_mode = "per_sample"         # durations and frame sizes differ per clip
    streams = ("video",)
    ARGS = {"ckpt": ArgSpec(type=str, default="/ckpts/mymodel.pt")}

    def __init__(self, bench, args):
        super().__init__(bench, args)
        self.model = load_my_model(self.args["ckpt"]).eval().cuda()   # weights before the first sample

    def describe(self):             # recorded with every number; never what makes numbers comparable
        return {"model": "mymodel", "reading": "v",       # the name results are filed under (same in every bridge of the model)
                "weights_sha256": fingerprint_state_dict(self.model), "weights_ref": self.args["ckpt"],
                "dtype": self.dtype, "batch_size": self.batch_size, "streams": list(self.streams), "pos_mode": self.pos_mode,
                "layer": "last", "pooling": "none", "views": 1,
                "input": {"video": {"temporal": TEMPORAL, "spatial": SPATIAL,
                                    "norm": {"scale": 1 / 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                                             "source": "mymodel@<commit> transforms.py"},
                                    "decoder": decoder_info("video")}},
                "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
                "args": dict(self.args)}

    def encode(self, samples):      # one clip at a time inside: instance-wise, never padded to the batch maximum
        feats, pos, used = [], [], []
        for s in samples:
            st = s.streams["video"]                     # the stored mp4, undecoded; st.info = header facts
            frames, pts = frames_at(st, rule_times(TEMPORAL, st.info))   # YOU decode (or st.open() with your own stack)
            feats.append(self.model(preprocess(frames, SPATIAL)))          # [N, D]
            region = crop_region(SPATIAL, st.info["width"], st.info["height"])
            t = (pts.reshape(8, 2).mean(1) / st.info["duration_s"]).tolist()  # 2-frame tubelets, media coordinates
            pos.append(grid_pos(8, 14, 14, stream=0, t_centres=t, region=region))
            used.append({"video": {"times": pts.tolist()}})                  # what was consumed, checked by validate
        return np.stack(feats), np.stack(pos), used
```

The bridge never sees labels; every model-side decision (frames, crop, normalization and its source, decoder, tap, pooling, precision) is declared in `describe()` and recorded with every number. It is never part of a path, a key or what makes numbers comparable: results are filed under `runs/<benchmark>/<model>/<reading>/` and compared by `eval_hash` (benchmark + probe). Its digest `describe_sha256` keeps one feature set to one configuration: an argument, rule, precision or weights change re-extracts on resume and shows as a new configuration of the same `(model, reading)` row (`W_BRIDGE_CHANGED`, newest shown; the replaced configuration's output is moved to `probe/_stale/` and listed under `--detail`); a path, batch-size or decoder-build change does nothing. Two configurations you want side by side (another layer, another checkpoint) are two `model` names.

### 4. Smoke-test, then extract

```bash
yavih bridge check bridges/mymodel_vggsounder.py --bench vggsounder --arg ckpt=/ckpts/mymodel.pt \
      --out bridges/mymodel-vggsounder.profile.json
# 32-clip smoke: describe(), hard checks (rerun bit-identical, batch- and order-invariance), the declared rules
# against `used`, positions in media coordinates, clips/s, profile.json (model, reading, describe_sha256)
yavih bridge check bridges/mymodel_vggsounder.py --bench vggsounder --synthetic
# the same checks on tiny generated media with vggsounder's streams and facts: before any data is staged, no profile
```

Single machine (`--out` is the run directory; features land in `<out>/features/<split>/`):

```bash
RUN=runs/vggsounder/mymodel/v
yavih extract --bench vggsounder --bridge bridges/mymodel_vggsounder.py --split train --shard 0/1 --out $RUN --arg ckpt=/ckpts/mymodel.pt
yavih extract --bench vggsounder --bridge bridges/mymodel_vggsounder.py --split test  --shard 0/1 --out $RUN --arg ckpt=/ckpts/mymodel.pt
```

Cluster: describe models × benchmarks once, let yavih cut it into units, run them as an array. A suite is a named
benchmark set (`default` is what "benchmark my model" means, `minimal` a smoke pass; [suites](https://github.com/Bizilizi/yavih/blob/main/docs/suites.md)),
and `plan init` writes its matrix and one bridge stub per benchmark:

```bash
yavih init                                # once per repository: yavih-bench/ (config, bridges/, matrices/, suites/, benchmarks/, tools/)
yavih suite show default                  # capabilities, benchmarks, readings
yavih plan init --suite default --model mymodel   # -> yavih-bench/matrices/mymodel.yaml + bridges/mymodel_<b>.py (--launcher unless the site config names one)
```

or by hand, one entry per bridge and benchmark:

```yaml
# yavih-bench/matrices/mymodel.yaml
schema_version: 1
out: null                                 # the configured runs root
defaults: {launcher: mymodel, cpus: 8}    # a [launchers] name from yavih.toml (per site), a python path, or null
bridges:
  mymodel-vggsounder: {script: ../bridges/mymodel_vggsounder.py, benchmark: vggsounder,
                       args: {ckpt: /ckpts/mymodel.pt}, profile: mymodel.profiles/mymodel-vggsounder.profile.json}
```

```bash
M=yavih-bench/matrices/mymodel.yaml
yavih plan build $M --sample 32 --out trial.jsonl && yavih plan run trial.jsonl --local   # capped run first: 32 rows per split
yavih plan build $M --out plan.jsonl                     # extract shards (GPU) -> probe (CPU/GPU) -> report
yavih exec plan.jsonl --index $SLURM_ARRAY_TASK_ID --deadline slurm   # one unit; idempotent; exit 4 = resubmit
yavih plan status plan.jsonl
yavih suite missing default --model mymodel                 # what the suite still lacks for this model
```

### 5. Validate, probe, report

```bash
yavih features validate runs/vggsounder/mymodel/v/features
yavih probe run        runs/vggsounder/mymodel/v --seeds 0,1,2
yavih report build     'runs/*' --fmt md
```

Every command takes `--json` and returns one envelope `{status, data, errors, warnings, paths}`; exit codes are `0` done / `1` usage / `2` contract violation to fix / `3` not ready / `4` deadline or preemption (resubmit) / `5` internal.

## For agents

The package ships one skill that is the single entrance for a coding agent:

```bash
yavih skill install                      # -> .claude/skills/yavih/  (SKILL.md + references/)
yavih skill install --target codex       # or cursor (.cursor/rules/yavih.mdc), agents-md (a block in AGENTS.md)
```

`SKILL.md` ([`src/yavih/skill/yavih/SKILL.md`](https://github.com/Bizilizi/yavih/blob/main/src/yavih/skill/yavih/SKILL.md)) is the nine-step journey with a stop condition per step (what output must be seen before moving on) and the guardrail checklist; `references/` hold the bridge contract, per-benchmark data notes and SLURM templates. Each benchmark folder's `setup.md` is written for the same reader: an agent that has to download a dataset it has never seen, prepare it, and connect a model to it. What it cannot obtain on its own (files a person must download by hand, a licence a human must accept, data only its owner has: `obtain` in `yavih bench list`) it records as not ready and skips, finishing the other benchmarks and ending with a to-do list for you (SKILL.md "When blocked").

The nine steps (install, setup.md, prepare, bridge, smoke, extract, validate, probe, report) with their exact commands and stop conditions are the table in [`SKILL.md`](https://github.com/Bizilizi/yavih/blob/main/src/yavih/skill/yavih/SKILL.md); this README does not repeat it, so the two cannot drift. Asked to "benchmark my model", the agent runs the `default` suite (`yavih plan init --suite default`) without asking which benchmarks.

Adding a benchmark is also written for an agent: [contributing a benchmark](https://github.com/Bizilizi/yavih/blob/main/docs/contributing-a-benchmark.md) is one ordered list of steps from a fork to the pull request, each with its commands and a "done when" line. `yavih bench trial <name> --synthetic` runs a new folder's whole pipeline on a generated stand-in, and the required [trial run](https://github.com/Bizilizi/yavih/blob/main/docs/trial-run.md) has a fresh agent set the benchmark up from the built wheel and the skill alone before the pull request is opened.

Where to read next: [design](https://github.com/Bizilizi/yavih/blob/main/docs/design.md) (why yavih works this way), the [reference](https://github.com/Bizilizi/yavih/blob/main/docs/README.md#reference) (what exactly it does: benchmark folders, the bridge contract, the feature format, validation codes, readouts, hashing, the CLI), [development](https://github.com/Bizilizi/yavih/blob/main/docs/development.md) (working on yavih itself), and the [releases](https://github.com/Bizilizi/yavih/releases) (what changed).

## What yavih enforces

- A bridge missing a `describe()` key or an explicit precision block, a torch module left in training mode, or a configuration that drifted from its `bridge check` profile (`E_CONFIG_DRIFT`) is refused before a single clip is encoded. `bridge check` also refuses an unfilled template (`E_TEMPLATE_UNFILLED`) and features that do not change between clips (`E_CONSTANT_FEATURES`). Which guardrail is checked by which code, and which are advisory, is the table in SKILL.md "Guardrails".
- Features are checked against the manifest: every row exactly once, finite, one `(N, D, dtype, batch size, describe_sha256, model, reading)` across shards; `used` checked against the declared rules (`E_RULE_MISMATCH`, `E_TIMES_OUT_OF_MEDIA`), token positions inside the crop region and the media span consumed, canaries (`E_BATCH_INVARIANCE`), dead token rows, duplicated clips, `pos` ranges.
- The probe refuses stale hashes, training on an eval split, train ∩ eval ≠ ∅ and eval coverage below the benchmark's threshold. Pooled features on a dense readout become a recorded `not_applicable`, never a number.
- Every readout emits its floor. The report greys out rows that do not clear it, and never a row because of a bridge decision (a `custom` rule, unfingerprinted weights).
<!-- yavih:bench-table controls begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
- `avsync` and `vggss_box` emit controls (`floor_violated`, `audio_blind`); the report greys out the rows they void.
<!-- yavih:bench-table controls end -->

## Feature format

```
runs/<benchmark>/<model>/<reading>/features/<split>/
  shard-000003.tar            000012345.json  000012345.feat.npy [N, D]  000012345.pos.npy [N, 4]  c000012345.* (canary)
  shard-000003.json           sidecar = DONE marker: row range, sha256, batch size, describe() (declared rules), hashes, timing
  _meta.json                  written by validate: coverage, fingerprint, eligibility verdict
runs/<benchmark>/<model>/<reading>/probe/<eval_hash12>/seed<k>/{result.json, DONE}
results/<model>/results_<timestamp>.json      lm-eval style, entries keyed <benchmark>:<reading>
```

Stock WebDataset tars keyed by manifest row; `.npy` members are memmapped in place by the probe, so a 64 GB token set never has to fit in RAM.

## Repository layout

```
src/yavih/
  benchmark.py sample.py media/      protocol side: benchmark.yaml, manifests, raw files + facts (census), rules, optional decode helpers
  bridge.py extract.py io/           bridge contract, extraction, feature format, validation
  probes/ metrics/ readouts/         standardized heads, metric registry, result.json
  report/ plan/ cli.py               comparability tables, units/exec, CLI
  benchmarks/<name>/                 README.md setup.md benchmark.yaml manifest.csv.gz classes.csv probe.py bridge_template.py make_manifest.py prepare/ tests/ (an optional reader.py)
  skill/yavih/                       SKILL.md + references/
docs/                                design, reference pages, developer guide (not shipped in the wheel)
tests/                               contract tests incl. an end-to-end run on synthetic benchmarks
.github/workflows/                   CI (ruff, pytest on 3.10–3.13, wheel smoke test) and the PyPI release
```

## Status

The core pipeline, all four readouts, the agent skill and every packaged benchmark folder (the table above) are implemented and covered by the test suite (core, per-folder and skill), which CI runs on Python 3.10–3.13; the core imports without torch, and the wheel carries every folder and the skill. Every folder's manifest is frozen and was verified against locally staged data. Protocol versions are pinned per benchmark; deliberate divergences from the published protocols are recorded in each folder's README ("## In yavih"). Bugs and questions: [GitHub issues](https://github.com/Bizilizi/yavih/issues).

## Releasing

Versions come from git tags; pushing a tag `vX.Y.Z` builds, publishes to PyPI and creates the GitHub Release with generated notes. See [docs/releasing.md](https://github.com/Bizilizi/yavih/blob/main/docs/releasing.md).

## License

yavih is released under the [MIT License](https://github.com/Bizilizi/yavih/blob/main/LICENSE). The benchmark datasets are not part of yavih and keep their own licences and terms of use (each folder's `benchmark.yaml` `source.licence` and setup.md "## Obtain the data").
