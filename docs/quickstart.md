# Quickstart by hand

Every step of benchmarking a model without a coding agent, on one benchmark (VGGSounder, visual reading), with a
complete bridge. With an agent you only install yavih, run `yavih skill install` and ask it to benchmark your
model; the agent follows the same steps, from the skill ([`SKILL.md`](../src/yavih/skill/yavih/SKILL.md)).

## 1. Install into the model's environment

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

## 2. Discover and prepare a benchmark

```bash
yavih bench list
yavih bench info vggsounder --setup   # the folder setup.md: obtain, dependencies, input/output contracts, verify, gotchas
# ...follow setup.md "Obtain the data" and "Prepare & verify", then:
yavih bench verify vggsounder --census   # manifest hash, presence per split, read sample, media-info table, coverage
```

Data roots are resolved from `~/.config/yavih/config.toml` (`[data] root = ...`, per-benchmark overrides under `[data.<name>]`, shared trees under `[datasets]`), the project file `yavih-bench/yavih.toml` found upward from the current directory (or `--config FILE`), per-cluster `[site.<id>]` tables, `YAVIH_DATA__<name>__<root>` environment variables, and `--data <name>.<root>=PATH`; `yavih config show --bench <name>` prints every value with where it came from (skill reference `cli.md`, "Configuration").

A repository keeps everything yavih-related in one folder, `yavih-bench/` (`yavih init` writes it): the project configuration with one `[site.<id>]` per cluster, `bridges/`, `matrices/` with their profiles, `suites/`, project-local `benchmarks/` and the project's own `tools/`. One per repository, found from any directory below the root; `yavih doctor` warns about a second project file or a stray matrix ([project layout](project-layout.md)).

## 3. Write a bridge

Run every command below from the repository root; every path is relative to it. Create the project folder once,
then copy the benchmark's template into it and fill it in (shipped files are reached through `yavih bench
files|show|prepare`, `yavih bridge new` and `yavih skill show`, the same from a wheel and a checkout):

```bash
yavih init                                                 # once per repository: yavih-bench/ (config, bridges/, matrices/, suites/, benchmarks/, tools/)
yavih bench show vggsounder setup                          # the contract (= yavih bench info vggsounder --setup)
yavih bridge new vggsounder --model mymodel                # → yavih-bench/bridges/mymodel_vggsounder.py, model name filled in
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
    """My model: video only. The reading is an argument (vggsounder scores a, v and av; `plan init` plans all three).

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
    ARGS = {"reading": ArgSpec(type=str, default="v", choices=("a", "v", "av"), help="which streams the model reads"),
            "ckpt": ArgSpec(type=str, default="/ckpts/mymodel.pt")}

    def __init__(self, bench, args):
        super().__init__(bench, args)
        self.reading = self.args["reading"]
        if self.reading != "v":     # a video-only model: the a and av entries are recorded as not_applicable
            raise yavih.NotApplicable(f"video-only model: no {self.reading!r} reading")
        self.model = load_my_model(self.args["ckpt"]).eval().cuda()   # weights before the first sample

    def describe(self):             # recorded with every number; never what makes numbers comparable
        return {"model": "mymodel", "reading": self.reading,   # the name results are filed under (same in every bridge of the model)
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

The bridge never sees labels; every model-side decision (frames, crop, normalization and its source, decoder, tap, pooling, precision) is declared in `describe()` and recorded with every number. It is never part of a path, a key or what makes numbers comparable: results are filed under `<runs root>/<benchmark>/<model>/<reading>/` and compared by `eval_hash` (benchmark + probe). Its digest `describe_sha256` keeps one feature set to one configuration: an argument, rule, precision or weights change re-extracts on resume and shows as a new configuration of the same `(model, reading)` row (`W_BRIDGE_CHANGED`, newest shown; the replaced configuration's output is moved to `probe/_stale/` and listed under `--detail`); a path, batch-size or decoder-build change does nothing. Two configurations you want side by side (another layer, another checkpoint) are two `model` names.

## 4. Smoke-test, then extract

```bash
yavih bridge check yavih-bench/bridges/mymodel_vggsounder.py --bench vggsounder --synthetic
# smoke checks on tiny generated media with vggsounder's streams and facts: before any data is staged, no profile
yavih bridge check yavih-bench/bridges/mymodel_vggsounder.py --bench vggsounder --arg reading=v --out yavih-bench/matrices/mymodel.profiles/mymodel-vggsounder-v.profile.json
# 32-clip smoke on the staged data: describe(), hard checks (rerun bit-identical, batch- and order-invariance), the
# declared rules against `used`, positions in media coordinates, clips/s, and the profile (model, reading,
# describe_sha256) of the matrix entry mymodel-vggsounder-v below, next to its matrix (the command `plan init`
# prints for that entry; with --arg reading=a or av the bridge raises NotApplicable and the profile records it)
```

Single machine (`--out` is the run directory under the runs root; features land in `<out>/features/<split>/`):

```bash
RUNS=$(yavih config get runs.root)        # the runs root: yavih-bench/runs unless [runs] root puts it on scratch
RUN=$RUNS/vggsounder/mymodel/v
yavih extract --bench vggsounder --bridge yavih-bench/bridges/mymodel_vggsounder.py --split train --shard 0/1 --out $RUN --arg reading=v
yavih extract --bench vggsounder --bridge yavih-bench/bridges/mymodel_vggsounder.py --split test  --shard 0/1 --out $RUN --arg reading=v
```

Cluster: describe models × benchmarks once, let yavih cut it into units, run them as an array. A suite is a named
benchmark set (`default` is what "benchmark my model" means, `minimal` a smoke pass; [suites](suites.md)).
For a whole suite, `plan init` writes the matrix and one bridge stub per benchmark (keeping the bridge written
above), and prints the `bridge check --out <profile>` command of every entry and the `plan build` command:

```bash
yavih suite show default                  # capabilities, benchmarks, readings
yavih plan init --suite default --model mymodel   # -> yavih-bench/matrices/mymodel.yaml + yavih-bench/bridges/mymodel_<b>.py (--launcher unless the site config names one)
```

or by hand, one entry per bridge and benchmark (paths in a matrix are relative to the matrix file; this one uses
the bridge and the profile of steps 3 and 4):

```yaml
# yavih-bench/matrices/mymodel.yaml
schema_version: 1
out: null                                 # the configured runs root (default yavih-bench/runs)
defaults: {launcher: mymodel, cpus: 8}    # a [launchers] name from yavih.toml (per site), a python path, or null
bridges:
  mymodel-vggsounder-v: {script: ../bridges/mymodel_vggsounder.py, benchmark: vggsounder,
                         args: {reading: v}, profile: mymodel.profiles/mymodel-vggsounder-v.profile.json}
```

```bash
M=yavih-bench/matrices/mymodel.yaml
P=$RUNS/_plans                            # plan files are run state: under the runs root, never in yavih-bench/
yavih plan build $M --sample 32 --out $P/mymodel-trial.jsonl && yavih plan run $P/mymodel-trial.jsonl --local   # capped run first: 32 rows per split
yavih plan build $M --out $P/mymodel.jsonl                 # extract shards (GPU) -> probe (CPU/GPU) -> report
yavih exec $P/mymodel.jsonl --index $SLURM_ARRAY_TASK_ID --deadline slurm   # one unit; idempotent; exit 4 = resubmit
yavih plan status $P/mymodel.jsonl
yavih suite missing default --model mymodel                 # what the suite still lacks for this model
```

## 5. Validate, probe, report

```bash
yavih features validate $RUN/features
yavih probe run        $RUN --seeds 0,1,2
yavih report build     "$RUNS/*" --fmt md
```

Every command takes `--json` and returns one envelope `{status, data, errors, warnings, paths}`; exit codes are `0` done / `1` usage / `2` contract violation to fix / `3` not ready / `4` deadline or preemption (resubmit) / `5` internal.


## What yavih enforces

- A bridge missing a `describe()` key or an explicit precision block, a torch module left in training mode, or a configuration that drifted from its `bridge check` profile (`E_CONFIG_DRIFT`) is refused before a single clip is encoded. `bridge check` also refuses an unfilled template (`E_TEMPLATE_UNFILLED`) and features that do not change between clips (`E_CONSTANT_FEATURES`). Which guardrail is checked by which code, and which are advisory, is the table in SKILL.md "Guardrails".
- Features are checked against the manifest: every row exactly once, finite, one `(N, D, dtype, batch size, describe_sha256, model, reading)` across shards; `used` checked against the declared rules (`E_RULE_MISMATCH`, `E_TIMES_OUT_OF_MEDIA`), token positions inside the crop region and the media span consumed, canaries (`E_BATCH_INVARIANCE`), dead token rows, duplicated clips, `pos` ranges.
- The probe refuses stale hashes, training on an eval split, train ∩ eval ≠ ∅ and eval coverage below the benchmark's threshold. Pooled features on a dense readout become a recorded `not_applicable`, never a number.
- Every readout emits its floor. The report greys out rows that do not clear it, and never a row because of a bridge decision (a `custom` rule, unfingerprinted weights).
<!-- yavih:bench-table controls begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
- `avsync` and `vggss_box` emit controls (`floor_violated`, `audio_blind`); the report greys out the rows they void.
<!-- yavih:bench-table controls end -->

