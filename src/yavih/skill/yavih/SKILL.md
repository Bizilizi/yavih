---
name: yavih
description: >
  Evaluate or benchmark a video, audio, image or audio-visual model with FROZEN features through
  yavih (frozen-feature benchmark harness): reading a benchmark's setup.md, writing a yavih bridge
  that turns the benchmark's raw files into tokens, extracting features, running the standardized
  probe readouts (clf / seq / grid / box) and reporting with floors. Use whenever the request
  mentions yavih, a bridge, frozen-feature or linear/attentive probing, or one of the benchmarks
  K400, SSv2, ESC-50, AudioSet-20K, FSD50K, VGGSounder, IN1K, AVSync, AVE, NYUv2 depth or VGG-SS
  boxes, or a benchmark suite ("benchmark my model" = the default suite). Nine steps, each with the
  exact command and the output that must be seen before moving on; contracts live in references/.
---

# yavih — nine steps, one stop condition each

yavih is a CLI + library: `yavih <group> <cmd> [--json]` (or `python -m yavih …`). With `--json` (or
`YAVIH_JSON=1`) every command prints exactly one envelope `{"status","data","errors","warnings","paths"}`;
warnings are `[yavih] warning CODE: …` lines on stderr. Exit codes: 0 done/skipped/not_applicable,
1 usage, 2 contract violation (fix bridge or data), 3 not ready (data, dependency or authors' package
missing), 4 deadline/preempted (resumable), 5 internal.

How the work splits: a benchmark hands your bridge the RAW stored files (mp4, webm, wav, jpg, png) plus
facts (duration, fps, sample rate, frame count, size, "the box is annotated at 5.0 s"). It never
decodes, cuts, crops, resizes, selects frames or pads. Everything model-side is yours and is declared
in `describe()`, recorded with every number and never part of what makes numbers comparable (that is
the benchmark and the probe, folded into one `eval_hash`). Your model is a NAME you choose
(`describe()["model"]`), read in a `reading` (`a`, `v`, `av`); results are filed under both. The
bridge returns `[N, D]` tokens per clip and, for localized labels, token positions in MEDIA
coordinates (t = fraction of the clip's duration, (y, x) = fraction of the ORIGINAL frame). Each
benchmark writes both contracts in its `setup.md`: read it first.

Use this skill when a model must be scored on a frozen benchmark, a bridge has to be written or
debugged, features must be validated, or a probe/report must be read. Not for training or
fine-tuning: yavih reads models, it never updates them.

## 0. Install (into the MODEL env, never a new one)

```bash
# the model's own python; --no-deps keeps its torch untouched, and assumes numpy>=1.24 and pyyaml>=6 are there
pip install --no-deps yavih     # PyPI; else "git+https://github.com/Bizilizi/yavih"; offline node: a prebuilt wheel
python -c "import numpy, yaml"                                  # --no-deps installs neither
yavih init                     # once per repository: yavih-bench/ (yavih.toml, bridges/, matrices/, suites/, benchmarks/, tools/)
yavih doctor --bench <b>
```

**Project folder.** Everything yavih-related lives in `yavih-bench/` at the repository root (`yavih init`; one per
repository, found from any directory below the root; `yavih doctor` warns `W_PROJECT_NESTED` / `W_PROJECT_STRAY`
/ `W_PROJECT_VERSION`): `yavih.toml` with one `[site.<id>]` per cluster (data roots, `launcher`, `[launchers]`),
`bridges/`, `matrices/<model>.yaml` + `<model>.profiles/`, `suites/`, `benchmarks/`, `tools/` (your drivers and
scripts; yavih never reads them). Clusters are site tables, never separate folders or copies; runs and results go
to the configured roots, never into the repository; personal roots belong in `~/.config/yavih/config.toml`. Fine
outside the folder: a bridge shipped in a model package (`script: pkg.mod:Class`), a benchmark folder that is the
repository's deliverable (`[project] benchmarks`), job scripts kept with the project's infrastructure.

**Hard rule — shipped files.** Everything yavih ships (setup.md, README.md, benchmark.yaml, `bridge_template.py`,
`prepare/` scripts, this skill's references) is reached ONLY through `yavih bench files <b>` (list, absolute paths),
`yavih bench show <b> setup|readme|yaml|template|prepare/<file>`, `yavih bench prepare <b> [<step>]`,
`yavih bridge new <b> --model <name>` and `yavih skill show --ref <name>`. Never search site-packages or a repository
checkout for them: a wheel install has no checkout, and these commands answer the same everywhere. Exploring a benchmark:

```bash
yavih suite list                          # benchmark sets: default, minimal (`suite show <s>`, `suite new <s>`)
yavih bench list                          # all benchmarks: readout, clips, obtain tier, data status
yavih bench files <b>                     # every shipped file, absolute path + kind (--json)
yavih bench show <b> setup                # contracts, data, gotchas: read first
yavih bench show <b> readme|yaml|probe    # paper/protocol, machine spec, extra metrics (probe.py: only some benchmarks)
yavih bench show <b> prepare/<file>       # read a prepare script before running it
yavih bench prepare <b> [<step>]          # list / run prepare steps
yavih bench path <b> [<file>]             # a real path when a tool needs one (sbatch, cp)
yavih skill show --ref bridge|data|readouts|slurm|cli
```

The bridge decodes with the model's own stack, or with the optional torch-free `yavih.media.decode`
helpers (`pip install av soundfile pillow`). The probe env needs torch and the benchmark's authors'
packages (`external`, installed exactly as setup.md "## Dependencies" says; never a yavih extra).
Data roots and config: references/data.md. `yavih doctor --selftest` runs the shipped ESC-50
reference bridge.

## 1–9. The journey

| step | command | continue only when you have seen |
|---|---|---|
| 1 install | `pip install --no-deps yavih` (model env); `yavih init` (once per repository); `yavih doctor --bench <b>` | `torch.untouched == true`; `project` line present, no `W_PROJECT_*`; every data root `exists`; `media_read … ok`; the `external` packages listed (probe env, step 2) |
| 2 setup.md | `yavih suite show default` names the benchmarks (the default set; `minimal` = a smoke pass); per benchmark `yavih bench info <b> --setup` (= `yavih bench show <b> setup`) FIRST, then `yavih bench list` (the `obtain` column), `yavih bench info <b>`, `yavih bench files <b>` | source obtainable here (else "When blocked"); budget fits; every "## Dependencies" package installed and `ok` in `yavih doctor --bench <b>` (probe env); "## Input contract" and "## Output contract" read |
| 3 prepare | setup.md "## Obtain the data" and "## Prepare & verify": `yavih bench prepare <b>` lists the steps, `yavih bench prepare <b> <step> [-- args]` runs one; `yavih bench verify <b> --census` | eval coverage ≥ 95 %; `labels_ok`; `read` sample ok; media-info table written; `derived_sha_ok` (nyuv2_depth) true where not null |
| 4 bridge | `yavih bridge new <b> --model <name>` copies that benchmark's `bridge_template.py` to the project's `bridges/<name>_<b>.py` (a bridge file serves one benchmark); paste the model's preprocessing source into its docstring; fill `describe()`: `model`, `reading`, the `input` rules (a `custom` rule with a one-line description where none fits), `views: 1`, explicit `precision`; decode in `prepare()` (it overlaps the forward pass; `encode()` reads `sample.prepared`); `encode()` returns `(feats, pos, used)`, pos in media coordinates | `describe()` matches the pasted source line by line; `model` is the same string in every bridge of the model |
| 5 smoke | `yavih bridge check <bridge.py> --bench <b> --arg k=v --out matrices/<model>.profiles/<model>-<b>-<r>.profile.json` per matrix entry (the exact command `plan init` prints; `--synthetic` before data is staged) | rerun bit-identical; batch/order invariance within `1e-5` (or a declared ≤ `1e-3` with a reason); no `E_RULE_MISMATCH`, `E_TIMES_OUT_OF_MEDIA`, `E_NETWORK`, `E_TEMPLATE_UNFILLED`; profile written |
| 6 extract | `yavih plan init --suite default --model <name>` writes `matrices/<name>.yaml` + bridge stubs (keeps existing bridges; `--launcher <python>` unless `yavih.toml` names one per site), then `yavih plan build yavih-bench/matrices/<name>.yaml --out plan.jsonl` (`--sample N` first: a capped trial run); run the units (SLURM: references/slurm.md; no SLURM: `yavih plan run plan.jsonl --local [--slots K]`); `yavih plan status plan.jsonl` | `pending == 0`, `preempted == 0`, `failed == 0` (`not_ready` → fix data/dependency) |
| 7 validate | `yavih features validate runs/<b>/<model>/<reading>/features` | `ok` per split; coverage 100 % or accepted; `rules checked:` names your rules (`N_RULE_CUSTOM` fine; `W_RULE_UNCHECKED` understood); `eligible for` read (ineligible = recorded `not_applicable`) |
| 8 probe | `yavih exec plan.jsonl --id <probe unit>` or `yavih probe run runs/<b>/<model>/<reading> --seeds 0,1,2 --log-samples` | `status: ok`; headline above its floor; no `floor_violated` / `audio_blind`; `external` versions recorded |
| 9 report | `yavih report build 'runs/*' --out <dir>/report.md` (`--merge-results`: one file per model) | table (`benchmark`, `reading`, `version`, `eval_hash`, `readout`, `metric`, `value`, `ci95`, `floor`) written; `results/<model>/results_<timestamp>.json` present; every reading of an AV model present; compare only at equal `eval_hash`; `W_BRIDGE_CHANGED` understood; no greyed row presented as a result |

Step 4, the template copy (the model name filled in; every other TODO stays, so `bridge check` refuses the copy with
`E_TEMPLATE_UNFILLED` until `weights_ref`, the input rules and the norm constants come from the model's own code):

```bash
yavih bench show <b> setup                       # the contract the bridge must meet
yavih bridge new <b> --model <name> [--reading a|v|av]   # → yavih-bench/bridges/<name>_<b>.py (--out, --force)
yavih bench show <b> template                    # the pristine template, if you need to compare
```

**One model on many benchmarks**: "benchmark my model" means the `default` suite: run it without asking which
benchmarks, and say so in one line ("running the default suite: …"). A custom set is `yavih suite new <name>`
(a project copy to edit); before reporting, `yavih suite missing default --model <name>` lists what is still
missing. Share model loading, the forward pass and common `describe()` keys in `bridges/<model>_common.py`
(each `<model>_<b>.py` subclasses it; the bridge directory is on `sys.path`, references/bridge.md §7).
Each matrix entry (file + args) is one reading of one model.

## The two contracts (setup.md, per benchmark)

- **Input contract** — what the reader hands over per stream (`input_contract.handed_over`), the
  `RawStream.info` keys guaranteed non-null (`input_contract.info`), the label facts in
  `Sample.meta["facts"]` (`input_contract.facts`: ave `num_segments`, `segment_seconds`; vggss_box
  `annotated_second`; nyuv2_depth `target_region`), and the prepare-time materialization the LABEL
  requires (avsync only). Nothing else: seconds, frame count, crop, resolution, resampling, padding
  and normalization are decisions you declare.
- **Output contract** — one fixed `[N, D]` for every row of every split; `pos (stream, t, y, x)` in
  media coordinates; `output_contract.requires` = the readout's eligibility: `seq` (ave) needs video
  tokens with `t` on ≥ 8 of the 10 scored seconds (`requires.streams: [video]`, so an audio-only model
  is `not_applicable` on ave); `grid` (nyuv2_depth) needs image tokens whose full-frame `(y, x)` cover
  ≥ 50 % of the 14×14 cells of the target region; `box` (vggss_box) needs video tokens covering ≥ 25 %
  of the full frame's 14×14 cells plus tokens tagged `audio`. Boxes and depth live in full-frame
  coordinates, so any crop works. `validate` checks it; a profile that cannot meet it is planned as
  `not_applicable`.

## Guardrails: who checks what

Every bridge decision (frames, window, crop, resolution, front end, normalization, tap, pooling,
precision, dtype, batch size) is declared in `describe()`, recorded, never compared. Where the rule
vocabulary does not fit, a `custom` rule with a one-line description is a legitimate opt-out (the note
`N_RULE_CUSTOM`, never penalized). "advisory" = nothing can check it: you are trusted.

**If a check is wrong for your model, waive it with a reason — don't edit yavih**: `describe()["waive"] = {CODE: reason}`
(features already extracted: `yavih features validate <dir> --waive CODE=reason`) makes it `W_WAIVED`, recorded and
marked `†` in the report, never greyed. Waivable: `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`, `E_POS_RANGE`,
`E_BATCH_INVARIANCE`, `E_RERUN_DRIFT`, `E_CONSTANT_FEATURES`, `E_DEAD_ROWS`, `E_DUPLICATE_CLIPS`, `E_ARGS_UNCONSUMED`;
structural and identity checks and `E_TEMPLATE_UNFILLED` never (`E_WAIVE_INVALID`; references/bridge.md §6).

| rule | checked by | outcome |
|---|---|---|
| filled template: no `TODO` in `model`, `weights_ref`, `norm.source`; no null `norm` constant | `yavih.extract.unfilled_describe` (bridge check) | `E_TEMPLATE_UNFILLED` |
| the bridge reads its input (features differ between clips) | `yavih.extract.constant_features` (bridge check); per token `yavih.io.validate.validate_features` | `E_CONSTANT_FEATURES`, `E_DEAD_ROWS` |
| one deterministic view (`views: 1`; no random crops, jitter, dropout, TTA) | `yavih.bridge.validate_describe`; the rerun pass of `yavih.extract.bridge_check` | exit 2, `E_RERUN_DRIFT` |
| instance-wise: no batch statistics, no padding to the batch maximum | alone + shuffled passes of `yavih.extract.bridge_check`; canaries in `yavih.io.validate.validate_features` | `E_BATCH_INVARIANCE` |
| no cross-sample state (no cache keyed on clip id or path, no online adaptation) | `yavih.extract.scramble` (alone pass, fresh instance, scrambled identity) | `E_BATCH_INVARIANCE` |
| normalization constants from the model's own pretraining recipe, never dataset statistics | advisory (`norm.source` is recorded, not verified) | — |
| media only: never manifest, labels or split | by construction (`yavih.bridge.BenchmarkView`) | — |
| frozen weights, `.eval()`, `inference_mode` | `yavih.extract._find_training_modules`; `yavih.extract._inference_ctx` | `E_TRAIN_MODE` |
| explicit precision | `yavih.bridge.validate_describe`; `yavih.extract._enforce_precision` | `E_PRECISION_UNDECLARED` |
| fixed `N` and `D`, finite values, fp16 range | `yavih.io.writer.FeatureWriter` | exit 2 |
| declared rules = what was read (`used`); a checkable rule's facts guaranteed | `yavih.io.validate.check_member` (bridge check, validate); `yavih.bridge.validate_describe` | `E_RULE_MISMATCH`, `E_TIMES_OUT_OF_MEDIA`, `E_INPUT` |
| positions in media coordinates, in range | `yavih.io.validate.check_member` | `E_POS_RANGE` |
| `used` reported per clip and stream | `yavih.extract._Engine._check_used` | `E_USED` |
| no network inside `encode`; weights loaded before the first sample | `yavih.extract._audit_hook`; loading order advisory | `E_NETWORK` |
| no writes outside the run directory | `yavih.extract._audit_hook` | `W_SIDE_EFFECT` (warning) |
| the model's preprocessing source pasted into the docstring | `yavih.bridge.preprocessing_source_check` | `W_NO_PREPROCESSING_SOURCE` (warning) |
| `describe()` matches the pasted source line by line | advisory | — |
| extraction runs the configuration `bridge check` profiled | `yavih.extract.config_drift` | `E_CONFIG_DRIFT` |

Corrupt media your decoder cannot read: raise `yavih.MediaUndecodable(clip_ids, reason)` (those rows
become missing); any other exception aborts the shard. `probe` refuses missing authors' packages (exit 3
with the pip line), stale hashes, train ∩ eval ≠ ∅ and low eval coverage; `report` greys rows below the
floor (never for a bridge decision). Hashes cover a benchmark's declaration, never code: `yavih bench conformance`
holds a folder's `protocol_version` to its numbers (references/cli.md).

## Operational rules

- Install with `--no-deps`; never let pip touch the model env's torch;
  install original-author packages independently, exactly as setup.md says.
- Read setup.md before anything else for every benchmark; the input and
  output contracts there are what the bridge must comply with.
- Default dtype float16; switch to float32 when `bridge check` warns about
  range.
- Many small shards, never one unit per GPU; resubmit from `plan status`.
- `validate` before `probe`; read the floor before the metric; a CI
  overlapping the floor, `floor_violated` or `audio_blind` is not a result.
- Write all readings (audio, video, av) of an audio-visual model; never
  present a joint number alone.
- One model, one `model` name in every bridge; two configurations you want
  side by side (another layer, another checkpoint) are two names. Numbers
  compare only at equal `eval_hash`; the bridge's configuration is recorded,
  not compared.
- Opting out of the rule vocabulary is fine when it does not fit: declare
  `{"rule": "custom", "description": "<one line>"}`; validate notes it
  (`N_RULE_CUSTOM`) and nothing is greyed.
- Evaluate a comparison on one cluster; cross-hardware probe noise is ~1 pt.

## When blocked

A benchmark you cannot finish here is a recorded outcome, not a reason to stop or to ask mid-run.

1. `yavih bench list` shows `obtain` per benchmark: `auto`, `manual-download` (a person downloads it from an
   official portal), `needs-licence` (a human signs a licence), `needs-data-owner` (only its holder has it),
   `site-only`; `yavih bench info <b>` says how. Never work around a licence or fetch a private source another way.
2. Record it: one line per benchmark in your run log, `<b>: not_ready — <reason> (<what unblocks it>)`.
   yavih's exit 3 means the same: data, dependency or authors' package missing.
3. Continue with the other benchmarks. Still write the blocked one's bridge and smoke-test it with
   `yavih bridge check <file> --bench <b> --synthetic` (tiny generated media, no data, no profile), so
   only staging remains. Never fake a reading (zeros, noise, `NotApplicable` for missing data).
4. End with a to-do list for the user: per blocked benchmark the exact action (accept the licence at
   <url>, download <files> from <url> into <dir>, request <files> from the dataset authors or data
   holder, install <package>) and the commands that resume it (`yavih bench verify <b> --census`,
   `yavih bridge check … --out …`, `yavih plan build`).

## What the outputs look like (values illustrative)

```
$ yavih --json bridge check bridges/x_k400.py --bench k400 --out bridges/x-k400.profile.json   # envelope, abridged
  data.model "x"  data.reading "v"  data.N 1568  data.D 1024  data.pos_mode "per_sample"
  data.invariance {"rerun_bit_identical": true, "batch_rel_l2": 3e-07, "order_rel_l2": 3e-07, "tolerance": 1e-05}
$ yavih features validate runs/k400/x/v/features
  shards 31/32 done, 0 in progress, 1 missing: --shard 7/32           # ← rerun that shard, exit 2
  rules checked: video uniform + spatial
  eligible for clf: yes
$ yavih report build 'runs/*'
  | benchmark | reading | version | eval_hash | readout | metric | value | ci95 | floor |
  | ave | av | 2 | 10914c9d2195 | seq | seg_acc | 0.412 | [0.36, 0.47] | 0.164 |
```

The run directory is `runs/<b>/<model>/<reading>/` with `features/<split>/shard-NNNNNN.{tar,json}`,
`_meta.json` (written by validate) and `probe/<eval_hash12>/seed<k>/{result.json,DONE}`; the results
store is `results/<model>/results_<timestamp>.json` (entries keyed `<b>:<reading>`; +
`samples_<b>_<reading>_<timestamp>.jsonl` with `--log-samples`), next to the runs root unless
`YAVIH_RESULTS` or `--results DIR` says otherwise.

The matrix (`yavih plan init` writes `matrices/<model>.yaml`, one per model): its keys, an example and when `plan
build` refuses a profile are in references/slurm.md "matrix.yaml".

## Contributing a benchmark

Adding a benchmark to yavih itself: clone https://github.com/Bizilizi/yavih and follow
https://github.com/Bizilizi/yavih/blob/main/docs/contributing-a-benchmark.md step by step (`yavih bench new`
scaffolds the folder; every step ends with a "Done when" check). The fresh-agent trial run of
https://github.com/Bizilizi/yavih/blob/main/docs/trial-run.md is required before the pull request.

## References

Print one with `yavih skill show --ref <name>` (bridge, data, readouts, slurm, cli).

- `references/bridge.md` — RawStream/Sample/BenchmarkView; `describe()["input"]` rules with a worked
  example per media kind; `used`; `grid_pos`; invariance checks; shared `<model>_common.py`; template walkthrough.
- `references/data.md` — obtainability and budget per benchmark, data roots, inode rules, prepare-time transforms.
- `references/readouts.md` — what each readout measures, floors, controls, CIs, eligibility, results files.
- `references/slurm.md` — `matrix.yaml`, array templates, driver loop, preemption.
- `references/cli.md` — command table and flags (`init`, suites, `plan init`, `--sample`, `bench trial`), `--json`
  envelope, exit codes, every code, the configuration layers (`[project]`, sites, launchers).

Install this skill where your agent reads skills: `yavih skill install --target claude|codex|cursor|agents-md
[--path P]` (stamped: `yavih doctor` warns `W_SKILL_STALE` after a yavih upgrade); print it with `yavih skill show`,
one reference with `yavih skill show --ref bridge` (`--all` appends every reference).
