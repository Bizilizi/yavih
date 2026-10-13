# Trial run

Before a benchmark pull request is opened ([contributing a benchmark](contributing-a-benchmark.md), step 13), a
fresh agent sets the new benchmark up from the built wheel and the shipped skill alone, connects a toy model to
it and runs the whole chain on a small sample of the real data. It is the check nothing else makes: `bench lint`
and the folder tests read the files, `yavih bench trial --synthetic` runs a generated stand-in, and only an agent
that has never seen the folder finds where `setup.md`, the bridge template or the skill make it guess. It also
exercises the folder's own `probe.py` hooks and `external` packages, and measures the budgets.

It costs one sub-agent session and a few minutes of CPU on a sample of a few dozen rows per split, plus whatever
obtaining the data costs (point the worker at data that is already staged to skip that).

## Roles

- **Reviewer**: the contributor's agent, working in the pull request's checkout. It builds the wheel, prepares the
  workspace, starts the worker, reads the report, fixes what the report found and pastes the final report into
  the pull request body.
- **Worker**: a fresh sub-agent with no access to the contributor's checkout, the repository documentation or
  the reviewer's conversation. It has the built wheel, an empty workspace, the network (for pip and the
  benchmark's public downloads) and, when given, the staged data. It follows only the yavih skill.

## Protocol

1. **Build the wheel** from the pull request branch, after committing (the wheel's version then names the commit
   the report is about):

   ```bash
   TRIAL=$(mktemp -d)
   pip wheel --no-deps . -w "$TRIAL/wheel"          # -> $TRIAL/wheel/yavih-<version>-py3-none-any.whl
   ```

2. **Prepare an empty workspace**, `mkdir "$TRIAL/work"`, outside the checkout and with nothing copied from it.
   When the data is already staged, note its root (`yavih config get <name>.media_root` in the checkout) for the
   prompt; otherwise the worker obtains it as setup.md says, which only works unattended for `obtain.tier: auto`.
3. **Start the worker** as a separate sub-agent whose working directory is `$TRIAL/work`, with the prompt below and
   its placeholders filled. Never pass it the checkout's path.
4. **Read the report** (`$TRIAL/work/TRIAL_REPORT.md`) against the [criteria](#reviewer-criteria). Fix the
   friction it found in the branch: setup.md wording, a prepare step, the bridge template, budgets. When the fix
   changes what an agent reads or runs, rebuild the wheel and run a new worker; never resume the old one, since it
   has learned the workaround.
5. **Paste the final report** into the pull request body under "Trial report".

## Worker prompt

Placeholders: `<BENCH>` the benchmark's name; `<WHEEL>` the absolute path of the wheel; `<WORK>` the workspace;
`<PYTHON>` a Python 3.10+ interpreter to build the venv from; `<DATA>` either `the data is staged at <path>: use
it as the <root> data root` or `no data is staged: obtain it as setup.md says`; `<N>` the sample size (32 to 64).

```text
You are checking whether a new benchmark of yavih (a frozen-feature benchmark harness) can be set up by an agent
that has never seen it. Work only inside <WORK>. You have the yavih wheel <WHEEL>, the network for pip and for the
benchmark's public downloads, and: <DATA>. You do not have the yavih source repository: do not look for it, and do
not read files out of site-packages. Reach every shipped file through `yavih` commands, as the skill says. Do not
edit anything yavih ships; when something does not work, work around it if you can and record it.

1. Toy model. Create a small installable model repository <WORK>/toymodel (pyproject.toml + toymodel/__init__.py)
   holding a tiny deterministic encoder: torch.manual_seed(0), then for each stream kind a 2-layer MLP
   (Linear(F, 64) -> GELU -> Linear(64, 32)) on CPU, in eval mode, with no downloads. Its preprocessing, which you
   will paste into the bridge as the model's own:
   - video: 8 frames at uniform times over the clip; per frame the mean RGB of each cell of a 4x4 grid scaled to
     [0, 1] (F = 3 per cell token), or the 48 values as one token per frame when the benchmark's output contract
     needs no spatial positions;
   - audio: the mono waveform at its stored sample rate, cut into 8 equal windows; per window the log energies of
     16 equal-width rfft bands (F = 16), one token per window;
   - image: the mean RGB of each cell of a 4x4 grid (F = 3), one token per cell.
   Audio-visual readings concatenate both streams' tokens. N and D are fixed for every clip.
2. Environment. <PYTHON> -m venv <WORK>/venv; with its pip: pip install torch --index-url
   https://download.pytorch.org/whl/cpu; pip install numpy pyyaml av soundfile pillow; pip install --no-deps
   <WHEEL>; pip install --no-deps -e <WORK>/toymodel. This one venv is both the model env and the probe env.
3. From here on follow only the skill: `yavih skill show`, `yavih skill show --ref <name>` when it points to a
   reference, and `yavih bench show <BENCH> setup` before anything else for the benchmark. Install the packages
   setup.md "## Dependencies" lists, exactly as it says.
4. Data: obtain and prepare it as setup.md says (`yavih bench prepare <BENCH> ...`), or configure the staged root,
   then `yavih bench verify <BENCH> --census`.
5. A project suite with just this benchmark: `yavih suite new trial --from minimal`, edit the new suites/trial.yaml
   to one group with one entry, <BENCH> with every reading its bridge template declares (`join: true` when the
   folder's probe.py defines a join), and check it with `yavih suite show trial`.
6. Bridge and matrix: `yavih plan init --suite trial --model toy --launcher <WORK>/venv/bin/python`. Fill the
   bridge stub(s) it wrote with the toy model: the preprocessing source pasted into the docstring, every TODO,
   decoding in prepare() and the forward pass in encode(). Then `yavih bridge check ... --synthetic` and the
   `yavih bridge check ... --out <profile>` lines plan init printed, one per entry.
7. A capped run: `yavih plan build matrix.yaml --sample <N> --out plan.jsonl`, `yavih plan run plan.jsonl --local`,
   `yavih plan status plan.jsonl`.
8. Read the outputs by hand, under the capped runs root (`<runs root>-sample<N>`): `yavih features validate` on each
   run's features directory, `yavih probe run <run dir> --no-results` (a probe the plan already finished is
   skipped, which is fine), and `yavih report build '<runs root>-sample<N>/*' --suite trial`.
9. Write <WORK>/TRIAL_REPORT.md from the template below. Time every stage (wall clock). Report what happened: a
   stage that failed is reported as failed, never as a number you did not see.
```

## Report template

````markdown
# Trial run: <BENCH>

## Environment

yavih <version> (wheel <file name>), Python <x.y>, torch <x.y> (CPU), OS / CPU <...>, decode workers <n>.
Data: <staged at ... | obtained with ... in <minutes>>. Sample: `--sample <N>`.

## Stages

| stage | command(s) | wall time | outcome |
|---|---|---|---|
| obtain / prepare | | | |
| verify --census | | | |
| suite new + edit | | | |
| plan init + bridge filled | | | |
| bridge check --synthetic | | | |
| bridge check --out | | | |
| plan build --sample | | | |
| plan run --local | | | |
| features validate | | | |
| probe run | | | |
| report build | | | |

## Failed commands

<every command that exited non-zero: the command, the exit code, the error code and message, the cause, what you
did about it; "none" if none>

## Warnings

<every W_ / N_ code seen: where, and whether you understood it (what it means here) or not>

## Friction

<every place where setup.md, the skill or the bridge template made you guess, with the quoted text and what would
have answered it; "none" if none>

## Budgets

| | measured | benchmark.yaml | ratio |
|---|---|---|---|
| extract minutes per 1k clips (1000 / `clips_per_s` / 60, from `bridge check`) | | `extract_min_per_1k_clips_ref` | |
| probe minutes, scaled to the sample (wall x full rows / sampled rows) | | `probe_minutes_ref` | |

## Headline vs floor

| reading | metric | value | ci95 | floor | partial | controls |
|---|---|---|---|---|---|---|

## Readings covered

<the readings the template declares, and which of them ran end to end>
````

## Reviewer criteria

The trial passes when:

- **every stage finished**: each command exited 0, or a refusal is the documented outcome (a `not_applicable`
  reading the output contract predicts);
- **no unexplained warning**: every warning in the report is understood, either by the worker or, after reading
  it, by the reviewer, who then says so in the pull request;
- **the friction is gone**: every friction point is fixed in the branch (setup.md, a prepare step, the template)
  or explained in the pull request body;
- **the budgets hold**: the measured extract and scaled probe minutes are within 2× of
  `budget.extract_min_per_1k_clips_ref` and `budget.probe_minutes_ref`, or those values are re-measured and updated
  in the branch;
- **the chain produces a number**: every reading has a headline value with its floor and the controls the folder
  declares. A toy model is expected near the floor; what is checked is that the pipeline produces a number and the
  floor and controls logic fires, not that the model is good;
- **every reading is covered**: `a`, `v` and `av` wherever the bridge template declares them;
- **the report is in the pull request body**, verbatim.
