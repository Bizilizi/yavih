# Fleet worker brief

The prompt for one agent that adds one benchmark folder to yavih, as part of a fleet that adds many in parallel. The
integrator fills the placeholders, starts the worker in its own worktree ([launch](launch.md)), and merges what it
reports in batches ([integrate](integrate.md)).

| placeholder | value |
|---|---|
| `<NAME>` | the queue entry's `name`: the folder name, reserved in [queue.yaml](queue.yaml) |
| `<ENTRY>` | the entry itself, as `python fleet/check.py entry <NAME>` prints it |
| `<BASE>` | the commit every worker of the batch starts from |
| `<WORKTREE>` | the worker's own git worktree, on branch `bench/<NAME>` created from `<BASE>` |
| `<PYTHON>` | a Python with yavih's dependencies, torch and the `media` extra installed (shared by the fleet) |
| `<DATA_ROOT>` | the directory staged benchmark data goes under (`<DATA_ROOT>/<NAME>`) |
| `<DISK_CAP_GB>` | the most this worker may download and stage |
| `<SCRATCH>` | a private scratch directory for this worker |

## The prompt

```text
You are adding ONE benchmark to yavih, a frozen-feature benchmark harness for video, audio and audio-visual models:
the folder benchmarks/<NAME>/ in the repository checked out at <WORKTREE> (branch bench/<NAME>, from <BASE>). Many
agents add other benchmarks in parallel; an integrator merges your branch after an independent review.

The queue entry for your benchmark (what the survey knows about it and the suggested route):
<ENTRY>
(A worker is only ever given a `kind: benchmark` entry. A `kind: suite` entry is the integrator's: a suite file in
the core, written once its components are shipped. If you were given one, stop and report it.)

Environment, for every command:
  cd <WORKTREE>
  export PYTHONPATH=<WORKTREE>/src YAVIH_HOME=<SCRATCH>/yavih-home     # your checkout's code, your own caches
  PY=<PYTHON>; yavih() { "$PY" -m yavih "$@"; }
Data goes under <DATA_ROOT>/<NAME> (never into the repository), at most <DISK_CAP_GB> GB. Scratch files go under
<SCRATCH>. Never build a wheel inside <WORKTREE> (pip leaves build/ and *.egg-info there, which break the repository's
tests): build from a copy, `mkdir -p <SCRATCH>/src && git -C <WORKTREE> archive HEAD | tar -x -C <SCRATCH>/src`, then
`SETUPTOOLS_SCM_PRETEND_VERSION=$("$PY" -m setuptools_scm) "$PY" -m pip wheel --no-deps <SCRATCH>/src -w <SCRATCH>/wheel`
(the archive has no .git, so the build is told the version git would compute).

THE FOOTPRINT RULE. Your branch changes files under benchmarks/<NAME>/ and nowhere else: not src/ (the core), not
tests/, docs/, README.md, .github/, fleet/ (the queue is the integrator's), catalog/index.json, the generated
benchmark tables, other benchmark folders. Your folder's header image and README figures are yours (benchmarks/<NAME>/assets/, step 11).
Before you report, this must print `footprint ok`:
  "$PY" fleet/check.py footprint HEAD --base <BASE> --name <NAME>
The core is read-only BY DESIGN, not as a restriction to work around: whatever your protocol needs that yavih does not
ship (a scorer, a trained head or loss, metrics, an output type, a bridge-check gate, an eligibility rule, a synthetic
stand-in, a reader, readings, capability tags, label kinds) goes into your folder through the extension points of
docs/extending-through-a-folder.md. `yavih bench extensions` lists the folders that already do this; read the closest
one and scaffold from it (`yavih bench new <NAME> --like <that folder> --dir benchmarks`). Whatever your folder
implements that the core might one day absorb goes into your report's "unification candidates" (what it is, which
file, which other folders might share it); `yavih bench info <NAME>` shows the `extends:` line the integrator checks it
against. Do not write folder notes: benchmarks/*/notes/ is ignored by git and never reaches the integrator.
If the protocol truly cannot be expressed through the folder, stop, describe the gap in your report's "core request",
put the one-line unblocking action in `blocked_by`, and report the entry blocked; never edit the core.

WHAT TO DO. Follow docs/contributing-a-benchmark.md, which is the contract, with these differences for a fleet
worker:
  - steps 1-13 are yours (scaffold, protocol, frozen manifest, prepare steps, setup.md and README.md, bridge
    template, eval.py, hashes and conformance fixture, folder tests, lint and tests, the header image and figures,
    synthetic trial, the fresh-agent trial run); step 5, "setup.md and README.md": setup.md is for the agent that
    sets the benchmark up, README.md is the benchmark's page for people, written per docs/writing-a-readme.md
    (nothing an agent needs goes into it: how to read the numbers goes into benchmark.yaml `metrics.notes`, the
    re-freeze recipe into the make_manifest.py docstring), with its "at a glance" block written by
    `python -m yavih.doctables --glance <NAME> --write` (rerun after every yaml edit; never `--write` alone);
    step 11, "Header image and figures": draft the figure spec with
    `python -m yavih.previews <NAME> --suggest --write`, edit its `rows` if the draw misrepresents the task (keep the
    pixels-and-licences rule of docs/design-system.md), then, when a headless Chrome or Chromium is at hand, render and
    commit the header with `python -m yavih.headers <NAME> --chrome <path>` and the figures from your staged data with
    `python -m yavih.previews <NAME> --chrome <path>` (ffmpeg for a motion figure; both write only
    benchmarks/<NAME>/assets/); without Chrome, ffmpeg or the data, commit the spec only and say so in your report:
    the integrator renders the rest;
  - the integrator does the rest: step 0 (your checkout exists), the generated tables and catalog/index.json (never
    run `python -m yavih.doctables --write` or `python -m yavih.catalog --write`), step 14
    (the pull request) and the queue update;
  - the reference for every file is docs/benchmark-folder.md; the route in your entry is a suggestion: take a better
    one if the paper's protocol says so, and say why.
Before writing code, find the benchmark's official protocol: paper, official splits and annotation files, metric,
licence, where the media comes from. Freeze the manifest only from public upstream files anyone can download.

DATA AND OBTAIN TIER. Decide the tier first (benchmark.yaml `obtain.tier`, docs/contributing-a-benchmark.md step 2):
  - auto (fetchable unattended): stage it under <DATA_ROOT>/<NAME> with the folder's own prepare steps. If the full
    data exceeds <DISK_CAP_GB> GB, stage what fits for the trial (a sample of every split) and say so.
  - manual-download, needs-licence, needs-data-owner, site-only: never work around a portal, a licence or an owner
    (no mirrors, no other routes). Build everything that needs no media (the manifest from the public annotation
    files, `make_manifest.py --no-media --metadata`, the prepare steps, the docs, the template, the scorer, the
    synthetic trial), then report the entry blocked with the exact unblocking action: who must act, at which URL,
    which files, into which directory.
Never fake a reading, a number or a budget. A budget you could not measure stays the scaffold's value with a `TODO`
and is listed under "Budgets" in your report as not measured, with the reason; one you derived (media_gb from the
official file sizes, minutes scaled from a sample) says how.

CHECKS, in this order, all inside your folder:
  yavih bench lint <NAME>                              # nothing but W_TODO for budgets you could not measure and the
                                                       # figure warnings of what you could not render (never
                                                       # E_README_GLANCE: rerun doctables --glance after yaml edits)
  "$PY" -m pytest benchmarks/<NAME> -q -p no:cacheprovider -o addopts="--import-mode=importlib"
  "$PY" -m ruff check benchmarks/<NAME>
  yavih bench conformance <NAME> --record               # once the protocol and scorer are final; re-record after edits
  yavih bench trial <NAME> --synthetic --readout-fast
Reading the trial: `trial ok` (exit 0) passes; `trial FAILED at <stage>` (exit 2, or 3 when that stage was not ready)
is a bug to fix; `trial UNVERIFIED` (exit 3) means nothing was checked (the authors' package the conformance stage
needs is missing): install it as setup.md "## Dependencies" says and rerun, it never counts as done. Stages reported
`not_applicable` mean no synthetic stand-in covers your scorer: give the folder a `testing.py` stand-in
(docs/extending-through-a-folder.md "A stand-in of the folder's own") so the trial runs end to end.
LAST, commit everything on bench/<NAME> (the trial run builds from the committed tree), then the fresh-agent trial
run of docs/trial-run.md on a small sample of the real data, with the wheel built from an
archive copy as above. If you cannot start a sub-agent, or the data is blocked, record `trial run: not run` and why.

FINISH. Commit on bench/<NAME> in small commits (never push, never open a pull request, never edit fleet/queue.yaml),
run the footprint check, and end with this report, filled in exactly:

## <NAME>: report
- status: review | blocked
- blocked_by: <the exact unblocking action, or none>
- obtain tier: <tier>; data staged: <what, where, GB, or none and why>
- route: <scorer route taken; why, if it differs from the queue entry's>
- commits: <hashes, one line each>
- footprint: <the footprint check's last line>
- lint / folder tests / ruff: <outcome each>
- conformance: <recorded | skipped (why)>
- synthetic trial: <verdict, exit code, stages not ok and why>
- figures: rendered | spec only (no Chrome | no ffmpeg | no data) | none (why)
- fresh-agent trial run: <pass | fail | not run> with the TRIAL_REPORT.md path; friction found and fixed
- budgets: media_gb, extract_min_per_1k_clips_ref, probe_minutes_ref: value and measured | derived (how) | not measured (why)
- unification candidates: <one bullet per item: what, which file, which other folders might share it; or none>
- core request: <the gap the folder cannot express, if blocked for that reason; or none>
- open questions for the integrator: <or none>
```
