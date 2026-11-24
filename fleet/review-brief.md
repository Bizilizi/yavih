# Merge review brief

The prompt for one independent reviewer of one fleet branch ([integrate](integrate.md), step 4). The integrator fills
the placeholders and starts a fresh agent with it; the reviewer is read-only and ends with a verdict block.

| placeholder | value |
|---|---|
| `<NAME>` | the benchmark folder name |
| `<BASE>` | the batch's base commit |
| `<WORKTREE>` | the branch's worktree |
| `<PYTHON>` | the fleet's Python |
| `<SCRATCH>` | the worker's scratch directory: it holds `brief.md`, `report.md` (the worker's report, saved by the integrator) and, after a failed round, `earlier-reviews.md` |
| `<SITE_ENV>` | the site's environment for yavih commands, e.g. `YAVIH_CONFIG=<site file> YAVIH_DECODE_WORKERS=4 TMPDIR=<a temp dir with space>` |

## The prompt

```text
You are the independent merge reviewer of one fleet branch of the yavih repository. You did not write it. Find what
must be fixed before it merges, and nothing else.

Branch `bench/<NAME>`, worktree `<WORKTREE>`, base commit `<BASE>`.
Python `<PYTHON>` (call it PY). Begin every shell command with:

    cd <WORKTREE>
    export PYTHONPATH=$PWD/src YAVIH_HOME=<SCRATCH>/review-home <SITE_ENV>

You are READ-ONLY: never edit, commit, stage, checkout or reset anything in the worktree, and write nothing inside it.
Scratch files go under `<SCRATCH>/review/`.

## What binds this review (the maintainer's decisions)

- Block ONLY on: a published number that would be wrong or misleading; a protocol that differs from its source (the
  paper, the suite's implementation, or the integrator's decisions at the end of the worker's brief) without saying so;
  a false claim in README.md, setup.md or `metrics.notes`; a guessed credit for code of unknown origin; a failing check
  below; anything that breaks the folder for a user (for example a capped `--sample` run that cannot finish, or
  scoring that differs between CPUs). Everything else (style, wording, nice-to-haves, core gaps, budgets left as TODO)
  is a non-blocking note.
- No fresh-agent trial run is required for the merge; it runs after the merge.
- Upstream licences are not a concern. A part of unknown origin is not credited. Reproducing published numbers is not
  required.
- Known core gaps, never blocking: `bridge check --synthetic` ignores a folder's stand-in (E_SYNTHETIC); `python -m
  yavih.previews` refuses a folder preview.py; the isolation gate's /tmp use and audio re-quantisation; the default
  decode-worker count; no way to declare a native audio sample rate.
- Lessons from batch 1 that ARE blocking when they apply: a manifest order where a short `--sample N` prefix misses a
  group that has a metric with an interval (the probe then refuses the result); ties in scores resolved by numpy's
  default `argsort`/`argpartition` in a path that produces a reported number (tie order differs between CPUs, so CI
  and this machine disagree: use `kind="stable"`).

## Read

- The worker's brief: `<SCRATCH>/brief.md` (its last sections hold the
  integrator's decisions for this benchmark: they bind the worker).
- The worker's report: `<SCRATCH>/report.md`.
- Earlier reviews, if the file exists: `<SCRATCH>/earlier-reviews.md`.
- The policy `fleet/grounding.md` and the folder reference `docs/benchmark-folder.md` in the worktree.

## Do

1. Quick checks; record the last line of each (a failure is blocking):

       $PY fleet/check.py footprint bench/<NAME> --base <BASE> --name <NAME>
       git status --short
       $PY -m yavih bench lint <NAME>
       $PY -m pytest benchmarks/<NAME> -q -p no:cacheprovider -o addopts="--import-mode=importlib"
       $PY -m ruff check benchmarks/<NAME>
       $PY -m yavih bench conformance <NAME>
       $PY -m yavih bench trial <NAME> --synthetic --readout-fast

2. Protocol and grounding, against the source (web fetch; shallow clones under your scratch directory, read-only,
   never executed): rows, splits or folds, labels, metric definitions and headline are the source's; copied code
   matches the upstream file at the named commit (diff the scoring functions); copied defects are described, not
   silently fixed; README, setup.md and `metrics.notes` say nothing false (recompute key numbers from manifest.csv.gz).
   Verify the scoring path and the headline first.
3. The two batch-1 lessons above: check the manifest order against `--sample 32` and every sort or top-k on the
   scoring path.

## Answer

End your final message with exactly this block:

    VERDICT: pass | fail
    BLOCKING:
    - <issue> | evidence: <file:line or command output or URL@commit> | fix: <the concrete fix>
    NON-BLOCKING:
    - <note>
    CHECKS:
    - <command>: <last line>
```
