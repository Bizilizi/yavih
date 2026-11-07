# README migration brief

The prompt for one agent that rewrites ONE shipped benchmark folder's legacy README (`## Description` / `## Citation`
/ `## In yavih`) as the human page of [writing a README](../docs/writing-a-readme.md), and moves what the old README
told agents to where agents read it. The integrator picks the folders marked `readme: legacy` in
[queue.yaml](queue.yaml), fills the placeholders, starts one worker per folder in its own worktree
([launch](launch.md)), and merges what they report ([integrate](integrate.md), "README migration batch").

| placeholder | value |
|---|---|
| `<NAME>` | the folder name, a `status: shipped`, `readme: legacy` entry of [queue.yaml](queue.yaml) |
| `<ENTRY>` | the entry itself, as `python fleet/check.py entry <NAME>` prints it |
| `<BASE>` | the commit every worker of the batch starts from |
| `<WORKTREE>` | the worker's own git worktree, on branch `readme/<NAME>` created from `<BASE>` |
| `<PYTHON>` | a Python with yavih's dependencies and the `media` extra installed (shared by the fleet) |
| `<CONFIG>` | the configuration file that names this site's data roots (exported as `YAVIH_CONFIG`), or `none` |
| `<CHROME>` | a headless Chrome or Chromium binary, or `none` |
| `<SCRATCH>` | a private scratch directory for this worker |

## The prompt

```text
You are rewriting the README of ONE benchmark folder of yavih, a frozen-feature benchmark harness for video, audio
and audio-visual models: benchmarks/<NAME>/ in the repository checked out at <WORKTREE> (branch readme/<NAME>, from
<BASE>). Other agents rewrite other folders in parallel; an integrator merges your branch after an independent review.

The queue entry of your benchmark:
<ENTRY>

Environment, for every command:
  cd <WORKTREE>
  export PYTHONPATH=<WORKTREE>/src YAVIH_HOME=<SCRATCH>/yavih-home     # your checkout's code, your own caches
  export YAVIH_CONFIG=<CONFIG>                                         # skip this line when <CONFIG> is none
  PY=<PYTHON>; yavih() { "$PY" -m yavih "$@"; }
Scratch files go under <SCRATCH>. You download nothing.

WHY. A folder's README.md is the benchmark's page for PEOPLE: a researcher deciding whether the benchmark is for
them. Agents never need it: they read setup.md, benchmark.yaml (`yavih bench info`), metrics.notes and the
make_manifest.py docstring. Your folder's README is still in the legacy shape, whose "## In yavih" section mixes
agent material (split tables, scorer details, noise guidance, re-freeze commands) into the page. You split it by
audience: what a person needs stays, rewritten as a short project page; what an agent needs moves to its home; what
the yaml already says is dropped.

READ FIRST, completely: docs/writing-a-readme.md (the guide: audience, tone, length, the sections, which figure for
which task, what never goes in a README and where it goes, a worked example); the "README.md contract" section of
docs/benchmark-folder.md; then your folder: README.md (the old page), benchmark.yaml, setup.md, the module docstring
of make_manifest.py, eval.py if present, tests/test_folder.py; and the paper behind the links line.

THE FOOTPRINT. Your branch changes files under benchmarks/<NAME>/ and nowhere else (not src/, tests/, docs/,
fleet/, the generated tables, catalog/index.json, other folders). This must print `footprint ok` before you report:
  "$PY" fleet/check.py footprint HEAD --base <BASE> --name <NAME>
Inside the folder you may change only:
  - README.md: rewritten;
  - benchmark.yaml: ONLY the `metrics.notes` key (added); never another key, never the changelog, never
    protocol_version;
  - make_manifest.py: ONLY its module docstring, and `argparse.ArgumentParser(description=__doc__)` (or its first
    paragraph) where the parser does not take it yet;
  - setup.md: facts moved from the old README into "## Dependencies" or "## Gotchas";
  - eval.py, reader.py, prepare/*, tests/*.py: ONLY comments and docstrings that point at README "## In yavih"
    (eval.py's digest ignores comments and docstrings), and folder tests that pin README.md wording (below);
  - assets/: preview.yaml and the rendered figures (step 6).
No hash may move. Before you start and before you report, run
  yavih --json bench info <NAME> | "$PY" -c "import json,sys; d=json.load(sys.stdin)['data']; print(d['protocol_hash'], d['input_hash'], d['scoring_hash'])"
and keep both lines: they must be identical. `yavih bench lint <NAME>` reports E_CHANGELOG_HASH if one moved.

STEPS
1. Take the old README apart. For every sentence of "## Description" and "## In yavih", decide where it goes:
   - the task, the data, the official splits in prose, the licence, the metric in words, chance and the floor in
     words, what is NOT this benchmark (variants, aliases, near-misses), published reference numbers with their
     citation: the new README, rewritten for a person;
   - floor values, CI widths, noise, the smallest gap that means something, "a number near the floor on one fold
     means a broken extraction": benchmark.yaml `metrics.notes` (a list of at most 8 sentences, each at most 200
     characters; plain sentences, no markdown);
   - how the manifest is re-frozen (upstream files and pins, curation, audits, the commands with --check, a
     `yavih bench prepare <NAME> <step>` re-freeze step): the make_manifest.py docstring, which must name `--check`
     in a `Usage::` block; add only what it lacks, never weaken what it states;
   - which code is the authors' and which is yavih's transcription, and how they differ; a dependency fact: setup.md
     "## Dependencies" (or "## Gotchas" for a pitfall);
   - split tables, row counts, scorer names and configuration, `protocol_version`, hashes, the headline id, CI
     kinds, what the bridge gets and returns: dropped (the glance block, `yavih bench info` and setup.md say it).
   Keep a short list of what went where; it goes into your report.
2. Write metrics.notes into benchmark.yaml (only that key; keep the file's comments and layout). Check:
   `yavih bench info <NAME>` prints your `note:` lines and the hashes above did not move.
3. Check the make_manifest.py docstring against the old re-freeze paragraph and fix the gaps (step 1). Check:
   `yavih bench show <NAME> make_manifest` and `"$PY" benchmarks/<NAME>/make_manifest.py --help` show the recipe.
4. Move transcription and dependency facts into setup.md. Fix every reference inside the folder to README
   "## In yavih" (comments, docstrings, setup.md); `grep -rn "In yavih" benchmarks/<NAME>` must print nothing.
   Folder tests that pin README.md wording (a quoted number, a heading, a phrase): delete those assertions (the lint
   checks the structure; prose is not tested); where the pinned fact is real (a floor value), pin it against
   benchmark.yaml or metrics.notes instead.
5. Write the new README.md: the header image line and the links line as they were (add a [Licence](…) link when
   the licence has a page), then the glance block, written with
     "$PY" -m yavih.doctables --glance <NAME> --write
   (never by hand; rerun it after any yaml edit), then ## The task, ## The data, ## How it is scored, the optional
   ## Reference numbers, and ## Citation with the old bibtex block. 60 to 120 lines. Follow the guide's tone: plain,
   specific, for a person; words before identifiers; no command sequences for agents. The figure lines go under
   "## The task": `![<what it shows>](assets/preview.png)` and, for a video task, `![<what it shows>](assets/clip.gif)
   · [mp4](assets/clip.mp4)`.
6. Figures. Draft the spec with `"$PY" -m yavih.previews <NAME> --suggest --write`, then change its `rows` until the
   figure shows the task (clear examples, different classes, the pixels-and-licences rule of
   docs/design-system.md). Render when <CHROME> is not none and `yavih bench list` shows the data present (status
   `verified`, or the rows the figures draw present):
     "$PY" -m yavih.previews <NAME> --chrome <CHROME>
   (ffmpeg on PATH for a motion figure). Look at every rendered PNG and GIF yourself before you commit: the task must
   be readable at a glance. Without Chrome or the data, commit the spec and keep the figure lines in the README as
   they are: the integrator renders them (`W_README_ASSET` / `W_PREVIEW_MISSING` remain until then). Never commit
   assets/_work/ or a filled HTML page.

CHECKS, in this order, all inside your folder:
  yavih bench lint <NAME>          # no error; no W_README_LEGACY; warnings only W_PREVIEW_* / W_README_ASSET for
                                   # figures you could not render, W_CONFORMANCE_SKIPPED where a package is absent
  "$PY" -m pytest benchmarks/<NAME> -q -p no:cacheprovider -o addopts="--import-mode=importlib"
  "$PY" -m ruff check benchmarks/<NAME>
  the hash line of THE FOOTPRINT, identical to the one you recorded first
  "$PY" fleet/check.py footprint HEAD --base <BASE> --name <NAME>
Never run `python -m yavih.doctables --write` (all folders) or `python -m yavih.catalog --write`, never edit
fleet/queue.yaml, never add a changelog entry.

FINISH. Commit on readme/<NAME> in small commits (never push, never open a pull request), and end with this report,
filled in exactly:

## <NAME>: README migration report
- commits: <hashes, one line each>
- footprint: <the footprint check's last line>
- hashes: <protocol_hash input_hash scoring_hash, before = after>
- lint / folder tests / ruff: <outcome each; the warnings left>
- README: <line count>; sections <list>
- moved: metrics.notes <n notes>; make_manifest.py docstring <what was added, or unchanged>; setup.md <what moved,
  or nothing>; dropped as derivable <one line>
- folder tests: <README assertions removed or retargeted, or none>
- figures: rendered | spec only (no Chrome | no data | no ffmpeg) | none (why)
- open questions for the integrator: <or none>
```
