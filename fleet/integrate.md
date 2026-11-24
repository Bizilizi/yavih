# Integrating fleet branches

The recipe for the maintainer (or the maintainer's agent) who runs the fleet: pick a batch from the
[queue](queue.yaml), start one worker per entry ([launch](launch.md), [worker brief](worker-brief.md)), review and
merge what they report, and keep the queue current. Work in batches of 5 to 8 entries; every command runs from a
checkout of the integration branch with `PY` the fleet's Python and `yavih() { "$PY" -m yavih "$@"; }`.

## 1. Pick and claim a batch

```bash
"$PY" fleet/check.py queue                                  # the queue is valid before anything changes
grep -B1 -A12 'status: todo' fleet/queue.yaml | less        # choose 5-8 entries
```

Never claim a `kind: suite` entry for a worker: a suite is a file in `src/yavih/suites/` (a core change, outside any
worker's footprint) that the integrator writes once its components are shipped, listing them by capability
(`docs/suites.md`). `python fleet/check.py queue` refuses a suite entry that is `claimed` or `review`. Prefer a mix of
routes and data sizes, and entries whose `obtain` tier is likely `auto`. For each chosen entry set
`status: claimed` and `branch: bench/<name>` in `fleet/queue.yaml`, run the check again and commit the claim on the
integration branch. That commit is the batch's `<BASE>`: every worker starts from it.

## 2. Launch the workers

One worktree and branch per entry, created from `<BASE>`, and one worker started with the filled
[worker brief](worker-brief.md) ([launch](launch.md)). Workers never touch the queue; they end with the brief's report.

## 3. Check each reported branch

```bash
"$PY" fleet/check.py footprint bench/<name> --base <BASE> --name <name>      # must print `footprint ok`
git -C <worktree> status --short                                               # nothing left uncommitted
```

A branch that fails the footprint check is returned to its worker with the offending paths: a change outside the
folder is never merged, whatever it fixes. A change to the core that the benchmark really needs arrives as the
"core request" of the worker's report (the entry is `blocked`) and goes to the maintainer as its own core change.

## 4. The merge review

One independent reviewer per branch, a fresh agent that has not seen the worker's conversation, started with the
filled [merge review brief](review-brief.md). It runs the quick checks itself (footprint, lint, folder tests, ruff,
conformance, the synthetic trial) and checks the protocol and its grounding against the source: the paper or the
suite's implementation, the upstream code at the named commit (diffing the scoring functions), and the numbers the
README, setup.md and `metrics.notes` state (recomputed from the manifest). It blocks only on what would make a number
wrong or misleading, a protocol that differs from its source without saying so, a false claim, a guessed credit, a
failing check, or a folder that breaks for its user; everything else is a note. Two failures found after merging in
batch 1 are always blocking:

- **a capped run that cannot finish**: `--sample N` takes the first `N` manifest rows of each split, so a manifest
  sorted by task leaves a short prefix without some group, and the interval of that group's metric refuses the
  result. The rows are ordered so that a short prefix (32 rows) covers every group with a metric;
- **scores that differ between CPUs**: numpy's default `argsort` and `argpartition` order tied scores differently on
  different CPUs (its AVX-512 kernels), and a constant floor ties everywhere. Every sort and top-k on a scoring path
  is `kind="stable"`.

A branch that fails goes to a fresh fixer with the reviewer's findings appended to the worker's brief, then to the
reviewer once more. One repair round; what is still open goes to the maintainer.

The fresh-agent trial run ([trial run](../docs/trial-run.md)) does not hold a merge. The integrator runs it after the
merge, from `main`, on a small sample of the staged data, and fixes its friction in a follow-up: it finds setup
friction, rarely a wrong number, and it costs an hour per folder on a shared machine.

## 5. Merge the batch

```bash
git switch -c batch/<n> <BASE>
for b in <passing branches>; do git merge --no-ff "$b" -m "Merge $b"; done   # folders are disjoint: no conflicts
"$PY" -m yavih.doctables --write && git commit -am "doctables: the batch's folders"
"$PY" -m yavih.catalog --write && git add catalog/index.json && git commit -m "catalog: the batch's folders"
"$PY" -m pytest tests/test_no_internal_refs.py -q -p no:cacheprovider    # a folder citing the maintainer's own paper
```

A folder that cites one of the maintainer's own papers trips `tests/test_no_internal_refs.py`: add its README and
`benchmark.yaml` to that test's allow-list, as for `vggsounder`. The catalog index names the commit it was generated at, so write it after the folders are committed. On `main` the
`catalog` workflow regenerates both after a push; writing them on the batch branch keeps its tests honest. Header
images and README figures live in each folder's `assets/` and are committed: a worker with Chrome commits its own; for
every folder of the batch that `bench lint` reports with `W_HEADER_MISSING` or `W_HEADER_STALE`, render and commit them
here with `python -m yavih.headers <name> ... --chrome <chrome>`. For a folder reported with `W_PREVIEW_MISSING` or
`W_PREVIEW_STALE`: where `yavih bench list` shows its data present (`verified`, or the rows a figure draws present), render the figures here with
`python -m yavih.previews <name> --chrome <chrome>` (ffmpeg for motion) and commit them; elsewhere leave the warning and
list the folder under "figures pending data" in the batch hand-over. An `E_PREVIEW_SPEC` or a `W_ASSET_SIZE` goes back
to the worker (an invalid spec, or figures over the budget of docs/design-system.md).

## 6. Update the queue

For every entry of the batch, on `batch/<n>`:

| outcome | `status` | also set |
|---|---|---|
| merged | `shipped` | `obtain` from the report |
| blocked on data, a licence, an owner or a core change | `blocked` | `blocked_by`: the report's exact unblocking action; keep `branch` |
| abandoned (not a benchmark after all, duplicate) | `dropped` | a reason in `survey_note` |
| still being fixed | `claimed` | unchanged |

```bash
"$PY" fleet/check.py ship        # every merged folder's entry becomes shipped (only those `status:` lines change)
# then set `obtain` (and clear `blocked_by`) by hand from each worker's report: ship changes the status only
"$PY" fleet/check.py queue && git commit -am "queue: batch <n>"
```

On `main` the `catalog` workflow runs the same `ship` after every push, so a single benchmark pull request (which may
not touch `fleet/`) still leaves the queue truthful.

Collect the batch's unification candidates from `yavih bench extensions` and the workers' reports; when several folders
implement the same thing, unifying it is a core change of its own (moving a folder's scorer into the core bumps that
folder's `protocol_version`, see `docs/hashing.md`).

## 7. Test the batch, then hand it over

```bash
yavih bench lint --all
"$PY" -m pytest tests benchmarks -q -p no:cacheprovider -o addopts="--import-mode=importlib"    # once per batch
```

Then an independent reviewer checks the batch branch as a whole (the merge, the regenerated files, the queue, the
full-suite result) before it becomes the next batch's base or goes to `main` as one pull request. A pull request
labelled `benchmark` must change exactly one folder; a batch pull request is not labelled `benchmark` and runs the
whole suite.

## README migration batch

The packaged folders still carry the legacy README (`## In yavih`); `fleet/queue.yaml` marks each `readme: legacy`
and `python fleet/check.py queue` keeps the marks true (a folder whose README still has `## In yavih` must be
`legacy` or `migrating`, a rewritten one must not be). One worker per folder rewrites it with the
[README migration brief](readme-brief.md). The same batch rhythm, with a smaller gate:

1. **Claim.** Pick 5 to 8 `readme: legacy` entries, preferring folders whose data is staged here (`yavih bench list`
   shows `verified`): their workers can render the figures. Set `readme: migrating` and `branch: readme/<name>` on
   each, run `"$PY" fleet/check.py queue`, and commit the claim: that commit is `<BASE>`.
2. **Launch.** As in [launch](launch.md), with branches `readme/<name>` instead of `bench/<name>` and the filled
   [README migration brief](readme-brief.md) (`<CONFIG>` the site configuration naming the data roots, `<CHROME>`
   the browser, or `none`).
3. **Gate.** An independent reviewer (a fresh agent) checks each branch: the footprint check prints `footprint ok`;
   `yavih bench lint <name>` has no error and no `W_README_LEGACY`; the folder tests and `ruff check
   benchmarks/<name>` pass; `protocol_hash`, `input_hash` and `scoring_hash` are those of `<BASE>`; nothing an agent
   needs was lost (every number and caveat of the old "## In yavih" is now in the new README, `metrics.notes`, the
   `make_manifest.py` docstring or setup.md, or is derivable from the yaml: the worker's "moved" line is checked
   against `git show <BASE>:benchmarks/<name>/README.md`); and the README reads as a project page for a person
   ([writing a README](../docs/writing-a-readme.md)): the task first, plain words, figures that show the task.
4. **Merge.** As in step 5 above (`git merge --no-ff readme/<name>` per passing branch), then
   `"$PY" -m yavih.doctables --write` (the tables, and every glance block), and for each folder whose figures are
   not rendered and whose data is staged here, `python -m yavih.previews <name> --chrome <chrome>`; list the others
   under "figures pending data" in the hand-over. Then `"$PY" -m yavih.catalog --write` (the folders' trees moved).
5. **Queue.** Set `readme: done` (or drop the key) and `branch: null` for every merged folder, keep `migrating` for
   a branch that goes back to its worker, and run `"$PY" fleet/check.py queue` before committing.

When no `readme: legacy` entry is left, the follow-up is a core change: the legacy lint branch goes,
`W_README_LEGACY` and `W_MAKE_MANIFEST_DOC` become errors, and the `readme` queue key retires.
