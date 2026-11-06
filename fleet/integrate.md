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

## 4. The batch review gate

An independent reviewer, a fresh agent that has not seen the workers' conversations, checks every branch of the
batch against the contract (`docs/contributing-a-benchmark.md`, `docs/benchmark-folder.md`) and against the
benchmark's paper, and writes one verdict per branch. Give it the branch names, `<BASE>`, the workers' reports and
this list; it runs everything itself:

- the protocol is the paper's: splits, labels, metric, headline, floors; the route is justified;
- `python benchmarks/<name>/make_manifest.py --check` reproduces the frozen manifest from the public files, and the
  curation sums hold (`yavih bench lint <name>`);
- `yavih bench lint <name>`, the folder's tests and `ruff check benchmarks/<name>` pass on the branch;
- `yavih bench trial <name> --synthetic --readout-fast` ends `trial ok`, or the report explains each
  `not_applicable` stage; `trial UNVERIFIED` is never accepted;
- the fresh-agent trial report meets [the trial-run criteria](../docs/trial-run.md#reviewer-criteria), or the
  report says why it was not run (blocked data, no sub-agent) and the entry stays `blocked` or `review`;
- a branch whose fresh-agent trial run was not run for lack of a sub-agent is merged only after the integrator has
  run it from the branch (`docs/trial-run.md`) and it passed; one not run because the data is blocked keeps its entry
  `blocked` and is merged only when the maintainer accepts a folder that has not met real data yet;
- every budget is measured, or marked derived (how) or not measured (why); nothing is invented;
- `yavih bench info <name>` prints an `extends:` line that matches the report's unification candidates.

A branch that fails goes back with the reviewer's findings, to the same worker or to a fresh one started on the same
branch with the findings appended to its brief (a fresh agent is required again for the trial run). It stays
`claimed` until it passes. A fix changes the branch, so the reviewer runs again on it.

## 5. Merge the batch

```bash
git switch -c batch/<n> <BASE>
for b in <passing branches>; do git merge --no-ff "$b" -m "Merge $b"; done   # folders are disjoint: no conflicts
"$PY" -m yavih.doctables --write && git commit -am "doctables: the batch's folders"
"$PY" -m yavih.catalog --write && git add catalog/index.json && git commit -m "catalog: the batch's folders"
```

The catalog index names the commit it was generated at, so write it after the folders are committed. On `main` the
`catalog` workflow regenerates both after a push; writing them on the batch branch keeps its tests honest. Header
images: `python -m yavih.headers <name> ... --chrome <chrome>` renders them; whether PNGs are committed is the
maintainer's decision (the `headers` workflow also renders them into an artifact).

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
