# Launching fleet workers

How to start the workers of one batch ([integrate](integrate.md), step 2). Each worker gets its own worktree, its own
branch, its own scratch directory and yavih home, and the [worker brief](worker-brief.md) with its placeholders filled.

```bash
BASE=$(git rev-parse HEAD)                       # the batch's claim commit
for n in <names>; do
  git worktree add -b "bench/$n" "<worktrees>/$n" "$BASE"
  mkdir -p "<scratch>/$n"
  "$PY" fleet/check.py entry "$n" > "<scratch>/$n/entry.yaml"
done
```

Fill the brief's prompt (the fenced block of `worker-brief.md`) once per name: `<NAME>`, `<ENTRY>` (the contents of
`entry.yaml`), `<BASE>`, `<WORKTREE>`, `<PYTHON>`, `<DATA_ROOT>`, `<DISK_CAP_GB>`, `<SCRATCH>`.

## With a coding agent's sub-agents

Start every worker of the batch in one message, each as a background sub-agent whose prompt is its filled brief and
whose working directory is its worktree. Give each one a disk cap that, summed over the batch, fits the data volume,
and keep the batch at 5 to 8 workers: their downloads, decodes and test runs share the machine. The brief's last step
starts a further sub-agent (the fresh-agent trial run); where a worker cannot start one, it reports `not run` and the
integrator runs the trial run from the branch afterwards ([trial run](../docs/trial-run.md)).

## With a workflow script

The same batch as a workflow: one `agent()` call per name with the filled brief as the prompt, run in parallel, and a
reviewer `agent()` per finished branch for the batch review gate ([integrate](integrate.md), step 4) fed the worker's
report and the branch name. The footprint check and the merge stay with the integrator.

## What not to share

The workers share the Python environment and the data root, nothing else: never two workers on one worktree or one
`YAVIH_HOME` (the conformance cache and fetched folders live there), and never a worker running in the integrator's
checkout.
