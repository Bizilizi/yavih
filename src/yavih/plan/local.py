"""``yavih plan run --local``: run a plan's pending units on this machine in dependency order.

The local runner for a workstation, a single GPU box, a Kubernetes pod or an interactive SLURM allocation: no
driver loop, no ``--deps`` bookkeeping. Every unit goes through ``yavih exec``'s own checks (dependencies, DONE,
deadline headroom, records, logs), so a plan can be finished by a mix of ``plan run --local`` and SLURM arrays.

- A unit starts once every ``depends_on`` unit has a matching DONE marker (extract → probe → report).
- ``--slots K``: up to K units at a time, each a separate ``yavih exec --id`` process on the ``g``-th inherited GPU
  (``exec.device_masks``); ``K = 1`` runs the units one after another in this process.
- A unit that ends ``failed`` / ``misconfigured`` / ``not_ready`` is not retried in this run, and its dependents are
  reported as ``blocked``. A ``deadline`` / ``preempted`` result or a signal stops new starts; running units finish
  (or stop on their own deadline).

There is no SLURM mode: yavih never writes or submits sbatch (references/slurm.md has the array templates).
stdlib only, like ``exec`` for extract units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yavih.plan import exec as X

if TYPE_CHECKING:  # pragma: no cover
    from yavih.plan.units import Unit

__all__ = ["run_local", "FINISHED", "TERMINAL"]

#: Statuses after which a unit's dependents may start.
FINISHED = frozenset({"done", "skipped", "not_applicable"})
#: Statuses a local run does not retry: its dependents are ``blocked``.
TERMINAL = frozenset({"failed", "misconfigured", "not_ready", "deadline", "preempted"})


def run_local(plan_path: str, *, slots: int = 1, deadline: str | float | None = None,
              deadline_margin: float = X.DEFAULT_DEADLINE_MARGIN_S, root: str | None = None, cpus: int | None = None,
              deps: str | None = None) -> dict[str, Any]:
    """Run every unit of ``plan_path`` that is not done yet, dependencies first (module docstring). Returns
    ``{units: [last…], counts: {status: n}, blocked: [id], not_started: [id], already_done, total, status, exit_code}``: exit 0 when
    every unit is done, skipped or not_applicable, else the worst unit's code (3 when units are only blocked or left
    unstarted behind a not-ready dependency, 4 when the deadline stopped the run)."""
    from yavih.plan.units import check_done, read_plan

    if slots < 1:
        raise X.ExecError("--slots must be >= 1")
    if cpus is not None and cpus < 1:
        raise X.ExecError("--cpus must be >= 1")
    units = read_plan(plan_path, root=root)
    by_id: dict[str, Unit] = {}
    if deps:
        by_id.update((u.id, u) for u in read_plan(deps, root=root))
    by_id.update((u.id, u) for u in units)
    dl = X.resolve_deadline(deadline, margin_s=deadline_margin)

    finished: set[str] = set()
    pending: list[Unit] = []
    for u in units:
        if check_done(u) is not None:
            finished.add(u.id)
        else:
            pending.append(u)
    pending_ids = {u.id for u in pending}  # units this run will start: never re-read from disk while they wait
    already_done = len(finished)
    results: dict[str, dict] = {}
    blocked: dict[str, str] = {}
    running: set[str] = set()

    def dep_state(u: Unit) -> str:
        """``ready`` / ``wait`` / ``blocked`` for ``u``'s dependencies."""
        state = "ready"
        for d in u.depends_on:
            if d in finished:
                continue
            rec = results.get(d)
            if (rec is not None and rec.get("status") in TERMINAL) or d in blocked:
                return "blocked"
            if d in pending_ids or d in running:
                state = "wait"
                continue
            dep = by_id.get(d)
            if dep is not None and d not in results and check_done(dep) is not None:  # a --deps unit done elsewhere
                finished.add(d)
                continue
            # not in the plan, or a --deps unit that is not done: this run cannot produce it (exec would say not_ready)
            return "blocked"
        return state

    def pick() -> Unit | None:
        for u in list(pending):
            st = dep_state(u)
            if st == "blocked":
                pending.remove(u)
                pending_ids.discard(u.id)
                bad = next((d for d in u.depends_on if d not in finished and d not in pending_ids and d not in running), "?")
                blocked[u.id] = bad
                continue
            if st == "ready":
                pending.remove(u)
                pending_ids.discard(u.id)
                running.add(u.id)
                return u
        return None

    def on_done(u: Unit, rec: dict) -> None:
        running.discard(u.id)
        results[u.id] = rec
        if rec.get("status") in FINISHED:
            finished.add(u.id)

    X._PREEMPT.clear()
    X._FORWARDED.clear()
    with X._Signals():
        if slots == 1:
            while not X._PREEMPT.is_set():
                u = pick()
                if u is None:
                    break
                rec = X._run_unit(by_id, u, deadline=dl, cpus=cpus)
                on_done(u, rec)
                if X._stops_the_run(rec):
                    break
        else:
            from yavih.resources import available_cpus

            per_slot = cpus if cpus is not None else max(1, available_cpus() // slots)  # the machine shared by K slots
            X.run_pool(plan_path, pick=pick, on_done=on_done, slots=slots, deadline=dl, deps=deps, root=root, cpus=per_slot)

    lasts = [results[u.id] for u in units if u.id in results]
    counts: dict[str, int] = {}
    for r in lasts:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    not_started = [u.id for u in pending if u.id not in blocked]
    codes = [int(r.get("exit_code", 5)) for r in lasts]
    worst = max(codes, default=0)
    if worst == 0 and (blocked or not_started):
        worst = 3
    if worst == 0:
        status = "done"
    elif codes and max(codes) > 0:
        status = max(lasts, key=lambda r: int(r.get("exit_code", 5)))["status"]
    else:
        status = "not_ready"  # nothing failed, but units are blocked or were never started
    for uid, dep in blocked.items():
        X._log(f"[yavih] unit {uid} blocked: dependency {dep} did not finish in this run")
    X._log(f"[yavih] plan run --local: {len(lasts)} unit(s) ran ({', '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'none'}), "
           f"{len(finished)} done in total, {len(blocked)} blocked, {len(not_started)} not started; exit {worst}")
    return {"units": lasts, "counts": counts, "blocked": sorted(blocked), "not_started": not_started,
            "already_done": already_done,
            "status": status, "exit_code": worst, "total": len(units)}
