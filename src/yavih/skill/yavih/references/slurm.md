# Execution reference — matrix.yaml, SLURM arrays, driver loop, preemption; local, Docker and Kubernetes runs

No SLURM? Skip to [Without SLURM](#without-slurm-workstation-docker-kubernetes): `yavih plan run
plan.jsonl --local` runs a plan on one machine in dependency order. yavih never writes or submits
sbatch; everything below is a template you adapt.

The unit of scheduling is one line of `plan.jsonl`; one array task runs
`exec yavih exec <pending file> --deps plan.jsonl --index $SLURM_ARRAY_TASK_ID --deadline slurm`.
`exec` checks dependencies (exit 3 if a `depends_on` DONE is missing), skips a unit whose DONE
already matches (exit 0 `skipped`), refuses to start with less than `max(10 min, 0.25 ·
needs.minutes)` left before the deadline (exit 4), writes a `running` record, runs the unit,
writes `<out>/logs/last.<id>.json` (and, for every unit, `<out>/logs/<id>.<job>.<utc>.log`: the
child's output for extract, the traceback of a failed probe/report), and prints `[yavih] unit
<id> <kind> <status> wall_s=… exit=N`. `--deadline slurm` = `SLURM_JOB_END_TIME − 900 s`
(`--deadline-margin`, deliberately larger than the `--signal=B:TERM@600` lead below); unset
inside a job (older Slurm releases do not export it) → `E_DEADLINE_SPEC`, exit 1: pass
`--deadline +<seconds>` (the job's `--time` minus a margin) instead; outside SLURM → none. Every
other form works everywhere: `none`, `+3600`, `+90m`, `2h`, an epoch, an ISO time.

Two rules every script below follows:

- **`--deps plan.jsonl`.** A `--pending --kind probe` file holds only probe units; their extract
  dependencies are in the full plan. Without `--deps` every probe exits 3 `dependency … is not in
  the plan`. `--index`/`--id` still select from the pending file.
- **`exec yavih exec …`** (the shell's `exec`): `--signal=B:TERM@600` signals only the batch
  shell, and a non-interactive bash neither forwards TERM to a foreground child nor runs a trap
  until that child exits. Replacing the shell makes `yavih exec` the signalled process. A script
  that must clean up afterwards (Template C) runs it in the background and forwards TERM itself.

Partition, account, QoS, paths and container names below are placeholders (`<partition>`,
`<account>`, `<qos>`, `$PROJECT`, `/path/to/…`); **replace them with your site's**.

## matrix.yaml

`yavih plan init --suite <s> --model <name> --launcher <python>` writes one entry per suite
`(benchmark, reading)` and the bridge stubs; by hand, one entry per bridge × benchmark:

```yaml
schema_version: 1
out: runs/
seeds: [0, 1, 2]
shard_minutes: 30            # target wall per extract unit; many small shards, never one unit per GPU
canaries: 8
defaults:
  launcher: "/envs/model/bin/python"       # python → replaces argv[0]; otherwise a prefix (apptainer …)
  env: {HF_HUB_OFFLINE: "1", OMP_NUM_THREADS: "8"}
  cpus: 8                                  # decode workers = cpus - 1 = your bridge's decode budget
bridges:
  model-k400:
    script: bridges/model_k400.py
    benchmark: k400
    args: {layer: 12}
    profile: bridges/model-k400.profile.json   # from step 5 with the SAME --arg values
```

`plan build` exits 1 if `profile.bench != benchmark`, `profile.args` differ from the entry's `args`,
`profile.describe` fails the schema, or the profile records no passed hard checks. A profile that
already misses the output contract yields a plan-time `not_applicable` probe unit, no extracts.
`yavih plan build matrix.yaml --sample N` plans a capped run first (the first `N` rows per split, under
`<out>-sample<N>`, results never stored, report rows marked partial).

## Resources per unit → sbatch

`needs = {gpu, cpus, mem_gb, scratch_gb, minutes}` in every unit. Build one pending file per
`--kind` (units of one kind have the same shape), take the maxima over it, and request:
`--gres=gpu:<needs.gpu>`, `--cpus-per-task=<matrix cpus>` (= `needs.cpus`; decode workers =
cpus − 1 = the bridge's own decoding budget and the byte read-ahead pool, and `bridge check`
measured `clips_per_s`, decoding included, at exactly that count), `--mem=<max
mem_gb>G`, `--time=<max minutes + margin>`. Any A/B of extraction speed must use equal
`--cpus-per-task` on both arms, cold (disjoint) rows, and a compute node (never a login node).

```bash
# helper: maxima over a pending file
needs() { python - "$1" <<'EOF'
import json,sys
u=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n=lambda k: max((x["needs"][k] for x in u), default=0)
print(f"COUNT={len(u)} GPU={n('gpu')} CPUS={n('cpus')} MEM={n('mem_gb')} SCRATCH={n('scratch_gb')} MIN={n('minutes')}")
EOF
}
```

## Driver loop (`until plan --pending == 0; sbatch --wait`)

```bash
set -euo pipefail
cd "$PROJECT"                                     # the directory matrix.yaml paths are relative to
yavih plan build matrix.yaml --out plan.jsonl     # full plan; plan status reads this one
for KIND in extract probe report; do
  while :; do
    N=$(yavih plan build matrix.yaml --pending --kind "$KIND" --out "pending.$KIND.jsonl" --format count)
    [ "$N" -eq 0 ] && break
    eval "$(needs "pending.$KIND.jsonl")"           # COUNT GPU CPUS MEM SCRATCH MIN
    sbatch --wait --parsable \
      --array="0-$((COUNT-1))%${THROTTLE:-8}" --gres="gpu:${GPU}" --cpus-per-task="$CPUS" \
      --mem="${MEM}G" --time="$((MIN+20))" \
      "$KIND.sbatch" "pending.$KIND.jsonl" plan.jsonl || true   # 2nd arg = --deps; a non-zero array exit is expected: preempted / not_ready units
    yavih plan status plan.jsonl                    # summary: counts per kind, then misconfigured / failed / repeated / not_ready / preempted units with their last message
    # stop on real failures: exit 1 (misconfigured) / 2 / 5 units never heal by resubmission, nor do units that
    # exited 3 or 4 the same way 3+ times in a row (`repeated`: data missing, a deadline too short for one shard)
    yavih plan status plan.jsonl --json | python -c 'import json,sys; d=json.load(sys.stdin)["data"]; sys.exit(1 if d["failed"] or d["misconfigured"] or d["repeated"] else 0)' \
      || { echo "failed / misconfigured / repeated units: see plan status above and their logs"; exit 2; }
  done
done
yavih report build 'runs/*'
```

`--pending` re-plans only units whose DONE is missing or stale, so a loop iteration resubmits
exactly the `preempted` (exit 4) and `not_ready` (exit 3) units, and units killed without any
final record (`plan status` lists them under `stale running`). Extract units are keyed by the
benchmark's input hash and probe units by its protocol hash: after a scoring-side edit of a
benchmark (`probe.py`, a metric or its pinned package, a readout version) `--pending` re-plans the
probes and no extraction. A `plan.jsonl` written by an older yavih (older plan schema) is refused:
rebuild it with `plan build` (finished shards are kept). `--time` is `needs.minutes` plus
20 min: the 15 min `--deadline-margin` plus start-up. `yavih plan calibrate runs/`
between iterations raises `needs.minutes` ×1.5 for units their own deadline stopped while running
(not for start refusals or preemptions); keep `--time` inside the partition limit. A shard longer
than the partition's maximum wall never finishes (extraction resumes per shard, not inside one):
its split's layout is frozen (below), so move that split directory aside and re-plan with a smaller
`shard_minutes` (or a matrix `shards` pin). To guard `--index` against a pending file rebuilt
while the array is queued, pass `--plan-sha` with the file's sha256 (`yavih plan build … --format
count --json` reports it as `data.sha256`). Instead of
`--wait`, chain kinds with `--dependency=afterany:<jobid>` (`afterany`, not `afterok`: a
preempted array task exits 4 and must not block the probe array's DONE checks, which `exec`
performs itself).

## Preemption and deadlines

Add `--signal=B:TERM@600 --requeue` to every array, and start `yavih exec` with the shell's
`exec` (or forward TERM yourself, Template C). Three things stop a unit, in this order:

1. **exec's own deadline** (`--deadline slurm` = end − 900 s): a probe stops between epochs with
   `probe_ckpt.pt` written, an extract child finishes its batch and drops the shard's `.tmp`; exit
   4, `status: deadline`. A unit that would start with too little time left is refused (exit 4).
2. **SLURM's `B:TERM` 600 s before the wall** (and on preemption) reaches `yavih exec` only
   because the shell was replaced; `exec` forwards it to the extract child, or sets the in-process
   probe's stop flag; exit 4, `status: preempted`.
3. **SIGKILL at the wall** leaves only the `running` record; `plan status` reports the unit as
   `pending` and lists it under `stale running`.

`plan status` counts 1 and 2 as `preempted` (resubmittable, never `failed`). `--requeue` only
requeues on **preemption or node failure**; a task stopped by its time limit is not requeued — the
driver loop's next `--pending` iteration resubmits it, and `plan calibrate` raises its
`needs.minutes` ×1.5. Either way the next run resumes (shard skipped if complete, probe resumed
from the checkpoint). `exec` also maps a child killed by a signal (negative rc / 128 + n) to exit
4. Use `exec` (never raw `python -m yavih.extract`) so `YAVIH_DEADLINE` reaches the child.

## Template A — single-node GPU array

Facts to look up for your site: the partition (and QoS / account, if required), the wall-time
limit, per-user caps on running and submitted jobs (array tasks usually count individually: keep
`COUNT` under the submit cap per array and loop again), and how to request a GPU type (an untyped
`--gres=gpu:1` may land on a small or old card: pin a type with the exact `sinfo` gres spelling or a
`--constraint`). If `/tmp` is a tmpfs it is charged to `--mem`. Make `logs/` before submitting.

```bash
#!/bin/bash
#SBATCH --job-name=yavih-extract
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1                          # pin a GPU type with your site's gres name if needed (gpu:<type>:1)
#SBATCH --cpus-per-task=8                     # = matrix defaults.cpus (decode workers 7)
#SBATCH --mem=48G                             # ≥ max needs.mem_gb of the pending file
#SBATCH --time=00:50:00                       # ≥ max needs.minutes + 20 min (shard_minutes 30; exec stops units 15 min before the wall)
#SBATCH --partition=<partition>
##SBATCH --qos=<qos>                          # only if your site requires one
##SBATCH --account=<account>                  # only if your site requires one
#SBATCH --array=0-31%8                        # %K throttle: K concurrent tasks (stay within your per-user GPU cap)
#SBATCH --signal=B:TERM@600 --requeue
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
cd "$PROJECT"                                   # where matrix.yaml lives
export YAVIH_SCRATCH="${SLURM_TMPDIR:-/tmp}"  # probe staging (on a tmpfs: size scratch_gb into --mem)
PLAN="${1:-pending.extract.jsonl}"
DEPS="${2:-plan.jsonl}"                         # the full plan: dependency lookup for a pending file
# `exec` replaces the shell, so --signal=B:TERM reaches yavih (a bash parent would die without forwarding it)
exec /path/to/probe-env/bin/yavih exec "$PLAN" --deps "$DEPS" --index "$SLURM_ARRAY_TASK_ID" --deadline slurm
```

The model env's python is the matrix `launcher` (`/path/to/model-env/bin/python`); `yavih` itself may
run from any env that has it — `exec` only needs stdlib for extract units, the probe/report
units need `yavih[probe]` in the env that runs `yavih exec` (so run those arrays from the probe env).

## Template B — offline compute nodes, one GPU per job

Many sites have **no internet on compute nodes**: pre-stage model weights and set
`HF_HUB_OFFLINE=1` (and `HF_HOME` to a pre-filled cache). Some allow only single-GPU jobs — yavih
never needs multi-GPU jobs: scale out with more array tasks. Match the torch build to the node's
CUDA driver. Without a system FFmpeg, PyAV wheels bundle their own libav, but a bridge that decodes
with torchcodec/torchaudio needs FFmpeg shared libraries on `LD_LIBRARY_PATH`. Launch the venv's
python directly, never `uv run` (it re-resolves and can modify a shared venv). Use the node-local
`$TMPDIR` for scratch. Long queues favour bigger shards (`shard_minutes 60`).

```bash
#!/bin/bash
#SBATCH --job-name=yavih-extract
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>                           # if your site requires one
#SBATCH --account=<account>                   # if your site requires one
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16                    # = matrix defaults.cpus (decode workers 15)
#SBATCH --mem=128G
#SBATCH --time=02:00:00                       # inside your partition / QoS limit
#SBATCH --array=0-39%10                       # throttle to your per-user job cap
#SBATCH --signal=B:TERM@600 --requeue
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
cd "$PROJECT"                                   # where matrix.yaml lives
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/path/to/hf-cache
export LD_LIBRARY_PATH="/path/to/ffmpeg/lib:${LD_LIBRARY_PATH:-}"   # only if the bridge decodes with torchcodec/torchaudio
export YAVIH_SCRATCH="$TMPDIR"
export YAVIH_DATA__K400__MEDIA_ROOT=/path/to/k400_wds   # data roots (or config.toml on this host)
PLAN="${1:-pending.extract.jsonl}"
DEPS="${2:-plan.jsonl}"
exec /path/to/yavih-env/bin/yavih exec "$PLAN" --deps "$DEPS" --index "$SLURM_ARRAY_TASK_ID" --deadline slurm   # exec: TERM reaches yavih
```

`matrix.yaml`: `launcher: "/path/to/model-env/bin/python"`, `cpus: 16`,
`env: {HF_HUB_OFFLINE: "1", OMP_NUM_THREADS: "16"}`.

## Template C — whole node with 4 GPUs, Apptainer, RAM scratch

For sites that allocate whole nodes (here 4 GPUs) and run models in containers, often with
diskless offline nodes: datasets, weights and the `.sif` live on a shared filesystem,
`HF_HUB_OFFLINE=1`, and node-local scratch is RAM (`/dev/shm`, `/tmp`), charged to the job and to be
deleted on exit. On such sites `--gpus=4` usually allocates every CPU of the node whatever
`--cpus-per-task` says.
Pack 4 units per node with `--slots 4`: `exec` runs units `4·index … 4·index+3` concurrently, each
slot a separate `yavih exec --id` process on the `g`-th GPU of the inherited `CUDA_VISIBLE_DEVICES`
(`0…3` when nothing is inherited) with `OMP_NUM_THREADS=1`; probe slots are processes too, so
each probe has its own GPU and torch state. `exec --slots 4` gives each slot
`needs.cpus // 4` CPUs (= matrix `cpus` // 4) and rewrites the child's
`--decode-workers` to that minus 1 — it does NOT multiply. So set matrix `cpus: 64` for whole-node
entries (one unit = the whole node when run without slots), keep `--cpus-per-task=64` (= matrix
`cpus`), and measure the profile with `yavih bridge check … --decode-workers 15` so `clips_per_s`
matches the 15 workers each slot really gets (with `cpus: 16` every slot would get 4 CPUs / 3
workers, 48 cores would idle and the array would under-plan ~5×). `needs.mem_gb` then reads
`16 + 4·64 = 272` GB per unit — a whole-node estimate, not per slot: size `--mem` to node RAM minus
the `/dev/shm` scratch you use. The array has `ceil(COUNT / 4)` tasks.

```bash
#!/bin/bash
#SBATCH --job-name=yavih-extract
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus=4                              # the whole node
#SBATCH --cpus-per-task=64                    # = matrix cpus 64; exec gives each of 4 slots 16 → 15 decode workers
#SBATCH --mem=400G                            # ≥ needs.mem_gb (272, whole-node estimate) + /dev/shm scratch; ≤ node RAM
#SBATCH --time=12:00:00                       # your partition's limit; exec --deadline slurm stops units 15 min before
#SBATCH --array=0-7                           # ceil(COUNT / 4) tasks
#SBATCH --signal=B:TERM@600 --requeue
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
cd "$PROJECT"                                   # where matrix.yaml lives
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/path/to/hf-cache
export YAVIH_SCRATCH="/dev/shm/yavih-${SLURM_JOB_ID}"; mkdir -p "$YAVIH_SCRATCH"
trap 'rm -rf "$YAVIH_SCRATCH"' EXIT             # RAM scratch must not outlive the job
export YAVIH_DATA__ROOT=/path/to/datasets       # or [data] root in config.toml on this host
PLAN="${1:-pending.extract.jsonl}"
DEPS="${2:-plan.jsonl}"
# `yavih exec` runs on the HOST (a small env with yavih; extract units need only stdlib there): the matrix launcher
# (`apptainer exec … model.sif python`) wraps each extract child in the container. Running `yavih exec` inside
# model.sif as well would nest Apptainer in Apptainer. Probe/report arrays run in-process and need yavih[probe]: run
# them from a probe env on the host, or as `apptainer exec --nv probe.sif yavih exec …` (probe units have no launcher).
# The cleanup trap above needs the shell alive, so no `exec`: run in the background and forward TERM.
# (bash runs a trap only between commands; a foreground child would block it until the child exits.)
/path/to/yavih-env/bin/yavih exec "$PLAN" --deps "$DEPS" --index "$SLURM_ARRAY_TASK_ID" --slots 4 --deadline slurm &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null' TERM
rc=0; wait "$pid" || rc=$?                      # returns early (rc > 128) when the TERM trap fires ...
while kill -0 "$pid" 2>/dev/null; do rc=0; wait "$pid" || rc=$?; done   # ... so wait for exec's own exit code
exit "$rc"
```

`matrix.yaml`: `launcher: "apptainer exec --nv --bind /path/to/project /path/to/model.sif python"`
(the last token is `python`, so it replaces `cmd[0]`; this is the only Apptainer layer — `yavih
exec` itself runs on the host), `cpus: 64` (the whole node; `--slots 4` splits it into 4 × 16),
`shard_minutes: 30`, profiles measured with `--decode-workers 15`. Probe units: a full-token
VGGSounder train set does not fit a typical `/dev/shm` (~284 GB fp16) — point `YAVIH_SCRATCH` at a
shared scratch filesystem for those or keep the probe array single-slot. Walls longer than 12 h are
never needed: units are ≤ `shard_minutes`, and a probe resumes from `probe_ckpt.pt` across requeues.

## Shard layout and shared run directories

The first extract unit of a split freezes its layout in `features/<split>/_layout.json` (`n`, the
batch alignment, every shard's row range). `plan build` then plans exactly that layout, whatever
the profile's `clips_per_s` (another GPU type) or `--shard-minutes` (another partition) say, so two
clusters sharing one run directory, or a re-run `bridge check`, never move finished shards to
`_stale/`. A unit from a plan built before the freeze with another `n` exits 1 `E_LAYOUT`: re-run
`plan build`. Only another configuration (model, reading, `describe_sha256`, the benchmark's input
hash) owns another layout; its shards and layout are moved aside together. Each shard is written by one task at a time:
a second task on the same shard (a requeue while the first still runs, a pod restart) exits 3
`E_SHARD_CLAIMED`; a claim whose heartbeat stopped for 15 min (`YAVIH_CLAIM_LEASE_S`) is taken over.

## Without SLURM: workstation, Docker, Kubernetes

The plan, `exec` and the records are the same everywhere; only who starts the units differs.

```bash
yavih plan build matrix.yaml --out plan.jsonl
yavih plan run plan.jsonl --local                      # every pending unit, dependencies first, one at a time
yavih plan run plan.jsonl --local --slots 2            # two at a time, unit g on the g-th visible GPU
yavih plan run plan.jsonl --local --deadline +8h       # stop starting units 8 h from now (a booked machine)
yavih plan status plan.jsonl                            # what ran, what is blocked and why
```

A failed, misconfigured or not-ready unit is not retried within one `plan run`, and its
dependents are reported `blocked`; fix it and run `plan run` again (finished units are skipped).
Lower-level pieces, for your own runner:

- `yavih exec plan.jsonl --range 0:40` runs units 0 … 39 in order in one process (stops at the
  deadline); add `--slots 4` for four at a time. `yavih exec plan.jsonl --index 7` runs one.
- GNU parallel with one GPU per job slot (`{%}` = slot number 1…J):
  `parallel -j2 'CUDA_VISIBLE_DEVICES=$(({%}-1)) yavih exec pending.extract.jsonl --deps plan.jsonl --index {}' ::: $(seq 0 $((N-1)))`.
- Docker / Podman: mount the project and keep `plan.jsonl` inside it — the plan stores paths
  relative to the plan file, so `docker run -v $PWD:/workspace … yavih exec /workspace/plan.jsonl
  --index 0` works although it was built on the host. A plan kept elsewhere takes `--root
  /workspace`. CPU quotas (`--cpus 4`) are read from the cgroup; `exec --cpus N` overrides
  (decode workers `N − 1`). With `--gpus device=4,5`, `--slots 2` uses devices 4 and 5.
- Kubernetes: an Indexed Job with `completions = <units>` runs `yavih exec /workspace/plan.jsonl
  --index $JOB_COMPLETION_INDEX --deadline +<activeDeadlineSeconds − 900>`; a restarted pod resumes
  (a finished shard is skipped, a probe resumes from its checkpoint). GPU UUIDs in
  `CUDA_VISIBLE_DEVICES` work with `--slots`.
- CPU-only machines: a profile measured without a GPU (`peak_gpu_gb: 0`) plans `needs.gpu: 0`;
  `yavih probe run --device cpu` forces the probe onto the CPU.
- A bridge that calls a local inference server (Triton, vLLM) needs `allow_loopback: true` in its
  matrix entry (`--allow-loopback`): loopback only, recorded in every sidecar.

## Checklist before `sbatch`

- [ ] `yavih plan build matrix.yaml --pending --kind <k> --format count` > 0 and `plan status` shows no `failed`, `misconfigured` or `repeated`
- [ ] `--cpus-per-task` == matrix `cpus` (for whole-node entries matrix cpus is the node's 64; exec splits it by `--slots`); `--mem` ≥ max `needs.mem_gb` (+ scratch on tmpfs sites)
- [ ] `--time` ≥ max `needs.minutes` + 20 min, inside the partition limit; `--signal=B:TERM@600 --requeue`
- [ ] the script starts `exec yavih exec …` (or forwards TERM to a background `yavih exec` on the host, Template C — never `yavih exec` inside the launcher's container) and passes `--deps plan.jsonl`
- [ ] `--deadline slurm` only where the cluster exports `SLURM_JOB_END_TIME` (else `--deadline +<seconds>`)
- [ ] site-required flags present (partition, GPU type, QoS, account); `--slots` matches the GPUs per task
- [ ] offline env vars and data roots exported (or in that host's `config.toml`); `yavih doctor --bench <b>` passed ON A COMPUTE NODE of that cluster
- [ ] `logs/` exists; `sbatch --test-only <script>` passes
