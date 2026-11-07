# The fleet

Maintainer tooling for adding many benchmarks in parallel, one agent per benchmark folder. Nothing here ships in the
wheel or is read by yavih itself.

| file | what |
|---|---|
| [worker-brief.md](worker-brief.md) | the prompt for one worker agent: the footprint rule (`benchmarks/<name>/` only), extending through the folder, the obtain-tier policy, the checks, the trial verdicts, the report template |
| [readme-brief.md](readme-brief.md) | the prompt for one README migration worker: rewrite a shipped folder's legacy README as the human page, move its agent material to `metrics.notes`, the `make_manifest.py` docstring and setup.md, no hash moves |
| [queue.yaml](queue.yaml) | one entry per candidate benchmark: its reserved folder name, survey class, suggested route and status |
| [check.py](check.py) | `footprint` (a branch changes one folder only), `queue` (the queue is valid against `benchmarks/`, `readme` marks included), `entry` (one entry, for a brief) |
| [integrate.md](integrate.md) | the integrator's recipe: claim a batch, check and review each branch, merge, regenerate, update the queue, test; the README migration batch |
| [launch.md](launch.md) | starting a batch of workers, each in its own worktree |

A worker renders its folder's header image and README figures when it has Chrome (and the data, and ffmpeg for
motion); otherwise it commits the figure spec and the integrator renders where the data is staged
([design system](../docs/design-system.md)).

**Reserving a name.** Every folder under `benchmarks/` has an entry here, and a pull request labelled `benchmark`
passes CI only when its folder's name has one. A name that is not yet queued is reserved by a separate pull request
that adds only its entry (every key described at the top of `queue.yaml`, `status: todo`), or by a maintainer, before the benchmark pull request.

The rules behind it: a benchmark contribution changes its folder only and extends yavih through the folder
([extending through a folder](../docs/extending-through-a-folder.md)); folders are integrated in reviewed batches; what
several folders implement alike is unified into the core later, as a core change of its own. CI enforces the
footprint of a pull request labelled `benchmark` and the validity of the queue (`.github/workflows/ci.yml`, job
`fleet`).
