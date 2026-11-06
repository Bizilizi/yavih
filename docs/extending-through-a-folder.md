# Extending yavih through a benchmark folder

A benchmark contribution never edits the core. Whatever a protocol needs that the core does not ship (a scorer, a
head, a loss, a metric, an output type, a bridge-check gate, an eligibility rule, a synthetic stand-in, a reading, a
capability tag, a label kind) is written inside the benchmark folder, through the extension points below, and the
core runs it. This is the intended way to add a benchmark, not a workaround: yavih is a collection of community
benchmark implementations with shared primitives a folder may use and is never obliged to. Logic that many folders
end up sharing is moved into the core later, deliberately, as a `protocol_version` bump of each folder that adopts
it (record the candidates in the folder's `notes/UNIFY.md`).

`yavih bench info <name>` prints what a folder extends on its `extends:` line; `yavih bench extensions` lists every
folder that extends anything (`--all`: every folder).

## The extension points

| point | where | what the core does with it |
|---|---|---|
| scorer | `eval.py` `SCORER = Scorer(name="<folder>", ...)` | registers it under the folder's name; `benchmark.yaml` says `scorer: <folder>`. Any kind: `probe` (trains), `stateless`, `direct` ([adding a scorer](adding-a-scorer.md)) |
| heads and losses | `Scorer.build_head`, `Scorer.loss_fn` with `head_families` / `losses` | the core trainer builds the folder's head (`build_head_for`) and its loss (`per_sample_loss`); torch is imported when called |
| own driver | `Scorer.score(ctx)` | the scorer's driver (`bench info`: `own driver`); it may use the core trainer (`yavih.readouts._fit`) or train with its own loop |
| metrics | `eval.py` `METRICS`, `PROTOCOL_KEYS` | metrics of the folder's own (an authors' scorer, a transcription), with `per_clip` / `interval` for intervals and `floor_of` for floors ([benchmark folder](benchmark-folder.md#evalpy-hooks)) |
| hooks | `eval.py` `targets`, `score`, `join` | target construction, the authors' scorer, a cross-reading join |
| samples lines | `Scorer.sample_lines(part, seed=, protocol=)` | the samples file of `eval run --log-samples` for the scorer's parts |
| conformance | `Scorer.conformance_population(ctx, tag)` | the generated population the folder's fixture is computed on ([conformance](conformance.md)) |
| output types | `eval.py` `CODECS` | codecs of the folder's own ([codecs](codecs.md#writing-a-codec)) |
| bridge-check gates | `eval.py` `BRIDGE_CHECKS` | run by `yavih bridge check` after its hard checks (below) |
| eligibility | `eval.py` `eligible(view)` | a rule of the folder's own on the validated features, after the scorer's verdict (below) |
| stand-in | `testing.py` `make_stand_in`, `write_bridge` | `yavih bench trial --synthetic` runs the whole chain on the folder's own stand-in (below) |
| reader | `reader.py` `iter_samples` | replaces the generic sample reader ([benchmark folder](benchmark-folder.md#readerpy)) |
| readings | `eval.py` `READINGS = ("...",)` | readings beyond `a`, `v`, `av`, `t` are the folder's own (`bench lint` warns `W_READING_UNKNOWN` for an undeclared one) |
| capability | `capability: [..., "<tag>"]` | a tag beyond the core vocabulary is listed under its own heading (`W_CAPABILITY_UNKNOWN`, a note) |
| label kind | `manifest.label.kind: <kind>` | a kind beyond the core ones is kept as the cell's string; valid when the folder's scorer lists it in `label_kinds` |

Name a folder's own head families and losses after the folder (`<folder>_...`): a name resolves to one scorer, so a
second folder scorer that declares a loss or head family another already builds is refused when it registers, and
`yavih bench new --like` renames the copied names that follow this convention.

Every point keeps the folder's numbers honest the same way: hashes cover the declaration (`benchmark.yaml`), never
code; the conformance fixture pins the folder's scoring path, whatever code it runs, to its `protocol_version`.

## A scorer that trains its own head

```python
# eval.py: module scope imports only the standard library, numpy and yavih core (torch inside functions)
from yavih.readouts.registry import Scorer


def _head(num_classes=None, T=None, **_):
    return {"family": "mybench_head_v1", "out": num_classes, "layout": f"time:{T}"}


def _build_head(cfg, *, visual_tag=None, conditioning_tag=None):
    def build(dim):
        import torch
        ...  # return an nn.Module: head(x) or head(x, pos)
    return build


def _loss_fn(name):
    import torch
    ...  # return (pred, target) -> per-sample loss [B]


def _score(ctx):  # the driver: train with yavih.readouts._fit.fit (the core trainer) or a loop of your own
    ...


SCORER = Scorer(
    name="mybench", version=1, module=__name__, kind="probe", label_kinds=frozenset({"multi"}),
    train=dict(lr=1e-3, lr_grid=(), weight_decay=0.05, max_epochs=30, batch_size=64, patience=5, warmup_fraction=0.1),
    head=_head, loss=lambda label_kind: "mybench_loss", head_families=frozenset({"mybench_head_v1"}),
    losses=frozenset({"mybench_loss"}), build_head=_build_head, loss_fn=_loss_fn, score=_score,
    floor_outputs=..., conformance_population=...,
)
```

The pinned configuration (`ProbeConfig.pinned`) and its hash come from `train`, `head` and `loss` as for a core
probe; `version` is on the scoring side of `protocol_hash`. A scorer that declares `family` (for example `clf`)
shares that family's metrics, floors and conformance population and may reuse its driver.

## Bridge-check gates

`BRIDGE_CHECKS` is a tuple of library gate names (`yavih.extract.LIBRARY_GATES`: `stream`, the future-invariance and
real-time gates of a streaming bridge) and callables `gate(gc) -> dict | None`. `gc` is a `yavih.extract.GateContext`:
`bench`, `describe`, `smoke` (the smoke samples), `outputs` (pass A per smoke index), `tol`, `deterministic`,
`encode(samples)` (the bridge again, on samples the gate builds, e.g. perturbed copies through
`yavih.sample.with_files`) and `fail(code, message)` (an `E_*` code of the folder's own; exit 2). The returned dict is
recorded in the profile under `gates.<name>`. Without `BRIDGE_CHECKS` the defaults of the output contract run (`stream`
for a streaming benchmark); with it, exactly the listed gates run.

## Eligibility

`eligible(view)` receives the `yavih.readouts.eligibility.EligibilityView` that `yavih features validate` measured and
returns what is missing (empty = eligible) or `(missing, reasons)`. It runs after the scorer's own verdict; its
thresholds belong in `protocol` keys the folder declares in `PROTOCOL_KEYS`.

## A stand-in of the folder's own

`testing.py` (imported like `eval.py`; torch-free at module scope) may define:

- `make_stand_in(work, real, *, n_eval, n_train) -> folder`: a folder with the real folder's name and scorer over tiny
  generated media (`yavih.testing` has the media writers and `make_synthetic_benchmark` to start from);
- `write_bridge(path, kind, *, D, bench_folder) -> path`: the stand-in's `oracle` (features that encode the label) or
  `random` bridge, when the core test bridges cannot encode its labels.

`yavih bench trial --synthetic` then runs every stage on it. Without `testing.py`, a folder scorer that declares a core
family (`family="clf"`, `seq`, `grid`, `box`, `knn`) trials on that family's synthetic fixture: the stand-in carries the
folder's scorer name and a copy of its `eval.py` and sibling modules, so the folder's own head, loss and driver run in
every stage (an `eval.py` whose hooks or protocol keys do not fit the family's synthetic rows is `E_TRIAL_NOT_READY`:
ship `testing.py`). A folder whose scorer neither the core fixtures nor a `testing.py` cover (no family, or a family
without a fixture such as `stream`) runs the `conformance` stage and records the later stages as `not_applicable`
(exit 0); the fresh-agent [trial run](trial-run.md) on real data covers it.

## Decisions

How the extension points draw their lines, and why:

- **A label kind outside the core vocabulary is valid only when the folder's scorer lists it in `label_kinds`.** A
  `targets` hook alone does not admit one: a core scorer would still be handed cells it cannot read. A folder that
  needs its own label kind declares a scorer (of family `clf` when it reuses the clf driver) that lists the kind.
- **Without `BRIDGE_CHECKS` the output contract's default gates run** (`stream` for a streaming benchmark, none
  otherwise), so a folder that declares nothing keeps the gates its contract implies; `BRIDGE_CHECKS = ()` opts out,
  and a declared tuple is exactly what runs.
- **A folder's eligibility thresholds are `protocol` keys** listed in `PROTOCOL_KEYS`, read by `eligible(view)`:
  `output_contract.requires` keeps its fixed keys (an unknown one is a schema error), so its hashing stays as it is.
- **`yavih bench new --scorer` stays** until the benchmark catalog can find the closest folder: it now falls back to
  the first folder that uses the scorer when the scaffold has no default for it; `--like` is the way to choose.
