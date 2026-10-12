# Adding a readout

A readout is a task type: how features become predictions and how those predictions are scored (`clf`,
`seq`, `grid`, `box` ship with yavih; see [readouts](readouts.md)). Readouts form a registry, so a new
one touches no core module: benchmark validation, `protocol_hash`, the pinned probe configuration, floors,
validation's dense-readout checks, `plan` and the metric table all ask the registry instead of a
hard-coded list.

Add a readout only when no existing one can score the task. A new benchmark that is a classification,
per-second labelling, dense map or box task should use the existing readout.

## The two parts

A readout is one module that exports:

1. `READOUT = yavih.readouts.registry.Readout(...)`, a frozen descriptor of everything the core needs;
2. a driver `run_<name>(ctx: ReadoutContext) -> dict` that trains and scores one seed.

**The module must import without torch.** The descriptor is read in the model environment, where
`protocol_hash` is computed and torch may be missing or must not be touched. Import torch code lazily
inside the driver (the built-in readouts use `yavih.readouts.base.lazy_module`).

## The descriptor

```python
from yavih.readouts.registry import Readout

VERSION = 1   # enters protocol_hash (scoring side); bump on any change to head, loss, floor, metric or protocol

READOUT = Readout(
    name="toy", version=VERSION, module=__name__,
    label_kinds=frozenset({"single"}),        # manifest.label.kind values it scores
    protocol_keys=frozenset(),                 # per-benchmark protocol keys its driver consumes
    param_keys=frozenset(),                    # the exact readout_params key set
    requires_t_bins=False, requires_grid=False,  # output_contract.requires: True required, False forbidden, None optional
    dense=False,                               # features need media positions (pos)?
    train=dict(lr=1e-3, lr_grid=(), weight_decay=0.0, max_epochs=10, batch_size=64, patience=3, warmup_fraction=0.0),
    head=lambda **kw: {"family": "toy_head_v1", "out": kw.get("num_classes")},
    loss=lambda label_kind: "toy_loss",
    head_families=frozenset({"toy_head_v1"}), losses=frozenset({"toy_loss"}),
    geometry=lambda cfg: {"num_classes": cfg.head.get("out")},
    floor_outputs=lambda train_targets, n_eval, **kw: ...,   # label-only constant output, scored by the floor metrics
)
```

The fields, grouped by who reads them:

| group | fields | read by |
|---|---|---|
| schema | `label_kinds`, `protocol_keys`, `param_keys`, `targets` (labels from `benchmark.yaml` `targets`), `full_frame` (`target_frame` must be `full_frame`), `requires_t_bins`, `requires_grid`, `validate_spec(readout_params, requires, protocol)` | `Benchmark.load` |
| identity | `version` (scoring side), `input_keys` (extra `readout_params` / `protocol` keys that `check_outputs` reads; they enter `input_hash`) | hashing |
| probe | `train` (the pinned training block), `head(...)`, `loss(label_kind)`, `task_sizes(bench)`, `geometry(cfg)`, `head_families`, `losses` | `ProbeConfig.pinned`, `probe_hash` |
| scoring | `floor_outputs(...)`, `dense`, `check_outputs(view, requires)` (an optional extra output-contract check run by validation), `synthetic` (a tiny fixture builder for `yavih.testing`) | probe, validate, plan, tests |

## The driver

`yavih probe run` builds a `ReadoutContext` per seed (benchmark, pinned config, roles, protocol, metric
specs, feature readers, deadline, device) and calls `run_<name>(ctx)`. The driver:

- reads features only through the context (`ctx.reader(split)`, present rows, clip ids) and targets from
  the manifest or `ctx.bench.targets(split)`;
- honours the roles: trains on `train` only (validation is carved from it), scores `eval`, scores `floor`
  splits with the same trained probe, loops over `folds` when declared;
- calls the folder's `probe.py` hooks when present (`ctx.hook_targets`, `ctx.scorer`) so a benchmark can
  supply targets or the authors' scorer;
- scores through `yavih.readouts.base.evaluate_metrics`, which evaluates every name in `metrics.list`,
  `floors` and `controls` with intervals;
- returns the metric, floor, control and interval blocks plus `probe`, `fit`, `timing`, `eval_outputs`,
  `eval_rows`, `extra_arrays`, `backend` and, with `--log-samples`, `samples`. `run_clf` in
  `src/yavih/readouts/clf.py` is the reference.

Checkpointing, deadlines, `result.json`, DONE markers and the results store are handled by
`yavih.readouts.base`, not by the driver.

## Metrics

Metrics live in `yavih.metrics.registry` as `MetricSpec`s: a lazy function reference
(`"yavih.metrics.<module>:<function>"`), the readouts it serves, the protocol keys it consumes, its kind
(metric, floor, control), `floor_of`, `higher_is_better` and admissible intervals. A metric of another
readout is refused for yours; `n_eval` serves every readout. Register new ones with
`yavih.metrics.registry.register(MetricSpec(...))`. Benchmark-specific metrics belong in the folder's
`probe.py` `METRICS` instead.

## Registering it

- **Built in**: add the module to `BUILTIN` in `yavih/readouts/registry.py` and a default `--like` folder
  in `yavih/scaffold.py`.
- **From a package**: declare an entry point; no yavih change is needed.

  ```toml
  [project.entry-points."yavih.readouts"]
  toy = "mypkg.toy_readout"
  ```

- **At run time** (tests, an in-process plugin): `registry.register(READOUT)`; `registry.reset()` forgets
  run-time registrations.

A broken plugin is reported by name and does not break the core. `yavih.versions.READOUT_VERSION` reads
every registered readout's version.

## Testing it

`tests/test_extensibility.py` shows the minimum: a toy readout registered through an entry point validates a
benchmark that uses it, hashes, pins its probe config and builds its floor. A new built-in readout also
needs:

- a synthetic fixture (`synthetic=`) so `yavih.testing.make_synthetic_benchmark(tmp, "<name>")` works;
- an end-to-end test where `oracle_bridge` clears the floor and `random_bridge` does not;
- metric tests against an independent implementation where one exists;
- a [readouts](readouts.md) section and, if it introduces validation codes, rows in [validation](validation.md).
