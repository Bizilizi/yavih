# Adding a scorer

A scorer is a task type: how a benchmark's stored outputs become numbers. The four shipped scorers are the
pinned probes `clf`, `seq`, `grid` and `box` ([readouts](readouts.md)): they train a fixed head on extracted
features. A scorer need not train anything: a `stateless` scorer reads stored outputs as they are (nearest
neighbours, retrieval), a `direct` scorer scores stored predictions (a folder's own metrics or the benchmark
authors' package). Scorers form a registry, so a new one touches no core module: benchmark validation,
`protocol_hash`, the configuration hash that fills the second slot of `eval_hash`, floors, eligibility, the
conformance population, `plan` and the metric table all ask the registry instead of a hard-coded list.

Use an existing scorer when one scores the task. A benchmark that is a classification, per-second labelling,
dense map or box task uses the shipped probe. A benchmark whose protocol is its own (an authors' scorer, a set-level
distance, a generation metric) owns its scorer in the folder's `eval.py`, under the folder's name.

## The two parts

A scorer is one module that exports:

1. `SCORER = yavih.readouts.registry.Scorer(...)`, a frozen descriptor of everything the core needs;
2. a driver: `score(ctx) -> dict` on the descriptor, or a module function `run_<name>(ctx)` for the shared kinds.

**The module must import without torch.** The descriptor is read in the model environment, where
`protocol_hash` is computed and torch may be missing or must not be touched. Import torch code lazily
inside the driver (the built-in scorers use `yavih.readouts.base.lazy_module`).

## Three kinds

| kind | trains | configuration (second slot of `eval_hash`) | driver |
|---|---|---|---|
| `probe` | the shared pinned head (`train`, `head`, `loss`, `head_families` required) | `probe_hash` of the pinned `ProbeConfig` | `run_<name>(ctx)` |
| `stateless` | nothing: scores stored arrays as they are | the constant `scorer_config_hash(name, version)` (`ScorerConfig`) | `score(ctx)` |
| `direct` | nothing: scores stored predictions, through the folder's metrics or an authors' package | the same constant | `score(ctx)` |

For a non-probe scorer the configuration is a constant, so `eval_hash` moves only with `protocol_hash`: the
scorer's `version` is on the scoring side of it. `standard` is always true for such a scorer unless a testing
configuration overrides it (`YAVIH_TESTING=1`).

## The descriptor

```python
from yavih.readouts.registry import Scorer

VERSION = 1   # enters protocol_hash (scoring side); bump on any change to head, loss, floor, metric or protocol

SCORER = Scorer(
    name="toy", version=VERSION, module=__name__, kind="probe",
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
| schema | `label_kinds`, `protocol_keys`, `param_keys`, `targets` (labels from `benchmark.yaml` `targets`), `full_frame` (`target_frame` must be `full_frame`), `requires_t_bins`, `requires_grid`, `validate_spec(readout_params, requires, protocol)`, `output_keys` (the `output_contract.schema` keys a feature set must carry) | `Benchmark.load`, validate |
| identity | `version` (scoring side), `input_keys` (extra `readout_params` / `protocol` keys that eligibility reads; they enter `input_hash`) | hashing |
| kind | `kind`, `family` (the scorer whose metrics, targets and conformance population this one shares, e.g. a kNN scorer of family `clf`), `needs_gpu` | load, `plan` |
| probe | `train` (the pinned training block), `head(...)`, `loss(label_kind)`, `task_sizes(bench)`, `geometry(cfg)`, `head_families`, `losses` | `ProbeConfig.pinned`, `probe_hash` |
| scoring | `score(ctx)`, `floor_outputs(...)` (`None` = the scorer declares no floors and the folder's `metrics.floors` must be `[]`), `dense`, `eligibility(view)`, `config_hash(cfg)`, `pinned(bench)`, `conformance_population(ctx, tag)` | eval, validate, plan, conformance |

Metrics are allowed for a scorer when their `readouts` name it or its `family` (`Scorer.families()`).

## The driver

`yavih eval run` builds a `ReadoutContext` per seed (benchmark, pinned configuration, roles, protocol, metric
specs, feature readers, deadline, device) and calls the driver. The driver:

- reads features only through the context (`ctx.reader(split)`, present rows, clip ids) and targets from
  the manifest or `ctx.bench.targets(split)`;
- honours the roles: trains on `train` only (validation is carved from it), scores `eval`, scores `floor`
  splits with the same trained probe, loops over `folds` when declared;
- calls the folder's `eval.py` hooks when present (`ctx.hook_targets`, `ctx.scorer`) so a benchmark can
  supply targets or the authors' scorer;
- scores through `yavih.readouts.base.evaluate_metrics`, which evaluates every name in `metrics.list`,
  `floors` and `controls` with intervals;
- returns the metric, floor, control and interval blocks plus `probe`, `fit`, `timing`, `eval_outputs`,
  `eval_rows`, `extra_arrays`, `backend` and, with `--log-samples`, `samples`. `run_clf` in
  `src/yavih/readouts/clf.py` is the reference. A non-probe driver returns `probe: {}` (the fit fields are
  recorded as null) and no `s_per_epoch`.

Checkpointing, deadlines, `result.json`, DONE markers and the results store are handled by
`yavih.readouts.base`, not by the driver.

## Eligibility

`yavih features validate` measures the facts of a feature set once over its seeded sample (position axes,
stream tags, grid-cell coverage, scored seconds, the distance to an annotated frame) and asks the scorer:
`eligibility(view)` with a `yavih.readouts.eligibility.EligibilityView`, returning what is missing (empty =
eligible) or `(missing, reasons)`. A scorer without the hook gets the default verdict, the
`output_contract.requires` rules (`requires_missing`). The scorer's `output_keys` must be present in the set
whatever the hook says ([validation](validation.md#eligibility)).

## Conformance

Every folder carries a conformance fixture ([conformance](conformance.md)). The population it is computed on
comes from the scorer: `conformance_population(ctx, tag)` when declared, else the built-in population of the
scorer's family (`clf`, `seq`, `grid`, `box`). A scorer with neither has no fixture, which `bench lint` refuses.

## Metrics

Metrics live in `yavih.metrics.registry` as `MetricSpec`s: a lazy function reference
(`"yavih.metrics.<module>:<function>"`), the scorers it serves (`readouts`), the protocol keys it consumes, its kind
(metric, floor, control), `floor_of`, `higher_is_better` and admissible intervals. A metric of another
scorer is refused for yours unless yours names that scorer as its `family`; `n_eval` serves every scorer.
Register new ones with `yavih.metrics.registry.register(MetricSpec(...))`. Benchmark-specific metrics belong in
the folder's `eval.py` `METRICS` instead.

## Registering it

- **Built in**: add the module to `BUILTIN` in `yavih/readouts/registry.py` and a default `--like` folder
  in `yavih/scaffold.py`.
- **From a package**: declare an entry point; no yavih change is needed.

  ```toml
  # pyproject.toml of the plugin package (not a yavih configuration file)
  [project.entry-points."yavih.scorers"]
  toy = "mypkg.toy_scorer"
  ```

- **In a benchmark folder**: `eval.py` exports `SCORER = Scorer(name="<the folder's name>", ...)`;
  `Benchmark.load` registers it (`registry.register_folder`) and the yaml says `scorer: <the folder's name>`.
  A folder scorer may never carry the name of a built-in or entry-point scorer.
- **At run time** (tests, an in-process plugin): `registry.register(SCORER)`; `registry.reset()` forgets
  run-time and folder registrations.

A broken plugin is reported by name and does not break the core. `yavih.versions.READOUT_VERSION` reads
every registered scorer's version.

## Testing it

`tests/test_extensibility.py` shows the minimum: a toy scorer registered through an entry point validates a
benchmark that uses it, hashes, pins its configuration and builds its floor. A new built-in scorer also
needs:

- a synthetic fixture so `yavih.testing.make_synthetic_benchmark(tmp, "<name>")` works;
- an end-to-end test where `oracle_bridge` clears the floor and `random_bridge` does not;
- metric tests against an independent implementation where one exists;
- a [readouts](readouts.md) section and, if it introduces validation codes, rows in [validation](validation.md).
