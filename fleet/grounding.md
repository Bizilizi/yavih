# Grounding a benchmark's evaluation

A fleet worker does not write an evaluation from the paper's text when an implementation of it exists. Every part
of a folder's evaluation is grounded in one named implementation, chosen by the ladder below, and the folder says
which one. yavih is a community-driven evaluation repository: it carries other people's evaluations faithfully and
credits them.

A benchmark that belongs to a suite (the queue entry's `parents`) is grounded in that suite's implementation of it.
The ladder is for every other benchmark, and for a suite member the suite's code does not cover.

## The parts of an evaluation

An evaluation can have up to three parts, and each is grounded on its own: they often have different sources.

| part | what it is | example |
|---|---|---|
| **scorer** | what turns the model's outputs and the ground truth into the number: metrics, answer parsing, aggregation | TAP-Vid's `compute_tapvid_metrics`; the DAVIS J and F measures; MVBench's `check_ans` |
| **procedure** | a training-free computation between the model's features and the scorer's input | label propagation of the first-frame mask for DAVIS |
| **readout** | a trained head on frozen features, with its loss and training recipe, where no core scorer fits | a temporal detection head for an event-localization benchmark |

A benchmark's authors usually publish the scorer and rarely the other two (their own model is often trained end to
end). The scorer is then official and the procedure or readout comes from a reference project.

## The ladder

For each part, take the first rung that applies.

| rung | the implementation | what the folder does |
|---|---|---|
| 1 | **official, released as a wheel**: the benchmark authors' code on PyPI, installed with `pip install <name>==<version>`, whose module that computes the number imports in a current model environment | imports it: an `external` entry in `benchmark.yaml`, every call through `yavih.external.require(name, pip, version_pin)`, which asserts the import and the version |
| 2 | **official, not a released wheel**: the authors' code exists as a repository, scripts or a notebook | copies it into the folder |
| 3 | **a reference implementation**: no official code for this part; the part as a well-recognised project implements it (see "Choosing the reference") | copies it into the folder, also when the reference is an installable package |
| 4 | **the paper only**: no usable implementation anywhere | writes it from the paper, and says so. The report lists where the worker looked. |

**Official** means the code the benchmark's own release points to: its paper, its dataset card, its README. A later
script by the same group, in another repository, is a rung-3 candidate.

**Why copying is the default.** Only a released wheel is imported. An agent that sets a benchmark up will run
`pip install <name>==<version>`; asked to install from a git URL, many agents refuse and hand the step to a
person, which breaks the one thing yavih promises: that the agent does the wiring. Copied code also sits in the
folder where it can be read, diffed against its source and audited. So official code is rung 2 whenever:

- it is not on PyPI: a repository that only installs from git (`pip install git+<url>`), even with a
  `pyproject.toml` or `setup.py`; a notebook; a script inside a model repository;
- it is on PyPI, but the module that computes the number imports frameworks a torch environment should not need
  for it (TensorFlow, JAX), or its pins exclude a current numpy or torch and `--no-deps` does not get around them;
- its authors document the file as stand-alone and meant to be copied.

**Glue around an import.** A rung-1 package often keeps part of the evaluation outside the importable functions: a
headline formula in a command-line script, a loop that reads result files from disk. That glue is transcribed into
`eval.py` and cited like copied code.

## Choosing the reference (rung 3)

The reference is the implementation a reader of a recent paper would expect the number to come from. In order:

1. It implements **this part for this benchmark**, with public code that can be read at a pinned commit. A project
   that reports the benchmark without releasing that code is named as a user of the convention, never as the source.
2. It is **well recognised**: the code of a widely cited model from a major venue or a major lab, or the evaluation
   toolkit the field reports with.
3. It is **recent**: prefer the newest project that satisfies 1 and 2, since it shows the convention in use today.
   When its file is a copy of an older one with the scored computation unchanged, take the older file as the source
   and name the recent project as its user. A file that rewrites the computation is itself the source.
4. Between equals, prefer the one whose evaluation is closest to how yavih will run the benchmark (a frozen backbone
   with a readout for a probe benchmark; the model's own answers for a question-answering benchmark), then the one
   whose code needs the fewest changes to run inside a folder.

When the best-recognised project's code is missing or incomplete, the next well-recognised project with complete
code is the reference, and the report says so. A readout may take its head from one project and its training
recipe from another; each is named. When recognised projects disagree on settings that move the number (context length,
neighbourhood, prompt wording), the folder takes those of the chosen reference and lists the other variants in
`metrics.notes`.

**A readout without a recognised reference is the core's.** When no well-recognised project publishes a
frozen-backbone readout for the task (the authors train end to end, the known work released no code, what exists is
a little-used repository), the worker neither copies the little-used one nor designs a head in the folder. yavih
defines one standard readout for that family of tasks, as it does for classification: a change to the core, made
once by the maintainer and shared by every benchmark of the family. The worker grounds the scorer, builds
everything that does not need the readout, and reports the entry blocked with `core request: a standard readout
for <task family>`, naming what it found.

The worker names the candidates it considered and why the chosen one won, in one or two lines each. Reproducing the
reference model's published number is not required: it is often impossible, and it is never a condition for a folder.

## Copying (rungs 2 and 3)

- **Take as much as the benchmark needs**: the scorer and its metrics, split and label handling, answer parsing,
  prompt templates, heads and losses, target construction, post-processing. The folder must be self-sufficient: it
  runs without the upstream repository, and nothing is fetched from it at evaluation time.
- **Keep it recognisable.** Copy functions whole and change only what must change to run in the folder (imports,
  removed dependencies, globals, data in and out). A reader should be able to diff the folder's code against the
  upstream file.
- **Copy it as it is, defects included.** The copied behaviour is the headline, also where it is demonstrably wrong
  (a parser that accepts an empty answer). Nothing is fixed silently: the defect is described in `metrics.notes`,
  the folder adds a diagnostic metric that shows when it matters (an empty-answer rate), and the worker reports it.
- **Say where it came from, in the code.** Every copied file, or copied block in a file, starts with a comment
  naming the project, the repository URL, the commit, the upstream path, its licence, and the changes made.
- **Keep the licence.** The upstream licence text or notice stays with the copied code. An upstream that states no
  licence, a non-commercial one, or is adapted from such code is copied all the same: the comment records the
  licence as found (`none stated`), and the worker does not stop or ask.

## Protocol or bridge

Copied code mixes what the benchmark fixes with what the reference chose for its own model. Each piece goes where
yavih keeps it.

| goes into the protocol (`eval.py`, the manifest, `benchmark.yaml`) | becomes a default of `bridge_template.py`, named as the reference's |
|---|---|
| what computes the number, and every setting of it | which frames, how many, at what rate |
| prompt text, system text, an answer prefix, the option format: text the model is handed | how that text is placed in the model's chat template; a bridge that cannot prefill an answer says so in `describe()` |
| the resolution the scorer works at, when a metric is defined in pixels | the resolution, crop and normalisation the model is run at |
| the media form the grounded implementation reads (raw videos with start and end times, or pre-cut clips); another upstream form only when the folder declares it | decoding, batching, the layer tapped, pooling |

A scorer setting counted in the model's own units (a neighbourhood of 12 token cells) depends on the bridge's
resolution and patch size. The folder states that dependence in `metrics.notes`, or restates the setting in pixels
or seconds.

A readout trained on another dataset than the one it is scored on (a point-tracking head trained on synthetic
video) makes that dataset part of the folder: its own prepare step, its source and licence in `setup.md`, and the
folder's `obtain.tier` is the most restrictive of its datasets.

## Acknowledging (every rung)

- `README.md` "How it is scored": one sentence naming whose implementation scores the benchmark, with a link.
- `README.md` "Citation": the BibTeX of the reference project next to the benchmark's own, when it is not the
  benchmark's authors.
- `setup.md` "## Dependencies": per part, the rung, the repository, the commit and which folder files carry the
  copied code (rung 1: the `external[].pip` line, as that section already requires).
- `benchmark.yaml` `metrics.notes`: what computed the number, and the differences from other widely used
  implementations that can move it, so `yavih bench info` and the report say why two published numbers differ.

## In the worker's report

One line per part the folder has:

```
- grounding (scorer | procedure | readout): rung <1-4>; <project> <repository URL> @ <commit>; licence <licence>;
  <upstream files> -> <folder files>; candidates considered: <name: why not>, ...
```
