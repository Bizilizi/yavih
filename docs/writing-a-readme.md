# Writing a benchmark README

A benchmark folder's `README.md` is the benchmark's page for people: a short project page that a researcher finds on
GitHub (or through the catalog) and reads to decide whether the benchmark measures something they care about. It
introduces the task, shows the data, says how a number is earned and what it can be compared with, and gives the
citation. This page is the guide for writing one, for a contributor adding a folder and for a fleet worker
rewriting an old one. The structure is a contract checked by `yavih bench lint`
([README.md contract](benchmark-folder.md#readmemd-contract)); the wording is yours.

## Who reads it, and who does not

The reader knows machine learning, has never used this dataset, and is skimming: the header, the glance table, the
figure, the first sentence of each section. Answer their questions in that order: what is the task, what does the
data look like, how is it scored, how do published models do, how do I cite it.

**Nothing an agent needs lives in README.md.** yavih is driven by agents, and every fact an agent acts on has a home
that a command prints: the protocol in `benchmark.yaml` (`yavih bench info <name>`, `yavih bench show <name> yaml`),
the setup and both contracts in `setup.md` (`yavih bench show <name> setup`), how to read the numbers in
`metrics.notes` (printed by `bench info` and the report), and the re-freeze recipe in the `make_manifest.py`
docstring (`yavih bench show <name> make_manifest`). `bench info` names the README's path and never prints it. So the
README never repeats a split table, a scorer configuration, a hash or a command sequence for the sake of an agent,
and never has to stay in sync with one.

## Tone and length

- Plain, specific, present tense. "Each clip is five seconds of one environmental sound" beats "a diverse,
  challenging collection".
- Numbers with their units and their source: 2 000 clips, 44.1 kHz, 50 classes, five folds of 400.
- Neutral about models. The page describes the benchmark, never a model's strengths.
- Words before identifiers. Say "mean accuracy over the five folds"; the glance table carries `acc_mean`. A
  yavih term (scorer, readout, `protocol_version`) appears only where the reader needs it to compare numbers.
- Short paragraphs (two to five sentences), a table only where it beats prose, one figure per point.
- About 60 to 120 lines in total, the citation included. A longer page is usually carrying agent material.
- Links only to what anyone can open: the paper, the dataset page, the licence, `setup.md`, and yavih's docs by their
  `https://github.com/Bizilizi/yavih/blob/main/docs/<page>.md` URL (the folder is fetched and shipped without
  `docs/`, `bench lint` enforces it).

## The page, top to bottom

```markdown
![<header.name>](assets/header.png)

[Paper](…) · [Homepage](…) · [Dataset](…) · [Licence](…) · [setup.md](setup.md)

<!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance <name> --write) -->
… generated, never edited by hand …
<!-- yavih:glance end -->

## The task
## The data
## How it is scored
## Reference numbers        (optional)
## Citation
```

**Header image and links.** The first line is the header image, rendered from the yaml's `header:` block
(`python -m yavih.headers <name>`). The second paragraph is the links line: paper, project page, dataset, licence,
whichever exist, and always `[setup.md](setup.md)`. No H1: the image carries the name.

**At a glance.** A two-column table generated from `benchmark.yaml`, in words: the task and capability tags, the
modality, the size (rows per split, classes), the headline metric with chance, the readings when there are several,
where the data comes from and who must act to get it, its size, the licence, and in the last row the identifiers
(version, scorer, the yavih releases it needs, the fetch command). Write it with
`python -m yavih.doctables --glance <name> --write` and again after every yaml edit; `bench lint` fails on a missing
or stale block (`E_README_GLANCE`). Because these facts are in the table, the prose never has to repeat them.

**The task.** What a model is asked to do, in one or two paragraphs a newcomer understands: the input, the expected
answer, and one concrete example ("given a five-second clip, name its one sound: `dog`, `rain`, `chainsaw`, …").
Then the main figure, `![<what it shows>](assets/preview.png)`, and for a task on video the motion clip,
`![<what it shows>](assets/clip.gif) · [mp4](assets/clip.mp4)` (the GIF plays inline, the MP4 is its full-quality
source). The alt text says what the figure shows, not "figure".

**The data.** What the recordings are and where they come from (who made them, from which source, how they were
annotated), what one item looks like (length, rate, resolution), the official splits in prose ("five folds of 400
clips, grouped by source recording so a recording never spans two folds"), and the licence in words, including what
it forbids. How to download it belongs to `setup.md`; say only whether it is public, behind a portal or a licence.

**How it is scored.** The metric in words and why it fits the task (balanced classes make accuracy meaningful;
multi-label makes mAP the standard), what chance is and what the floor is in words (the score a model earns from the
label distribution alone), and **what is not this benchmark**: a subset, a variant, a near-namesake or a
neighbouring protocol whose numbers are not these numbers (ESC-10 is not ESC-50; one fold is not five). Mention
that yavih scores frozen features with a fixed probe when the comparison with published numbers depends on it.

**Reference numbers** (optional, right before Citation). Published numbers with their citation, and why they are or
are not comparable with a frozen-feature probe (fine-tuned models, other splits, other metrics). Leave the section
out when nothing is published.

**Citation.** The paper's BibTeX in a fenced `bibtex` block, and that of the annotation source where it differs.

## Which figure for which task

`python -m yavih.previews <name> --suggest --write` drafts `assets/preview.yaml` from the folder's streams, label kind,
scorer and capability, with rows drawn at random from the eval split. Read the draft and change its `rows` until the
figure shows the task: clear examples, different classes, the labels readable, no borderline clip as the first thing a
reader sees. The kinds, sizes and budgets are in the [design system](design-system.md#figures).

| task family | main figure (`preview`) | motion (`clip`) |
|---|---|---|
| image classification | `images`: 8 labelled tiles | |
| video classification | `frames`: a 6-frame strip with timestamps | `motion` |
| audio classification, tagging, sound event detection | `audio`: two clips, waveform over mel, one chip per label | |
| audio-visual events (labelled seconds or intervals) | `av`: frames, waveform and mel on one time axis, events as bands | `motion` |
| temporal detection or online detection on video | `timeline`: frames over the labelled intervals | `motion` |
| sound source localisation (boxes) | `boxes`: the annotated frame with its box, a card with the class and the sound | `motion` |
| depth or another dense target | `depth`: three image → target pairs | |
| generation (text or video to audio) | `generation`: what the model is given → the reference sound | `motion` for video input |
| cross-modal retrieval | `retrieval`: a query → five candidates, the match ringed | `motion` |
| data whose licence forbids redistribution | `classes`: sampled class names with counts, no media | none |

Two rules hold for every figure: every image renders on the dark design background, never on white, and a figure
shows a handful of downscaled stills and at most five seconds of media
([pixels and licences](design-system.md#pixels-and-licences)). A folder whose task none of the kinds shows can add its
own kind in `preview.py` ([adding a kind](design-system.md#adding-a-kind)).

## What never goes in a README, and where it goes

| content | its home | printed by |
|---|---|---|
| split names, roles and row counts; headline id, chance, floors, controls; `protocol_version`; aliases | `benchmark.yaml` (and the generated glance block) | `yavih bench info <name>`, `yavih bench show <name> yaml` |
| floor values, CI widths, noise, the smallest gap that means something, "a number near the floor on one fold means …" | `benchmark.yaml` `metrics.notes` (at most 8 sentences of at most 200 characters) | `yavih bench info <name>` (`note:` lines), `yavih report build` |
| how the manifest is re-frozen: upstream files and pins, curation, audits, the commands (`--check`) | the `make_manifest.py` module docstring | `yavih bench show <name> make_manifest`, `make_manifest.py --help` |
| what the bridge gets and returns, decoding and preprocessing pitfalls | `setup.md` "Input contract", "Output contract", "Gotchas" | `yavih bench show <name> setup` |
| dependencies, and where the folder's scorer transcribes the authors' code (and how it differs) | `setup.md` "Dependencies" (and "Gotchas") | `yavih bench show <name> setup` |
| the probe configuration, hashes, the scorer's internals | the core and `benchmark.yaml` | `yavih bench info <name>`, `yavih bench hash <name> --explain` |
| the version history | `benchmark.yaml` `changelog` | `yavih bench info <name>` |
| maintainer notes, decisions, provenance logs | the folder's `notes/` (ignored by git, never shipped) | nobody |

Two more things never appear: a `## Changelog`, `## History`, `## Checklist` or `## Maintainers` heading (`bench lint`
refuses them), and a relative link into the repository (`docs/…`, `notes/…`, `src/…`): a fetched folder has none of
them.

## A worked example: esc50

````markdown
![ESC-50](assets/header.png)

[Paper](https://dl.acm.org/doi/10.1145/2733373.2806390) · [Homepage](https://github.com/karolpiczak/ESC-50) · [Dataset](https://dx.doi.org/10.7910/DVN/YDEPUT) · [Licence](https://creativecommons.org/licenses/by-nc/3.0/) · [setup.md](setup.md)

<!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance esc50 --write) -->
| At a glance | ESC-50 |
|---|---|
| **Task** | Sound classification · audio |
| **Modality** | Audio |
| **Size** | 2 000 in folds 1–5 (400 each) · 50 classes |
| **Headline** | Mean accuracy over the folds (`acc_mean`) · chance 2 % |
| **Data** | Public download; fetched unattended · 0.88 GB of media · how: [setup.md](setup.md) |
| **Licence** | CC BY-NC 3.0 for the dataset as a whole; the ESC-10 subset (esc10 == True, 400 clips) CC BY 3.0; per-clip Freesound attributions in the upstream LICENSE file |
| **Protocol** | Version 1 · scorer `clf` · needs yavih >=1.1,<2 · `yavih bench fetch esc50` |
<!-- yavih:glance end -->

## The task

Hear a five-second recording and name the one environmental sound in it: a dog barking, rain, a chainsaw, a
crackling fire, a clock ticking. There are 50 classes in five broad groups (animals; natural soundscapes and water;
human non-speech sounds; domestic sounds; urban noises), and every clip belongs to exactly one.

![Two ESC-50 clips, waveform over mel spectrogram, each with its class](assets/preview.png)

## The data

ESC-50 (Karol J. Piczak, ACM Multimedia 2015) is 2 000 clips cut from public field recordings on Freesound, 40 per
class, each exactly 5.0 s of 44.1 kHz mono audio. The authors arranged them into five folds of 400 clips, eight per
class per fold, keeping the clips cut from one source recording in one fold where possible, so the standard protocol
is five-fold cross-validation. The dataset is licensed CC BY-NC 3.0 (non-commercial); the 400-clip ESC-10 subset is
CC BY 3.0. It is a public download of 0.88 GB of audio.

## How it is scored

Each fold is held out once: a probe trained on the other four folds names the classes of the held-out 400 clips, and
the score is the accuracy averaged over the five folds. The classes are perfectly balanced, so accuracy is
meaningful and chance is 2 %; the floor, a model that always answers the most frequent training class, is also 2 %.

Not this benchmark: the ESC-10 subset (10 classes, 400 clips) and an accuracy on a single fold are different
numbers, and the upstream numeric class ids follow another order than yavih's class list.

## Reference numbers

The ESC-50 repository lists human listeners at 81.3 % (Piczak 2015), AST at 95.7 % (Gong et al., Interspeech 2021)
and BEATs at 98.1 % (Chen et al., ICML 2023). These models are fine-tuned end to end; a frozen-feature probe is
not comparable with them.

## Citation

```bibtex
@inproceedings{piczak2015dataset,
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  year = {2015},
  pages = {1015--1018},
  doi = {10.1145/2733373.2806390}
}
```
````

What moved out of the old "## In yavih" section: the split table and the `acc_mean` / CI / `n_folds` details (the
glance block and `bench info`), "treat gaps below 0.015 at one seed as unresolved" and the per-fold warning
(`metrics.notes`), and the re-freeze paragraph with its commands (the `make_manifest.py` docstring).

## Commands

```bash
python -m yavih.doctables --glance <name> --write    # write or refresh the "at a glance" block
python -m yavih.previews <name> --suggest --write     # draft assets/preview.yaml; then fix its rows
python -m yavih.previews <name> --chrome <chrome>     # render the figures (the data staged; ffmpeg for motion)
python -m yavih.headers <name> --chrome <chrome>      # render the header image
yavih bench lint <name>                               # the structure: E_README, E_README_GLANCE, W_README_ASSET
yavih bench info <name>                               # what an agent sees instead: the yaml facts and note: lines
yavih bench show <name> make_manifest                 # the re-freeze recipe
```

## Rewriting a legacy README

The packaged folders were written with a `## Description` / `## Citation` / `## In yavih` README. `bench lint`
accepts that shape with the warning `W_README_LEGACY` until the folder is rewritten; `fleet/queue.yaml` marks such
folders `readme: legacy`, and a fleet worker rewrites one folder per branch with
[the README migration brief](../fleet/readme-brief.md). Take the old README apart by audience: the description
becomes "The task" and "The data"; floors, CI widths and noise guidance become `metrics.notes`; the re-freeze
paragraph is checked against the `make_manifest.py` docstring (which gains what it lacks); transcription and
dependency facts move to `setup.md`; published numbers and "what is not this benchmark" stay, as prose, in "How it is
scored" and "Reference numbers"; everything the glance block or `bench info` already shows is dropped.
