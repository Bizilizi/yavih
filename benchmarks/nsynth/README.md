![nsynth](assets/header.png)

TODO: [Paper](https://…) · [Homepage](https://…) · [Dataset](https://…) · [Licence](https://…) · [setup.md](setup.md)

<!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance nsynth --write) -->
| At a glance | nsynth |
|---|---|
| **Task** | Sound classification · audio |
| **Modality** | Audio |
| **Size** | 2 000 in folds 1–5 (400 each) · 50 classes |
| **Headline** | Mean accuracy over the folds (`acc_mean`) · chance 2 % |
| **Data** | Public download; fetched unattended · 0.88 GB of media · how: [setup.md](setup.md) |
| **Licence** | CC BY-NC 3.0 for the dataset as a whole; the ESC-10 subset (esc10 == True, 400 clips) CC BY 3.0; per-clip Freesound attributions in the upstream LICENSE file |
| **Protocol** | Version 1 · scorer `clf` · needs yavih >=1.1,<2 · `yavih bench fetch nsynth` |
<!-- yavih:glance end -->

## The task

TODO: what a model is asked to do, in human terms, for a reader deciding whether this benchmark is for them: one
example input and the answer it expects. How to write this page: https://github.com/Bizilizi/yavih/blob/main/docs/writing-a-readme.md

<!-- once rendered (python -m yavih.previews nsynth --suggest --write, then python -m yavih.previews nsynth), un-comment:
![TODO: what the figure shows](assets/preview.png)
-->

## The data

TODO: what the recordings are and where they come from, the official splits in prose, the licence in words.
(Scaffolded from `esc50`: the glance above shows `esc50`'s numbers until this benchmark's manifest replaces them.)

## How it is scored

TODO: the metric and why it fits the task; chance and the floor in words; what is not this benchmark (variants,
aliases, near-misses), one sentence each.

## Reference numbers

TODO: published numbers with their citation and why they are (not) comparable to a frozen-feature probe; or delete
this optional section.

## Citation

```bibtex
TODO: the BibTeX entry of the paper
```
