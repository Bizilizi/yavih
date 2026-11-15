![mimetics](assets/header.png)

TODO: [Paper](https://…) · [Homepage](https://…) · [Dataset](https://…) · [Licence](https://…) · [setup.md](setup.md)

<!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance mimetics --write) -->
| At a glance | mimetics |
|---|---|
| **Task** | Video classification · visual |
| **Modality** | Video |
| **Size** | 20 000 train / 13 492 val · 400 classes |
| **Headline** | Accuracy (`acc`) · chance 0.25 % |
| **Data** | Public torrent; fetched unattended · 20 GB of media · how: [setup.md](setup.md) |
| **Licence** | Kinetics-400 annotations: CC BY 4.0 (DeepMind). The media are YouTube clips re-encoded by a third party (Academic Torrents upload 2020-11-07, height 256; portrait clips are 256×144, short side 144) and remain subject to YouTube's terms and … |
| **Protocol** | Version 1 · scorer `clf` · needs yavih >=1.1,<2 · `yavih bench fetch mimetics` |
<!-- yavih:glance end -->

## The task

TODO: what a model is asked to do, in human terms, for a reader deciding whether this benchmark is for them: one
example input and the answer it expects. How to write this page: https://github.com/Bizilizi/yavih/blob/main/docs/writing-a-readme.md

<!-- once rendered (python -m yavih.previews mimetics --suggest --write, then python -m yavih.previews mimetics), un-comment:
![TODO: what the figure shows](assets/preview.png)
-->

## The data

TODO: what the recordings are and where they come from, the official splits in prose, the licence in words.
(Scaffolded from `k400`: the glance above shows `k400`'s numbers until this benchmark's manifest replaces them.)

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
