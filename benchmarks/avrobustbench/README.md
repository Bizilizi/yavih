![avrobustbench](assets/header.png)

TODO: [Paper](https://…) · [Homepage](https://…) · [Dataset](https://…) · [Licence](https://…) · [setup.md](setup.md)

<!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance avrobustbench --write) -->
| At a glance | avrobustbench |
|---|---|
| **Task** | Audio-visual classification · audio-visual, temporal |
| **Modality** | Video + audio |
| **Size** | 24 000 train / 3 000 test / 5 997 stationary · 9 classes |
| **Headline** | Accuracy (`acc`) · chance 11 % |
| **Readings** | Audio · video · audio+video |
| **Data** | Derived from another dataset; media on request from the data owner · 74 GB of media · how: [setup.md](setup.md) |
| **Licence** | Derived from VGGSound (annotations CC BY 4.0; media are YouTube clips fetched by the user, subject to YouTube's terms). The offsets and the clip selection are model-free derived data of this benchmark; no new media. |
| **Protocol** | Version 1 · scorer `clf` · needs yavih >=1.1,<2 · `yavih bench fetch avrobustbench` |
<!-- yavih:glance end -->

## The task

TODO: what a model is asked to do, in human terms, for a reader deciding whether this benchmark is for them: one
example input and the answer it expects. How to write this page: https://github.com/Bizilizi/yavih/blob/main/docs/writing-a-readme.md

<!-- once rendered (python -m yavih.previews avrobustbench --suggest --write, then python -m yavih.previews avrobustbench), un-comment:
![TODO: what the figure shows](assets/preview.png)
-->

## The data

TODO: what the recordings are and where they come from, the official splits in prose, the licence in words.
(Scaffolded from `avsync`: the glance above shows `avsync`'s numbers until this benchmark's manifest replaces them.)

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
