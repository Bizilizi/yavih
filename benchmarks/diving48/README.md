![diving48](assets/header.png)

TODO: [Paper](https://…) · [Homepage](https://…) · [Dataset](https://…) · [Licence](https://…) · [setup.md](setup.md)

<!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance diving48 --write) -->
| At a glance | diving48 |
|---|---|
| **Task** | Video classification · visual |
| **Modality** | Video |
| **Size** | 20 000 train / 24 777 val · 174 classes |
| **Headline** | Accuracy (`acc`) · chance 0.57 % |
| **Data** | Official portal; needs a signed licence · 4 GB of media · how: [setup.md](setup.md) |
| **Licence** | Something-Something v2 (TwentyBN, now distributed by Qualcomm Technologies): free download after registration and acceptance of the dataset licence on the Qualcomm developer portal; research use under those terms, no redistribution of the … |
| **Protocol** | Version 1 · scorer `clf` · needs yavih >=1.1,<2 · `yavih bench fetch diving48` |
<!-- yavih:glance end -->

## The task

TODO: what a model is asked to do, in human terms, for a reader deciding whether this benchmark is for them: one
example input and the answer it expects. How to write this page: https://github.com/Bizilizi/yavih/blob/main/docs/writing-a-readme.md

<!-- once rendered (python -m yavih.previews diving48 --suggest --write, then python -m yavih.previews diving48), un-comment:
![TODO: what the figure shows](assets/preview.png)
-->

## The data

TODO: what the recordings are and where they come from, the official splits in prose, the licence in words.
(Scaffolded from `ssv2`: the glance above shows `ssv2`'s numbers until this benchmark's manifest replaces them.)

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
