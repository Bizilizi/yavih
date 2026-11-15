![llp_avvp](assets/header.png)

TODO: [Paper](https://…) · [Homepage](https://…) · [Dataset](https://…) · [Licence](https://…) · [setup.md](setup.md)

<!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance llp_avvp --write) -->
| At a glance | llp_avvp |
|---|---|
| **Task** | Temporal detection, multi-label · audio-visual, temporal |
| **Modality** | Video + audio |
| **Size** | 6 489 train / 2 167 test · 100 classes |
| **Headline** | mAP averaged over tIoU thresholds (`map_avg`) |
| **Readings** | Audio · video · audio+video |
| **Data** | Public download; fetched unattended · 53 GB of media · how: [setup.md](setup.md) |
| **Licence** | The HF dataset card (ttgeng233/UnAV-100) declares CC BY-NC-SA 4.0 (non-commercial, share-alike); the project page shows a CC BY 4.0 badge; the code repository is MIT. The media are YouTube videos (VGGSound sources) subject to YouTube's … |
| **Protocol** | Version 1 · scorer `llp_avvp` (the folder's own) · needs yavih >=1.1,<2 · `yavih bench fetch llp_avvp` |
<!-- yavih:glance end -->

## The task

TODO: what a model is asked to do, in human terms, for a reader deciding whether this benchmark is for them: one
example input and the answer it expects. How to write this page: https://github.com/Bizilizi/yavih/blob/main/docs/writing-a-readme.md

<!-- once rendered (python -m yavih.previews llp_avvp --suggest --write, then python -m yavih.previews llp_avvp), un-comment:
![TODO: what the figure shows](assets/preview.png)
-->

## The data

TODO: what the recordings are and where they come from, the official splits in prose, the licence in words.
(Scaffolded from `unav100`: the glance above shows `unav100`'s numbers until this benchmark's manifest replaces them.)

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
