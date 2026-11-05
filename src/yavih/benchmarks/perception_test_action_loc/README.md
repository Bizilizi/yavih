![PT · action loc](assets/header.png)

[Paper](https://arxiv.org/abs/2305.13786) · [Homepage](https://github.com/google-deepmind/perception_test) · [Dataset](https://github.com/google-deepmind/perception_test#download-the-data-and-annotations) · [setup.md](setup.md)

## Description

The Perception Test (Pătrăucean et al., NeurIPS 2023 Datasets and Benchmarks) is a diagnostic benchmark of 11.6 k
real-world videos (about 23 s each, 6.7 to 36 s) filmed by crowd-sourced participants to show perceptually
interesting situations, annotated for six tasks. In the temporal action localisation task a model receives a video
and must localise and classify every action in it: each action is one of 63 templated classes ("Putting something
into something", "Pretending to open something", "Clapping hands", …) plus its start and end time. The yearly
Perception Test challenge (eval.ai) trains on 2 009 videos and evaluates on a 5 359-video validation split (public
labels) and a held-out test split (hidden labels). The metric is detection mean average precision averaged over
temporal IoU thresholds 0.1, 0.2, 0.3, 0.4 and 0.5 (the official ActionFormer baseline's `ANETdetection`). The data
is CC-BY 4.0.

## Citation

```bibtex
@inproceedings{patraucean2023perception,
  title     = {Perception Test: A Diagnostic Benchmark for Multimodal Video Models},
  author    = {P{\u{a}}tr{\u{a}}ucean, Viorica and Smaira, Lucas and Gupta, Ankush and Recasens Continente, Adri{\`a} and
               Markeeva, Larisa and Banarse, Dylan and Koppula, Skanda and Heyward, Joseph and Malinowski, Mateusz and
               Yang, Yi and Doersch, Carl and Matejovicova, Tatiana and Sulsky, Yury and Miech, Antoine and
               Fr{\'e}chette, Alex and Klimczak, Hanna and Koster, Raphael and Zhang, Junlin and Winkler, Stephanie and
               Aytar, Yusuf and Osindero, Simon and Damen, Dima and Zisserman, Andrew and Carreira, Jo{\~a}o},
  booktitle = {Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)},
  year      = {2023},
  url       = {https://openreview.net/forum?id=HYEGXFnPoq}
}
```

## In yavih

### Protocol

Scorer `perception_test_action_loc`, the folder's own temporal-detection scorer (`eval.py` and `detection.py`; `yavih bench info perception_test_action_loc`
names it on its `extends:` line), `T = 128`: the probe reads frozen tokens that carry a time coordinate, places
each at its absolute second relative to the clip's annotated duration, and predicts on 128 bins over the clip which
of the 63 classes is active and, per bin, the distance to the action's start and end; trained with a sigmoid focal plus a 1-D GIoU loss; the decoded
segments are scored like the challenge. `protocol_version: 1`.

| split value | role | rows (videos) | events | notes |
|---|---|---|---|---|
| `train` | train | 2 009 | 13 097 | `challenge_action_localisation_train_annotations.zip`, every video (60 of the 63 classes occur); the probe carves its validation split from it |
| `valid` | eval | 5 359 | 35 440 | `challenge_action_localisation_valid_annotations.zip` = `localisation_challenge_valid_id_list.csv`; all 63 classes occur |

One row per video; nothing excluded or capped. The challenge test split (3 233 videos) has hidden labels and is not
in the manifest. Three eval classes never occur in train (`Pretending to launch something`, `Pretending to plug
something into something`, `Pretending to tear something into pieces`: 7 eval events); they count in the class mean
as the challenge counts them, and no probe can score on them.

Headline **`tmap`** = the challenge metric: per class present in the eval ground truth, the detection AP at a tIoU
threshold (predictions sorted by score, greedy one-to-one matching inside each video, VOC-2011 interpolated
precision), averaged over classes and over the thresholds 0.1 to 0.5 (`protocol.tiou_thresholds`); `tmap_per_tiou`
lists the five. It is a transcription of `ANETdetection` from the baseline repository
(github.com/ptchallenge-workshop/actionformer_release_PT at `10684a4`, `libs/utils/metrics.py`), which is not
pip-installable; one documented difference: a prediction of a class absent from the eval ground truth is dropped,
where the reference would alias it onto another class through `DataFrame.replace`. Floor **`tmap_floor` = 0.012**
(per bin and class, the train frequency of that class at that relative time with the median train extents, decoded
and scored the same way). Clip-bootstrap 95 % interval for `tmap` (1 000 resamples, seed 0). Disclosures:
`n_events` (35 440), `n_eval`. No chance value, no controls. The pinned decoding (threshold 0.001, top 2 000
candidates, class-wise NMS at tIoU 0.5, 100 segments per clip) reaches `tmap` = 0.96 when fed the eval targets
themselves (one regression per bin cannot separate every overlapping action of one class).

The bridge gets two streams: `video`, the stored mp4 (H.264 up to 1920x1080, ~30 fps), and `audio`, the official
32-bit PCM wav (44.1 or 48 kHz, mono or stereo); frames, windows, crop and audio front end are the bridge's. It
returns a fixed `[N, D]` per clip with `pos.t` in media coordinates; tokens tagged `video` with `t` are required
(`requires.streams: [video]`), so the readings are `v` (headline, like the official video baseline) and `av`; an
audio-only reading is not applicable. Details: [setup.md](setup.md).

### Variants and aliases

No aliases. Sibling benchmark: `perception_test_sound_loc` (the same videos, temporal sound localisation, 11
classes). Not this benchmark: the README's `action_localisation_train_annotations.zip` (2 135 train videos, a
superset of the 2 009 the challenge baseline trains on, same 60 classes), the hidden-label test split (eval.ai), and
numbers from models fine-tuned on the task.

### Reference numbers

None quoted: the published baselines (ActionFormer on TSP video features, and the challenge entries) train a
detector end to end on top of pre-extracted features, not a pinned frozen probe, so they are not comparable to a
yavih number.

### Re-freezing the manifest

`python "$(yavih bench path perception_test_action_loc)/make_manifest.py" --check` re-derives `manifest.csv.gz`
and `classes.csv` from the two public annotation zips (sha256-pinned; `yavih bench prepare
perception_test_action_loc 01` puts them under `<media_root>/annotations`) and prints the `manifest.upstream` block.
`tests/test_folder.py` re-checks the derivation on a stand-in rebuilt from the committed manifest, the curation
sums, the events cells and the reader on synthetic media.
