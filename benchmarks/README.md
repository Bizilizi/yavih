# yavih benchmarks

The repository's benchmark folders (`yavih bench fetch <name>` brings one), one frozen protocol each. Every benchmark hands
the bridge its raw stored files and states two contracts in its `setup.md`; everything
model-side is the bridge's declared decision. Click a name for the
benchmark's README (description, citation, the protocol in yavih).

## Benchmarks by capability

Grouped by each folder's `benchmark.yaml` `capability` tags (the vocabulary suites use); a benchmark with several
capabilities is listed under each.

<!-- yavih:bench-table index begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
### audio (5)

| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [audiocaps_t2a](audiocaps_t2a/README.md) | AudioCaps text-to-audio generation (776 surviving full-length test clips; FD, KL, IS, CLAP score via av-benchmark) | audiocaps_t2a | text (cell) | public (obtain: auto) |
| [epic_sounds_det](epic_sounds_det/README.md) | EPIC-SOUNDS audio-based interaction detection (untrimmed kitchen recordings, 44 classes); alias `epic_sounds_detection` | epic_sounds_det | audio | public (obtain: auto) |
| [esc50](esc50/README.md) | ESC-50 environmental sound classification (50 classes, official 5-fold CV, mean accuracy) | clf | audio | public (obtain: auto) |
| [fsd50k](fsd50k/README.md) | FSD50K 200-class multi-label audio tagging (Freesound clips, macro mAP) | clf | audio | public (obtain: auto) |
| [perception_test_sound_loc](perception_test_sound_loc/README.md) | Perception Test temporal sound localisation (challenge train / valid) | perception_test_sound_loc | video + audio | public (obtain: auto) |

### visual (6)

| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [in1k](in1k/README.md) | ImageNet-1K image classification (frozen 50/class train draw, full 50k val); aliases `imagenet1k`, `imagenet` | clf | image | manual portal (obtain: needs-licence) |
| [k400](k400/README.md) | Kinetics-400 action recognition (Academic-Torrents _256 release, 400 classes, video-only probe) | clf | video | torrent (obtain: auto) |
| [nyuv2_depth](nyuv2_depth/README.md) | NYUv2 dense depth (12 000 GeoNet train frames, 654 labelled test images, 14x14 grid) | grid | image | manual portal (obtain: manual-download) |
| [perception_test_action_loc](perception_test_action_loc/README.md) | Perception Test temporal action localisation (challenge train / valid) | perception_test_action_loc | video + audio | public (obtain: auto) |
| [ssv2](ssv2/README.md) | Something-Something-v2 temporal action classification (frozen 20k class-balanced train draw, full 24 777 val) | clf | video | manual portal (obtain: needs-licence) |
| [thumos14_oad](thumos14_oad/README.md) | THUMOS'14 online action detection (streaming; 20 actions + background, per-frame mAP at 4 fps) | stream | video + audio | public (obtain: auto) |

### audio-visual (14)

| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [audioset20k](audioset20k/README.md) | AudioSet-20K (balanced) 527-class multi-label audio-visual tagging, published raw clips (macro mAP) | clf | video + audio | public (obtain: auto) |
| [ave](ave/README.md) | AVE audio-visual event localization (28 events + background, per-second labels) | seq | video + audio | public (obtain: auto) |
| [ave_causal](ave_causal/README.md) | AVE causal per-second event recognition (streaming; 28 events + background) | stream | video + audio | public (obtain: auto) |
| [avsync](avsync/README.md) | AV synchronization-offset classification (9 offsets, transient VGGSound clips, stationary floor) | clf | video + audio (materialized at prepare time) | derived (obtain: needs-data-owner) |
| [epic100_anticipation_stream](epic100_anticipation_stream/README.md) | EPIC-KITCHENS-100 streaming action anticipation (tau_a = 1 s; verb, noun, action class-mean top-5 recall) | stream | video + audio | public (obtain: auto) |
| [perception_test_action_loc](perception_test_action_loc/README.md) | Perception Test temporal action localisation (challenge train / valid) | perception_test_action_loc | video + audio | public (obtain: auto) |
| [perception_test_sound_loc](perception_test_sound_loc/README.md) | Perception Test temporal sound localisation (challenge train / valid) | perception_test_sound_loc | video + audio | public (obtain: auto) |
| [unav100](unav100/README.md) | UnAV-100 dense audio-visual event localization in untrimmed videos | unav100 | video + audio | public (obtain: auto) |
| [unav100_causal](unav100_causal/README.md) | UnAV-100 causal audio-visual event recognition (streaming; 100 classes, one-second bins, multilabel) | stream | video + audio | public (obtain: auto) |
| [vggsound_retrieval](vggsound_retrieval/README.md) | VGGSound test-set audio–video retrieval (A→V and V→A R@1 over 15 440 clips; the CAV-MAE Sync 1 545-clip subset alongside) | vggsound_retrieval | video + audio | public (obtain: needs-data-owner) |
| [vggsounder](vggsounder/README.md) | VGGSounder audio / visual / audio-visual multilabel probe (VGGSound-train probe, 309 classes) | clf | video + audio | public (obtain: needs-data-owner) |
| [vggsounder_v2a](vggsounder_v2a/README.md) | VGGSounder video-to-audio generation (3 000 audible clips; FD, KL, IS, ImageBind score, DeSync via av-benchmark) | vggsounder_v2a | video (materialized at prepare time) | public (obtain: needs-data-owner) |
| [vggss_box](vggss_box/README.md) | VGG-SS audio-conditioned single-box regression | box | video + audio | public (obtain: needs-data-owner) |
| [vggss_loc](vggss_loc/README.md) | VGG-SS training-free sound-source localization (cIoU@0.5, AUC on heatmaps) | vggss_loc | video + audio | public (obtain: needs-data-owner) |

### temporal (10)

| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [ave](ave/README.md) | AVE audio-visual event localization (28 events + background, per-second labels) | seq | video + audio | public (obtain: auto) |
| [ave_causal](ave_causal/README.md) | AVE causal per-second event recognition (streaming; 28 events + background) | stream | video + audio | public (obtain: auto) |
| [avsync](avsync/README.md) | AV synchronization-offset classification (9 offsets, transient VGGSound clips, stationary floor) | clf | video + audio (materialized at prepare time) | derived (obtain: needs-data-owner) |
| [epic100_anticipation_stream](epic100_anticipation_stream/README.md) | EPIC-KITCHENS-100 streaming action anticipation (tau_a = 1 s; verb, noun, action class-mean top-5 recall) | stream | video + audio | public (obtain: auto) |
| [epic_sounds_det](epic_sounds_det/README.md) | EPIC-SOUNDS audio-based interaction detection (untrimmed kitchen recordings, 44 classes); alias `epic_sounds_detection` | epic_sounds_det | audio | public (obtain: auto) |
| [perception_test_action_loc](perception_test_action_loc/README.md) | Perception Test temporal action localisation (challenge train / valid) | perception_test_action_loc | video + audio | public (obtain: auto) |
| [perception_test_sound_loc](perception_test_sound_loc/README.md) | Perception Test temporal sound localisation (challenge train / valid) | perception_test_sound_loc | video + audio | public (obtain: auto) |
| [thumos14_oad](thumos14_oad/README.md) | THUMOS'14 online action detection (streaming; 20 actions + background, per-frame mAP at 4 fps) | stream | video + audio | public (obtain: auto) |
| [unav100](unav100/README.md) | UnAV-100 dense audio-visual event localization in untrimmed videos | unav100 | video + audio | public (obtain: auto) |
| [unav100_causal](unav100_causal/README.md) | UnAV-100 causal audio-visual event recognition (streaming; 100 classes, one-second bins, multilabel) | stream | video + audio | public (obtain: auto) |

### spatial (3)

| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [nyuv2_depth](nyuv2_depth/README.md) | NYUv2 dense depth (12 000 GeoNet train frames, 654 labelled test images, 14x14 grid) | grid | image | manual portal (obtain: manual-download) |
| [vggss_box](vggss_box/README.md) | VGG-SS audio-conditioned single-box regression | box | video + audio | public (obtain: needs-data-owner) |
| [vggss_loc](vggss_loc/README.md) | VGG-SS training-free sound-source localization (cIoU@0.5, AUC on heatmaps) | vggss_loc | video + audio | public (obtain: needs-data-owner) |

### retrieval (1)

| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [vggsound_retrieval](vggsound_retrieval/README.md) | VGGSound test-set audio–video retrieval (A→V and V→A R@1 over 15 440 clips; the CAV-MAE Sync 1 545-clip subset alongside) | vggsound_retrieval | video + audio | public (obtain: needs-data-owner) |

### generation (2)

| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [audiocaps_t2a](audiocaps_t2a/README.md) | AudioCaps text-to-audio generation (776 surviving full-length test clips; FD, KL, IS, CLAP score via av-benchmark) | audiocaps_t2a | text (cell) | public (obtain: auto) |
| [vggsounder_v2a](vggsounder_v2a/README.md) | VGGSounder video-to-audio generation (3 000 audible clips; FD, KL, IS, ImageBind score, DeSync via av-benchmark) | vggsounder_v2a | video (materialized at prepare time) | public (obtain: needs-data-owner) |

### streaming (4)

| Benchmark | Description | Readout | Media | Source |
|---|---|---|---|---|
| [ave_causal](ave_causal/README.md) | AVE causal per-second event recognition (streaming; 28 events + background) | stream | video + audio | public (obtain: auto) |
| [epic100_anticipation_stream](epic100_anticipation_stream/README.md) | EPIC-KITCHENS-100 streaming action anticipation (tau_a = 1 s; verb, noun, action class-mean top-5 recall) | stream | video + audio | public (obtain: auto) |
| [thumos14_oad](thumos14_oad/README.md) | THUMOS'14 online action detection (streaming; 20 actions + background, per-frame mAP at 4 fps) | stream | video + audio | public (obtain: auto) |
| [unav100_causal](unav100_causal/README.md) | UnAV-100 causal audio-visual event recognition (streaming; 100 classes, one-second bins, multilabel) | stream | video + audio | public (obtain: auto) |
<!-- yavih:bench-table index end -->

`Media` names what the reader hands over; `Source` is `benchmark.yaml` `source.kind` in words.
`yavih bench list` prints the same set with sizes, `obtain` (who must act before the data can be staged:
`auto`, `manual-download`, `needs-licence`, `needs-data-owner`, `site-only`) and what is present at your site.

## Folder layout

Each folder holds `README.md` (for people: a header image, links, then description, citation and the
protocol in yavih), `setup.md` (for the agent: obtain the data, dependencies to install independently, the input
contract, the output contract, prepare & verify, gotchas), `benchmark.yaml` (the machine-readable protocol), the frozen
`manifest.csv.gz` and `classes.csv` (the generic reader `yavih.sample.iter_samples` hands over the stored files
and their facts as `benchmark.yaml` declares them; an out-of-tree folder may add its own `reader.py`),
an optional `eval.py` (targets, folder-specific metrics with their intervals, and the authors' scorer through an
assert-import), a `bridge_template.py` to copy, `make_manifest.py` (byte-identical re-freeze, through the shared
`yavih.freeze` helpers), `prepare/` and `tests/test_folder.py`. A folder is data + scripts, not a Python package: no `__init__.py` in it or
in its `tests/`; `eval.py` (and an optional `reader.py`) are loaded by path and never hashed:
`protocol_hash = H(input_hash, scoring_hash)` covers the declaration in `benchmark.yaml` only, and their behaviour
is held by `protocol_version` plus the conformance fixture (`conformance/expected.json`), so a `eval.py` fix is a
re-record and a `protocol_version` bump, and never invalidates extracted features. The newest
`changelog` entry of `benchmark.yaml` records the `protocol_hash` and `input_hash` the folder computes. `Benchmark.load(<name>)` resolves folders through
`importlib.resources` (`packaged_names()`, `packaged_folder()`).

## Setting a benchmark up

Read the folder's `setup.md` first, before downloading anything or writing a bridge
(`yavih bench info <name> --setup` prints it; `FOLDER=$(yavih bench path <name>)` is the folder its
`"$FOLDER/prepare/…"` commands run from):
<!-- yavih:bench-table setup_links begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
[audiocaps_t2a](audiocaps_t2a/setup.md) ·
[audioset20k](audioset20k/setup.md) ·
[ave](ave/setup.md) ·
[ave_causal](ave_causal/setup.md) ·
[avsync](avsync/setup.md) ·
[epic100_anticipation_stream](epic100_anticipation_stream/setup.md) ·
[epic_sounds_det](epic_sounds_det/setup.md) ·
[esc50](esc50/setup.md) ·
[fsd50k](fsd50k/setup.md) ·
[in1k](in1k/setup.md) ·
[k400](k400/setup.md) ·
[nyuv2_depth](nyuv2_depth/setup.md) ·
[perception_test_action_loc](perception_test_action_loc/setup.md) ·
[perception_test_sound_loc](perception_test_sound_loc/setup.md) ·
[ssv2](ssv2/setup.md) ·
[thumos14_oad](thumos14_oad/setup.md) ·
[unav100](unav100/setup.md) ·
[unav100_causal](unav100_causal/setup.md) ·
[vggsound_retrieval](vggsound_retrieval/setup.md) ·
[vggsounder](vggsounder/setup.md) ·
[vggsounder_v2a](vggsounder_v2a/setup.md) ·
[vggss_box](vggss_box/setup.md) ·
[vggss_loc](vggss_loc/setup.md).
<!-- yavih:bench-table setup_links end -->

## Adding a benchmark, and the checks that pin these documents

A new benchmark is a new folder and nothing else: `yavih bench new <name> --scorer clf --like esc50 [--dir D]`
writes a scaffold (with `TODO` where facts are needed), `yavih bench lint <name|path>` checks it against the folder
contract (`yavih.lint`), and `python -m yavih.doctables --write` regenerates the tables above, the repo README's and
the skill's from the folders' `benchmark.yaml`. A folder outside the package loads from a directory on
`$YAVIH_BENCH_PATH` or from a pip-installed pack (entry-point group `yavih.benchmarks`).

`yavih bench lint` checks, for every folder: the required files and no `__init__.py`; a strict `benchmark.yaml`
whose `name` is the folder's, with a `header:` block (name, paper title, venue, a short abstract) from which
`python -m yavih.headers <name>` renders the README's header image in the source repository, and a `capability:`
list (the groups above); README.md, the benchmark's page for people, opens with
that image, a links line and the "at a glance" block generated from the yaml (`python -m yavih.doctables --glance
<name> --write`), and its `##` headings are exactly The task, The data, How it is scored, the optional Reference
numbers, Citation (with a fenced `bibtex` block); a README still in the earlier shape (Description, Citation, In
yavih) is accepted with the warning `W_README_LEGACY` until it is rewritten. setup.md's H1 is
`# <name> — setup` and its `##` headings are exactly Obtain the data, Dependencies, Input contract, Output contract,
Prepare & verify, Gotchas; neither file has a Changelog, History, Checklist or Maintainers heading (the
`benchmark.yaml` `changelog` is the version record). "## Dependencies" carries every `external[].pip` line verbatim (or "None" when
`external` is empty); the last fenced `yaml` block of "## Input contract" / "## Output contract" parses to
`input_contract` / `output_contract`; "## Prepare & verify" contains `yavih bench verify <name>`; the shipped docs
cite nothing a pip-installed user cannot open; `eval.py` (and any `reader.py`) import without torch; the newest
`changelog` entry records the hashes the folder computes. The repo tests run it
over every packaged folder and check that the generated tables are current. Run one folder's own tests with
`python -m pytest "$(yavih bench path <name>)/tests" -q -p no:cacheprovider`.
