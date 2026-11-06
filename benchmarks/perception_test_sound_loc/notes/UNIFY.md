# Unification candidates: perception_test_sound_loc

What this folder implements that the core might absorb once enough folders share it. Maintainer notes: not in the
wheel and not a shipped file (`yavih bench files` omits them, no shipped file links them), but a fetched folder
carries this directory, and an edit here changes the folder's git tree (a documentation-only revision).

## The shared temporal-detection scorer, `detection.py`

`detection.py` is byte-identical in four folders: this one, `unav100`, `perception_test_action_loc`, `epic_sounds_det`
(sha256 `8d6d61fe6abdc03fd4554324de4a63c0240106a670a1c0412ca7d0d204eb35d9` when this note was written). `eval.py` loads it by path through its `_sibling()` helper and builds
the folder's scorer with `make_scorer("perception_test_sound_loc", ...)` and its metrics with `make_metrics(...)`; it adds only the event
encoding (`class:start_us:end_us`) and the metric names. Moving it into the core would make it one registered scorer for
timed `(class, start, end)` events. What it holds:

- **The scorer.** A `probe`-kind scorer with its own driver (`run`), training settings (`TRAIN`), floor
  (`floor_outputs`: the label-only constant output, decoded and scored like a model's), conformance population
  (`conformance_population`) and samples lines (`sample_lines`).
- **Head and loss.** The head is the core's `dense_query_v1` (layout `time:T`, `C + 2` outputs per bin), declared,
  not built, so nothing to absorb there. The loss is the folder's own `perception_test_sound_loc_focal_giou` (`focal_giou`: sigmoid focal
  loss over bins and classes plus `1 - GIoU` of the 1-D extents on positive bins), a core-loss candidate.
- **Targets.** `parse_events` and `EVENT_FORMATS` (the events-column encodings), `row_events` (a row's events and
  annotated duration), `dense_targets` (`T` bins over the annotated duration, class and start/end distance targets)
  and `task_pos_fn` (token times into the task frame).
- **Decoding.** `decode` / `decode_row` with `DECODE` (score threshold, top-k, per-class NMS, `max_per_row`).
- **Metrics.** The ActivityNet detection AP over tIoU thresholds (`segment_iou`, `interpolated_prec_rec`,
  `match_flags`, `ap_from_flags`, `ap_matrix`) with a clip-bootstrap interval (`resampled_map`), wrapped by
  `make_metrics` under each folder's metric names. This folder reports `tmap`, `tmap_per_tiou`, `n_events`, floor `tmap_floor`.

## The `_sibling()` loader in `eval.py`

Every folder that splits its scorer over several files repeats the same 15-line loader (import a file next to
`eval.py` by path, keyed on its content). A core helper for loading a folder's own modules would remove it.

## Unifying

Moving `detection.py` into the core changes this folder's scorer from its own (`scorer: perception_test_sound_loc`) to a core one, which
changes the scoring identity: a `protocol_version` bump with `supersedes` in the changelog, and a conformance fixture
re-recorded under the new scorer (see the hashing migration rules). Do it for all four folders at once, and keep
the conformance numbers equal as the check that nothing moved.
