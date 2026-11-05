# vggss_box — setup

Read this first. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

The annotations are one public JSON file; the media are VGGSound test clips that you must already have (VGG-SS has
no media download of its own). Every prepare step is a read-only check except step 1's single `curl` when the json
is absent:

```sh
yavih bench prepare vggss_box      # lists the steps (fallback: bash "$(yavih bench path vggss_box)/prepare/<file>")
yavih bench prepare vggss_box 01   # vggss.json into the annotations root
yavih bench prepare vggss_box 02   # check the VGGSound media (manual: stage them first)
yavih bench prepare vggss_box 03   # optional: frame sizes of the staged media vs the manifest (PyAV)
yavih bench prepare vggss_box 04   # optional: make_manifest.py --check (byte identity of the manifest)
yavih bench prepare vggss_box 05   # yavih bench verify vggss_box --census
```

1. `01_annotations_vggss.sh`: `curl -fL -o <annotations_root>/vggss.json
   https://raw.githubusercontent.com/hche11/Localizing-Visual-Sounds-the-Hard-Way/master/metadata/vggss.json`
   (no auth). Expected **620 306 B, sha256 `ae300b4a259267e6fe839970d495862edd70bfe3da63d925bcdb34054ed3b15e`**,
   5 158 entries, 221 classes, 1–7 boxes each (normalised `x1, y1, x2, y2` of the full frame at t = 5.0 s). 12
   boxes carry a corrupt coordinate `-2.8823e16`; the official `max(x, 0)` rule zeroes them, and none is on a frozen
   row. If the sha differs, the upstream file moved: `make_manifest.py` refuses it. The authors' repository states
   no licence; check it before redistributing the boxes. The annotations root defaults to `${data.root}/vggss`
   (`yavih config get vggss_box.annotations_root`); only `make_manifest.py` reads it.
2. `02_vggsound_media.sh`: **manual step**: provide the VGGSound test clips from an existing VGGSound copy. Every
   entry is a VGGSound **test** clip, read in place from the VGGSound tree (CC BY 4.0 listing; YouTube media):
   `<ROOT>/video/<ytid>_<start6>.mp4` (native resolution), `<ROOT>/audio/<ytid>_<start6>.wav` (16 kHz mono),
   `<ROOT>/metadata/vggsound.csv` (the official listing, sha256
   `ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8`). The layout and rel paths are the same as in
   `vggsounder`; without a copy, `vggsounder`'s `prepare/02_download_vggsound.sh` re-downloads from YouTube, where
   clips disappear over time. A re-download's frame sizes may differ: the boxes are fractions, so they survive a
   uniform rescale but not a cut or re-framed clip (step 3 checks the sizes). The script checks the listing's
   sha256 and that all 5 158 `video/<id>.mp4` + `audio/<id>.wav` exist.
3. `03_geometry_sidecars.sh`: checks the two geometry sidecars shipped in `prepare/` (`frame_sizes.json`,
   `video_geometry.json`; sha256-pinned in `make_manifest.py`) and compares PyAV's coded frame size of every frozen
   row with `frame_w`/`frame_h` (needs `av`).
4. `04_make_manifest.sh`: `make_manifest.py --check`, see README.md "Re-freezing the manifest".
5. `05_verify.sh`: `yavih bench verify vggss_box --census`.

Disk / inode budget: the 5 158 clips are 10.54 GB; the frozen rows touch 4 614 clips = **9.35 GB** (mp4 + wav,
9 228 files) inside the much larger VGGSound tree. **Never extract the bulk VGGSound tars in place** next to the
tree. Nothing is materialized at prepare time (`data.derived: {}`).

## Dependencies

None. `external: []`: no original-author package is used. The official VGG-SS evaluation code
(`hche11/Localizing-Visual-Sounds-the-Hard-Way` `test.py`) scores localization heatmaps (cIoU, AUC), not a
regressed box, so it does not apply to this readout; yavih's box metrics (`iou_mean`, `acc@0.5`, `acc@0.3`,
`center_in_gt`) are its own, and only the box sanitisation `max(x, 0)` is transcribed from `test.py`.
`yavih doctor --bench vggss_box` lists no external package.

Prepare-time tools only: `curl` (step 1), Python with numpy (`make_manifest.py`; no torch), and for the optional
checks PyAV (step 3) and soundfile (`make_manifest.py --census`): `pip install av soundfile`. The bridge decodes
with its own stack or with `yavih.media.decode` (same two packages).

## Input contract

What the reader hands each bridge per row (the generic reader `yavih.sample.iter_samples`; the folder ships no
`reader.py`):

- `video`: a `RawStream` over the **stored VGGSound mp4**, byte for byte (`path`, `open()`, `bytes()`); nothing
  decoded, cut, cropped or resized. `info` guarantees `duration_s, fps, width, height, container, bytes` of the
  video track (from the site census written by `yavih bench verify vggss_box --census`, else a header probe).
  Native sizes: 2 231 rows 1280 × 720, 787 480 × 360, 529 640 × 360, 339 320 × 240, and 182 portrait clips
  (406 × 720, 202 × 360, …); 2 825 rows are 16:9, 1 242 are 4:3. Durations ~10 s; 221 rows' video is shorter
  than 10 s, 41 shorter than 9 s, 20 shorter than 8.5 s, 16 (all eval) shorter than 8.0 s, the shortest
  `vhUM-UOKpSk_000053` 5.09 s; every video holds a frame at the annotated 5.0 s.
- `audio`: a `RawStream` over the **stored wav** (16 kHz mono PCM for all 4 614 rows); nothing decoded,
  resampled, cut or padded. `info` guarantees `duration_s, sample_rate, n_samples, channels, container, bytes`.
  487 rows (384 train / 103 eval) are shorter than 10 s, 67 (43 / 24) shorter than 9 s, 37 (15 / 22) shorter
  than 8.5 s; min 4.94 s.
- `meta["facts"] == {"annotated_second": 5.0}`: the box marks the sounding object in the frame at 5.0 s
  (`input_contract.facts`). Nothing else: no box, class name, `frame_w`/`frame_h`, row identity or media key.
- No materialization (`materialize: null`).

What the bridge decides and declares in `describe()["input"]`: which frames (e.g. `explicit` times `1.0 … 8.0` s
with one frame on the annotated second, or `uniform` over the clip), the spatial rule (e.g. Resize(256) +
CenterCrop(224), or the literature's `Resize((224, 224))` squash), the audio span (e.g. `full` padded to 10 s, or
`span` `[1, 9]`), resampling, front end, padding of the short wavs, normalization constants and their source, the
decoder. Pitfalls of this media: mixed and portrait geometry (positions differ per clip → `pos_mode:
per_sample`), short wavs (declare the padding), short videos down to 5.09 s (declare `edge: clamp` or a rule that
fits short media, e.g. `uniform` over the clip).

```yaml
input_contract:
  handed_over:
    video: "the VGGSound mp4 exactly as stored (native size, 1280x720 for 2 231 rows, 182 portrait; ~10 s, 221 rows shorter, min 5.09 s); nothing decoded, cut, cropped or resized"
    audio: "the VGGSound wav exactly as stored (16 kHz mono PCM, ~10 s, 487 rows shorter, min 4.94 s); nothing decoded, cut, resampled or padded"
  info:
    video: [duration_s, fps, width, height, container, bytes]
    audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
  facts: [annotated_second]
  materialize: null
```

## Output contract

- `feats [N, D]`, one fixed `(N, D)` for every row of both splits.
- `pos [N, 4] = (stream, t, y, x)` in **media coordinates**: `stream` indexes `describe()["streams"]`, which must
  contain both `video` and `audio`. `t` = media seconds / that stream's `duration_s`. `(y, x)` = fractions of the
  **ORIGINAL** frame (`info.height`, `info.width`), before any resize or crop: map a crop back with
  `yavih.media.rules.crop_region` (`grid_pos(..., region=…)`). Video tokens need `y, x ≥ 0`. Audio tokens may carry
  `y = x = -1`: every audio-tagged token is a conditioning row.
- Targets: `box` = `x1|y1|x2|y2` in **full-frame fractions** (`target_frame: full_frame`; the scored region is the
  whole frame). The readout bins video tokens into the 14 × 14 cells tiling the full frame by their `(y, x)`; IoU,
  GIoU and `center_in_gt` are computed in full-frame fractions (IoU is invariant under per-axis scaling, so it
  equals pixel IoU). A bridge that cropped leaves the cells it did not see empty.
- Eligibility (validate, `N_ELIGIBILITY`): `pos` has `y`, `x`; both stream tags appear; the median over the
  validate sample of the fraction of the 196 cells holding ≥ 1 video-token centre is **≥ 0.25**; and the median
  over the sample of `min |t · duration_s − 5.0|` over the video tokens with `t ≥ 0` is **≤ 0.5 s**
  (`annotated_frame`; the measured median and min are recorded as `annotated_frame_dt`). Reference: a Resize(256) +
  CenterCrop(224) ViT/16 covers 96 / 196 = 0.49 on 16:9 clips, 120 / 196 = 0.61 on 4:3; median 0.49 over all
  rows, min 0.245 (6 rows). A squash bridge covers 1.0. Pooled features (`N == 1`), a video-only or an audio-only
  bridge, or a 7 × 7 token grid inside a centre crop are recorded `not_applicable` (exit 0), not failed.
- What validate checks per member: `E_TIMES_OUT_OF_MEDIA` (a reported frame time outside `[0, duration_s]` or an
  audio range outside `[0, n_samples]`), `E_RULE_MISMATCH` (`used` vs the declared temporal rule evaluated on the
  member's media facts; a video token centre outside `crop_region` of the declared spatial rule; a token whose
  `t · duration_s` lies outside the frames / audio samples reported in `used`), `E_BATCH_INVARIANCE` (canaries),
  `E_POS_RANGE`, fixed `N`/`D`.

```yaml
output_contract:
  shape: fixed
  coordinates: media
  target_frame: full_frame
  requires:
    pos: [y, x]
    streams: [video, audio]
    grid: {h: 14, w: 14, min_cell_coverage: 0.25}
    t_bins: null
    annotated_frame: {fact: annotated_second, stream: video, tol_s: 0.5}  # a video token within 0.5 s of the annotated 5.0 s frame
```

## Prepare & verify

```sh
# the media root defaults to ${datasets.vggsound}; a per-site override (never edit a shared config):
export YAVIH_DATA__VGGSS_BOX__MEDIA_ROOT=/path/to/vggsound   # or [data.vggss_box] media_root in a site config

yavih bench verify vggss_box --census      # presence, read sample, labels, curation; writes the media-info table
yavih bench verify vggss_box               # later runs
```

Success: `manifest sha ok; labels ok; curation ok; … status verified`, `train present 3700/3700`, `eval present
914/914`, read and decode samples `ok=32 failed=0`. The manifest's decompressed sha256 is
`fbf93c348d424ed48dc598ce80f512bd1aa8b529c60748624c61011d429898db` (`manifest_hash` `4739a39a14b9…`, no classes).

## Gotchas

Bridge-side mistakes that pass validation (or fail it late):

1. **Positions in the original frame.** A bridge that crops and then calls `grid_pos` without `region` claims its
   tokens span the whole frame: every token lands in the wrong cell while the boxes stay put. Declare the spatial
   rule and pass `region = crop_region(spatial, width, height)`; validate's `E_RULE_MISMATCH` then catches a
   mismatch. A `custom` spatial rule (`N_RULE_CUSTOM`) skips that check, so the boxes are only as right as your
   `region`.
2. **Constant positions are wrong here.** Frame sizes (and therefore crop regions) and durations differ per clip:
   use `pos_mode: per_sample`.
3. **`t` is a fraction of the clip's duration**, not of your window: `t = pts / duration_s`. The `grid_pos`
   default `(k + 0.5) / T` is right only for a `uniform` rule over the full clip. With a 2-frame tubelet, average
   each pair. Validate refuses a `t` outside the frames reported in `used` (`E_RULE_MISMATCH`).
4. **Grid coverage.** A 14 × 14 or 16 × 16 patch grid in a centre crop is fine (0.49 on 16:9); a 7 × 7 grid
   (pooled / strided) in a centre crop covers 0.21 (median over rows) → `not_applicable`, and the same grid over
   the squashed full frame covers exactly 0.25. CLS/register tokens: drop them or give them `y = x = -1`.
5. **Crop vs squash are two model readings.** A centre-crop bridge never sees the frame edges (35 rows' GT lies
   entirely outside a Resize(256)+CenterCrop(224), so it scores them as hard cases); a squash bridge sees
   everything distorted. Both are scored against the same full-frame boxes, but they are different configurations:
   give them two `model` names and never quote one as the other.
6. **Short wavs.** 487 wavs end before 10 s (min 4.94 s). A model trained on 10 s needs padding: declare it (`full`
   with `pad_to_s`, or `span` with `pad: zeros`) and report it in `used`. Normalise as the model was pretrained:
   its fixed constants, or per-clip statistics only when they ARE its own recipe (declared in `norm` with their
   `source`); never per-batch statistics.
7. **Short videos.** 20 rows end before 8.5 s (16 eval rows last 5.09 … 8.0 s). A frame time past the last frame
   is `E_TIMES_OUT_OF_MEDIA` under `edge: error`: declare `edge: clamp` (`explicit` `1.0 … 8.0` clamps on those
   rows) or a rule over the clip's own duration (`uniform`, `span: full`).
8. **The annotated frame.** The box is drawn on the frame at 5.0 s (`meta["facts"]`). Unless the median over
   validate's sample of the nearest video token's distance to 5.0 s is ≤ 0.5 s, the feature set is
   `not_applicable` (a frame-0 bridge, or one that averages 16 frames over 10 s into tokens timed elsewhere).
   `explicit` times `annotated_second + (k − n // 2) · step` put one frame on it for any `n`.
9. **Edge boxes.** 566 of 4 614 full-frame boxes touch a frame edge; the sigmoid output never reaches 0 or 1, so a
   small residual IoU loss is structural and identical for every model.
10. **No cross-sample caches.** A decode cache keyed on clip id or path (e.g. reused from `vggsounder`, read with
    another recipe) serves the wrong frames while numbers still look plausible. Decode per sample.
    `RawStream.path` is an opaque handle that differs between canaries and the passes of `bridge check`, so a
    path-keyed cache fails `E_BATCH_INVARIANCE` there.
