# vggss_loc — setup

Read this first. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

The annotations are one public JSON file; the media are VGGSound test clips that you must already have (VGG-SS has
no media download of its own). Every prepare step is a read-only check except step 1's single `curl` when the json
is absent:

```sh
yavih bench prepare vggss_loc      # lists the steps (fallback: bash "$(yavih bench path vggss_loc)/prepare/<file>")
yavih bench prepare vggss_loc 01   # vggss.json into the annotations root
yavih bench prepare vggss_loc 02   # check the VGGSound media (manual: stage them first)
yavih bench prepare vggss_loc 03   # optional: make_manifest.py --check (byte identity of the manifest)
yavih bench prepare vggss_loc 04   # yavih bench verify vggss_loc --census
```

1. `01_annotations_vggss.sh`: `curl -fL -o <annotations_root>/vggss.json
   https://raw.githubusercontent.com/hche11/Localizing-Visual-Sounds-the-Hard-Way/master/metadata/vggss.json`
   (no auth). Expected **620 306 B, sha256 `ae300b4a259267e6fe839970d495862edd70bfe3da63d925bcdb34054ed3b15e`**,
   5 158 entries, 221 classes, 1–7 boxes each (normalised `x1, y1, x2, y2` of the full frame at t = 5.0 s). 12
   boxes carry a corrupt coordinate `-2.8823e16`; the official `max(x, 0)` rule zeroes them. If the sha differs, the
   upstream file moved: `make_manifest.py` refuses it. The authors' repository states no licence; check it before
   redistributing the boxes. The annotations root defaults to `${data.root}/vggss`
   (`yavih config get vggss_loc.annotations_root`); only `make_manifest.py` reads it.
2. `02_vggsound_media.sh`: **manual step**: provide the VGGSound test clips from an existing VGGSound copy. Every
   entry is a VGGSound **test** clip, read in place from the VGGSound tree (CC BY 4.0 listing; YouTube media):
   `<ROOT>/video/<ytid>_<start6>.mp4` (native resolution), `<ROOT>/audio/<ytid>_<start6>.wav` (16 kHz mono),
   `<ROOT>/metadata/vggsound.csv` (the official listing, sha256
   `ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8`). The layout and rel paths are the same as in
   `vggsounder`; without a copy, `vggsounder`'s `prepare/02_download_vggsound.sh` re-downloads from YouTube, where
   clips disappear over time. A re-download's frame sizes may differ: the boxes are fractions, so they survive a
   uniform rescale but not a cut or re-framed clip. The script checks the listing's sha256 and that all 5 158
   `video/<id>.mp4` + `audio/<id>.wav` exist.
3. `03_make_manifest.sh`: `make_manifest.py --check`, see README.md "Re-freezing the manifest".
4. `04_verify.sh`: `yavih bench verify vggss_loc --census`.

Disk / inode budget: the 5 158 clips are 10.54 GB; the frozen rows touch 5 154 clips = **10.53 GB** (mp4 +
wav, 10 308 files) inside the much larger VGGSound tree. **Never extract the bulk VGGSound tars in place** next to
the tree. Nothing is materialized at prepare time (`data.derived: {}`).

## Dependencies

None. `external: []`: the official VGG-SS evaluation (`hche11/Localizing-Visual-Sounds-the-Hard-Way` `test.py` /
`utils.py`: the box rasteriser, the median threshold, cIoU and AUC) is transcribed in the folder's `loc.py`
and pinned against a verbatim copy in the folder's `tests/test_folder.py`; no package is installed. `yavih doctor --bench
vggss_loc` lists no external package.

Prepare-time tools only: `curl` (step 1) and Python with numpy (`make_manifest.py`; no torch). The bridge decodes
with its own stack or with `yavih.media.decode` (PyAV, soundfile: `pip install av soundfile`).

## Input contract

What the reader hands each bridge per row (the generic reader `yavih.sample.iter_samples`; the folder ships no
`reader.py`):

- `video`: a `RawStream` over the **stored VGGSound mp4**, byte for byte (`path`, `open()`, `bytes()`); nothing
  decoded, cut, cropped or resized. `info` guarantees `duration_s, fps, width, height, container, bytes` of the
  video track (from the site census written by `yavih bench verify vggss_loc --census`, else a header probe).
  Native sizes are mixed (mostly 1280 × 720, 480 × 360, 640 × 360, 320 × 240; some portrait clips); durations ~10 s,
  some shorter; every video holds a frame at the annotated 5.0 s. The mp4 carries its own AAC track: the visual
  tokens must not read it (`E_MODALITY_LEAK`).
- `audio`: a `RawStream` over the **stored wav** (16 kHz mono PCM); nothing decoded, resampled, cut or padded.
  `info` guarantees `duration_s, sample_rate, n_samples, channels, container, bytes`. Some wavs are shorter than
  10 s.
- `meta["facts"] == {"annotated_second": 5.0}`: the boxes mark the sounding object in the frame at 5.0 s
  (`input_contract.facts`). Nothing else: no boxes, class name, row identity or media key.
- No materialization (`materialize: null`).

What the bridge decides and declares in `describe()["input"]`: which frame (one at the annotated second; a model
that needs several frames declares them, the tokens it returns are the annotated frame's), the spatial rule (the
literature's `Resize((224, 224))` squash, or Resize(256) + CenterCrop(224)), the audio span (e.g. `full` padded to
10 s), resampling, front end, padding of the short wavs, normalization constants and their source, the decoder.

```yaml
input_contract:
  handed_over:
    video: "the VGGSound mp4 exactly as stored (native size, mixed 16:9 / 4:3 and some portrait; ~10 s, some shorter, every one holds the annotated 5.0 s frame); nothing decoded, cut, cropped or resized"
    audio: "the VGGSound wav exactly as stored (16 kHz mono PCM, ~10 s, some shorter); nothing decoded, cut, resampled or padded"
  info:
    video: [duration_s, fps, width, height, container, bytes]
    audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
  facts: [annotated_second]
  materialize: null
```

## Output contract

A declared schema (`output_contract.schema`), three keys per row, `describe()["outputs"]` lists them:

- `audio_emb [D]` (float16 / float32): the clip's audio embedding in the model's shared audio–visual space (the
  audio projection it retrieves with).
- `video_tokens [N, D]`: the visual tokens of the frame at 5.0 s projected into the **same** space (one per patch;
  `N` may differ per row). Not the encoder's raw tokens: the scorer compares them with `audio_emb` by cosine.
- `video_pos [N, 2]` (float64, exact): `(y, x)` centre of each token in fractions of the **ORIGINAL** frame
  (`info.height`, `info.width`), before any resize or crop: map a crop back with `yavih.media.rules.crop_region`
  (`grid_pos(..., region=…)[:, 2:]`). A token outside `[0, 1]` (CLS, registers at `-1, -1`) is dropped.
- Targets: `boxes` = `x1|y1|x2|y2` quadruples `;`-joined in full-frame fractions; the scorer rasterises their union
  at `protocol.gt_size` 224 as the published evaluator does. The 14 × 14 cells (`readout_params.grid`) tile the
  full frame; a bridge that cropped leaves the cells it did not see empty (they score below every filled cell).
- Isolation (`output_contract.isolation`): `audio_emb` reads `audio` only; `video_tokens` and `video_pos` read
  `video` only. `bridge check` encodes copies with one stream swapped for another clip's and with the mp4's audio
  track blanked; a key that changes fails `E_MODALITY_LEAK`.
- Eligibility: the scorer refuses a set whose median fraction of cells holding a token is below
  `min_cell_coverage` 0.25 (`E_POS_MISSING`). A Resize(256) + CenterCrop(224) ViT/16 covers about half the cells
  on 16:9 clips; a squash bridge covers them all; a 7 × 7 grid inside a centre crop does not qualify.
- What validate checks per member: `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH` (`used` vs the declared temporal
  rule), `E_BATCH_INVARIANCE` (canaries), the codec checks of each key (dtype, shape binding `D` across keys and
  rows, finite values).

```yaml
output_contract:
  shape: fixed
  coordinates: media
  target_frame: null
  requires: {pos: [], streams: [], grid: null, t_bins: null}
  schema:
    audio_emb:    {type: array, dtype: [float16, float32], shape: [D], stats: true}
    video_tokens: {type: array, dtype: [float16, float32], shape: ["*", D], stats: true}
    video_pos:    {type: array, dtype: float64, shape: ["*", 2], exact: true}
  isolation: {audio_emb: [audio], video_tokens: [video], video_pos: [video]}
```

## Prepare & verify

```sh
# the media root defaults to ${datasets.vggsound}; a per-site override (never edit a shared config):
export YAVIH_DATA__VGGSS_LOC__MEDIA_ROOT=/path/to/vggsound   # or [data.vggss_loc] media_root in a site config

yavih bench verify vggss_loc --census      # presence, read sample, curation; writes the media-info table
yavih bench verify vggss_loc               # later runs
```

Success: `manifest sha ok; curation ok; … status verified`, `test present 5154/5154`, read and decode samples
`ok=32 failed=0`. The manifest's decompressed sha256 is `c934591d0b4c820695feaeecf7402fe190501281e52d3d58f7ad7a0ecc39cbce`.

## Gotchas

Bridge-side mistakes that pass validation (or fail it late):

1. **Positions in the original frame.** A bridge that crops and then lays its grid over the whole frame claims its
   tokens span it: every token lands in the wrong cell while the boxes stay put. Declare the spatial rule and pass
   `region = crop_region(spatial, width, height)` to `grid_pos`; with a `custom` spatial rule the boxes are only
   as right as your `region`.
2. **The shared space, both sides.** The audio projection head's output against the visual projection head applied
   to every token. The encoder's pre-projection tokens against a pooled audio vector are a different, unaligned
   space: the map is then noise, `audio_blind` true, and the number reads like a failure of the model.
3. **The mp4's audio track.** Decoding the mp4 with audio and letting a fused encoder see it leaks the audio into
   `video_tokens`; `bridge check`'s blanked-track copy catches it (`E_MODALITY_LEAK`). Decode frames only.
4. **No fused forward.** A model whose visual tokens attend to the audio in one forward cannot be scored here: its
   tokens would change when the audio is swapped (`E_MODALITY_LEAK`). Run the two encoders separately.
5. **The ceiling.** Half the frame is always predicted, so clips whose box union is below a quarter of the frame
   (2 551 of 5 154) cannot reach cIoU 0.5 whatever the map. Do not read a `ciou@0.5` of 0.45 as "45 % localized";
   read `auc` and `ciou_mean` next to it and the floor.
6. **Crop vs squash are two model readings.** A centre-crop bridge never sees the frame edges; a squash bridge sees
   everything distorted. Both are scored against the same full-frame boxes, but they are different
   configurations: give them two `model` names and never quote one as the other.
7. **Short wavs.** Some wavs end before 10 s. A model trained on 10 s needs padding: declare it (`full` with
   `pad_to_s`, or `span` with `pad: zeros`) and report it in `used`. Normalise as the model was pretrained: its
   fixed constants, or per-clip statistics only when they ARE its own recipe (declared in `norm` with their
   `source`); never per-batch statistics.
8. **No cross-sample caches.** A decode cache keyed on clip id or path serves the wrong frames while numbers still
   look plausible. Decode per sample. `RawStream.path` is an opaque handle that differs between canaries and the
   passes of `bridge check`, so a path-keyed cache fails `E_BATCH_INVARIANCE` there.
