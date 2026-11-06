# vggsound_retrieval — setup

Read this first. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

The rows come from the official VGGSound listing and the shipped subset list; the media are the VGGSound test clips
that you must already have. Every prepare step is a read-only check:

```sh
yavih bench prepare vggsound_retrieval      # lists the steps (fallback: bash "$(yavih bench path vggsound_retrieval)/prepare/<file>")
yavih bench prepare vggsound_retrieval 01   # check the VGGSound media and listing (manual: stage them first)
yavih bench prepare vggsound_retrieval 02   # optional: make_manifest.py --check (byte identity of the manifest)
yavih bench prepare vggsound_retrieval 03   # yavih bench verify vggsound_retrieval --census
```

1. `01_vggsound_media.sh`: **manual step**: provide the VGGSound test clips from an existing VGGSound copy, read in
   place from the VGGSound tree (CC BY 4.0 listing; YouTube media): `<ROOT>/video/<ytid>_<start6>.mp4` (native
   resolution, H.264 with its own AAC track), `<ROOT>/audio/<ytid>_<start6>.wav` (16 kHz mono),
   `<ROOT>/metadata/vggsound.csv` (the official listing, sha256
   `ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8`). The layout and rel paths are the same as in
   `vggsounder`; without a copy, `vggsounder`'s `prepare/02_download_vggsound.sh` re-downloads from YouTube, where
   clips disappear over time (a re-download is new media). The script checks the listing's sha256 and that all
   15 446 test clips exist.
2. `02_make_manifest.sh`: `make_manifest.py --check`, see README.md "Re-freezing the manifest". The subset list
   `cavmae_sync_subset.csv` (1 545 clip ids, sha256-pinned in `make_manifest.py`) is the `video_id` column of
   `datafiles/vgg_test_5_per_class_for_retrieval_template.json` of `edsonroteia/cav-mae-sync` at commit
   `0add2e7cbfcb2ff0315456b9db7ad347893ba54b` (sha256 `09c577a64d31ad6205dd2494917ed833c4e1123704b16ca32c1379be57828abd`).
3. `03_verify.sh`: `yavih bench verify vggsound_retrieval --census`.

Disk / inode budget: the frozen rows touch 15 440 clips = **31.12 GB** (mp4 + wav, 30 880 files) inside the
much larger VGGSound tree. **Never extract the bulk VGGSound tars in place** next to the tree. Nothing is
materialized at prepare time (`data.derived: {}`).

## Dependencies

None. `external: []`: the ranking (the folder's `retrieval.py`) is a vectorised cosine ranking, pinned
against a naive loop in the folder's `tests/test_folder.py`; the CAV-MAE authors' retrieval script is not installed.
`yavih doctor --bench vggsound_retrieval` lists no external package.

Prepare-time tools only: Python (`make_manifest.py`; no numpy, no torch). The bridge decodes with its own stack or
with `yavih.media.decode` (PyAV, soundfile: `pip install av soundfile`).

## Input contract

What the reader hands each bridge per row (the generic reader `yavih.sample.iter_samples`; the folder ships no
`reader.py`):

- `video`: a `RawStream` over the **stored VGGSound mp4**, byte for byte (`path`, `open()`, `bytes()`); nothing
  decoded, cut, resized or padded. `info` guarantees `duration_s, fps, width, height, container, bytes` of the video
  track (from the site census written by `yavih bench verify vggsound_retrieval --census`, else a header probe).
  Native sizes are mixed (about half 720 × 1280, the rest 360 × 480 / 360 × 640 / 240 × 320, some portrait);
  ~10 s at ~30 fps. The mp4 carries its own AAC track: the video embedding must not read it (`E_MODALITY_LEAK`).
- `audio`: a `RawStream` over the **stored wav** (16 kHz mono 16-bit PCM); nothing decoded, resampled, cut or
  padded. `info` guarantees `duration_s, sample_rate, n_samples, channels, container, bytes`. About a tenth of the
  wavs are shorter than 10 s, down to ~1 s; 120 are silent (an all-zero embedding ties with the whole gallery and
  ranks in its middle).
- No facts (`input_contract.facts: []`), no row identity or media key, no materialization (`materialize: null`).

What the bridge decides and declares in `describe()["input"]`: which frames (the model's own evaluation frames,
e.g. 10 at 1 s or the central ones), the spatial rule, the audio span (e.g. `full` padded to 10 s), resampling,
front end, padding of the short wavs, normalization constants and their source, the decoder.

```yaml
input_contract:
  handed_over:
    video: "the VGGSound mp4 exactly as stored (H.264, native and mixed frame size, ~10 s at ~30 fps, with its own AAC track, which the video embedding must not read); nothing decoded, cut, resized or padded"
    audio: "the VGGSound wav exactly as stored (16 kHz mono 16-bit PCM, nominal 10 s; some files shorter, down to ~1 s); nothing resampled, trimmed or padded"
  info:
    video: [duration_s, fps, width, height, container, bytes]
    audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
  facts: []
  materialize: null
```

## Output contract

A declared schema (`output_contract.schema`), two keys per row, `describe()["outputs"]` lists them:

- `audio_emb [D]` and `video_emb [D]` (float16 / float32): one clip's two projections into the model's shared
  audio–visual space, the vectors its own objective aligned (CAV-MAE: the projection heads' outputs pooled as for
  the contrastive loss). Un-normalised is fine: the scorer L2-normalises. `D` is bound across both keys and every
  row.
- Isolation (`output_contract.isolation`): `audio_emb` reads `audio` only, `video_emb` reads `video` only. `bridge
  check` encodes copies with one stream swapped for another clip's and with the mp4's audio track blanked; a key
  that changes fails `E_MODALITY_LEAK`. Two independent single-modality forwards satisfy it; a fused forward does
  not.
- Targets: none in the manifest (`label.kind: none`): the true pair of a row is the row itself. The subset flag
  `cavmae_sync_5pc` is an `extra` column the scorer reads (`readout_params.subsets`).
- What validate checks per member: `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH` (`used` vs the declared temporal rule),
  `E_BATCH_INVARIANCE` (canaries), `E_DEAD_ROWS` / `E_DUPLICATE_CLIPS` (a constant dimension, identical embeddings
  across clips), the codec checks of each key (dtype, shape, finite values).

```yaml
output_contract:
  shape: fixed
  coordinates: media
  target_frame: null
  requires: {pos: [], streams: [], grid: null, t_bins: null}
  schema:
    audio_emb: {type: array, dtype: [float16, float32], shape: [D], stats: true}
    video_emb: {type: array, dtype: [float16, float32], shape: [D], stats: true}
  isolation: {audio_emb: [audio], video_emb: [video]}
```

## Prepare & verify

```sh
# the media root defaults to ${datasets.vggsound}; a per-site override (never edit a shared config):
export YAVIH_DATA__VGGSOUND_RETRIEVAL__MEDIA_ROOT=/path/to/vggsound   # or [data.vggsound_retrieval] media_root in a site config

yavih bench verify vggsound_retrieval --census      # presence, read sample, curation; writes the media-info table
yavih bench verify vggsound_retrieval               # later runs
```

Success: `manifest sha ok; curation ok; … status verified`, `test present 15440/15440`, read and decode samples
`ok=32 failed=0`. The manifest's decompressed sha256 is `61ecee9919eed820bf02cc50939b5ee7add4e0e9dca4f511c2738dbc4a42d23f`.

## Gotchas

Bridge-side mistakes that pass validation (or fail it late):

1. **The shared space, not the encoder.** The vectors to return are the ones the model's own retrieval evaluation
   uses: after the projection heads, pooled as the contrastive loss pools them. The mean of the encoder tokens of
   two towers that never shared a loss scores at chance and the number reads like a failure of the model.
2. **The mp4's audio track.** Decoding the mp4 with audio and feeding a fused model leaks the audio into
   `video_emb`; `bridge check`'s blanked-track copy catches it (`E_MODALITY_LEAK`). Decode frames only; take the
   audio from the wav.
3. **No fused forward.** A model whose video embedding attends to the audio in one forward cannot be scored here:
   it changes when the audio is swapped (`E_MODALITY_LEAK`). Run the two encoders separately, even when the model
   can run them jointly.
4. **One deterministic view.** The published CAV-MAE numbers average several frames per clip; a bridge declares the
   frames it feeds (`explicit` times, `edge: clamp` for the shorter clips) and pools inside the model as the model's
   own evaluation does. Random frame sampling fails `E_RERUN_DRIFT`.
5. **Short wavs.** About a tenth of the wavs end before 10 s. A model trained on 10 s needs padding: declare it
   (`full` with `pad_to_s`, or `span` with `pad: zeros`) and report it in `used`. Normalise as the model was
   pretrained: its fixed constants, never per-batch statistics.
6. **The two galleries.** `r@1` is over 15 440 candidates, `r@1_cavmae` over 1 545: quote the subset numbers against
   the papers and the full-split numbers against other bridges, never one for the other.
7. **No cross-sample caches.** A decode cache keyed on clip id or path serves the wrong frames while numbers still
   look plausible. Decode per sample. `RawStream.path` is an opaque handle that differs between canaries and the
   passes of `bridge check`, so a path-keyed cache fails `E_BATCH_INVARIANCE` there.
