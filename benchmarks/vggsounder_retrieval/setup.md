# vggsounder_retrieval — setup

Read this first. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

The rows come from the VGGSounder labels (the `vggsounder` 1.0 wheel on PyPI) and the official VGGSound listing,
both frozen into the shipped `manifest.csv.gz`; the media are VGGSound test clips that you must already have. The
folder reads the same VGGSound tree as `vggsounder` and stages nothing of its own. Every prepare step is a
read-only check:

```sh
yavih bench prepare vggsounder_retrieval      # lists the steps (fallback: bash "$(yavih bench path vggsounder_retrieval)/prepare/<file>")
yavih bench prepare vggsounder_retrieval 01   # check the VGGSound listing and the 5 294 clips (manual: stage them first)
yavih bench prepare vggsounder_retrieval 02   # optional: make_manifest.py --check (downloads the vggsounder 1.0 wheel, no install)
yavih bench prepare vggsounder_retrieval 03   # yavih bench verify vggsounder_retrieval --census
```

1. `01_vggsound_media.sh`: **manual step** (`obtain.tier: needs-data-owner`): provide the VGGSound test clips from
   an existing VGGSound copy, read in place (CC BY 4.0 listing; YouTube media): `<ROOT>/video/<ytid>_<start6>.mp4`
   (native resolution, H.264 with its own AAC track), `<ROOT>/audio/<ytid>_<start6>.wav` (16 kHz mono),
   `<ROOT>/metadata/vggsound.csv` (the official listing, sha256
   `ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8`). Layout and rel paths are those of
   `vggsounder`; without a copy, `vggsounder`'s `prepare/02_download_vggsound.sh` re-downloads from YouTube, where
   clips disappear over time (a re-download is new media). The script checks the listing's sha256 and that the
   5 294 frozen clips exist (`VGGSOUND_ROOT=/path` overrides the configured root).
2. `02_make_manifest.sh`: `make_manifest.py --check` (`yavih bench show vggsounder_retrieval make_manifest` prints
   the recipe). Its input is `vggsounder/data/vggsounder.csv` of `vggsounder==1.0` (sha256
   `400979ea3dee34bed145d3fd96966e16689352df63b645c4730caf8eb2cacb06`), fetched with
   `pip download --no-deps --python-version 3.11 --only-binary=:all: vggsounder==1.0` unless
   `VGGSOUNDER_CSV=/path/to/vggsounder.csv` is set. vggsounder 1.0 requires Python >= 3.11: without those two flags,
   pip on Python 3.10 finds only 0.1.x and the step fails. With them it fetches the 3.11 wheel whatever the
   interpreter (only its data file is read, nothing is installed).
3. `03_verify.sh`: `yavih bench verify vggsounder_retrieval --census`.

Disk budget: the frozen rows touch 5 294 clips = **10.72 GB** (9.03 GB mp4 + 1.69 GB wav, 10 588 files) inside the
much larger VGGSound tree. Nothing is materialized (`data.derived: {}`).

## Dependencies

None. `external: []`: no package of the benchmark's authors is imported at scoring time. The scorer is the folder's
own code, per part:

- instance-level ranks (headline): `retrieval.py`, byte-identical to `vggsound_retrieval`'s (the CAV-MAE
  retrieval protocol, rung 3: YuanGongND/cav-mae `src/retrieval.py` @ `68fe8c2`, re-implemented vectorised there
  with the average tie rank; pinned against a naive loop in `tests/test_folder.py`);
- class-level mAP: `class_level.py` `calc_map`, copied (rung 3) from DCMH, jiangqy/DCMH-CVPR2017 @
  `67d0e84c0425fdac3fad30d67d5a2beb5e345cea`, `DCMH_tensorflow/DCMH_tensorflow/utils/calc_hammingranking.py`
  (no licence stated); Hamming distance replaced by cosine similarity, per-query APs returned;
- class-level recall@k: `class_level.py` `c_recall_hits`, copied (rung 3) from Roth et al.,
  Confusezius/Revisiting_Deep_Metric_Learning_PyTorch @ `efddbf23ccbe267f055867b4e1c7c6693e2447c9`,
  `metrics/c_recall.py` (MIT); multi-label relevance, per-query hits returned.

`yavih doctor --bench vggsounder_retrieval` lists no external package. Prepare-time tools only: Python and pip
(`make_manifest.py` and step 02 read the `vggsounder==1.0` wheel's data file; `pip download --no-deps
--python-version 3.11 --only-binary=:all: vggsounder==1.0`, never imported for scoring). The bridge decodes with its own stack or with `yavih.media.decode`
(PyAV, soundfile: `pip install av soundfile`).

## Input contract

What the reader hands each bridge per row (the generic reader `yavih.sample.iter_samples`; the folder ships no
`reader.py`):

- `video`: a `RawStream` over the **stored VGGSound mp4**, byte for byte (`path`, `open()`, `bytes()`); nothing
  decoded, cut, resized or padded. `info` guarantees `duration_s, fps, width, height, container, bytes` of the video
  track (from the site census written by `yavih bench verify vggsounder_retrieval --census`, else a header probe).
  Native sizes are mixed (360p to 720p, some portrait); ~10 s at ~30 fps. The mp4 carries its own AAC track: the
  video embedding must not read it (`E_MODALITY_LEAK`).
- `audio`: a `RawStream` over the **stored wav** (16 kHz mono 16-bit PCM); nothing decoded, resampled, cut or
  padded. `info` guarantees `duration_s, sample_rate, n_samples, channels, container, bytes`. 530 of the 5 294 wavs
  are shorter than 10 s.
- No facts (`input_contract.facts: []`), no class, no row identity or media key, no materialization. The
  `labels_av` column is read by the scorer only.

What the bridge decides and declares in `describe()["input"]`: which frames (the model's own evaluation frames),
the spatial rule, the audio span (e.g. `full` padded to 10 s), resampling, front end, padding of the short wavs,
normalization constants and their source, the decoder.

```yaml
input_contract:
  handed_over:
    video: "the VGGSound mp4 exactly as stored (H.264, native and mixed frame size, ~10 s at ~30 fps, with its own AAC track, which the video embedding must not read); nothing decoded, cut, resized or padded"
    audio: "the VGGSound wav exactly as stored (16 kHz mono 16-bit PCM, nominal 10 s; some files shorter); nothing resampled, trimmed or padded"
  info:
    video: [duration_s, fps, width, height, container, bytes]
    audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
  facts: []
  materialize: null
```

## Output contract

A declared schema (`output_contract.schema`), two keys per row, `describe()["outputs"]` lists them:

- `audio_emb [D]` and `video_emb [D]` (float16 / float32): one clip's two projections into the model's shared
  audio-visual space, the vectors its own objective aligned. Un-normalised is fine: the scorer L2-normalises. `D` is
  bound across both keys and every row.
- Isolation (`output_contract.isolation`): `audio_emb` reads `audio` only, `video_emb` reads `video` only. `bridge
  check` encodes copies with one stream swapped for another clip's and with the mp4's audio track blanked; a key
  that changes fails `E_MODALITY_LEAK`. Two independent single-modality forwards satisfy it; a fused forward does
  not.
- Targets: none in the manifest (`label.kind: none`): the instance-level true pair of a row is the row itself; the
  class-level relevance is the `labels_av` column (`readout_params.classes_column`), read by the scorer.
- What validate checks per member: `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH` (`used` vs the declared temporal rule),
  `E_BATCH_INVARIANCE` (canaries), `E_DEAD_ROWS` / `E_DUPLICATE_CLIPS`, the codec checks of each key (dtype, shape,
  finite values).

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
export YAVIH_DATA__VGGSOUNDER_RETRIEVAL__MEDIA_ROOT=/path/to/vggsound   # or [data.vggsounder_retrieval] media_root in a site config

yavih bench verify vggsounder_retrieval --census      # presence, read sample, curation; writes the media-info table
yavih bench verify vggsounder_retrieval               # later runs
```

Success: `manifest sha ok; labels ok; curation ok; … status verified`, `test present 5294/5294`, read and decode
samples `ok=32 failed=0` (the census probes 10 588 files, about a minute). The manifest's decompressed sha256 is
`7dbc0abb50fae18f30745f2c3083efbd3946ea678cbeea711abe3d26a26df6b9`.

## Gotchas

Bridge-side mistakes that pass validation (or fail it late):

1. **The shared space, not the encoder.** Return the vectors the model's own retrieval evaluation uses: after the
   projection heads, pooled as the contrastive loss pools them. Mean encoder tokens of two towers that never shared
   a loss score at chance.
2. **The mp4's audio track.** Decode frames only; take the audio from the wav. A fused model fed the mp4 leaks the
   audio into `video_emb` (`E_MODALITY_LEAK`).
3. **No fused forward.** Run the two encoders separately, even when the model can run them jointly.
4. **One deterministic view.** Declare the frames you feed (`explicit` times, `edge: clamp` for shorter clips) and
   pool as the model's own evaluation does. Random frame sampling fails `E_RERUN_DRIFT`.
5. **Short wavs.** 530 wavs end before 10 s. Declare the padding (`full` with `pad_to_s`, or `span` with
   `pad: zeros`) and report it in `used`; normalise with the model's fixed constants, never per-batch statistics.
6. **Two galleries.** This folder's 5 294 clips are a different gallery from `vggsound_retrieval`'s 15 440 (and from
   the CAV-MAE 1 545-clip subset): never quote one folder's numbers as the other's.
7. **No cross-sample caches.** `RawStream.path` differs between canaries and `bridge check` passes; a path- or
   clip-keyed cache fails `E_BATCH_INVARIANCE` there or silently serves the wrong frames.
