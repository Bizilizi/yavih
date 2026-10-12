# k400 — setup

Read this first. `benchmark.yaml` is the machine truth; `README.md` describes the benchmark.

## Obtain the data

`source.kind: torrent`. The release is "Kinetics400 Dataset: The Kinetics Human Action Video Dataset" on Academic
Torrents (uploaded 2020-11-07), <https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26>,
infohash `184d11318372f70018cf9a72ef867e2fb9ce1d26`, 157.31 GB in two files:

| file | bytes | sha256 | content |
|---|---|---|---|
| `kinetics400.zip` | 144 172 278 021 | `335dd19846e60078204b492c725003cfcbbb3f0bbff4dca525916785bd60e30f` | `kinetics400/train_256/`, `kinetics400/test_256/` |
| `kinetics400_validation.zip` | 13 139 875 063 | `33a69475d026e94b3a2f3ed708152b7beaf04a520b4ee455de0c1c883900483b` | `val_256/` |

This is a third-party height-256 re-encode, **not** the ~450 GB CVDF release
(<https://github.com/cvdfoundation/kinetics-dataset>). A CVDF copy has different files, durations and containers,
and its paths do not match the frozen manifest, so using it is a new `protocol_version`. The torrent has no web
seeds and few seeders, so expect the download to take hours. Never scrape YouTube for K400. Annotations are
CC BY 4.0 (DeepMind). The media are YouTube clips, subject to YouTube's terms and the uploaders' rights, for
research use only. No agreement or account is needed.

```bash
yavih bench prepare k400                      # lists the steps
DIR=$(yavih config get data.root)/k400_download
# 0. the .torrent (1 500 770 B; needs web access)
mkdir -p "$DIR" && curl -L -o "$DIR/kinetics400.torrent" \
  https://academictorrents.com/download/184d11318372f70018cf9a72ef867e2fb9ce1d26.torrent
# 1. aria2 download (val zip first, then train), size + sha256 check; zips land in $DIR/kinetics400_dataset/
yavih bench prepare k400 01                   # SKIP_SHA=1 skips the ~1 h sha256 pass
# 2. unzip val_256 -> <media_root>/val, train_256 -> <media_root>/train (test_256 is not extracted)
yavih bench prepare k400 02
# 3. container census + presence of every frozen row + yavih bench verify k400
yavih bench prepare k400 03
# 4. optional, inode-capped filesystems: pack the frozen rows into tar shards with a tar_index
yavih bench prepare k400 04 -- /path/to/k400_wds
```

- `01_download_torrent.sh` needs aria2 ≥ 1.37 (`ARIA2=/path/to/aria2c` if it is not on `PATH`). It downloads with
  public trackers and DHT, checks both sizes and sha256s, and exits 3 on a mismatch. It also runs as a CPU batch
  job (`sbatch`).
- `02_stage.sh` uses `SRC` (the zip directory) and `ROOT` (default `yavih config get k400.media_root`). It checks the
  staged class dirs against `classes.csv`. An interrupted run leaves `<ROOT>/.val-tmp` or `.train-tmp`: delete
  them and run it again (a finished split is skipped).
- `04_pack_webdataset_optional.sh` writes about 1 GiB tars and an `index.json`, then prints how to read them as
  a `tar_index` copy of the folder. That copy has a different `protocol_hash` (Gotchas 10).

Disk and inodes: the zips take 157.3 GB (keep them: they are the provenance record of the media bytes). The staged
train + val trees hold 260 022 files in 800 dirs. The 33 492 frozen files occupy 20.2 GB (`budget.media_gb`). On an
inode-capped filesystem, use the tar shards of step 4 instead of loose files. Nothing is materialized or derived
(`data.derived: {}`).

## Dependencies

None. `external: []`: the Kinetics authors ship no scorer package, so top-1 is yavih's registry metric `acc`, there
is no `probe.py`, and `yavih doctor --bench k400` lists no external package.

Tools outside yavih:
- download and staging: aria2 ≥ 1.37 (e.g. `conda install -c conda-forge aria2`), `unzip`, `sha256sum`, `curl`;
- model env: the bridge's own video decoder (PyAV, decord, torchvision.io, …) or the `yavih[media]` helpers
  (`yavih.media.decode`), declared in `describe()["input"]["video"]["decoder"]`. Without a census table,
  `RawStream.info` comes from a live header probe that needs `yavih[media]`. After one run of
  `yavih bench verify k400 --census`, the model env needs no yavih decoder.

## Input contract

There is one stream, `video` (`kind: video`, `layout: dir` under `media_root`, `required: true`). It is the
Kinetics-400 mp4 **exactly as stored** in the Academic-Torrents `_256` release, handed over as a `RawStream`:
`path` (`<media_root>/<split>/<class>/<clip_id>.mp4`), `bytes()`, `open()` (which also works for a `tar_index`
member) and `info`. Nothing is decoded, cut, resized, padded or normalized. `meta == {"facts": {}}`: the label,
class vocabulary, split and clip id never reach the bridge.

The guaranteed `info` keys are non-null on every row: `duration_s` (the video stream's duration), `fps`
(`avg_frame_rate`), `width`, `height`, `container` and `bytes`. `n_frames` may be null. Facts of the 33 492 frozen
files (PyAV header census):

| fact | value |
|---|---|
| container / codec | mp4 / h264 for all 33 492; one video stream each; 33 479 also carry an audio track; no rotation tags |
| height | 256 for every file |
| width | 138–1 424; mostly 454 or 340; **4 013 portrait** (144, 192, 142 or 170 wide × 256 high), 317 square |
| fps | 6–60; mostly 29.97, 30 or 25; 325 files below 10 fps; some non-integer averages (e.g. 8.667) |
| duration (video stream) | train median 10.0 s (min 0.567), val median 12.0 s (min 0.533); **4 555 < 10 s** (train 2 756, val 1 799) |
| container vs video duration | the container (audio) runs past the last video frame on 1 415 files, by up to 2.62 s |
| frames | 8 (`val/dodgeball/q0X_uY4pyU8.mp4`, 0.53 s at 15 fps) to ~360 |
| frame timing | first frame at pts 0, last at `duration_s − 1/fps`, constant frame rate |

The bridge decides the following and declares them in `describe()["input"]["video"]`:
- **temporal**: which frames, over which span, and how a span longer than the clip is served. `uniform` over
  `span: "full"` reads every clip whole (train ~10 s, val ~12 s). `uniform` over `[0.0, 10.0]` with
  `edge: "clamp"` repeats the last frame on short clips. `edge: "error"` with a fixed span gives
  `E_TIMES_OUT_OF_MEDIA` on the 4 555 short clips. Times are media seconds of the video stream.
- **spatial**: resize and crop of a 256-high frame of any width. A short-side resize upscales portrait clips.
- **norm**: the model's own pretraining constants, with their source. **decoder**: the library and its version.
- The audio track inside the mp4 is not part of the protocol, so `describe()["input"]` names only `video`.

```yaml
handed_over:
  video: >-
    the Kinetics-400 mp4 exactly as stored in the Academic-Torrents _256 release (h264, 256 pixels high,
    138-1424 wide, 4 013 portrait; 6-60 fps; train ~10 s, val ~12 s, 4 555 frozen clips shorter than 10 s,
    min 0.53 s); nothing decoded, cut, resized, padded or normalized
info:
  video: [duration_s, fps, width, height, container, bytes]
facts: []
materialize: null
```

## Output contract

- **Shape**: one fixed `[N, D]` per clip for every row of every split, `float16` or `float32`.
- **Positions**: `clf` requires none, so any feature set with `N ≥ 1` is eligible: pooled (`N = 1`,
  `pos_mode: none`) or token features for the attentive probe. Positions a bridge does write are in media
  coordinates: `t` is the token centre in seconds divided by `info.duration_s` (`-1` for a token centred in padding
  the bridge added), and `(y, x)` are fractions of the ORIGINAL 256-high frame (map a crop back with
  `yavih.media.rules.crop_region` and `grid_pos(..., region=...)`). Clips differ in duration and frame size, so
  positions are `per_sample`.
- **No spatial target** (`target_frame: null`).
- **What validate checks**: `E_TIMES_OUT_OF_MEDIA` (a reported frame time outside `[0, duration_s]` + 1 ms),
  `E_RULE_MISMATCH` (the declared rule re-evaluated on the clip's facts vs the reported `used` times, tolerance
  `0.5 / fps + 1 ms`; token centres outside the declared crop region), `E_POS_RANGE`, `E_BATCH_INVARIANCE`, a fixed
  `N`/`D`, and `E_DEAD_ROWS` / `E_DUPLICATE_CLIPS`.

```yaml
shape: fixed
coordinates: media
target_frame: null
requires:
  pos: []
  streams: []
  grid: null
  t_bins: null
```

## Prepare & verify

```sh
# media_root defaults to ${data.root}/kinetics400; override per site with either
export YAVIH_DATA__K400__MEDIA_ROOT=/path/to/kinetics400
# or in ~/.config/yavih/config.toml:
# [data.k400]
# media_root = "/path/to/kinetics400"

yavih bench verify k400 --census   # presence + per-file media facts (writes the media-info table)
yavih bench verify k400 --json
yavih bridge check my_k400_bridge.py --bench k400 --out k400.profile.json
```

A pass prints `status verified`, `train present 20000/20000`, `val present 13492/13492` and a decode sample with
0 failures. With `--census`, every file's `info` carries the six guaranteed keys. `bridge check` must report a
bit-identical rerun and batch and order invariance.

## Gotchas

These are bridge-side mistakes that pass validation, or fail it late:

1. **Frame times are yours.** Declare the rule (`uniform` or `fps`, span, start, edge). Any declared rule is legal
   and none is required.
2. **Short clips.** 4 555 frozen clips (13.6 %) end before 10 s, 659 of them before 5 s (min 0.53 s). A fixed
   `[0, 10]` span needs `edge: "clamp"` (late times get the last frame, so a 0.53-s clip yields ~2 distinct
   frames out of 16), or it fails with `E_TIMES_OUT_OF_MEDIA`. `span: "full"` spreads the frames over the clip
   instead. Never drop or re-time a clip, and return the same `N` for it. Report the pts of the frames actually
   fed: validate compares times past the last frame (`duration_s − 1/fps`) against that last frame.
3. **Val clips are ~12 s, train ~10 s.** `span: "full"` covers 0–10 s on train and 0–12 s on val (a shift in frame
   spacing between train and eval). A fixed `[0, 10]` span covers the same seconds on both. Both are legal;
   declare which one you use.
4. **Use the video stream's duration.** On 1 415 files the container runs up to 2.62 s past the last video frame.
   Times computed from the container duration fall outside the video. `info["duration_s"]` is the video stream's.
5. **Geometry.** Height is 256 for every file, but widths range from 138 to 1 424 and 4 013 clips are portrait.
   "Resize the short side to 256, centre-crop 224" upscales portrait clips and crops their top and bottom. Use the
   model's own eval transform, paste it into the bridge's preprocessing-source slot, and map positions (if
   written) back to the original frame per clip.
6. **No multi-view TTA.** One clip, one view (`views: 1`). V-JEPA-style 16×8×3 multi-view averaging is not part of
   the protocol.
7. **Label order is frozen, not a sort.** `classes.csv` is the en_US-locale `ls val | sort` order, which differs
   from Python `sorted()` / `LC_ALL=C sort` at 15 indices (`brushing_hair` < `brush_painting`, `pushing_*` <
   `push_up`, `skiing_*` < `ski_jumping`, `spraying` < `spray_painting`). Post-process `eval_logits.npy` with
   `classes.csv`, never with a re-derived vocabulary. Class names are the release's dir names: underscores, with
   `(`, `)` and `'` replaced by `-` in 10 names (e.g. `hurling_-sport-`, `massaging_person-s_head`).
8. **Clip ids follow the release's file names.** Train ids are `<ytid>_<start6>_<end6>` (35 are
   `youtube-<ytid>-<start>-<end>`). Val ids are the bare ytid, with leading `-`/`_` stripped and separator runs
   collapsed. A bridge never keys anything on `clip_id` (`bridge check` scrambles it). No two rows share media.
9. **Dead rows and ignored media.** A model whose features ignore the frames shows up as `E_DUPLICATE_CLIPS` or
   `E_DEAD_ROWS`. Very short clips legitimately give near-constant frame stacks, but their features must still be
   distinct. Reading the file's audio track would be a different benchmark, not a K400 reading.
10. **Layout forks the hash.** `streams` is hashed, so the `tar_index` copy of step 4 has a different
    `protocol_hash` from the packaged `dir` layout, even though every handed-over byte is identical, and `report`
    never merges the two. Extract all readings of a comparison from one layout.
