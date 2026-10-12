# in1k — setup

The agent's first read for `in1k`. The law is `benchmark.yaml` (hashed); this file explains it. People-facing
summary: [`README.md`](README.md).

## Obtain the data

`source.kind: manual_portal`. yavih downloads nothing. ImageNet (ILSVRC2012) is distributed by image-net.org
under its terms of access (non-commercial research and education, account and signed agreement, no
redistribution): <https://image-net.org/challenges/LSVRC/2012/>, <https://image-net.org/download-images.php>.
Request access there and download the files below, or use an existing torchvision ImageNet root.

| file | bytes | checksum |
|---|---|---|
| `ILSVRC2012_img_train.tar` | 147 897 477 120 | md5 `1d675b47d978889d74fa0da5fadfb00e` (torchvision) |
| `ILSVRC2012_img_val.tar` | 6 744 924 160 | sha256 `c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0` |
| `ILSVRC2012_devkit_t12.tar.gz` | 2 568 145 | sha256 `b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953` |
| `meta.bin` (written by torchvision) | 303 087 | sha256 `c0c241b17643e8cd3a3f5028d8bd93be950aec0c84fa9bf42079d2c1199c3db3` |

The expected layout is a `torchvision.datasets.ImageNet` root: the tars, `meta.bin`, and the extracted
`train/` (1 000 wnid dirs, 1 281 167 JPEGs) and `val/` (1 000 wnid dirs × 50, sorted by the devkit ground
truth). The frozen manifest touches 100 000 of those JPEGs: **12 432 228 705 B (12.4 GB)**, 100 000 inodes
(`budget.media_gb: 12.43`). Disk budget: full root 155 GB + 1 331 167 inodes + 2 000 dirs; the 100 000-file
subset 12.4 GB. There is no prepare-time transform. The `prepare/` scripts' header comments carry the expected
sizes and digests. Never extract tars in place next to copies you do not own (`02` refuses to write over an
existing `train/`).

```bash
yavih bench prepare in1k   # lists the steps
# 1. locate the torchvision root (exit 0 usable / 2 tars only / 1 nothing); prints the config line to use
yavih bench prepare in1k 01 -- [ROOT]
# 2. only with tars and no extracted root: build a torchvision root in a NEW directory
#    (checks the val/devkit sha256 and the train md5, then torchvision.datasets.ImageNet extracts)
yavih bench prepare in1k 02 -- <TARS_DIR> <NEW_ROOT>
# 3. optional: stage only the 100 000 frozen files (rel_media column of manifest.csv.gz) into a new root
yavih bench prepare in1k 03 -- <SRC_ROOT> <DEST_ROOT>
# 4. verify: "Prepare & verify"
yavih bench prepare in1k 04 -- [ROOT]
```

Set `media_root` to the root (it must contain a populated `val/`):

```bash
export YAVIH_DATA__IN1K__MEDIA_ROOT=/path/to/imagenet
```

```toml
# ~/.config/yavih/config.toml (equivalent)
[data.in1k]
media_root = "/path/to/imagenet"
```

## Dependencies

None. `external: []`: no authors' package is installed or called; `yavih doctor --bench in1k` lists no
external package.

Prepare-time tools: `rsync` (step 3); `torchvision >= 0.9` and pillow (step 2 only). Model side: whatever
decoder the bridge declares (`pillow` in `bridge_template.py`; `yavih.media.decode.image` needs `yavih[media]`).

## Input contract

One stream, `image` (`kind: image`, `layout: dir`, `media_column: rel_media`, `root: media_root`). Per row the
reader hands over a `RawStream` whose `path` is the stored file `<media_root>/{train,val}/<wnid>/<file>.JPEG`
and whose `bytes()` / `open()` return it byte for byte. Nothing is decoded, colour-converted, resized or
cropped.

- `info` guarantees `width`, `height` (the ORIGINAL stored size in pixels), `container` and `bytes`, from the
  census (`yavih bench verify in1k --census`) or a header probe.
- `meta["facts"]` is `{}`.
- `path` contains the wnid directory, i.e. the label: a bridge MUST NOT parse `path`.
- The bridge decides and declares (`describe()["input"]["image"]`): the decoder and its colour conversion
  (`decoder`), `spatial` (`resize`, `crop`, `interp`, `antialias`) and `norm` (constants and their `source` in
  the model's own recipe). An image stream has no `temporal` rule; `used[i]["image"] == {}`.

The stored files (header census of the 100 000 frozen files):

| | train (50 000) | val (50 000) |
|---|---|---|
| container (pillow `format`) | JPEG 50 000 | JPEG 50 000 |
| colour mode | RGB 49 267, greyscale `L` 731, CMYK 2 (`n02077923_14822`, `n04596742_4225`) | RGB 49 100, `L` 899, CMYK 1 (`ILSVRC2012_val_00019877`) |
| width min / median / max | 29 / 500 / 4 752 | 54 / 500 / 4 288 |
| height min / median / max | 41 / 375 / 5 935 | 56 / 375 / 5 005 |
| most common sizes | 500×375 (11 576), 500×333, 375×500, 333×500, 500×334 | 500×375 (15 487), 375×500, 500×333, 500×500, 500×334 |
| landscape / portrait / square | 35 620 / 12 786 / 1 594 | 34 757 / 13 252 / 1 991 |
| short side < 224 / < 256 | 3 973 / 5 283 | 903 / 1 373 |
| bytes | 5 725 668 386 | 6 706 560 319 |

The known PNG-inside-`.JPEG` file of the full train set (`n02105855_2933`) is not in the frozen draw.

```yaml
handed_over:
  image: >-
    the ILSVRC2012 file exactly as stored in the torchvision root (train/<wnid>/<wnid>_<n>.JPEG,
    val/<wnid>/ILSVRC2012_val_<8d>.JPEG; sizes and aspect vary per row; mostly RGB JPEG, a few
    greyscale or CMYK); nothing decoded, colour-converted, resized or cropped
info:
  image: [width, height, container, bytes]
facts: []
materialize: null
```

## Output contract

`encode` returns `(feats, pos, used)`: `feats` `[B, N, D]` with one `(N, D)` for every row of both splits
(`shape: fixed`; resize/crop to a fixed grid inside the bridge, never pad to the batch max). `clf` has no
positional requirement: pooled `N = 1`, a CLS token or a token grid are all eligible, and `pos_mode: "none"`
is fine. Positions, if written, are in media coordinates: `t = -1` (an image has no time axis), `(y, x)` =
fraction of the ORIGINAL image, i.e. patch centres mapped back through the declared crop with
`yavih.grid_pos(..., region=yavih.media.rules.crop_region(spatial, width, height))`; since image sizes vary,
that means `pos_mode: "per_sample"`.

Validate checks fixed `N`/`D` and dtype (`E_MEMBERS`), `used` with an `image` entry (`E_USED`),
`E_RULE_MISMATCH` (every `image` token centre inside `crop_region(spatial, width, height)` when the spatial
rule is not `custom`), `E_POS_RANGE`, `E_FINITE`, `E_BATCH_INVARIANCE` on the canaries, `E_DEAD_ROWS` and
`E_DUPLICATE_CLIPS`.

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

```bash
yavih bench verify in1k --census    # or: yavih bench prepare in1k 04 -- [ROOT]
```

Success: `status verified`, `train present 50000/50000`, `val present 50000/50000`, `decode sample ok=32
failed=0`, `media_info present rows=100000`, `media_fingerprint b57a811274e3…`. The `--census` run probes every
file once (minutes on networked storage); later runs take seconds. `val present 0/50000` means the root's
`val/` is unpopulated (Gotchas 1). Reading is I/O-bound: with a JPEG decode, expect tens of images per second
per node on networked storage.

Then `yavih bridge check <bridge.py> --bench in1k --out in1k.profile.json` must pass its hard checks (rerun
bit-identical, batch and order invariance).

## Gotchas

1. **Empty `val/`.** A root with the val tar but no extracted `val/<wnid>/` (a common partial copy) shows
   `val present 0/50000` in `bench verify`, and the probe refuses. Extract val (step 2) or point `media_root`
   at a complete root. `yavih bench fetch` copies the folder, not media.
2. **Colour mode.** Decode with an explicit RGB conversion (pillow `convert("RGB")` or your stack's
   equivalent) and declare it in `decoder`: 1 630 greyscale and 3 CMYK files are in the frozen rows. A stack
   that returns 1 or 4 channels crashes or silently mis-normalises (CMYK→RGB conversions differ between
   libraries).
3. **Small images.** 4 876 frozen images have a short side under 224 px (minimum 29 px): a resize-short rule
   upsamples them; a crop-without-resize rule would need per-image padding, which the bridge must declare.
4. **Geometry is yours, declare it.** A common choice is `Resize(256) → CenterCrop(224)`. Whatever you pick
   goes in `describe()["input"]["image"]["spatial"]`, recorded with every number. Two crops shown side by side
   are two `model` names; one name re-run with another crop is one report row, newest shown
   (`W_BRIDGE_CHANGED`).
5. **Normalisation.** Use the model's own preprocessing (ImageNet mean/std, [-1, 1], …; paste it into the
   bridge docstring and declare `norm.source`): features of a mean/std model fed [-1, 1] inputs are garbage that
   still passes every validator. Never compute mean/std over the batch or the dataset (`E_BATCH_INVARIANCE`).
6. **Replicate-to-T for video towers.** A video encoder needs `[3, T, H, W]`: repeat the image T times inside
   the bridge and declare it as a TOP-LEVEL `describe()` key (`bridge_template.py`:
   `"image_as_video": {"frames": T, "how": "repeat"}` next to `"input"`, not inside `input.image`, which
   accepts only `decoder`, `norm` and `spatial` and refuses other keys with `E_INPUT`). Compute scales with T
   for no information gain; image tokens have `t = -1`.
7. **Positions through the crop.** If you emit a token grid, map patch centres through `crop_region` of your
   declared spatial rule; `grid_pos` without `region` claims the full image and fails `E_RULE_MISMATCH` for a
   centre crop.
8. **Label order.** The logit vocabulary is `classes.csv` `index` = sorted wnids. Never derive it from
   `meta.bin`, `synset_words.txt` or alphabetical class names.
9. **Shards.** 100 000 rows: use many small shards (`plan` default); decode workers (= cpus − 1) matter more
   than the GPU for small models (`Bridge.decode_workers`).
