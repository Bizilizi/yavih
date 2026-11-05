# nyuv2_depth — setup

## Obtain the data

The eval set is public; the GeoNet train frames cannot be fetched by script (manual Google Drive / Baidu links in
the GeoNet README; request access from the dataset authors if they are unavailable). NYU Depth Dataset V2
(Silberman, Hoiem, Kohli, Fergus, ECCV 2012) publishes no explicit licence (academic research, cite the paper);
GeoNet's train frames (Qi et al., CVPR 2018) come via the links in <https://github.com/xjqi/GeoNet> without a data
licence. Treat both as research-only, no redistribution. URLs:
<http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat>,
<https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip>,
<https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>, Probe3D <https://github.com/mbanani/probe3d>.

Stage Probe3D's data layout under `geonet_root`: `nyu_geonet/` (GeoNet `data1.zip` + `data2.zip` merged) and
`nyuv2/` (labelled `.mat` + the FAIR SN40 split files); the recipe is in the header of `prepare/01_locate_mirror.sh`.
The source layout is only read; derived files go into a `derived_root` you own. If a built `derived_root` (PNGs +
targets, 8.2 GB) is available to you, copying it replaces steps 1–5.

| file | bytes | sha256 |
|---|---|---|
| `nyu_geonet/*.mat` (30 916 MATLAB v7 files; `img`, `depth`, `mask`, `norm`, `grid`) | 142 791 955 406 | not hashed per file (12 000 frozen files: 55 163 911 114 B) |
| `nyuv2/nyu_depth_v2_labeled.mat` (MATLAB v7.3 / HDF5) | 2 972 037 809 | `2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa` |
| `nyuv2/test_SN40.json` | 54 282 | `2aee4e157046cc7991161ddfea1a8e8f76b5f9883f519615fcb900b168de6b21` |
| `nyuv2/train_SN40.json` | 65 985 | `694c3ba6a1a2ba712c3a9df59a6fc29e93f39438e197442d474fbb3f81d341e3` |

What the frozen benchmark reads once prepared (`budget.media_gb`): the 12 654 PNGs (5.65 GB) + targets 2.54 GB
under `derived_root`; prepare-time only: the 12 000 frozen `.mat` (55.16 GB, read in place) and the eval index
(1.41 GB). Inodes: 12 654 PNGs + ~8 files.

The numbered scripts in `prepare/` locate, derive, re-container, stage and verify; their header comments carry
the expected sizes and digests:

```bash
yavih bench prepare nyuv2_depth   # lists the prepare steps; each runs with FOLDER and YAVIH_BENCH set
# 1. locate the Probe3D layout (exit 0 usable); prints the config line. MANUAL STEP if absent: build the layout
#    from the upstream recipe in the script header (GeoNet data1/data2.zip, NYU wget, FAIR SN40 zip).
yavih bench prepare nyuv2_depth 01 -- [GEONET_ROOT]
#    expects nyu_geonet/ with 30916 .mat (or the 12000-file subset), labelled .mat 2972037809 B

# 2. derive the eval index from the labelled .mat (h5py; reads images + rawDepths only)
#    SKIP when derived_root already holds the three files with these digests
yavih bench prepare nyuv2_depth 02 -- <GEONET_ROOT>/nyuv2 <DERIVED_ROOT>
#    eval_rgb.npy   602726528 B sha256 019589449649ab4a1e5602854f65746a70b326a4ea84f72a3e6a2dc08d732a39
#    eval_depth.npy 803635328 B sha256 fdbc9659f04c11274b2a72bf8e79ac8069df2559603b181103d237583a401fe5

# 3. optional: copy only the 12 000 frozen .mat files (55 163 911 114 B) out of a full layout
yavih bench prepare nyuv2_depth 03 -- <SRC_ROOT> <DEST_ROOT>

# 4. derive the targets into derived_root (numpy + scipy; + the 16-row torch cross-check when torch imports;
#    ~10 min train, ~25 s eval with 16 workers)
yavih bench prepare nyuv2_depth 04 -- <GEONET_ROOT> <DERIVED_ROOT>
#    train_depth_224.npy 2408448128 B sha256 af4988cbc3f5eefb1e93e5d5a6f6d59793208b49ba2ef8c2156c760df3933b49
#    eval_depth_224.npy   131260544 B sha256 08a16cbafc34647b60f30c10669cc6ae507699161bcba108862f3533162a6888

# 5. re-container the RGB of every row as a lossless PNG into derived_root/rgb/<split>/<clip_id>.png
#    (numpy + scipy; ~10 min with 16 workers)
yavih bench prepare nyuv2_depth 05 -- <GEONET_ROOT> <DERIVED_ROOT>
#    12654 PNGs, 5653048724 B; listing rgb_png.sha256.csv sha256 02639980f22166fea9092decf943e6a6f9e0380e4cc68c381286f9a7a4c4f95e

# 6. optional pre-flight: FULL scipy.io.loadmat of the 12 000 train files, sharded (e.g. 32 shards; expected 0 failures)
yavih bench prepare nyuv2_depth 06 -- <GEONET_ROOT> <OUT_DIR> <SHARD> 32

# 7. verify: "Prepare & verify"
```

Disk: `derived_root` needs ~9.6 GB (PNGs 5.65 GB + targets 2.54 GB + eval index 1.41 GB, symlinks suffice for the
latter); copy only the 12 000 frozen GeoNet files, never the rest of the layout.

**Configure the roots** (precedence: kwargs → env → `[data.<name>]` → yaml default). Defaults:
`derived_root = ${data.root}/nyuv2` (everything the reader and the probe read), `geonet_root =
${data.root}/probe3d_nyuv2_navi` (prepare-time source only):

```bash
export YAVIH_DATA__NYUV2_DEPTH__DERIVED_ROOT=/path/to/nyuv2_derived
export YAVIH_DATA__NYUV2_DEPTH__GEONET_ROOT=/path/to/probe3d_nyuv2   # prepare only
```

```toml
# ~/.config/yavih/config.toml (equivalent)
[data.nyuv2_depth]
derived_root = "/path/to/nyuv2_derived"
geonet_root = "/path/to/probe3d_nyuv2"
```

## Dependencies

Probe3D's depth-metric code, in the **probe env** only (the model env needs nothing beyond yavih core and the
bridge's own decoder). Install it independently — never into yavih's requirements — with exactly:

```bash
pip install loguru -e git+https://github.com/mbanani/probe3d.git@a1f14640076e38b8bc07b66d0fe2d01d15691e9d#egg=evals
```

That is `external[0].pip` (`name: evals`, `version_pin: "1.0"`). The assert-import is
`yavih.external.require("evals", <pip>, "1.0")` followed by `import evals.utils.metrics` and a sha256 check of that
file (`39c0c919f0474b1a7b6d09eedba9dccda19abff921381343250c5058a44a34a8`, commit `a1f14640`; the distribution
version is `1.0` at every commit, so the digest is what pins the code); `yavih doctor --bench nyuv2_depth` shows
it. It needs torch (the probe env has it) and `loguru`. The install MUST be editable (`-e`): Probe3D's `setup.py`
declares `packages=["evals"]` while `evals/` has no `__init__.py`, so a regular `pip install git+…` installs an
empty distribution and `import evals` fails. A missing package or digest makes `probe` exit 3 before training;
nothing is silently re-implemented.

Where that pip line cannot run (a uv venv has no pip; an env you may not install into), clone the commit and
either install the clone editable or put it on `PYTHONPATH` without installing anything:

```bash
git clone https://github.com/mbanani/probe3d.git /path/probe3d
git -C /path/probe3d checkout a1f14640076e38b8bc07b66d0fe2d01d15691e9d
uv pip install loguru -e /path/probe3d            # uv venv: editable from the local clone (dist metadata 1.0)
# or, installing nothing into the env:
export PYTHONPATH=/path/probe3d:$PYTHONPATH        # plus loguru importable (e.g. pip install --target DIR loguru on PYTHONPATH)
```

Without distribution metadata, `yavih.external.require` accepts `evals` only when the directory it imports from
lies in a git checkout whose `HEAD` is the commit of the pip line, `setup.py` says `version="1.0"` and (when
`git` runs) no tracked file under `evals/` is modified; the `metrics.py` sha256 check above still runs. `yavih
doctor --bench nyuv2_depth` then shows `evals 1.0+a1f14640 (git checkout)`. A copy without `.git` is refused (its
commit cannot be checked).

Prepare-time tools: numpy, scipy (GeoNet `.mat`, steps 4–6), h5py (step 2), torch (optional: step 4's
`--check-torch` cross-check). PNGs are written by `prepare/recontainer_png.py` with stdlib `zlib` (no pillow);
pillow is used by `--verify` only when importable.

## Input contract

One stream, `image` (`kind: image`, `layout: dir`, `media_column: rel_media`, `root: derived_root`, `required:
true`). Per row the reader hands over a `RawStream` whose `path` is `<derived_root>/rgb/<split>/<clip_id>.png` and
whose `bytes()` / `open()` return it byte for byte. Nothing is decoded, cropped, resized or normalised; no
statistic is computed.

- **What the PNG is.** 8-bit RGB, 640×480 (every row), no ancillary chunks, Paeth-filtered, zlib level 6,
  written by `prepare/recontainer_png.py`. Train pixels: GeoNet `img` truncated to `[:480, :640]` + channel offsets
  `(244.350, 232.338, 207.016)` in float32, `rint`, clip → uint8 (every pixel lands on an exact integer in
  [0, 254]; Probe3D's recovery). Per-file sha256 and decoded-pixel sha256 are listed in `rgb_png.sha256.csv`
  (pinned in `data.derived`; a gzip copy `rgb_png.sha256.csv.gz` ships in this folder).
- `info` guarantees `width` (640), `height` (480), `container` (`png`) and `bytes`, from the site
  census (`yavih bench verify nyuv2_depth --census`) or a header probe.
- `meta["facts"]` = `{"target_region": [0.0, 0.125, 1.0, 0.875]}`: the part of the frame the depth targets
  cover (Probe3D's 480×480 centre square), in full-frame fractions `[y0, x0, y1, x1]`. It is a fact, not an
  instruction: the bridge may feed the centre square, the whole frame, or anything else.
- The bridge never sees the depth, the split, the manifest or the GeoNet path; `path` encodes the split and
  clip id, which a bridge MUST NOT parse.
- The bridge decides and declares (`describe()["input"]["image"]`): the decoder (`decoder`),
  `spatial` (`resize`, `crop`, `interp`, `antialias`), `norm` (constants + their `source` in the model's own
  recipe), and for a video tower how many identical frames it feeds; for an audio-visual model what its audio
  branch receives (zeros, a mask). An image stream has no `temporal` rule; `used[i]["image"] == {}`.

```yaml
handed_over:
  image: >-
    a lossless PNG (8-bit RGB, 640x480, no ancillary chunks) holding exactly the upstream RGB pixels of the
    frame, re-containered at prepare time because both upstream containers also hold the depth labels: train =
    the GeoNet .mat img truncated to 480x640 with the channel offsets added back and rounded (exact uint8);
    eval = images[i] of nyu_depth_v2_labeled.mat. Nothing cropped, resized or normalised; the depth never
    reaches the bridge
info:
  image: [width, height, container, bytes]
facts: [target_region]
materialize: null
```

## Output contract

`encode` returns `(feats, pos, used)`: `feats` `[B, N, D]` with one `(N, D)` for every row of both splits
(`shape: fixed`); `pos` `[B, N, 4]` (`per_sample`) or one `[N, 4]` (`constant`; legal here because every frame is
640×480); `used[i] = {"image": {}}`.

- **Positions in media coordinates** (`coordinates: media`): `stream = 0` (the tag `image`), `t = -1` (an image
  has no time), `(y, x)` = the token's centre as a fraction of the ORIGINAL 480×640 frame — patch centres mapped
  back through the declared crop with `yavih.grid_pos(1, H, W, 0, region=yavih.media.rules.crop_region(spatial,
  640, 480))`. CLS/register tokens get `(0, -1, -1, -1)` or are dropped.
- **Targets live in full-frame coordinates** (`target_frame: full_frame`): `float32 [n, 224, 224]` metres, the
  nearest-exact raster of `target_region` (rows 0–479, columns 80–559 of the frame), invalid = 0.
- **Eligibility** (`requires`, decided by validate only): `pos: [y, x]`, `streams: [image]`, `grid: {h: 14, w:
  14, min_cell_coverage: 0.5}` — per sampled row, the fraction of the 196 cells tiling `target_region` that
  hold ≥ 1 `image` token centre; eligible iff the median ≥ 0.5. Resize(224) + CenterCrop(224) of the 640×480
  frame covers `target_region` (to one resized pixel: torchvision resizes to 298×224 and crops from column 37,
  `crop_region` = `[0, 0.1242, 1, 0.8758]`; 196/196 cells for a 14×14 patch grid); the whole frame squashed to
  224×224 puts 10 of 14 token columns inside it (140/196 = 0.71 cells); a pooled `N = 1` bridge fills one
  cell and is ineligible (recorded `not_applicable`, exit 0).
- What validate checks: fixed `N`/`D` and dtype (`E_MEMBERS`); `used` with an `image` entry;
  `E_RULE_MISMATCH` — every `image` token centre inside `crop_region(spatial, 640, 480)` when the spatial rule is
  not `custom`; `E_POS_RANGE`; `E_FINITE`; `E_BATCH_INVARIANCE` on the canaries (rel_l2 1e-5, or a declared
  tolerance ≤ 1e-3); `E_DEAD_ROWS`; `E_DUPLICATE_CLIPS`. `bridge check` additionally hard-checks rerun
  bit-identity and batch-/order-invariance.

```yaml
shape: fixed
coordinates: media
target_frame: full_frame
requires:
  pos: [y, x]
  streams: [image]
  grid: {h: 14, w: 14, min_cell_coverage: 0.5}
  t_bins: null
```

## Prepare & verify

**Derived files** (`data.derived`, sha256 pinned): `eval_rgb.npy`, `eval_depth.npy` (step 2), `train_depth_224.npy`,
`eval_depth_224.npy` (step 4), `rgb_png.sha256.csv` (step 5, the listing of the 12 654 PNGs). Target rule
(`derive_targets.py`): truncate `[:480, :640]` → `valid = 0 < d <= 10` on the source → keep `target_region` =
columns `[80, 560)` → depth by torch **nearest-exact** (`src = floor((i + 0.5)·480/224)`), mask by torch
**nearest** (`src = floor(i·480/224)`) — two different index tables, both float32 — → `mask & 0 < d_small <= 10`
→ `where(mask, d_small, 0)`. `--check-torch N` bit-compares the numpy rule with `torch.nn.functional.interpolate`
on N rows per split. PNG rule: see "## Input contract"; `prepare/recontainer_png.py --verify` re-reads every PNG
and checks its sha256 against the listing, falling back to the decoded-pixel sha256 when another zlib build
produced different bytes.

```bash
yavih bench verify nyuv2_depth --census   # the first time: also writes the media-info table
yavih bench verify nyuv2_depth
# or the whole chain (re-freeze check when the GeoNet listing and torch are available + PNG listing check + census):
yavih bench prepare nyuv2_depth 07 -- [GEONET_ROOT] [DERIVED_ROOT]
```

Success looks like (abridged):

```
nyuv2_depth: manifest sha ok; labels ok; curation ok; targets_sha_ok=True; derived_sha_ok=True; status verified
  train    present 12000/12000  missing_first10=[]
  eval     present 654/654  missing_first10=[]
  read sample ok=32 failed=0
  decode sample ok=32 failed=0
  media_info present rows=12654
```

and `recontainer_png.py --verify` reports `checked 12654`, `bad 0`, `missing 0`. A `W_EXTERNAL` warning about
`evals` is expected in the model env (Probe3D belongs to the probe env only).

Then `yavih bridge check <bridge.py> --bench nyuv2_depth --out nyuv2_depth.profile.json` must pass its hard
checks (rerun bit-identical, batch and order invariance); a pooled `N = 1` profile is planned `not_applicable`.

## Gotchas

- **Eligibility is a token grid over `target_region`.** Pooled features (`N = 1`), `pos_mode: none`, or
  positions only in `t` → `features validate` records `eligible: false` and the probe writes `not_applicable`.
  An audio-only bridge raises `NotApplicable`.
- **`y, x` are fractions of the ORIGINAL 640×480 frame**, not of your model input. A bridge that crops the centre
  square and then reports positions as fractions of the crop puts every token in the wrong cell (shifted toward
  the centre) and still passes `E_POS_RANGE`; `E_RULE_MISMATCH` catches it only if the declared crop is right.
  Use `crop_region(spatial, width, height)` as `grid_pos`'s `region`.
- **A second crop loses cells.** Resize-256 + centre-crop-224 on the 640×480 frame shows only rows 0.0625–0.9375
  of the frame and columns 0.17–0.83 (144/196 = 0.73 of the `target_region` cells for 14×14 tokens: eligible,
  but the outer cell rows and columns are blind). Resize(224) + CenterCrop(224) shows the region (to one resized
  pixel).
- **Token order.** `grid_pos` is row-major `(t, y, x)`; a model that flattens `(x, y)` or a transposed patch grid
  passes validation and silently learns a mirrored map. Check one image.
- **CLS / registers**: give them `(stream, -1, -1, -1)` (`grid_pos(..., cls=True)` for one leading CLS) or drop
  them; never assign them a cell.
- **T identical frames.** A video tower fed the replicated still produces T' identical tubelet slices: average
  over time to 14×14, or keep them with `t = -1`; declare the replication.
- **Normalization is the model's**: use its pretraining constants and declare their source. The PNG holds plain
  uint8 RGB; nothing upstream normalized it. Never compute dataset statistics.
- **Silent input.** There is no audio stream. An AV model's audio branch must be fed something (zeros, a mask);
  that choice is the bridge's — declare it in `describe()` (recorded with every number).
- **Do not read the `.mat` files from a bridge.** They hold the depth labels; the reader never opens them and a
  bridge has no path to them (`source_mat` is a manifest column the bridge never sees).
- **Probe env needs `derived_root`** (`Benchmark.targets`, 2.5 GB memmaps) and Probe3D (editable install,
  or a git checkout at the pinned commit on `PYTHONPATH`, "## Dependencies"); a missing or re-derived target
  file with a different sha256 refuses to load.
- **Probe memory and time scale with N.** The dense probe, not extraction, dominates wall time; allow ≥ 1 h per
  seed, and budget RAM by the token count (12 000 images × a few thousand tokens reaches tens of GB).
- **Partial `.mat` reads hide truncation.** `recontainer_png.py` and `derive_targets.py` read with
  `variable_names=[...]`, which a truncated file survives; step 6's full load is the only guard.
