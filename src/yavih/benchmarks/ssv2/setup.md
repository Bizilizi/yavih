# ssv2 — setup

The agent's first read for this benchmark: how to obtain the data and what the bridge receives and must
return. `benchmark.yaml` (hashed) is the contract.

## Obtain the data

`source.kind: manual_portal`. Something-Something v2 (TwentyBN; distribution taken over by Qualcomm) is a
free download after registration and licence acceptance on the Qualcomm developer portal:
<https://developer.qualcomm.com/software/ai-datasets/something-something> (paper: Goyal et al. 2017,
<https://arxiv.org/abs/1706.04261>). Research use under the portal's terms, no redistribution — read the
current terms there. **A human must accept the licence**; no agent may work around it. The Hugging Face repo
`HuggingFaceM4/something_something_v2` is not gated but hosts **no media** (loading script only, pointing
back at the portal).

Upstream files of the official release (`source.bytes` = 19 450 115 851 B = the two parts + the labels zip):

| file | bytes | sha256 |
|---|---|---|
| `20bn-something-something-v2-00` | 10 000 000 000 | `aab5bf3badcdf58aed2ac4b6c219a910ea97069b346bcb3ab19202b42dc8f0aa` |
| `20bn-something-something-v2-01` | 9 444 975 475 | `65b505fd26e7e72b745421d33508a5b5fcb7544c6b5510e39aa754282bad8d24` |
| `20bn-something-something-download-package-labels.zip` | 5 140 376 | `8e9859f2862c3626dd1cd013dbc0bccccc12ab48d7c18812e3f302988d6f2c0c` |
| `labels/labels.json` | 9 234 | `d98d8b657a5d418d7c05dbf630dedd34763025b3d764ceef19bbfd1dc654f54f` |
| `labels/train.json` | 26 042 001 | `b24e46090960a49d732a0d3633fd939030b4eb717e16728871a3180b8704192b` |
| `labels/validation.json` | 3 733 168 | `fbdd69ac72b40b9e3ed3aaf484e1575777152565caed693b0ced2d4959fd3958` |
| `labels/test.json` | 448 045 | `60702636202db04cf60c76c76905d821b8fb281376fc1ee4e140087afe108560` |
| `labels/test-answers.csv` | 1 255 143 | `82fe3127545b172d5e1a92641e5192bccfbea689d4b793d79f091239ce5cd2ad` |

The two parts are byte slices of ONE gzip'd tar (`cat … | tar -xz`) that yields a flat
`20bn-something-something-v2/` with 220 847 webm (train + val + test, 19 GB). Older 20BN distributions split
the same tar into more parts (`-00` … `-19`); trust the extracted count, not the part count. This folder reads
loose files only (`layout: dir`): a WebDataset or other repacking must be unpacked first.

What the frozen manifest touches: 44 777 webm, **4 020 523 417 B (4.02 GB)**, 44 777 inodes
(`budget.media_gb: 4.02`; train 1.84 GB, val 2.18 GB). The 27 157 test clips are on disk but not in the
manifest. Disk/inode budget: download 19.5 GB + extracted 19.4 GB + 220 849 inodes in ONE flat directory (full
release), or 4.02 GB + 44 777 inodes (frozen subset). **Never extract inside a directory you do not own**;
`02` extracts into a temp dir and renames after the count check.

```bash
yavih bench prepare ssv2   # lists the prepare steps; each runs with FOLDER and YAVIH_BENCH set (fallback: bash "$(yavih bench path ssv2)/prepare/<file>")
# 1. MANUAL: a human downloads from https://developer.qualcomm.com/software/ai-datasets/something-something
#    into a new DL_DIR; then verify sizes + sha256:
yavih bench prepare ssv2 01 -- DL_DIR
# 2. cat parts | tar -xz into ROOT/20bn-something-something-v2 (count must be 220847) + labels zip → ROOT/labels
yavih bench prepare ssv2 02 -- DL_DIR ROOT
# 3. (alternative to 1-2) copy an already-extracted release into a root you own: rsync into
#    DEST_ROOT/.20bn-tmp, count check, atomic rename (~2 h for 220 847 files over NFS; run it as a batch job)
yavih bench prepare ssv2 03 -- SRC_ROOT DEST_ROOT
#    small disks: only the 44 777 frozen files (4.02 GB)
yavih bench prepare ssv2 03 -- SRC_ROOT DEST_ROOT --frozen-only
# 4. verify + site census: "Prepare & verify"
```

**Configure the root** (precedence: kwargs → env → `[data.<name>]` → yaml default). The yaml default is
`${data.root}/ssv2`; otherwise set, in your own shell or your own config:

```bash
export YAVIH_DATA__SSV2__MEDIA_ROOT=/path/to/ssv2      # holds 20bn-something-something-v2/ (+ labels/)
```

```toml
# ~/.config/yavih/config.toml (equivalent)
[data.ssv2]
media_root = "/path/to/ssv2"
```

## Dependencies

None. `external: []` — the SSv2 release ships no loader, scorer or metric code; top-1 accuracy is yavih's
registered `acc`, so nothing has to be installed independently for the probe.

Tools, all outside yavih's requirements:

- Model env: whatever decodes vp9 webm for your bridge — the model's own stack (decord, torchvision.io, PyAV)
  or `yavih[media]` (PyAV with an FFmpeg that has the vp9 decoder; `yavih.media.decode.frames_at`). Declare
  it in `describe()["input"]["video"]["decoder"]`.
- Site census (`yavih bench verify ssv2 --census`): `yavih[media]` (PyAV) once per site, to write the
  media-info table; afterwards the model env needs no yavih decoder for the facts.
- `prepare/*.sh`: bash, coreutils (`sha256sum`, `stat`), `tar` + `gzip`, `rsync`, `unzip`.

## Input contract

One stream per row, `video` (kind `video`, `layout: dir`, `media_column: rel_media` =
`20bn-something-something-v2/<id>.webm`, `root: media_root`, `required`): a `RawStream` over the stored
webm, byte for byte (`path`, `bytes()`, `open()`). Nothing is decoded, cut, re-timed, resized or cropped.

- **Guaranteed `info` keys** (non-null on every row): `duration_s` (the container duration; webm carries no
  per-stream duration — on all 44 777 clips it equals the last frame's pts + 1/12 s, measured from 0, so a
  leading offset is inside it), `fps` (`avg_frame_rate`, 12.0 on every clip), `n_frames` (webm headers hold
  no frame count: this is the site census's demuxed packet count, equal to the decoded frame count on every
  clip checked — **run `yavih bench verify ssv2 --census` once per site**, else a live probe cannot supply
  it), `width`, `height`, `container` (`webm`), `bytes`.
- **`meta["facts"]`**: empty — SSv2 has no label facts a bridge may use.
- **Materialization**: none (`materialize: null`).
- **What the bridge decides and declares** (`describe()["input"]["video"]`): how many frames and
  at which times (`uniform` over `span: full` = bin centres `(k + 0.5)·D/n`;
  `index_linspace` = `round(linspace(0, n_frames − 1, T))`; `fps` windows), how short clips are
  served (repeat the nearest frame; never pad to the batch maximum), resize, crop, interpolation,
  normalisation constants with their source, the decoder.

The media (census of the 44 777 frozen clips, every packet demuxed):

| fact | value |
|---|---|
| codec / container | vp9 in webm on every clip; no audio track |
| frame rate | `avg_frame_rate` 12.0 on every clip; frame pts on a 1/12-s grid, but not every grid slot holds a frame |
| frame size | height 240 on every clip; width 240–483 in 30 values: 427 (26 557 clips, 59 %), 320 (9 011, 20 %), 360 (4 336, 10 %), 424 (1 358), 293 (1 335), 436 (597), 432 (455), …, 240 (63) |
| duration `D` | 0.833–14.166 s; p01 1.583, median 4.0, p99 6.0; 1 275 clips < 2 s, 199 > 6 s (train median 3.916, max 6.333; val median 4.166, max 14.166) |
| frames `n_frames` | 5–142, median 48 |
| leading offset | 1 315 clips start at pts > 0: 0.083 s (1 101 clips), 0.167 (70), 0.25 (30), 0.5 (21), 0.75 (21), …, 1.417 s (12) |
| holes | 2 762 clips have an inter-frame gap > 1.5 frames (largest 3.083 s; p99 of the per-clip largest gap 0.25 s); 3 769 clips hold fewer frames than `D · 12` (shortfall up to 4.58 s) |
| errors | 0 open/demux/decode errors |

Pitfalls of this media: frame times are NOT `k / fps` on the 3 769 irregular clips, and a requested time can
sit in a hole or before the first frame (see "Gotchas"); widths vary, so the same crop covers a different
fraction of each clip.

```yaml
handed_over:
  video: >-
    the SSv2 webm exactly as stored (vp9, 12 fps nominal with leading offsets and holes in ~8 % of clips,
    height 240, width 240-483, no audio track, 0.83-14.17 s, median 4.0 s), byte for byte; nothing
    decoded, cut, re-timed, resized or cropped
info:
  video: [duration_s, fps, n_frames, width, height, container, bytes]
facts: []
materialize: null
```

## Output contract

- `feats`: `[N, D]` per row, one fixed `(N, D)` for every row of both splits (`shape: fixed`); any token
  layout is eligible for `clf`, pooled `N = 1` included. `float16` or `float32`, finite.
- `pos`: not required (`requires: {pos: [], streams: []}`); `pos_mode` `none`, `constant` or `per_sample`.
  When written, in **media coordinates** (`coordinates: media`): `t` = the token's time / `duration_s` of
  that clip, `(y, x)` = fractions of the ORIGINAL 240-pixel-high frame before any resize or crop. Because
  widths differ, a centre crop's region differs per clip → `per_sample` (`grid_pos(..., region=crop_region(
  spatial, width, height), t_centres=...)`, as in `bridge_template.py`); `constant` pos is legal only
  without `y`/`x`/`t`.
- No spatial target (`target_frame: null`); labels are the manifest's `label` column, never seen by the
  bridge.
- `used`: per clip `{"video": {"times": [...]}}` — the pts of the frames actually fed, in feed order.
- What validate checks: `E_TIMES_OUT_OF_MEDIA` (a reported time outside `[0, duration_s]`, 1 ms
  slack), `E_RULE_MISMATCH` (the declared temporal rule re-evaluated on each row's `info` vs `used`, within
  `0.5 / fps + 1 ms`; token centres outside `crop_region` of the declared spatial rule),
  `E_BATCH_INVARIANCE` (canaries, `rel_l2 ≤ 1e-5` unless a looser tolerance is declared), `E_POS_RANGE`,
  `E_FINITE`, `E_DEAD_ROWS`; `bridge check` adds rerun bit-identity and batch/order invariance as hard
  checks. On the 3 769 irregular clips the nearest stored frame can lie farther than `0.5 / fps` from the
  rule's time (`uniform` n = 8 on 1 885 clips, n = 16 on 2 495; `index_linspace` on all 3 769), so
  `E_RULE_MISMATCH` can fire there for a correct bridge; a `custom` temporal rule is the declared way around it.
- Eligibility (`N_ELIGIBILITY`): always eligible for `clf` with a well-formed feature set.

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

Nothing to derive. Verify the staged media and write the site census once per site (`n_frames` exists only
through it):

```bash
yavih bench verify ssv2 --census
```

Success: `status verified`, train 20000/20000 and val 24777/24777 present, read and decode samples with
`failed=0`, `media_info present rows=44777`. Later runs without `--census` take seconds.

Then `yavih bridge check <bridge.py> --bench ssv2 --out ssv2.profile.json` must pass its hard checks
(rerun bit-identical, batch and order invariance) and report your declared temporal rule in `rules_checked`.
Frozen manifest sha256: `80def82d6f17565978e963312f048e92732dcb85fca2626bd26c462cc4ec4777`.

## Gotchas

Bridge-side mistakes that pass validation (or fail it for the wrong reason) on ssv2:

1. **Synthesizing frame times from indices.** On the 3 769 irregular clips (1 315 with a leading offset up to
   1.417 s, 2 762 with holes up to 3.08 s) frame `i` is not at `i / 12`. Report the pts of the frames you fed
   (`frames_at` returns them), never `idx / fps`; `index_linspace` assumes `t = idx / fps` and is off on
   exactly those clips. Do not "fix" offsets by shifting times.
2. **Re-timing per model family.** Declare one temporal rule and use it for every clip; the rule is recorded
   with every number, so a hidden per-checkpoint switch (e.g. a fixed-fps window for some models) makes
   `describe()` wrong.
3. **Short clips** (0.83–14.17 s; `n_frames` down to 5): 16 frames from a 0.83-s clip repeat frames (nearest
   pts). That is a legal, declared choice; do not pad, loop or resample to a fixed duration, and never pad to
   the batch maximum (batch invariance).
4. **Geometry.** Frames are 240 pixels high with widths 240–483; resize to the model's eval geometry and
   centre-crop to a FIXED grid so `N` is constant. A short side of 256 is an **upscale** of 240p — legal, but
   declare it. A 224 centre crop keeps the middle 56 % of a 427-wide clip's width at short side 224 (49 % of
   the width and 87.5 % of the height at 256); pos `(y, x)` must say so (`crop_region`), or `E_RULE_MISMATCH`
   fires on the spatial check.
5. **Normalisation scope.** Use the model's fixed statistics with their source (e.g. `x*2-1` when that is the
   model's pretraining convention) — never per-batch or per-clip statistics.
6. **`pos` time axis.** `t` is a fraction of the clip's own `duration_s`, which includes a leading offset;
   with a tubelet of `ts` frames pass `t_centres` = the mean of each group's reported pts / `D`. `grid_pos`'s
   default `(k + 0.5) / T` is right only for `uniform` over `span: full` on a regular clip. A `t` outside the
   frames reported in `used` is `E_RULE_MISMATCH` (validate, `bridge check`).
7. **Label order.** The logit vocabulary is `classes.csv` `index` = `labels.json` order; labels never reach the
   bridge. 12 class names contain commas (quoted CSV) — never split cells on `,` by hand.
