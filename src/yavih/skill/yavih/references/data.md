# Data reference — obtainability, budget, data roots, inode rules, failure modes

Each benchmark's `setup.md` is the first read and the authority on how to obtain and prepare its
data (`yavih bench info <b> --setup` prints it; §5 below lists them). This file is the overview:
what is obtainable where, how big it is, what the reader hands over, how data roots are
configured, and the failure modes the rules guard against. Every row count and
`budget.media_gb` below is the value of that folder's frozen `benchmark.yaml` (pinned against it by
`tests/test_skill.py`; `yavih bench list` / `yavih bench info <b>` are authoritative). yavih never
downloads media: `bench fetch` copies a benchmark folder, `bench verify` checks presence and, with
`--census`, writes the media-info table (facts per file: duration, fps, frame count, sample rate,
size) that lets the model env run without a yavih decoder. Without a census row, a stream's facts
come from a live header probe of the same file the first time they are read: slower, still exact
(the census probes files by path, bounded in memory, with a progress line every 1 000 cells; the
full avsync census takes ~4.5 min at ~350 MB). Where your site keeps the data is your site
config (§2), never this file; any path below is an example to replace.

## 1. Obtainability + budget

Who must act first is each folder's `obtain.tier` (`yavih bench list`, the `obtain` column below): `auto` (a script
can fetch it unattended), `manual-download` (a person downloads the files by hand from an official portal or the
authors' links, possibly after a free registration or click-through terms; no data owner involved), `needs-licence`
(a human signs a licence / terms-of-access agreement for their own account), `needs-data-owner` (decayed YouTube media
or private files: request access from the dataset authors or whoever holds them), `site-only`. A benchmark you cannot obtain is recorded and skipped (SKILL.md
"When blocked"). The table is generated from each folder's frozen `benchmark.yaml`; what exactly is handed over,
the sizes of the upstream release and the layout rules are in that folder's `setup.md`.

<!-- yavih:bench-table data begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
| benchmark | readout | source | media touched (frozen) | frozen rows | handed over | obtain | how to stage it |
|---|---|---|---|---|---|---|---|
| audioset20k | clf (multi) | `public_url`; 66.1 GB = `source.bytes` | 66 GB = `budget.media_gb` | 17 726 train / 16 223 eval | video + audio | auto | `yavih bench info audioset20k --setup` |
| ave | seq (segments) | `public_url`; 13.4 GB = `source.bytes` | 4.02 GB = `budget.media_gb` | 2 624 train / 316 test | video + audio | auto | `yavih bench info ave --setup` |
| avsync | clf (floor role) | `derived`; 407.0 GB = `source.bytes` | 74.4 GB = `budget.media_gb` | 24 000 train / 3 000 test / 5 997 stationary | video + audio (materialized at prepare time) | needs-data-owner | `yavih bench info avsync --setup` |
| esc50 | clf (folds) | `public_url`; 0.6 GB = `source.bytes` | 0.88 GB = `budget.media_gb` | 2 000 in folds 1–5 (400 each) | audio | auto | `yavih bench info esc50 --setup` |
| fsd50k | clf (multi) | `public_url`; 24.7 GB = `source.bytes` | 22.9 GB = `budget.media_gb` | 20 000 train / 10 231 eval | audio | auto | `yavih bench info fsd50k --setup` |
| in1k (aliases `imagenet1k`, `imagenet`) | clf | `manual_portal`; 154.6 GB = `source.bytes` | 12.43 GB = `budget.media_gb` | 50 000 train / 50 000 val | image | needs-licence | `yavih bench info in1k --setup` |
| k400 | clf | `torrent`; 157.3 GB = `source.bytes` | 20.2 GB = `budget.media_gb` | 20 000 train / 13 492 val | video | auto | `yavih bench info k400 --setup` |
| nyuv2_depth | grid | `manual_portal`; 145.8 GB = `source.bytes` | 8.2 GB = `budget.media_gb` | 12 000 train / 654 eval | image | manual-download | `yavih bench info nyuv2_depth --setup` |
| ssv2 | clf | `manual_portal`; 19.5 GB = `source.bytes` | 4.02 GB = `budget.media_gb` | 20 000 train / 24 777 val | video | needs-licence | `yavih bench info ssv2 --setup` |
| vggsounder | clf (multi_sets) | `public_url`; 407.0 GB = `source.bytes` | 67 GB = `budget.media_gb` | 20 000 train / 12 008 test | video + audio | needs-data-owner | `yavih bench info vggsounder --setup` |
| vggss_box | box (box) | `public_url`; 10.5 GB = `source.bytes` | 9.35 GB = `budget.media_gb` | 3 700 train / 914 eval | video + audio | needs-data-owner | `yavih bench info vggss_box --setup` |
<!-- yavih:bench-table data end -->

Budget rules of thumb: `bytes_per_row = N · D · itemsize (+ N·16 for
per_sample pos, the common case under media coordinates)`; fp16 K400 train with 1 568 × 1 024
tokens ≈ 64 GB, IN1K train 196 × 768 ≈ 15 GB, VGGSounder eval 1 968 × 768 ≈ 36 GB per reading;
pooled bridges (`N = 1`) are < 100 MB everywhere. `plan` keeps every shard ≤ 2 GiB and
~`shard_minutes` of wall from the profile's `clips_per_s`. Decoding now happens inside your bridge:
`clips_per_s` includes it, and the decode budget is matrix `cpus − 1` (typical: 7–29
clips/s per GPU at 8 CPUs; throughput can collapse past ~68 decode workers per node). Probe RAM
`mem_gb = max(16, 1.3 · bytes_train / 2³⁰ + 8)`; above the RAM budget the probe stages to
`YAVIH_SCRATCH` (memmap). Indicative wall per model, one-GPU medians (pooled 64-token tap):
AS-20K 1.75 h, VGGSounder 1.5 h (three readings), IN1K 1.0 h, SSv2 0.4 h, K400 0.4 h, ESC-50
0.14 h, AVE ~24 min; the NYUv2 dense head dominates its wall (give NYUv2 ≥ 1 h). Full-token taps
were 3–17 h per benchmark on one GPU.

## 2. Data roots per site

Resolution (references/cli.md "Configuration"): `Benchmark.load(data=…)` → `--data
<bench>.<root>=PATH` → env `YAVIH_DATA__<BENCH>__<ROOT>` → project `yavih.toml` → user
`~/.config/yavih/config.toml` `[data.<bench>] <root>` (each file's active `[site.<id>]` table first) →
`benchmark.yaml` default with `${data.root}` from `[data] root` (or `YAVIH_DATA__ROOT`) and
`${datasets.<name>}` from `[datasets] <name>` (else `${data.root}/<name>`). `yavih config show --bench
<b>` prints each root with the file:line, env var or default that set it; root names per benchmark are
also in setup.md. Roots marked `prepare_only` are read only by `prepare/` and `make_manifest.py`;
`doctor` reports them missing as a note.

<!-- yavih:bench-table prepare_roots begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
Roots marked `prepare_only`: avsync `media_root` and `curation_root`; nyuv2_depth `geonet_root`; vggss_box `annotations_root`.
<!-- yavih:bench-table prepare_roots end -->

Every folder script resolves its defaults through the same
layers (`yavih.config.folder_root`; shell scripts call `yavih config get <bench>.<root>`), so a flag
like `--media-root` is only an override. The user file is per user and per site (never edit another
user's); one shared HOME across several machines uses `[site.<id>]` tables, each selected by its
`hosts` patterns:

```toml
[data]
root = "/path/to/datasets"                                   # ${data.root}: <root>/<name> is each benchmark's default
[data.in1k]
media_root = "/path/to/imagenet"                             # a dataset kept outside [data] root (e.g. a group mirror)
[data.audioset20k]                                           # the hf download dir: 62 tars + index.json + SHA256SUMS
media_root = "/path/to/audioset20k"
[datasets]
vggsound = "/path/to/datasets/vggsound"                      # one entry for vggsounder, vggss_box and avsync (= the default)
[site.<id>]
hosts = ["<hostname glob>"]
[site.<id>.data]
root = "/other/path/to/datasets"
[site.<id>.datasets]
vggsound = "/other/path/to/datasets/vggsound_wds"
[runs]
root = "/path/to/yavih-runs"
[results]
root = "/path/to/yavih-results"                              # optional; default = <runs root parent>/results
```

Per job, the same thing as env: `YAVIH_DATA__K400__MEDIA_ROOT=/path/to/k400_wds` (upper-case
benchmark and root names, double underscores). `bench verify` writes the resolved roots, a
`media_fingerprint` and the media-info table's sha256 to `$YAVIH_HOME/data_status/<b>-<roots digest>.json`
(keyed by the resolved roots, so two sites sharing a HOME never void each other's stamp);
features extracted from differently staged media are flagged `W_MEDIA_FP` by `features validate`.

## 3. Inode rules

- On a fileset with a file-count (inode) quota, stage tar shards + `index.json` (VGGSound
  377 GiB in 382 inodes; K400 144 tars; SSv2 24 tars; AS-20K as its 62 published tars; AVE as its original tar), never
  loose trees; never put a per-clip cache there (a per-clip tree of VGGSound alone can exhaust a
  file-count quota). yavih reads `tar_index` layouts in place with
  `os.pread` (no decoder needed); `RawStream.open()` gives your decoder a seekable view of the
  member, so nothing is extracted to disk.
- yavih writes NO decode cache; feature shards are a handful of ≤ 2 GiB tars per split. Your bridge
  must not write one either (writes outside the run directory are `W_SIDE_EFFECT`). Scratch for the
  probe (`YAVIH_SCRATCH`) is node-local: often `/dev/shm/<job>` or a per-job tmpfs `/tmp`, which
  count against the job's memory, or the scheduler's `$TMPDIR`.
- An `index.json` with ABSOLUTE shard paths silently ignores the data root when relocated; rewrite shard entries to basenames before
  shipping and verify a member by sha on the target.

## 4. Prepare-time transforms (only where the label requires it)

- **avsync — materialization.** The label is the audio-video offset, so the folder's `prepare/`
  writes one 8 s mp4 per row (32 997 files): the VGGSound video over `[1 + δ/2, 9 + δ/2]`
  re-encoded frame-accurately with libx264, the VGGSound audio over `[1 − δ/2, 9 − δ/2]` cut
  sample-exactly and stored as ALAC (AAC's priming samples would shift the audio; the pinned
  ffmpeg 4.4 cannot put PCM in mp4). The ffmpeg version and every argument are pinned in
  `input_contract.materialize` and hashed; `prepare/` refuses another ffmpeg build and verifies each
  output by decoding it. The recipe is the pin (its sha256 is in setup.md and in every log line), not
  the output bytes: re-encoded files can differ between hosts, so no per-row sha256 ships. Budget: the source media (46.3 GB of VGGSound mp4 + wav) are read once; the output is
  ≈ 72.9 GB (`budget.media_gb`, the folder's estimate from a 216-row validation run; setup.md has
  the measured number once materialized). Plan disk for both (~120 GB) during preparation.
- **nyuv2_depth — re-containering.** GeoNet `.mat` files and the labelled `.mat` hold depth next to
  the RGB, which the bridge must never see, so `prepare/` writes lossless PNGs of the RGB only
  (GeoNet: truncate 481 × 641 → 480 × 640, add back the channel offsets, round to exact uint8;
  labelled: `images[i]`), sha256 list in `data.derived`. 12 654 PNGs of 640 × 480 RGB, 5.65 GB
  measured (the listing `rgb_png.sha256.csv.gz` is in the folder). The depth targets are the
  `*_depth_224.npy` memmaps (2.4 GB + 0.13 GB) in `derived_root`; the probe env reads its targets from that
  root (the Probe3D metric package it also needs is in setup.md "## Dependencies").
- Everywhere else nothing is transformed: `bytes()` is the file on disk.

## 5. Per-benchmark `setup.md`

`yavih bench info <b> --setup` (= `yavih bench show <b> setup`) prints the folder's setup.md: "## Obtain the data"
(commands, sizes, sha256; the prepare steps run as `yavih bench prepare <b> <step>`, `yavih bench prepare <b>` lists them), "## Dependencies" (packages to install independently, with exact pip
lines; "None." for most), "## Input contract", "## Output contract", "## Prepare & verify" (the
expected `yavih bench verify <b>` output), "## Gotchas" (bridge mistakes that pass validation on
this benchmark). One per folder:
<!-- yavih:bench-table names begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
`audioset20k`, `ave`, `avsync`, `esc50`, `fsd50k`, `in1k`, `k400`, `nyuv2_depth`, `ssv2`, `vggsounder`, `vggss_box`.
<!-- yavih:bench-table names end -->

## 6. Failure modes yavih guards against

| failure mode | what yavih does |
|---|---|
| **Class order drift**: re-deriving class indices by sorting names disagrees with the published order (on AVE, `background` must be index 28; a drift relabels segments with every shape still valid) | `classes.csv` is frozen in the folder and hashed; never re-derive an index by sorting |
| **Single-fold ESC-50 numbers** reported as the benchmark | `n_folds` is a metric; probe refuses `n_folds ≠ 5` |
| **Silent protocol-knob drop**: a model-side option is ignored and two "different" configurations produce byte-identical numbers, exit 0 | every model-side knob is declared in `describe()` and recorded with the number (`describe_sha256` keeps one feature set to one configuration); unknown `--arg` is exit 1; `args` must be echoed (`E_ARGS_UNCONSUMED`); a configuration that drifted from its `bridge check` profile is refused (`E_CONFIG_DRIFT`) |
| **Incomplete downloads**: a handful of clips present gives a near-zero mAP that looks like a real result | `bench verify` coverage; probe `E_COVERAGE` below 95 %; `effective_pos_classes` disclosed; AS-20K reads one pinned, sha256-verified public release (`11hu83/audioset20k` @ revision), never an ad hoc download |
| **Self-made splits** (e.g. 80/20 of a test set) and **permutation-invariant box heads** (a constant floor for every model) | frozen manifests with roles; floors in every result; the `box` head is positional; report greys rows at the floor; the official `vggsounder` package scores VGGSounder |
| **Window clamping** as a shortcut: a shifted window that clamps duplicates the edge frame = perfect offset detector; **normalization scoped to the window** moves with the label | avsync's shift is materialized once at prepare time, never by the bridge; out-of-media times are `E_TIMES_OUT_OF_MEDIA` unless the bridge declares `edge: clamp`; normalization constants come only from the model's recipe (`norm.source`) |
| **Batch-composition drift**: re-compositing batches changes features (rel-L2 ~1e-5 … 1e-4) | a bridge bug: `bridge check` alone/shuffled passes and canaries → `E_BATCH_INVARIANCE` above 1e-5 (a declared tolerance ≤ 1e-3 with a reason) |
| **Frame offset bugs**: frames read half a second early (`[::2]` vs `[1::2]`), unmeasured | the bridge reports the times it fed (`used`); validate recomputes the declared rule on each row's facts (`E_RULE_MISMATCH`) and positions are media coordinates |
| **OOM / wall-time kills** on full-token feature sets (hundreds of GiB as fp32 memmaps) | fp16 default, `W_FP16_RANGE`; probe memmap backend with `YAVIH_SCRATCH` pre-flight; `plan` sizes `mem_gb`/`scratch_gb`; many small shards |
| **CUDA OOM in the validation loop**, **device-placement crashes after full extraction** → "COMPLETED" jobs with zero metrics | probe micro-batches by `max_tokens_per_forward`; a `failed` status never writes DONE; report ignores results without DONE |
| **Loose-file staging** exhausts a fileset's inode quota; **absolute shard paths**; a **frame cache** larger than the media on the quota fileset | §3 above; no decode cache anywhere |
| **Stale in-archive annotation files** that disagree with the published root file (AVE); **`ytid` not unique** | manifest frozen from the published root file; `clip_id = <ytid>#<row>`; rows sharing media are exempt from the duplicate check |
| **Partial `loadmat`** hides truncated files; **positional deletes** in a published split | corrupt files recorded in `upstream.curation.excluded.corrupt`; rows named by filename; PNGs and targets with sha256 |
| **A frame-rate option silently overriding the sampling rule**; "clips are 10 s" true for one split only (K400 val ~12 s) | the frame rule is the bridge's DECLARED rule (`uniform`, `index_linspace`, `fps`, …), checked per row against the media's facts; a short or long clip is served the way the bridge declared |
| **Cross-hardware noise** (fp16 on one cluster vs fp32 on another, identical hashes) | evaluate a comparison on one cluster; CIs in every result; single-seed rows flagged |
| **Completion sentinel keyed on a metric the model never writes** → resubmit-forever loops; fail-open deadline parsing | `NOT_APPLICABLE.json` / `not_applicable` results count as done; `exec --deadline slurm` fails closed; `plan status` separates `preempted` from `failed` |
| **Decode throughput** collapses past ~68 workers/node; an A/B with unequal loader workers "proves" a speedup | `--decode-workers = cpus − 1`, capped at 16 per task; `bridge check` and `extract` measure at the same count; equal worker counts for any A/B (slurm.md) |
