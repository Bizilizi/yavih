# llp_avvp — setup

The agent's first read for LLP (AVVP). `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

`obtain.tier: auto`: public, ungated, no account. Two prepare steps; both check sha256s.

- **Annotations** (step 01): the six tab-separated files of the authors' repository
  `https://github.com/YapengTian/AVVP-ECCV20` at commit `fde56118f2c2e7c19188dfefa2569a37b70054fe`
  (`source.revision`), `data/AVVP_{train,val_pd,test_pd,eval_audio,eval_visual,dataset_full}.csv`, 1 033 239 B in
  all, into the prepare-only `annotations_root` (default `${data.root}/llp_avvp/annotations`). Only
  `make_manifest.py` reads them; the frozen manifest already carries every label. sha256s: `benchmark.yaml`
  `source.sha256`.
- **Media** (step 02): the authors distribute no videos (their `scripts/download_dataset.py` fetches each clip from
  YouTube and cuts `[start, start + 10 s]` with ffmpeg). The public Hugging Face dataset `CserDu123/LLP-dataset` at
  revision `31727951b0ea2a573501cda490005c68de6822d0` holds all 11 849 clips cut that way: `llp_video.zip`,
  13 438 121 116 B, sha256 `36093363b2d6cc540d1da665fd78af73d339cc6e3eeeb0993778b967e75211d3`. Step 02 downloads it
  into `<media_root>/download/` (one connection, resumable), checks it, extracts `llp_videos/<ytid>.mp4` (13.5 GB,
  every clip with both an H.264 and an AAC track) and deletes the zip unless `KEEP_ZIP=1`. The dataset's
  `audio_data.zip` (mp3s) is not used. Peak disk: 27 GB while the zip and the clips coexist, 13.5 GB after.
- Licence: annotations and code GPL-3.0 (the repository README); the clips are YouTube (AudioSet) content that
  stays its uploaders', redistributed by the HF dataset under a tag that cannot relicense them. Research use; cite
  Tian et al., ECCV 2020.
- An existing AudioSet copy holds most of the clips too (the segments match the HF clips within 30 ms), but under
  other names and cuts: it is not a supported source.

```bash
yavih bench prepare llp_avvp                 # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
yavih bench prepare llp_avvp 01              # the six annotation files (ROOT= overrides annotations_root)
yavih bench prepare llp_avvp 02              # llp_video.zip -> llp_videos/*.mp4 (ROOT= overrides media_root; KEEP_ZIP=1 keeps the zip)
```

## Dependencies

None to install: `external: []`. The LLP authors publish no package; the folder carries their code.

- Scorer: rung 2 (official code, not a wheel). `avvp_eval_metrics.py` is `utils/eval_metrics.py` of
  `YapengTian/AVVP-ECCV20` @ `fde56118f2c2e7c19188dfefa2569a37b70054fe` byte for byte; `avvp.py`
  (`video_scores`, `summary`) transcribes the evaluation loop of `main_avvp.py::eval` (threshold 0.5, the
  video-level mask, the ground-truth matrices, the per-video means, Type@AV and Event@AV). GPL-3.0, kept with the
  copies.
- Readout: rung 2. `han.py` is `nets/net_audiovisual.py` of the same commit (HAN + attentive MMIL pooling, the
  authors' baseline on frozen features) with the input layers resized to the bridge's width and the R(2+1)D
  branch removed; `avvp.py` (`han_loss`, `_fit_han`) transcribes `main_avvp.py::train` and `main` (Adam 3e-4,
  StepLR 10 / 0.1, 40 epochs, batch 16, label smoothing 0.9 on the visual guide, best val Type@AV kept).
- The probe needs torch (the `probe` extra). The census (`yavih bench verify llp_avvp --census`) needs PyAV
  (`yavih[media]`); `make_manifest.py` with media needs PyAV too. Step 02 needs `curl`, `sha256sum` and `unzip`
  (or Python's `zipfile`).

## Input contract

Per row, two streams over ONE stored file, `llp_videos/<ytid>.mp4` under `media_root` (`layout: dir`). Both
`RawStream`s return the same bytes (read once).

- `video` — the mp4 exactly as stored; `info` describes its H.264 track: `duration_s` (6.77–10.15 s; 780 tracks
  shorter than 9.9 s), `fps` (6–32; 7 636 clips at about 30, 2 215 at 25, 66 below 8), `width` × `height` (128×88
  to 1280×720, mixed: 3 219 × 480×360, 2 584 × 1280×720, 2 296 × 640×360, 1 238 × 320×240, portrait clips such as
  406×720), `container`, `bytes`.
- `audio` — the same mp4; `info` describes its AAC track: `duration_s` (6.21–10.02 s; 702 shorter than 9.9 s),
  `sample_rate` (44 100 everywhere), `n_samples`, `channels` (9 133 stereo, 2 716 mono), `container`, `bytes`.
- `meta["facts"] == {"num_segments": 10, "segment_seconds": 1.0}`: the labels are per media second 0–9. Nothing
  else reaches the bridge (no label, event, split or media key).
- Materialization: none. Nothing is decoded, cut, resampled, resized or padded.

The bridge decides, and declares in `describe()["input"]`: which frames at which times, the resize and crop, the
audio front end (resampling from 44.1 kHz, mono downmix, fbank/mel, normalization from the model's recipe), how
much audio it reads and how it pads short tracks, and how a frame time beyond a short video track is served. The
template's defaults follow the authors' own features (80 frames per clip, 224 centre crop, 16 kHz audio in ten
1-s windows); they are not protocol.

```yaml
handed_over:
  video: "the LLP clip mp4 (llp_videos/<ytid>.mp4 of the extracted CserDu123/LLP-dataset zip) exactly as stored; its H.264 track (6-32 fps, 128x88 to 1280x720, 6.8-10.2 s) is described by info; nothing decoded, cut, resized or padded"
  audio: "the same mp4 bytes as video (read once); its AAC track (44.1 kHz, mono or stereo, 6.2-10.0 s) is described by info; nothing decoded, resampled, cut or padded"
info:
  video: [duration_s, fps, width, height, container, bytes]
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: [num_segments, segment_seconds]
materialize: null
```

## Output contract

`encode` returns `(feats, pos, used)`: `feats` `[B, N, D]` with one fixed `(N, D)` for every row of every split;
`pos` `[B, N, 4]` = `(stream, t, y, x)` in media coordinates — `t` = token centre seconds / that stream's
`info.duration_s`, `(y, x)` a fraction of the original frame or `-1` (unused here), `-1` = no axis. Durations
differ between clips, so `pos_mode` is `per_sample`. `used` reports the decoded frame pts (`video`) and the sample
ranges and padding fed (`audio`).

What the scorer does with them: every token goes to the media second `floor(t · duration_s / 1.0)` of its own
stream; per stream and second the tokens are averaged (a second without a token of that stream reads zeros); the
ten audio and ten video vectors feed HAN's audio and visual branches. A reading with one stream only (`a` or `v`)
feeds that stream to both branches. Eligibility (`yavih features validate`): tokens need `t` (`pos: [t]`) and on
the median sampled clip at least **8 of the 10 seconds** must hold a timed token of some stream (`t_bins: 8`);
otherwise the probe writes `not_applicable`. Targets live in time only (`target_frame: null`).

```yaml
shape: fixed
coordinates: media
target_frame: null
requires:
  pos: [t]
  streams: []
  grid: null
  t_bins: 8
```

## Prepare & verify

```bash
# point yavih at the data: either
export YAVIH_DATA__LLP_AVVP__MEDIA_ROOT=/path/to/datasets/llp_avvp
export YAVIH_DATA__LLP_AVVP__ANNOTATIONS_ROOT=/path/to/datasets/llp_avvp/annotations
#   or [data] root = "/path/to/datasets" in ~/.config/yavih/config.toml (then ${data.root}/llp_avvp is used)
yavih bench prepare llp_avvp 01
yavih bench prepare llp_avvp 02
python "$(yavih bench path llp_avvp)/make_manifest.py" --check     # the staged annotations re-derive the frozen manifest
yavih bench verify llp_avvp --census                               # presence, a read and decode sample, the media-info table
```

There are no derived files and no materialization. Success looks like:

```
llp_avvp: manifest sha ok; labels ok; curation ok; targets_sha_ok=None; derived_sha_ok=None; status verified
  train    present 10000/10000  missing_first10=[]
  val      present 649/649  missing_first10=[]
  test     present 1200/1200  missing_first10=[]
  read sample ok=30 failed=0
  decode sample ok=30 failed=0 ...
  media_info present rows=23698 sha256=3d31272fed0ea39211ce1c34ae1b532bf649718720df6f376c0af3e5238c92b4
  media_fingerprint ed6868070503deea74c33d167706904054f128b3a48aecff6077a00a6ef3cabc
```

The census takes about 1.5 min on network storage (300 cells/s); a plain `yavih bench verify llp_avvp` afterwards
takes seconds. Extract all three splits (`train`, `val`, `test`): the probe fits on `train`, picks the epoch on
`val` and scores `test`.

## Gotchas

Bridge-side mistakes that pass validation:

- **Time-pooled tokens.** A CLS token, learned latent tokens or a mean over time carry no per-second information;
  with a `t` they pass eligibility and score near the floor. Give them `t = -1` and expose per-frame / per-window
  tokens.
- **Wrong `t`.** `t` is a fraction of the stream's own `duration_s`, from the pts of the frames actually decoded
  (or a window's centre), never from the requested times or a fixed 10 s. Audio windows centred in padding the
  bridge added carry `t = -1`. A `t` that is wrong but inside the consumed span passes `E_RULE_MISMATCH`.
- **Assuming 10 s.** 780 video and 702 audio tracks are shorter than 9.9 s (6.2 s at the least). A rule that asks
  for frames or samples beyond the media is `E_TIMES_OUT_OF_MEDIA` unless declared (`edge: clamp`, `pad: zeros`,
  `pad_to_s`). Seconds the media does not cover are still scored.
- **Low frame rates.** 66 clips run below 8 fps: 80 uniform frames then repeat decoded frames (the pts say so).
- **Per-batch / per-clip normalization** of frames or spectrograms; use the model's recipe (`norm.source`). Batch
  statistics fail `bridge check` batch invariance.
- **Resolution and aspect.** Native frames go up to 1280×720 and mix 4:3, 16:9 and portrait clips; decode dominates
  cost; resize in the bridge.
- **No `bridge check --synthetic`.** The core's generated stand-in does not cover this folder's own scorer
  (`E_SYNTHETIC`); check a bridge on the staged clips with `yavih bridge check <file> --bench llp_avvp`.
- **Extract `val`.** It is a train-role split here (the epoch is chosen on it), not an unused one: a run without its
  features is refused (`E_SPLIT_MISSING`).
- **Leakage of the pretraining set.** Every clip is an AudioSet segment (train, val and test alike); a model
  pretrained on AudioSet may have seen the test videos and their audio labels. Disclose it next to any number.
