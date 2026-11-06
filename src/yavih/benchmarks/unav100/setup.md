# unav100 — setup

The agent's first read for UnAV-100. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

`obtain.tier: auto`: everything comes from the authors' public, ungated Hugging Face dataset
`https://huggingface.co/datasets/ttgeng233/UnAV-100`, pinned at revision `daeb88bc14eeff48f8a4ac2836b67f6c350f847b`.
No account, token or licence click. The dataset card declares CC BY-NC-SA 4.0 (the project page
`https://unav100.github.io/` shows a CC BY 4.0 badge); the media are YouTube videos (VGGSound sources) under
YouTube's terms. Research use; cite Geng et al., CVPR 2023.

- annotations `data/{train,validation,test}.parquet` (323 290 / 116 962 / 115 568 B; sha256 `d8b1302f…`,
  `551e1781…`, `36a7d528…`, full digests in `benchmark.yaml` `source.sha256`): one row per event, columns
  `video_id, duration, timestamp [start, end], event_label`; 18 123 / 6 036 / 5 900 events over 6 489 / 2 134 /
  2 167 videos. They are the authors' `unav100_annotations.json` (Google Drive) row for row.
- media `unav_raw_videos.tar.gz.part_00` … `part_12`: one gzip tar split into 13 byte ranges (12 × 5 368 709 120 B
  + 1 351 094 873 B = 65.78 GB), sha256 per part in `source.sha256`. 10 790 mp4 members
  `ssd3/gengtiantian/audio_visual_data/unav100/raw_videos/<ytid>.mp4`, already cut to the annotated span (10–60 s).
  The 8 656 train + test videos the manifest reads are ~52.8 GB (`budget.media_gb`).

Disk: 65.8 GB of parts + ~66 GB for the joined, decompressed tar (step 02; `KEEP_PARTS=0` deletes the parts once the
tar is complete, so the staged data ends at ~66 GB in 2 files + `index.json`). **Never extract the tar**: every read
goes through `index.json` byte offsets.

```bash
yavih bench prepare unav100                    # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
export ROOT=/path/to/datasets/unav100          # default: `yavih config get unav100.media_root`
yavih bench prepare unav100 01                 # 3 parquets + 13 parts: resumable, size + sha256 checked (~1 h at 20 MB/s)
KEEP_PARTS=0 yavih bench prepare unav100 02    # cat parts | gunzip -> $ROOT/unav_raw_videos.tar, checked with tar -t
```

## Dependencies

None in the probe env: `external: []`. The authors' scorer (`ANETdetection` in `ttgeng233/UnAV@1609deba`
`libs/utils/metrics.py`) is not an installable package (no `setup.py`, its package imports torch and a C++ NMS
extension, and it uses `np.float`, removed in numpy 1.24); the folder's `detection.py` (its own scorer)
transcribes it and reproduces its numbers on the same predictions (including its `np.linspace(0.1, 0.9, 9)`
thresholds, which is why `protocol.tiou_thresholds` holds 0.30000000000000004 and 0.7000000000000001).

- Prepare-time tools: `curl`, `gzip`, `tar`, `sha256sum` (steps 01–02); Python 3.10+ stdlib (step 03);
  `pyarrow` for `make_manifest.py` reading the parquets (`pip install pyarrow`; step 04 runs it; or pass
  `--annotations-json` with the authors' json, stdlib only).
- The census (`yavih bench verify unav100 --census`) needs `yavih[media]` (PyAV).
- The bridge decodes with its own stack (PyAV, torchaudio, decord, …) or with `yavih.media.decode`
  (`yavih[media]`). The ffmpeg CLI cannot read a tar member at an offset; use `RawStream.open()` / `bytes()`.

## Input contract

Per row, two streams over ONE stored file: the member `video/<ytid>.mp4` of `unav_raw_videos.tar`, read in place
through `index.json` (`layout: tar_index`). Both `RawStream`s return the same bytes (read once); `RawStream.open()`
is a seekable file object over them.

- `video` — the mp4 exactly as stored; `info` describes its H.264 track: `duration_s` (10–60 s, 42 s on average;
  within ~0.1 s of the annotated duration), `fps` (about 12–30; 25, 29.97 and 30 are common), `width` × `height`
  (from 202×360 portrait to 1280×720, 4:3 and 16:9 mixed), `container`, `bytes`.
- `audio` — the same mp4; `info` describes its AAC track: `duration_s`, `sample_rate` (44 100), `n_samples`,
  `channels` (mono or stereo), `container`, `bytes`. The audio and video tracks differ in length by up to ~0.2 s.
- `meta["facts"] == {}`: the events, their classes and the annotated duration live in the manifest; nothing of
  them reaches the bridge.
- Materialization: none. Nothing is decoded, cut, resampled, resized or padded.

The bridge decides, and declares in `describe()["input"]`: how many frames at which times (a `uniform` rule over
`span: full` spreads a fixed count over the whole clip, whatever its length), the resize/crop, the audio front end
(resampling from 44.1 kHz, downmix, fbank/mel, normalization constants from the model's pretraining recipe), how
much audio it reads (the whole track: `full` with `max_s: null`), and whether it pools space (the probe reads time
only).

```yaml
handed_over:
  video: the UnAV-100 mp4 (tar member video/<ytid>.mp4 of unav_raw_videos.tar) exactly as stored, read in place; its H.264
    track (10-60 s, about 12-30 fps, up to 1280x720) is described by info; nothing decoded, cut, resized or padded
  audio: the same mp4 bytes as video (one tar member, read once); its AAC track (44.1 kHz, mono or stereo) is described by
    info; nothing decoded, resampled, cut or padded
info:
  video: [duration_s, fps, width, height, container, bytes]
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: []
materialize: null
```

## Output contract

`encode` returns `(feats, pos, used)`: `feats` `[B, N, D]` with one fixed `(N, D)` for every row of every split
(clips last 10 to 60 s, so a fixed token COUNT spread over the clip, not a fixed rate); `pos` `[B, N, 4]` =
`(stream, t, y, x)` in media coordinates — `t` = token centre seconds / that stream's `info.duration_s`, `(y, x)` =
fraction of the original frame or `-1` — with `pos_mode: per_sample`; `used` reports the decoded frame pts
(`video`) and the sample ranges fed (`audio`).

Eligibility (decided by `yavih features validate`): tokens need a `t` axis (`pos: [t]`); no stream is required, so
the `a`, `v` and `av` readings are all eligible. The folder's detection scorer tiles the clip with `T = 128` bins and reads every
token at `u = t · duration_s(stream) / D`, `D` the annotated duration: the fraction of the clip it lies at. Targets
live in time only (`target_frame: null`): bin `k` is positive for class `c` iff an event of `c` covers its centre
(or holds its midpoint), and regresses the distances to the shortest such event's ends. A bridge whose tokens carry no `t` (pooled over time) is `not_applicable`.

```yaml
shape: fixed
coordinates: media
target_frame: null
requires:
  pos: [t]
  streams: []
  grid: null
  t_bins: null
```

## Prepare & verify

```bash
# 3. index the tar in place (one header pass, never extracts); stores the bare tar name, resolved next to index.json
yavih bench prepare unav100 03                 # or: -- --root $ROOT

# 4. point yavih at the root, then check the manifest against the parquets and verify (needs pyarrow)
export YAVIH_DATA__UNAV100__MEDIA_ROOT=$ROOT   # or [data.unav100] media_root = "..." in ~/.config/yavih/config.toml
yavih bench prepare unav100 04                 # = make_manifest.py --check + yavih bench verify unav100 --census
```

There are no derived files and no materialization. Success looks like:

```
[make_manifest unav100] 10790 videos, 30059 events, 100 classes from HF ttgeng233/UnAV-100 @ daeb88bc ...
[make_manifest unav100] presence: 0 missing {}
[make_manifest unav100] --check: manifest.csv.gz identical, classes.csv identical
unav100: manifest sha ok; labels ok; curation ok; ... status verified
  train    present 6489/6489  missing_first10=[]
  test     present 2167/2167  missing_first10=[]
  read sample ok=32 failed=0
  decode sample ok=32 failed=0 ...
```

The `validation` split (2 134 videos) is `unused`: it is neither read nor verified. Budget: decode-bound; a 32-frame
bridge decodes ~5 clips/s on 15 CPU workers, so extraction of the 8 656 rows is roughly 30 min plus the model's
forward time.

## Gotchas

Bridge-side mistakes that pass validation:

- **Reading a fixed window.** A bridge that reads the first 10 s (an AudioSet habit) passes every check and leaves
  three quarters of the average clip unseen; the events there are unfindable. Spread the tokens over the whole clip.
- **Time-pooled tokens.** A CLS token, learned latents or mean pooling over time carry no time; with a `t` they pass
  eligibility and score near the floor. Give them `t = -1` and expose per-frame / per-window tokens.
- **Wrong `t`.** `t` is a fraction of the stream's own `duration_s`, from the pts actually decoded (or the window's
  sample span for audio), never from absolute seconds or a nominal length. `E_RULE_MISMATCH` catches a `t` outside
  what `used` reports, not a wrong `t` inside it.
- **Too few tokens.** Events are as short as a fraction of a second and the strict thresholds (tIoU 0.8, 0.9) need
  boundaries to within a few percent of the event length; 8 tokens over a 60 s clip cannot place them. 32 or more
  per stream is a reasonable floor.
- **Per-batch / per-clip normalization** of frames or spectrograms; use the model card's constants (`norm.source`).
- **Pretraining leakage.** Every UnAV-100 video is a VGGSound source video (the paper collected them from VGGSound's
  URLs), so a model pretrained on VGGSound has seen 10 s of most train and test videos. The official splits are kept;
  disclose the overlap next to any number.
