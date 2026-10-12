# ave — setup

The agent's first read for AVE. `benchmark.yaml` is the machine truth; `README.md` says what the benchmark is.

## Obtain the data

`source.kind: public_url`, ungated, no account.

- media `https://huggingface.co/datasets/Helios1208/AVE_Dataset/resolve/main/AVE_Datset.tar` (the filename typo is
  upstream's), 13 353 553 920 B, sha256 `479905dc043ebae0e445c6ed1a109e5295a8f72b65ae959511db0a2bfc3a02c3`
  (hashing it takes ~12 min on network storage). 4 097 mp4 members `AVE_Dataset/AVE/<ytid>.mp4` (both streams in
  one file) + ReadMe/split txt + a **stale** `Annotations.txt`. The 2 924 distinct train + test videos the manifest
  reads are 4.02 GB (`budget.media_gb`).
- annotations `https://huggingface.co/datasets/Helios1208/AVE_Dataset/resolve/main/Annotations.txt` (the HF
  **root** copy), 144 330 B, sha256 `b0aabcea3281abb4c884e19ad4ff90314b199bea4742efe8556585a18f1396e7`, 4 143 rows
  `Category&VideoID&Quality&StartTime&EndTime`.
- official splits `https://github.com/YapengTian/AVE-ECCV18/tree/master/data` → `{train,val,test}_order.h5`
  (int64 `order` indexing Annotations rows, 3 339 / 402 / 402); sha256s in `benchmark.yaml` `source.sha256`.
- leakage-filter input (not downloaded here; staged with VGGSound): `vggsound/metadata/vggsound.csv`, sha256
  `ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8`. Only `make_manifest.py` and step 02 read it.
- Licence: none recorded on the GitHub repository or the HF mirror; the media are YouTube (AudioSet) clips under
  YouTube's terms. Research use; cite Tian et al., ECCV 2018. Check both pages before redistributing anything.
- The AudioSet-mirror copy of AVE (1.5 fps frames) is not a supported source.

Budget: 13.36 GB, 6 files (the tar, 4 small upstream files, `index.json`). **Never extract the tar** (4 097 mp4 +
4 776 other members would cost ~9 k inodes and 13 GB more; everything is read in place by byte offset).

```bash
yavih bench prepare ave                    # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
export ROOT=/path/to/datasets/ave          # default: `yavih config get ave.media_root`
yavih bench prepare ave 01                 # tar + Annotations.txt + 3 order.h5, with resume, size and sha256 checks
yavih bench prepare ave 02                 # re-check sizes + sha256 (SKIP_TAR=1 skips the 13 GB hash); 4 144 lines in Annotations.txt
```

Use the HF-root `Annotations.txt`, never `AVE_Dataset/Annotations.txt` from inside the tar (2 757 rows / 19
categories, not indexable by the split files; `make_manifest.py` refuses it on the row count).

## Dependencies

None. AVE's authors ship no installable scorer (`external: []`); the official accuracy (`compute_acc` in
YapengTian/AVE-ECCV18 `supervised_main.py`) is a few lines that yavih's `seg_acc` transcribes.

- Reading the tar members in place needs nothing beyond yavih core (the reader hands over bytes). The census
  (`yavih bench verify ave --census`, header facts per member) needs `yavih[media]` (PyAV, `pip install av`); with
  the census present the model env needs no yavih decoder.
- The bridge decodes with its own stack (PyAV, torchaudio, decord, …) or with `yavih.media.decode`
  (`yavih[media]`). The ffmpeg CLI cannot read a tar member at an offset; use `RawStream.open()` / `bytes()`.
- `prepare/04_media_census.py` (optional) needs PyAV + numpy; `make_manifest.py` needs h5py.

## Input contract

Per row, two streams over ONE stored file: the member `video/<ytid>.mp4` of `AVE_Datset.tar`, read in place through
`index.json` (`layout: tar_index`, `data.tar_index`). Both `RawStream`s return the same bytes (read once);
`RawStream.open()` is a seekable file object over them, so PyAV/torchaudio decode the member without extracting
anything.

- `video` — the mp4 exactly as stored; `info` describes its video track: `duration_s` (9.96–10.08 s), `fps`
  (16–46; 2 226 videos ≈ 29.97–30, 695 exactly 25), `width` × `height` (mixed: 1 213 × 1280×720, 916 × 480×360,
  262 × 320×240, 245 × 640×360, … down to 128×96), `container`, `bytes`.
- `audio` — the same mp4; `info` describes its AAC track: `duration_s` (= `n_samples / sample_rate`; 9.73–10.05 s
  by decoded sample count; AAC header counts differ from the decoded count by a few hundred samples — priming and
  padding — so use `info`, never a constant), `sample_rate` (44 100 everywhere), `n_samples`, `channels` (2 620
  stereo, 612 mono), `container`, `bytes`.
- `meta["facts"] == {"num_segments": 10, "segment_seconds": 1.0}`: labels are per one-second segment of the media,
  seconds 0 … 9. Nothing else reaches the bridge (no label, event range, split or media key).
- Materialization: none. Nothing is decoded, cut, resampled, resized or padded.

The bridge decides, and declares in `describe()["input"]`: which frames at which times (a `uniform` rule over
`span: full` is the natural choice; 8 frames hit 8 distinct seconds, 10 or 16 hit all ten), the resize/crop (the
aspect ratio varies, so the kept region of the original frame varies per clip), the audio front end (resampling
from 44.1 kHz, mono downmix, fbank/mel, normalization constants from the model's pretraining recipe), how much
audio it reads and how it pads the 74 rows whose tracks are shorter than 10 s (`full` with `pad_to_s`, or `span`
with `pad: zeros`), and how a frame time beyond a short video track is served (`edge: clamp` or `error`). The
decoded audio track of some clips starts 0–36 ms after the video (AAC priming); samples are indexed from the first
decoded sample.

```yaml
handed_over:
  video: "the AVE mp4 (tar member video/<ytid>.mp4 of AVE_Datset.tar) exactly as stored, read in place; its video track (16-46 fps, 128x96 to 1280x720, 9.96-10.08 s) is described by info; nothing decoded, cut, resized or padded"
  audio: "the same mp4 bytes as video (one tar member, read once); its AAC audio track (44.1 kHz, mono or stereo, 9.7-10.1 s) is described by info; nothing decoded, resampled, cut or padded"
info:
  video: [duration_s, fps, width, height, container, bytes]
  audio: [duration_s, sample_rate, n_samples, channels, container, bytes]
facts: [num_segments, segment_seconds]
materialize: null
```

## Output contract

`encode` returns `(feats, pos, used)`: `feats` `[B, N, D]` with one fixed `(N, D)` for every row of every split;
`pos` `[B, N, 4]` = `(stream, t, y, x)` in media coordinates — `t` = token centre seconds / that stream's
`info.duration_s`, `(y, x)` = fraction of the ORIGINAL frame (before the bridge's resize or crop;
`grid_pos(..., region=crop_region(spatial, width, height))`), `-1` = no axis. Because durations and frame sizes
differ between clips, `pos_mode` is `per_sample` for any token bridge. `used` reports per clip the decoded frame pts
(`video`) and the sample ranges and padding fed (`audio`).

Eligibility (decided by `yavih features validate`): tokens need a `t` axis (`pos: [t]`), the `video` stream must
appear among the timed tokens (`streams: [video]`), and on the median sampled clip at least **8 of the 10 scored
seconds** must hold a timed token, at absolute second `floor(t · duration_s / 1.0)` (`t_bins: 8`). An audio-only
model is not applicable here (raise `yavih.NotApplicable` from the bridge; recorded `not_applicable`), and neither
is a time-pooled token set. The seq probe embeds each token at its absolute second, so any frame count, fps or
window the bridge chose is read where it lies in the media. Targets live in time only (`target_frame: null`):
segment `s` of row r is its class iff `event_start ≤ s < event_end`, else `background` (28). A second the media
covers only partly (74 rows end between 9.73 and 9.99 s) is still scored.

```yaml
shape: fixed
coordinates: media
target_frame: null
requires:
  pos: [t]
  streams: [video]
  grid: null
  t_bins: 8
```

## Prepare & verify

```bash
# 1. index the tar in place (never extracts; seconds warm, minutes cold on network storage); the default
#    --shard-path relative stores the bare tar name, resolved next to index.json
yavih bench prepare ave 03 -- --root $ROOT
#    an index copied from elsewhere: `yavih bench prepare ave 03 -- --relocate $ROOT/index.json` (offsets unchanged)

# 2. point yavih at the root: either
export YAVIH_DATA__AVE__MEDIA_ROOT=$ROOT
#    or in ~/.config/yavih/config.toml: [data.ave] media_root = "/path/to/datasets/ave"
#    (or set [data] root so that ${data.root}/ave is the AVE root)

# 3. verify the data and write the media-info table (header facts of every member)
yavih bench verify ave --census
```

There are no derived files (`data.derived: {}`) and no materialization. Success looks like:

```
ave: manifest sha ok; labels ok; curation ok; ... status verified
  train    present 2624/2624  missing_first10=[]
  test     present 316/316  missing_first10=[]
  read sample ok=32 failed=0
  decode sample ok=32 failed=0 ...
  media_info present rows=5848 sha256=cf1d638c137e16d551c281b9fa76d849f9dc7cc981baeca0db7e4de7ef4a4b73
  media_fingerprint 61cd7931613621d0ed7794dc4c986d4b709bf95fe92fdbbaee6e7e782ba1cbd8
```

A plain `yavih bench verify ave` afterwards prints the same in seconds. Budget: ~1.5 GPU-min per 1 000 clips for a
16-frame ViT-B bridge (decode-bound), ~4.5 GPU-min for the 2 940 rows read; the seq probe plans ~2 h per seed at
N = 2080, D = 768.

## Gotchas

Bridge-side mistakes that pass validation:

- **Time-pooled tokens.** Learned memory/latent tokens, a CLS token or mean pooling over time have no per-second
  information. If they carry a `t` they pass eligibility and score near the floor; give them `t = -1` and expose
  per-frame / per-tubelet tokens instead.
- **Wrong `t`.** `t` is a fraction of the stream's own `duration_s`, computed from the pts of the frames actually
  decoded (or the tubelet's mean pts), not from the requested times and never from absolute seconds or from a fixed
  10 s. Audio tokens: centre of the patch's sample span / audio `duration_s`; a patch centred in padding the bridge
  added carries `t = -1`. Validate (and `bridge check`) refuse a token whose `t · duration_s` lies outside the
  frames / samples the bridge reported in `used` (`E_RULE_MISMATCH`), so a window-relative `t` or `grid_pos`'s
  default `(k + 0.5)/T` under a non-`full` rule does not pass; a `t` that is wrong but inside the consumed span
  still does.
- **Assuming 10 s.** 74 rows have a track shorter than 10 s (audio down to 9.73 s, video down to 9.96 s). A rule
  that asks for a frame or samples beyond the media is `E_TIMES_OUT_OF_MEDIA` unless declared (`edge: clamp`,
  `pad: zeros`, `pad_to_s`).
- **Reading only part of the clip.** All ten seconds are scored. A bridge that feeds only an 8 s window (e.g.
  [1, 9) s) still qualifies if its tokens hit 8 seconds, but the seconds it never saw are guessed from context;
  declare the span and expect a lower `seg_acc`.
- **Per-batch / per-clip normalization** of frames or spectrograms; use the model card's constants (`norm.source`).
  Batch statistics fail `bridge check` batch invariance; per-clip statistics pass it and are usually not the
  model's recipe — unless the model was pretrained with them (e.g. a log-mel standardised per clip over its padded
  buffer): then keep them and declare them in `norm` with their `source`, since replacing them departs from the
  model's recipe.
- **Resolution and aspect.** Native frames go up to 1280×720 (1 213 videos) and mix 4:3 and 16:9; decode dominates
  cost; resize in the bridge and map positions back with `crop_region`.
- **Mono downmix.** 2 620 of 3 232 videos are stereo; declare the downmix (`channels: mean` or the model's).
- **Duplicated features.** The two events of one video (16 train videos) get identical features by construction
  (one `media_key`); validate exempts them from `E_DUPLICATE_CLIPS`. Do not add per-row noise or cache by path to
  "fix" it.
- **Audio priming offset.** The decoded audio of some clips starts 0–36 ms after the video (AAC priming). Do not
  "correct" it in the bridge.
- **Leakage of the pretraining set.** The freeze drops AVE videos that are in VGGSound train (886 rows). A model
  pretrained on AudioSet (AVE clips are AudioSet segments) or anything else that contains AVE test videos is NOT
  covered: audit it yourself before claiming a number.
