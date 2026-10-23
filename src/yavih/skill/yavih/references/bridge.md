# Bridge reference

A bridge is the only code you write: a `yavih.Bridge` subclass in one `.py` file that turns the
benchmark's raw files into a fixed-shape token set `[N, D]` per clip. A bridge file serves one
benchmark; each matrix entry (file + args) is one reading of one model. A model read on five
benchmarks has five bridge files: thin subclasses of one shared `<model>_common.py` (§7). yavih
reads bytes, batches, writes shards, validates, probes and reports. The
benchmark owns only what the label depends on; you own everything model-side and declare it.

Read the benchmark's setup.md first (`yavih bench show <b> setup`, the same as `yavih bench info <b> --setup`; every
shipped file of the folder is listed by `yavih bench files <b>` — never search site-packages or a checkout): its "## Input contract"
says exactly what you receive, its "## Output contract" what must come back, and "## Gotchas"
lists the bridge mistakes that pass validation on that benchmark.

## 1. What `encode()` receives: raw files and facts

```python
class RawStream:                     # read-only
    name: str                        # stream name from benchmark.yaml ("video", "audio", "image"); the pos tag vocabulary
    kind: str                        # "video" | "audio" | "image"
    path: str | None                 # an OPAQUE absolute path to the stored file (a hash-named symlink: the stored name can carry
                                     #   the label or split, e.g. k400 val/<class>/); None for a tar member
    member: tuple | None             # (tar path, data offset, size) for a tar member; None otherwise
    info: dict                       # facts: duration_s, fps, n_frames, sample_rate, n_samples, channels, width, height, container, bytes,
                                     #   has_track (False: the file holds no such track; its track facts are None)
    def bytes(self) -> bytes         # the stored file, byte for byte (tar member: the member's bytes)
    def open(self) -> BinaryIO       # read-only seekable file object over the same bytes (PyAV, soundfile, pillow accept it)

@dataclass(frozen=True)
class Sample:
    row: int; clip_id: str; split: str         # bookkeeping: MUST NOT change what you compute (bridge check scrambles them)
    streams: dict[str, RawStream]              # every present stream under its declared name
    meta: dict                                 # {"facts": {...}}: label facts only, e.g. {"annotated_second": 5.0}

@dataclass(frozen=True)
class BenchmarkView:                           # what Bridge.__init__ receives: no manifest, labels, splits or roots
    name: str; aliases: list[str]; readout: str; protocol_version: int; protocol_hash: str
    streams: dict[str, str]                    # {name: kind}
    input_contract: dict; output_contract: dict
    facts: dict                                # = Sample.meta["facts"] of every row
```

Raw means stored: `bytes()` is the file exactly as it lies on disk (avsync: the prepare-time
materialized mp4 whose audio is shifted by the label; nyuv2_depth: the lossless PNG re-containering
of the upstream RGB; audioset20k: the original clip mp4 from its published tar). Nothing is decoded, resampled, trimmed,
cropped, resized, padded, selected or normalized, and no statistic is computed. Two streams may be
the two tracks of one mp4 (ave, avsync, audioset20k): both return the same bytes, each `info` describes its own
track.

`info` keys listed in setup.md's `input_contract.info` are guaranteed non-null on every row; others
may be null (webm headers have no frame count, so SSv2 takes `n_frames` from the census's packet
count). Facts come from the site's media-info table (`yavih bench verify <b> --census`), so the
model env needs no yavih decoder. A row the table lacks (no census yet, or a file staged after it) gets
its facts from a live header probe of the same file the first time `info` is read: slower, but the same
facts, never a guess. A row whose media is missing never reaches you (missing row,
coverage). A file WITHOUT the track a stream names (audioset20k: 3 mp4s have no audio track) is not
missing: it reaches you with `info["has_track"] is False` and the track facts `None`. Branch on it
before any rule or decode, feed your own declared choice (e.g. silence; `describe()["no_track"]`),
return the same fixed-shape row and report `used` as `{"times": []}` / `{"samples": [], "pad": [0, 0]}`. A clip your decoder cannot read: raise `yavih.MediaUndecodable([clip_id], reason)`;
run_extract re-encodes the batch without it (exact, because you are batch-invariant) and records it
missing. Any other exception aborts the shard (exit 5): a bridge bug is not coverage.

Decoding is yours. Use the model's own stack (torchvision/torchaudio/decord/PIL, opened on
`stream.open()` or `stream.path`), or the optional torch-free helpers (`[media]` extra), which
carry NO protocol semantics:

```python
from yavih.media.decode import frames_at, audio, image, decoder_info
frames, pts = frames_at(stream, times)        # uint8 [n, H, W, 3], float64 [n]: nearest decoded frame per requested time
wav, sr = audio(stream, ranges)               # float32 [L, C] at the native rate, ranges = int64 [k, 2] sample ranges
img = image(stream)                           # uint8 [H, W, 3]
decoder_info()                                # {lib, version, libav} for describe()["input"][s]["decoder"]
from yavih.media.rules import rule_times, audio_ranges, crop_region   # numpy only; validate uses the same code
```

`run_extract` sets `self.decode_workers` (= matrix `cpus − 1`) before the first `encode`: your CPU
budget for decoding (PyAV, soundfile and pillow release the GIL).

**Decode in `prepare(sample)`, not in `encode`** (every packaged template does): yavih runs
`prepare` per sample in `decode_workers` threads up to `--read-ahead + 1` batches ahead, so decoding
overlaps the forward pass, and hands its return value to `encode` as `s.prepared`. Contract: a pure,
deterministic function of that one sample (no model/GPU state, nothing kept across calls;
`bridge check` prepares every smoke sample twice: `E_PREPARE_NONDETERMINISTIC`); it may raise
`MediaUndecodable([sample.clip_id], reason)` / `MediaMissing` (missing row); it runs under
`inference_mode` and the same network/write audit as `encode`. The stored bytes are released once
`prepare` returns, so `encode` uses `s.prepared` + `info` and never `bytes()` / `open()` (set
`keep_bytes = True` to keep them). Without `prepare`, decode inside `encode`.
Full contract: https://github.com/Bizilizi/yavih/blob/main/docs/bridge.md#prepare

## 2. The `Bridge` class

```python
import yavih
from yavih.bridge import ArgSpec, fingerprint_state_dict
from yavih import grid_pos, NotApplicable, MediaUndecodable

class MyBridge(yavih.Bridge):
    name = "vjepa-video"              # optional label of this file, recorded only (the model's NAME is describe()["model"])
    batch_size = 8                    # recorded in describe(); shards are aligned to it
    dtype = "float16"                 # "float16" | "float32"
    pos_mode = "per_sample"           # "none" (pooled) | "constant" ([N,4] valid for every row) | "per_sample" ([B,N,4])
    streams = ("video",)              # stream tags in pos[:, 0], order = tag index
    ARGS = {"layer": ArgSpec(type=int, default=12)}   # one ArgSpec per --arg k=v you consume; unknown key → exit 1

    def __init__(self, bench, args):  # bench: BenchmarkView; weights loaded HERE, never inside encode
        super().__init__(bench, args)
        ...; self.model.eval(); self.sha = fingerprint_state_dict(self.model)
    def describe(self) -> dict: ...   # REQUIRED, JSON primitives only (table below)
    def prepare(self, sample): ...    # optional: decode + CPU preprocessing of ONE sample → s.prepared
    def encode(self, samples):        # REQUIRED → (feats [B, N, D], pos None | [N, 4] | [B, N, 4], used: list of B dicts)
        ...
```

`feats` may be an ndarray or anything with `.detach().float().cpu().numpy()`. `N` and `D` are
fixed for every row of every split (the writer fixes them from the first clip and refuses a
mismatch by `clip_id`): pad or crop to a FIXED length inside the bridge, never to the batch
maximum. `prepare` and `encode` run under `torch.inference_mode()`; a module left in training mode is
`E_TRAIN_MODE` (call `.eval()` yourself). A network connection during `encode` or `prepare` is
`E_NETWORK`; a write outside the run directory is `W_SIDE_EFFECT`.

### `describe()` — MUST contain

`describe()` is RECORDED with every number (profile, sidecars, `_meta.json`, `result.json`, results
files) and NEVER part of what makes numbers comparable: comparability is the benchmark and the probe
(`eval_hash`, references/readouts.md), the things yavih controls. The middle column says whether a key
is covered by `describe_sha256`, the consistency digest that keeps one feature set to one
configuration, re-extracts a resumed shard written under another one, re-probes changed features and
tells configurations of one model name apart in the report (`W_BRIDGE_CHANGED`).

| key | in `describe_sha256` | content |
|---|---|---|
| `model` | yes | the model's NAME, chosen by you, `[A-Za-z0-9_.-]{1,64}` starting with a letter or digit (`_stale`, `.x`, `-x` are refused): the SAME string in every bridge of that model, on every benchmark and in every reading. Results are filed under it: run dir `<runs root>/<b>/<model>/<reading>/`, results `<results root>/<model>/`, report rows `(model, reading)`. Two configurations you want side by side (another layer, another checkpoint) are two names; one name re-run with another configuration is one row, newest shown (`W_BRIDGE_CHANGED`). |
| `weights_sha256` | yes | `fingerprint_state_dict(model_or_state_dict)`; null is accepted and recorded as the warning `W_WEIGHTS_UNFINGERPRINTED` |
| `weights_ref` | **no** | path or hub id; provenance only |
| `dtype`, `pos_mode`, `streams` | yes | equal to the class attributes |
| `batch_size` | **no** | equal to the class attribute; provenance (the canaries check batch invariance), so another batch size on another GPU is the same configuration |
| `reading` | yes | `"a"`, `"v"` or `"av"`: what the model sees, one vocabulary across benchmarks (`[A-Za-z0-9_-]{1,32}`, first character a letter or digit; another declared tag is legal). It names the run subdirectory and the results key `<benchmark>:<reading>`, so a different spelling, e.g. `"audio"`, is a different row; the report marks a reading `MISSING` only where the benchmark has its modalities, and `report --join` routes `a`/`v`/`av` |
| `layer`, `pooling` | yes | the tap; `"none"`, `"mean"`, `"cls"`, … |
| `views` | yes | MUST be `1`: one deterministic view, no multi-crop/multi-clip/TTA averaging |
| `input` | yes, except each stream's `decoder` block | per stream you READ: the declared rules of §3 below; `decoder` (`{lib, version, …}`) is recorded as provenance, like the torch or CUDA version |
| `precision` | yes | `{autocast: null\|"bf16"\|"fp16", matmul_precision: "highest"\|"high"\|"medium", allow_tf32: bool, deterministic: bool}`; enforced by run_extract; a null other than `autocast` → `E_PRECISION_UNDECLARED` |
| `args` | yes | the consumed `--arg` values with defaults filled; an unechoed one → `E_ARGS_UNCONSUMED` |
| `batch_tolerance` | **no** | optional `{rel_l2: ≤ 1e-3, reason: "…"}` when you cannot reach `1e-5` |
| `waive` | yes | optional `{CODE: reason}` (`{"*": reason}` = every waivable check): checks that are wrong for this model, each with a reason (§6, "Waiving a check") |

Any further key is allowed, recorded and covered by `describe_sha256` (= sha256 of `describe()`
minus `weights_ref`, `batch_tolerance`, `batch_size` and every `input.<stream>.decoder`): a path
edit, another batch size or a PyAV / soundfile patch release changes nothing; a frame count, crop,
front end, normalization source, argument, precision or dtype change re-extracts on resume and shows
in the report as a new configuration of the same `(model, reading)` row. None of it changes a path,
a key or what a number is compared with. Runs, profiles and results written by a yavih whose digest
still covered the batch size and decoders are recognised and not re-run.

## 3. `describe()["input"]`: declared rules, one per stream read

```python
"input": {"video": {"temporal": {...}, "spatial": {...}, "norm": {...}, "decoder": {...}},
          "audio": {"temporal": {...}, "front_end": {...}, "norm": {...}, "decoder": {...}}}
```

`temporal` is a closed vocabulary so validate can recompute what you should have read from each
row's facts and compare it with your `used` report (`E_RULE_MISMATCH`):

| rule | kinds | params | expected `used` |
|---|---|---|---|
| `uniform` | video | `n`, `span: "full" \| [t0, t1]`, `edge: "error" \| "clamp"` | bin centres `a + (k + 0.5)(b − a)/n` |
| `fps` | video | `fps`, `n`, `start: float \| "centre" \| "centre_clamped"`, `edge`; optional `phase` (default `0.5`), `duration_from: "duration_s" \| "frame_count"` (default `duration_s`) | `s + (k + phase)/fps`: `phase: 0` = window-start sampling. `centre` = `(D − n/fps)/2`, `centre_clamped` = `max(0, …)`. `D` = the container's `duration_s`, or with `frame_count` the track's `n_frames/fps` (on vggsounder / vggss_box the two differ by ~0.1–0.3 s); with `frame_count` the `edge` bound is the last frame `(n_frames − 1)/fps` and the rule needs `n_frames`, refused (`E_INPUT`) where the benchmark does not guarantee it |
| `index_linspace` | video | `n` | indices `round(linspace(0, n_frames − 1, n))`, `t = idx / fps` |
| `explicit` | video | `times`, `edge` | the given absolute times (e.g. a frame on vggss's annotated second) |
| `all` | video | — | every frame (only where the count is constant across the benchmark) |
| `span` | audio | `start_s: float \| "centre"`, `length_s`, `pad: "zeros" \| "none"` | one range of `round(length·sr)` native samples, cut at `n_samples`, shortfall = right pad |
| `full` | audio | `max_s`, `pad_to_s` | `[0, min(n_samples, max_s·sr))`, right-padded to `pad_to_s` |
| `segments` | audio | `n`, `length_s`, `span`, `pad` | `n` ranges centred on the `uniform` times of `span` |
| `custom` | any | `description` (one line: what you do instead) | the OPT-OUT, legitimate whenever the vocabulary does not fit: not evaluated, validate notes `N_RULE_CUSTOM` (never a warning); out-of-media, range and position checks still apply |

With `edge: error` a time outside `[0, duration_s]` is `E_TIMES_OUT_OF_MEDIA`; `edge: clamp`
moves it to the nearer end. How short media are served is your declared decision, not the
benchmark's. Tolerances: 1 ms for explicit times, `0.5/fps + 1 ms` for decoded video
(nearest frame), ±1 sample for audio ranges and padding.

`spatial` (video, image): `{"resize": null | {"short": S} | {"long": S} | {"size": [h, w]},
"crop": null | {"kind": "center", "size": [h, w]} | {"kind": "region", "region": [y0, x0, y1, x1]},
"interp": "bilinear", "antialias": true}` or `{"rule": "custom", "description": …}`.
`crop_region(spatial, width, height)` is the part of the ORIGINAL frame your input shows; validate
checks that your tokens' `(y, x)` lie inside it (skipped for a `custom` spatial rule; the `[0, 1]`
range check stays). `front_end` (audio): free-form, at least the model
`sample_rate`, the resampler and the feature type. `norm`: the constants and their `source` (the
model's own pretraining recipe: repository, commit, file); per-clip statistics only when they ARE that
recipe; never batch or dataset statistics. A model that normalizes nothing declares `{"source": "none:
<why>"}`; a `TODO` source or a null constant fails `bridge check` (`E_TEMPLATE_UNFILLED`). `decoder`:
`{lib, version, …}` as the library reports it.

### What `used` reports, per clip and stream

| kind | `used[i][stream]` |
|---|---|
| video | `{"times": [...]}` — media seconds of the frames fed, in feed order: the pts of the decoded frames |
| audio | `{"samples": [[lo, hi], ...], "pad": [left, right]}` — half-open native-rate sample ranges of the file, and the padding you added |
| image | `{}` |

run_extract copies each row's `info` beside your `used` into the shard (you cannot alter it), and
refuses a missing or ill-formed `used` (`E_USED`).

### Opting out of the rule vocabulary

When none of the rules above describes what your model does — repeat or reflect padding, the model's
own clip sampler with irregular but deterministic spacing, a front end that streams the whole file in
chunks — declare `{"rule": "custom", "description": "<one line: what the bridge does>"}` for that
`temporal` or `spatial` entry. That is fine and often right; do not bend the model to fit the
vocabulary. Validate records the note `N_RULE_CUSTOM` (stream, rule, your description, what still ran)
and `rules_checked` says `"custom"`; the report never greys, flags or footnotes the row. What still
applies to every rule, custom included: reported times and sample ranges inside the media
(`E_TIMES_OUT_OF_MEDIA`), token positions in range and inside the span you reported (`E_POS_RANGE`,
the position part of `E_RULE_MISMATCH`), canaries and the bridge check invariance passes. Where
`validate_describe` refuses a rule because the benchmark's facts cannot evaluate it (`E_INPUT`, e.g.
`index_linspace` on k400), its message offers a rule those facts support or this opt-out.

## 4. Positions in media coordinates and `grid_pos`

`pos` rows are `(stream, t, y, x)`: `stream` = index into `describe()["streams"]`; `t` = the
token's temporal centre as a fraction of ITS stream's `duration_s`; `(y, x)` = its spatial centre
as fractions of the ORIGINAL frame's height and width, before your resize or crop (AS-20K: the
stored frame); `-1` = no coordinate on that axis (images have `t = -1`; a token centred in padding
you added has `t = -1`); all four `-1` = a global token (CLS, pooled). Readouts read tokens where
they lie in the media, so any frame count, crop or resolution works; `clf` ignores pos.

```python
grid_pos(T, H, W, stream, t_centres=None, region=(0.0, 0.0, 1.0, 1.0), cls=False)   # float32 [(1 if cls) + T*H*W, 4]
```

Rows are `(t, y, x)` row-major; `t_centres` = fractions of the duration (e.g. `np.mean(times
of a tubelet) / duration_s`), defaulting to `(k + 0.5)/T` (correct only for `uniform` over `span:
full`); `region = crop_region(spatial, width, height)` maps patch centres into the part of the
original frame you kept: `y = y0 + (i + 0.5)(y1 − y0)/H`. Concatenate per-stream blocks in token
order (e.g. `[cls, video, audio]`). When durations or frame sizes differ between clips (almost
always), `t` and `region` differ too: use `pos_mode = "per_sample"` (≈ 1 % of the feature bytes at
D = 768 fp16). `constant` is legal only when one `[N, 4]` is right for every row. Slice tokens
that are constant across clips (registers, padding) out of the set, or validate refuses
`E_DEAD_ROWS` (unless you waive it with a reason, §6: `"waive": {"E_DEAD_ROWS": "4 register tokens"}`).

## 5. Worked examples, one per media kind

**mp4 / webm video** (k400, ssv2, vggsounder video, audioset20k video): 16 frames as bin centres over the whole clip,
resize-short 256 + centre crop 224, ViT tubelet 2 × 16 × 16 → 8 × 14 × 14 tokens.

```python
spatial = {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}
"video": {"temporal": {"rule": "uniform", "n": 16, "span": "full"}, "spatial": spatial,
          "norm": {"scale": 1/255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "source": "<repo>@<commit> <file>"},
          "decoder": decoder_info()}
# encode, per clip s:
v = s.streams["video"]; d = v.info["duration_s"]
times = rule_times({"rule": "uniform", "n": 16, "span": "full"}, v.info)
frames, pts = frames_at(v, times)                        # report what was really fed: the decoded pts
region = crop_region(spatial, v.info["width"], v.info["height"])
pos_i = grid_pos(8, 14, 14, stream=0, t_centres=pts.reshape(8, 2).mean(1) / d, region=region)
used_i = {"video": {"times": pts.tolist()}}
```

A fixed 10 s K400 window is `{"rule": "uniform", "n": 16, "span": [0, 10], "edge": "clamp"}`; index
sampling as is common on SSv2 is `{"rule": "index_linspace", "n": 16}`.

**wav audio** (esc50, fsd50k, vggsounder audio; the audio track of an mp4 — ave, avsync, audioset20k — the same way
with PyAV): the first 10 s at the native
rate, zero-padded on the right, resampled to 16 kHz, 128-bin fbank, 16 × 16 patches.

```python
"audio": {"temporal": {"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"},
          "front_end": {"sample_rate": 16000, "resample": "torchaudio.functional.resample (sinc_interp_hann, width 6)",
                        "features": "kaldi_fbank", "n_mels": 128, "win_ms": 25, "hop_ms": 10, "channels": "mean"},
          "norm": {"mean": -4.268, "std": 4.569, "source": "<AudioMAE repo>@<commit> (AudioSet-2M recipe)"}, "decoder": {...}}
ranges, (pad_l, pad_r) = audio_ranges(rule, a.info)     # [[0, min(n_samples, 10·sr)]], shortfall on the right
used_i = {"audio": {"samples": ranges.tolist(), "pad": [pad_l, pad_r]}}
# pos: audio patch k covers seconds [k·Δ, (k+1)·Δ) of the 10 s input → t = (k + 0.5)·Δ / duration_s if inside the media, else -1
```

Pad to the fixed length inside the bridge, per clip; never pad to the longest clip of the batch
(that is batch-dependent and fails the alone pass). FSD50K runs 0.3–64 s: decide (and declare)
whether you read the head, the centre or `segments`.

**jpg / png image** (in1k, nyuv2_depth): `{"spatial": {...}, "norm": {...}, "decoder": {...}}`, no
`temporal`; `used = {"image": {}}`; `pos = grid_pos(1, 14, 14, 0, t_centres=[-1.0], region=crop_region(spatial, w, h))`
(`t_centres=[-1.0]`: images have no time). NYUv2 is 640 × 480 and depth is scored on the
centre square `target_region [0, 0.125, 1, 0.875]`: resize-short 224 + centre crop 224 covers it
exactly; feeding the whole frame is equally legal (more cells outside the region).

**audio-visual** (ave, vggss_box, avsync, vggsounder av, audioset20k av): declare both streams in `input`, set
`streams = ("video", "audio")`, tag each token block with its stream index, report `used` for both.
vggss_box hands over `meta["facts"]["annotated_second"] = 5.0`: a single-frame bridge may declare
`{"rule": "explicit", "times": [5.0]}`. avsync's offset is the label and never reaches you; the
shifted audio is already inside the mp4.

## 6. Instance-wise, deterministic: passing `bridge check`

`yavih bridge check` encodes 32 present rows four times at your `batch_size`: **A** (row order),
**rerun** (A again: feats, pos and `used` bit-identical, else `E_RERUN_DRIFT`), **alone** (every
clip in a batch of one with `row`, `clip_id`, `split` scrambled: `rel_l2 ≤ 1e-5` against A, else
`E_BATCH_INVARIANCE`), **shuffled** (permuted and re-batched: same bound). Typical failures and
fixes: batch statistics or BatchNorm in train mode → per-clip normalization from the model's
recipe, `.eval()`; padding to the batch maximum → fixed-length padding plus an attention mask;
a cache keyed on clip id → none; dropout or random crops → off (`views: 1`); non-deterministic
kernels → `precision.deterministic: true`, `allow_tf32: false`, `matmul_precision: "highest"`.
bf16/fp16 autocast kernels often differ between batch sizes by ~1e-4: then declare
`"batch_tolerance": {"rel_l2": 1e-3, "reason": "bf16 SDPA kernels differ between batch sizes"}`
(recorded, printed as `W_LOOSE_TOLERANCE`; never above 1e-3). During extraction, canaries
re-encode rows in another batch composition and validate applies the same bound.

Before any row is read, an unfilled template is refused (`E_TEMPLATE_UNFILLED`: `TODO` left in `model`,
`weights_ref` or a `norm.source`, or a null `norm` constant); after the passes, features equal on every
smoke clip are refused (`E_CONSTANT_FEATURES`: the bridge ignores its input). Which rule is checked by
which code, and which are advisory: SKILL.md "Guardrails". No data staged yet: `yavih bridge check <file>
--bench <b> --synthetic` runs every check on tiny generated media with the benchmark's stream names,
kinds, guaranteed facts and label facts (no profile is written; speed and numbers mean nothing there).

### Waiving a check

When a check is wrong for your model, waive it with a reason — do not edit yavih, do not turn validation off.
`describe()["waive"] = {CODE: reason}` (part of `describe()` and of `describe_sha256`, never of an input, protocol or eval
hash) turns that error into the warning `W_WAIVED` (`"<CODE>: <message> — waived: <reason>"`) in `bridge check` and
validate; every waiver in force is recorded (`profile.json` and `_meta.json` `waived: [{code, reason, source, count}]`,
`result.json`, the results store) and the report marks the row with `†` and a footnote. Nothing is greyed, excluded or
penalized: like a `custom` rule, a waiver with a reason is a legitimate, recorded decision.

Waivable: `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`, `E_POS_RANGE` (coordinates only), `E_BATCH_INVARIANCE`,
`E_RERUN_DRIFT`, `E_CONSTANT_FEATURES`, `E_DEAD_ROWS`, `E_DUPLICATE_CLIPS`, `E_ARGS_UNCONSUMED`; `{"*": reason}`
waives all of them. Never waivable (`E_WAIVE_INVALID`, exit 2): the structural checks (`E_FORMAT`, `E_SHARD_SET`,
`E_RANGE_TILING`, `E_MEMBERS`, `E_ROWS`, `E_TAR_SHA`, `E_POS_SHAPE`, `E_FINITE`, `E_NA_AND_SHARDS`), the identity checks
(`E_HASH_MANIFEST`, `E_HASH_INPUT`, `E_DESCRIBE_MISMATCH`) and `E_TEMPLATE_UNFILLED`; a waiver needs a non-empty reason;
warnings need none. `batch_tolerance` is not a waiver: prefer it while the drift fits under `1e-3`.

Example — VGGSounder, audio reading, a bucketed front end that standardizes each 4-s window: every digitally silent
window becomes the same all-zero input, so ~1.6 % of test clips get byte-identical features although the bridge reads
its input correctly, and validate says `E_DUPLICATE_CLIPS`:

```python
def describe(self):
    return {
        ...,
        "waive": {"E_DUPLICATE_CLIPS": "digitally silent audio: the per-window standardized fbank of silence is all zeros"},
    }
```

For features already extracted, the operator does the same without touching the bridge (no hash changes;
recorded with `source: cli`): `yavih features validate <runs root>/vggsounder/<model>/a/features --waive
E_DUPLICATE_CLIPS="digitally silent audio"` (repeatable; `all=<reason>`), `yavih probe run … --waive …`, or the matrix
entry key `waive: {E_DUPLICATE_CLIPS: "digitally silent audio"}` for the plan's probe units.

## 7. Many benchmarks, one model: `<model>_common.py`

Put what every benchmark shares (model build, weight loading, the forward pass and tap, the common
`describe()` keys) in one module next to the bridges, and keep each per-benchmark file thin:

```python
# bridges/mymodel_common.py
import yavih
from yavih.bridge import ArgSpec, fingerprint_state_dict

class MyModelBase(yavih.Bridge):
    batch_size, dtype, pos_mode = 8, "float16", "per_sample"
    ARGS = {"layer": ArgSpec(type=int, default=12)}
    def __init__(self, bench, args):
        super().__init__(bench, args)
        self.model = build_and_load(...).eval(); self.sha = fingerprint_state_dict(self.model)
    def common_describe(self) -> dict:      # model, weights, layer, pooling, views, precision, args
        return {"model": "mymodel", "weights_sha256": self.sha, "weights_ref": "...", "dtype": self.dtype, ...}

# bridges/mymodel_k400.py
from mymodel_common import MyModelBase

class MyModelK400(MyModelBase):
    streams = ("video",)
    def describe(self):
        return {**self.common_describe(), "reading": "v", "streams": list(self.streams), "input": {"video": {...}}}
    def encode(self, samples): ...           # this benchmark's rule, decode, pos, used
```

yavih inserts the bridge file's directory at the front of `sys.path` before importing it, under `bridge
check` and in every extract unit, whatever the cwd; an import error is shown with its traceback. A bridge
file must define exactly one `Bridge` subclass of its own (or name it in a module-level `BRIDGE`); the
imported base does not count. Run `bridge check` once per matrix entry: each loads the weights itself.

## 8. Template walkthrough (`bridge_template.py` of the benchmark folder)

Start from a copy: `yavih bridge new <b> --model <name> [--reading a|v|av] [--out DIR|FILE] [--force]` writes
the project's `bridges/<name>_<b>.py` with `describe()["model"]` (and the reading, when given) filled in and a header naming the
contract command; `yavih bench show <b> template` prints the pristine template. The template carries the benchmark's stream names,
a `describe()` with the `input` block to fill, `used` reporting, `grid_pos` with `region` and
`t_centres`, TODO markers and a "Preprocessing source" docstring slot. Fill in this order:

1. **Docstring**: paste the model's own preprocessing source verbatim (frame sampling, resize/crop,
   normalization, resampling, mel front end). `bridge check` warns `W_NO_PREPROCESSING_SOURCE`
   when the slot is empty.
2. **Class attributes**: `batch_size`, `dtype`, `pos_mode`, `streams`, `ARGS` (`name` is an optional label).
3. **`__init__`**: load weights from `self.args` (offline; nothing downloaded in `encode`),
   `.eval()`, device, `fingerprint_state_dict` once.
4. **`describe()`**: the table keys — `model` (the one name every bridge of this model uses) and
   `reading` first —, `views: 1`, explicit `precision`, and `input` rules that restate the docstring
   line by line (a `custom` rule with a one-line description where the vocabulary does not fit).
5. **`prepare()` / `encode()`**: `prepare` evaluates your rule on `s.streams[x].info` and decodes
   exactly that (plus the CPU transform); `encode` runs the model on `s.prepared`, reports `used`,
   builds pos; stack to `[B, N, D]`. `NotApplicable("audio-only bridge: no audio stream")`
   from `__init__` or the first `encode` when the model cannot serve the benchmark (recorded,
   exit 0); never fake a reading with zeros or noise.

An unchanged copy — also a fresh `bridge new` copy, which fills only the model name — fails `bridge check`
(`E_TEMPLATE_UNFILLED`: `weights_ref`, `norm.source`, null norm constants; then `E_CONSTANT_FEATURES`): fill it first.

The shipped example is `yavih/bridges/examples/hf_reference.py` (one public HF audio model on
ESC-50, `N = 1`, `pos_mode = "none"`); `yavih doctor --selftest` runs it.

## 9. `bridge check` output explained

`profile.json` (= the `--json` `data`) holds `{bench, args, describe, describe_sha256, model,
reading, N, D, pos_mode, streams, used, media, invariance: {rerun_bit_identical, batch_rel_l2,
order_rel_l2, tolerance, tolerance_reason}, rules_checked, clips_per_s, decode_workers,
peak_gpu_gb, warnings, notes}`.

- `model`, `reading`: copied from `describe()`; `plan` names the run directory
  `<runs root>/<b>/<model>/<reading>/` from them and refuses two matrix entries resolving to one
  `(benchmark, model, reading)`.
- `describe_sha256`, `N`, `D`: what the writer will fix (a pooled bridge → `N = 1`; ViT-B/16 on
  16 × 224² with tubelet 2 → 1 568 tokens). Every extract unit carries `--expect-profile`, so a
  configuration that drifted from the profile (another `describe_sha256`, `model` or `reading`) is
  refused (`E_CONFIG_DRIFT`, naming the differing keys) before writing.
- `used` / `media`: the first row's report and facts: check the times sit where you meant inside
  `duration_s`, audio ranges inside `n_samples`.
- `invariance`: the four passes; a failed hard check exits 2 and writes no profile.
- `rules_checked`: per stream the temporal rule re-evaluated (`"custom"` for an opt-out) and whether
  the spatial rule was checked (`false` for a custom one).
- `notes`: `N_RULE_CUSTOM` per opted-out rule, with your description; nothing to fix.
- `clips_per_s at decode_workers=K`: measured at the worker count `plan` will request (`matrix cpus
  − 1`); `needs.minutes = ceil(1.5 · rows / clips_per_s / 60)`.
- warnings: `W_FP16_RANGE` (use float32), `W_WEIGHTS_UNFINGERPRINTED`, `W_SIDE_EFFECT`,
  `W_NO_PREPROCESSING_SOURCE`, `W_RULE_UNCHECKED` (a checkable rule a null media fact kept from being
  evaluated), `W_LOOSE_TOLERANCE`.

## 10. Running one shard by hand

```bash
python -m yavih.extract --bench <b> --bridge bridges/x.py --split <split> --shard i/n --out $RUN \
    [--arg k=v]... [--canaries 8] [--decode-workers N] [--read-ahead 2] [--deadline slurm|<epoch_s>] \
    [--expect-profile P] [--json]
```

`--out` is the RUN directory (by convention `<runs root>/<b>/<model>/<reading>`, what `plan` passes): shards land in
`$RUN/features/<split>/shard-NNNNNN.{tar,json}`; each member's
`.json` holds `{row, clip_id, media_key, media, used}`. A matching sidecar → skipped (exit 0); one
written under another configuration (`describe_sha256`, `model`, `reading`) or for another benchmark
input side (`benchmark.input_hash`: manifest, splits, streams, contracts) is moved to `_stale/<utc>/`
and re-extracted. A benchmark edit on the scoring side only (`probe.py`, a metric or its pinned
package, a readout version) never touches finished shards: it re-probes. The first unit of a reconfigured split moves EVERY shard of the old configuration or
shard count (also the ones the new plan no longer writes) and the old `_meta.json` there in one step;
with `--expect-profile` nothing moves before the drift check passed (`E_CONFIG_DRIFT`). SIGTERM/SIGUSR1 or the deadline finishes the current
batch, discards the shard's `.tmp` and exits 4. `--read-ahead` batches of bytes are read (and, with
`prepare`, prepared) by `--decode-workers` threads while you encode (`--decode-workers = max(1, SLURM_CPUS_PER_TASK − 1)`
inside SLURM, else `cpu_count // 2`, cap 16).
