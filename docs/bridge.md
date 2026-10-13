# Bridge contract

A bridge is a `yavih.Bridge` subclass in one `.py` file that turns one benchmark's stored media into a
fixed-shape token set `[N, D]` per clip. yavih reads the bytes, batches the rows, writes the shards, validates,
probes and reports; the benchmark owns only what the label depends on. Everything model-side (which frames, at
what times, which crop, resolution, resampler and audio front end, which normalization constants, which tap,
pooling, precision, storage dtype and batch size) is the bridge's decision, and the bridge declares each one in
`describe()`.

This page is the normative reference for that contract. Related pages: what the features look like on disk is
[features](features.md); every validation code and the waiver rules are in [validation](validation.md);
`describe_sha256` is defined in [hashing](hashing.md); read-ahead and decode workers are in
[execution](execution.md); the benchmark's `streams` and `input_contract` are in
[benchmark-folder](benchmark-folder.md).

## What a bridge receives

`Bridge.__init__` receives a `BenchmarkView`; `prepare` receives one `Sample`, `encode` lists of them. None of
them carries the manifest, labels, splits or data roots.

```python
class RawStream:                          # read-only (__slots__; assignment raises AttributeError)
    name: str                             # stream name from benchmark.yaml `streams` (the pos tag vocabulary)
    kind: str                             # "video" | "audio" | "image"
    path: str | None                      # OPAQUE absolute path to the stored file (layout dir); None for a tar member
    member: tuple[str, int, int] | None   # (tar path, data offset, size) for layout tar_index; None otherwise
    info: dict                            # MediaInfo (below), a fresh dict on every access
    def bytes(self) -> bytes              # the stored file byte for byte (tar member: the member's bytes); cached after the first read
    def open(self) -> BinaryIO            # read-only, seekable file object over the same bytes

@dataclass(frozen=True)
class Sample:
    row: int                              # manifest row index within the split: the feature key
    clip_id: str                          # manifest key, unique per row (several rows may share a media file)
    split: str
    streams: dict[str, RawStream]         # every present stream, under its declared name
    meta: dict                            # {"facts": {key: protocol[key] for key in input_contract.facts}}
    _canary_of: int | None                # private: a bridge MUST NOT read it
    prepared: Any                         # what Bridge.prepare returned for this sample (None without prepare)

@dataclass(frozen=True)
class BenchmarkView:                      # Benchmark.view(); JSON primitives only
    name: str; aliases: list[str]; readout: str; protocol_version: int; protocol_hash: str
    streams: dict[str, str]               # {name: kind}
    input_contract: dict; output_contract: dict
    facts: dict                           # = Sample.meta["facts"] of every row
    input_hash: str                       # the benchmark's input-side digest (hashing.md)
```

PyAV, soundfile and pillow all accept the object `open()` returns, so a tar member decodes in place.

### Rules

- **Raw means stored.** `bytes()` returns the stored bytes. Nothing is decoded, resampled, trimmed, cropped,
  resized, padded, selected or normalized, and no statistic of any kind is computed. What "stored" means for a
  benchmark is its `input_contract.handed_over` and its setup.md (avsync: the materialized mp4 written at prepare
  time; nyuv2_depth: the lossless PNG re-containering; audioset20k: the published clip mp4).
- **Shared files.** Two streams may be the two tracks of one file (ave, avsync, audioset20k: `video` and `audio`
  of one mp4). Both RawStreams return the same bytes, read once, and each carries its own track's `info`. They
  also share one opaque `path`.
- **A file without the track is still handed over.** When the stored file of a `video` or `audio` stream is
  readable but holds no track of that kind (3 audioset20k mp4s have no audio track), the row is not missing. The
  stream arrives with the same bytes, `info["has_track"]` is `False`, the track facts (video: `duration_s, fps,
  n_frames, width, height`; audio: `duration_s, sample_rate, n_samples, channels`) are null whatever
  `input_contract.info` guarantees, and `container` / `bytes` stay set. The benchmark never decides for the
  model: the bridge picks what to feed (its own silence, a blank frame, a learned "absent" input), declares it in
  `describe()` (convention: `"no_track": {"audio": "<one line>"}`), still returns one fixed-shape row, and
  reports an empty consumption for that stream (`{"times": []}` or `{"samples": [], "pad": [0, 0]}`). Branch
  on `has_track` before evaluating a rule or decoding: `yavih.media.decode` raises `DecodeError` on the absent
  track, and raising `MediaUndecodable` would drop the row.
- **`meta` holds facts only**: the protocol keys named in `input_contract.facts` (vggss_box:
  `{"facts": {"annotated_second": 5.0}}`). It never holds a label, a label-derived value (avsync's offset), a
  manifest column, the canary flag or the media key.
- **Identity fields do not change the output.** `row`, `clip_id`, `split`, `_canary_of`, the `path` handle and
  anything in `meta` other than `meta["facts"]` MUST NOT change what the bridge computes. `bridge check`
  scrambles them (below).
- **`path` is an opaque handle, not a name.** It is a symlink in a per-process temporary directory
  (`yavih-media-*`, removed at exit) whose basename is `sha256(salt, stored path)[:24]` plus the stored extension,
  so decoders that sniff the extension keep working. The stored directories and file name (k400
  `val/<class>/`, in1k `val/<wnid>/`, ESC-50 `…-<target>.wav`) never reach the bridge. The salt is the manifest
  hash, split and row; a canary and every `bridge check` pass after the first get their own salt, so equal media
  under two rows, a canary and its original, and two check passes have different paths. Two streams of one file
  in one Sample share one handle.
- **Missing media never reaches the bridge.** A row whose media is absent or unreadable (absent file, absent
  index entry, short read, or a header that cannot supply a fact `input_contract.info` guarantees) raises
  `MediaMissing(clip_id, stream, reason)` in the reader or the read-ahead; `run_extract` records it as missing
  ([features](features.md)). A `required: false` stream whose media is absent is omitted from `Sample.streams`.
- **Batches.** `run_extract` hands `encode` lists of up to `batch_size` Samples in row order. A batch can be
  shorter (the last one, or after missing rows were dropped), and canaries arrive in batches of their own.

## Media facts

`RawStream.info` always carries every key below, null where unknown or not applicable. The keys
`input_contract.info[stream]` lists are non-null on every handed-over row, except the track facts of a stream
with `has_track: false`. `input_contract.info` may only name keys valid for the stream's kind (and never
`has_track` or `rotation`).

| key | kinds | meaning |
|---|---|---|
| `duration_s` | video, audio | the track's duration in seconds: video = the stream's duration, else the container's; audio = `n_samples / sample_rate` |
| `fps` | video | average frame rate (`avg_frame_rate`) |
| `n_frames` | video | frames in the track: the container's count, else the census's demuxed packet count (never decoded) |
| `sample_rate`, `n_samples`, `channels` | audio | native rate, per-channel sample count, channel count |
| `width`, `height` | video, image | stored frame size in pixels (coded size; a rotation tag is recorded as `rotation` when present) |
| `container` | all | format name: `mp4`, `webm`, `wav`, `jpeg`, `png`, ... |
| `bytes` | all | stored size |
| `has_track` | video, audio | `true` when the file holds a track of the stream's kind, `false` when a readable file holds none; null for image streams and when a census does not record it |

### Where facts come from

Facts are header facts, read without decoding a sample. In order of preference:

1. **The site census**, `$YAVIH_HOME/media_info/<name>-<manifest_hash12>.jsonl.gz`: one line
   `{"stream", "key", "info"}` per distinct media cell, written by `yavih bench verify <b> --census`, which
   probes every present file. Its sha256 is recorded in `data_status.json` and copied into sidecars as
   `benchmark.media_info_sha256`. With the census present, the model env needs no yavih decoder.
2. **A live probe** of the same file the first time `info` is read: `yavih.media.info.probe_info(file, kind)`
   with PyAV, soundfile or pillow (`yavih[media]`); a video whose header lacks a guaranteed `n_frames` (webm) is
   demuxed for its packet count.
3. **The `ffprobe` CLI** when those Python backends are absent: the same libav and the same facts (a tar
   member's bytes go through stdin), only slower.

Facts are site facts: provenance, never hashed. `run_extract` copies each row's `info` into the member next to
the bridge's `used` report; validate reads that copy and never re-probes.

### DecoderMissing

A stream kind with no census and none of the header backends would fail every row with `DecoderMissing`.
`extract` and `bridge check` refuse at start instead (exit 3, naming the fixes: install `yavih[media]`, put
`ffprobe` on `PATH`, or copy a census made elsewhere), and `yavih doctor --bench` warns `W_HEADER_BACKEND`.

## The `yavih.media` helpers

The helpers carry no protocol semantics. A bridge may decode with its own stack (torchvision, torchaudio,
decord, PIL, opened on `stream.open()` or `stream.path`) and loses nothing.

- **`yavih.media.rules`** (core, numpy only; validate uses the same code):
  - `rule_times(temporal, info) -> float64 [n] | None`: the media seconds a video rule feeds.
  - `audio_ranges(temporal, info) -> (int64 [k, 2], (pad_left, pad_right)) | None`: the half-open native-rate
    sample ranges an audio rule feeds and the padding it adds.
  - `crop_region(spatial, width, height) -> (y0, x0, y1, x1) | None`: the part of the original frame a spatial
    rule keeps, in fractions.
  - `select_nearest(pts, times) -> int64 [n]`: `nearest_ties_earlier` (below).
  - Each returns `None` for a `custom` rule; a time outside the media under `edge: "error"` (or an audio span
    past the end under `pad: "none"`) raises `OutOfMedia`.
- **`yavih.media.decode`** (`[media]` extra): accepts a `RawStream`, a path or bytes.
  - `frames_at(stream, times) -> (uint8 [n, H, W, 3], pts float64 [n])`: per requested time, the decoded frame
    nearest by pts (ties to the earlier frame). A time outside the
    media gets the first or last frame: the declared `edge` decides whether that is legal, not the helper. The
    returned `pts` are what the bridge reports as `used[stream]["times"]`.
  - `audio(stream, ranges=None) -> (float32 [L, C], rate)`: the native-rate samples of the given ranges,
    concatenated; no resampling, downmix or padding.
  - `image(stream) -> uint8 [H, W, 3]`.
  - `decoder_info(kind=None) -> {lib, version, libav}` for `describe()["input"][s]["decoder"]`.

`nearest_ties_earlier`: for each requested time `t`, choose the index `i` minimising `round(|t - pts[i]|, 9)`
with a strict `<` scan in ascending `i`, so an exact tie goes to the earlier frame. On frames with pts
`0, 0.625, ..., 9.375`, the 8 `uniform` times over `[0, 8]` select indices `[1, 2, 4, 6, 7, 9, 10, 12]`; on pts
`0, 1, ..., 9` they select `[0..7]`.

```python
from yavih.media.decode import frames_at, audio, decoder_info
from yavih.media.rules import rule_times, audio_ranges, crop_region

v = s.streams["video"]
times = rule_times({"rule": "uniform", "n": 16, "span": "full"}, v.info)
frames, pts = frames_at(v, times)                     # report pts, the frames actually fed
a = s.streams["audio"]
ranges, (pad_l, pad_r) = audio_ranges({"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"}, a.info)
wav, sr = audio(a, ranges)                            # float32 [L, C] at the native rate
```

## Canaries

With `canaries=K` the reader yields, after a shard's regular rows, `K` Samples that are exact copies of regular
rows of the same shard. They are drawn without replacement by
`default_rng(sha256(f"{manifest_hash}:{split}:{shard_index}:{canary_seed}"))` from rows at least `2 * batch_size`
positions before the end of the shard; the effective count is `min(K, n_rows_in_shard // 4)`. A canary carries
the private `_canary_of = row` (never in `meta`: a bridge cannot tell a canary from its original) and its own
opaque `path` handles, so a feature cache keyed on the path cannot replay the original. A cache keyed on a hash
of the bytes cannot be detected this way and stays forbidden by the guardrails.

`run_extract` encodes the canaries as their own batches after the regular rows, so each meets a different batch
composition than its original; the drift feeds `E_BATCH_INVARIANCE` in validate. The reader's default is
`K = 0`; `run_extract`, `yavih extract` and `plan` default to 8.

## The `Bridge` class

```python
import yavih
from yavih.bridge import ArgSpec, fingerprint_state_dict

class MyBridge(yavih.Bridge):
    """...

    Preprocessing source:
        <repo>@<commit> <file>::<function>
        <the model's own preprocessing lines, verbatim>
    """
    name = "mymodel-k400"        # optional label, [A-Za-z0-9_.-]{1,64} (default: the class name); recorded only
    batch_size = 8
    dtype = "float16"            # storage dtype: "float16" | "float32"
    pos_mode = "per_sample"      # "none" | "constant" | "per_sample"
    streams = ("video",)         # stream tags used in pos[:, 0]; order = tag index
    ARGS = {"layer": ArgSpec(type=int, default=12)}
    keep_bytes = False           # True: encode() still reads the stored bytes after prepare()

    def __init__(self, bench, args):       # bench: BenchmarkView
        super().__init__(bench, args)      # self.bench, self.args (typed, defaults filled)
        self.model = load(...).eval()      # weights loaded here, never inside encode
        self.sha = fingerprint_state_dict(self.model)

    def describe(self) -> dict: ...        # REQUIRED
    def prepare(self, sample): ...         # optional: per-sample decode ahead of encode -> sample.prepared
    def encode(self, samples): ...         # REQUIRED -> (feats, pos, used)
    def profile(self) -> dict | None: ...  # optional, e.g. {"peak_gpu_gb": ...}
```

- **`ARGS`** maps each consumed `--arg k=v` to an `ArgSpec(type, default, choices, help)` (or an equivalent
  dict). `Bridge.__init__` coerces the values (`bool` accepts `true/false/yes/no/on/off/1/0`), fills defaults and
  checks `choices`; `self.args` has exactly the keys of `ARGS`, so passing a default explicitly changes nothing.
  An unknown or malformed `--arg` is `E_ARG` (exit 1).
- **`decode_workers`** is set by `run_extract` and `bridge check` before the first `encode`: the bridge's CPU
  budget for its own decoding. With `prepare` it is the number of threads `prepare` runs in; a bridge that
  decodes inside `encode` may use it for its own thread pool over the batch ([execution](execution.md)).
- **`prepare` and `keep_bytes`** are optional and not part of `describe()`, so neither moves `describe_sha256`.
  A bridge without `prepare` decodes inside `encode`: no prepare threads, and the stored
  bytes stay readable throughout `encode` ([`prepare()`](#prepare)).
- **One bridge, one benchmark.** `__init__` receives that benchmark's view, and each matrix entry names one
  benchmark. A model read on several benchmarks has one bridge file per benchmark, typically thin subclasses of a
  shared `<model>_common.py`: yavih puts the bridge file's directory first on `sys.path` before importing it.
- **Loading.** A bridge file must define exactly one `Bridge` subclass of its own (an imported base does not
  count), or name the one to use in a module-level `BRIDGE`; otherwise it is a usage error (exit 1).
  `module:Class` is accepted for packaged bridges.

### Start-up sequence

`run_extract` and `bridge check` run the same preflight before any row is read: import the bridge, parse
`--arg`, construct it with the `BenchmarkView` (`NotApplicable` allowed), call `describe()` and
`validate_describe` (exit 2 on any violation), then, when torch is importable in the model env:

- walk the bridge instance's attributes (recursively through containers and objects), the class attributes and
  the bridge module's globals for `torch.nn.Module` objects; any module in training mode is `E_TRAIN_MODE`
  (exit 2, naming it). The bridge calls `.eval()` itself; yavih never fixes it silently.
- enforce `describe()["precision"]` (`torch.set_float32_matmul_precision`, TF32 flags for matmul and cuDNN,
  `torch.use_deterministic_algorithms`; with `deterministic: true` also `CUBLAS_WORKSPACE_CONFIG` and
  `cudnn.benchmark = False`). `autocast` is declared, not applied: the bridge wraps its own forward.
- run every `prepare` and `encode` under `torch.inference_mode()`. Precision flags that changed during extraction are
  `E_PRECISION_DRIFT` (exit 2). The live values go to the sidecar `env`.

## `describe()`

`describe()` MUST return JSON primitives only and contain the keys below. Every key is recorded wherever the
features go (profile, sidecars, `_meta.json`, `result.json`, results files). None of it is comparability, a
path component or a result key: the model is a NAME the user chooses, read in a declared reading. The third
column says whether the key is covered by `describe_sha256`, the consistency digest ([hashing](hashing.md)).

| key | type | digest | meaning |
|---|---|---|---|
| `model` | str `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}` | yes | the model's name, shared by every bridge of that model on every benchmark and in every reading. It names `runs/<benchmark>/<model>/<reading>/` and `<results_root>/<model>/`; with `reading` it is the report's row key. Two configurations to show side by side are two names; one name re-run with another configuration is one row, newest shown (`W_BRIDGE_CHANGED`) |
| `reading` | str `[A-Za-z0-9][A-Za-z0-9_-]{0,31}` | yes | what the model sees: `"a"`, `"v"`, `"av"` (the vocabulary `report --join` routes) or another declared tag. It names the run subdirectory and the results key `<benchmark>:<reading>`, so another spelling is another row |
| `weights_sha256` | 64 lower-case hex \| null | yes | `fingerprint_state_dict(model_or_state_dict)`: sha256 over sorted `(name, dtype, shape, raw bytes)`. `null` is accepted and surfaces as `W_WEIGHTS_UNFINGERPRINTED`; nothing is gated by it |
| `weights_ref` | str | no | path or hub id; provenance only |
| `dtype` | str | yes | equals `Bridge.dtype` |
| `batch_size` | int >= 1 | no | equals `Bridge.batch_size`; provenance (batch invariance is checked by `bridge check` and the canaries) |
| `streams` | list[str] | yes | equals `Bridge.streams`; no duplicates; each a stream of the benchmark |
| `pos_mode` | str | yes | equals `Bridge.pos_mode` |
| `layer` | str \| int | yes | the tap |
| `pooling` | non-empty str | yes | `"none"`, `"mean"`, `"cls"`, ... |
| `views` | int | yes | MUST be `1`: one deterministic view per clip (no multi-crop, multi-clip or test-time-augmentation averaging) |
| `input` | dict | yes, minus each `input.<stream>.decoder` | the declared rules, one entry per stream the bridge reads (below) |
| `precision` | dict | yes | `{autocast: null \| "bf16" \| "fp16", matmul_precision: "highest" \| "high" \| "medium", allow_tf32: bool, deterministic: bool}`; only `autocast` may be null, else `E_PRECISION_UNDECLARED` |
| `args` | dict | yes | echoes every consumed `--arg` with defaults filled (`== self.args`), else `E_ARGS_UNCONSUMED`; a key not in `ARGS` is refused too |
| `batch_tolerance` | dict, optional | no | `{rel_l2, reason}`: a batch- and order-invariance tolerance looser than the default `1e-5`, in `(0, 1e-3]`, with a non-empty reason |
| `waive` | dict, optional | yes | `{CODE: reason}` (`{"*": reason}` = every waivable check): checks this bridge waives, each with a non-empty reason. A non-waivable code or a missing reason is `E_WAIVE_INVALID` (exit 2). See [validation](validation.md) |

Further keys are allowed, recorded and covered by `describe_sha256`; `no_track` (what is fed for a file without
the track) is the conventional one.

`describe_sha256 = sha256(canon(describe_subset(describe())))`, the subset being `describe()` minus
`weights_ref`, `batch_tolerance`, `batch_size` and every `input.<stream>.decoder`. A path edit, a looser declared
tolerance, another batch size or another decoder build therefore neither re-extracts nor counts as configuration
drift; a change of an argument, rule, normalization source, precision or dtype does both. `describe()` must be a
pure function of the bridge's arguments and weights: `bridge check` builds a second instance and refuses one
that describes itself differently.

`validate_describe(d, bridge, bench)` refuses (exit 2; `E_DESCRIBE` unless named): a non-dict or non-JSON
`describe()`; a missing key; a `model`, `reading` or `Bridge.name` outside its charset; a `dtype`,
`batch_size`, `pos_mode` or `streams` that differs from the class attribute; `views != 1`; an `input` block that
fails the rules below (`E_INPUT`); a malformed `precision` (`E_PRECISION_UNDECLARED` for a missing or null field
other than `autocast`); a `batch_tolerance` outside `(0, 1e-3]` or without reason; an invalid `waive`
(`E_WAIVE_INVALID`); an `args` that does not echo `self.args` unless `waive` names `E_ARGS_UNCONSUMED`.

## Declared rules: `describe()["input"]`

`input` is non-empty, its keys are a subset of the benchmark's streams and a superset of `describe()["streams"]`
(a tagged stream is read and declared). Each entry has exactly these keys, by kind:

| kind | keys |
|---|---|
| video | `temporal`, `spatial`, `norm`, `decoder` |
| audio | `temporal`, `front_end`, `norm`, `decoder` |
| image | `spatial`, `norm`, `decoder` |

```python
"input": {
  "video": {
    "temporal": {"rule": "uniform", "n": 16, "span": "full"},
    "spatial":  {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]},
                 "interp": "bilinear", "antialias": True},
    "norm":     {"scale": 1 / 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                 "source": "<repo>@<commit> <file>"},
    "decoder":  {"lib": "pyav", "version": "13.1.0", "libav": "61.19.100"},
  },
  "audio": {
    "temporal":  {"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"},
    "front_end": {"sample_rate": 16000, "resample": "torchaudio.functional.resample(sinc_interp_hann, width 6)",
                  "features": "kaldi_fbank", "n_mels": 128, "win_ms": 25, "hop_ms": 10, "channels": "mean"},
    "norm":      {"mean": -4.268, "std": 4.569, "source": "<repo>@<commit> dataset.py (pretraining recipe)"},
    "decoder":   {"lib": "soundfile", "version": "0.12.1", "libav": None},
  },
}
```

### Temporal rules

A closed vocabulary that `yavih.media.rules` evaluates on each row's `info`, so validate can compare the reported
`used` with it (`E_RULE_MISMATCH`). Unknown parameters are refused.

| rule | kinds | params | expected `used` |
|---|---|---|---|
| `uniform` | video | `n`, `span: "full" \| [t0, t1]`; optional `edge: "error" \| "clamp"` (default `error`) | bin centres `t_k = a + (k + 0.5)(b - a)/n` over `(0, duration_s)` or `[t0, t1]` |
| `fps` | video | `fps`, `n`, `start: float \| "centre" \| "centre_clamped"`; optional `edge`, `phase` in `[0, 1)` (default `0.5`), `duration_from: "duration_s" \| "frame_count"` (default `duration_s`) | `t_k = s + (k + phase)/fps` with `s = start`, `(D - n/fps)/2` (`centre`) or `max(0, (D - n/fps)/2)` (`centre_clamped`); `D = duration_s`, or `n_frames/fps` with `frame_count`, whose `edge` bound is the last frame `(n_frames - 1)/fps` |
| `index_linspace` | video | `n` | indices `round(linspace(0, n_frames - 1, n))`, `t_k = idx_k / fps` |
| `explicit` | video | `times` (non-empty); optional `edge` | the given absolute times |
| `all` | video | none | every frame; legal only where the count is constant across the benchmark (else the writer's `FeatureShapeError`) |
| `span` | audio | `start_s: float \| "centre"`, `length_s`, `pad: "zeros" \| "none"` | one range of `L = round(length_s·sr)` samples from `round(start_s·sr)` (`centre`: `max(0, round((n_samples - L)/2))`), cut at `n_samples`; the shortfall is `pad[1]` under `zeros`, `E_TIMES_OUT_OF_MEDIA` under `none` |
| `full` | audio | `max_s: float \| null`, `pad_to_s: float \| null` | `[0, min(n_samples, round(max_s·sr)))`, right-padded to `pad_to_s` |
| `segments` | audio | `n`, `length_s`, `span`, `pad` | `n` ranges of `length_s` centred on the `uniform` times of `span`; the cuts before 0 and after the end are the padding |
| `custom` | any | `description` (one line: what the bridge does instead) | not evaluated; validate records the note `N_RULE_CUSTOM` |

`edge: clamp` replaces a time outside `[0, duration_s]` by the nearer end; under `edge: error` such a time is
`E_TIMES_OUT_OF_MEDIA`. How short media are served is the bridge's declared decision, not the benchmark's.

**A checkable rule must be evaluable on every row.** The media facts it needs (`rules.rule_facts`: `uniform`,
`fps`, `explicit` need `duration_s`; `fps` with `duration_from: frame_count`, `index_linspace` and video `all`
need `n_frames` and `fps`; audio rules need `sample_rate` and `n_samples`) must
all be guaranteed by the benchmark's `input_contract.info[stream]`, else `validate_describe` refuses the
declaration with `E_INPUT` (e.g. `index_linspace` on k400, whose facts do not guarantee `n_frames`); the message
offers a rule those facts support, or the `custom` opt-out. Where a needed fact is nevertheless null on a row (a
stream the folder lists no `info` for), validate counts the row as unchecked: `W_RULE_UNCHECKED`, the only
warning about rules.

### Spatial, front end, norm, decoder

- **`spatial`** (video, image): exactly `{"resize": null | {"short": S} | {"long": S} | {"size": [h, w]},
  "crop": null | {"kind": "center", "size": [h, w]} | {"kind": "region", "region": [y0, x0, y1, x1]}, "interp":
  str, "antialias": bool}`, or the opt-out `{"rule": "custom", "description": str}`. `crop_region` maps the rule
  and a row's `width`, `height` to the region of the original frame the model input shows (torchvision's resize
  and centre-crop arithmetic); validate checks that the stream's token centres lie inside it. For a custom
  spatial rule that check is skipped and the `[0, 1]` range check of every coordinate remains.
- **`front_end`** (audio): free-form JSON naming at least `sample_rate` (the model's, a positive int),
  `resample` and `features`.
- **`norm`** (every kind): the constants and their non-empty `source`, the model's own pretraining recipe
  (repository, file, commit). Per-clip statistics are allowed only when they are that recipe; never batch or
  dataset statistics. A model that normalizes nothing declares `{"source": "none: <why>"}`.
- **`decoder`** (every kind): `{lib, version, ...}` as the library reports it (`decoder_info()` when using
  yavih's helpers). It is recorded as host provenance and excluded from `describe_sha256`.

**Opting out of the vocabulary.** A `custom` rule is not a lesser declaration. Repeat or reflect padding, a
model's own clip sampler with irregular but deterministic spacing, a front end that streams the whole file in
chunks: declare `{"rule": "custom", "description": "<one line>"}`. Validate records `N_RULE_CUSTOM` and
`rules_checked` says `temporal: "custom"` / `spatial: false`; the report never greys, flags or footnotes such a
row. What applies to every rule, custom included: reported times and sample ranges inside the media
(`E_TIMES_OUT_OF_MEDIA`), token positions in range and inside the media span reported in `used` (`E_POS_RANGE`,
the position part of `E_RULE_MISMATCH`), canaries and the `bridge check` invariance passes.

## `prepare()`

```python
def prepare(self, sample: Sample) -> Any:   # optional; the return value arrives in encode as sample.prepared
```

`prepare` is the per-sample CPU work of the bridge (decode, cut the declared window, resize, crop, normalize,
compute a spectrogram) moved out of `encode`, so it overlaps the model's forward pass instead of idling the GPU.
The contract:

- **Pure and per-sample.** The result is a deterministic function of the one sample it is handed: no model or
  GPU state, no state kept across calls or shared between samples, no randomness. `bridge check` calls it twice
  on fresh copies of every smoke sample and compares the payloads (mappings by key, sequences by position,
  arrays and tensors by shape, dtype and bytes); any difference is `E_PREPARE_NONDETERMINISTIC` (exit 2).
- **Where it runs.** In a pool of `decode_workers` threads named `yavih-prepare-*`, under
  `torch.inference_mode()`, on the batches the byte read-ahead has delivered: up to `--read-ahead + 1` batches
  are prepared while `encode` runs on an earlier one. Every Sample of a batch, canaries included, is prepared
  before that batch reaches `encode`. The payload is set as `sample.prepared` and dropped once the batch is
  written.
- **Bytes.** The read-ahead has read the stored bytes and resolved `info` before `prepare` is called. Unless the
  bridge sets `keep_bytes = True`, `run_extract` releases the bytes as soon as `prepare` returns: `encode` must
  work from `sample.prepared` and `info`, never `bytes()` / `open()` (a read there goes back to the store).
  `bridge check` keeps the bytes of its smoke set for all passes.
- **Missing rows.** It may raise `MediaUndecodable([sample.clip_id], reason)` or `MediaMissing` for media it
  cannot decode: the row is recorded as missing and never reaches `encode`. A `MediaUndecodable` naming another
  clip, or any other exception, is `E_BRIDGE` (exit 5); `DecoderMissing` stops the run (exit 3).
- **Guardrails.** The prepare threads are always audited: a network connect is `E_NETWORK` and a write outside
  the run directory `W_SIDE_EFFECT`, as in `encode`.
- **Payload.** Anything `encode` can consume; the packaged templates return a dict (decoded `frames` / `wav` /
  `x`, their `pts` or sample ranges, the per-stream `used` entry). It is not hashed and never written.

The time spent in `prepare` (summed over samples, across threads) is the sidecar's `timing.prepare_s`.

```python
def prepare(self, sample):
    v = sample.streams["video"]
    frames, pts = frames_at(v, rule_times(TEMPORAL, v.info))      # decode the declared frames
    return {"x": transform(frames), "pts": pts}                      # CPU preprocessing, per clip

def encode(self, samples):
    x = torch.from_numpy(np.stack([s.prepared["x"] for s in samples])).cuda()
    ...                                                              # forward, pos from s.prepared["pts"] and info
```

## `encode()`

```python
def encode(self, samples: list[Sample]) -> tuple[feats, pos, used]:
    # feats: [B, N, D], np.ndarray or anything with .detach().float().cpu().numpy()
    # pos:   None (pos_mode "none") | [N, 4] ("constant") | [B, N, 4] ("per_sample"), float32
    # used:  list of B dicts, one entry per stream in describe()["input"]
```

- **Shapes.** `feats` is `[B, N, D]` with `B = len(samples)`; the writer stores it in `Bridge.dtype`. `N` and `D`
  are fixed for every row of every split: the writer fixes them from the first clip and refuses a mismatch by
  `clip_id` (`FeatureShapeError`). Pad or crop to a fixed length inside the bridge, per clip, never to the batch
  maximum. A pooled bridge has `N = 1`. A `float16` value beyond the float16 range is refused; `bridge check`
  warns `W_FP16_RANGE` above `1e3`. A wrong shape or a `pos` that does not match `pos_mode` is `E_FEATURES`
  (exit 2).
- **`pos` in media coordinates.** Each row is `(stream, t, y, x)`: `stream` = index into
  `describe()["streams"]` (or `-1`); `t` = the token's temporal centre as a fraction of its stream's
  `duration_s`; `(y, x)` = its spatial centre as fractions of the original stored frame's height and width,
  before the bridge's resize or crop. `-1` on an axis means no coordinate there: image tokens have `t = -1`, a
  token centred in padding the bridge added has `t = -1`, all four `-1` mark a global token (CLS, pooled).
  Coordinates outside `[0, 1] ∪ {-1}` are `E_POS_RANGE`; a non-integer or out-of-range stream column is
  `E_POS_SHAPE`. The format is in [features](features.md).
- **`pos_mode`.** `"none"`: no positions (`pos` is `None`). `"constant"`: one `[N, 4]` valid for every row.
  `"per_sample"`: `[B, N, 4]`, needed whenever durations or frame sizes differ between clips, because `t` and
  the crop region then differ too.
- **Stream tags.** `Bridge.streams` lists the stream tags in tag order; concatenate per-stream token blocks in
  token order (e.g. `[cls, video, audio]`), each tagged with its index.
- **`used`** reports per clip what the bridge actually consumed, one entry per stream in `describe()["input"]`:

| stream kind | `used[i][stream]` |
|---|---|
| video | `{"times": [float, ...]}`: media seconds of the frames fed, in feed order (the pts of the decoded frames) |
| audio | `{"samples": [[lo, hi], ...], "pad": [left, right]}`: half-open native-rate sample ranges of the file (ints), and the samples of padding added before/after (ints >= 0) |
| image | `{}` |

A missing or ill-formed `used` (wrong length, a declared stream absent, wrong types, not JSON) is `E_USED`
(exit 2). `run_extract` writes `used` into the member with the row's `info` copied from the RawStreams (the
bridge cannot alter it), where validate checks it against the declared rules ([validation](validation.md)).

`yavih.grid_pos` builds the token centres of a `(t, y, x)` grid:

```python
grid_pos(T, H, W, stream, t_centres=None, region=(0.0, 0.0, 1.0, 1.0), cls=False)  # float32 [(1 if cls) + T*H*W, 4]

# a video tokenized 8 x 14 x 14 with tubelet 2 over 16 decoded frames
region = crop_region(spatial, v.info["width"], v.info["height"])
pos_i = grid_pos(8, 14, 14, stream=0, t_centres=pts.reshape(8, 2).mean(1) / v.info["duration_s"], region=region)
used_i = {"video": {"times": pts.tolist()}}
```

Rows are row-major over `(t, y, x)`. `t_centres` are fractions of the duration in `[0, 1]` or `-1`, defaulting
to `(k + 0.5)/T` (correct only for a `uniform` rule over `span: "full"`). Patch centres map into `region = (y0,
x0, y1, x1)`: `y = y0 + (i + 0.5)(y1 - y0)/H`. The optional CLS row comes first and is `(stream, -1, -1, -1)`.

### Exceptions from the bridge

| raised | where | effect |
|---|---|---|
| `yavih.NotApplicable(reason)` | `__init__`, or `prepare` / `encode` of the first batch | `run_extract` writes `NOT_APPLICABLE.json` and exits 0 ([features](features.md)); later it is `E_BRIDGE` (exit 5) |
| `yavih.MediaUndecodable(clip_ids, reason)` | `encode` | the named rows are recorded as missing and the batch is re-encoded without them (exact, by batch invariance); `MediaMissing` from the bridge's own read is treated the same way |
| `yavih.MediaUndecodable([sample.clip_id], reason)`, `MediaMissing` | `prepare` | the row is recorded as missing and never reaches `encode`; naming another clip is `E_BRIDGE` |
| anything else | `prepare` or `encode` | a bridge bug, not coverage: `E_BRIDGE`, the shard is aborted (exit 5) |

## Guardrails

Requirements on every bridge. yavih enforces them where it can; the rest is advised.

| guardrail | enforced by |
|---|---|
| instance-wise extraction: no batch or dataset statistics, no padding to the batch maximum, normalization constants only from the model's own pretraining recipe | `bridge check` batch- and order-invariance; canaries (`E_BATCH_INVARIANCE` in validate); `norm.source` declared |
| deterministic single view: no random crops, jitter, dropout, test-time augmentation or multi-view averaging | `views == 1`; rerun identity in `bridge check`; `E_TRAIN_MODE` |
| no cross-sample state: no caches keyed on clip id, path or content, no online adaptation, no reading another row's media | every `bridge check` pass after the first gets new Samples under fresh opaque paths; the alone pass runs on a fresh instance with scrambled identities; canaries carry their own paths. A cache keyed on a hash of the bytes in module-global state cannot be detected: advised |
| media only, never the manifest, labels or split | `BenchmarkView`; `meta` holds facts only; opaque `path`; scrambled `row` / `clip_id` / `split` |
| frozen weights under `inference_mode` | `E_TRAIN_MODE`, `torch.inference_mode()` |
| fixed `N` and `D` for every sample | `FeatureShapeError` in the writer; `E_MEMBERS` |
| positions in media coordinates | `E_POS_RANGE`, `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH` (token centres outside the declared crop region; token `t · duration_s` outside the media span reported in `used`; image tokens with `t >= 0`) |
| weights loaded before the first sample; no network and no writes outside the run directory during extraction | an audit hook armed inside `encode` and in every `yavih-prepare` thread: a network `connect` is `E_NETWORK` (exit 2); opening a file for writing outside the run directory is `W_SIDE_EFFECT`, listing the paths. With `--allow-loopback` (or matrix `allow_loopback`), connects to `127.0.0.0/8`, `::1` or `localhost` pass and are recorded |
| `prepare` deterministic and per-sample | `bridge check` prepares every smoke sample twice: `E_PREPARE_NONDETERMINISTIC` |
| the model's own preprocessing source pasted into the bridge and matched by `describe()` line by line | `W_NO_PREPROCESSING_SOURCE` when the docstring slot is unfilled; matching is advised |

**The preprocessing source slot** is a docstring line starting `Preprocessing source` (its paragraph of prose up
to the line ending in `:` is ignored) followed by the pasted source as an indented block (blank lines allowed),
or an inline body after `Preprocessing source: `. A body with a line starting `TODO`, `FIXME` or `TBD`, or
holding only `<...>` placeholders and "e.g." / "paste" prose, is unfilled. The class docstring is searched first,
then the module's; the first filled slot wins, and the warning names every docstring searched.

## `yavih bridge check`

```bash
yavih bridge check bridges/mymodel_k400.py --bench k400 [--n 32] [--arg k=v]... \
    [--decode-workers N] [--out profile.json] [--synthetic] [--allow-loopback]
```

`bridge check` is the gate before extraction: `plan` schedules a matrix entry only from its profile
([execution](execution.md)). It runs the same preflight and the same read-ahead and encode engine as
`run_extract`, then:

1. **Refusals before any row.** `yavih.testing`'s oracle bridge is refused (`E_ORACLE`) unless
   `YAVIH_TESTING=1`. An unfilled template is `E_TEMPLATE_UNFILLED` (exit 2): `TODO` left in `model`,
   `weights_ref` or a `norm.source`, a null `norm`, or a null `norm` constant.
2. **The smoke set**: the first `--n` present rows (default 32) of the first split with role `eval` (else
   `train`, `folds`, `floor`), read once and cached. Rows with missing media are skipped and reported as
   `W_MISSING_ROWS`; no present row at all is `E_SMOKE` (exit 3).
3. **Four passes at the declared `batch_size`**:
   - **A**: batches in row order. A bridge with `prepare` then has every sample A encoded prepared twice more,
     on fresh copies; payloads that differ are `E_PREPARE_NONDETERMINISTIC` ([`prepare()`](#prepare)).
   - **rerun**: A again in the same process; feats, pos and `used` bit-identical, else `E_RERUN_DRIFT`. With
     `precision.deterministic: false` the rerun is held to the tolerance instead (per clip `rel_l2 <= tol`,
     pos and `used` equal), and a non-zero drift within it is `W_RERUN_NONDETERMINISTIC`.
   - **order**: the smoke set permuted with `default_rng(0)` and re-batched; per clip `rel_l2 <= tol` against
     A, pos and `used` equal, else `E_BATCH_INVARIANCE`.
   - **alone**: on a fresh bridge instance (the first is released; the second must describe itself identically,
     else `E_RERUN_DRIFT`, and be in eval mode, else `E_TRAIN_MODE`), every Sample in a batch of its own with
     `row`, `clip_id`, `split` replaced by `10**9 + k`, `"bridge-check-<k>"`, `"bridge-check"`; same test,
     else `E_BATCH_INVARIANCE`.

   Every pass after A receives new Sample objects whose streams hand over the same cached bytes under fresh
   opaque paths, so a cache keyed on the path, the stream object or the clip id cannot replay pass A.
4. **Constant features.** Every smoke clip encoding to the same features is `E_CONSTANT_FEATURES` (the bridge
   ignores its input).
5. **Per-member checks** of validate on pass A: `E_TIMES_OUT_OF_MEDIA`, `E_RULE_MISMATCH`,
   `E_POS_SHAPE` / `E_POS_RANGE`, fixed `N` / `D` (`E_MEMBERS`), `E_FINITE`, the float16 range; plus
   `E_NETWORK`, `W_SIDE_EFFECT`, `W_NO_PREPROCESSING_SOURCE`, `W_WEIGHTS_UNFINGERPRINTED`, `W_RULE_UNCHECKED`,
   `W_LOOSE_TOLERANCE` (a declared `batch_tolerance`) and the note `N_RULE_CUSTOM` per opted-out rule.

`rel_l2 = ||c - o||_2 / max(||o||_2, 1e-12)` per clip over the features in float32. `tol` is
`describe()["batch_tolerance"]["rel_l2"]`, default `1e-5`, at most `1e-3`; a declared looser value is recorded
in the profile and every sidecar and surfaces as `W_LOOSE_TOLERANCE`.

A failed hard check exits 2 and writes no profile. A check the bridge waives in `describe()["waive"]`
(`E_RERUN_DRIFT`, `E_BATCH_INVARIANCE`, `E_CONSTANT_FEATURES`, the waivable per-member checks,
`E_ARGS_UNCONSUMED`) does not exit 2: it becomes the warning `W_WAIVED` (`"<CODE>: <message> — waived:
<reason>"`), and the profile is written with the measured numbers and `waived: [{code, reason, source, count}]`.
A second instance that describes itself differently stays `E_RERUN_DRIFT` whatever is waived.

`--synthetic` runs every check on a tiny generated stand-in of the benchmark (its stream names and kinds,
guaranteed facts and label facts) before any data is staged; it writes no profile and refuses `--out`.

### `profile.json`

Written atomically to `--out` (and returned as the `--json` `data`):

| key | content |
|---|---|
| `bench`, `split`, `n` | the benchmark, the smoke split and the number of smoke rows |
| `bridge_name`, `source_sha256`, `source_path`, `args` | the bridge label, its file's digest and path, the parsed args |
| `describe`, `describe_sha256`, `model`, `reading` | the declaration; `plan` names the run directory from `model` / `reading`, and every extract unit carries `--expect-profile`, so a drifted configuration is `E_CONFIG_DRIFT` before anything is written |
| `N`, `D`, `pos_mode`, `streams`, `batch_size`, `dtype` | what the writer will fix |
| `used`, `media` | the first smoke row's report and facts |
| `invariance` | `{rerun_bit_identical, rerun_rel_l2, batch_rel_l2, order_rel_l2, tolerance, tolerance_reason}` |
| `rules_checked` | per stream read: the temporal rule validate re-evaluates (`"custom"` for an opt-out), whether the spatial rule is checked, and `temporal_unchecked_rows` where a fact was null |
| `clips_per_s`, `decode_workers`, `peak_gpu_gb`, `env` | throughput of pass A at the given worker count, peak GPU memory, the live torch / CUDA / precision state |
| `missing`, `warnings`, `notes`, `waived` | skipped smoke rows, warnings (`{code, message, where}`), `N_RULE_CUSTOM` notes, waivers in force |
| `status`, `reason` | `ok`, or `not_applicable` with the bridge's reason |
| `allow_loopback`, `loopback_connects` | only under `--allow-loopback` |
| `yavih_version`, `created` | provenance |

## Example bridges

- Each benchmark folder ships a `bridge_template.py` with the benchmark's stream names, a `describe()` whose
  `input` block is to be filled, `used` reporting, `grid_pos` with `region` and `t_centres`, TODO markers and a
  "Preprocessing source" slot. `yavih bridge new <b> --model <name> [--reading a|v|av]` copies it to
  `bridges/<name>_<b>.py` with the model name filled in; an unchanged copy fails `bridge check`
  (`E_TEMPLATE_UNFILLED`, then `E_CONSTANT_FEATURES`).
- `yavih.bridges.examples.hf_reference` (inside the wheel; `[media]` and torch at call time) is one public HF audio
  model on ESC-50 (`N = 1`, `pos_mode = "none"`), decoding with `yavih.media.decode`; `yavih doctor --selftest`
  runs it.
- `yavih.testing.random_bridge` derives features from a seed and the sha256 of the bytes of every stream it
  reads: a pure function of the media that passes every invariance check, declares `custom` rules and reports
  empty `used`. `yavih.testing.oracle_bridge` encodes the label on purpose (a test fixture that breaks the
  guardrails) and is refused by `bridge check` unless `YAVIH_TESTING=1`.
