# Output schemas and codecs

What a bridge returns per row is declared by the benchmark folder. Without a declaration it is the feature matrix
every 1.0 benchmark uses: one `[N, D]` token array plus optional media positions `[N, 4]` (the default schema
below). A folder whose protocol needs something else (a generated audio clip, per-option log-likelihoods, a text
answer, a depth map) declares `output_contract.schema`, and yavih stores, checks, compares and reads each key
through a **codec**. The core defines no closed list of output types: a folder that needs a type the built-ins do
not cover registers its own codec.

The implementation is `yavih.codecs` (numpy and the standard library; the `wav` codec uses `soundfile` from the
`media` extra when installed and the standard library's `wave` module otherwise).

## Declaring a schema

```yaml
output_contract:
  shape: fixed                 # every symbol of the schema is fixed per feature set
  coordinates: media
  target_frame: null
  requires: {pos: [], streams: [], grid: null, t_bins: null}
  sampling: required           # optional | required: the bridge generates and declares describe()["sampling"]
  schema:
    audio:  {type: wav, channels: 1, sample_rate: 16000, duration_s: [9.5, 10.5]}
    score:  {type: array, dtype: float32, shape: [K], stats: true}
    answer: {type: json, kind: str}
```

Each key names one value per row. Key names match `[a-z][a-z0-9_]{0,31}` and are none of `json`, `used`, `media`,
`row`, `clip_id`, `media_key`, `canary_of`. Every entry has a `type` (a codec name) and the codec's constraints,
plus the keys every codec accepts:

| key | default | meaning |
|---|---|---|
| `optional` | `false` | the bridge may leave the key out; a feature set carries one key set throughout |
| `stats` | `false` | run the codec's sanity statistics (dead positions, duplicate clips, constant clips, float16 range) |
| `exact` | the codec's | compare canaries and passes for equality instead of within the tolerance |
| `constant` | `never` | `allowed`: the value may be identical for every row (the default schema's `pos`) |

`requires` measures media positions of the default feature keys, so a schema without them declares every
`requires` entry empty; a probe scorer (`clf`, `seq`, `grid`, `box`) reads the feature matrix and needs the
default keys.

## The default schema

```python
DEFAULT_SCHEMA = {
    "feat": {"type": "array", "dtype": ["float16", "float32"], "shape": ["N", "D"], "stats": True},
    "pos":  {"type": "array", "dtype": "float32", "shape": ["N", 4], "optional": True, "constant": "allowed", "exact": True},
}
```

A folder that declares no schema has it, and so does one that declares exactly it: both hash the same, and the
11 packaged folders' hashes did not move when schemas were introduced. Its feature sets are written exactly as in
format 4 ([features](features.md)).

## Built-in codecs

| codec | value | constraints | distance |
|---|---|---|---|
| `array` | a numpy array (a tensor is converted) | `dtype` (one or a list), `shape` (ints, symbols `[A-Z][A-Z0-9_]*`, or `"*"`), `finite` (default true), `fp16_guard` (default true: refuse `max\|x\| > 1e4`, warn above `1e3`) | `rel_l2` in float64, net of one float16 ulp; `0` for identical bytes |
| `json` | any JSON value | `kind` (`any`, `str`, `number`, `bool`, `object`, `list`), `max_bytes` | exact: `0` iff the canonical JSON is equal |
| `bytes` | an opaque file | `container` (the member extension, required), `max_bytes` | exact: sha256 equality |
| `wav` | a wav file's bytes | `channels`, `sample_rate` (one or a list), `duration_s` `[lo, hi]`, `max_bytes` | `rel_l2` over the decoded samples; `inf` when the rate or length differs |

A symbol binds on the first row that carries it and must hold the same value for every other row and key of the
feature set (`feat [N, D]` and `pos [N, 4]` share `N`); `"*"` is free per row.

## Writing a codec

A codec is a `yavih.codecs.Codec`: `name`, `version`, `ext` (the member extension), `encode(value, spec) -> bytes`,
`decode(bytes, spec) -> value`, `check(value, spec, bound) -> [CodecFinding]`, `distance(a, b, spec) -> float`, and
optionally `exact`, `spec_check(spec) -> [problem]`, `stats(spec, bound) -> Stats`, `synthetic(spec, bound, rng)`
and `nbytes(spec, bound)`. A folder exports its codecs from `eval.py`:

```python
from yavih import codecs as C

CODECS = {"pointcloud": C.Codec(name="pointcloud", version=1, ext="ply", encode=..., decode=..., check=...,
                                distance=..., synthetic=...)}
```

A package shares codecs through the entry-point group `yavih.codecs` (a module exporting `CODECS`). A name that
redefines a built-in is refused.

**The self-test.** A codec whose distance never sees a change would let any bridge through the determinism gates.
`yavih bench lint` therefore runs `yavih.codecs.self_test` on every folder or entry-point codec the schema names
(`E_CODEC_SELFTEST`): a synthetic value must round-trip through `encode` / `decode` losslessly, `distance(x, x)`
must be `0`, `distance(x, y)` must be positive for a second sample, and `check` must accept the codec's own
sample.

## What enters the hash

`output_contract` is on the input side of the protocol hash through `hashing.output_contract_view`: the block as
written, without a `schema` that is absent or equal to the default, plus `codecs: {name: version}` for the codecs
a declared schema names. Bumping a codec's `version` therefore moves the input hash of exactly the folders that
use it. A codec's source is provenance, held to its declared version by the folder's conformance fixture and the
self-test, never hashed ([hashing](hashing.md)).
