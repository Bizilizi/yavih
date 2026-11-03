# Streaming benchmarks

A streaming benchmark scores an encoder that observes a recording causally: at every update time it may use what it
has seen up to then, never what comes after. yavih hands the bridge one recording per manifest row; the bridge returns
the recording's whole update sequence; the `stream` scorer trains causal heads on the sequences and reads, at every
scored time, the latest prediction that was available by then.

Two execution modes are scored separately:

- **causal** (unrestricted time): every update on the grid is answered and the stream pauses while the model
  computes. This isolates what the representation can predict from how fast the hardware is.
- **real time**: the recording is replayed at 1x on one device; an update counts from the moment it is ready.
  A slow model answers fewer requests, later.

## The folder

```yaml
scorer: stream
output_contract:
  shape: sequence                     # one row = one recording, its update sequence per row
  coordinates: media
  target_frame: null
  requires: {pos: [], streams: [], grid: null, t_bins: null}
  stream: {update_interval_s: 0.5}    # updates at k * 0.5 s, k >= 1, up to the recording's duration
  schema:                             # exactly yavih.codecs.STREAM_SCHEMA, written out (its codec versions are hashed)
    emb:     {type: array, dtype: [float16, float32], shape: ["*", D], stats: true}
    support: {type: array, dtype: float64, shape: ["*", 4], exact: true, finite: false}
    cutoff:  {type: array, dtype: float64, shape: ["*"], exact: true}
    timing:  {type: json, kind: object, optional: true, volatile: true}
readout_params:
  groups: {action: {loss: bce, classes: 100}}   # label groups: ce (one class per target) or bce (multilabel)
metrics:
  list: [map_linear, map_gru]          # every metric names the head it scores: _linear or _gru
```

`eval.py` defines `targets(bench, split, rows)`, returning per row the scored times and their labels:
`{"times": [m], "labels": {group: int [m] (ce, -1 = not scored) | float [m, C] (bce, NaN = not scored)}, "meta":
{name: [m]}}`. Its metrics receive `{group: probabilities [M, C]}`, `{group: labels}` and a `split_meta` with the
per-target `recording` (intervals resample recordings: `yavih.metrics.stream.recording_interval`), `time`, `meta`,
`groups` and `head`. `yavih.metrics.stream` has the online-detection metrics (per-frame mAP, calibrated AP, per-stage
AP, macro AP over bins, class-mean top-k recall), checked against the reference evaluators.

## The bridge

A streaming bridge is an ordinary bridge with a declared output schema. `encode(samples)` returns, per recording,
`emb [n, D]`, `support [n, 4]` (`[video_start, video_end, audio_start, audio_end]` in seconds of the recording, NaN for
a modality the update did not read), `cutoff [n]` and, in real time, `timing`. The helpers in `yavih.stream` run the
schedule; the bridge supplies `step(cutoff, since) -> (emb [D], support [4])`:

```python
from yavih import stream

def encode(self, samples):
    outs = []
    for s in samples:
        step = self.stepper(s)             # decodes up to each cutoff; a native model keeps its state in the closure
        dur = stream.recording_duration({n: st.info for n, st in s.streams.items()})
        outs.append(stream.causal(step, duration_s=dur, update_interval_s=self.dt))   # or stream.replay(...)
    ...
```

`stream.replay` is the real-time mode: requests arrive at the grid cutoffs on a 1x clock; one job runs at a time; a
request that arrives while one is pending replaces it (the replaced one is logged as skipped); a job answers its
request's cutoff; `ready_at` is taken after the step returns (and after `torch.cuda.synchronize()`). `buffer_s` bounds
the raw-observation buffer (an evicted interval is logged as a gap). Every model uses this one scheduler. Under `bridge
check`'s forced causal passes `replay` runs the causal mode and returns `timing: None` (the engine drops it), so a
bridge returns the same keys in both.

`describe()["stream"]` records `{mode, native, video_window_s, audio_window_s, num_frames, frame_sampling,
startup_policy, gap_policy, device}`. A real-time bridge has `batch_size: 1`, lists `timing` in `outputs` and its
`reading` ends with `-rt`, so its numbers are rows of their own; the report shows a real-time row as
`reading@device` (a real-time number is comparable only on one hardware configuration).

## What yavih checks

| check | where | code |
|---|---|---|
| no update's support ends after its cutoff or spans more than the declared window | extract, validate | `E_FUTURE_SUPPORT` |
| cutoffs strictly increasing, on the grid, within the recording; every grid cutoff in the causal mode | extract, validate | `E_FEATURES` / `E_MEMBERS` |
| the real-time log: each request answered or skipped once, no job before its request, serial jobs, `ready_at >= started_at` | extract, validate | `E_REPLAY_LOG` |
| determinism (rerun, shuffled, alone) on the causal path of the same code (`stream.forced_causal()`) | bridge check | `E_RERUN_DRIFT`, `E_BATCH_INVARIANCE` |
| future invariance: lossless copies cut a little after a cutoff, one intact and one blanked after the cutoff (black video, silent audio); every update up to the cutoff must be equal | bridge check | `E_FUTURE_LEAK` |
| a real-time smoke run: the replay's updates before its first skip equal the causal run's | bridge check | `E_REPLAY_LOG` |

The timing log is `volatile`: stored and checked, never compared by the determinism gates; a real-time set's canaries
are not compared (its determinism is the bridge check's).

## The `stream` scorer

Both heads are trained per seed on the train split's update sequences and selected on a held-out tenth of the train
recordings by validation loss over the pinned lr grid:

- `linear`: the held embedding, standardized with train statistics, through one linear layer per label group.
- `gru` (`causal_gru`): `[emb, elapsed / update_interval]` projected to 512, one unidirectional GRU layer of 512, a
  linear layer per group; one step per completed update, the state carried through the recording (truncated
  back-propagation through chunks of 256 updates).

A target at time `t` reads the latest update available by `t`: its cutoff in the causal mode, its `ready_at` plus
the head's measured per-update latency in real time. Before the first available update the prediction is the train
prior; that interval is scored like any other. Floors are the prior at every target.
