"""Bridge skeleton for **unav100_causal** (read setup.md and https://github.com/Bizilizi/yavih/blob/main/docs/streaming.md first). Copy, rename, fill the TODOs.

``yavih bridge check <file> --bench unav100_causal`` runs the determinism gates on the causal path, the future-invariance
gate (lossless copies blanked after a cutoff: every earlier update must be equal) and, for ``mode=realtime``, a 1x
replay. A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model.

The "Preprocessing source" slot below is REQUIRED: paste the model's own input pipeline into it, and make
``describe()["input"]`` and ``describe()["stream"]`` match it line by line.

Preprocessing source: TODO <repo>@<commit> <file>::<function>, e.g. the model's build_eval_transform
    TODO paste the source lines: frames per window and their sampling over the window, resize / crop,
    TODO normalization constants (the pretraining recipe's), audio resampler, fbank/mel parameters, audio window
    TODO length and how a window that starts before the recording is padded (with constants, never future input).

What unav100_causal hands ``encode`` (setup.md "Input contract"): per Sample, two RawStreams over the SAME stored mp4
(raw_videos/<video_id>.mp4: h264 + AAC, 9.5-60 s, 6-52 fps, up to 1280x720). ``encode`` returns, per recording, its
update sequence: one update every 0.5 s of the recording (cutoffs 0.5, 1.0, ..., up to its duration; ~84 for a 42 s
video):

    emb      [n, D]   the embedding available after each update (same D for every recording)
    support  [n, 4]   [video_start, video_end, audio_start, audio_end] in seconds of the recording, of the input the
                      update actually read (NaN for a modality it did not read); no end after the update's cutoff
    cutoff   [n]      the update's cutoff (yavih.stream fills it)
    timing            real time only: the replay log (yavih.stream.replay fills it)

The schedule is yavih's: ``yavih.stream.causal`` (unrestricted time) or ``yavih.stream.replay`` (1x real time, one
job at a time, the latest pending request wins). Your ``step(cutoff, since)`` reads only input at or before ``cutoff``
and returns ``(emb [D], support [4])``. An offline encoder reads its trailing window ``[cutoff - W, cutoff]``; a
native streaming model consumes ``(since, cutoff]`` and keeps its state in the closure (reset per recording).

Gotchas (setup.md "Gotchas"):
    * Never read past the cutoff: no centered window, no look-ahead audio frame, no normalization over the recording.
      bridge check's future-invariance gate (E_FUTURE_LEAK) compares updates on copies blanked after a cutoff.
    * Report the support you actually read (decoded frame pts, first/last audio sample), not the window you asked
      for; a window that starts before 0 is padded with constants and its support starts at 0.
    * A real-time reading ends with ``-rt`` and runs with batch_size 1 on one declared device.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih import stream
from yavih.bridge import ArgSpec, Bridge

# ---- TODO: the model's choices (placeholders: an encoder over a trailing window of 2 s, 8 frames, and 2 s of audio)
VIDEO_WINDOW_S = 2.0
AUDIO_WINDOW_S = 2.0
N_FRAMES = 8
D = 768


class Unav100CausalBridge(Bridge):
    """TODO: the model's name. A trailing-window encoder over the stream (offline baseline)."""

    batch_size = 1
    outputs = ("emb", "support", "cutoff")
    ARGS = {
        "reading": ArgSpec(default="av", choices=("av", "v", "av-rt", "v-rt"), help="modalities; -rt = the 1x real-time replay"),
        "device": ArgSpec(default="cuda:0", help="the one device a real-time replay runs on (recorded)"),
    }

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        self.dt = float(bench.output_contract["stream"]["update_interval_s"])
        self.realtime = str(self.args["reading"]).endswith("-rt")
        if self.realtime:
            self.outputs = ("emb", "support", "cutoff", "timing")
        self.audio = str(self.args["reading"]).startswith("av")
        # TODO: load the model in eval mode on self.args["device"]; fingerprint its weights
        self.model = None

    def describe(self) -> dict:
        return {
            "model": "TODO-model-name",
            "reading": self.args["reading"],
            "weights_sha256": None,  # TODO: yavih.bridge.fingerprint_state_dict(self.model)
            "weights_ref": "TODO <hub id>@<revision>",
            "batch_size": self.batch_size,
            "views": 1,
            "outputs": list(self.outputs),
            "input": {"video": {"norm": {"source": "TODO the pretraining recipe's mean/std"},
                                "decoder": {"lib": "TODO pyav", "version": "TODO"},
                                "temporal": {"rule": "custom", "description": f"TODO {N_FRAMES} frames uniform over the trailing "
                                             f"{VIDEO_WINDOW_S} s window ending at each cutoff"},
                                "spatial": {"rule": "custom", "description": "TODO resize / crop"}},
                      **({"audio": {"norm": {"source": "TODO"}, "decoder": {"lib": "TODO", "version": "TODO"},
                                    "temporal": {"rule": "custom", "description": f"TODO the trailing {AUDIO_WINDOW_S} s ending at each cutoff"},
                                    "front_end": {"sample_rate": 16000, "resample": "TODO", "features": "TODO"}}} if self.audio else {})},
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),
            "stream": {"mode": "realtime" if self.realtime else "causal", "native": False,
                       "video_window_s": VIDEO_WINDOW_S, "audio_window_s": AUDIO_WINDOW_S if self.audio else None,
                       "num_frames": N_FRAMES, "frame_sampling": "TODO uniform over the trailing window",
                       "startup_policy": "TODO pad a window that starts before 0 with its first frame / zeros",
                       "gap_policy": "none (stateless windows)", "device": str(self.args["device"])},
        }

    def _step_fn(self, sample: Any):
        """TODO: decode once per recording, then encode the trailing window at each cutoff."""
        def step(cutoff: float, since: float):
            v0 = max(0.0, cutoff - VIDEO_WINDOW_S)
            a0 = max(0.0, cutoff - AUDIO_WINDOW_S)
            emb = np.zeros(D, dtype=np.float32)  # TODO: self.model(frames in [v0, cutoff], audio in [a0, cutoff])
            support = [v0, cutoff, a0 if self.audio else np.nan, cutoff if self.audio else np.nan]  # TODO: the pts actually read
            return emb, np.asarray(support, dtype=np.float64)

        return step

    def encode(self, samples: list[Any]) -> dict[str, Any]:
        outs = []
        for s in samples:
            dur = stream.recording_duration({n: st.info for n, st in s.streams.items()})
            run = stream.replay if self.realtime else stream.causal
            outs.append(run(self._step_fn(s), duration_s=dur, update_interval_s=self.dt, dim=D))
        res = {k: [o[k] for o in outs] for k in self.outputs}
        res["used"] = [{"video": {"times": [float(t) for t in o["support"][:, 1]]},
                        **({"audio": {"samples": [], "pad": [0, 0]}} if self.audio else {})} for o in outs]
        return res
