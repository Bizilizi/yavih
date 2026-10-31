# vggsounder_v2a — setup

## Obtain the data

This benchmark reads the VGGSound corpus as staged for the `vggsounder` benchmark (`yavih bench info vggsounder
--setup`): `video/<clip_id>.mp4` and `audio/<clip_id>.wav` under its media root (`${datasets.vggsound}`). Only the
3 000 frozen clips are read (about 4.5 GB of muted copies are written). YouTube clips decay: prefer an existing
copy of the corpus over a re-download. Nothing else is downloaded for the data; the evaluation code and its
weights are below.

```sh
yavih bench prepare vggsounder_v2a                               # lists the steps
yavih bench prepare vggsounder_v2a 01 -- --workers 8             # mute the 3 000 frozen clips into derived_root
yavih bench prepare vggsounder_v2a 02                            # bench verify --census + reference audio present
```

Disk: about 4.5 GB under `derived_root` for the muted videos, plus about 1 GB for the reference feature caches the
scorer writes there on first use.

## Dependencies

The scorer runs the MMAudio authors' evaluation code from a git checkout pinned to one commit (it loads its
weights from `<checkout>/weights`):

```bash
pip install -e git+https://github.com/hkchengrex/av-benchmark.git@f351b9a6fc6abde746d5f8e1d4c47c883319cb41#egg=av_bench
```

then download its two weight files into the checkout's `weights/` (as its README says):

```bash
cd <checkout>   # e.g. src/av-bench
mkdir -p weights
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt -O weights/music_speech_audioset_epoch_15_esc_89.98.pt
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth -O weights/synchformer_state_dict.pth
```

It needs torch >= 2.5.1 and a CUDA GPU in the scoring environment, and installs PaSST, ImageBind, LAION-CLAP and
MS-CLAP forks; `yavih doctor --bench vggsounder_v2a` checks the
assert-import (`av_bench`, version 1.0.0). Prepare-time: ffmpeg `4.4.2-0ubuntu0.22.04.1` (the pinned build that mutes
the videos).

Its four model forks are git dependencies without a pinned commit, and its other dependencies are unpinned. The
folder was validated with this resolution (record yours with `pip freeze`):

| package | version |
|---|---|
| `laion-clap` | git+https://github.com/hkchengrex/CLAP.git@590b52df03a5e480634b8714135d3534313be860 |
| `passt` (hear21passt) | git+https://github.com/hkchengrex/passt_hear21.git@1ab778984f67cca09dd8bbad11f728a0122cce24 |
| `imagebind` | git+https://github.com/hkchengrex/ImageBind.git@9989650c87d393d7e8c144194182cbf124cd03a0 |
| `msclap` | git+https://github.com/hkchengrex/MS-CLAP.git@3ad265d353aea4d6d4baa4bec2545184f0a841e1 |
| `torch` / `torchaudio` / `torchvision` | 2.5.1 / 2.5.1 / 0.20.1 (CUDA 12.1 builds) |
| `transformers` | 4.49.0: keep it below 4.50; transformers 5 disables itself on torch 2.5 and av-benchmark's encoders fail to import |

Keep the installed torch when installing (a constraints file holding the three torch versions above). On first use the
encoders download their own checkpoints into the user's caches (about 8 GB): PANNs Cnn14 (`audioldm_eval`), VGGish
(`torch.hub` from `harritaylor/torchvggish` at its `master` branch: not pinned), PaSST, ImageBind huge and MS-CLAP. The
two weight files above are checked by sha256 before every scoring run (`eval.py` `WEIGHTS`).

## Input contract

`video`: the muted mp4 of the clip. prepare/01 copies the stored VGGSound video stream (`-an -c:v copy`): no frame is
decoded, re-encoded, cut or resized, and the soundtrack never reaches the bridge. `info` guarantees `duration_s`, `fps`,
`width`, `height`, `container` and `bytes`. The fact `audio_length` (8.0) is in `Sample.meta["facts"]`.

The bridge decides which frames it reads, at what rate and size, how it generates (model, sampler, steps, guidance,
seed) and declares all of it in `describe()` (`input` and `sampling`).

```yaml
input_contract:
  handed_over:
    video: 'the VGGSound mp4 of the clip with its audio track removed (stream copy: the video bytes are the stored
      ones, nothing re-encoded, cut or resized); the soundtrack is the label and never reaches the bridge'
  info:
    video:
    - duration_s
    - fps
    - width
    - height
    - container
    - bytes
  facts:
  - audio_length
  materialize:
    kind: strip_audio
    source_columns:
      video: rel_video
    tool:
      name: ffmpeg
      version_pin: 4.4.2-0ubuntu0.22.04.1
      args:
      - -an
      - -c:v
      - copy
      - -map_metadata
      - '-1'
```

## Output contract

`encode()` returns `{"audio": [one wav per sample], "used": [...]}`. Each wav is MONO and covers the first
`audio_length` = 8 s of the clip (every frozen clip lasts at least that long): 8.0 to 8.05 s (the `wav` codec checks it).
Any sample rate. `describe()["sampling"]` is required. `bridge check` reruns every clip (the audio must be identical
when `precision.deterministic` is true) and re-encodes clips alone and in shuffled batches; validation compares the
canaries through the wav codec's distance.

```yaml
output_contract:
  shape: fixed
  coordinates: media
  target_frame: null
  requires:
    pos: []
    streams: []
    grid: null
    t_bins: null
  sampling: required
  schema:
    audio:
      type: wav
      channels: 1
      duration_s:
      - 8.0
      - 8.05
```

## Prepare & verify

```bash
yavih bench prepare vggsounder_v2a 01 -- --rows 32      # a validation run: 32 muted copies, each verified
yavih bench prepare vggsounder_v2a 01 -- --workers 16   # all 3 000
yavih bench prepare vggsounder_v2a 02                   # yavih bench verify vggsounder_v2a --census + the reference audio
```

Success: `yavih bench verify vggsounder_v2a` reports 3 000 / 3 000 present and the census written; step 02 reports
the 3 000 reference wavs present.

## Gotchas

- The container's frame-count header is wrong in many VGGSound mp4s, and edit lists hide packets; the mute step
  verifies each copy by its decoded frame timestamps, not by `nb_frames` or packet counts.
- Every frozen clip lasts at least 8 s. av-benchmark's synchrony model refuses shorter audio and its loader then
  scores the next file under the wrong name, so a generation shorter than 8 s is refused by the `wav` codec instead.
- Seeded generation is part of the configuration: another seed is another `describe_sha256` and another row in the
  report. Do not average candidates; declare `n_candidates` and the selection rule instead.
- av-benchmark's default batches exceed a 24 GB GPU: 128 audio clips in PaSST, and 8 videos in `extract_video.py`
  (ImageBind huge and Synchformer, compiled). The scorer extracts 16 audio clips and 2 videos at a time, which changes
  no number beyond floating-point noise.
- `extract_video.py` skips a video it cannot decode without an error, and av-benchmark's audio loader substitutes the
  next file for one it cannot read; the scorer refuses to score when a feature file does not hold exactly the scored
  clips.
