# audiocaps_t2a — setup

## Obtain the data

Two public downloads, no account: the official AudioCaps test captions (`test.csv` from github.com/cdjkim/audiocaps
at commit `80fb3ade6d400d4311354a360db25da051a48cc1`, sha256-checked) and the test split of the
[OpenSound/AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps) Hugging Face dataset at revision
`b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded` (41 parquet files, 3.4 GB, CC BY-NC 4.0). The mirror holds the 24 kHz wav
of each caption row; step 02 writes one reference wav per frozen clip and checks it against the sha256 the manifest
froze, so another mirror revision is caught rather than scored.

```sh
yavih bench prepare audiocaps_t2a                 # lists the steps
yavih bench prepare audiocaps_t2a 01              # test.csv + the mirror's test parquet into source_root
yavih bench prepare audiocaps_t2a 02              # one reference wav per frozen clip into derived_root, sha256-checked
yavih bench prepare audiocaps_t2a 03              # bench verify + every staged wav's sha256
```

Disk: 3.4 GB under `source_root` (not read after step 02), about 0.8 GB of wavs under `derived_root`, plus about
0.1 GB for the reference feature caches the scorer writes there on first use.

## Dependencies

The scorer runs the MMAudio authors' evaluation code from a git checkout pinned to one commit (it loads its
weights from `<checkout>/weights` and runs the checkout's `extract_text.py` for the caption features):

```bash
pip install -e git+https://github.com/hkchengrex/av-benchmark.git@f351b9a6fc6abde746d5f8e1d4c47c883319cb41#egg=av_bench
```

then download its two weight files into the checkout's `weights/` (as its README says; av-benchmark builds every
encoder it knows, so the synchrony model's weights are needed even though this benchmark has no video):

```bash
cd <checkout>   # e.g. src/av-bench
mkdir -p weights
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt -O weights/music_speech_audioset_epoch_15_esc_89.98.pt
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth -O weights/synchformer_state_dict.pth
```

It needs torch >= 2.5.1 and a CUDA GPU in the scoring environment, and installs PaSST, ImageBind, LAION-CLAP and
MS-CLAP forks.
`yavih doctor --bench audiocaps_t2a` checks the assert-import (`av_bench`, version 1.0.0). Prepare-time: curl, the
Hugging Face CLI (`hf`) and pyarrow.

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

`caption`: a cell field, handed over as `Sample.cells["caption"]`, a str. It is the clip's first caption in the
official `test.csv`, verbatim. No file is handed over and the recording never reaches the bridge. The fact
`audio_length` (10.0) is in `Sample.meta["facts"]`.

The bridge decides how it encodes the text and how it generates (model, sampler, steps, guidance, seed) and
declares it in `describe()`: `input` is `{"caption": {"note": "..."}}`, naming any prompt template, and `sampling`.

```yaml
input_contract:
  handed_over:
    caption: one human caption of the clip, verbatim (the first of its five in the official test.csv); the recording
      is the reference and never reaches the bridge
  info: {}
  facts:
  - audio_length
  materialize: null
```

## Output contract

`encode()` returns `{"audio": [one wav per sample], "used": [{} per sample]}`. Each wav is MONO and lasts exactly
`audio_length` = 10 s: 10.0 to 10.05 s (the `wav` codec checks it). Any sample rate. `describe()["sampling"]` is
required. `bridge check` reruns every caption (the audio must be identical when `precision.deterministic` is true)
and re-encodes captions alone and in shuffled batches; validation compares the canaries through the wav codec's
distance.

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
      - 10.0
      - 10.05
```

## Prepare & verify

```bash
yavih bench prepare audiocaps_t2a 01   # downloads (resumable)
yavih bench prepare audiocaps_t2a 02   # 776 reference wavs, sha256-checked
yavih bench prepare audiocaps_t2a 03   # yavih bench verify audiocaps_t2a + every staged wav's sha256
```

Success: `yavih bench verify audiocaps_t2a` reports 776 / 776 present; step 03 reports 776 / 776 staged with the
frozen sha256.

## Gotchas

- Other AudioCaps copies hold other subsets of the test clips (886 to 975 in the literature). Numbers on another
  subset are not comparable with this folder's; the manifest names every clip and its recording's sha256.
- The captions of a clip differ in wording and detail; the benchmark hands over the first one only, so a model is
  never scored on its best-matching caption.
- Seeded generation is part of the configuration: another seed is another `describe_sha256` and another row in the
  report. Do not average candidates; declare `n_candidates` and the selection rule instead.
- av-benchmark pairs files by name and only prints a row it cannot pair; the scorer refuses instead when a feature
  file does not hold exactly the scored clips.
- av-benchmark's `extract_text.py` looks for the LAION-CLAP checkpoint one level above the checkout, not in the
  `<checkout>/weights` its README names. The scorer runs the script's own `extract` in-process with the path pointed at
  `<checkout>/weights`; running the script by hand needs a `weights` folder next to the checkout.
- av-benchmark's default extraction batch (128 ten-second clips) exceeds a 24 GB GPU in PaSST; the scorer extracts 16
  clips at a time, which changes no number beyond floating-point noise.
