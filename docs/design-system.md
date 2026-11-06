# Design system

Every image yavih commits (the repository header, each benchmark folder's README header and its README figures) is an
HTML page filled from one set of tokens and components and screenshotted by a headless Chrome. One design means a
GitHub page that lists many folders reads as one project, whichever machine rendered each image.

The design lives in the source repository, not in the wheel:

```
assets/design/
  tokens.css        the palette, the type stacks, radii, the grid step
  components.css    the page background and the pieces the headers and figures are drawn with
  fonts.css         @font-face for the bundled fonts (relative urls)
  fonts/inter/            Inter 400/500/600/800, Latin subset woff2 + OFL.txt
  fonts/jetbrains-mono/   JetBrains Mono 400/600, Latin subset woff2 + OFL.txt
  figures/<kind>.html     one string.Template page per core figure kind (and base.html for a folder's own kinds)
assets/benchmark-header/template.html   the folder header layout
assets/header.html                      the repository header (links the design CSS)
```

`yavih.design` (shipped, so `bench lint` can tell whether the design is at hand) concatenates the CSS
(`base_css()`), sizes the page (`page_css(w, h)`), digests the design (`design_sha256()`), links the fonts next to a
filled page (`stage_fonts`), finds Chrome and screenshots (`render_html`), palette-compresses PNGs (`compress_png`,
with `pip install imagequant`) and tiles contact sheets. `yavih.headers` and `yavih.previews` build on it; the drawing
helpers are `yavih.media.figures` (waveforms as inline SVG, spectrograms and depth maps as PNG data URIs coloured with
the palette). There is no plotting library: boxes, bands and timelines are positioned `<div>`s and Chrome rasterises.

## Palette

| token | value | role |
|---|---|---|
| `--bg0` | `#070a12` | page background (dark end), letterbox, an invalid depth pixel |
| `--bg1` | `#0f172a` | page background (light end), contact sheets |
| `--ink` | `#e5e7eb` | text |
| `--muted` | `#94a3b8` | secondary text, kickers, timestamps, axis ticks |
| `--line` | `rgba(148,163,184,.16)` | the background grid, centre lines, track outlines |
| `--a` | `#22d3ee` (cyan) | first accent: the media chip, waveforms, boxes, the first event band, the retrieval hit, the motion time track |
| `--b` | `#a78bfa` (violet) | second accent: the task chip, the second event band |
| `--c` | `#f472b6` (pink) | third accent: the third event band, the end of the name gradient |
| `--pill` | `rgba(255,255,255,.05)` | chip and track fills |
| `--ok` / `--warn` | `#4ade80` / `#fbbf24` | status words in the repository header |

Spectrograms run `--bg0` → deep violet → `--b` → `--c` → warm white (quiet to loud, `figures.SPEC_STOPS`); depth maps
run warm white → `--a` → `--b` → deep violet → `--bg0` (near to far, `figures.DEPTH_STOPS`).

**The dark-background rule:** every committed image renders on the `--bg0 → --bg1` gradient (with the two accent glows
and the masked grid of `body.ds`), never on white, so it sits the same on GitHub's dark and light themes. The motion
overlay is the one transparent page: ffmpeg composes it over `--bg0`.

## Typography

`--sans` is Inter, `--mono` JetBrains Mono, both bundled (SIL Open Font License 1.1, licences next to the files) as
subsets covering Latin, Latin Extended, Greek, punctuation, arrows, sub- and superscripts and math operators. The
system stacks after them only catch a glyph outside the subset. Sizes in CSS px at device scale 2:

| step | size | weight | use |
|---|---|---|---|
| display | 68 to 148 | 800 | the header name (shrinks with its length), the repository word mark |
| title | 27 to 30 | 600 | a header's paper title, a figure's title (`.figtitle`) |
| body | 17 to 18.5 | 400 | the header abstract |
| label | 15 to 16 | 600 | tile captions, label chips |
| kicker / chip | 13 to 16 | 400 / 600 mono | `yavih · <name> · <split>` kickers (upper case), chips, timestamps |

## Figures

A folder's README figures are declared in `benchmarks/<name>/assets/preview.yaml` (never hashed, never in the wheel)
and rendered by `python -m yavih.previews <name>`. Each still figure is a 1600 CSS px wide page rendered at scale 2:
a `.figbar` (kicker, title, chips) over a `.figbody`.

| kind | shows | rows | page (CSS px) | needs |
|---|---|---|---|---|
| `images` | 2 x 4 labelled tiles | 8 | 1600 x 620 | an image stream, labels |
| `frames` | a strip of 6 frames with timestamps (`options.times` or uniform) | 1 | 1600 x 330 | a video stream |
| `timeline` | 6 frames over the labelled intervals of the video (online action detection) | 1 | 1600 x 420 | a video stream, intervals |
| `boxes` | the frame at the annotated time with the ground-truth box; a card with the class, the time, waveform and mel | 1 | 1600 x 620 | a video stream, `label.kind: box`, audio optional |
| `audio` | two clips: waveform over mel spectrogram, one label chip per label | 2 | 1600 x 520 | an audio stream |
| `av` | 8 frames, waveform and mel on one time axis, the events as bands | 1 | 1600 x 560 | video and audio, intervals |
| `depth` | three image → depth pairs (the depth colours) | 3 | 1600 x 440 | an image stream, dense targets |
| `generation` | what the model is given (a caption, or 4 frames) → the reference audio (waveform and mel) | 1 | 1600 x 420 | a caption cell or a video stream; the reference through `options.reference_column` / `reference_root` or the audio stream |
| `retrieval` | a query video → five candidate sounds, the match ringed (4 drawn by `options.seed`) | 1 | 1600 x 460 | video and audio |
| `classes` | sampled class names with their counts; no media | 0 | 1600 x 360 | `classes.csv` |
| `motion` | `<id>.gif` 640 x 400 at 10 fps, 256 colours, and `<id>.mp4` 1280 x 800 H.264 CRF 18 yuv420p with its audio: the clip over a label bar and a filling time track | 1 | 640 x 400 overlay | a video stream (audio kept) |

Intervals (`timeline`, `av`, `motion`) come from `label.kind: segments`, else `options.intervals: {column, format:
json_triples, names?}` (a json cell `[[start_s, end_s, class], ...]`), else the folder's `preview.py` `intervals`
hook.

**File names.** The figure with id `preview` writes `assets/preview.png`, the README's main figure; any other still
figure `assets/preview-<id>.png`; a motion figure `assets/<id>.gif` and `assets/<id>.mp4` (the README shows the GIF
and links the MP4). A `preview*` or `clip*` file the spec does not declare is stale.

**Provenance.** `assets/preview.src.json` records per figure its kind, split, rows, files with their sizes, the media
sha256 when the manifest carries one, and `source_sha256`: the canonical spec entry, the template, the component
version, `PREVIEW_VERSION` and `design_sha256()`. Nothing decoded enters it: pixels differ between machines and decoder
builds and that is accepted; what was asked for does not. The header images record the sha256 of their filled HTML
(`assets/header.src.sha256`) the same way. A changed token, component or font makes every figure stale (and every
header, whose HTML inlines the CSS): re-render, and review the contact sheets.

**Size budget.** A PNG at most 1.5 MB, a GIF 4 MB, an MP4 2 MB, the folder's `assets/` 10 MB: the folder's files
(figures included) are fetched with it. Lint warns `W_ASSET_SIZE` above a budget: fewer or smaller tiles, a shorter
excerpt, and `pip install imagequant` (PNGs are then saved as 256-colour palettes).

## Pixels and licences

A figure shows a handful of downscaled stills (at most 8 tiles, at most 960 px on a side) and at most 5 s of audio
or video (`MAX_SECONDS`), as documentation of the task. A folder whose data licence forbids redistribution (ImageNet
first; in general `obtain.tier: needs-licence`, which `--suggest` maps to it) uses the data-free `classes` kind and
shows no media at all. The reviewer of a folder checks this with the figures.

## Rendering

```
python -m yavih.previews <name> --suggest --write    # draft assets/preview.yaml; check its rows show the task
python -m yavih.previews <name> --chrome <chrome>    # render (needs the data, Chrome; ffmpeg for motion)
python -m yavih.previews <name> --only clip          # one figure; the other provenance entries are kept
python -m yavih.previews <name> --html-only          # the filled pages under assets/_work/ (git-ignored)
python -m yavih.previews --all --check               # ok / missing / stale / invalid / too-big per folder
python -m yavih.headers --all --contact-sheet sheet.png
```

`--data-root KEY=PATH` points a data root elsewhere for one run. Rendering writes the files and their provenance only
after every figure of the run succeeded. `bench lint` reports the figures (`preview` check: `E_PREVIEW_SPEC`,
`W_PREVIEW_MISSING`, `W_PREVIEW_STALE`, `W_ASSET_SIZE`) and README links to absent assets (`W_README_ASSET`); only an
invalid spec is an error, since figures may be rendered after a merge where the data is staged. The `render` workflow
renders every header and the committed figures' contact sheet on pull requests.

## Adding a kind

A kind every benchmark family can use belongs in the core: a template `assets/design/figures/<kind>.html` (the
placeholders `$base_css $page_css $bench $kicker $title $chips $body`; layout CSS in the page, pieces from
`components.css`) and a `Component` in `yavih.previews.CORE_COMPONENTS` with its size, row count, options and a `fill`
that returns `{"title", "chips", "body"}` from a `RenderContext` (rows as samples, `frames`, `audio`, `image`,
`target`, `labels`, `intervals`, and the drawing helpers `data_uri`, `wave_svg`, `mel_uri`, `depth_uri`). Add it to the
table above and a case to `tests/test_previews.py`.

A kind only one folder needs stays in the folder, in `preview.py` (imported like `eval.py`, torch-free at module
scope; `bench extensions` lists it as the `preview` point):

```python
from yavih.previews import Component

def fill(ctx):
    return {"title": ..., "chips": [("a", "audio")], "body": "..."}

COMPONENTS = [Component("myfolder_grid", 1, (1600, 420), 1, "base.html", fill, css=".x { ... }", options={})]

def intervals(bench, split, extra):          # optional: [(start_s, end_s, label)] of one row
    ...
```

A core kind's name is refused. Bump a component's `version` when its `fill` draws differently: the figures it made
become stale.
