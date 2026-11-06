"""README figures of the benchmark folders: ``python -m yavih.previews NAME ... [--all]``.

A folder's README shows its task with pre-rendered figures in ``assets/``: ``preview.png`` (the main figure),
``preview-<id>.png`` (more stills) and ``<id>.gif`` + ``<id>.mp4`` (motion: the GIF plays inline on GitHub, the MP4 is
its lossless source, linked). What to draw is the folder's ``assets/preview.yaml``, never hashed or shipped::

    schema_version: 1
    figures:
      - id: preview            # -> assets/preview.png; any other id -> assets/preview-<id>.png
        kind: audio            # a core kind (CORE_KINDS) or one the folder's preview.py registers
        split: "1"
        rows: [1-100032-A-0, 1-100038-A-14]   # clip_ids in display order, the count the kind wants
        options: {seconds: 5.0}               # the kind's options; an unknown key is a SpecError
      - id: clip               # -> assets/clip.gif + assets/clip.mp4
        kind: motion
        split: "1"
        rows: [1-100032-A-0]
        options: {start_s: 0.0, seconds: 4.0}

Every figure is an HTML page filled from a ``string.Template`` in ``assets/design/figures/<kind>.html`` of the source
checkout with the design system (``yavih.design``) and screenshotted by a headless Chrome at device scale 2; a motion
figure renders a transparent overlay frame that ffmpeg composes over the clip. The drawing helpers are
``yavih.media.figures`` (no plotting library). Rendering reads the media through the folder's own data roots
(``--data-root KEY=PATH`` overrides one).

Provenance: ``assets/preview.src.json`` records per figure its kind, split, rows, files and sizes and
``figure_source_sha256`` (the spec entry, the template, the component version, ``PREVIEW_VERSION`` and
``design_sha256()``; no decoded pixel enters it, so the record is machine-independent while the pixels are not).
``bench lint`` (check ``preview``) reports a missing or invalid spec, missing, stale or undeclared files and files over
``SIZE_BUDGET`` (``E_PREVIEW_SPEC``, ``W_PREVIEW_MISSING``, ``W_PREVIEW_STALE``, ``W_ASSET_SIZE``); ``--check`` prints the
same per folder. ``--suggest`` drafts a spec from the yaml (kinds from the streams, label kind, scorer and capability;
rows drawn by a seeded, class-stratified draw from the eval split); ``--write`` saves it.

A folder adds a kind of its own in ``preview.py`` (imported like ``eval.py``; torch-free at module scope)::

    from yavih.previews import Component
    COMPONENTS = [Component("mykind", 1, (1600, 420), 1, "base.html", fill, css=".x { ... }")]
    def intervals(bench, split, extra): ...   # optional: [(start_s, end_s, label)] of a row (timeline, av)

The rules (palette, sizes, budgets, pixels and licences) are https://github.com/Bizilizi/yavih/blob/main/docs/design-system.md.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import yaml

from yavih import design
from yavih.design import render_html

__all__ = ["CORE_KINDS", "Component", "Figure", "PREVIEW_VERSION", "PreviewError", "PreviewSpec", "RenderContext",
           "SIZE_BUDGET", "SpecError", "check_folder", "components", "figure_files", "figure_html",
           "figure_source_sha256", "load_spec", "main", "render_figure", "render_folder", "render_motion",
           "suggest_spec", "validate_spec", "write_provenance"]

PREVIEW_VERSION = 1
SPEC_FILE = "preview.yaml"
PROVENANCE_FILE = "preview.src.json"
#: The folder's own kinds and interval hook (``COMPONENTS``, ``intervals``).
FOLDER_HOOK = "preview.py"
TEMPLATES_DIR = design.DESIGN_DIR / "figures"
WIDTH = 1600
#: Bytes per committed file type and for the whole ``assets/`` directory.
SIZE_BUDGET: dict[str, int] = {"png": 1_500_000, "gif": 4_000_000, "mp4": 2_000_000, "folder": 10_000_000}
#: The longest media excerpt a figure shows (pixels and licences: docs/design-system.md).
MAX_SECONDS = 5.0
#: The motion overlay page (CSS px): the GIF is rendered at this size, the MP4 at twice it.
MOTION_SIZE = (640, 400)
#: Motion geometry (CSS px): the video area, then a time track, then the label bar.
_MOTION_VIDEO_H, _MOTION_TRACK_H = 336, 4
_MOTION_MP4_FPS = 25
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
#: Files in assets/ the figures own (an undeclared one is stale).
_FIGURE_FILE = re.compile(r"^(preview|clip)[^/]*\.(png|gif|mp4)$")


class SpecError(ValueError):
    """``assets/preview.yaml`` (or a folder ``preview.py``) is invalid."""


class PreviewError(design.RenderError):
    """A figure cannot be rendered (no data, no decoder, no ffmpeg, Chrome failed)."""


# ----------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class Figure:
    """One ``figures:`` entry of ``assets/preview.yaml``."""

    id: str
    kind: str
    split: str | None
    rows: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "split": self.split, "rows": list(self.rows), "options": dict(self.options)}


@dataclass(frozen=True)
class PreviewSpec:
    schema_version: int
    figures: tuple[Figure, ...]
    path: Path | None = None

    def figure(self, fid: str) -> Figure:
        for f in self.figures:
            if f.id == fid:
                return f
        raise KeyError(fid)


def _parse_spec(raw: Any, where: str) -> PreviewSpec:
    if not isinstance(raw, dict):
        raise SpecError(f"{where}: not a mapping")
    extra = sorted(set(raw) - {"schema_version", "figures"})
    if extra:
        raise SpecError(f"{where}: unknown key(s) {extra} (schema_version, figures)")
    if raw.get("schema_version") != 1:
        raise SpecError(f"{where}: schema_version must be 1, got {raw.get('schema_version')!r}")
    figs = raw.get("figures")
    if not isinstance(figs, list) or not figs:
        raise SpecError(f"{where}: figures must be a non-empty list")
    out: list[Figure] = []
    for k, f in enumerate(figs):
        at = f"{where}: figures[{k}]"
        if not isinstance(f, dict):
            raise SpecError(f"{at}: not a mapping")
        unknown = sorted(set(f) - {"id", "kind", "split", "rows", "options"})
        if unknown:
            raise SpecError(f"{at}: unknown key(s) {unknown} (id, kind, split, rows, options)")
        fid, kind = f.get("id"), f.get("kind")
        if not isinstance(fid, str) or not _ID.match(fid):
            raise SpecError(f"{at}: id must match {_ID.pattern}, got {fid!r}")
        if not isinstance(kind, str) or not kind:
            raise SpecError(f"{at}: kind must be a non-empty string")
        split = f.get("split")
        if split is not None and not isinstance(split, (str, int)):
            raise SpecError(f"{at}: split must be a string")
        rows = f.get("rows") or []
        if not isinstance(rows, list) or not all(isinstance(r, (str, int)) for r in rows):
            raise SpecError(f"{at}: rows must be a list of clip_ids")
        options = f.get("options") or {}
        if not isinstance(options, dict):
            raise SpecError(f"{at}: options must be a mapping")
        if any(o.id == fid for o in out):
            raise SpecError(f"{at}: duplicate id {fid!r}")
        out.append(Figure(fid, kind, None if split is None else str(split), tuple(str(r) for r in rows), dict(options)))
    return PreviewSpec(1, tuple(out))


def load_spec(folder: str | Path) -> PreviewSpec | None:
    """The folder's ``assets/preview.yaml`` (``None`` when absent; ``SpecError`` when it does not parse)."""
    path = Path(folder) / "assets" / SPEC_FILE
    if not path.is_file():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SpecError(f"assets/{SPEC_FILE}: {e}") from e
    spec = _parse_spec(raw, f"assets/{SPEC_FILE}")
    return PreviewSpec(spec.schema_version, spec.figures, path)


# ----------------------------------------------------------------------------- components


@dataclass(frozen=True)
class Component:
    """A figure kind. ``size`` is the page in CSS px (``MOTION_SIZE`` for motion); ``rows`` the exact number of
    ``rows`` a figure lists (``None``: one or more); ``template`` a file name under ``assets/design/figures/``, a
    ``Path`` or the HTML text itself (placeholders ``$base_css $page_css $bench $kicker $title $chips $body``);
    ``fill(ctx) -> {"title", "chips", "body"}`` (``chips``: ``[(css class, inner html)]``); ``css`` joins the page CSS;
    ``options`` the accepted option keys with their defaults; bump ``version`` when ``fill`` draws differently."""

    kind: str
    version: int
    size: tuple[int, int]
    rows: int | None
    template: str | Path
    fill: Callable[[RenderContext], dict[str, Any]]
    css: str = ""
    motion: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def template_text(self) -> str:
        t = self.template
        if isinstance(t, Path):
            return t.read_text(encoding="utf-8")
        if "<" in t:
            return t
        path = TEMPLATES_DIR / t
        if not path.is_file():
            raise PreviewError(f"{path} not found: figure rendering runs from a source checkout of yavih")
        return path.read_text(encoding="utf-8")


def _folder_hook(bench: Any) -> Any | None:
    path = Path(bench.folder) / FOLDER_HOOK
    if not path.is_file():
        return None
    from yavih.benchmark import _load_module

    return _load_module(path, "yavih_benchmark_preview")


def components(bench: Any) -> dict[str, Component]:
    """The kinds a folder can draw: the core ones and its ``preview.py`` ``COMPONENTS`` (a core name is refused)."""
    out = dict(CORE_COMPONENTS)
    mod = _folder_hook(bench) if bench is not None else None
    for comp in getattr(mod, "COMPONENTS", None) or ():
        if not isinstance(comp, Component):
            raise SpecError(f"{FOLDER_HOOK}: COMPONENTS holds {type(comp).__name__}, not yavih.previews.Component")
        if comp.kind in CORE_COMPONENTS:
            raise SpecError(f"{FOLDER_HOOK}: kind {comp.kind!r} is a core kind; a folder kind takes a name of its own")
        out[comp.kind] = comp
    return out


def figure_files(folder: str | Path | None, fig: Figure, *, motion: bool | None = None) -> list[str]:
    """The ``assets/``-relative files ``fig`` writes: ``<id>.gif`` and ``<id>.mp4`` for motion, else ``preview.png``
    (id ``preview``) or ``preview-<id>.png``. ``motion`` defaults to ``fig.kind == "motion"``."""
    if motion is None:
        motion = fig.kind == "motion"
    if motion:
        return [f"{fig.id}.gif", f"{fig.id}.mp4"]
    return ["preview.png" if fig.id == "preview" else f"preview-{fig.id}.png"]


def figure_source_sha256(bench: Any, fig: Figure) -> str:
    """sha256 (hex) of what a figure is drawn from, without decoding anything: the canonical spec entry, the
    component's template, CSS and version, ``PREVIEW_VERSION`` and ``design_sha256()``."""
    comp = components(bench)[fig.kind]
    h = hashlib.sha256(f"yavih-preview:{PREVIEW_VERSION}\n".encode())
    h.update(json.dumps(fig.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")
    h.update(f"{comp.kind}:{comp.version}:{comp.size[0]}x{comp.size[1]}:{int(comp.motion)}\n".encode())
    h.update(comp.template_text().encode("utf-8") + b"\0" + comp.css.encode("utf-8") + b"\0")
    h.update(design.design_sha256().encode())
    return h.hexdigest()


def validate_spec(bench: Any, spec: PreviewSpec) -> list[str]:
    """What is wrong with ``spec`` for ``bench`` (empty when it is valid): an unknown kind, split, clip_id or option,
    a wrong row count, a motion excerpt over ``MAX_SECONDS``."""
    try:
        comps = components(bench)
    except Exception as e:  # noqa: BLE001 - a broken preview.py is a spec problem
        return [f"{FOLDER_HOOK}: {e.__class__.__name__}: {e}"]
    splits = set(bench.spec.splits.all_values())
    out = []
    for fig in spec.figures:
        at = f"figure {fig.id!r}"
        comp = comps.get(fig.kind)
        if comp is None:
            out.append(f"{at}: unknown kind {fig.kind!r}; known: {sorted(comps)}")
            continue
        unknown = sorted(set(fig.options) - set(comp.options))
        if unknown:
            out.append(f"{at}: unknown option(s) {unknown} for kind {fig.kind!r}; known: {sorted(comp.options)}")
        if comp.rows is not None and len(fig.rows) != comp.rows:
            out.append(f"{at}: kind {fig.kind!r} draws {comp.rows} row(s), the spec lists {len(fig.rows)}")
        elif comp.rows is None and not fig.rows:
            out.append(f"{at}: kind {fig.kind!r} needs at least one row")
        if fig.split is None:
            if fig.rows:
                out.append(f"{at}: rows need a split")
        elif fig.split not in splits:
            out.append(f"{at}: unknown split {fig.split!r}; declared: {sorted(splits)}")
        elif fig.rows:
            known = set(bench.manifest(fig.split).clip_id)
            missing = [r for r in fig.rows if r not in known]
            if missing:
                out.append(f"{at}: clip_id(s) {missing[:3]} not in split {fig.split!r}")
        if len(set(fig.rows)) != len(fig.rows):
            out.append(f"{at}: a clip_id is listed twice")
        secs = fig.options.get("seconds")
        if secs is not None and (not isinstance(secs, (int, float)) or not 0 < float(secs) <= MAX_SECONDS):
            out.append(f"{at}: seconds must be in (0, {MAX_SECONDS:g}] (pixels and licences: a preview shows a short excerpt)")
    return out


# ----------------------------------------------------------------------------- render context


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _num(n: int) -> str:
    return f"{n:,}".replace(",", "\u2009")


def _fmt_t(t: float) -> str:
    return f"{t:.1f} s" if t < 60 else f"{int(t // 60)}:{t % 60:04.1f}"


class RenderContext:
    """What a component's ``fill`` draws from: the loaded ``bench``, the ``figure``, its rows as ``Sample`` s (by
    position ``i`` in ``figure.rows``) and the options merged over the component's defaults. The helpers decode
    through ``yavih.media.decode`` (the ``[media]`` extra)."""

    def __init__(self, bench: Any, figure: Figure, component: Component, *, work_dir: Path) -> None:
        self.bench, self.figure, self.component, self.work_dir = bench, figure, component, Path(work_dir)
        self.split = figure.split
        self.options: dict[str, Any] = {**dict(component.options), **dict(figure.options)}
        self.manifest = bench.manifest(figure.split) if figure.split is not None else None
        self.index = [self.manifest.index_of(c) for c in figure.rows] if self.manifest is not None else []
        self._samples: dict[int, Any] = {}
        self._classes = [c["name"] for c in bench.classes()] if bench.spec.manifest.classes else []

    # -- rows
    def __len__(self) -> int:
        return len(self.index)

    def clip_id(self, i: int) -> str:
        return self.figure.rows[i]

    def sample(self, i: int, *, manifest_index: int | None = None) -> Any:
        k = self.index[i] if manifest_index is None else manifest_index
        if k not in self._samples:
            from yavih.sample import build_sample

            try:
                roots = self.bench.resolved_roots()
            except Exception as e:  # noqa: BLE001 - a root that does not resolve: no data here
                raise PreviewError(f"{self.bench.name}: data roots do not resolve ({e}); give --data-root KEY=PATH") from e
            self._samples[k] = build_sample(self.bench, self.split, self.manifest, k, roots)
        return self._samples[k]

    def stream(self, i: int, kind: str, *, manifest_index: int | None = None) -> Any:
        """The row's first stream of ``kind`` (``video``, ``audio``, ``image``) in the yaml's order."""
        smp = self.sample(i, manifest_index=manifest_index)
        for name, st in self.bench.spec.streams.items():
            if st.kind == kind and name in smp.streams:
                return smp.streams[name]
        raise PreviewError(f"{self.bench.name}: row {self.clip_id(i)!r} has no {kind} stream")

    def has_stream(self, kind: str) -> bool:
        return any(s.kind == kind for s in self.bench.spec.streams.values())

    def labels(self, i: int, *, manifest_index: int | None = None) -> list[str]:
        """The class names of a row (``classes.csv``), in label order."""
        k = self.index[i] if manifest_index is None else manifest_index
        lab = self.manifest.label_ids[k]
        ids = [] if lab is None else ([lab] if isinstance(lab, (int, np.integer)) else list(lab))
        return [self._classes[j] if 0 <= j < len(self._classes) else str(j) for j in ids]

    def extra(self, i: int) -> dict[str, Any]:
        return dict(self.manifest.extra[self.index[i]])

    def duration(self, i: int) -> float | None:
        for kind in ("video", "audio"):
            if self.has_stream(kind):
                try:
                    d = self.stream(i, kind).info.get("duration_s")
                except Exception:  # noqa: BLE001 - a fact that cannot be read: the caller falls back
                    d = None
                if d:
                    return float(d)
        return float(self.bench.spec.protocol.get("clip_seconds") or 0) or None

    def intervals(self, i: int) -> list[tuple[float, float, str]]:
        """``[(start_s, end_s, label)]`` of a row: ``label.kind: segments`` (segment indices x ``segment_seconds``),
        else ``options.intervals {column, format: json_triples, names?}``, else the folder ``preview.py``
        ``intervals(bench, split, extra)``, else none."""
        k = self.index[i]
        proto = self.bench.spec.protocol
        if self.bench.spec.manifest.label.kind == "segments" and self.manifest.segments is not None:
            seg = float(proto.get("segment_seconds") or 1.0)
            off = float(proto.get("segment_offset") or 0) * seg
            a, b = (int(x) for x in self.manifest.segments[k])
            names = self.labels(i)
            return [(off + a * seg, off + b * seg, names[0] if names else "event")] if b > a else []
        spec = self.options.get("intervals")
        if spec:
            col, fmt, names = spec.get("column"), spec.get("format", "json_triples"), spec.get("names") or []
            if fmt != "json_triples":
                raise SpecError(f"options.intervals.format {fmt!r}: only json_triples ([[start_s, end_s, class], ...])")
            cell = self.manifest.extra[k].get(col)
            cell = json.loads(cell) if isinstance(cell, str) else (cell or [])
            out = []
            for s, e, c in cell:
                c = int(c)
                name = names[c] if 0 <= c < len(names) else (self._classes[c] if 0 <= c < len(self._classes) else f"class {c}")
                out.append((float(s), float(e), str(name)))
            return out
        mod = _folder_hook(self.bench)
        fn = getattr(mod, "intervals", None)
        if callable(fn):
            return [(float(s), float(e), str(lab)) for s, e, lab in fn(self.bench, self.split, self.manifest.extra[k])]
        return []

    # -- decoding
    def frames(self, i: int, times: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        from yavih.media.decode import frames_at

        return frames_at(self.stream(i, "video"), list(times))

    def image(self, i: int) -> np.ndarray:
        from yavih.media.decode import image

        return image(self.stream(i, "image"))

    def audio(self, i: int, *, start_s: float = 0.0, seconds: float | None = None, manifest_index: int | None = None
              ) -> tuple[np.ndarray, int]:
        """Mono float32 samples of the row's audio (channels averaged) from ``start_s`` for ``seconds``."""
        from yavih.media.decode import audio

        y, rate = audio(self.stream(i, "audio", manifest_index=manifest_index))
        return _cut(y, rate, start_s, seconds), rate

    def target(self, i: int) -> np.ndarray:
        """The row's dense target (``targets[split]``, e.g. a depth map)."""
        return np.asarray(self.bench.targets(self.split).rows([self.index[i]])[0])

    # -- drawing
    @staticmethod
    def data_uri(rgb: np.ndarray, max_side: int | None = 960) -> str:
        from yavih.media.figures import png_data_uri

        return png_data_uri(rgb, max_side=max_side)

    @staticmethod
    def wave_svg(y: np.ndarray, width: int, height: int) -> str:
        from yavih.media.figures import svg_wave

        return svg_wave(y, width, height)

    @staticmethod
    def mel_uri(y: np.ndarray, rate: int, *, columns: int | None = None, n_mels: int = 64) -> str:
        """The mel spectrogram as a PNG data URI, low bands at the bottom, ``columns`` wide when given."""
        from yavih.media.figures import SPEC_STOPS, colormap, mel_spectrogram, png_data_uri

        db = mel_spectrogram(y, rate, n_mels=n_mels)
        img = colormap(((db[::-1] + 80.0) / 80.0) ** 1.6, SPEC_STOPS)
        if columns and img.shape[1] != columns:
            xs = np.linspace(0, img.shape[1] - 1, int(columns)).round().astype(int)
            img = img[:, xs]
        return png_data_uri(img)

    @staticmethod
    def depth_uri(depth: np.ndarray, max_side: int | None = 640) -> str:
        """A depth map (0 = invalid) through the design's depth colours, near bright, scaled per map (log)."""
        from yavih.media.figures import DEPTH_STOPS, colormap, png_data_uri

        d = np.asarray(depth, dtype=np.float64)
        valid = d > 0
        v = np.zeros_like(d)
        if valid.any():
            lg = np.log(d[valid])
            lo, hi = np.percentile(lg, 2), np.percentile(lg, 98)
            v[valid] = (lg - lo) / max(hi - lo, 1e-6)
        img = colormap(v, DEPTH_STOPS)
        img[~valid] = (7, 10, 18)
        return png_data_uri(img, max_side=max_side)


def _cut(y: np.ndarray, rate: int, start_s: float, seconds: float | None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = y.mean(axis=1) if y.ndim == 2 else y.reshape(-1)
    a = max(0, int(round(float(start_s) * rate)))
    b = y.size if seconds is None else min(y.size, a + int(round(float(seconds) * rate)))
    return y[a:b]


def _chip(text: str, cls: str = "") -> tuple[str, str]:
    return cls, _esc(text)


def _media_chip(bench: Any) -> tuple[str, str]:
    kinds = {s.kind for s in bench.spec.streams.values()}
    return "a", " + ".join(k for k in ("video", "image", "audio") if k in kinds) or "data"


def _uniform(duration: float | None, n: int) -> list[float]:
    d = float(duration or 10.0)
    return [round(d * (k + 0.5) / n, 3) for k in range(n)]


def _frame_tiles(ctx: RenderContext, i: int, times: Sequence[float], *, max_side: int = 520) -> str:
    frames, pts = ctx.frames(i, times)
    return "".join(f'<div class="frame"><img src="{ctx.data_uri(f, max_side)}"><span class="ts">{_fmt_t(float(p))}</span></div>'
                   for f, p in zip(frames, pts, strict=True))


def _bands(intervals: Sequence[tuple[float, float, str]], duration: float, *, cls: str = "band") -> str:
    out, palette = [], ("", "b", "c")
    names = list(dict.fromkeys(lab for _, _, lab in intervals))
    for s, e, lab in intervals:
        a, b = max(0.0, s) / duration * 100, min(duration, e) / duration * 100
        if b <= a:
            continue
        tone = palette[names.index(lab) % len(palette)]
        out.append(f'<div class="{cls} {tone}" style="left:{a:.3f}%;width:{b - a:.3f}%"><span class="lab">{_esc(lab)}</span></div>')
    return "".join(out)


def _ticks(duration: float, n: int = 6) -> str:
    step = max(1.0, round(duration / n))
    out, t = [], 0.0
    while t <= duration + 1e-9:
        edge = " first" if t == 0 else (" last" if duration - t < step / 2 else "")
        out.append(f'<span class="{edge.strip()}" style="left:{t / duration * 100:.3f}%">{_fmt_t(t)}</span>')
        t += step
    return '<div class="ticks">' + "".join(out) + "</div>"


# ----------------------------------------------------------------------------- core kinds


def _fill_images(ctx: RenderContext) -> dict[str, Any]:
    tiles = []
    for i in range(len(ctx)):
        cap = ", ".join(ctx.labels(i)) or ctx.clip_id(i)
        tiles.append(f'<div class="tile"><img src="{ctx.data_uri(ctx.image(i), 720)}"><div class="cap">{_esc(cap)}</div></div>')
    n = ctx.bench.num_classes
    return {"title": f"{len(ctx)} images, {n} classes" if n else f"{len(ctx)} images",
            "chips": [_media_chip(ctx.bench), _chip(f"split {ctx.split}")], "body": '<div class="grid">' + "".join(tiles) + "</div>"}


def _fill_frames(ctx: RenderContext) -> dict[str, Any]:
    n = int(ctx.options["n"])
    times = ctx.options.get("times") or _uniform(ctx.duration(0), n)
    labels = ", ".join(ctx.labels(0))
    return {"title": labels or ctx.clip_id(0), "chips": [_media_chip(ctx.bench), _chip(ctx.clip_id(0))],
            "body": f'<div class="strip">{_frame_tiles(ctx, 0, times)}</div>'}


def _fill_timeline(ctx: RenderContext) -> dict[str, Any]:
    dur = ctx.duration(0) or 10.0
    ivs = ctx.intervals(0)
    strip = _frame_tiles(ctx, 0, _uniform(dur, int(ctx.options["n"])))
    names = list(dict.fromkeys(lab for _, _, lab in ivs))
    title = ", ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else "") if names else "no annotated action"
    return {"title": title, "chips": [_media_chip(ctx.bench), _chip(f"{len(ivs)} intervals"), _chip(_fmt_t(dur))],
            "body": f'<div class="strip">{strip}</div><div class="track"><div class="timeline">{_bands(ivs, dur)}</div>{_ticks(dur)}</div>'}


def _fill_boxes(ctx: RenderContext) -> dict[str, Any]:
    k = ctx.index[0]
    box = ctx.manifest.boxes[k] if ctx.manifest.boxes is not None else None
    if box is None:
        raise PreviewError(f"{ctx.bench.name}: kind boxes needs manifest.label.kind box")
    t = ctx.options.get("time_s")
    t = float(t if t is not None else ctx.bench.spec.protocol.get("annotated_second") or (ctx.duration(0) or 10.0) / 2)
    frame, pts = ctx.frames(0, [t])
    h, w = frame.shape[1:3]
    x1, y1, x2, y2 = (float(v) for v in box)
    name = ctx.extra(0).get("class_name") or ", ".join(ctx.labels(0)) or "sound source"
    side = [f'<div class="lab">class</div><div class="big">{_esc(name)}</div>',
            f'<div class="lab">annotated frame</div><div class="mid">{_fmt_t(float(pts[0]))}</div>']
    if ctx.has_stream("audio"):
        try:
            y, rate = ctx.audio(0, seconds=MAX_SECONDS, start_s=max(0.0, t - MAX_SECONDS / 2))
            side.append(f'<div class="lab">audio around it</div>{ctx.wave_svg(y, 560, 64)}<img class="spec" src="{ctx.mel_uri(y, rate, columns=560)}">')
        except Exception:  # noqa: BLE001 - a frame without its audio is still the task
            pass
    pic = (f'<div class="pic" style="aspect-ratio:{w}/{h}"><img src="{ctx.data_uri(frame[0], 1100)}">'
           f'<div class="box" style="left:{x1 * 100:.3f}%;top:{y1 * 100:.3f}%;width:{(x2 - x1) * 100:.3f}%;height:{(y2 - y1) * 100:.3f}%">'
           f'<span class="lab">{_esc(name)}</span></div></div>')
    return {"title": f"where is the sound? {name}", "chips": [_media_chip(ctx.bench), _chip(ctx.clip_id(0))],
            "body": f'<div class="row">{pic}<div class="side card">{"".join(side)}</div></div>'}


def _fill_audio(ctx: RenderContext) -> dict[str, Any]:
    rows, secs, start = [], float(ctx.options["seconds"]), float(ctx.options["start_s"])
    for i in range(len(ctx)):
        y, rate = ctx.audio(i, start_s=start, seconds=secs)
        chips = "".join(f'<span class="chip {"a" if k == 0 else ""}">{_esc(lab)}</span>' for k, lab in enumerate(ctx.labels(i)))
        rows.append(f'<div class="clip"><div class="meta"><div class="chips">{chips}</div><div class="ts">{_esc(ctx.clip_id(i))}'
                    f' · {_fmt_t(len(y) / rate)} · {rate // 1000 if rate % 1000 == 0 else rate / 1000:g} kHz</div></div>'
                    f'<div class="sig">{ctx.wave_svg(y, 1100, 56)}<img class="spec" src="{ctx.mel_uri(y, rate, columns=1100)}"></div></div>')
    names = [", ".join(ctx.labels(i)) for i in range(len(ctx))]
    return {"title": " · ".join(n for n in names if n) or "two clips", "chips": [_media_chip(ctx.bench), _chip(f"split {ctx.split}")],
            "body": "".join(rows)}


def _fill_av(ctx: RenderContext) -> dict[str, Any]:
    dur = ctx.duration(0) or 10.0
    n = int(ctx.options["n"])
    strip = _frame_tiles(ctx, 0, _uniform(dur, n), max_side=400)
    y, rate = ctx.audio(0)
    y = y[: int(dur * rate)]
    ivs = ctx.intervals(0)
    names = list(dict.fromkeys(lab for _, _, lab in ivs)) or ctx.labels(0)
    return {"title": ", ".join(names) or ctx.clip_id(0), "chips": [_media_chip(ctx.bench), _chip(f"{len(ivs)} event(s)"), _chip(_fmt_t(dur))],
            "body": (f'<div class="stack"><div class="spans">{_bands(ivs, dur, cls="band span")}</div><div class="strip">{strip}</div>'
                     f'<div class="sig">{ctx.wave_svg(y, 1500, 72)}<img class="spec" src="{ctx.mel_uri(y, rate, columns=1500)}"></div></div>'
                     f'{_ticks(dur)}')}


def _fill_depth(ctx: RenderContext) -> dict[str, Any]:
    pairs = []
    for i in range(len(ctx)):
        rgb, depth = ctx.image(i), ctx.target(i)
        valid = depth[depth > 0]
        rng = f"{valid.min():.1f}–{valid.max():.1f} m" if valid.size else ""
        pairs.append(f'<div class="pairbox"><div class="pair"><div class="tile"><img src="{ctx.data_uri(rgb, 520)}"></div>'
                     f'<span class="arrow">→</span><div class="tile"><img src="{ctx.depth_uri(depth, 520)}"></div></div>'
                     f'<div class="ts">{_esc(ctx.clip_id(i))} · {rng}</div></div>')
    return {"title": "image → depth", "chips": [_media_chip(ctx.bench), _chip("near bright · far dark")], "body": "".join(pairs)}


def _fill_generation(ctx: RenderContext) -> dict[str, Any]:
    o = ctx.options
    secs = float(o["seconds"])
    if o["given"] == "frames":
        given = '<div class="strip">' + _frame_tiles(ctx, 0, _uniform(ctx.duration(0), 4), max_side=400) + "</div>"
        title = "video → audio"
    else:
        cell = o.get("cell") or next(iter(ctx.bench.spec.cells or {}), None)
        smp = ctx.sample(0)
        text = (smp.cells or {}).get(cell) if cell else None
        if text is None:
            raise PreviewError(f"{ctx.bench.name}: no caption cell {cell!r} (options.cell)")
        given = f'<div class="quote">“{_esc(text)}”</div>'
        title = "text → audio"
    if o.get("reference_column"):
        from yavih.media.decode import audio

        root = ctx.bench.resolved_roots()[o["reference_root"]] if o.get("reference_root") else ""
        path = Path(root) / str(ctx.extra(0)[o["reference_column"]])
        y, rate = audio(str(path))
        y = _cut(y, rate, 0.0, secs)
    else:
        y, rate = ctx.audio(0, seconds=secs)
    ref = f'{ctx.wave_svg(y, 760, 72)}<img class="spec" src="{ctx.mel_uri(y, rate, columns=760)}">'
    return {"title": title, "chips": [_chip("generation", "a"), _chip(ctx.clip_id(0))],
            "body": (f'<div class="card given"><div class="lab">given</div>{given}</div><span class="arrow">→</span>'
                     f'<div class="card ref"><div class="lab">reference · {_fmt_t(len(y) / rate)}</div>{ref}</div>')}


def _fill_retrieval(ctx: RenderContext) -> dict[str, Any]:
    o = ctx.options
    n = int(o["candidates"])
    k0 = ctx.index[0]
    pool = [k for k in range(len(ctx.manifest)) if k != k0]
    rng = random.Random(int(o["seed"]))
    picks = rng.sample(pool, min(n - 1, len(pool))) + [k0]
    rng.shuffle(picks)
    dur = ctx.duration(0)
    frames, pts = ctx.frames(0, [(dur or 10.0) / 2])
    query = f'<div class="frame"><img src="{ctx.data_uri(frames[0], 600)}"><span class="ts">{_fmt_t(float(pts[0]))}</span></div>'
    query += f'<div class="ts">{_esc(ctx.clip_id(0))}</div>'
    cards = []
    for k in picks:
        y, rate = ctx.audio(0, seconds=MAX_SECONDS, manifest_index=k)
        hit = k == k0
        cards.append(f'<div class="cand tile{" hit" if hit else ""}"><img class="spec" src="{ctx.mel_uri(y, rate, columns=260)}">'
                     f'{ctx.wave_svg(y, 260, 40)}<div class="ts">{_esc(ctx.manifest.clip_id[k])}</div>'
                     f'{"<span class=badge>match</span>" if hit else ""}</div>')
    name = ", ".join(ctx.labels(0))
    return {"title": "which sound belongs to this video?" + (f" ({name})" if name else ""),
            "chips": [_media_chip(ctx.bench), _chip(f"1 of {len(picks)} shown")],
            "body": (f'<div class="query card"><div class="lab">query video</div><div class="qstrip">{query}</div></div>'
                     f'<span class="arrow">→</span><div class="gallery">{"".join(cards)}</div>')}


def _fill_classes(ctx: RenderContext) -> dict[str, Any]:
    b = ctx.bench
    classes = [c["name"] for c in b.classes()]
    counts = np.zeros(len(classes), dtype=np.int64)
    man = b.manifest(ctx.split)
    for lab in man.label_ids:
        for j in ([] if lab is None else ([lab] if isinstance(lab, (int, np.integer)) else lab)):
            if 0 <= j < len(classes):
                counts[j] += 1
    rng = random.Random(int(ctx.options["seed"]))
    pick = sorted(rng.sample(range(len(classes)), min(int(ctx.options["n"]), len(classes))), key=lambda j: (-counts[j], classes[j]))
    chips = "".join(f'<span class="chip">{_esc(classes[j])} <b>{_num(int(counts[j]))}</b></span>' for j in pick)
    where = f"split {ctx.split}" if ctx.split is not None else "all splits"
    return {"title": f"{_num(len(classes))} classes · {_num(len(man))} clips",
            "chips": [_media_chip(b), _chip(where), _chip(f"{len(pick)} sampled")],
            "body": f'<div class="cloud chips">{chips}</div><div class="note ts">no media shown: the data licence keeps it with its owners</div>'}


def _fill_motion(ctx: RenderContext) -> dict[str, Any]:
    labels = ctx.labels(0)
    ivs = []
    try:
        ivs = ctx.intervals(0)
    except Exception:  # noqa: BLE001 - the label bar falls back to the row's labels
        pass
    title = ", ".join(dict.fromkeys([lab for _, _, lab in ivs] or labels)) or ctx.clip_id(0)
    o = ctx.options
    secs = float(o["seconds"])
    return {"title": title, "chips": [_chip(f"{float(o['start_s']):g}–{float(o['start_s']) + secs:g} s")], "body": ""}


def _core(kind: str, height: int | None, rows: int | None, fill: Callable, options: Mapping[str, Any], *, motion: bool = False) -> Component:
    return Component(kind, 1, MOTION_SIZE if motion else (WIDTH, int(height or 0)), rows, f"{kind}.html", fill, motion=motion, options=options)


#: The core kinds (docs/design-system.md has the table).
CORE_COMPONENTS: dict[str, Component] = {c.kind: c for c in (
    _core("images", 620, 8, _fill_images, {}),
    _core("frames", 330, 1, _fill_frames, {"n": 6, "times": None}),
    _core("timeline", 420, 1, _fill_timeline, {"n": 6, "intervals": None}),
    _core("boxes", 620, 1, _fill_boxes, {"time_s": None}),
    _core("audio", 520, 2, _fill_audio, {"seconds": 5.0, "start_s": 0.0}),
    _core("av", 560, 1, _fill_av, {"n": 8, "intervals": None}),
    _core("depth", 440, 3, _fill_depth, {}),
    _core("generation", 420, 1, _fill_generation,
          {"given": "caption", "cell": None, "reference_column": None, "reference_root": None, "seconds": 5.0}),
    _core("retrieval", 460, 1, _fill_retrieval, {"seed": 0, "candidates": 5}),
    _core("classes", 360, 0, _fill_classes, {"n": 24, "seed": 0}),
    _core("motion", None, 1, _fill_motion, {"start_s": 0.0, "seconds": 4.0, "intervals": None}, motion=True),
)}
CORE_KINDS: tuple[str, ...] = tuple(CORE_COMPONENTS)


# ----------------------------------------------------------------------------- render


def figure_html(bench: Any, fig: Figure, *, work_dir: Path) -> str:
    """The filled page of one figure (decodes the rows' media)."""
    comp = components(bench)[fig.kind]
    ctx = RenderContext(bench, fig, comp, work_dir=work_dir)
    try:
        filled = comp.fill(ctx)
    except (PreviewError, SpecError):
        raise
    except Exception as e:  # noqa: BLE001 - missing media, a decoder, a target: one message
        raise PreviewError(f"{bench.name}: figure {fig.id!r} ({fig.kind}): {e.__class__.__name__}: {e}") from e
    w, h = comp.size
    chips = "".join(f'<span class="chip {cls}">{inner}</span>' for cls, inner in filled.get("chips", []))
    kicker = f"<b>yavih</b> · {_esc(bench.name)}" + (f" · {_esc(fig.split)}" if fig.split is not None else "")
    return Template(comp.template_text()).substitute(
        base_css=design.base_css(), page_css=design.page_css(w, h) + comp.css, bench=_esc(bench.name), kicker=kicker,
        title=_esc(filled.get("title", "")), chips=chips, body=filled.get("body", ""))


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise PreviewError("motion figures need the ffmpeg CLI (apt install ffmpeg)")
    return exe


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise PreviewError(f"ffmpeg failed ({proc.returncode}): {(proc.stderr or '').strip()[-600:]}")


def render_motion(video: str | Path, frame_png: str | Path, mp4: Path, gif: Path, *, start_s: float, seconds: float,
                  audio: str | Path | None = None, audio_from_video: bool = True) -> tuple[Path, Path]:
    """Compose the MP4 (``2 x MOTION_SIZE``, H.264 CRF 18 yuv420p, audio kept) — the page background, the clip scaled
    into the video area, a time track filling over ``seconds``, the transparent ``frame_png`` on top — and derive the
    GIF (``MOTION_SIZE`` at 10 fps, a 256-colour palette) from it."""
    ff = _ffmpeg()
    W, H = MOTION_SIZE[0] * 2, MOTION_SIZE[1] * 2
    vh, th = _MOTION_VIDEO_H * 2, _MOTION_TRACK_H * 2
    d, r = f"{float(seconds):g}", _MOTION_MP4_FPS
    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"color=c=0x070a12:s={W}x{H}:r={r}:d={d}",
           "-ss", f"{float(start_s):g}", "-t", d, "-i", str(video),
           "-f", "lavfi", "-i", f"color=c=0x22d3ee:s={W}x{th}:r={r}:d={d}",
           "-loop", "1", "-framerate", str(r), "-t", d, "-i", str(frame_png)]
    amap = "1:a?" if audio_from_video else None
    if audio is not None:
        cmd += ["-ss", f"{float(start_s):g}", "-t", d, "-i", str(audio)]
        amap = "4:a?"
    graph = (f"[1:v]scale={W}:{vh}:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1,fps={r}[v];"
             f"[0:v][v]overlay=x=({W}-w)/2:y=({vh}-h)/2:eof_action=pass[b];"
             f"[b][2:v]overlay=x='-{W}+{W}*t/{d}':y={vh}[c];"
             f"[c][3:v]overlay=0:0,format=yuv420p[out]")
    cmd += ["-filter_complex", graph, "-map", "[out]"] + (["-map", amap] if amap else [])
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k", "-t", d, str(mp4)]
    _run(cmd)
    _run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
          f"fps=10,scale={MOTION_SIZE[0]}:{MOTION_SIZE[1]}:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", str(gif)])
    return mp4, gif


def render_figure(bench: Any, fig: Figure, *, chrome: str | None = None, html_only: bool = False,
                  work_dir: Path, out_dir: Path | None = None) -> list[Path]:
    """Render one figure: its files written into ``out_dir`` (default ``<folder>/assets/``) only after every step
    succeeded, or with ``html_only`` the filled page ``<work_dir>/<id>.html`` (fonts staged next to it). Returns the
    paths written."""
    comp = components(bench)[fig.kind]
    work_dir.mkdir(parents=True, exist_ok=True)
    page = figure_html(bench, fig, work_dir=work_dir)
    src = work_dir / f"{fig.id}.html"
    src.write_text(page, encoding="utf-8")
    design.stage_fonts(work_dir)
    if html_only:
        return [src]
    assets = Path(out_dir) if out_dir is not None else Path(bench.folder) / "assets"
    names = figure_files(bench.folder, fig, motion=comp.motion)
    staged = [work_dir / f"_out_{n}" for n in names]
    if comp.motion:
        frame = render_html(src, work_dir / f"{fig.id}.frame.png", size=comp.size, scale=2, chrome=chrome, compress=False)
        ctx = RenderContext(bench, fig, comp, work_dir=work_dir)
        from yavih.sample import stored_file

        video = stored_file(ctx.stream(0, "video"), work_dir)
        audio, from_video = None, True
        if ctx.has_stream("audio"):
            a = stored_file(ctx.stream(0, "audio"), work_dir)
            if a != video:
                audio, from_video = a, False
        o = ctx.options
        render_motion(video, frame, staged[1], staged[0], start_s=float(o["start_s"]), seconds=float(o["seconds"]),
                      audio=audio, audio_from_video=from_video)
    else:
        render_html(src, staged[0], size=comp.size, scale=2, chrome=chrome)
    assets.mkdir(parents=True, exist_ok=True)
    out = []
    for n, s in zip(names, staged, strict=True):
        shutil.move(str(s), assets / n)
        out.append(assets / n)
    return out


def _provenance(folder: Path) -> dict[str, Any]:
    try:
        d = json.loads((folder / "assets" / PROVENANCE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _media_sha256(bench: Any, fig: Figure) -> dict[str, str]:
    """The rows' recorded media sha256s when the manifest carries a ``*sha256`` column."""
    if fig.split is None or not fig.rows:
        return {}
    man = bench.manifest(fig.split)
    out = {}
    for cid in fig.rows:
        extra = man.extra[man.index_of(cid)]
        shas = [str(v) for k, v in extra.items() if k.endswith("sha256") and v]
        if shas:
            out[cid] = shas[0]
    return out


def write_provenance(folder: Path, bench: Any, rendered: Mapping[str, list[Path]], spec: PreviewSpec) -> Path:
    """Merge the rendered figures into ``assets/preview.src.json`` (entries of other figures are kept while the spec
    still declares them)."""
    folder = Path(folder)
    prev = _provenance(folder).get("figures") or {}
    ids = {f.id for f in spec.figures}
    figs = {k: v for k, v in prev.items() if k in ids}
    for fid, files in rendered.items():
        fig = spec.figure(fid)
        entry = {"kind": fig.kind, "split": fig.split, "rows": list(fig.rows), "source_sha256": figure_source_sha256(bench, fig),
                 "files": {p.name: p.stat().st_size for p in files}}
        media = _media_sha256(bench, fig)
        if media:
            entry["media_sha256"] = media
        figs[fid] = entry
    doc = {"preview_version": PREVIEW_VERSION, "design_version": design.DESIGN_VERSION,
           "figures": {k: figs[k] for k in sorted(figs, key=[f.id for f in spec.figures].index)}}
    path = folder / "assets" / PROVENANCE_FILE
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def render_folder(name: str, *, chrome: str | None = None, html_only: bool = False, only: Sequence[str] | None = None,
                  data: Mapping[str, str] | None = None) -> list[Path]:
    """Render the figures of benchmark ``name`` (``only``: these ids) and record their provenance; ``html_only`` writes
    the filled pages under ``assets/_work/`` instead (no provenance). ``data``: data-root overrides."""
    from yavih.benchmark import Benchmark

    bench = Benchmark.load(name, data=dict(data) if data else None)
    folder = Path(bench.folder)
    spec = load_spec(folder)
    if spec is None:
        raise SpecError(f"{bench.name}: no assets/{SPEC_FILE}; draft one with `python -m yavih.previews {bench.name} --suggest --write`")
    problems = validate_spec(bench, spec)
    if problems:
        raise SpecError(f"{bench.name}: " + "; ".join(problems))
    figs = [f for f in spec.figures if not only or f.id in only]
    if only and len(figs) != len(set(only)):
        raise SpecError(f"{bench.name}: --only {sorted(set(only) - {f.id for f in figs})} not in the spec")
    if html_only:
        work = folder / "assets" / "_work"
        return [p for f in figs for p in render_figure(bench, f, chrome=chrome, html_only=True, work_dir=work)]
    rendered: dict[str, list[Path]] = {}
    assets = folder / "assets"
    with tempfile.TemporaryDirectory(prefix="yavih-preview-") as tmp:
        staged = {f.id: render_figure(bench, f, chrome=chrome, work_dir=Path(tmp) / f.id, out_dir=Path(tmp) / "out")
                  for f in figs}  # every figure first: a failure leaves assets/ as it was
        for fid, files in staged.items():
            rendered[fid] = [Path(shutil.move(str(p), assets / p.name)) for p in files]
    write_provenance(folder, bench, rendered, spec)
    return [p for files in rendered.values() for p in files]


# ----------------------------------------------------------------------------- check


def _budget_findings(folder: Path) -> list[tuple[str, str, str]]:
    assets = folder / "assets"
    if not assets.is_dir():
        return []
    out, total = [], 0
    for p in sorted(assets.rglob("*")):
        if not p.is_file() or "_work" in p.relative_to(assets).parts or p.suffix == ".html":
            continue
        size = p.stat().st_size
        total += size
        limit = SIZE_BUDGET.get(p.suffix.lstrip(".").lower())
        if limit is not None and size > limit:
            out.append(("W_ASSET_SIZE", f"assets/{p.relative_to(assets).as_posix()}",
                        f"{size / 1e6:.2f} MB > the {p.suffix[1:]} budget {limit / 1e6:g} MB: fewer or smaller tiles, a shorter "
                        "excerpt; PNGs are palette-compressed when imagequant is installed (pip install imagequant)"))
    if total > SIZE_BUDGET["folder"]:
        out.append(("W_ASSET_SIZE", "assets/", f"assets/ holds {total / 1e6:.2f} MB > the folder budget {SIZE_BUDGET['folder'] / 1e6:g} MB "
                    "(every fetch downloads it; pip install imagequant compresses PNGs)"))
    return out


def check_folder(folder: str | Path, bench: Any) -> list[tuple[str, str, str]]:
    """``[(code, file, message)]`` of the folder's figures: ``E_PREVIEW_SPEC`` (an invalid spec), ``W_PREVIEW_MISSING``
    (no spec, or a declared file absent), ``W_PREVIEW_STALE`` (provenance absent or naming another source; an
    undeclared ``preview*`` / ``clip*`` file), ``W_ASSET_SIZE``. Shared by ``bench lint`` and ``--check``."""
    folder = Path(folder)
    assets, name = folder / "assets", bench.name
    out: list[tuple[str, str, str]] = []
    spec_rel = f"assets/{SPEC_FILE}"
    try:
        spec = load_spec(folder)
        problems = validate_spec(bench, spec) if spec is not None else []
    except SpecError as e:
        spec, problems = None, [str(e)]
    out += [("E_PREVIEW_SPEC", spec_rel, p) for p in problems]
    declared: set[str] = set()
    if spec is None and not problems:
        out.append(("W_PREVIEW_MISSING", spec_rel, f"no figure spec: draft one with `python -m yavih.previews {name} --suggest "
                    f"--write`, check its rows, render with `python -m yavih.previews {name} --chrome <chrome>`"))
    elif spec is not None and not problems:
        comps = components(bench)
        prov = _provenance(folder).get("figures") or {}
        for fig in spec.figures:
            files = figure_files(folder, fig, motion=comps[fig.kind].motion)
            declared |= set(files)
            missing = [f for f in files if not (assets / f).is_file()]
            if missing:
                out.append(("W_PREVIEW_MISSING", f"assets/{missing[0]}", f"figure {fig.id!r}: {', '.join(missing)} not rendered: "
                            f"`python -m yavih.previews {name} --only {fig.id} --chrome <chrome>` where the data is staged"))
                continue
            entry = prov.get(fig.id) or {}
            want = figure_source_sha256(bench, fig)
            if entry.get("source_sha256") != want:
                got = entry.get("source_sha256")
                out.append(("W_PREVIEW_STALE", f"assets/{files[0]}", f"figure {fig.id!r} was not rendered from the current spec, "
                            f"template and design (assets/{PROVENANCE_FILE} {'records ' + str(got)[:12] + '…' if got else 'has no entry'}, "
                            f"the source is {want[:12]}…): re-render with `python -m yavih.previews {name} --only {fig.id}`"))
    if not problems and assets.is_dir():
        for p in sorted(assets.iterdir()):
            if p.is_file() and _FIGURE_FILE.match(p.name) and p.name not in declared:
                out.append(("W_PREVIEW_STALE", f"assets/{p.name}", f"assets/{p.name} is no figure of assets/{SPEC_FILE}: delete it or declare it"))
    return out + _budget_findings(folder)


def _state(findings: list[tuple[str, str, str]]) -> str:
    codes = {c for c, _, _ in findings}
    for code, word in (("E_PREVIEW_SPEC", "invalid"), ("W_PREVIEW_MISSING", "missing"), ("W_PREVIEW_STALE", "stale"),
                       ("W_ASSET_SIZE", "too-big")):
        if code in codes:
            return word
    return "ok"


# ----------------------------------------------------------------------------- suggest


def _eval_split(bench: Any) -> str:
    s = bench.spec.splits
    for vals in (s.eval, s.folds or (), s.train):
        if vals:
            return str(vals[0])
    raise SpecError(f"{bench.name}: no split to draw from")


def _draw(bench: Any, split: str, n: int, rng: random.Random, *, keep: Callable[[int], bool] | None = None) -> list[str]:
    """``n`` clip_ids of ``split``: one per class in a random class order first (class-stratified), then any."""
    man = bench.manifest(split)
    rows = [k for k in range(len(man)) if keep is None or keep(k)] or list(range(len(man)))
    by_class: dict[Any, list[int]] = {}
    for k in rows:
        lab = man.label_ids[k]
        key = lab if isinstance(lab, (int, np.integer)) else (tuple(lab)[0] if lab else None)
        by_class.setdefault(key, []).append(k)
    keys = list(by_class)
    rng.shuffle(keys)
    picked: list[int] = []
    while len(picked) < min(n, len(rows)):
        for key in keys:
            left = [k for k in by_class[key] if k not in picked]
            if left and len(picked) < n:
                picked.append(rng.choice(left))
    return [man.clip_id[k] for k in picked]


def suggest_spec(bench: Any, seed: int = 0) -> dict[str, Any]:
    """A draft ``assets/preview.yaml`` (a mapping): the main figure's kind inferred from the capability, the streams,
    the label kind and the scorer; a ``clip`` motion figure for a video folder; rows drawn by ``random.Random(seed)``,
    class-stratified, from the eval split. A licence-restricted folder (``obtain.tier: needs-licence``) gets the
    data-free ``classes`` kind."""
    spec = bench.spec
    kinds = {s.kind for s in spec.streams.values()}
    caps = set(spec.capability)
    rng = random.Random(seed)
    split = _eval_split(bench)
    tier = (spec.raw.get("obtain") or {}).get("tier")
    options: dict[str, Any] = {}
    if tier == "needs-licence" and spec.manifest.classes:
        kind = "classes"
    elif "generation" in caps:
        kind = "generation"
        if not spec.cells:
            options["given"] = "frames"
        cols = [c for c in spec.manifest.columns if c.startswith("rel_audio")]
        if cols and "audio" not in kinds:
            roots = spec.raw["data"]["roots"]
            usable = [k for k, v in roots.items() if not (isinstance(v, dict) and v.get("prepare_only"))]
            options.update({"reference_column": cols[0], "reference_root": (usable or list(roots) or [None])[0]})
    elif "retrieval" in caps:
        kind = "retrieval"
    elif spec.manifest.label.kind == "box":
        kind = "boxes"
    elif spec.targets and kinds == {"image"}:
        kind = "depth"
    elif kinds == {"image"}:
        kind = "images"
    elif kinds == {"audio"}:
        kind = "audio"
    elif spec.manifest.label.kind == "segments":
        kind = "av" if "audio" in kinds else "timeline"
    else:
        events = [c for c, cs in spec.manifest.columns.items() if cs.type == "json" and cs.role == "label_aux"]
        if events and ("temporal" in caps or "streaming" in caps):
            kind = "av" if "audio" in kinds and "streaming" not in caps else "timeline"
            options["intervals"] = {"column": events[0], "format": "json_triples"}
        else:
            kind = "frames"
    n = CORE_COMPONENTS[kind].rows or 1
    rows: list[str] = [] if kind == "classes" else _draw(bench, split, n, rng)
    figs: list[dict[str, Any]] = [{"id": "preview", "kind": kind, "split": split, "rows": rows, "options": options}]
    if "video" in kinds and kind not in ("classes",):
        motion_opts: dict[str, Any] = {"start_s": 0.0, "seconds": 4.0}
        if "intervals" in options:
            motion_opts["intervals"] = dict(options["intervals"])
        figs.append({"id": "clip", "kind": "motion", "split": split, "rows": rows[:1] or _draw(bench, split, 1, rng),
                     "options": motion_opts})
    for f in figs:
        if not f["options"]:
            del f["options"]
    return {"schema_version": 1, "figures": figs}


# ----------------------------------------------------------------------------- CLI


def _sheet_images(folder: Path, spec: PreviewSpec | None) -> list[Path]:
    if spec is None:
        return []
    out = []
    for f in spec.figures:
        if f.kind == "motion":
            continue
        p = folder / "assets" / figure_files(folder, f)[0]
        if p.is_file():
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yavih.previews", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="benchmark names (default with --all: every packaged folder)")
    ap.add_argument("--all", action="store_true", help="every packaged benchmark folder")
    ap.add_argument("--data-root", action="append", default=[], metavar="KEY=PATH", help="override a data root (repeatable)")
    ap.add_argument("--chrome", help="Chrome/Chromium executable (default: $CHROME or the first on PATH)")
    ap.add_argument("--only", help="comma-separated figure ids")
    ap.add_argument("--html-only", action="store_true", help="write the filled pages under assets/_work/ instead of rendering")
    ap.add_argument("--check", action="store_true", help="render nothing: ok / missing / stale / invalid / too-big per folder, exit 1 unless all ok")
    ap.add_argument("--suggest", action="store_true", help="print a draft assets/preview.yaml")
    ap.add_argument("--write", action="store_true", help="with --suggest: write it (refused when one exists)")
    ap.add_argument("--seed", type=int, default=0, help="with --suggest: the row draw's seed")
    ap.add_argument("--contact-sheet", type=Path, help="also tile the folders' still figures into this PNG")
    ap.add_argument("--committed", action="store_true", help="with --contact-sheet: tile the committed files, render nothing")
    a = ap.parse_args(argv)
    names = list(a.names)
    if a.all:
        from yavih.benchmarks import packaged_names

        names = list(packaged_names())
    if not names:
        ap.error("give benchmark names or --all")
    if a.write and not a.suggest:
        ap.error("--write goes with --suggest")
    data: dict[str, str] = {}
    for kv in a.data_root:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            ap.error(f"--data-root wants KEY=PATH, got {kv!r}")
        data[k] = v
    only = [s for s in (a.only or "").split(",") if s] or None
    from yavih.benchmark import Benchmark

    status = 0
    sheet: list[Path] = []
    for n in names:
        try:
            bench = Benchmark.load(n, data=data or None)
            folder = Path(bench.folder)
            if a.check:
                found = check_folder(folder, bench)
                state = _state(found)
                print(f"{state:8s} {bench.name}" + (f"  ({found[0][2][:100]})" if found and state != "ok" else ""))
                status = max(status, 0 if state == "ok" else 1)
                continue
            if a.suggest:
                draft = yaml.safe_dump(suggest_spec(bench, a.seed), sort_keys=False, allow_unicode=True)
                if a.write:
                    target = folder / "assets" / SPEC_FILE
                    if target.exists():
                        print(f"error: {target} exists; edit it or delete it first", file=sys.stderr)
                        status = max(status, 1)
                        continue
                    target.parent.mkdir(exist_ok=True)
                    target.write_text(draft, encoding="utf-8")
                    print(target)
                else:
                    print(f"# {bench.name}: assets/{SPEC_FILE}\n{draft}")
                continue
            if a.committed:
                sheet += _sheet_images(folder, load_spec(folder))
                continue
            for p in render_folder(n, chrome=a.chrome, html_only=a.html_only, only=only, data=data or None):
                print(p)
            if not a.html_only:
                sheet += _sheet_images(folder, load_spec(folder))
        except (SpecError, design.RenderError) as e:
            print(f"error: {e}", file=sys.stderr)
            status = 2
    if a.contact_sheet and sheet:
        print(design.contact_sheet(sheet, a.contact_sheet))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
