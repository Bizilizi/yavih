"""README header images of the benchmark folders: ``python -m yavih.headers [NAME ...] [--all]``.

Each folder's ``README.md`` opens with ``assets/header.png``, a 1600 x 500 (CSS px) banner rendered at device scale 2
from the folder's ``benchmark.yaml`` ``header:`` block (``name``, paper ``title``, ``venue``, a short ``abstract``)
and chips derived from the yaml: the stream kinds, the readout (multi-label when the labels are), the clip and class
counts, the folds or readings or train/eval sizes, and the headline metric by its readable name (``metric_label``).
The template (``template.html`` and ``base.css``) lives in the source repository under ``assets/benchmark-header/``; rendering needs a headless Chrome or
Chromium. Both are maintainer tools: the images are for the repository's README pages and are not shipped in the
wheel, and ``bench lint`` requires the ``header:`` block, not the image.

``--main`` re-renders the repository's own ``assets/header.png`` from ``assets/header.html`` with the same command.
``--html-only`` writes the filled HTML next to the image instead of rendering it; ``--contact-sheet PATH`` tiles the
rendered images of the selected folders into one PNG (needs Pillow).
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from string import Template

__all__ = ["CHROME_ARGS", "METRIC_LABELS", "TEMPLATE_DIR", "chips", "contact_sheet", "header_html", "metric_label", "render_folder",
           "render_html"]

#: ``assets/benchmark-header/`` of the source checkout (not shipped).
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "assets" / "benchmark-header"
#: The page size in CSS px and the device scale of the PNG (3200 x 1000 pixels).
WIDTH, HEIGHT, SCALE = 1600, 500, 2
CHROME_ARGS: tuple[str, ...] = ("--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                                f"--force-device-scale-factor={SCALE}", f"--window-size={WIDTH},{HEIGHT}",
                                "--default-background-color=00000000")
_CHROMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")
_KIND_ORDER = ("video", "image", "audio")
#: The headline metric as people read it (the metric registry carries ids only); any other id is shown as is.
METRIC_LABELS: dict[str, str] = {
    "acc": "accuracy", "acc_mean": "mean acc", "map": "mAP", "seg_acc": "segment acc", "d1": "δ₁", "iou_mean": "mean IoU",
}


class HeaderError(RuntimeError):
    """A header cannot be built or rendered (no ``header:`` block, no template, no Chrome)."""


def _num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def metric_label(metric: str) -> str:
    """``METRIC_LABELS[metric]``; ``auc_pr_<set>`` reads ``AUC-PR`` (``AUC-PR (<set>)`` for a set other than
    ``all``); the id itself otherwise."""
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    if metric.startswith("auc_pr_") and not metric.endswith("_floor"):
        label_set = metric[len("auc_pr_"):]
        return "AUC-PR" if label_set == "all" else f"AUC-PR ({label_set})"
    return metric


def chips(bench) -> list[tuple[str, str]]:
    """``[(css class, inner html)]`` of the chips row, derived from the loaded ``Benchmark``."""
    from yavih.lint import template_readings

    spec = bench.spec
    kinds = {s.kind for s in spec.streams.values()}
    modality = " + ".join(k for k in _KIND_ORDER if k in kinds) or "+".join(sorted(kinds))
    readout = spec.readout + (" · multi-label" if spec.manifest.label.kind in ("multi", "multi_sets") else "")
    out = [("a", html.escape(modality)), ("b", html.escape(readout))]
    clips = sum(int(v) for v in spec.budget.clips.values())
    out.append(("", f"<b>{_num(clips)}</b> {'images' if kinds == {'image'} else 'clips'}"))
    if spec.manifest.classes:
        try:
            n_classes = len(bench.classes())
        except Exception:  # noqa: BLE001 - a header without the class chip is still a header
            n_classes = None
        if n_classes:
            out.append(("", f"<b>{_num(n_classes)}</b> classes"))
    folds = list(spec.splits.folds or ())
    tpl = Path(bench.folder) / "bridge_template.py"
    readings = template_readings(tpl.read_text(encoding="utf-8")) if tpl.is_file() else set()
    if folds:
        out.append(("", f"<b>{len(folds)}</b> folds"))
    if len(readings) > 1:
        out.append(("", " / ".join(r for r in ("a", "v", "av") if r in readings)))
    elif not folds:
        roles = {"train": 0, "eval": 0}
        for split, n in spec.budget.clips.items():
            role = spec.splits.role_of(split)
            if role in roles:
                roles[role] += int(n)
        if roles["train"] and roles["eval"]:
            out.append(("", f"<b>{_num(roles['train'])}</b> train · <b>{_num(roles['eval'])}</b> eval"))
    out.append(("", html.escape(metric_label(spec.metrics.headline))))
    return out


def _size(text: str, steps: tuple[tuple[int, str], ...], last: str) -> str:
    return next((px for limit, px in steps if len(text) <= limit), last)


def header_html(bench) -> str:
    """The filled template for one loaded ``Benchmark`` (``HeaderError`` without a ``header:`` block)."""
    h = bench.spec.header
    if h is None:
        raise HeaderError(f"{bench.name}: benchmark.yaml has no header: block")
    tpl, css = TEMPLATE_DIR / "template.html", TEMPLATE_DIR / "base.css"
    if not tpl.is_file() or not css.is_file():
        raise HeaderError(f"{TEMPLATE_DIR} not found: header rendering runs from a source checkout of yavih")
    chip_html = "".join(f'<span class="chip {cls}">{inner}</span>' for cls, inner in chips(bench))
    return Template(tpl.read_text(encoding="utf-8")).substitute(
        base_css=css.read_text(encoding="utf-8"),
        bench=html.escape(bench.name),
        name=html.escape(h.name),
        name_px=_size(h.name, ((8, "110px"), (11, "96px"), (14, "82px")), "68px"),
        title=html.escape(h.title),
        title_px=_size(h.title, ((60, "27px"), (90, "24px")), "21px"),
        venue=html.escape(h.venue),
        abstract=html.escape(h.abstract),
        abstract_px=_size(h.abstract, ((330, "18.5px"),), "17px"),
        chips=chip_html,
    )


def _chrome(explicit: str | None = None) -> str:
    for c in ([explicit] if explicit else []) + [os.environ.get("CHROME", "")] + list(_CHROMES):
        if c and shutil.which(c):
            return shutil.which(c)  # type: ignore[return-value]
    raise HeaderError("no Chrome/Chromium found (give --chrome PATH or set CHROME)")


def render_html(html_path: Path, png_path: Path, *, chrome: str | None = None) -> Path:
    """Screenshot ``html_path`` into ``png_path`` with headless Chrome (``CHROME_ARGS``)."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        cmd = [_chrome(chrome), *CHROME_ARGS, f"--user-data-dir={profile}", f"--screenshot={png_path.resolve()}",
               html_path.resolve().as_uri()]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not png_path.is_file():
        raise HeaderError(f"chrome failed ({proc.returncode}): {(proc.stderr or '').strip()[-500:]}")
    return png_path


def render_folder(name: str, *, chrome: str | None = None, html_only: bool = False) -> Path:
    """Render ``<folder>/assets/header.png`` of benchmark ``name`` (or write ``header.html`` with ``html_only``)."""
    from yavih.benchmark import Benchmark

    bench = Benchmark.load(name)
    out_dir = Path(bench.folder) / "assets"
    out_dir.mkdir(exist_ok=True)
    page = header_html(bench)
    if html_only:
        target = out_dir / "header.html"
        target.write_text(page, encoding="utf-8")
        return target
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "header.html"
        src.write_text(page, encoding="utf-8")
        return render_html(src, out_dir / "header.png", chrome=chrome)


def contact_sheet(images: list[Path], out: Path, *, columns: int = 2, width: int = 1600, gap: int = 24) -> Path:
    """Tile ``images`` (scaled to ``width`` px each) into one PNG at ``out``."""
    from PIL import Image

    tiles = [Image.open(p).convert("RGB") for p in images]
    tiles = [t.resize((width, round(t.height * width / t.width))) for t in tiles]
    th = max(t.height for t in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width + (columns + 1) * gap, rows * th + (rows + 1) * gap), (24, 24, 27))
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet.paste(t, (gap + c * (width + gap), gap + r * (th + gap)))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yavih.headers", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="benchmark names (default with --all: every packaged folder)")
    ap.add_argument("--all", action="store_true", help="every packaged benchmark folder")
    ap.add_argument("--main", action="store_true", help="also re-render assets/header.png from assets/header.html")
    ap.add_argument("--html-only", action="store_true", help="write <folder>/assets/header.html instead of the PNG")
    ap.add_argument("--chrome", help="Chrome/Chromium executable (default: $CHROME or the first on PATH)")
    ap.add_argument("--contact-sheet", type=Path, help="also tile the rendered images into this PNG")
    a = ap.parse_args(argv)
    names = list(a.names)
    if a.all:
        from yavih.benchmarks import packaged_names

        names = list(packaged_names())
    if not names and not a.main:
        ap.error("give benchmark names, --all or --main")
    done: list[Path] = []
    try:
        if a.main:
            root = TEMPLATE_DIR.parent
            print(render_html(root / "header.html", root / "header.png", chrome=a.chrome))
        for n in names:
            p = render_folder(n, chrome=a.chrome, html_only=a.html_only)
            done.append(p)
            print(p)
        if a.contact_sheet and done and not a.html_only:
            print(contact_sheet(done, a.contact_sheet))
    except HeaderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
