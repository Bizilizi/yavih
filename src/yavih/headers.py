"""README header images of the benchmark folders: ``python -m yavih.headers [NAME ...] [--all]``.

Each folder's ``README.md`` opens with ``assets/header.png``, a 1600 x 500 (CSS px) banner rendered at device scale 2
from the folder's ``benchmark.yaml`` ``header:`` block (``name``, paper ``title``, ``venue``, a short ``abstract``)
and chips derived from the yaml: the stream kinds, the readout (multi-label when the labels are), the clip and class
counts, the folds or readings or train/eval sizes, and the headline metric by its readable name (``metric_label``).
The template (``assets/benchmark-header/template.html``) lives in the source repository and draws on the design system
(``yavih.design``: ``assets/design/`` tokens, components and the bundled fonts); rendering needs a headless Chrome or
Chromium. Both are maintainer tools: the images are for the repository's README pages and are not shipped in the
wheel, and ``bench lint`` requires the ``header:`` block, not the image.

Provenance: every render also writes ``assets/header.src.sha256``, the sha256 of the filled HTML
(``header_source_sha256``), so ``bench lint`` can tell a PNG rendered from an older ``header:`` block, chip set,
template or design (``W_HEADER_STALE``, a warning: the pixels are machine-dependent, the HTML is not). The PNG and the
sha file are committed together; the repository's ``render`` CI workflow renders every header of a pull request as an
artifact and reports (never fails on) the sha files that are not current. ``--check`` prints ``ok``, ``missing`` or
``stale`` per folder and exits 1 when any is not ``ok``.

``--main`` re-renders the repository's own ``assets/header.png`` from ``assets/header.html`` with the same command.
``--html-only`` writes the filled HTML and the sha file next to the image instead of rendering it;
``--contact-sheet PATH`` tiles the rendered images of the selected folders into one PNG (needs Pillow).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import sys
import tempfile
from pathlib import Path
from string import Template

from yavih import design
from yavih.design import RenderError, compress_png, contact_sheet

__all__ = ["CHROME_ARGS", "METRIC_LABELS", "SOURCE_SHA_FILE", "TEMPLATE_DIR", "HeaderError", "check_folder", "chips",
           "compress_png", "contact_sheet", "header_html", "header_source_sha256", "metric_label", "render_folder",
           "render_html"]

#: ``assets/benchmark-header/`` of the source checkout (not shipped).
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "assets" / "benchmark-header"
#: The page size in CSS px and the device scale of the PNG (3200 x 1000 pixels).
WIDTH, HEIGHT, SCALE = 1600, 500, 2
#: ``<folder>/assets/header.src.sha256``: ``header_source_sha256`` of the HTML the committed PNG was rendered from.
SOURCE_SHA_FILE = "header.src.sha256"
CHROME_ARGS: tuple[str, ...] = design.chrome_args(WIDTH, HEIGHT, SCALE)
_KIND_ORDER = ("video", "image", "audio")
#: The media kind an output codec carries, for the "given → generated" chip of a generation benchmark.
_CODEC_KIND = {"wav": "audio"}
#: The headline metric as people read it (the metric registry carries ids only); any other id is shown as is.
METRIC_LABELS: dict[str, str] = {
    "acc": "accuracy", "acc_mean": "mean acc", "map": "mAP", "seg_acc": "segment acc", "d1": "δ₁", "iou_mean": "mean IoU",
    "auc_pr": "mAP",  # vggsounder: the package's auc_pr, the paper's mAP
    "fd_passt": "FD (PaSST)",
    # streaming benchmarks: <metric>_<head>
    "sec_acc_gru": "segment acc (causal GRU)", "map_gru": "mAP (causal GRU)", "cap_gru": "cAP (causal GRU)",
    "event_map_gru": "event mAP (causal GRU)", "top5_recall_action_gru": "top-5 recall (causal GRU)",
}


class HeaderError(RenderError):
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
    if spec.header is not None and spec.header.chips:  # the folder says what its task is (its own scorer, readings)
        own = [html.escape(c) for c in spec.header.chips]
        return [("a", own[0]), *(("b" if i == 1 else "", c) for i, c in enumerate(own[1:], start=1))]
    kinds = {s.kind for s in spec.streams.values()}
    modality = " + ".join(k for k in _KIND_ORDER if k in kinds) or "+".join(sorted(kinds))
    scorer_label = spec.readout
    if spec.readout == bench.name:  # the folder's own scorer: name the family it belongs to, else its kind
        try:
            from yavih.readouts import registry

            sc = registry.get(spec.readout)
            scorer_label = sc.family or f"{sc.kind} (own)"
        except Exception:  # noqa: BLE001 - a header still renders
            pass
    readout = scorer_label + (" · multi-label" if spec.manifest.label.kind in ("multi", "multi_sets") else "")
    if "generation" in spec.capability:  # what the model is given and what it generates; the readout is its scorer
        given = [k for k in _KIND_ORDER if k in kinds] + (["text"] if any(c.kind == "text" for c in (spec.cells or {}).values()) else [])
        made = sorted({_CODEC_KIND.get(str(o.get("type")), name) for name, o in spec.output_contract.schema.items()})
        modality, readout = f"{' + '.join(given)} → {' + '.join(made)}", "generation"
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
        from yavih.lint import folder_readings

        own = sorted((readings & folder_readings(Path(bench.folder))) - {"a", "v", "av"})  # the folder's declared readings
        out.append(("", html.escape(" / ".join([r for r in ("a", "v", "av") if r in readings] + own))))
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
    tpl = TEMPLATE_DIR / "template.html"
    if not tpl.is_file() or not design.available():
        raise HeaderError(f"{TEMPLATE_DIR} or {design.DESIGN_DIR} not found: header rendering runs from a source checkout of yavih")
    chip_html = "".join(f'<span class="chip {cls}">{inner}</span>' for cls, inner in chips(bench))
    return Template(tpl.read_text(encoding="utf-8")).substitute(
        base_css=design.base_css(),
        page_css=design.page_css(WIDTH, HEIGHT),
        bench=html.escape(bench.name),
        name=html.escape(h.name),
        name_px=_size(h.name, ((8, "110px"), (11, "96px"), (14, "82px"), (20, "68px"), (24, "60px")), "52px"),
        title=html.escape(h.title),
        title_px=_size(h.title, ((60, "27px"), (90, "24px")), "21px"),
        venue=html.escape(h.venue),
        abstract=html.escape(h.abstract),
        abstract_px=_size(h.abstract, ((330, "18.5px"),), "17px"),
        chips=chip_html,
    )


def header_source_sha256(bench) -> str:
    """sha256 (hex) of ``header_html(bench)`` encoded as UTF-8: what ``render_folder`` records in
    ``assets/header.src.sha256`` and ``bench lint`` compares (``HeaderError`` as ``header_html``)."""
    return hashlib.sha256(header_html(bench).encode("utf-8")).hexdigest()


def render_html(html_path: Path, png_path: Path, *, chrome: str | None = None) -> Path:
    """Screenshot ``html_path`` (a 1600 x 500 page) into ``png_path`` with headless Chrome (``design.render_html``)."""
    return design.render_html(html_path, png_path, size=(WIDTH, HEIGHT), scale=SCALE, chrome=chrome)


def render_folder(name: str, *, chrome: str | None = None, html_only: bool = False) -> Path:
    """Render ``<folder>/assets/header.png`` of benchmark ``name`` (or write ``header.html`` with ``html_only``) and
    record the HTML's sha256 in ``<folder>/assets/header.src.sha256`` (after a successful render only)."""
    from yavih.benchmark import Benchmark

    bench = Benchmark.load(name)
    out_dir = Path(bench.folder) / "assets"
    out_dir.mkdir(exist_ok=True)
    page = header_html(bench)
    sha = hashlib.sha256(page.encode("utf-8")).hexdigest() + "\n"
    if html_only:
        target = out_dir / "header.html"
        target.write_text(page, encoding="utf-8")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "header.html"
            src.write_text(page, encoding="utf-8")
            design.stage_fonts(tmp)
            target = render_html(src, out_dir / "header.png", chrome=chrome)
    (out_dir / SOURCE_SHA_FILE).write_text(sha, encoding="utf-8")
    return target


def check_folder(folder: Path, bench) -> str:
    """``ok`` (the PNG exists and its sha file names the current header HTML), ``missing`` (no PNG) or ``stale``."""
    folder = Path(folder)
    if not (folder / "assets" / "header.png").is_file():
        return "missing"
    try:
        sha = (folder / "assets" / SOURCE_SHA_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return "stale"
    return "ok" if sha == header_source_sha256(bench) else "stale"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yavih.headers", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="benchmark names (default with --all: every packaged folder)")
    ap.add_argument("--all", action="store_true", help="every packaged benchmark folder")
    ap.add_argument("--main", action="store_true", help="also re-render assets/header.png from assets/header.html")
    ap.add_argument("--html-only", action="store_true", help="write <folder>/assets/header.html (and the sha file) instead of the PNG")
    ap.add_argument("--chrome", help="Chrome/Chromium executable (default: $CHROME or the first on PATH)")
    ap.add_argument("--contact-sheet", type=Path, help="also tile the rendered images into this PNG")
    ap.add_argument("--check", action="store_true", help="render nothing: print ok / missing / stale per folder, exit 1 unless all ok")
    a = ap.parse_args(argv)
    names = list(a.names)
    if a.all:
        from yavih.benchmarks import packaged_names

        names = list(packaged_names())
    if not names and not a.main:
        ap.error("give benchmark names, --all or --main")
    if a.check:
        from yavih.benchmark import Benchmark

        bad = 0
        for n in names:
            b = Benchmark.load(n)
            try:
                state = check_folder(Path(b.folder), b)
            except RenderError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
            bad += state != "ok"
            print(f"{state:8s} {b.name}")
        return 1 if bad else 0
    done: list[Path] = []
    try:
        if a.main:
            root = TEMPLATE_DIR.parent  # assets/: header.html links design/*.css by relative url
            print(render_html(root / "header.html", root / "header.png", chrome=a.chrome))
        for n in names:
            p = render_folder(n, chrome=a.chrome, html_only=a.html_only)
            done.append(p)
            print(p)
        if a.contact_sheet and done and not a.html_only:
            print(contact_sheet(done, a.contact_sheet))
    except RenderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
