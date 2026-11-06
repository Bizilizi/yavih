"""The yavih design system as code: the shared CSS and the Chrome plumbing every committed image is rendered with.

The README header images (``yavih.headers``) and the README figures (``yavih.previews``) are HTML pages filled from
``assets/design/`` of the source checkout (``tokens.css``, ``components.css``, ``fonts.css`` and the bundled Inter and
JetBrains Mono woff2 files, SIL OFL 1.1) and screenshotted with a headless Chrome or Chromium. The CSS and the fonts are
not shipped in the wheel (``available()`` says whether they are at hand); this module is, so the lint can ask.

A filled page inlines ``base_css()`` and names the fonts by relative url (``fonts/<family>/<file>.woff2``):
``stage_fonts(html_dir)`` links ``fonts/`` next to the page before Chrome opens it, so the page itself stays free of
machine paths and its sha256 is the same on every machine (the provenance the headers and figures record).
``design_sha256()`` digests ``DESIGN_VERSION``, the CSS and the font files: a figure rendered under another design is
stale. The rules (palette, typography, the dark-background rule, sizes, budgets, pixels and licences) are in
https://github.com/Bizilizi/yavih/blob/main/docs/design-system.md.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

__all__ = ["CSS_FILES", "DESIGN_DIR", "DESIGN_VERSION", "FONTS_DIR", "RenderError", "SCALE", "available", "base_css",
           "chrome", "chrome_args", "compress_png", "contact_sheet", "design_sha256", "page_css", "render_html",
           "stage_fonts"]

#: ``assets/design/`` of the source checkout (not shipped).
DESIGN_DIR = Path(__file__).resolve().parents[2] / "assets" / "design"
FONTS_DIR = DESIGN_DIR / "fonts"
#: Bumped when the rendering changes in a way the CSS text does not show (a new Chrome flag, a new compression).
DESIGN_VERSION = 1
#: The stylesheets ``base_css()`` concatenates, in cascade order.
CSS_FILES: tuple[str, ...] = ("fonts.css", "tokens.css", "components.css")
#: Device scale of every committed PNG (a 1600 CSS px page is 3200 pixels wide).
SCALE = 2
_CHROMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")
#: Background of the contact sheets (``--bg1``).
_SHEET_BG = (15, 23, 42)


class RenderError(RuntimeError):
    """A page cannot be built or rendered (no design files, no Chrome, Chrome failed)."""


def available() -> bool:
    """Whether the design files are at hand (a source checkout; an installed wheel has none)."""
    return all((DESIGN_DIR / f).is_file() for f in CSS_FILES)


def base_css() -> str:
    """``fonts.css`` + ``tokens.css`` + ``components.css``: what every filled page inlines (``RenderError`` without
    the design files)."""
    if not available():
        raise RenderError(f"{DESIGN_DIR} not found: rendering runs from a source checkout of yavih")
    return "\n".join((DESIGN_DIR / f).read_text(encoding="utf-8").rstrip("\n") for f in CSS_FILES) + "\n"


def page_css(width: int, height: int) -> str:
    """The fixed page box of a rendered image (CSS px)."""
    return f"html, body {{ width: {int(width)}px; height: {int(height)}px; overflow: hidden; }}\n"


def design_sha256() -> str:
    """sha256 (hex) of ``DESIGN_VERSION``, the CSS files and the font files (path and bytes)."""
    h = hashlib.sha256(f"yavih-design:{DESIGN_VERSION}\n".encode())
    files = [DESIGN_DIR / f for f in CSS_FILES]
    if FONTS_DIR.is_dir():
        files += sorted(p for p in FONTS_DIR.rglob("*") if p.is_file() and p.suffix == ".woff2")
    for p in files:
        if not p.is_file():
            raise RenderError(f"{p} not found: rendering runs from a source checkout of yavih")
        h.update(p.relative_to(DESIGN_DIR).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def stage_fonts(html_dir: str | os.PathLike) -> Path:
    """Make ``<html_dir>/fonts`` the design's ``fonts/`` (a symlink; a copy where links are refused), so a filled page
    in ``html_dir`` finds its fonts by relative url. An existing ``fonts`` entry is kept."""
    target = Path(html_dir) / "fonts"
    if target.exists() or target.is_symlink():
        return target
    Path(html_dir).mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(FONTS_DIR, target_is_directory=True)
    except OSError:
        shutil.copytree(FONTS_DIR, target)
    return target


def chrome(explicit: str | None = None) -> str:
    """The Chrome / Chromium executable: ``explicit``, ``$CHROME``, or the first of the usual names on ``PATH``."""
    for c in ([explicit] if explicit else []) + [os.environ.get("CHROME", "")] + list(_CHROMES):
        if c and shutil.which(c):
            return shutil.which(c)  # type: ignore[return-value]
    raise RenderError("no Chrome/Chromium found (give --chrome PATH or set CHROME)")


_find_chrome = chrome  # render_html's ``chrome`` argument shadows the function


def chrome_args(width: int, height: int, scale: int = SCALE) -> tuple[str, ...]:
    """The headless screenshot flags for a ``width`` x ``height`` CSS px page at device ``scale`` (transparent
    default background: a page that paints none renders with alpha, the motion overlay)."""
    return ("--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
            f"--force-device-scale-factor={scale}", f"--window-size={int(width)},{int(height)}",
            "--default-background-color=00000000", "--virtual-time-budget=10000")


def render_html(html_path: Path, png_path: Path, *, size: tuple[int, int] = (1600, 500), scale: int = SCALE,
                chrome: str | None = None, compress: bool = True) -> Path:
    """Screenshot ``html_path`` into ``png_path`` with headless Chrome (``chrome_args``); ``compress_png`` after."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    exe = _find_chrome(chrome)
    with tempfile.TemporaryDirectory() as profile:
        cmd = [exe, *chrome_args(*size, scale), f"--user-data-dir={profile}", f"--screenshot={png_path.resolve()}",
               Path(html_path).resolve().as_uri()]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not png_path.is_file():
        raise RenderError(f"chrome failed ({proc.returncode}): {(proc.stderr or '').strip()[-500:]}")
    if compress:
        compress_png(png_path)
    return png_path


def compress_png(path: Path) -> Path:
    """Re-save ``path`` as a 256-colour palette PNG with libimagequant (``pip install imagequant``, a maintainer-only
    tool): ~3.5x smaller with no visible change, gradients and accent colours kept. Without it the screenshot is kept
    as is (Pillow's own quantizers band the background or drop the accent colours)."""
    try:
        import imagequant
        from PIL import Image
    except ImportError:
        return path
    with Image.open(path) as im:
        q = imagequant.quantize_pil_image(im.convert("RGBA"), dithering_level=1.0, max_colors=256)
    q.save(path, optimize=True)
    return path


def contact_sheet(images: list[Path], out: Path, *, columns: int = 2, width: int = 1600, gap: int = 24) -> Path:
    """Tile ``images`` (scaled to ``width`` px each, rows as tall as their tallest tile) into one PNG at ``out``."""
    from PIL import Image

    tiles = [Image.open(p).convert("RGB") for p in images]
    tiles = [t.resize((width, round(t.height * width / t.width))) for t in tiles]
    rows = [tiles[i:i + columns] for i in range(0, len(tiles), columns)]
    heights = [max(t.height for t in r) for r in rows]
    sheet = Image.new("RGB", (columns * width + (columns + 1) * gap, sum(heights) + (len(rows) + 1) * gap), _SHEET_BG)
    y = gap
    for r, h in zip(rows, heights, strict=True):
        for c, t in enumerate(r):
            sheet.paste(t, (gap + c * (width + gap), y))
        y += h + gap
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
