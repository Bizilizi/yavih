"""``yavih.headers``: the README header image of a benchmark folder is filled from its ``header:`` block and chips
derived from the yaml (rendering itself needs Chrome and is not run here); every render records the sha256 of its HTML,
which ``bench lint`` compares (``W_HEADER_STALE``)."""

from __future__ import annotations

import html

import pytest

from yavih import design, headers
from yavih.benchmark import Benchmark

pytestmark = pytest.mark.skipif(not (headers.TEMPLATE_DIR / "template.html").is_file(),
                                reason="needs the source checkout's assets/benchmark-header/")


def test_chips_are_derived_from_the_yaml():
    chips = [inner for _, inner in headers.chips(Benchmark.load("esc50"))]
    assert chips == ["audio", "clf", "<b>2 000</b> clips", "<b>50</b> classes", "<b>5</b> folds", "mean acc"]
    multi = [inner for _, inner in headers.chips(Benchmark.load("vggsounder"))]
    assert multi[:2] == ["video + audio", "clf · multi-label"] and "a / v / av" in multi
    images = [inner for _, inner in headers.chips(Benchmark.load("in1k"))]
    assert images[2].endswith(" images") and any(" train · " in c for c in images)
    t2a = [inner for _, inner in headers.chips(Benchmark.load("audiocaps_t2a"))]
    assert t2a == ["text → audio", "generation", "<b>776</b> clips", "FD (PaSST)"]
    assert [inner for _, inner in headers.chips(Benchmark.load("vggsounder_v2a"))][:2] == ["video → audio", "generation"]


def test_headline_metric_reads_as_a_name():
    labels = {name: headers.chips(Benchmark.load(name))[-1][1] for name in ("audioset20k", "ave", "nyuv2_depth", "vggsounder", "vggss_box", "k400")}
    assert labels == {"audioset20k": "mAP", "ave": "segment acc", "nyuv2_depth": "δ₁", "vggsounder": "mAP", "vggss_box": "mean IoU",
                      "k400": "accuracy"}
    assert headers.metric_label("auc_pr_audible") == "AUC-PR (audible)" and headers.metric_label("rmse") == "rmse"


def test_header_html_fills_every_field():
    b = Benchmark.load("esc50")
    page = headers.header_html(b)
    h = b.spec.header
    assert "$" not in page  # every placeholder substituted
    for text in (h.name, h.title, h.venue, h.abstract):
        assert html.escape(text) in page
    assert design.base_css() in page and design.page_css(headers.WIDTH, headers.HEIGHT) in page  # the design system
    assert 'url("fonts/inter/' in page and str(design.DESIGN_DIR) not in page  # fonts by relative url: no machine path
    assert design.design_sha256() in page  # the design digest (fonts included) is part of the recorded HTML


def test_header_sha_follows_the_design_digest(monkeypatch):
    b = Benchmark.load("esc50")
    sha = headers.header_source_sha256(b)
    monkeypatch.setattr(design, "design_sha256", lambda: "0" * 64)  # e.g. a font swapped under the same file name
    assert headers.header_source_sha256(b) != sha


def test_header_html_needs_the_header_block(tmp_path):
    import shutil

    import yaml

    dst = tmp_path / "esc50"
    shutil.copytree(Benchmark.load("esc50").folder, dst, ignore=shutil.ignore_patterns("__pycache__", "assets"))
    y = yaml.safe_load((dst / "benchmark.yaml").read_text(encoding="utf-8"))
    del y["header"]
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(headers.HeaderError):
        headers.header_html(Benchmark.load(str(dst)))


def test_abstract_word_limit_is_a_schema_error(tmp_path):
    import shutil

    import yaml

    from yavih.benchmark import SchemaError

    dst = tmp_path / "esc50"
    shutil.copytree(Benchmark.load("esc50").folder, dst, ignore=shutil.ignore_patterns("__pycache__", "assets"))
    y = yaml.safe_load((dst / "benchmark.yaml").read_text(encoding="utf-8"))
    y["header"]["abstract"] = "word " * 61
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(SchemaError, match="at most 60"):
        Benchmark.load(str(dst))


def test_explicit_header_chips_replace_the_derived_ones(tmp_path):
    """``header.chips`` (1 to 6 strings of at most 40 characters) replace the derived chips; seven are a schema error."""
    import shutil

    import yaml

    from yavih.benchmark import SchemaError

    dst = tmp_path / "esc50"
    shutil.copytree(Benchmark.load("esc50").folder, dst, ignore=shutil.ignore_patterns("__pycache__", "assets"))
    y = yaml.safe_load((dst / "benchmark.yaml").read_text(encoding="utf-8"))
    y["header"]["chips"] = ["audio", "own task", "2 000 clips", "a <b>"]
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert headers.chips(Benchmark.load(str(dst))) == [("a", "audio"), ("b", "own task"), ("", "2 000 clips"), ("", "a &lt;b&gt;")]
    y["header"]["chips"] = [f"chip {i}" for i in range(7)]
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(SchemaError, match="1 to 6"):
        Benchmark.load(str(dst))


def test_contact_sheet_tiles_images(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (320, 100), (i * 40, 0, 0)).save(p)
        paths.append(p)
    out = headers.contact_sheet(paths, tmp_path / "sheet.png", columns=2, width=160, gap=4)
    w, h = Image.open(out).size
    assert (w, h) == (2 * 160 + 3 * 4, 2 * 50 + 3 * 4)


def _esc50_copy(tmp_path):
    import shutil

    dst = tmp_path / "esc50"
    shutil.copytree(Benchmark.load("esc50").folder, dst, ignore=shutil.ignore_patterns("__pycache__", "assets"))
    return dst


def test_html_only_writes_sha(tmp_path):
    """``--html-only`` writes the filled HTML and ``assets/header.src.sha256`` = sha256 of exactly that HTML."""
    import hashlib

    dst = _esc50_copy(tmp_path)
    assert headers.main([str(dst), "--html-only"]) == 0
    assert (dst / "assets" / "fonts").exists()  # the page opens with its fonts
    page = (dst / "assets" / "header.html").read_bytes()
    sha = (dst / "assets" / headers.SOURCE_SHA_FILE).read_text(encoding="utf-8")
    assert sha == hashlib.sha256(page).hexdigest() + "\n" == headers.header_source_sha256(Benchmark.load(str(dst))) + "\n"
    assert not (dst / "assets" / "header.png").exists()


def test_check_prints_ok_missing_stale(tmp_path, capsys):
    """``--check`` renders nothing: ``missing`` without a PNG, ``stale`` when the sha file names another HTML, ``ok``
    otherwise; exit 1 unless every folder is ok."""
    import yaml

    dst = _esc50_copy(tmp_path)
    assert headers.main([str(dst), "--check"]) == 1 and capsys.readouterr().out.startswith("missing")
    assert headers.main([str(dst), "--html-only"]) == 0
    (dst / "assets" / "header.png").write_bytes(b"png")
    capsys.readouterr()
    assert headers.main([str(dst), "--check"]) == 0 and capsys.readouterr().out.startswith("ok")
    from PIL import Image

    Image.new("RGBA", (8, 4), (0, 0, 0, 0)).save(dst / "assets" / "header.png")  # an incomplete render
    assert headers.main([str(dst), "--check"]) == 1 and capsys.readouterr().out.startswith("stale")
    from yavih.lint import lint_folder

    found = [f for f in lint_folder(dst, checks=["header"]).warnings if f.code == "W_HEADER_STALE"]
    assert len(found) == 1 and "transparent pixels" in found[0].message
    Image.new("RGBA", (8, 4), (7, 10, 18, 255)).save(dst / "assets" / "header.png")
    assert headers.main([str(dst), "--check"]) == 0
    capsys.readouterr()
    y = yaml.safe_load((dst / "benchmark.yaml").read_text(encoding="utf-8"))
    y["header"]["venue"] = "Another venue"
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert headers.main([str(dst), "--check"]) == 1 and capsys.readouterr().out.startswith("stale")


def test_render_writes_sha_after_the_png(tmp_path, monkeypatch):
    dst = _esc50_copy(tmp_path)

    def fake_render(html_path, png_path, *, chrome=None):
        png_path.write_bytes(b"png")
        return png_path

    monkeypatch.setattr(headers, "render_html", fake_render)
    assert headers.render_folder(str(dst)) == dst / "assets" / "header.png"
    sha = (dst / "assets" / headers.SOURCE_SHA_FILE).read_text(encoding="utf-8").strip()
    assert sha == headers.header_source_sha256(Benchmark.load(str(dst))) and not (dst / "assets" / "header.html").exists()

    def failing_render(html_path, png_path, *, chrome=None):
        raise headers.HeaderError("no chrome")

    (dst / "assets" / headers.SOURCE_SHA_FILE).unlink()
    monkeypatch.setattr(headers, "render_html", failing_render)
    with pytest.raises(headers.HeaderError):
        headers.render_folder(str(dst))
    assert not (dst / "assets" / headers.SOURCE_SHA_FILE).exists()  # no render, no provenance


def test_lint_warns_stale_header(tmp_path, monkeypatch):
    """``W_HEADER_STALE`` (a warning) when the PNG exists and its sha file is missing or names another header; silent
    without a PNG and where the template is absent (an installed wheel)."""
    import yaml

    from yavih.lint import lint_folder

    dst = _esc50_copy(tmp_path)

    def stale() -> list:
        rep = lint_folder(dst, checks=["header"])
        assert rep.ok
        return [f for f in rep.warnings if f.code == "W_HEADER_STALE"]

    assert stale() == []  # no image: nothing to compare (W_HEADER_MISSING instead, below)
    (dst / "assets").mkdir()
    (dst / "assets" / "header.png").write_bytes(b"png")
    found = stale()
    assert len(found) == 1 and found[0].severity == "warning" and "is missing" in found[0].message
    assert headers.main([str(dst), "--html-only"]) == 0
    assert stale() == []
    y = yaml.safe_load((dst / "benchmark.yaml").read_text(encoding="utf-8"))
    y["header"]["title"] = "Another title"
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert [f.file for f in stale()] == ["assets/header.png"]
    monkeypatch.setattr(headers, "TEMPLATE_DIR", tmp_path / "no-template")
    assert stale() == []



def test_lint_warns_missing_header_in_a_source_checkout_only(tmp_path, monkeypatch):
    """The header image lives in the folder: without ``assets/header.png`` lint warns ``W_HEADER_MISSING`` (never an
    error) where the header template is at hand; an installed wheel (no template) is silent."""
    from yavih.lint import lint_folder

    dst = _esc50_copy(tmp_path)

    def missing() -> list:
        rep = lint_folder(dst, checks=["header"])
        assert rep.ok
        return [f for f in rep.warnings if f.code == "W_HEADER_MISSING"]

    found = missing()
    assert len(found) == 1 and found[0].severity == "warning" and found[0].file == "assets/header.png"
    (dst / "assets").mkdir()
    (dst / "assets" / "header.png").write_bytes(b"png")
    assert missing() == []
    (dst / "assets" / "header.png").unlink()
    monkeypatch.setattr(headers, "TEMPLATE_DIR", tmp_path / "no-template")
    assert missing() == []
