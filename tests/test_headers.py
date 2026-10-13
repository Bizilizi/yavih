"""``yavih.headers``: the README header image of a benchmark folder is filled from its ``header:`` block and chips
derived from the yaml (rendering itself needs Chrome and is not run here); every render records the sha256 of its HTML,
which ``bench lint`` compares (``W_HEADER_STALE``)."""

from __future__ import annotations

import html

import pytest

from yavih import headers
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


def test_headline_metric_reads_as_a_name():
    labels = {name: headers.chips(Benchmark.load(name))[-1][1] for name in ("audioset20k", "ave", "nyuv2_depth", "vggsounder", "vggss_box", "k400")}
    assert labels == {"audioset20k": "mAP", "ave": "segment acc", "nyuv2_depth": "δ₁", "vggsounder": "AUC-PR", "vggss_box": "mean IoU",
                      "k400": "accuracy"}
    assert headers.metric_label("auc_pr_audible") == "AUC-PR (audible)" and headers.metric_label("rmse") == "rmse"


def test_header_html_fills_every_field():
    b = Benchmark.load("esc50")
    page = headers.header_html(b)
    h = b.spec.header
    assert "$" not in page  # every placeholder substituted
    for text in (h.name, h.title, h.venue, h.abstract):
        assert html.escape(text) in page


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
    page = (dst / "assets" / "header.html").read_bytes()
    sha = (dst / "assets" / headers.SOURCE_SHA_FILE).read_text(encoding="utf-8")
    assert sha == hashlib.sha256(page).hexdigest() + "\n" == headers.header_source_sha256(Benchmark.load(str(dst))) + "\n"
    assert not (dst / "assets" / "header.png").exists()


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

    assert stale() == []  # no image: nothing to compare
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

