"""``yavih.previews``: every core figure kind fills its template from synthetic folders; the provenance is computed
without decoding; renders write their files and provenance only after success; the ``preview`` lint check; the
suggested spec; a folder kind in ``preview.py``; motion writes a GIF and an MP4. Chrome is never run here: a
monkeypatched ``previews.render_html`` stands in for it."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil

import pytest
import yaml

from yavih import design, previews
from yavih.benchmark import Benchmark

pytestmark = pytest.mark.skipif(not design.available() or not previews.TEMPLATES_DIR.is_dir(),
                                reason="needs the source checkout's assets/design/")

pytest.importorskip("av")
pytest.importorskip("soundfile")
pytest.importorskip("PIL")
pytest.importorskip("scipy")


def _synth(root, readout, name, *, streams=None, n_classes=5, n_eval=8):
    from yavih.testing import make_synthetic_benchmark

    return make_synthetic_benchmark(root, readout, n_train=2, n_eval=n_eval, n_classes=n_classes, name=name, streams=streams)


def _with_events(folder, events):
    """Add a thumos-style ``events`` json column ([[start_s, end_s, class], ...]) to every row of the manifest."""
    from yavih.testing import _csv_bytes, read_manifest_rows

    rows = read_manifest_rows(folder)
    header = [*rows[0], "events"]
    data = _csv_bytes(header, [[*r.values(), json.dumps(events)] for r in rows])
    with open(folder / "manifest.csv.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(data)
    sha = hashlib.sha256(data).hexdigest()
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["manifest"]["sha256"] = y["manifest"]["upstream"]["sha256"] = sha
    y["manifest"]["columns"]["events"] = {"type": "json", "role": "extra"}
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return folder


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = tmp_path_factory.mktemp("previews")
    return {
        "image": _synth(root, "clf", "p_image", streams={"image": "image"}, n_classes=8),
        "video": _synth(root, "clf", "p_video"),
        "audio": _synth(root, "clf", "p_audio", streams={"audio": "audio"}),
        "av": _synth(root, "clf", "p_av", streams={"video": "video", "audio": "audio"}, n_classes=3),
        "seq": _synth(root, "seq", "p_seq"),
        "box": _synth(root, "box", "p_box"),
        "grid": _synth(root, "grid", "p_grid"),
        "events": _with_events(_synth(root, "clf", "p_events"), [[1.0, 3.5, 0], [6.0, 8.0, 2], [7.0, 9.0, 1]]),
    }


def _rows(folder, n, start=0):
    return list(Benchmark.load(str(folder)).manifest("val").clip_id[start:start + n])


#: kind -> (folder, rows (n, start), options, {needle: count} in the filled page)
CASES = {
    "images": ("image", (8, 0), {}, {"data:image/png": 8}),
    "frames": ("video", (1, 0), {}, {"data:image/png": 6}),
    "timeline": ("seq", (1, 1), {}, {"data:image/png": 6, 'class="band': 1}),
    "boxes": ("box", (1, 0), {}, {"data:image/png": 2, "<svg": 1, 'class="box"': 1}),
    "audio": ("audio", (2, 0), {}, {"data:image/png": 2, "<svg": 2}),
    "av": ("seq", (1, 1), {}, {"data:image/png": 9, "<svg": 1, 'class="band span': 1}),
    "depth": ("grid", (3, 0), {}, {"data:image/png": 6}),
    "generation": ("av", (1, 0), {"given": "frames"}, {"data:image/png": 5, "<svg": 1}),
    "retrieval": ("av", (1, 0), {}, {"data:image/png": 6, "<svg": 5, " hit": 1}),
    "classes": ("video", (0, 0), {}, {"data:image/png": 0, 'class="chip"': 5}),
    "motion": ("av", (1, 0), {"seconds": 2.0}, {"data:image/png": 0}),
}


@pytest.mark.parametrize("kind", sorted(CASES))
def test_every_core_kind_fills_its_template(folders, kind, tmp_path):
    key, (n, start), options, needles = CASES[kind]
    folder = folders[key]
    fig = previews.Figure("preview", kind, "val", tuple(_rows(folder, n, start)), options)
    bench = Benchmark.load(str(folder))
    assert previews.validate_spec(bench, previews.PreviewSpec(1, (fig,))) == []
    page = previews.figure_html(bench, fig, work_dir=tmp_path)
    assert "$" not in page.replace("$ ", "")  # every placeholder substituted
    assert design.base_css() in page and f"width: {previews.components(bench)[kind].size[0]}px" in page
    for needle, count in needles.items():
        assert page.count(needle) == count, (needle, page.count(needle))


def test_options_are_type_and_range_checked():
    """The pixels-and-licences bounds and the option types are refused when the spec is read, not when it renders."""
    ok = {"n": 8, "times": [0.5, 1.0], "seed": 3, "candidates": 5, "start_s": 0.0, "seconds": 5.0}
    assert previews._option_problems("frames", ok) == []
    assert previews._option_problems("classes", {"n": 32}) == []
    for bad in ({"n": 9}, {"n": 2.0}, {"times": [1.0] * 9}, {"times": [-1]}, {"seed": "x"}, {"candidates": 1},
                {"start_s": -0.5}, {"seconds": 0}, {"n": True}):
        assert previews._option_problems("frames", bad), bad
    assert previews._option_problems("classes", {"n": 65})


def test_json_intervals_draw_bands(folders, tmp_path):
    folder = folders["events"]
    bench = Benchmark.load(str(folder))
    opts = {"intervals": {"column": "events", "format": "json_triples"}}
    page = previews.figure_html(bench, previews.Figure("preview", "timeline", "val", tuple(_rows(folder, 1)), opts), work_dir=tmp_path)
    assert page.count('class="band') == 3 and "class_02" in page and "3 intervals" in page
    named = {**opts, "intervals": {**opts["intervals"], "names": ["Basketball", "Diving", "Golf"]}}
    page = previews.figure_html(bench, previews.Figure("preview", "timeline", "val", tuple(_rows(folder, 1)), named), work_dir=tmp_path)
    assert "Golf" in page and "Diving" in page


def _spec(folder, figures):
    (folder / "assets").mkdir(exist_ok=True)
    (folder / "assets" / "preview.yaml").write_text(yaml.safe_dump({"schema_version": 1, "figures": figures}, sort_keys=False),
                                                    encoding="utf-8")


def _copy(folder, tmp_path):
    dst = tmp_path / folder.name
    shutil.copytree(folder, dst)
    return dst


def test_source_sha256_needs_no_media_and_follows_spec_template_design(folders, tmp_path, monkeypatch):
    folder = folders["audio"]
    bench = Benchmark.load(str(folder))
    fig = previews.Figure("preview", "audio", "val", tuple(_rows(folder, 2)), {"seconds": 5.0})
    sha = previews.figure_source_sha256(bench, fig)
    media = folder.parent / "media" / folder.name
    moved = tmp_path / "media-away"
    shutil.move(str(media), moved)  # no media: nothing is decoded
    try:
        assert previews.figure_source_sha256(bench, fig) == sha
    finally:
        shutil.move(str(moved), media)
    assert previews.figure_source_sha256(bench, previews.Figure("preview", "audio", "val", fig.rows, {"seconds": 4.0})) != sha
    monkeypatch.setattr(design, "DESIGN_VERSION", design.DESIGN_VERSION + 1)
    assert previews.figure_source_sha256(bench, fig) != sha
    monkeypatch.undo()
    default = previews.CORE_COMPONENTS["audio"]
    monkeypatch.setitem(previews.CORE_COMPONENTS, "audio", previews.Component(
        "audio", default.version, default.size, default.rows, default.template, default.fill,
        options={**default.options, "start_s": 1.0}))
    assert previews.figure_source_sha256(bench, fig) != sha  # a changed default the entry does not set
    monkeypatch.undo()
    comp = previews.CORE_COMPONENTS["audio"]
    monkeypatch.setitem(previews.CORE_COMPONENTS, "audio", previews.Component(
        "audio", comp.version, comp.size, comp.rows, comp.template_text() + "<!-- edited -->", comp.fill, options=comp.options))
    assert previews.figure_source_sha256(bench, fig) != sha


def _fake_render(calls=None, fail_on=None):
    def render(html_path, png_path, *, size=(1600, 500), scale=2, chrome=None, compress=True):
        if fail_on and fail_on in html_path.name:
            raise design.RenderError("no chrome")
        if calls is not None:
            calls.append(html_path.name)
        png_path.write_bytes(b"png:" + html_path.name.encode())
        return png_path
    return render


def test_render_writes_files_and_provenance_after_success_only(folders, tmp_path, monkeypatch):
    folder = _copy(folders["audio"], tmp_path)
    rows = _rows(folder, 2)
    _spec(folder, [{"id": "preview", "kind": "audio", "split": "val", "rows": rows},
                   {"id": "names", "kind": "classes", "split": "val"}])
    monkeypatch.setattr(previews, "render_html", _fake_render(fail_on="names"))
    with pytest.raises(design.RenderError):
        previews.render_folder(str(folder))
    assert not (folder / "assets" / "preview.png").exists() and not (folder / "assets" / previews.PROVENANCE_FILE).exists()
    calls: list[str] = []
    monkeypatch.setattr(previews, "render_html", _fake_render(calls))
    out = previews.render_folder(str(folder))
    assert sorted(p.name for p in out) == ["preview-names.png", "preview.png"]
    prov = json.loads((folder / "assets" / previews.PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert list(prov["figures"]) == ["preview", "names"] and prov["figures"]["preview"]["rows"] == rows
    bench = Benchmark.load(str(folder))
    spec = previews.load_spec(folder)
    assert prov["figures"]["names"]["source_sha256"] == previews.figure_source_sha256(bench, spec.figure("names"))
    before = prov["figures"]["preview"]
    calls.clear()
    assert previews.main([str(folder), "--only", "names"]) == 0 and calls == ["names.html"]
    assert json.loads((folder / "assets" / previews.PROVENANCE_FILE).read_text(encoding="utf-8"))["figures"]["preview"] == before


def _codes(folder, check="preview"):
    from yavih.lint import lint_folder

    rep = lint_folder(folder, checks=[check])
    return rep, sorted({(f.code, f.file) for f in rep.findings})


def test_lint_preview_codes(folders, tmp_path, monkeypatch):
    folder = _copy(folders["audio"], tmp_path)
    rep, found = _codes(folder)
    assert rep.ok and found == [("W_PREVIEW_MISSING", "assets/preview.yaml")]
    rows = _rows(folder, 2)
    _spec(folder, [{"id": "preview", "kind": "audio", "split": "val", "rows": rows}])
    assert _codes(folder)[1] == [("W_PREVIEW_MISSING", "assets/preview.png")]
    assert previews.main([str(folder), "--html-only"]) == 0  # the filled pages only: still nothing rendered
    assert (folder / "assets" / "_work" / "preview.html").is_file() and (folder / "assets" / "_work" / "fonts").exists()
    assert _codes(folder)[1] == [("W_PREVIEW_MISSING", "assets/preview.png")]
    monkeypatch.setattr(previews, "render_html", _fake_render())
    previews.render_folder(str(folder))
    shutil.rmtree(folder / "assets" / "_work")
    assert _codes(folder)[1] == []
    assert previews.main([str(folder), "--check"]) == 0
    _spec(folder, [{"id": "preview", "kind": "audio", "split": "val", "rows": rows, "options": {"seconds": 4.0}}])
    assert _codes(folder)[1] == [("W_PREVIEW_STALE", "assets/preview.png")]
    assert previews.main([str(folder), "--check"]) == 1
    _spec(folder, [{"id": "preview", "kind": "audio", "split": "val", "rows": rows}])
    (folder / "assets" / "clip.gif").write_bytes(b"gif")
    (folder / "assets" / "walk.mp4").write_bytes(b"mp4")  # a removed motion figure of another id
    assert _codes(folder)[1] == [("W_PREVIEW_STALE", "assets/clip.gif"), ("W_PREVIEW_STALE", "assets/walk.mp4")]
    (folder / "assets" / "clip.gif").unlink()
    (folder / "assets" / "walk.mp4").unlink()
    monkeypatch.setitem(previews.SIZE_BUDGET, "png", 3)
    assert _codes(folder)[1] == [("W_ASSET_SIZE", "assets/preview.png")]
    monkeypatch.undo()
    for bad in ({"rows": rows[:1]}, {"split": "nope"}, {"kind": "nope"}, {"options": {"colour": "red"}}, {"rows": [rows[0], "zz"]},
                {"options": {"seconds": 6}}, {"options": {"start_s": -1}}, {"options": {"start_s": "0"}}):
        _spec(folder, [{"id": "preview", "kind": "audio", "split": "val", "rows": rows, **bad}])
        rep, found = _codes(folder)
        assert not rep.ok and found == [("E_PREVIEW_SPEC", "assets/preview.yaml")], bad
    (folder / "assets" / "preview.yaml").write_text("schema_version: 1\nfigures: [", encoding="utf-8")
    assert _codes(folder)[1] == [("E_PREVIEW_SPEC", "assets/preview.yaml")]


def test_lint_warns_a_readme_linking_a_missing_asset(folders, tmp_path):
    folder = _copy(folders["audio"], tmp_path)
    readme = folder / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n![figure](assets/nope.png)\n", encoding="utf-8")
    rep, found = _codes(folder, "readme")
    assert rep.ok and ("W_README_ASSET", "README.md") in found


def test_suggest_spec_is_deterministic_and_valid(folders):
    for key, kind in (("audio", "audio"), ("seq", "av"), ("box", "boxes"), ("grid", "depth"), ("image", "images"), ("video", "frames")):
        bench = Benchmark.load(str(folders[key]))
        draft = previews.suggest_spec(bench, seed=3)
        assert draft == previews.suggest_spec(bench, seed=3)
        spec = previews._parse_spec(yaml.safe_load(yaml.safe_dump(draft)), "draft")
        assert spec.figures[0].kind == kind and previews.validate_spec(bench, spec) == [], key
    seq = Benchmark.load(str(folders["seq"]))
    draft = previews.suggest_spec(seq)
    clip = draft["figures"][1]
    man = seq.manifest("val")
    first = int(man.segments[man.index_of(clip["rows"][0])][0])  # 1-s segments: the event's start in seconds
    assert clip["kind"] == "motion" and clip["options"]["start_s"] == max(0.0, first - 1.0)
    image = Benchmark.load(str(folders["image"]))
    labels = [image.manifest("val").label_ids[image.manifest("val").index_of(c)] for c in previews.suggest_spec(image)["figures"][0]["rows"]]
    assert len(set(labels)) == 8  # class-stratified
    for name in ("esc50", "ave"):  # the packaged folders: no data needed to draft
        bench = Benchmark.load(name)
        assert previews.validate_spec(bench, previews._parse_spec(previews.suggest_spec(bench), name)) == []


def test_suggest_write_refuses_to_overwrite(folders, tmp_path):
    folder = _copy(folders["audio"], tmp_path)
    assert previews.main([str(folder), "--suggest", "--write"]) == 0 and (folder / "assets" / "preview.yaml").is_file()
    assert previews.main([str(folder), "--suggest", "--write"]) == 1


_PREVIEW_PY = '''
from yavih.previews import Component


def fill(ctx):
    return {"title": "own " + ctx.clip_id(0), "chips": [("a", "mine")], "body": '<div class="own">hello</div>'}


COMPONENTS = [Component("KIND", 1, (1600, 300), 1, "base.html", fill, css=".own { color: red; }", options={"tone": "red"})]


def intervals(bench, split, extra):
    return [(0.0, 2.0, "own event")]
'''


def test_a_folder_preview_py_registers_a_kind(folders, tmp_path):
    from yavih.extensions import folder_extensions

    folder = _copy(folders["video"], tmp_path)
    (folder / "preview.py").write_text(_PREVIEW_PY.replace("KIND", "ownstrip"), encoding="utf-8")
    bench = Benchmark.load(str(folder))
    assert "ownstrip" in previews.components(bench) and folder_extensions(bench)["preview"] == ["ownstrip", "intervals"]
    fig = previews.Figure("own", "ownstrip", "val", tuple(_rows(folder, 1)), {"tone": "blue"})
    assert previews.validate_spec(bench, previews.PreviewSpec(1, (fig,))) == []
    page = previews.figure_html(bench, fig, work_dir=tmp_path)
    assert '<div class="own">hello</div>' in page and ".own { color: red; }" in page and "$" not in page
    page = previews.figure_html(bench, previews.Figure("preview", "timeline", "val", fig.rows, {}), work_dir=tmp_path)
    assert "own event" in page  # the folder's intervals hook
    (folder / "preview.py").write_text(_PREVIEW_PY.replace("KIND", "audio"), encoding="utf-8")
    bench = Benchmark.load(str(folder))
    with pytest.raises(previews.SpecError, match="core kind"):
        previews.components(bench)
    assert "core kind" in previews.validate_spec(bench, previews.PreviewSpec(1, (fig,)))[0]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI")
def test_motion_writes_gif_and_mp4(folders, tmp_path, monkeypatch):
    from PIL import Image

    from yavih.media.ffmpeg import ffprobe_video

    def frame(html_path, png_path, *, size=(1600, 500), scale=2, chrome=None, compress=True):
        Image.new("RGBA", (size[0] * scale, size[1] * scale), (0, 0, 0, 0)).save(png_path)  # Chrome's transparent frame
        return png_path

    folder = _copy(folders["av"], tmp_path)
    _spec(folder, [{"id": "clip", "kind": "motion", "split": "val", "rows": _rows(folder, 1), "options": {"seconds": 2.0}}])
    monkeypatch.setattr(previews, "render_html", frame)
    out = previews.render_folder(str(folder))
    assert sorted(p.name for p in out) == ["clip.gif", "clip.mp4"]
    info = ffprobe_video(str(folder / "assets" / "clip.mp4"))
    assert (info["width"], info["height"]) == (1280, 800) and info["duration"] == pytest.approx(2.0, abs=0.2)
    with Image.open(folder / "assets" / "clip.gif") as gif:
        assert gif.size == previews.MOTION_SIZE and gif.n_frames >= 15
    assert _codes(folder)[1] == []
