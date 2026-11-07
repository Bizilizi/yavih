"""The README "at a glance" block (``yavih.doctables``): rendered from the folder's yaml, replaced in place."""

from __future__ import annotations

import pytest

from yavih import doctables
from yavih.benchmarks import packaged_folder

ESC50 = packaged_folder("esc50")


def test_render_glance_of_esc50():
    block = doctables.render_glance(ESC50)
    lines = block.splitlines()
    assert lines[0] == doctables.GLANCE_BEGIN.format(name="esc50") and lines[-1] == doctables.GLANCE_END
    rows = [ln.split(" | ")[0].strip("| *") for ln in lines[3:-1]]
    assert rows == ["Task", "Modality", "Size", "Headline", "Data", "Licence", "Protocol"]  # one reading: no Readings row
    assert lines[1] == "| At a glance | ESC-50 |"  # the header row names the benchmark (header.name)
    assert "Readings" in doctables.render_glance(packaged_folder("ave"))  # several readings: the row appears


def test_replace_glance_inserts_replaces_and_refuses_an_open_block():
    text = "![ESC-50](assets/header.png)\n\nlinks · [setup.md](setup.md)\n\n## The task\n\nbody\n"
    once = doctables.replace_glance(text, ESC50)
    assert doctables.glance_block(once) == doctables.render_glance(ESC50)
    assert once.index(doctables.GLANCE_END) < once.index("## The task")
    assert doctables.replace_glance(once, ESC50) == once  # idempotent
    stale = once.replace("2 000", "1 999")
    assert doctables.replace_glance(stale, ESC50) == once
    with pytest.raises(ValueError, match="has no"):
        doctables.replace_glance(once.replace(doctables.GLANCE_END, ""), ESC50)


def test_glance_cli(capsys):
    assert doctables.main(["--glance", "esc50"]) == 0
    assert capsys.readouterr().out.strip() == doctables.render_glance(ESC50)
