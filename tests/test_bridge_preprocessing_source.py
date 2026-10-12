"""``W_NO_PREPROCESSING_SOURCE``: which docstring layouts count as a filled ``Preprocessing source:`` slot.

The usual layout (heading, blank line, indented pasted code) is filled; the heading's own prose on the next line
("repository, commit and file:line ...") and a ``<paste ...>`` placeholder are not.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

import yavih
from yavih.bridge import Bridge, preprocessing_source, preprocessing_source_check

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "src" / "yavih" / "benchmarks").glob("*/bridge_template.py"))


def _bridge_from_docs(cls_doc: str | None, mod_doc: str | None, name: str = "PreprocProbe") -> type:
    """A Bridge subclass whose class docstring is ``cls_doc``, defined in a fresh module whose docstring is
    ``mod_doc`` (``preprocessing_source`` looks the module up in ``sys.modules``)."""
    modname = f"_yavih_preproc_test_{name}_{abs(hash((cls_doc, mod_doc)))}"
    mod = types.ModuleType(modname, mod_doc)
    sys.modules[modname] = mod
    cls = type(name, (Bridge,), {"__doc__": cls_doc, "__module__": modname})
    setattr(mod, name, cls)
    return cls


def _load_template(path: Path) -> tuple[types.ModuleType, list[type]]:
    name = f"_yavih_tmpl_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    classes = [v for v in vars(mod).values()
               if isinstance(v, type) and issubclass(v, yavih.Bridge) and v is not yavih.Bridge and v.__module__ == name]
    return mod, classes


# ----------------------------------------------------------------------------- the usual layout passes

EXAMPLE_MODULE_DOC = '''Bridge: example ViT-B -> esc50 (a).

The model loader lives in example_common.py.

Preprocessing source (pasted from example_common.py, matched by describe()["input"]):

    preprocessing source (repo org/model@0123abc):

        # audio_frontend.py  decode + log_mel
        wav = resample(wav, sr, 16000)
        mel = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=128, frame_shift=10)
'''

EXAMPLE_CLASS_DOC = '''Example ViT-B, esc50 / a.

    Preprocessing source:

        preprocessing source (repo org/model@0123abc):

            # audio_frontend.py  decode + log_mel
            wav = resample(wav, sr, 16000)
    '''


def test_heading_blank_line_indented_block_is_filled():
    cls = _bridge_from_docs(EXAMPLE_CLASS_DOC, EXAMPLE_MODULE_DOC, "ExampleBridge")
    text, why = preprocessing_source_check(cls)
    assert text is not None and "log_mel" in text and "resample" in text
    assert "ExampleBridge.__doc__" in why
    # the module docstring alone (parenthetical heading prose, blank line, nested indentation) is filled too
    only_module = _bridge_from_docs("A bridge.", EXAMPLE_MODULE_DOC, "ExampleBridgeModuleOnly")
    assert "kaldi.fbank" in (preprocessing_source(only_module) or "")


# ----------------------------------------------------------------------------- unfilled slots warn


@pytest.mark.parametrize("path", TEMPLATES, ids=[p.parent.name for p in TEMPLATES])
def test_every_unfilled_bridge_template_warns(path: Path):
    _, classes = _load_template(path)
    assert classes, f"{path}: no Bridge subclass"
    for cls in classes:
        text, why = preprocessing_source_check(cls)
        assert text is None, f"{path.parent.name}: the unfilled template slot reads as filled: {text!r}"
        assert "Preprocessing source" in why and "indented block" in why
        assert "template text" in why, why  # the template HAS a slot; the message says it is unfilled


@pytest.mark.parametrize("path", TEMPLATES, ids=[p.parent.name for p in TEMPLATES])
def test_filling_each_template_slot_as_instructed_passes(path: Path):
    """Replacing the TODO lines of a template's slot with pasted code is enough to satisfy the check."""
    mod, classes = _load_template(path)
    doc = mod.__doc__ or ""
    filled = []
    for line in doc.splitlines():
        if re.match(r"^\s*preprocessing\s+source\s*:\s*TODO\b", line, re.IGNORECASE):
            line = "Preprocessing source: org/model@0123abc src/transforms.py::eval_transform"
        elif re.match(r"^\s*TODO\b", line):
            indent = line[: len(line) - len(line.lstrip())]
            line = indent + "x = torchvision.transforms.Resize(256)(x)"
        filled.append(line)
    cls = _bridge_from_docs(classes[0].__doc__, "\n".join(filled), f"Filled_{path.parent.name}")
    assert preprocessing_source(cls) is not None, (path.parent.name, preprocessing_source_check(cls)[1])


def _readme_bridge_docstring() -> str:
    text = (ROOT / "README.md").read_text()
    block = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    assert block is not None
    tree = ast.parse(block.group(1))
    doc = next(ast.get_docstring(n, clean=False) for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Bridge")
    assert doc is not None
    return doc


def test_readme_example_slot_warns_until_filled():
    doc = _readme_bridge_docstring()
    assert "Preprocessing source" in doc
    assert preprocessing_source(_bridge_from_docs(doc, None, "ReadmeBridge")) is None
    # filled the way the README says (replace the TODO line(s) by the pasted source): passes
    filled = "\n".join("        torchvision.transforms.Resize(256)" if ln.strip().startswith("TODO") else ln for ln in doc.splitlines())
    assert preprocessing_source(_bridge_from_docs(filled, None, "ReadmeBridgeFilled")) is not None


@pytest.mark.parametrize(
    "doc",
    [
        # a `<paste …>` placeholder
        """Preprocessing source: <paste the model's own eval transform here, matched by describe() line by line>""",
        # heading prose continues on the next line (with a colon in it), body TODO
        """Bridge.

        Preprocessing source of the model (paste the model's own audio front end VERBATIM here, with the
        repository, commit and file:line it came from; describe()["input"]["audio"] must match it line by line):

            TODO (e.g. `torchaudio.compliance.kaldi.fbank(wav16k, htk_compat=True)`, then the
            model's PRETRAINING-DATASET mean/std)
        """,
        # heading prose only, no body at all
        """Preprocessing source of the model (paste it here, with the
        repository, commit and file:line it came from):
        """,
        # a heading followed by an unrelated section after a blank line (not indented: not the body)
        """Preprocessing source:

        What encode receives per Sample: s.streams["video"].
        """,
        # partially filled: a TODO line left in
        """Preprocessing source:
            Resize(256) -> CenterCrop(224)
            TODO normalization constants
        """,
        # placeholders only, over several lines
        """Preprocessing source:
            <repo>@<commit> <file>::<function>
            <the source lines,
             verbatim>
        """,
        "Preprocessing source: TODO",
        "Preprocessing source:",
    ],
)
def test_unfilled_slot_layouts_warn(doc: str):
    cls = _bridge_from_docs(doc, None)
    text, why = preprocessing_source_check(cls)
    assert text is None, text
    assert "PreprocProbe.__doc__" in why


@pytest.mark.parametrize(
    ("doc", "needle"),
    [
        ("Preprocessing source: facebookresearch/vjepa2@abc src/datasets/utils/video/transforms.py", "vjepa2"),
        # inline head + deeper continuation (tests/test_extract_fixtures.py NumpyBridge)
        ("""A bridge.

        Preprocessing source: example/model@abc123 transforms.py::eval_transform
            Resize(224) -> CenterCrop(224) -> x / 255
        """, "CenterCrop"),
        # a paragraph layout (same indentation, no blank line) counts
        ("""A bridge.

        Preprocessing source:
        torchvision Resize(256) -> CenterCrop(224), timm IMAGENET_DEFAULT_MEAN/STD
        """, "CenterCrop"),
        # a multi-line heading that ends with ":", then the indented block
        ("""PREPROCESSING SOURCE - paste the model's own EVAL pipeline here, verbatim, with repo, file, function and
        commit (bridge check warns W_NO_PREPROCESSING_SOURCE while this slot is empty):
            facebookresearch/AudioMAE@e3b0c44 dataset.py:120-160
            fbank = kaldi.fbank(wav, num_mel_bins=128)
        """, "AudioMAE"),
        # a URL in the heading line is not an inline body separator
        ("""Preprocessing source (see https://github.com/org/repo):
            transforms.Resize(256)
        """, "Resize"),
        # pasted code that contains a TODO comment is still pasted code
        ("""Preprocessing source:
            x = resize(x, 256)  # TODO(upstream): antialias
        """, "resize"),
    ],
)
def test_filled_slot_layouts_pass(doc: str, needle: str):
    assert needle in (preprocessing_source(_bridge_from_docs(doc, None)) or "")


def test_first_filled_slot_wins_class_then_module():
    cls = _bridge_from_docs("Preprocessing source: TODO", "Preprocessing source:\n    org/model@1 eval.py\n", "Mixed")
    text, why = preprocessing_source_check(cls)
    assert text == "org/model@1 eval.py"
    assert "module" in why


def test_explanation_names_docstrings_and_layout():
    cls = _bridge_from_docs("Just a bridge.", "A module without a slot.", "NoSlot")
    text, why = preprocessing_source_check(cls)
    assert text is None
    assert "NoSlot.__doc__: no line starts with 'Preprocessing source'" in why
    assert "module _yavih_preproc_test_NoSlot" in why
    assert "indented block" in why and "TODO" in why

    class Bare(Bridge):
        pass

    assert preprocessing_source(Bare) is None


def test_package_fixtures_keep_their_verdicts():
    """The yavih.testing synthetic template stays unfilled; the shipped hf_reference example is filled."""
    from yavih import testing
    from yavih.bridges.examples import hf_reference

    assert preprocessing_source(hf_reference.HFReferenceBridge) is not None
    tmpl = testing._BRIDGE_TEMPLATE_PY.format(name="synth", streams=("video",))
    tree = ast.parse(tmpl)
    assert preprocessing_source(_bridge_from_docs(None, ast.get_docstring(tree, clean=False), "SynthTemplate")) is None
