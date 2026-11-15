"""av_syncbench, step 3: re-derive the frozen manifest from the extracted metadata (sha256-pinned) and compare it byte for
byte with the committed one; also checks that every row's two files are present. Usage:
yavih bench prepare av_syncbench 03   (``-- --media-root DIR`` overrides the configured media_root)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

folder = Path(os.environ.get("FOLDER") or Path(__file__).resolve().parents[1])
spec = importlib.util.spec_from_file_location("av_syncbench_make_manifest", folder / "make_manifest.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
sys.exit(mm.main(["--check", *sys.argv[1:]]))
