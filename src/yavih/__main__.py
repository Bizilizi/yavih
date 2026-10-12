"""``python -m yavih …`` = the ``yavih`` command (for an env whose scripts directory is not on PATH)."""

import sys

from yavih.cli import main

if __name__ == "__main__":
    sys.exit(main())
