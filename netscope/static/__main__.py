"""Enable `python -m netscope.static <file.py>`."""
from __future__ import annotations

from netscope.static.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
