"""Static-graph CLI: emit the AST-derived graph of a file as JSON.

The VSCode extension calls `python -m netscope.static <file.py>` whenever a file
changes, to draw the on-type "skeleton" graph (stage/branch/vote structure)
WITHOUT running the user's code. Output is the standard SCHEMA_VERSION IR JSON.
"""
from __future__ import annotations

import sys

from netscope.static.ast_producer import analyze_file


def render_static_json(path: str, indent: int = 2) -> str:
    return analyze_file(path).to_json(indent=indent)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m netscope.static <file.py>", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(render_static_json(argv[0]))
        return 0
    except Exception as e:  # surface parse/IO errors to the extension cleanly
        print(f"netscope.static error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
