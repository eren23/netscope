"""`netscope <cmd>` / `python -m netscope <cmd>` — a thin dispatcher over the tools.

    netscope static   model.py            # static graph (declared-dim checks, no run)
    netscope playground [port]            # local live editor <-> graph
    netscope mcp                          # MCP server (JSON-RPC over stdio) for agents
    netscope diff     a.json b.json [...]  # diff two saved traces
    netscope views    graph.json "prompt"  # a prompt -> a view spec

Each subcommand delegates to that tool's own entry point, so `python -m
netscope.static …` etc. keep working unchanged.
"""
from __future__ import annotations

import sys

_USAGE = "usage: netscope {static|playground|mcp|diff|views} ...\n"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "static":
        from netscope.static.cli import main as run
        return run(rest)
    if cmd == "playground":
        from netscope.playground import main as run
        return run(rest)
    if cmd == "mcp":
        from netscope.mcp.__main__ import main as run
        return run()
    if cmd == "diff":
        from netscope.core.diff import _main as run
        return run(rest)
    if cmd == "views":
        from netscope.llm.views import _main as run
        return run(rest)
    sys.stderr.write(_USAGE)
    return 0 if cmd in ("", "-h", "--help", "help") else 2


if __name__ == "__main__":
    sys.exit(main())
