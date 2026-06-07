"""netscope MCP server — expose the live model graph to coding agents.

A stdlib-only Model Context Protocol server (JSON-RPC 2.0 over stdio, no SDK
dependency). Point Cursor / Claude Code at `python -m netscope.mcp` and the agent
can ground itself in REAL netscope data instead of guessing: trace a file, query a
node's actual tensor shapes + dataflow, list wiring mismatches, and get a grounded
explanation. See netscope/mcp/server.py for the tools.
"""
from netscope.mcp.server import Server

__all__ = ["Server"]
