"""Regression tests for the audit fixes (static / security / MCP batch).

The engine-level audit fixes live with their own suites (test_checks, test_diff,
test_session_safety); this file covers the static-analysis, prompt-safety, and MCP
findings.
"""
from __future__ import annotations

from netscope.core.checks import detect_mismatches
from netscope.static.ast_producer import analyze_source

_CNN_HEAD = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.net = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(),\n"
    "                                 nn.Flatten(), nn.Linear(400, 10))\n"
    "    def forward(self, x):\n"
    "        return self.net(x)\n"
)


def test_cnn_sequential_head_is_not_false_flagged():
    # Conv -> (ReLU, Flatten) -> Linear: the static pass must NOT wire Conv->Linear
    # across the implicit flatten and compare channels vs flattened features.
    g = analyze_source(_CNN_HEAD, "net.py")
    assert detect_mismatches(g) == []


def test_source_block_root_blocks_paths_outside_it(tmp_path):
    from netscope.llm.prompts import _source_block

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET\napi_key = sk-deadbeef\n")
    node = {"loc": {"file": str(secret), "line": 1}}

    # with a root that doesn't contain the file (the MCP untrusted path), no read
    assert _source_block(node, root=str(tmp_path / "elsewhere")) is None
    # under its own root, it's read (legit project source)
    assert _source_block(node, root=str(tmp_path)) is not None
    # unrestricted (the in-process library path) reads it — trusted there
    assert _source_block(node) is not None


def test_mcp_trace_file_rejects_unknown_mode():
    from netscope.mcp.server import _tool_trace_file

    r = _tool_trace_file({"file": "netscope/__init__.py", "mode": "bogus"})
    assert r.get("isError") is True


def test_cli_dispatcher_routes_known_and_rejects_unknown():
    import contextlib
    import io

    from netscope.__main__ import main

    assert main([]) == 0          # no args -> usage, clean exit
    assert main(["bogus"]) == 2   # unknown subcommand -> error code

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["static", "examples/mismatch_demo.py"])
    assert rc == 0 and '"nodes"' in buf.getvalue()   # routed to the static tool
