"""MCP server — expose the live graph to coding agents (Cursor / Claude Code).

A stdlib-only JSON-RPC 2.0 server over stdio (no mcp SDK dependency). It lets an
agent ground itself in REAL netscope data: trace a file, query a node's actual
shapes/dataflow, list wiring mismatches, and (if an LLM key is set) get a grounded
explanation. Tests drive the dispatcher directly with JSON-RPC request dicts.
"""
from __future__ import annotations

import json

from netscope.mcp.server import Server, TOOLS


def _srv():
    return Server()


def _call(srv, method, params=None, rid=1):
    return srv.handle({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})


# ---- protocol basics -----------------------------------------------------
def test_initialize_returns_server_info():
    r = _call(_srv(), "initialize", {"protocolVersion": "2024-11-05"})
    assert r["result"]["serverInfo"]["name"] == "netscope"
    assert "capabilities" in r["result"]


def test_tools_list_exposes_the_four_tools():
    r = _call(_srv(), "tools/list")
    names = {t["name"] for t in r["result"]["tools"]}
    assert {"trace_file", "query_node", "list_mismatches", "explain_node"} <= names
    # each tool has a JSON-schema inputSchema
    for t in r["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"


def test_unknown_method_returns_jsonrpc_error():
    r = _call(_srv(), "no/such/method")
    assert r["error"]["code"] == -32601   # method not found


# ---- trace_file ----------------------------------------------------------
def test_trace_file_static_returns_graph(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(8, 16)\n"
        "        self.b = nn.Linear(16, 4)\n"
        "    def forward(self, x):\n"
        "        return self.b(self.a(x))\n"
    )
    srv = _srv()
    r = _call(srv, "tools/call", {"name": "trace_file",
                                  "arguments": {"file": str(f), "mode": "static"}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["schema_version"]
    names = [n["name"] for n in payload["nodes"]]
    assert any("Linear" in nm for nm in names)


def test_trace_file_flags_a_static_mismatch(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text(
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(64, 256)\n"
        "        self.b = nn.Linear(128, 10)\n"
        "    def forward(self, x):\n"
        "        h = self.a(x)\n"
        "        return self.b(h)\n"
    )
    r = _call(_srv(), "tools/call", {"name": "trace_file",
                                     "arguments": {"file": str(f), "mode": "static"}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert len(payload["warnings"]) == 1
    assert "256" in payload["warnings"][0]["detail"]


# ---- query_node ----------------------------------------------------------
def _graph_json(tmp_path):
    """A small saved trace with real shapes + a mismatch, for query/list/explain."""
    g = {
        "schema_version": "1", "name": "demo",
        "nodes": [
            {"id": "a", "kind": "module", "name": "Encoder", "parent": None,
             "source": "runtime", "loc": {"file": "model.py", "line": 5},
             "meta": {"out_shape": [1, 256], "qualname": "encoder",
                      "dtype": "float32", "device": "cpu"}, "attrs": {}},
            {"id": "b", "kind": "module", "name": "Head", "parent": None,
             "source": "runtime", "loc": {"file": "model.py", "line": 9},
             "meta": {"in_shape": [1, 128], "qualname": "head"}, "attrs": {}},
        ],
        "edges": [{"src": "a", "dst": "b", "kind": "dataflow", "source": "runtime"}],
        "warnings": [{"src": "a", "dst": "b", "kind": "shape_mismatch",
                      "detail": "Encoder emits dim 256 but Head expects 128 (axis 1)",
                      "severity": "error"}],
    }
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(g))
    return str(p)


def test_query_node_by_qualname_returns_real_data(tmp_path):
    gp = _graph_json(tmp_path)
    r = _call(_srv(), "tools/call", {"name": "query_node",
              "arguments": {"graph": gp, "node": "encoder"}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["name"] == "Encoder"
    assert payload["meta"]["out_shape"] == [1, 256]
    assert payload["meta"]["dtype"] == "float32"
    # downstream neighbour is reported
    assert any("head" in (d.get("qualname") or d.get("name", "")).lower()
               for d in payload["downstream"])


def test_query_node_reports_its_mismatch(tmp_path):
    gp = _graph_json(tmp_path)
    r = _call(_srv(), "tools/call", {"name": "query_node",
              "arguments": {"graph": gp, "node": "head"}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["warnings"], "head's mismatch should be reported"
    assert "256" in payload["warnings"][0]["detail"]


def test_query_node_unknown_is_an_error(tmp_path):
    gp = _graph_json(tmp_path)
    r = _call(_srv(), "tools/call", {"name": "query_node",
              "arguments": {"graph": gp, "node": "nope"}})
    # tool errors are returned as isError content, not a JSON-RPC error
    assert r["result"]["isError"] is True


# ---- list_mismatches -----------------------------------------------------
def test_list_mismatches_returns_structured_clashes(tmp_path):
    gp = _graph_json(tmp_path)
    r = _call(_srv(), "tools/call", {"name": "list_mismatches",
              "arguments": {"graph": gp}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["count"] == 1
    assert payload["mismatches"][0]["kind"] == "shape_mismatch"
    # carries the loc so an agent can jump to the offending line
    assert payload["mismatches"][0]["loc"]["line"] == 9


# ---- explain_node (gated) ------------------------------------------------
def test_explain_node_gated_without_key(tmp_path, monkeypatch):
    for k in ("NETSCOPE_LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    gp = _graph_json(tmp_path)
    r = _call(_srv(), "tools/call", {"name": "explain_node",
              "arguments": {"graph": gp, "node": "head", "question": "why_warn"}})
    # no key -> a clean isError telling the agent to set one (not a crash)
    assert r["result"]["isError"] is True
    assert "key" in r["result"]["content"][0]["text"].lower()


# ---- stdio transport -----------------------------------------------------
def test_stdio_serve_roundtrips_jsonrpc_lines():
    import io
    from netscope.mcp.__main__ import serve
    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),  # no reply
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]) + "\n"
    out = io.StringIO()
    serve(stdin=io.StringIO(requests), stdout=out)
    lines = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    # 2 responses (the notification produced none)
    assert len(lines) == 2
    assert lines[0]["result"]["serverInfo"]["name"] == "netscope"
    assert any(t["name"] == "trace_file" for t in lines[1]["result"]["tools"])


def test_stdio_handles_malformed_line():
    import io
    from netscope.mcp.__main__ import serve
    out = io.StringIO()
    serve(stdin=io.StringIO("{not json\n"), stdout=out)
    resp = json.loads(out.getvalue().splitlines()[0])
    assert resp["error"]["code"] == -32700
