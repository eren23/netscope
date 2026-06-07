"""The MCP request dispatcher + tools (transport-agnostic; stdio in __main__).

Speaks JSON-RPC 2.0. `Server.handle(request_dict) -> response_dict` is pure and
fully testable; the stdio framing lives in __main__. Tools return the MCP
`{"content": [{"type": "text", "text": ...}], "isError": bool}` shape.
"""
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from netscope.core.ir import NVGraph

PROTOCOL_VERSION = "2024-11-05"

# --- tool schemas (advertised via tools/list) --------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "trace_file",
        "description": (
            "Analyze a Python file and return netscope's graph of the model it "
            "defines. mode='static' (default) parses WITHOUT running — structure + "
            "declared-dim wiring clashes. mode='run' executes the file with tracing "
            "for REAL tensor shapes (only for runnable scripts). Returns the IR "
            "graph JSON (nodes, edges, warnings)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "absolute path to the .py file"},
                "mode": {"type": "string", "enum": ["static", "run"], "default": "static"},
            },
            "required": ["file"],
        },
    },
    {
        "name": "query_node",
        "description": (
            "Return the REAL captured data for one node in a saved netscope trace: "
            "its tensor in/out shapes, dtype, device, params, upstream/downstream "
            "neighbours, and any shape mismatch touching it. Answers 'what actually "
            "flows into <layer>?'. `node` is a node id OR a qualified name "
            "(e.g. model.layers.2)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string", "description": "path to a saved trace JSON"},
                "node": {"type": "string", "description": "node id or qualname"},
            },
            "required": ["graph", "node"],
        },
    },
    {
        "name": "list_mismatches",
        "description": (
            "List the shape/rank wiring mismatches netscope found in a saved trace, "
            "as structured data (kind, detail, the producer/consumer, and the source "
            "loc to jump to). Lets an agent find + fix wiring bugs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string", "description": "path to a saved trace JSON"},
            },
            "required": ["graph"],
        },
    },
    {
        "name": "explain_node",
        "description": (
            "Get a netscope-GROUNDED explanation of a node (using the configured LLM "
            "if a key is set). question: explain | why_warn | suggest_fix. The answer "
            "is grounded in the node's real shapes + source, never invented."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string"},
                "node": {"type": "string", "description": "node id or qualname"},
                "question": {"type": "string",
                             "enum": ["explain", "why_warn", "suggest_fix"],
                             "default": "explain"},
            },
            "required": ["graph", "node"],
        },
    },
]


# --- helpers -----------------------------------------------------------------
def _load_graph(path: str) -> NVGraph:
    data = json.load(open(path, encoding="utf-8"))
    g = NVGraph(name=data.get("name", ""))
    for n in data.get("nodes", []):
        g.add_node(n["id"], kind=n["kind"], name=n["name"], parent=n.get("parent"),
                   source=n.get("source", "runtime"), loc=n.get("loc"),
                   meta=n.get("meta"), attrs=n.get("attrs"))
    for e in data.get("edges", []):
        g.add_edge(e["src"], e["dst"], kind=e["kind"],
                   tensor_meta=e.get("tensor_meta"), source=e.get("source", "runtime"),
                   condition=e.get("condition"))
    return g


def _find_node(graph: NVGraph, ref: str) -> Optional[dict]:
    """Resolve a node by id, then by qualname, then by name."""
    for n in graph.nodes():
        if n["id"] == ref:
            return n
    for n in graph.nodes():
        if (n.get("meta") or {}).get("qualname") == ref:
            return n
    for n in graph.nodes():
        if n["name"] == ref:
            return n
    return None


def _label(n: dict) -> str:
    return (n.get("meta") or {}).get("qualname") or n["name"]


def _text(payload: Any, is_error: bool = False) -> Dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


# --- tool implementations ----------------------------------------------------
def _tool_trace_file(args: dict) -> Dict[str, Any]:
    path = args.get("file")
    mode = args.get("mode", "static")
    if mode not in ("static", "run"):
        return _text(f"unknown mode {mode!r} (expected 'static' or 'run')", is_error=True)
    if not path or not os.path.exists(path):
        return _text(f"file not found: {path!r}", is_error=True)
    try:
        if mode == "run":
            import tempfile
            fd, out = tempfile.mkstemp(suffix=".json")   # mkstemp, not the race-prone mktemp
            os.close(fd)
            try:
                env = dict(os.environ, NETSCOPE_OUT=out)
                r = subprocess.run([sys.executable, path], env=env,
                                   capture_output=True, text=True, timeout=300)
                if not os.path.getsize(out):
                    # surface the REAL failure instead of the generic "no graph".
                    tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-12:])
                    if r.returncode != 0:
                        return _text(f"the script exited with code {r.returncode}:\n"
                                     f"{tail or '(no output)'}", is_error=True)
                    return _text("the script ran but produced no graph — wrap a forward "
                                 "in `with netscope.graph(\"name\"):`\n" + tail, is_error=True)
                return _text(json.load(open(out)))
            finally:
                try:
                    os.unlink(out)
                except OSError:
                    pass
        from netscope.static.ast_producer import analyze_file
        return _text(analyze_file(path).to_dict())
    except Exception as e:  # never crash the server on a bad file
        return _text(f"trace failed: {e}", is_error=True)


def _tool_query_node(args: dict) -> Dict[str, Any]:
    try:
        g = _load_graph(args["graph"])
    except Exception as e:
        return _text(f"could not load graph: {e}", is_error=True)
    node = _find_node(g, args.get("node", ""))
    if node is None:
        return _text(f"no node {args.get('node')!r} in the graph", is_error=True)

    by_id = {n["id"]: n for n in g.nodes()}
    up, down = [], []
    for edge in g.edges():
        if edge["kind"] != "dataflow":
            continue
        if edge["dst"] == node["id"] and edge["src"] in by_id:
            up.append(by_id[edge["src"]])
        if edge["src"] == node["id"] and edge["dst"] in by_id:
            down.append(by_id[edge["dst"]])
    warns = [w for w in g.to_dict()["warnings"]
             if w.get("src") == node["id"] or w.get("dst") == node["id"]]
    return _text({
        "id": node["id"], "name": node["name"], "kind": node["kind"],
        "qualname": (node.get("meta") or {}).get("qualname"),
        "loc": node.get("loc"), "meta": node.get("meta") or {},
        "upstream": [{"qualname": (n.get("meta") or {}).get("qualname"), "name": n["name"]} for n in up],
        "downstream": [{"qualname": (n.get("meta") or {}).get("qualname"), "name": n["name"]} for n in down],
        "warnings": warns,
    })


def _tool_list_mismatches(args: dict) -> Dict[str, Any]:
    try:
        g = _load_graph(args["graph"])
    except Exception as e:
        return _text(f"could not load graph: {e}", is_error=True)
    by_id = {n["id"]: n for n in g.nodes()}
    out = []
    for w in g.to_dict()["warnings"]:
        dst = by_id.get(w.get("dst"))
        out.append({
            "kind": w.get("kind"), "detail": w.get("detail"),
            "severity": w.get("severity"),
            "producer": _label(by_id[w["src"]]) if w.get("src") in by_id else w.get("src"),
            "consumer": _label(dst) if dst else w.get("dst"),
            "loc": (dst or {}).get("loc"),
        })
    return _text({"count": len(out), "mismatches": out})


def _tool_explain_node(args: dict) -> Dict[str, Any]:
    try:
        g = _load_graph(args["graph"])
    except Exception as e:
        return _text(f"could not load graph: {e}", is_error=True)
    node = _find_node(g, args.get("node", ""))
    if node is None:
        return _text(f"no node {args.get('node')!r} in the graph", is_error=True)
    from netscope.llm import available, explain, LLMUnavailable
    if not available():
        return _text("no LLM key configured — set NETSCOPE_LLM_API_KEY (or "
                     "OPENROUTER_API_KEY / OPENAI_API_KEY) to use explain_node.",
                     is_error=True)
    try:
        # the graph here came from a caller-supplied trace JSON (untrusted) — restrict
        # source-line reads to the project dir so a crafted loc.file can't be slurped.
        answer = explain(g, node["id"], question=args.get("question", "explain"),
                         source_root=os.getcwd())
        return _text(answer)
    except LLMUnavailable as e:
        return _text(str(e), is_error=True)
    except Exception as e:
        return _text(f"explain failed: {e}", is_error=True)


_DISPATCH = {
    "trace_file": _tool_trace_file,
    "query_node": _tool_query_node,
    "list_mismatches": _tool_list_mismatches,
    "explain_node": _tool_explain_node,
}


class Server:
    """JSON-RPC 2.0 dispatcher for the netscope MCP tools."""

    def handle(self, req: dict) -> Optional[dict]:
        rid = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}

        def ok(result):
            return {"jsonrpc": "2.0", "id": rid, "result": result}

        def err(code, message):
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

        if method == "initialize":
            return ok({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "netscope", "version": "0.1"},
            })
        if method in ("notifications/initialized", "initialized"):
            return None   # notification: no response
        if method == "tools/list":
            return ok({"tools": TOOLS})
        if method == "tools/call":
            name = params.get("name")
            fn = _DISPATCH.get(name) if isinstance(name, str) else None
            if fn is None:
                return err(-32602, f"unknown tool: {name}")
            try:
                return ok(fn(params.get("arguments") or {}))
            except Exception as e:
                return ok(_text(f"tool error: {e}", is_error=True))
        return err(-32601, f"method not found: {method}")
