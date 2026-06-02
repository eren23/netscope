"""Demo — drive the netscope MCP server in-process (what a coding agent does).

The MCP server exposes the live model graph to agents (Cursor / Claude Code) over
stdio. Here we call its dispatcher directly to show the four tools. To wire it into
a real agent, see the 'MCP server' section of the README — point the agent at:

    python -m netscope.mcp

    python examples/mcp_server_demo.py
"""
from __future__ import annotations

import json
import os
import tempfile

from netscope.mcp.server import Server


def main() -> None:
    # a model file with a wiring clash (512 out -> 256 in)
    src = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.backbone = nn.Linear(784, 512)\n"
        "        self.classifier = nn.Linear(256, 10)\n"
        "    def forward(self, x):\n"
        "        return self.classifier(self.backbone(x))\n"
    )
    f = tempfile.mktemp(suffix=".py")
    open(f, "w").write(src)

    srv = Server()

    def call(name, args):
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args}})
        return r["result"]

    print("tools:", [t["name"] for t in srv.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]])

    # 1) an agent traces the file (no execution)
    res = call("trace_file", {"file": f, "mode": "static"})
    graph = json.loads(res["content"][0]["text"])
    print(f"\ntrace_file: {len(graph['nodes'])} nodes, {len(graph['warnings'])} warning(s)")

    # save the graph so the other tools can read it
    gp = tempfile.mktemp(suffix=".json")
    open(gp, "w").write(json.dumps(graph))

    # 2) an agent lists the mismatches it should fix
    res = call("list_mismatches", {"graph": gp})
    mm = json.loads(res["content"][0]["text"])
    print(f"\nlist_mismatches: {mm['count']} clash(es)")
    for m in mm["mismatches"]:
        print(f"  {m['producer']} -> {m['consumer']}: {m['detail']}"
              f"  (line {(m.get('loc') or {}).get('line')})")

    os.remove(f); os.remove(gp)


if __name__ == "__main__":
    main()
