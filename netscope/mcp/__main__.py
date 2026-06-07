"""stdio transport for the netscope MCP server: `python -m netscope.mcp`.

Reads newline-delimited JSON-RPC requests from stdin, writes responses to stdout
(one JSON object per line). Point Cursor / Claude Code at this command as an MCP
server. Stdlib only.
"""
import json
import sys

from netscope.mcp.server import Server


def serve(stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = Server()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            # malformed frame -> a JSON-RPC parse error, best effort.
            stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }) + "\n")
            stdout.flush()
            continue
        resp = server.handle(req)
        if resp is not None:                      # notifications get no response
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
    return 0


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
