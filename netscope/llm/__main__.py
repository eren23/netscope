"""CLI: answer a question about a node in a saved graph JSON.

    python -m netscope.llm <graph.json> <node_id> [explain|why_warn|suggest_fix]

The extension shells out to this (same pattern as `python -m netscope.static` and
the NETSCOPE_OUT trace) and shows the answer in the node panel. Reads the graph
from a file so it works on any saved/streamed trace. Prints the answer to stdout;
a clear message + nonzero exit if no provider is configured.
"""
import json
import sys

from netscope.core.ir import NVGraph
from netscope.llm import LLMUnavailable, explain


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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: python -m netscope.llm <graph.json> <node_id> "
              "[explain|why_warn|suggest_fix]", file=sys.stderr)
        return 2
    graph_path, node_id = argv[0], argv[1]
    question = argv[2] if len(argv) > 2 else "explain"
    try:
        g = _load_graph(graph_path)
        sys.stdout.write(explain(g, node_id, question=question))
        return 0
    except LLMUnavailable as e:
        print(f"netscope.llm: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"netscope.llm error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
