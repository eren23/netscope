"""Infer dataflow edges between sibling stage nodes from hint semantics.

Runtime tensor-identity (see instrument/torch_nn.py) links modules whose tensors
literally flow. But across hinted stages the data is often rebuilt (a new tensor,
a python list of answers, a vote count), so identity can't see the connection.
This post-pass recovers the *intended* flow from the `branch` / `reduce` hints:

  plain stage      -> chains from the current frontier
  run of branches  -> fan OUT from the frontier, become the new frontier
  (a reduce is just a plain stage; the branch frontier fans IN to it)

Runs per parent group at session finalize, so nested pipelines stay isolated.
Pure-torch graphs (no `stage` nodes) are a no-op.
"""

def _order(node_id: str) -> int:
    tail = node_id.rsplit("#", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def infer_stage_flow(graph) -> None:
    by_parent: dict = {}
    for n in graph.nodes():
        if n["kind"] == "stage":
            by_parent.setdefault(n["parent"], []).append(n)

    for stages in by_parent.values():
        stages.sort(key=lambda n: _order(n["id"]))
        frontier: list = []
        i = 0
        while i < len(stages):
            if stages[i]["attrs"].get("branch"):
                run = []
                while i < len(stages) and stages[i]["attrs"].get("branch"):
                    run.append(stages[i])
                    i += 1
                for f in frontier:
                    for b in run:
                        graph.add_edge(f["id"], b["id"], kind="dataflow", source="hint")
                frontier = run
            else:
                for f in frontier:
                    graph.add_edge(f["id"], stages[i]["id"], kind="dataflow", source="hint")
                frontier = [stages[i]]
                i += 1
