"""Generation-step timeline — read an autoregressive trace as an ordered sequence.

Each ``with netscope.step():`` (see hints.api.step) leaves a stage node tagged
``attrs.step``. This aggregates those into a timeline: per step, its wall-time
(under ``profile=True``), how many modules ran, and the step's output shape — so
the sequence length growing across decode steps (the autoregressive signature) is
right there in the numbers.
"""
from __future__ import annotations


def _descendant_ids(graph, root_id) -> list:
    """All node ids under ``root_id`` (by the ``parent`` field), depth-first."""
    out, stack = [], list(graph.children(root_id))
    while stack:
        nid = stack.pop()
        out.append(nid)
        stack.extend(graph.children(nid))
    return out


def timeline(graph) -> list:
    """Ordered per-step summary of a generation trace.

    Returns a list of ``{step, label, time_ms, modules, out_shape}`` sorted by step
    index; empty if the trace has no ``netscope.step()`` markers. ``out_shape`` is
    the step's final sub-call output (watch its sequence axis grow per decode step).
    """
    nodes = {n["id"]: n for n in graph.nodes()}
    steps = []
    for n in graph.nodes():
        attrs = n.get("attrs") or {}
        if "step" not in attrs:
            continue
        # the step's output = the last direct child carrying an out_shape (the
        # model call inside the step); module count is over the whole subtree.
        out_shape = None
        for cid in graph.children(n["id"]):
            os_ = ((nodes.get(cid) or {}).get("meta") or {}).get("out_shape")
            if os_:
                out_shape = os_
        modules = sum(
            1 for did in _descendant_ids(graph, n["id"])
            if (nodes.get(did) or {}).get("kind") in ("module", "op", "model")
        )
        steps.append({
            "step": attrs["step"],
            "label": n.get("name"),
            "time_ms": (n.get("meta") or {}).get("time_ms"),
            "modules": modules,
            "out_shape": out_shape,
        })
    steps.sort(key=lambda s: s["step"])
    return steps
