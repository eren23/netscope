"""Fuse a runtime graph with a static graph by source location.

This is the one integration point between producers. A runtime node and a static
node that share the same ``loc`` describe the same code, seen two ways: the
runtime node knows real shapes/dtypes (the executed path); the static node knows
declared structure (names, branch counts, votes — including code that never ran).
The fused node keeps the runtime ``meta`` and gains the static ``attrs``.

Nodes present on only one side are carried through unchanged, so static-only
structure (e.g. a vote stage the tracer can't see) still appears.
"""
from __future__ import annotations

from typing import Optional

from netscope.core.ir import NVGraph


def _loc_key(node: dict) -> Optional[tuple]:
    loc = node.get("loc")
    if not loc:
        return None
    return (loc.get("file"), loc.get("line"))


def merge(runtime: NVGraph, static: NVGraph) -> NVGraph:
    fused = NVGraph(name=runtime.name or static.name)

    static_by_loc = {}
    for n in static.nodes():
        key = _loc_key(n)
        if key is not None:
            static_by_loc[key] = n

    matched_static_ids = set()

    # 1) runtime nodes, fused with any static node sharing their loc. A static
    #    node fuses into AT MOST ONE runtime node (first match wins) — two runtime
    #    nodes can legitimately share a loc (a submodule called twice, a loop
    #    body), and duplicating the static node's attrs across both is wrong.
    for rt in runtime.nodes():
        key = _loc_key(rt)
        st = static_by_loc.get(key) if key is not None else None
        if st is not None and st["id"] in matched_static_ids:
            st = None   # already fused into an earlier runtime node at this loc
        attrs = dict(rt.get("attrs") or {})
        source = rt["source"]
        if st is not None:
            attrs.update(st.get("attrs") or {})
            source = "fused"
            matched_static_ids.add(st["id"])
        fused.add_node(
            rt["id"], kind=rt["kind"], name=rt["name"], parent=rt.get("parent"),
            source=source, loc=rt.get("loc"), meta=rt.get("meta"), attrs=attrs,
        )

    # 2) runtime edges
    for e in runtime.edges():
        fused.add_edge(e["src"], e["dst"], kind=e["kind"],
                       tensor_meta=e.get("tensor_meta"), source=e["source"],
                       condition=e.get("condition"))

    # Does the runtime trace already have branch / reduce stages? (The user's
    # netscope.branch()/reduce() hint markers — which carry no loc, so they never
    # loc-match.) If so, the static AST's recovered "branch loop" / "vote" are
    # redundant DUPLICATES of them, and keeping them floats disconnected strays
    # in the fused view (the sfumato 'branch loop' + second 'vote' bug).
    runtime_has_branch = any(
        (n.get("attrs") or {}).get("branch") for n in runtime.nodes())
    runtime_has_reduce = any(
        (n.get("attrs") or {}).get("reduce") for n in runtime.nodes())

    # 3) static-only nodes (structure the runtime never saw — e.g. a branch fan
    #    or a vote stage when there were NO runtime hints). Drop the ones that are
    #    redundant with what the runtime already captured:
    #    - declared-dim nodes (the static pre-check; redundant with runtime modules)
    #    - branch/reduce stages already present as runtime branch/reduce markers
    for st in static.nodes():
        if st["id"] in matched_static_ids:
            continue
        attrs = st.get("attrs") or {}
        if attrs.get("declared_dim"):
            continue
        if attrs.get("branch") and runtime_has_branch:
            continue
        if attrs.get("reduce") and runtime_has_reduce:
            continue
        fused.add_node(
            st["id"], kind=st["kind"], name=st["name"], parent=st.get("parent"),
            source="static", loc=st.get("loc"), meta=st.get("meta"),
            attrs=attrs,
        )

    # 4) static-only edges whose BOTH endpoints survived into the fused graph (a
    #    vote/branch wiring the runtime never captured). Mirrors mergeByLoc.ts — an
    #    endpoint that fused into a runtime node by loc no longer exists under its
    #    static id, so that edge is dropped rather than left dangling.
    for e in static.edges():
        if fused.has_node(e["src"]) and fused.has_node(e["dst"]):
            fused.add_edge(e["src"], e["dst"], kind=e["kind"],
                           tensor_meta=e.get("tensor_meta"),
                           source=e.get("source", "static"), condition=e.get("condition"))

    return fused
