// Fuse a runtime graph with a static graph by source location.
// Mirrors netscope/core/merge.py so the editor's fused view matches the library's:
// a runtime node and a static node at the same loc describe the same code seen
// two ways — runtime knows real shapes (executed path), static knows declared
// structure (names, branch/vote, even un-run code). The fused node keeps the
// runtime meta and gains the static attrs. One-sided nodes are carried through.

import { NVGraph, NVNode, NVEdge, locKey } from "./ir";

export function mergeByLoc(runtime: NVGraph, staticG: NVGraph): NVGraph {
  const staticByLoc = new Map<string, NVNode>();
  for (const n of staticG.nodes) {
    const k = locKey(n.loc);
    if (k) staticByLoc.set(k, n);
  }

  const matchedStatic = new Set<string>();
  const nodes: NVNode[] = [];

  // 1) runtime nodes, fused with any static node sharing their loc. A static
  //    node fuses into AT MOST ONE runtime node (first match wins) — two runtime
  //    nodes can share a loc (a submodule called twice / a loop body), and
  //    duplicating the static attrs across both is wrong. Mirrors core/merge.py.
  for (const rt of runtime.nodes) {
    const k = locKey(rt.loc);
    let st = k ? staticByLoc.get(k) : undefined;
    if (st && matchedStatic.has(st.id)) st = undefined;
    const attrs = { ...rt.attrs };
    let source = rt.source;
    if (st) {
      Object.assign(attrs, st.attrs);
      source = "fused";
      matchedStatic.add(st.id);
    }
    nodes.push({ ...rt, source, attrs });
  }

  // does the runtime already have branch/reduce stages (the user's branch()/
  // reduce() hint markers — no loc, so they never loc-match)? If so the static
  // AST's "branch loop"/"vote" are redundant duplicates that would float as
  // disconnected strays in the fused view.
  const rtHasBranch = runtime.nodes.some((n) => n.attrs?.branch);
  const rtHasReduce = runtime.nodes.some((n) => n.attrs?.reduce);

  // 2) static-only nodes (structure the runtime never saw). Drop the redundant
  //    ones: declared-dim nodes, and branch/reduce stages the runtime already
  //    captured. Mirrors netscope/core/merge.py.
  for (const st of staticG.nodes) {
    if (matchedStatic.has(st.id)) continue;
    const a: Record<string, unknown> = st.attrs || {};
    if (a.declared_dim) continue;
    if (a.branch && rtHasBranch) continue;
    if (a.reduce && rtHasReduce) continue;
    nodes.push({ ...st, source: "static" });
  }

  // edges: runtime edges plus static edges whose endpoints both survive
  const ids = new Set(nodes.map((n) => n.id));
  const edges: NVEdge[] = [];
  for (const e of runtime.edges) edges.push(e);
  for (const e of staticG.edges) {
    if (ids.has(e.src) && ids.has(e.dst)) edges.push(e);
  }

  // carry warnings through: the shape/rank mismatches are computed on the runtime
  // trace (real shapes), and the editor's squiggles read them off the fused graph.
  // Dropping them here is why mismatches rendered in the graph but not on the line.
  return {
    schema_version: runtime.schema_version || staticG.schema_version,
    name: runtime.name || staticG.name,
    nodes,
    edges,
    warnings: [...(runtime.warnings || []), ...(staticG.warnings || [])],
  };
}
