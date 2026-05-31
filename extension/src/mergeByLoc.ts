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

  // 1) runtime nodes, fused with any static node sharing their loc
  for (const rt of runtime.nodes) {
    const k = locKey(rt.loc);
    const st = k ? staticByLoc.get(k) : undefined;
    const attrs = { ...rt.attrs };
    let source = rt.source;
    if (st) {
      Object.assign(attrs, st.attrs);
      source = "fused";
      matchedStatic.add(st.id);
    }
    nodes.push({ ...rt, source, attrs });
  }

  // 2) static-only nodes (structure the runtime never saw — e.g. un-run branches)
  for (const st of staticG.nodes) {
    if (matchedStatic.has(st.id)) continue;
    nodes.push({ ...st, source: "static" });
  }

  // edges: runtime edges plus static edges whose endpoints both survive
  const ids = new Set(nodes.map((n) => n.id));
  const edges: NVEdge[] = [];
  for (const e of runtime.edges) edges.push(e);
  for (const e of staticG.edges) {
    if (ids.has(e.src) && ids.has(e.dst)) edges.push(e);
  }

  return {
    schema_version: runtime.schema_version || staticG.schema_version,
    name: runtime.name || staticG.name,
    nodes,
    edges,
  };
}
