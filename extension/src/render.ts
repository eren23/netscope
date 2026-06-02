// IR -> Cytoscape elements. Mirrors netscope/sinks/html_sink.py:to_cytoscape so
// the webview renders graphs identically to the standalone HTML: `parent` maps
// to compound nodes, `contains` edges are implied by nesting (dropped).

import { NVGraph } from "./ir";

function label(n: any): string {
  const out = n.meta && (n.meta.out_shape as number[] | undefined);
  return out ? `${n.name}\n[${out.join(", ")}]` : n.name;
}

export function toElements(g: NVGraph): { nodes: any[]; edges: any[] } {
  // nodes/edges touched by a mismatch warning -> data.warn, so the renderer can
  // paint them red AND the node panel can offer a "why flagged?" assistant action.
  const warnIds = new Set<string>();
  for (const w of (g as unknown as { warnings?: { src: string; dst: string }[] }).warnings || []) {
    warnIds.add(w.src);
    warnIds.add(w.dst);
  }

  const nodes = g.nodes.map((n) => {
    const data: any = {
      id: n.id, name: n.name, label: label(n), kind: n.kind,
      meta: n.meta || {}, loc: n.loc,
    };
    if (n.parent) data.parent = n.parent;
    if (warnIds.has(n.id)) data.warn = true;
    return { data };
  });

  const edges: any[] = [];
  g.edges.forEach((e, i) => {
    if (e.kind === "contains") return; // implied by compound nesting
    const data: any = { id: `e${i}`, source: e.src, target: e.dst, kind: e.kind };
    const shape = e.tensor_meta && e.tensor_meta.shape;
    if (shape && shape.length) data.label = shape.join("x");
    edges.push({ data });
  });

  return { nodes, edges };
}
