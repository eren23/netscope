// IR -> Cytoscape elements. Mirrors netscope/sinks/html_sink.py:to_cytoscape so
// the webview renders graphs identically to the standalone HTML: `parent` maps
// to compound nodes, `contains` edges are implied by nesting (dropped).

import { NVGraph } from "./ir";

function label(n: any): string {
  const out = n.meta && (n.meta.out_shape as number[] | undefined);
  return out ? `${n.name}\n[${out.join(", ")}]` : n.name;
}

export function toElements(g: NVGraph): { nodes: any[]; edges: any[] } {
  const nodes = g.nodes.map((n) => {
    const data: any = {
      id: n.id, name: n.name, label: label(n), kind: n.kind,
      meta: n.meta || {}, loc: n.loc,
    };
    if (n.parent) data.parent = n.parent;
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
