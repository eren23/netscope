// IR -> Cytoscape elements. Mirrors netscope/sinks/html_sink.py:to_cytoscape so
// the webview renders graphs identically to the standalone HTML: `parent` maps
// to compound nodes, `contains` edges are implied by nesting (dropped).

import { NVGraph } from "./ir";

function label(n: any): string {
  const out = n.meta && (n.meta.out_shape as number[] | undefined);
  return out ? `${n.name}\n[${out.join(", ")}]` : n.name;
}

// Mirror of netscope/enrich/roles.py:node_role — coarse architectural role from
// the node's name + qualified path, for the transformer "color by role" lens.
const ROLE_KEYS: [string, string[]][] = [
  ["attention", ["attention", "attn", "mha", "self_attn", "selfattention", "crossattention"]],
  ["mlp", ["mlp", "feedforward", "feed_forward", "ffn", "swiglu", "geglu", "moe", "experts"]],
  ["norm", ["layernorm", "rmsnorm", "batchnorm", "groupnorm", "norm", "ln_f", "ln_1", "ln_2"]],
  ["embedding", ["embedding", "embed", "wte", "wpe", "tok_emb", "pos_emb", "rotary"]],
  ["activation", ["relu", "gelu", "silu", "swish", "sigmoid", "tanh", "softmax", "act_fn", "activation"]],
  ["conv", ["conv", "pool"]],
  ["linear", ["linear", "proj", "dense", "lm_head", "out_proj", "fc"]],
];

function nodeRole(n: any): string {
  const s = `${n.name || ""} ${(n.meta && n.meta.qualname) || ""}`.toLowerCase();
  for (const [role, keys] of ROLE_KEYS) {
    if (keys.some((k) => s.includes(k))) return role;
  }
  return "other";
}

export function toElements(g: NVGraph): { nodes: any[]; edges: any[]; warnings: any[] } {
  // nodes/edges touched by a mismatch warning -> data.warn, so the renderer paints
  // them red; the top-level `warnings` array drives the HUD ⚠ pill + warn list.
  const warns = g.warnings || [];
  const warnIds = new Set<string>();
  const warnPairs = new Set<string>();
  for (const w of warns) {
    warnIds.add(w.src);
    warnIds.add(w.dst);
    warnPairs.add(`${w.src}->${w.dst}`);
  }

  const nodes = g.nodes.map((n) => {
    const data: any = {
      id: n.id, name: n.name, label: label(n), kind: n.kind,
      meta: n.meta || {}, loc: n.loc, prov: n.source,   // prov = the panel's "source" row
    };
    if (n.parent) data.parent = n.parent;
    if (warnIds.has(n.id)) data.warn = true;
    data.role = nodeRole(n);              // transformer "color by role" lens

    const attrs = n.attrs || {};
    if (attrs.inferred) data.inferred = true;   // LLM-inferred -> dashed/dim styling
    // trace-diff tags (added|removed|changed|same) drive the green/amber styling;
    // the cost overlay reads meta.* directly, so it needs nothing extra here.
    if (attrs.diff) {
      data.diff = attrs.diff;
      if (attrs.diff_detail) data.diff_detail = attrs.diff_detail;
    }
    return { data };
  });

  const edges: any[] = [];
  g.edges.forEach((e, i) => {
    if (e.kind === "contains") return; // implied by compound nesting
    const data: any = { id: `e${i}`, source: e.src, target: e.dst, kind: e.kind };
    if (warnPairs.has(`${e.src}->${e.dst}`)) data.warn = true;   // paint the clash edge red
    const shape = e.tensor_meta && e.tensor_meta.shape;
    if (shape && shape.length) data.label = shape.join("x");
    edges.push({ data });
  });

  return { nodes, edges, warnings: warns };
}
