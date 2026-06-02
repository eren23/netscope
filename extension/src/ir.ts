// IR types mirroring the Python netscope schema (netscope/core/ir.py).
// The extension consumes two producers — the static CLI and a traced run —
// and fuses them; both speak this shape.

export interface Loc {
  file: string;
  line: number;
}

export interface NVNode {
  id: string;
  kind: "pipeline" | "stage" | "model" | "module" | "op";
  name: string;
  parent: string | null;
  source: "runtime" | "static" | "fused";
  loc: Loc | null;
  meta: Record<string, unknown>;
  attrs: Record<string, unknown>;
}

export interface NVEdge {
  src: string;
  dst: string;
  kind: "dataflow" | "control" | "contains";
  tensor_meta?: { shape?: number[]; dtype?: string } | null;
  source: string;
  condition?: string | null;
}

export interface NVWarning {
  src: string;
  dst: string;
  detail: string;
  severity?: string;
  kind?: string;
  source?: string;
}

export interface NVGraph {
  schema_version: string;
  name: string;
  nodes: NVNode[];
  edges: NVEdge[];
  warnings?: NVWarning[];
}

export function locKey(loc: Loc | null): string | null {
  return loc ? `${loc.file}:${loc.line}` : null;
}
