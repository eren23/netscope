// Pure unit test for mergeByLoc + render — runs under plain node (no VSCode host):
//   npm run compile && npm run test:unit
import * as assert from "assert";
import { mergeByLoc } from "../mergeByLoc";
import { toElements } from "../render";
import { NVGraph } from "../ir";

function g(name: string, nodes: any[], edges: any[] = []): NVGraph {
  return { schema_version: "1", name, nodes, edges };
}
function node(o: any): any {
  return {
    id: o.id, kind: o.kind || "stage", name: o.name, parent: o.parent ?? null,
    source: o.source || "runtime", loc: o.loc ?? null, meta: o.meta || {}, attrs: o.attrs || {},
  };
}

let passed = 0;
function check(label: string, fn: () => void) {
  fn();
  passed++;
  console.log(`ok - ${label}`);
}

check("fuses runtime+static at same loc (keeps meta, gains attrs)", () => {
  const rt = g("r", [node({ id: "rt", kind: "model", name: "Qwen",
    loc: { file: "m.py", line: 12 }, meta: { out_shape: [1, 32] } })]);
  const st = g("s", [node({ id: "st", kind: "stage", name: "plan", source: "static",
    loc: { file: "m.py", line: 12 }, attrs: { declared: true } })]);
  const f = mergeByLoc(rt, st);
  const n = f.nodes.find((x) => x.loc && x.loc.line === 12)!;
  assert.strictEqual(n.source, "fused");
  assert.deepStrictEqual((n.meta as any).out_shape, [1, 32]);
  assert.strictEqual((n.attrs as any).declared, true);
  assert.strictEqual(f.nodes.length, 1);
});

check("carries static-only nodes through", () => {
  const rt = g("r", [node({ id: "rt", kind: "model", name: "Qwen",
    loc: { file: "m.py", line: 12 } })]);
  const st = g("s", [node({ id: "vote", kind: "stage", name: "vote", source: "static",
    loc: { file: "m.py", line: 20 }, attrs: { reduce: true } })]);
  const f = mergeByLoc(rt, st);
  const names = new Set(f.nodes.map((n) => n.name));
  assert.ok(names.has("Qwen") && names.has("vote"));
  assert.strictEqual(f.nodes.find((n) => n.name === "vote")!.source, "static");
});

check("drops static edges with missing endpoints", () => {
  const rt = g("r", [node({ id: "a", name: "a" })]);
  const st = g("s", [node({ id: "a2", name: "a2", source: "static" })],
    [{ src: "a2", dst: "ghost", kind: "dataflow", source: "static" }]);
  const f = mergeByLoc(rt, st);
  assert.strictEqual(f.edges.length, 0); // ghost endpoint dropped
});

check("render maps parent->compound and drops contains edges", () => {
  const gr = g("r",
    [node({ id: "p", name: "plan" }), node({ id: "c", name: "Qwen", parent: "p", meta: { out_shape: [1, 8] } })],
    [{ src: "p", dst: "c", kind: "contains", source: "runtime" },
     { src: "c", dst: "p", kind: "dataflow", source: "runtime", tensor_meta: { shape: [1, 8] } }]);
  const el = toElements(gr);
  const child = el.nodes.find((n: any) => n.data.id === "c")!;
  assert.strictEqual(child.data.parent, "p");          // compound nesting
  assert.ok(child.data.label.includes("[1, 8]"));      // shape in label
  assert.strictEqual(el.edges.length, 1);              // contains dropped, dataflow kept
  assert.strictEqual(el.edges[0].data.kind, "dataflow");
});

console.log(`\n${passed} passed`);
