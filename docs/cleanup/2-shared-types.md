# Criterion 2 — Shared TypeScript types & cross-language schema drift

Scope: type definitions WITHIN the TypeScript extension (`extension/src`), plus a
field-by-field schema-drift audit of `extension/src/ir.ts` against
`netscope/core/ir.py` (and the two Cytoscape producers it constrains:
`extension/src/render.ts` vs `netscope/sinks/html_sink.py`). Read-only.

## Overall assessment

The TS side is already in good shape. The graph IR is centralized in
`extension/src/ir.ts` (`Loc`, `NVNode`, `NVEdge`, `NVWarning`, `NVGraph`) and is
consumed consistently by `mergeByLoc.ts`, `shapeDecorations.ts`, `traceStore.ts`,
`diagnostics.ts`, and `extension.ts`. The Python types live in `core/ir.py` as the
single contract. `madge --circular` reports no cycles across the 8 TS modules.

Two real items surfaced, both small and local:

1. `extension/src/render.ts` is the one genuinely untyped module — its return type
   is an inline `{ nodes: any[]; edges: any[]; warnings: any[] }` and it builds
   `data: any` element objects (lines 32, 45, 64, 67). Naming `CytoscapeNode` /
   `CytoscapeEdge` interfaces would document the renderer's actual output schema
   and let the compiler catch field-name drift against the webview consumer in
   `netscope/web/template.html`.

2. `extension/src/diagnostics.ts` declares a private `Warning` interface
   (lines 23-30) that is a byte-for-byte duplicate of the exported `NVWarning` in
   `ir.ts` (lines 30-37). `knip` flags `NVWarning` as an "unused exported type" for
   exactly this reason — the consumer re-declares its own copy instead of importing
   the canonical one. This is the clearest consolidation win.

Schema-drift audit between `ir.ts` and `core/ir.py`: the IR node/edge/graph shapes
match. There is ONE producer/consumer field gap between the two Cytoscape sinks:
the Python `to_cytoscape` sets an edge `flow` field (`html_sink.py:100`) that
`render.ts` never emits. This is a latent parity gap, not an active bug (see R3).

Cross-language schema codegen would be overkill here and is explicitly out of
scope; not recommended (low confidence at best).

## Tool output

- `npx knip` (in `extension/`): flagged `NVWarning interface src/ir.ts:30:18` under
  "Unused exported types" — it is exported but the only other place that needs it
  (`diagnostics.ts`) re-declares a local `Warning` instead of importing it. Also
  flagged `clear` (`diagnostics.ts:74`) and `disposeShapeDecorations`
  (`shapeDecorations.ts:60`) as unused exports, and the 4 `media/vendor/*.min.js`
  plus `src/test/mergeByLoc.test.ts` as unused files — these are out of scope for
  this criterion (vendor libs are runtime-injected; the test file is run by the
  headless harness, not imported). Not addressed here.
- `npx madge --circular --extensions ts src`: "No circular dependency found!" (8 files).
- Grep of `netscope/web/template.html` for the fields each producer emits confirmed
  the consumer reads `flow` only via `selector: 'edge[flow = "inferred"]'`
  (template.html:252).

## Recommendations

### R1 — Name `CytoscapeNode` / `CytoscapeEdge` and type `toElements` (medium)
`render.ts:32` returns `{ nodes: any[]; edges: any[]; warnings: any[] }` and builds
`const data: any` objects at lines 45 and 67. Introduce two interfaces describing the
`{ data: {...} }` element shape (matching the fields actually set: node `id, name,
label, kind, meta, loc, prov, parent?, warn?, role, inferred?, diff?, diff_detail?`;
edge `id, source, target, kind, warn?, label?`) and type the return as
`{ nodes: CytoscapeNode[]; edges: CytoscapeEdge[]; warnings: NVWarning[] }`. Medium
(not high) because: (a) the optional-field set is broad and dynamically attached, so a
faithful interface needs care to avoid over-tightening; (b) it touches the renderer's
output contract; (c) it is a quality improvement, not a correctness fix. Worth doing,
but should be reviewed rather than auto-applied. The `warnings` element of the return
can immediately be `NVWarning[]` instead of `any[]` (it is literally `g.warnings`).

### R2 — Delete the duplicate `Warning` interface in diagnostics.ts; import `NVWarning` (high)
`extension/src/diagnostics.ts:23-30` defines `interface Warning { src; dst; detail;
severity?; kind?; source?; }`, identical to `NVWarning` in `ir.ts:30-37`. Replace the
local interface with `import { ..., NVWarning } from "./ir"` and use `NVWarning[]` at
`diagnostics.ts:31` (`graph as unknown as { warnings?: NVWarning[] }`). This removes the
duplication, resolves knip's "unused exported type NVWarning", and re-couples the
consumer to the canonical type so future warning-field changes propagate. Safe and
clearly correct: the shapes are identical, and `NVWarning` is already exported and
imported elsewhere in the same package (`ir.ts`/`mergeByLoc` import chain). High.

### R3 — Edge `flow` field: Python sink emits it, render.ts does not (medium)
`netscope/sinks/html_sink.py:100` sets `data["flow"] = e.get("source")` on every edge;
`extension/src/render.ts:67` builds the edge data with only `id, source, target, kind`
(+ optional `warn`/`label`) and never sets `flow`. The webview consumes `flow` exactly
once: `template.html:252` `selector: 'edge[flow = "inferred"]'` (dashed/dim styling for
LLM-inferred edges). So the two Cytoscape producers have drifted by one field. This is
NOT currently a user-visible bug, because the extension never renders inferred edges:
inferred edges are created with `source="inferred"` only by the standalone CLI path
`netscope/llm/infer.py:188`, and the extension has no command that runs `infer`
(grep of `extension/src` + `package.json` finds no `infer`/augment path; the extension's
LLM features are ask-a-node `runLLM` and view-spec `runViewSpec` only). Recommend adding
`data.flow = e.source` in `render.ts` for producer/consumer parity, so that if the
extension ever loads an inferred or fused-with-inferred trace, inferred edges style
correctly — matching the standalone HTML. Medium: it is a real schema mismatch and a
latent correctness risk, but it cannot manifest through any current extension code path,
so it should be reviewed, not auto-applied.
NB: node-level `inferred` IS handled by render.ts (line 54) and the diff `flow`/edge
concern does not apply — `core/diff.py` tags only nodes (`attrs.diff`/`diff_detail`),
which render.ts handles at lines 57-60.

### R4 — Cross-language schema codegen (low — do not implement)
A generator emitting `ir.ts` from `core/ir.py` (or a shared JSON Schema) would eliminate
the possibility of drift like R3 entirely. For a ~50-line `ir.ts` mirrored against a
~150-line `ir.py` this is disproportionate machinery; the mirrors are intentional
producer/consumer parity and are already documented as such in both files' header
comments. Mentioned only for completeness. Low confidence; not recommended.

## Non-findings (verified, deliberately NOT flagged)
- `ir.ts` node/edge/graph fields match `core/ir.py`: node `{id, kind, name, parent,
  source, loc, meta, attrs}`; edge `{src, dst, kind, tensor_meta?, source, condition?}`;
  graph `{schema_version, name, nodes, edges, warnings?}`. No missing/extra fields.
- `NVWarning` fields (`src, dst, detail, severity?, kind?, source?`) match what
  `core/checks.py:detect_mismatches` produces (`src, dst, severity, kind, detail`) plus
  the `source` that static warnings carry. No drift.
- Python types are already centralized in `core/ir.py`; no Python-side consolidation
  needed for this criterion.
- `data.warn`, `data.role`, `data.kind`, `data.meta`, `data.loc`, `data.prov`,
  `data.label`, node `data.diff`/`diff_detail`/`inferred` are emitted identically by
  both render.ts and html_sink.py and consumed by template.html. No drift there.
