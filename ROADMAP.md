# netscope roadmap

netscope's pitch: **trace, visualize, and show errors in your neural-network
pipelines — as you write them.** v0.1 nails *trace + visualize + show-errors*.
The road ahead is mostly about the **"as you write"** half (live, in-editor) and
an **LLM-augmented** layer that reaches where deterministic analysis can't.

## v0.1.0 — shipped ✅

The full loop works, run-triggered:

- **Auto-trace** PyTorch / Hugging Face with zero decorators (`wrapt` post-import
  hooks + torch global module hooks, capture-once, metadata-only).
- **Interactive graph** — hierarchy as nested boxes, real tensor shapes, dataflow
  edges, repeated blocks folded; self-contained HTML (no CDN).
- **Show errors** — shape/rank mismatches flagged red + a warnings list.
- **Click-to-source** — every node carries `loc`; click jumps to the line.
- **Isolate a part** — click a node → re-run *just* that submodule on its real
  input (handles kwargs-taking transformer blocks).
- **VSCode/Cursor extension** — Run & Trace / Show Graph, shortcuts (⌘⌥T / ⌘⌥G),
  context menu, webview.
- Packaged: PyPI wheel + `.vsix`, both verified from a clean install.

## Guiding principle

Two engines speaking one IR, fused by source `loc`:
**static (what the code says)** ⊕ **runtime (what actually ran)** ⊕ **LLM (what
neither can infer alone)**. Every feature below is a producer or a consumer of
that one IR — nothing bolted on the side.

---

## v0.2 — "as you write" + the LLM layer (built together)

The differentiator and the LLM layer reinforce each other, so they grow in
parallel. The static skeleton gives the LLM ground truth; the LLM fills the gaps
the AST can't reach.

### Live static engine (deterministic core)
- **On-type skeleton.** Parse the file as you type → draw the architecture
  *before* you run. Overlay real shapes after a trace (the static⊕runtime fusion
  the whole design was built around).
- **Inline shape hints.** Each layer's real tensor shape as a ghost annotation on
  its line (from the last trace). The biggest "whoa, as I type" moment; cheap —
  we already have shapes + `loc`.
- **Mismatches as editor squiggles.** Surface the shape/rank clashes we already
  detect as red underlines on the offending line (VSCode diagnostics), not only
  in the graph.
- **Declared-dim checking.** Read `nn.Linear(256, 128)` etc. from source and flag
  a wiring clash *before* a single forward runs.

### LLM-augmented layer (four jobs, one provider abstraction)
1. **Augment static inference.** Where the AST gives up — dynamic indexing, custom
   `forward`, exotic layers — ask an LLM to infer the likely shape/dataflow, drawn
   as *provisional* (dashed) edges/nodes, clearly distinct from
   runtime-confirmed ones. Confidence shown; never silently presented as fact.
2. **Assistant over the graph.** "Explain this block", "why is this node red",
   "suggest a fix for this mismatch" — grounded in the IR + the source slice, not
   free-floating. Answers cite `loc`.
3. **netscope as agent context (MCP).** Expose the live graph + real shapes as an
   MCP server so Cursor / Claude Code can ask *"what actually flows into this
   layer?"* and get real captured shapes — grounding coding agents in reality
   instead of guesses.
4. **Generate custom views/analyzers.** Turn a prompt into a **declarative view
   spec** ("group by attention vs MLP", "highlight params > 1M", "color by
   dtype") that the renderer applies. Specs, not arbitrary code — safe and
   reproducible. This is the "make your own views" extensibility.

Design rules for the LLM layer:
- **Provider-agnostic.** A thin interface (Anthropic default; OpenAI/local
  pluggable). Bring-your-own-key. Never required for the core to work.
- **Grounded, not hallucinated.** The model only ever annotates/explains the IR;
  inferred elements are visually marked and confidence-scored.
- **Offline-first core.** Everything in v0.1 keeps working with no LLM, no network.

### Adoption
- README GIFs, the sfumato hero figure, more demos.
- Extension integration tests + CI (GitHub Actions: pytest + tsc + headless on
  push; build + publish on a version tag).

---

## v0.3+ — depth & reach

- **Trace diffing.** Compare two runs (before/after an edit, or two model
  variants): graph diff + shape diff. The iteration superpower.
- **`scope=` capture API** (isolation Level 2): `with netscope.graph(scope=model.layers[2])`
  — record only a subtree, pure-library.
- **Click-to-focus** (isolation Level 1): instant subtree focus in the graph, no
  re-run.
- **Richer metadata:** dtype, device, param + activation memory, FLOPs (thop,
  opt-in), per-layer timing.
- **LLM-specific views:** attention-head maps, KV-cache shapes, generation-step
  timeline — ties straight back to sfumato.

## Later / bigger bets

- **Persistent kernel** (isolation Level 4): keep the model in memory → isolate /
  tweak an input / re-run any part instantly, no file re-run.
- **More frameworks:** JAX/Flax, Keras.
- **Export bridges:** OpenTelemetry → Langfuse/Phoenix; shareable hosted graphs.

---

## Design seams already in place (so the above doesn't require rework)

- **Producer-agnostic IR + merge-by-`loc`** — static, runtime, and (soon) LLM all
  emit the same IR; `merge` is the only integration point.
- **Instrumentor registry** — add frameworks additively.
- **Sink interface** — JSON/HTML/mermaid today; websocket (live), OTel, and an
  MCP sink are drop-in.
- **Shared renderer** — one `template.html` for standalone HTML and the webview.
- **Schema versioning** — `SCHEMA_VERSION` guards IR evolution across lib +
  extension + future LLM annotations.
