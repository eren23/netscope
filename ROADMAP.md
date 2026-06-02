# netscope roadmap

netscope's pitch: **trace, visualize, and show errors in your neural-network
pipelines — as you write them.** v0.1 nails *trace + visualize + show-errors*.
The road ahead is mostly about the **"as you write"** half (live, in-editor) and
an **LLM-augmented** layer that reaches where deterministic analysis can't.

We ship this in **small 0.1.x increments**, not a big-bang 0.2 — each milestone is
independently useful and independently releasable. The keystone is a cheap
library fix (give every traced node a source `loc`) that unlocks the whole
editor-live experience.

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

## The 0.1.x line — "as you write" in the editor

Bring the trace's real shapes + mismatch warnings onto the line, in the editor,
without leaving the file. Ordered so each step is shippable and the cheap,
high-impact, LLM-free wins come first.

### M0 (0.1.1) — `loc` on every traced node *(keystone)*
Runtime module nodes don't carry a source location today, so the editor can't map
a node back to a line. New `netscope/static/module_loc.py` scans the model's
defining file (`self.x = nn.Conv2d(...)`, `Sequential`/`ModuleList` index naming)
to map each `meta.qualname` → `{file, line}`; the torch hook sets `loc` from it.
Pure-AST, best-effort, never raises. **Unlocks inline hints, squiggles, AND fixes
click-to-source at once.**

### M1 (0.1.2) — inline shape hints + mismatch squiggles *(the headline)*
Pure extension/TS on top of M0, no LLM:
- **Inline shape hints (InlayHints).** Each layer's real `out_shape` as faint
  end-of-line ghost text, from the last trace.
- **Mismatches as squiggles (Diagnostics).** The `warnings` we already compute
  (`checks.py`) rendered as red underlines on the offending line, not only in the
  graph.

### M2 (0.1.3) — declared-dim pre-check, no run required
Extend the static AST producer to read `nn.Linear(in, out)` / `nn.Conv2d(...)`
literal dims + `forward` call order and flag an obvious wiring clash *before* a
single forward runs. Same `warnings` channel + diagnostics path as M1, tagged
`source:"static"`. Conservative — literal + directly-wired only, no false alarms.

### M3 (0.1.4) — LLM layer, entry point ✅
**Assistant over the graph**, shipped. Click a node → "explain / why flagged /
suggest fix", grounded in the node's IR slice (shapes, neighbours, the concrete
warning) + the real source lines via `loc`. The model only ever annotates real
netscope data — it can't invent architecture.

- `netscope/llm/provider.py` — one thin client for ANY OpenAI-compatible
  `/chat/completions` endpoint (stdlib `urllib`, no SDK/dependency). Default
  gateway **OpenRouter** (→ many cheap models, e.g. Gemini Flash); point
  `NETSCOPE_LLM_BASE_URL`/`_MODEL` at OpenAI / Together / Groq / a local server.
- `netscope/llm/prompts.py` — the grounded message builder.
- `netscope/llm/__init__.py` — `available()` / `explain(graph, node, question=)`
  / `LLMUnavailable`; `python -m netscope.llm <graph.json> <node> <question>` CLI.
- Env-keyed, **hard-gated**: no key → the layer is simply unavailable and the rest
  of netscope works fully offline. Key precedence `NETSCOPE_LLM_API_KEY` →
  `OPENROUTER_API_KEY` → `OPENAI_API_KEY`; netscope never stores it.
- Extension: node-panel "ask" buttons → shells out to the CLI → answer rendered
  in the panel.

The other LLM jobs (augmented inference, MCP, generated views) follow below.

### Adoption (alongside the above)
- README GIFs, the sfumato hero figure, more demos.
- Extension integration tests + CI (GitHub Actions: pytest + tsc + headless on
  push; build + publish on a version tag).

---

## The LLM layer, in full (M3 and beyond)

The static skeleton gives the LLM ground truth; the LLM fills the gaps the AST
can't reach. Four jobs over one provider abstraction:

1. **Assistant over the graph** *(M3, first)*. Explain / diagnose / suggest-fix,
   grounded in IR + source slice. Answers cite `loc`.
2. **Augment static inference.** Where the AST gives up — dynamic indexing, custom
   `forward`, exotic layers — infer the likely shape/dataflow, drawn as
   *provisional* (dashed) edges/nodes, distinct from runtime-confirmed ones.
   Confidence shown; never silently presented as fact.
3. **netscope as agent context (MCP).** Expose the live graph + real shapes as an
   MCP server so Cursor / Claude Code can ask *"what actually flows into this
   layer?"* and get real captured shapes — grounding coding agents in reality.
4. **Generate custom views/analyzers.** Turn a prompt into a **declarative view
   spec** ("group by attention vs MLP", "highlight params > 1M", "color by
   dtype") that the renderer applies. Specs, not arbitrary code — safe.

Design rules for the LLM layer:
- **Provider-agnostic.** A thin interface (Anthropic default; OpenAI/local
  pluggable). Bring-your-own-key. Never required for the core to work.
- **Grounded, not hallucinated.** The model only ever annotates/explains the IR;
  inferred elements are visually marked and confidence-scored.
- **Offline-first core.** Everything in v0.1 keeps working with no LLM, no network.

---

## Later 0.1.x / v0.2 — depth & reach

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
