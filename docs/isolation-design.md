# Design: tracing parts of a network in isolation

**Status:** draft for review (no code yet)
**Date:** 2026-05-31
**Author:** netscope

## Goal

Let a user trace **only a chosen part** of a model instead of the whole forward
pass — both from code (`scope=` on a session) and interactively (click a node in
the graph → see just that part). Three levels, increasing power and effort. They
stack: each is a stepping stone to the next.

## The enabling fact (why this is cheap)

We already have the two primitives this needs:

1. **Per-node input shape.** The torch pre-hook records `meta.in_shape` for every
   module (`instrument/torch_nn.py::pre_hook`).
2. **The live module + its real input.** That same pre-hook is called as
   `pre_hook(module, args)` — `module` is the actual `nn.Module` instance and
   `args` is its **real input tensor(s)** at call time. We currently read only
   metadata (zero-overhead), but we *can* grab a specific module's real input
   when asked.

Because of (2), "trace this part in isolation" does **not** need synthetic inputs
or a long-running kernel for the common case — we reuse the real tensor that was
already flowing through.

## New shared primitive: a module → qualified-name map

One small addition unlocks all three levels and improves normal traces too.

Today the global hook sees a module instance but not *where it lives* (`model` has
the names, the hook doesn't). Fix: when the **outermost** module is entered (span
depth 0 → 1), treat it as the trace root and build
`id(submodule) -> "layers.2.attn.q_proj"` from `root.named_modules()` once. Then
every node can be tagged with its real qualified name.

- Cost: one `named_modules()` walk per top-level forward (cheap, once).
- Bonus: nicer node labels and a stable handle for click-to-source + isolation.
- Edge case: scripts that call several top-level modules → first one is treated as
  root per forward; fine for the single-model case, note for later.

Lands in: `instrument/torch_nn.py` (capture root + map), `core/ir.py` (carry
`qualname` on nodes).

---

## Level 1 — Click-to-focus (view filter)

**What:** Click a node in the graph → collapse everything except that node's
subtree (optionally its direct dataflow neighbours too). No re-run; pure webview.

**UX:** node context action "Focus" / double-click. A breadcrumb + "show all"
restores the full graph. Reuses the existing expand/collapse machinery.

**Mechanism:** webview-only. On focus, hide siblings/ancestors-outside-path and
`cy.fit()` to the subtree. Nothing leaves the browser.

**Files:** `web/template.html` (focus action + restore), synced to
`extension/media/`. No Python, no extension host changes.

**Effort:** ~1 hour. **Limitation:** filters the *existing* trace, not a fresh run.

---

## Level 2 — `scope=` capture API (code-level isolation)

**What:** record only a chosen submodule subtree.

```python
# by module object (exact, no qualname needed):
with netscope.graph("decoder-blk-2", scope=model.model.layers[2]) as g:
    model(input_ids)            # full forward runs; only layer 2 is recorded

# or by name pattern:
with netscope.graph("attn-only", include=r"\.attn") as g:
    model(input_ids)
with netscope.graph("no-norms", exclude=r"RMSNorm") as g:
    model(input_ids)
```

**Mechanism:** the pre-hook gains a predicate.
- `scope=<module>` → precompute `frozenset(id(m) for m in scope.modules())`; record
  a node only if `id(module)` is in the set.
- `include=/exclude=<regex>` → match against the qualname from the shared map.
Outside the predicate the hook still no-ops (zero-overhead gate unchanged).

**Files:** `core/capture.py` (`graph()` + `Capture` take `scope/include/exclude`),
`instrument/torch_nn.py` (predicate in `pre_hook`). Pure library — usable with no
editor.

**Effort:** ~half day + tests. **Limitation:** the full forward still executes
(the rest is just not recorded); it's a code change, not a click.

---

## Level 3 — True isolated re-run (the "trace it alone" feature)

**What:** select a module (clicked node in the graph, or symbol under the cursor)
→ get a fresh focused trace of **just that submodule**, run on its real input,
without the rest of the pipeline.

**Flow:**
1. You've done a normal Run & Trace once → nodes carry `qualname` + `in_shape`.
2. Click a node → "Isolate". Webview posts `{type:'isolate', qualname}` to the
   extension host (sits right next to the existing `{type:'reveal'}` handler).
3. Extension re-runs the file with `NETSCOPE_ISOLATE=<qualname>` set.
4. During that run, the pre-hook watches for the module whose qualname matches.
   On match it stashes `(module, detach-copy of its real input)` — exactly one
   tensor, deliberately retained.
5. At session finalize, if a target was captured, open a child
   `graph("isolate:<qualname>")`, run `target(stashed_input)` under it, and dump
   to `NETSCOPE_ISOLATE_OUT`.
6. Extension opens that focused JSON in the webview.

**Why no kernel:** step 4 reuses the **real** input that flowed by, so the
isolated run is faithful (right shape/dtype/device, real values) and fast — one
tiny forward, not the whole model.

**Files:** `instrument/torch_nn.py` (match + stash input), `core/capture.py`
(finalize-time isolated re-run + second dump), `extension/src/extension.ts`
(`isolate` message + a `netscope.isolateModule` command + keybinding),
`web/template.html` (node "Isolate" action).

**Effort:** ~1 day incl. tests. **Limitation:** you pick the target then it
re-runs the file (matches how Run & Trace already works). Fully interactive
isolation *without* re-running is Level 4.

---

## Level 4 — Persistent kernel (deferred)

Fully interactive: isolate *anything* repeatedly with no re-run, tweak an input,
re-run a subgraph live. Requires keeping the Python process + model in memory —
a long-lived "netscope kernel" the extension talks to over IPC (like a debug
adapter / Jupyter kernel).

**Recommendation: defer.** Levels 1–3 deliver ~90% of the felt experience with
none of the process-lifecycle, IPC, or state-management complexity. Revisit once
1–3 are in use and we know the real interaction patterns.

---

## Recommended build sequence

1. **Shared primitive** — module→qualname map + `qualname` on nodes. (Enables 2 & 3,
   improves labels/click-to-source.)
2. **Level 1** — click-to-focus (instant value, webview-only).
3. **Level 2** — `scope=/include=/exclude=`.
4. **Level 3** — isolate-on-rerun.
5. Level 4 only if demanded.

Each step is independently shippable and TDD'd (real fixtures: resnet18 for
`scope=model.layer2`, a tiny decoder for `include=r"\.attn"`, an isolated
`TransformerEncoderLayer` for Level 3). Existing 72 tests stay green.

## Open decisions for you

1. **Level 1 "focus" scope:** subtree only, or subtree + its direct dataflow
   neighbours (upstream/downstream nodes)?
2. **Level 3 trigger:** from the graph (click a node) only, or also a
   `netscope: Isolate Module Under Cursor` editor command on a `self.attn = ...`
   line? (The editor one needs a small static lookup to resolve the qualname.)
3. **Scope by name:** regex (`include=r"\.attn"`) vs glob (`include="*.attn.*"`) —
   which reads better to you?
4. **Naming:** `scope=` vs `only=` for the module-object form; `include/exclude`
   for patterns. Happy with these?
