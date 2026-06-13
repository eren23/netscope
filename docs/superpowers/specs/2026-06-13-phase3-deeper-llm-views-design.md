# Phase 3 — Deeper LLM views (design spec)

**Date:** 2026-06-13
**Status:** approved design → ready for implementation plan
**Branch:** `qa-hardening` (continues PR #3, as a further commit)

## Context & goal

netscope traces neural-network pipelines into an interactive graph, with a metadata-only,
zero-overhead default (shapes, dtype, device, params, bytes, opt-in wall-time). The ROADMAP's
"deeper LLM-specific views" (`ROADMAP.md`) are the stated differentiator for the sfumato/LLM
use case but were unbuilt. This spec covers all three, as one coherent feature set built on a
shared opt-in foundation:

1. **KV-cache shapes** surfaced on the (already-existing) generation timeline.
2. **Attention head-stats** — per-head reductions rendered as a graph overlay.
3. The **opt-in capture mechanism** both ride on.

**Hard invariant:** the default trace stays metadata-only and zero-retention. All new capture is
strictly opt-in and bounded; nothing here changes behavior when `capture` is unset.

## What already exists (build on, don't rebuild)

- Generation-step timeline: `netscope.step()` (`hints/api.py`) tags step nodes; `core/timeline.py`
  aggregates `{step,label,time_ms,modules,out_shape}`. The cost heatmap already paints per-step
  latency (`web/template.html`). → KV-cache adds a field here; no new timeline machinery.
- Overlay machinery in `web/template.html`: `vhi/vdim/vcolor` (highlight/filter/colorBy), the role
  lens (`rolecolor`), and the cost heatmap. → the attention overlay reuses this pattern.
- `enrich/roles.py` — a pure, name-heuristic classifier with a tiny tested surface. → `enrich/attention.py`
  mirrors its shape (pure reduction functions, unit-tested in isolation).

## Component 1 — Capture options (shared foundation)

- `graph(name, *, profile=False, capture=None)` in `core/capture.py`. `capture` is an iterable of
  string flags; normalized to a `frozenset`. Recognized: `"attention"`, `"kv_cache"`.
- **Env override**, parsed in `graph()` exactly like `NETSCOPE_PROFILE` (`capture.py:104`):
  `NETSCOPE_CAPTURE=attention,kv_cache`. Env flags union with the kwarg (so the extension/CLI can
  enable without editing user code).
- Stored on `Capture` as `self.capture: frozenset[str]`; helper `cap.wants(flag) -> bool`.
- Unknown flags: ignored with a one-time `warnings.warn` (permissive, never raises).
- **Default `capture=None` and no env → empty frozenset → every new code path below is gated off →
  byte-for-byte the current behavior.**

## Component 2 — KV-cache shapes (metadata path)

- Gated by `cap.wants("kv_cache")`. In the torch post-hook (`instrument/torch_nn.py`), inspect a
  module's `output` for a KV cache: HF returns `past_key_values` as a `Cache`/`DynamicCache` (v5)
  or a legacy tuple-of-tuples. A small helper extracts a shape summary **without retaining tensors**:
  `meta["kv_cache"] = {"layers": L, "shape": [b, n_heads, seq, head_dim], "seq": seq}`.
- `core/timeline.py`: add `kv_seq` (and optionally `kv_shape`) to each step dict, read from the
  step subtree's recorded `kv_cache` meta. Watching `kv_seq` grow across steps is the autoregressive
  signature, now in numbers.
- Memory: shapes only. No change when the flag is off.

## Component 3 — Attention head-stats (value path, reduced)

- Gated by `cap.wants("attention")`. Capture attention weights **transiently**, immediately reduce
  to per-head scalars, discard the tensor. Stored as
  `meta["attn_heads"] = [{"entropy": float, "dist": float, "last": float}, ...]` on the attention
  module node, where:
  - `entropy` — mean per-head attention entropy (focus vs. diffuse).
  - `dist` — mean attention distance (how far back each head looks; |query_pos − key_pos| weighted).
  - `last` — fraction of attention mass on the final key position.
- **Reduction lives in `enrich/attention.py`** (new): pure functions `head_stats(weights) -> list[dict]`
  taking a `[heads, q, k]` (or `[b, heads, q, k]`) tensor, returning the scalar dicts. Pure + unit-tested;
  the hook just calls it and stores the result.
- **Acquisition (recommended path):** request attentions from HF models — when `capture` includes
  `"attention"`, set `output_attentions=True` for the duration of the session (e.g. patch the
  config/generate kwargs through the existing `transformers_hf` seam). Best-effort: models that don't
  surface attention weights are silently skipped (no `attn_heads` meta). Documented as an HF-first v1.

## Component 4 — Rendering (`web/template.html`)

- New `⊕ attention` overlay button (sibling of `⊕ role`): colors attention nodes by a chosen
  head-stat (default `entropy`), reusing the `rolecolor`/heatmap recolor path. A small dropdown
  selects the stat (entropy / dist / last).
- Node detail panel: when a node has `attn_heads`, render a compact per-head table (head | entropy |
  dist | %last).
- Timeline view: add a `kv_seq` column so cache growth reads alongside per-step latency.
- **Sync rule (CONTRIBUTING #1):** author in `netscope/web/template.html`, then copy byte-identical
  to `extension/media/template.html`; `tests/test_web_sync.py` enforces equality. Mirror any new
  render data fields in `extension/src/render.ts` if the extension surfaces them.

## Testing (hermetic, existing style)

- `tests/test_attention.py` — `enrich/attention.head_stats` math on hand-built weight tensors
  (a one-hot-on-last head → entropy≈0, last≈1; a uniform head → max entropy); meta population via a
  tiny module that returns fake attentions under `capture={"attention"}`.
- `tests/test_kv_cache.py` — shape extraction from both a fake `Cache`-like object and a legacy tuple;
  `timeline()` exposes `kv_seq`; growth across two steps.
- **Zero-retention guards:** assert that with `capture=None` no `attn_heads`/`kv_cache` meta appears
  and no tensors are held (extend `tests/test_overhead.py` style).
- Env parsing: `NETSCOPE_CAPTURE` unions with the kwarg; unknown flag warns, doesn't raise.

## Files

- `netscope/core/capture.py` — `capture` param, env parse, `cap.capture` + `cap.wants()`.
- `netscope/instrument/torch_nn.py` — gated KV-cache + attention recording in the hooks.
- `netscope/instrument/transformers_hf.py` — request `output_attentions` when the flag is set.
- `netscope/enrich/attention.py` (new) — pure per-head reductions.
- `netscope/core/timeline.py` — `kv_seq` field.
- `netscope/web/template.html` (+ byte-copy to `extension/media/template.html`) — attention overlay,
  per-head panel, timeline kv column; `extension/src/render.ts` if needed.
- `tests/test_attention.py`, `tests/test_kv_cache.py` (+ overhead guard).

## Build sequence (incremental, each shippable)

1. Capture-options foundation + tests (no behavior change).
2. KV-cache shapes + timeline field + tests (metadata-only; lowest risk).
3. `enrich/attention.py` reductions + tests (pure, no integration).
4. Attention acquisition + hook wiring + tests (the risky integration).
5. Rendering (overlay + panel + timeline column) + web-sync.

## Risks / open questions

- **Attention acquisition fragility.** Forcing `output_attentions=True` is HF-specific and some
  architectures (e.g. fused/SDPA paths) won't return weights. v1 is HF-first, best-effort, skips
  silently. Non-HF / non-attention frameworks get nothing — acceptable.
- **Step↔KV association.** KV-cache meta is recorded on module nodes; `timeline()` must find it under
  the step subtree. If a model is traced without `step()` markers, KV shapes still attach to nodes but
  there's no per-step timeline — acceptable (the node-level data is still there).
- **Overhead when opted-in.** `output_attentions=True` has real compute cost; documented as the price
  of the flag, never on by default.

## Out of scope (explicit)

- Raw/downsampled attention matrices and a seq×seq heatmap panel (chose reduced per-head stats).
- A `scope=` capture API and click-to-focus isolation (separate ROADMAP items).
- Non-torch frameworks (the deferred JAX/Flax adapter; Phase 2 laid its groundwork).
