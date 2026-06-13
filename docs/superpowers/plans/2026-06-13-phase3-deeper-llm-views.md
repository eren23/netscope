# Phase 3 — Deeper LLM Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in capture of KV-cache shapes and per-head attention statistics, surfaced on the existing generation timeline and as a graph overlay — without changing the metadata-only, zero-retention default.

**Architecture:** A single `capture={...}` knob on `graph()` (plus `NETSCOPE_CAPTURE` env, mirroring `profile`) gates two new recording paths in the torch post-hook: (a) KV-cache *shapes* (metadata only) and (b) attention weights captured transiently and immediately reduced to per-head scalars via a new pure `enrich/attention.py`. The renderer gets an `⊕ attention` overlay and a KV column on the timeline. Spec: `docs/superpowers/specs/2026-06-13-phase3-deeper-llm-views-design.md`.

**Tech Stack:** Python 3.10+, pytest (+pytest-cov), torch (the traced runtime; imported lazily), vendored Cytoscape JS in `web/template.html`.

**Conventions for every task:** run `.venv/bin/python -m pytest ...` (the repo's venv). Keep `ruff check netscope tests` and `mypy` green. The default trace must record nothing new — every task includes/keeps a "capture off → unchanged" assertion.

---

## File Structure

- `netscope/core/capture.py` — **modify**: add `capture` param to `Capture` + `graph()`, env parse, `cap.wants()`.
- `netscope/enrich/attention.py` — **create**: pure `head_stats(weights)` reductions (entropy / dist / last).
- `netscope/instrument/torch_nn.py` — **modify**: `_kv_cache_shape()` + `_attention_weights()` helpers; gated recording in `post()`.
- `netscope/instrument/transformers_hf.py` — **modify**: request `output_attentions=True` on `generate` when attention capture is on.
- `netscope/core/timeline.py` — **modify**: add `kv_seq` to each step dict.
- `netscope/web/template.html` — **modify**: `⊕ attention` overlay + per-head panel + timeline KV column; then byte-copy to `extension/media/template.html`.
- `tests/test_capture_options.py`, `tests/test_attention.py`, `tests/test_kv_cache.py` — **create**.

---

## Task 1: Capture options foundation

**Files:**
- Modify: `netscope/core/capture.py`
- Test: `tests/test_capture_options.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capture_options.py
from __future__ import annotations

import netscope
from netscope.core import capture as capmod


def test_default_capture_is_empty():
    with netscope.graph("g") as g:  # noqa: F841
        cap = netscope.active_capture()
        assert cap.capture == frozenset()
        assert cap.wants("attention") is False


def test_capture_kwarg_sets_flags():
    with netscope.graph("g", capture={"attention", "kv_cache"}):
        cap = netscope.active_capture()
        assert cap.wants("attention") and cap.wants("kv_cache")


def test_env_capture_unions_with_kwarg(monkeypatch):
    monkeypatch.setenv("NETSCOPE_CAPTURE", "kv_cache")
    with netscope.graph("g", capture={"attention"}):
        cap = netscope.active_capture()
        assert cap.wants("attention") and cap.wants("kv_cache")


def test_unknown_flag_warns_not_raises(recwarn):
    with netscope.graph("g", capture={"bogus"}):
        cap = netscope.active_capture()
        assert cap.wants("bogus") is False          # dropped
    assert any("bogus" in str(w.message) for w in recwarn.list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_capture_options.py -q`
Expected: FAIL — `Capture` has no `capture`/`wants`, `graph()` rejects the `capture` kwarg.

- [ ] **Step 3: Implement in `netscope/core/capture.py`**

Add near the top (after imports):

```python
import warnings

_VALID_CAPTURE = frozenset({"attention", "kv_cache"})
```

Change `Capture.__init__` signature and body:

```python
    def __init__(self, name: str = "", profile: bool = False,
                 capture: "frozenset[str] | None" = None) -> None:
        self.graph = NVGraph(name=name)
        self._counter = itertools.count()
        self.profile = profile
        # opt-in extra capture (e.g. "attention", "kv_cache"). Empty by default ->
        # the steady-state trace stays metadata-only and zero-retention.
        self.capture: "frozenset[str]" = capture or frozenset()

    def wants(self, flag: str) -> bool:
        return flag in self.capture
```

Change `graph()` signature to `def graph(name: str = "", *, profile: bool = False, capture=None)` and, after the existing `NETSCOPE_PROFILE` block, add:

```python
    flags = set(capture or ())
    env_cap = os.environ.get("NETSCOPE_CAPTURE")
    if env_cap:
        flags |= {f.strip() for f in env_cap.split(",") if f.strip()}
    unknown = flags - _VALID_CAPTURE
    if unknown:
        warnings.warn(
            f"netscope: ignoring unknown capture flag(s): {sorted(unknown)}; "
            f"valid: {sorted(_VALID_CAPTURE)}",
            RuntimeWarning, stacklevel=2,
        )
    flags &= _VALID_CAPTURE
```

Then pass it through: `cap = Capture(name, profile=profile, capture=frozenset(flags))`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_capture_options.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Keep gates green + commit**

```bash
.venv/bin/ruff check netscope tests && .venv/bin/mypy
git add netscope/core/capture.py tests/test_capture_options.py
git commit -m "feat(capture): opt-in capture= flags + NETSCOPE_CAPTURE (default unchanged)"
```

---

## Task 2: Pure per-head attention reductions

**Files:**
- Create: `netscope/enrich/attention.py`
- Test: `tests/test_attention.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attention.py
from __future__ import annotations

import math

import torch

from netscope.enrich.attention import head_stats


def test_focused_head_low_entropy_high_last():
    # one head, q=k=4, every query attends ONLY to the last key.
    w = torch.zeros(1, 4, 4)
    w[0, :, -1] = 1.0
    (h,) = head_stats(w)
    assert h["entropy"] == 0.0          # all mass on one key
    assert h["last"] == 1.0
    assert h["dist"] > 0                 # looks back to the last key


def test_uniform_head_max_entropy():
    w = torch.full((1, 4, 4), 0.25)      # uniform over 4 keys
    (h,) = head_stats(w)
    assert math.isclose(h["entropy"], math.log(4), rel_tol=1e-5)
    assert math.isclose(h["last"], 0.25, rel_tol=1e-5)


def test_4d_input_is_meaned_over_batch_and_lists_each_head():
    w = torch.rand(2, 3, 5, 5)           # [batch, heads, q, k]
    stats = head_stats(w)
    assert len(stats) == 3               # one dict per head
    assert set(stats[0]) == {"entropy", "dist", "last"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_attention.py -q`
Expected: FAIL — module `netscope.enrich.attention` does not exist.

- [ ] **Step 3: Implement `netscope/enrich/attention.py`**

```python
"""Per-head attention reductions — the value-side LLM lens.

Attention weight tensors are huge ([batch, heads, q, k]); we never retain them.
Given a weight tensor (already in hand inside a forward hook), reduce it to a few
scalars per head and throw the tensor away:

- entropy : mean attention entropy per query (focused≈0, diffuse≈log k)
- dist    : mean attention distance |query_pos - key_pos| (how far back it looks)
- last    : fraction of attention mass on the final key position

Pure + framework-light: works on any torch tensor shaped [heads, q, k] or
[batch, heads, q, k] (batch is averaged out). Returns one dict per head.
"""
from typing import List


def head_stats(weights) -> List[dict]:
    import torch

    w = weights.detach().to(torch.float32)
    if w.dim() == 4:            # [batch, heads, q, k] -> mean over batch
        w = w.mean(dim=0)
    if w.dim() != 3:
        return []
    heads, q, k = w.shape
    # entropy per (head, query), averaged over queries
    p = w.clamp_min(0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    ent = -(p * (p.clamp_min(1e-9)).log()).sum(dim=-1).mean(dim=-1)   # [heads]
    # attention distance: sum_k p * |q_idx - k_idx|, averaged over queries
    qi = torch.arange(q).view(q, 1)
    ki = torch.arange(k).view(1, k)
    dist_mat = (qi - ki).abs().to(torch.float32)                     # [q, k]
    dist = (p * dist_mat).sum(dim=-1).mean(dim=-1)                   # [heads]
    last = p[:, :, -1].mean(dim=-1)                                  # [heads]
    return [
        {"entropy": round(float(ent[h]), 4),
         "dist": round(float(dist[h]), 4),
         "last": round(float(last[h]), 4)}
        for h in range(heads)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_attention.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Keep gates green + commit**

```bash
.venv/bin/ruff check netscope tests && .venv/bin/mypy
git add netscope/enrich/attention.py tests/test_attention.py
git commit -m "feat(enrich): per-head attention reductions (entropy/dist/last)"
```

---

## Task 3: KV-cache shape extraction helper

**Files:**
- Modify: `netscope/instrument/torch_nn.py` (add `_kv_cache_shape`)
- Test: `tests/test_kv_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kv_cache.py
from __future__ import annotations

import torch

from netscope.instrument.torch_nn import _kv_cache_shape


def test_legacy_tuple_of_kv():
    # legacy HF: past_key_values = ((k, v), (k, v), ...) per layer; k=[b,heads,seq,hd]
    k = torch.zeros(1, 8, 5, 64)
    v = torch.zeros(1, 8, 5, 64)
    out = {"past_key_values": ((k, v), (k, v))}
    info = _kv_cache_shape(out)
    assert info == {"layers": 2, "shape": [1, 8, 5, 64], "seq": 5}


def test_cache_object_with_key_cache():
    class _Cache:                       # mimics a v5 DynamicCache
        key_cache = [torch.zeros(1, 8, 7, 64)]
    out = {"past_key_values": _Cache()}
    info = _kv_cache_shape(out)
    assert info["seq"] == 7 and info["shape"][-1] == 64


def test_no_kv_returns_none():
    assert _kv_cache_shape(torch.zeros(2, 3)) is None
    assert _kv_cache_shape({"logits": torch.zeros(1, 5, 10)}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_kv_cache.py -q`
Expected: FAIL — `_kv_cache_shape` not defined.

- [ ] **Step 3: Implement `_kv_cache_shape` in `netscope/instrument/torch_nn.py`**

Add after `_act_bytes` (near the other helpers, ~line 102):

```python
def _kv_cache_shape(output) -> Optional[dict]:
    """Shape summary of a KV cache found in a module output — metadata only, no
    tensor retained. Handles HF's legacy tuple-of-(k,v)-per-layer and v5 Cache
    objects. Returns {layers, shape:[b,heads,seq,head_dim], seq} or None."""
    pkv = None
    if isinstance(output, dict):
        pkv = output.get("past_key_values")
    else:
        pkv = getattr(output, "past_key_values", None)
    if pkv is None:
        return None
    try:
        # v5 Cache object: a list of per-layer key tensors
        key_cache = getattr(pkv, "key_cache", None)
        if key_cache:
            k0 = key_cache[0]
            shape = list(k0.shape)
            return {"layers": len(key_cache), "shape": shape, "seq": int(shape[-2])}
        # legacy: ((k, v), (k, v), ...)
        if isinstance(pkv, (tuple, list)) and pkv and isinstance(pkv[0], (tuple, list)):
            k0 = pkv[0][0]
            shape = list(k0.shape)
            return {"layers": len(pkv), "shape": shape, "seq": int(shape[-2])}
    except Exception:
        return None
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_kv_cache.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check netscope tests && .venv/bin/mypy
git add netscope/instrument/torch_nn.py tests/test_kv_cache.py
git commit -m "feat(instrument): KV-cache shape extraction helper (metadata only)"
```

---

## Task 4: Record KV-cache + attention in the post-hook (gated)

**Files:**
- Modify: `netscope/instrument/torch_nn.py` (the `post()` closure, ~lines 278-310; add `_attention_weights` helper)
- Test: `tests/test_kv_cache.py`, `tests/test_attention.py` (append integration tests)

- [ ] **Step 1: Write the failing tests (append)**

```python
# append to tests/test_kv_cache.py
import netscope


class _KVModel(torch.nn.Module):
    def forward(self, x):
        k = torch.zeros(1, 8, x.shape[1], 64)
        return {"logits": x, "past_key_values": ((k, k),)}


def test_kv_cache_recorded_only_when_opted_in():
    m, x = _KVModel(), torch.zeros(1, 5, 16)
    with netscope.graph("on", capture={"kv_cache"}) as g:
        m(x)
    assert any((n.get("meta") or {}).get("kv_cache", {}).get("seq") == 5 for n in g.nodes())

    with netscope.graph("off") as g2:        # default: nothing recorded
        m(x)
    assert all("kv_cache" not in (n.get("meta") or {}) for n in g2.nodes())
```

```python
# append to tests/test_attention.py
import netscope


class _AttnModel(torch.nn.Module):
    def forward(self, x):
        attn = torch.softmax(torch.rand(1, 3, 4, 4), dim=-1)  # [b,heads,q,k]
        return (x, attn)


def test_attention_recorded_only_when_opted_in():
    m, x = _AttnModel(), torch.zeros(1, 4, 8)
    with netscope.graph("on", capture={"attention"}) as g:
        m(x)
    hit = [n for n in g.nodes() if (n.get("meta") or {}).get("attn_heads")]
    assert hit and len(hit[0]["meta"]["attn_heads"]) == 3

    with netscope.graph("off") as g2:
        m(x)
    assert all("attn_heads" not in (n.get("meta") or {}) for n in g2.nodes())
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_kv_cache.py tests/test_attention.py -q`
Expected: FAIL — post-hook records neither field yet.

- [ ] **Step 3: Implement the `_attention_weights` helper + gated recording**

Add helper after `_kv_cache_shape`:

```python
def _attention_weights(output):
    """First tensor in `output` that looks like attention weights: 4-D with equal
    last two dims ([batch, heads, q, k] with q==k). None if absent. Heuristic —
    best-effort; netscope requests output_attentions on HF models (see
    transformers_hf). Never retained beyond the immediate reduction."""
    for t in _iter_tensors(output):
        if _is_tensor(t) and t.dim() == 4 and t.shape[-1] == t.shape[-2]:
            return t
    return None
```

In the `post()` closure, after the existing `update` is built and before `cap.close_span(...)` (around line 304), add:

```python
            if cap.wants("kv_cache"):
                kv = _kv_cache_shape(output)
                if kv is not None:
                    update["kv_cache"] = kv
            if cap.wants("attention"):
                aw = _attention_weights(output)
                if aw is not None:
                    from netscope.enrich.attention import head_stats
                    stats = head_stats(aw)
                    if stats:
                        update["attn_heads"] = stats
                    del aw  # drop the tensor immediately — record stats, not values
```

(Note: `cap` is already `ctx.active_capture()` at the top of `post()`; the early
`if cap is None or not pending: return` guard already protects these calls.)

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_kv_cache.py tests/test_attention.py -q`
Expected: PASS (all, including the "off → nothing" guards).

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check netscope tests && .venv/bin/mypy
git add netscope/instrument/torch_nn.py tests/test_kv_cache.py tests/test_attention.py
git commit -m "feat(instrument): record KV-cache shapes + attention head-stats when opted in"
```

---

## Task 5: KV-cache on the generation timeline

**Files:**
- Modify: `netscope/core/timeline.py`
- Test: `tests/test_timeline.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
# append to tests/test_timeline.py
from netscope.core.ir import NVGraph


def test_timeline_surfaces_kv_seq():
    g = NVGraph(name="gen")
    g.add_node("s0", kind="stage", name="step 0", attrs={"step": 0})
    g.add_node("m0", kind="model", name="decoder", parent="s0",
               meta={"out_shape": [1, 1, 32000], "kv_cache": {"layers": 2, "shape": [1, 8, 6, 64], "seq": 6}})
    from netscope.core.timeline import timeline
    (row,) = timeline(g)
    assert row["step"] == 0 and row["kv_seq"] == 6
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_timeline.py::test_timeline_surfaces_kv_seq -q`
Expected: FAIL — `KeyError: 'kv_seq'`.

- [ ] **Step 3: Implement in `netscope/core/timeline.py`**

Inside the per-step loop, after computing `out_shape`/`modules`, scan the subtree for kv-cache meta:

```python
        kv_seq = None
        for did in _descendant_ids(graph, n["id"]):
            kv = ((nodes.get(did) or {}).get("meta") or {}).get("kv_cache")
            if kv and kv.get("seq") is not None:
                kv_seq = kv["seq"]          # last one wins (deepest decode call)
```

Then add `"kv_seq": kv_seq,` to the appended step dict.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_timeline.py -q`
Expected: PASS (existing timeline tests + the new one).

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check netscope tests && .venv/bin/mypy
git add netscope/core/timeline.py tests/test_timeline.py
git commit -m "feat(timeline): surface kv_seq per generation step"
```

---

## Task 6: Request `output_attentions` on HF generate (gated)

**Files:**
- Modify: `netscope/instrument/transformers_hf.py`
- Test: `tests/test_transformers_hf.py` (append)

**Why:** HF models only return attention weights when asked. When attention capture
is on, inject `output_attentions=True` so Task 4's `_attention_weights` finds them.

- [ ] **Step 1: Write the failing test (append)**

```python
# append to tests/test_transformers_hf.py
import netscope
from netscope.instrument.transformers_hf import _maybe_request_attentions


def test_injects_output_attentions_when_capturing_attention():
    with netscope.graph("g", capture={"attention"}):
        kwargs = _maybe_request_attentions({})
        assert kwargs.get("output_attentions") is True


def test_does_not_inject_by_default():
    with netscope.graph("g"):
        assert "output_attentions" not in _maybe_request_attentions({})


def test_respects_user_explicit_value():
    with netscope.graph("g", capture={"attention"}):
        kwargs = _maybe_request_attentions({"output_attentions": False})
        assert kwargs["output_attentions"] is False   # never override the user
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transformers_hf.py -q`
Expected: FAIL — `_maybe_request_attentions` not defined.

- [ ] **Step 3: Implement in `netscope/instrument/transformers_hf.py`**

Add:

```python
from netscope.core import context as ctx


def _maybe_request_attentions(kwargs: dict) -> dict:
    """When attention capture is on, ask HF to return attention weights (it won't
    by default). Uses setdefault so an explicit user value always wins."""
    cap = ctx.active_capture()
    if cap is not None and cap.wants("attention"):
        kwargs.setdefault("output_attentions", True)
    return kwargs
```

Then wire it into the generate wrapper. Replace the `safe_patch(...)` call with a
bespoke wrapt patch that can mutate kwargs (the generic `span_wrapper` cannot):

```python
def register() -> None:
    global _installed
    if _installed:
        return
    try:
        import wrapt
    except Exception:
        return

    @wrapt.patch_function_wrapper("transformers.generation.utils", "GenerationMixin.generate")
    def _wrapped(wrapped, instance, args, kwargs):
        if not ctx.is_capturing():
            return wrapped(*args, **kwargs)        # zero-overhead gate
        kwargs = _maybe_request_attentions(dict(kwargs))
        cap = ctx.active_capture()
        handle = cap.open_span(_gen_name(instance, args, kwargs), kind="model",
                               meta=_gen_meta(instance, args, kwargs))
        try:
            return wrapped(*args, **kwargs)
        finally:
            cap.close_span(handle)
    _installed = True
```

Keep `_gen_name` / `_gen_meta` as-is. Remove the now-unused `safe_patch` import if
nothing else uses it (run ruff — it flags unused imports).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transformers_hf.py tests/test_capture_hf.py -q`
Expected: PASS — including the existing `test_capture_hf` structural-wrap test (a
`generate` call inside a session still opens a `model` node).

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check netscope tests && .venv/bin/mypy
git add netscope/instrument/transformers_hf.py tests/test_transformers_hf.py
git commit -m "feat(hf): request output_attentions when attention capture is on"
```

---

## Task 7: Renderer — attention overlay, per-head panel, timeline KV column

**Files:**
- Modify: `netscope/web/template.html`
- Then byte-copy to: `extension/media/template.html`
- Test: `tests/test_sinks.py` (append a structural assert), `tests/test_web_sync.py` (already enforces byte-equality)

**Note:** This is vendored JS — not unit-TDD'd line-by-line. The test gate is (a) a
structural assertion that the new overlay exists in the emitted HTML, and (b) the
existing `test_web_sync.py` byte-equality between the two template copies.

- [ ] **Step 1: Write the failing structural test (append to `tests/test_sinks.py`)**

```python
def test_html_has_attention_overlay_control():
    from netscope.core.ir import NVGraph
    g = NVGraph(name="t")
    g.add_node("a", kind="module", name="self_attn",
               meta={"attn_heads": [{"entropy": 1.0, "dist": 2.0, "last": 0.1}]})
    html = g.to_html()
    assert "btn-attention" in html          # the new overlay button id
    assert "attn_heads" in html             # per-head data reaches the webview
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sinks.py::test_html_has_attention_overlay_control -q`
Expected: FAIL — template has no `btn-attention` / doesn't pass `attn_heads`.

- [ ] **Step 3: Edit `netscope/web/template.html`**

Make these concrete edits (mirror the existing role-lens implementation — search for
`btn-role`, `clearRole`, `rolecolor`, and the detail-panel builder near line 637):

1. **Toolbar button** — beside `btn-role` (~line 152):
   `<button class="hbtn" id="btn-attention" title="color attention nodes by a per-head stat (entropy/dist/%last)">⊕ attention</button>`
2. **Pass the data** — wherever node `data` is built for cytoscape (the `toElements`
   equivalent in the template's inline JS), include `attn_heads` and `kv_cache` from
   `node.meta` onto the element `data` (next to where `role`/`meta` are copied).
3. **Overlay handler** — clone the `role` overlay (`rolecolor`) into an `attncolor`:
   on `#btn-attention` click, for each node with `data.attn_heads`, compute the mean
   of the selected stat (default `entropy`) across heads, map cool→hot with the same
   palette helper the cost heatmap uses (`~line 476`), and set `attncolor`. Add a
   `node[attncolor]` style block cloned from the `node[rolecolor]` block (~line 281).
4. **Detail panel** — in the node-tap panel builder (~line 637, where `role`/`branch`
   rows are added), if `d.attn_heads`, append a compact table: one row per head with
   `entropy | dist | %last`.
5. **Timeline KV column** — in the timeline/heatmap section (~line 484, which already
   paints per-step latency), add a `kv_seq` column/label per step when present.

- [ ] **Step 4: Run the structural test**

Run: `.venv/bin/python -m pytest tests/test_sinks.py::test_html_has_attention_overlay_control -q`
Expected: PASS.

- [ ] **Step 5: Byte-copy to the extension + verify sync**

```bash
cp netscope/web/template.html extension/media/template.html
diff netscope/web/template.html extension/media/template.html   # must be empty
.venv/bin/python -m pytest tests/test_web_sync.py -q
```
Expected: PASS (byte-equality holds).

- [ ] **Step 6: Commit**

```bash
git add netscope/web/template.html extension/media/template.html tests/test_sinks.py
git commit -m "feat(render): attention overlay + per-head panel + timeline KV column"
```

---

## Task 8: Docs + changelog

**Files:**
- Modify: `docs/API.md`, `CHANGELOG.md`, `ROADMAP.md`

- [ ] **Step 1: Document the `capture=` API in `docs/API.md`**

Add a short section: `netscope.graph(name, *, profile=False, capture={"attention","kv_cache"})`
— what each flag records, the `NETSCOPE_CAPTURE` env equivalent, the memory note
(attention is reduced to per-head scalars; KV is shapes only), and that the default
is metadata-only. Add `timeline()`'s new `kv_seq` field to its entry.

- [ ] **Step 2: CHANGELOG entry**

Add an `Unreleased`/next-version block: "Deeper LLM views — opt-in `capture=` for
KV-cache shapes (on the generation timeline) and per-head attention stats (entropy /
distance / %last) as a graph overlay. Default trace unchanged (metadata-only)."

- [ ] **Step 3: Mark the ROADMAP item shipped**

In `ROADMAP.md`, mark the "Deeper LLM-specific views" bullet (attention-head maps,
KV-cache shapes) as ✅ / shipped.

- [ ] **Step 4: Commit**

```bash
git add docs/API.md CHANGELOG.md ROADMAP.md
git commit -m "docs: capture= API, KV/attention views, changelog + roadmap"
```

---

## Final verification (run before opening for review)

- [ ] Full suite + coverage gate: `.venv/bin/python -m pytest tests/ -q --cov=netscope --cov-report=term-missing --cov-fail-under=80` → all pass, coverage ≥ 80%.
- [ ] `.venv/bin/ruff check netscope tests` → clean.
- [ ] `.venv/bin/mypy` → 0 issues (the gated code lives in untyped hook bodies — `check_untyped_defs` is on, so watch for new dict-narrowing; annotate locals as `dict[str, object]` if needed, as in `torch_nn.py:233`).
- [ ] `core/` still imports no framework: `grep -rn "import torch" netscope/core/` → none (KV/attention helpers live in `instrument/`, reductions in `enrich/`).
- [ ] Extension: `cd extension && npm run compile && npm run lint && npm run knip && npm run madge && npm run test:unit` → green; `test:headless` if a torch venv is available.
- [ ] Manual smoke (optional): a tiny GPT-2 generate under `capture={"attention","kv_cache"}` → `.show()` the graph, toggle `⊕ attention`, and `netscope.timeline(g)` shows `kv_seq` growing.

---

## Self-review notes (author)

- **Spec coverage:** capture foundation (T1) ✓ · KV shapes (T3+T4) ✓ · KV on timeline (T5) ✓ · attention reductions (T2) ✓ · attention capture wiring (T4) ✓ · attention acquisition / output_attentions (T6) ✓ · rendering overlay+panel+timeline column (T7) ✓ · zero-retention/default-off guards (T1,T4) ✓ · docs (T8) ✓.
- **Type consistency:** `cap.wants(flag)`, `meta["kv_cache"]={layers,shape,seq}`, `meta["attn_heads"]=[{entropy,dist,last}]`, `timeline` field `kv_seq`, button id `btn-attention`, style key `attncolor` — used consistently across tasks.
- **Risk (documented in spec):** attention acquisition is HF-first/best-effort; the `_attention_weights` 4-D-square heuristic + `_AttnModel` test exercise the path without depending on real HF internals.
