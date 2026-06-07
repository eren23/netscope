# Criterion 6 — Defensive code (try/except, try/catch)

## Verdict

The defensive code is **almost entirely load-bearing and already in good shape.**
Inventory: 72 `except` blocks in `netscope/` (Python), 15 `catch` blocks in
`extension/src/extension.ts`, and 9 `try/catch` in the `template.html` webview JS.

- **Zero bare `except:`** clauses exist (`grep 'except\s*:'` → 0). Every block names
  at least `Exception` or a more specific type.
- **No error-hiding with a meaningful, recoverable error and no fallback.** Every
  `except … : pass` / `: return None` / `: return {}` either (a) guards introspection
  of an arbitrary user model — the tracer's untrusted input — or (b) guards an
  optional dependency / cosmetic side effect / best-effort file write that is
  *documented* as best-effort.
- `ruff --select BLE,E722,TRY,S110,S112` reports 75 hits, but **all of the BLE001
  (blind-except) and S110 (try/except/pass) hits map 1:1 to category-A guards** below.
  They are correct false positives for this codebase's purpose.

Recommendation count: **0 high-confidence removals.** Two low-confidence,
optional observability nudges only (NOT to be auto-implemented).

## Tool output (captured, read back)

`ruff check netscope/ --select BLE,E722,TRY,S110,S112` → 75 errors, breakdown:
- `BLE001` blind-except: 40 hits. Every one is a model/optional-dep guard (e.g.
  `torch_nn.py` ×11, `enrich/params.py` ×3, `enrich/flops.py` ×2, `fx_trace.py` ×4,
  `mcp/server.py` ×6, `llm/*` graceful-degrade ×5).
- `S110` try/except/pass: 11 hits (`__init__.py` 44/53, `capture.py` 149/155,
  `registry.py` 35/44, `base.py` 43, `torch_nn.py` 182/194/229, `file_sink.py` 22,
  `fx_trace.py` 76, `playground.py` 137). All deliberate best-effort/guards.
- `E722` bare-except: **0 hits.**
- `S112` try/except/continue: 0 hits.
- `TRY300`/`TRY003` hits are stylistic (move-to-else / long-message), not
  error-hiding; out of scope for this criterion and not worth churning.

## Classification (every block)

### Category A — LOAD-BEARING (KEEP). Representative groups:

**Arbitrary-user-model introspection guards (the tracer's core job; user models ARE
the untrusted input):**
- `netscope/instrument/torch_nn.py` — `_is_tensor` 66, `_dtype` 80, `_device` 90,
  `_act_bytes` 102, `_first_tensor`/`_iter_tensors` 136, `_freeze` 150,
  `_root_qualname_locs` 54/57, isolation grab 182/194, `named_modules()` scan 229,
  weakref of tensor subclass 323, `register_*_hook` `TypeError` shims 187/332.
  Each reads a property off / hooks into an arbitrary `nn.Module` that may override
  dunders, lazy-init, or be a custom tensor subclass. Failing must degrade the
  one node's metadata, never break the user's forward pass.
- `netscope/instrument/base.py:35` & `43` (`span_wrapper`) — MLflow `safe_patch`
  rule, stated in the module docstring: tracing NEVER breaks the wrapped call.
  `base.py:65` (`safe_patch`) returns False on patch failure by contract.
- `netscope/enrich/params.py` 14/25/32 — `module.parameters()` numel; `enrich/flops.py`
  19/40 — optional `thop` import + profile. Both documented best-effort, return
  0/None/False.
- `netscope/static/fx_trace.py` 29/38/42/47/76 — `torch.fx.symbolic_trace` famously
  fails on dynamic control flow; `None`/skip lets the caller fall back to runtime.
- `netscope/hints/api.py:83` (`_fn_loc`) — `inspect.getsourcefile` on a user fn.
- `netscope/core/checks.py:19` (`_shape`), `static/dims.py:207`,
  `static/module_loc.py:57/86` — `int()` coercion / `ast.parse` / file read of
  arbitrary user source; specific exceptions (`TypeError/ValueError`, `SyntaxError`,
  `OSError/UnicodeDecodeError`) — already narrow.

**Optional-dependency / framework import guards:**
- `netscope/__init__.py:33` (`import wrapt`) — degrades to no instrumentation.
- `netscope/__init__.py:44` & `53` (`_install_torch`/`_install_transformers`) —
  post-import hooks must never break `import torch`/`transformers`. (See REC-1 — the
  only block where surfacing *might* help debugging; low confidence.)

**Subprocess JSON.parse / file cleanup / best-effort writes:**
- `extension/src/extension.ts` — `JSON.parse(stdout)` of subprocess output
  (101, 115, 276, 518), `fs.unlinkSync` temp cleanup (70, 143, 171, 252, 278, 333,
  336, 521), `child.kill` 70, `fs.writeFileSync` persist 337. All guard untrusted
  subprocess output or non-critical FS ops. The `r.code !== 0` checks already surface
  the *real* failure with `explainFailure(r)` BEFORE these fallbacks — so the catch
  is for the residual "exited 0 but emitted garbage" case, which legitimately has no
  better recovery than null.
- `netscope/sinks/file_sink.py:22`, `netscope/core/capture.py:149/155` (isolation
  re-run + dump), `netscope/playground.py:137` (`webbrowser.open` cosmetic) — all
  documented best-effort, must not break the user's program.

**Graceful LLM degrade (optional, gated feature):**
- `netscope/llm/views.py:208` & `infer.py:163` — `except Exception: return {"ops":[]}` /
  `return graph` after `provider.complete()`. `complete()` raises `RuntimeError` on
  all known failures, but a custom `_transport` or odd network condition could raise
  other types; the docstrings explicitly promise "empty/no-op if no key or bad reply."
  This is graceful degradation of an optional feature, NOT error-hiding.
- `netscope/llm/provider.py` 110/114/121/128/132 — well-structured: narrow HTTPError /
  URLError/TimeoutError/OSError / KeyError/IndexError/TypeError handling that RE-RAISES
  as a `RuntimeError` with context. Exemplary, not a swallow. (114 reads error-body
  detail best-effort, falling back to `str(e)`.)
- `netscope/llm/views.py:85/91`, `infer.py:77/97/104`, `mcp/__main__.py:25` — narrow
  `json.JSONDecodeError`/`TypeError`/`ValueError` parsing untrusted LLM/RPC text with
  a defined fallback. Already specific.

**Surface-the-error blocks (the right pattern, KEEP):**
- `netscope/playground.py:122`, `static/cli.py:26`, `llm/__main__.py:48`,
  `mcp/server.py` 165/172/202/221/239/282 — catch broad `Exception` at a process /
  RPC / HTTP boundary and RETURN the error text (`f"{type(e).__name__}: {e}"`,
  `is_error=True`, non-zero exit). These are the opposite of error-hiding: they
  convert an arbitrary failure into a clean, surfaced message at the outer boundary,
  exactly as a server/CLI entrypoint should. Comments at server.py:165 ("never crash
  the server on a bad file") document the intent.
- `netscope/instrument/torch_nn.py:345` — already does the right thing: `warnings.warn`
  on a hook that won't remove (the one place surfacing matters, already surfaced).

### Category B — genuinely pointless / error-hiding with no reason

**None found.** No block swallows a meaningful, actionable error with no fallback and
no documented reason.

## Recommendations (ranked)

### REC-1 (LOW) — optionally gate a debug warning behind `NETSCOPE_DEBUG` in the install guards
`netscope/__init__.py:44` and `:53` (`_install_torch` / `_install_transformers`)
swallow any failure to register instrumentation. If `torch_nn.register()` ever fails,
the user gets a graph with no runtime nodes and no hint why. This is the single block
whose silent failure has a real (if rare) user-visible consequence. The silence is
deliberate (must never break `import torch`), so removing the guard is wrong. A
*non-default* nudge would be: emit a `warnings.warn` only when an env flag
(e.g. `NETSCOPE_DEBUG`) is set, leaving the default path silent. LOW confidence
because there is currently no `NETSCOPE_DEBUG`/logging facility (grep → none), this
adds a new convention, and the failure is genuinely rare; it is behavior-additive, not
a cleanup. Do NOT auto-implement.

### REC-2 (LOW) — optionally narrow `views.py:208` / `infer.py:163` to `RuntimeError`
`provider.complete()` is documented to raise `RuntimeError` after exhausting retries.
The two LLM call sites catch broad `Exception` and degrade to a no-op. Narrowing to
`except RuntimeError` would make the contract explicit AND let a genuine programming
bug (e.g. a `TypeError` from a bad `messages` shape) propagate instead of silently
yielding an empty spec. LOW confidence: a custom `_transport` injected in tests/embeds
can raise non-`RuntimeError` types, and the public docstrings promise a no-op on any
bad reply — so the broad catch is arguably correct as-is. Behavior-changing; do NOT
auto-implement. Flagged only for completeness.

## What NOT to touch (explicit)
- Do not remove or narrow any `torch_nn.py` / `enrich/*` / `fx_trace.py` /
  `static/*` / `base.py` / `capture.py` model-introspection or fx guards.
- Do not remove the `extension.ts` `JSON.parse`/`unlinkSync`/`kill` catches —
  subprocess-output + temp-file cleanup, with the real error already surfaced upstream.
- Do not touch `provider.py` (exemplary narrow+re-raise), the surface-the-error
  boundary blocks (`mcp/server.py`, `cli.py`, `playground.py`, `__main__.py`), or the
  `template.html` cytoscape/dagre `catch(_){}` blocks (third-party layout-lib calls
  that legitimately may not be loaded; fallback is `cy.fit`).
