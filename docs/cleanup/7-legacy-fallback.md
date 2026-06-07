# Criterion 7 — Deprecated / legacy / fallback code; make code paths singular

**Branch:** `cleanup/code-quality`  ·  **Scope:** READ-ONLY assessment
**Verdict:** The codebase is in good shape. There is a small, well-bounded set of
genuinely-dead version-compat paths, plus one large-but-cosmetic redundancy
(`from __future__ import annotations`). None of it is a correctness problem; this
is housekeeping that makes code paths singular against the *declared* minimums.

---

## Declared minimums (evidence from `pyproject.toml`)

- `requires-python = ">=3.10"` (line 9), classifiers list 3.10/3.11/3.12 (line 16).
- `torch = ["torch>=2.0"]` (line 22).
- `hf = ["transformers>=5"]` (line 24), with an inline comment "transformers v5
  (Python 3.10+)" (line 23).

Runtime in this env (informational, confirms the APIs exist today): torch 2.12.0,
transformers 5.10.2.

---

## Finding 1 — redundant `from __future__ import annotations` (38 netscope files)

PEP 563 (`from __future__ import annotations`) makes all annotations lazy strings.
It is a no-op on the *behavior* of this package because every annotated name is
either a builtin or already imported at runtime — there is no annotation that
needs string-deferral to avoid a runtime NameError. Under `requires-python>=3.10`
the import is purely cosmetic: nothing in netscope uses 3.12-only generic syntax
that would still require it, and forward-refs that *are* used are written as
explicit string literals anyway (e.g. `netscope/__init__.py:60` `def diff(before:
"NVGraph", ...)` — and `NVGraph` is in fact imported at line 18, so even those
quotes are belt-and-suspenders).

`grep -rl 'from __future__ import annotations' --include='*.py' netscope/` → **38
files** (matches the brief exactly). Full list:

```
netscope/__init__.py
netscope/__main__.py
netscope/core/capture.py
netscope/core/checks.py
netscope/core/context.py
netscope/core/diff.py
netscope/core/ir.py
netscope/core/merge.py
netscope/core/registry.py
netscope/core/stage_flow.py
netscope/core/timeline.py
netscope/enrich/flops.py
netscope/enrich/params.py
netscope/enrich/roles.py
netscope/hints/api.py
netscope/instrument/base.py
netscope/instrument/torch_nn.py
netscope/instrument/transformers_hf.py
netscope/llm/__init__.py
netscope/llm/__main__.py
netscope/llm/infer.py
netscope/llm/prompts.py
netscope/llm/provider.py
netscope/llm/views.py
netscope/mcp/__init__.py
netscope/mcp/__main__.py
netscope/mcp/server.py
netscope/playground.py
netscope/sinks/file_sink.py
netscope/sinks/html_sink.py
netscope/sinks/json_sink.py
netscope/sinks/mermaid_sink.py
netscope/static/__main__.py
netscope/static/ast_producer.py
netscope/static/cli.py
netscope/static/dims.py
netscope/static/fx_trace.py
netscope/static/module_loc.py
```

(Whole-repo count incl. tests/examples/scripts is 87: tests 34, examples 14,
scripts 1, netscope 38. The brief's "38" is the netscope package alone.)

**Confidence nuance.** This is a real redundancy but it is *purely stylistic*. It
does not make a code path "non-singular" in any behavioral sense — it changes
nothing at runtime. Two honest caveats keep this out of "trivially auto-remove":

1. Removing the future import would force a paired modernization of the many
   `typing.Optional[...]`, `typing.Dict[...]`, `typing.List[...]`, `typing.Tuple`
   annotations to PEP 604 / PEP 585 builtins, OR they keep working as-is (they do
   — they are real imported names, not deferred forms). So removal is *safe on its
   own*, but a reviewer may want it bundled with the `ruff --select UP006,UP007,
   UP037` modernization (ruff reports **42** such fixable findings in netscope/),
   to avoid a mixed old/new annotation style.
2. `ruff --select FA` (the flake8-future-annotations rule family, FA100/FA102)
   reports **"All checks passed!"** — i.e. the linter does *not* flag these as
   removable. The rule that *would* auto-remove a useless future import is not in
   ruff's stable set, so this is a deliberate manual call, not a tool mandate.

Given (1) and (2), I rate the bulk removal **medium** confidence as a standalone
change, and recommend it be done as part of an annotation-modernization pass
rather than a blind 38-file delete. It is correct to leave it entirely if the
project prefers PEP 563 style consistency.

---

## Finding 2 — dead `except TypeError` fallback for `with_kwargs` (HIGH confidence)

`netscope/instrument/torch_nn.py:185-197`, function `_attach_isolation_capture`:

```python
    try:
        return module.register_forward_pre_hook(grab, with_kwargs=True)
    except TypeError:
        # very old torch without with_kwargs: positional-only (kwargs lost, but
        # modules whose kwargs have defaults still re-run fine).
        def grab_pos(mod, args):
            ...
        return module.register_forward_pre_hook(grab_pos)
```

`nn.Module.register_forward_pre_hook` gained the `with_kwargs` keyword in **torch
2.0** (Module-level hooks support for kwargs landed in the 2.0 release). The
declared floor is `torch>=2.0` (pyproject:22), so under the *minimum supported*
torch the `try` branch always succeeds and the `except TypeError` branch is
**unreachable** — a dead pre-2.0 compat path. No test exercises it (the only
`with_kwargs` test, `tests/test_isolate.py:61 test_isolate_handles_kwargs`,
asserts the *happy* path; nothing monkeypatches torch to raise `TypeError`).

**Recommendation (high):** collapse to the single supported call:

```python
    return module.register_forward_pre_hook(grab, with_kwargs=True)
```

and drop the `grab_pos` fallback + its comment. This makes the code path singular
against `torch>=2.0`. The in-line comment at lines 171-174 ("Per-module hooks
support `with_kwargs` across torch versions") stays accurate.

---

## Finding 3 — dead `except TypeError` fallback for `always_call` (HIGH confidence)

`netscope/instrument/torch_nn.py:330-333`, `TorchForwardInstrumentor.on_enter`:

```python
        try:
            post_handle = register_module_forward_hook(post, always_call=True)
        except TypeError:
            post_handle = register_module_forward_hook(post)   # older torch: no always_call
```

`register_module_forward_hook`'s `always_call` keyword also landed in **torch
2.0** (same hook-API overhaul). Confirmed against the installed API:
`inspect.signature(register_module_forward_hook)` →
`(hook, *, with_kwargs=False, always_call=False)`. Under `torch>=2.0` the `try`
branch always succeeds; the `except TypeError` branch is dead pre-2.0 compat.

`always_call=True` is load-bearing for correctness (the comment at 327-329
explains it: the post-hook must fire on a raising forward so the span stack
unwinds; `tests/test_session_safety.py:53` documents the same). So the *kwarg*
must stay; only the fallback that drops it is dead.

**Recommendation (high):** collapse to:

```python
        post_handle = register_module_forward_hook(post, always_call=True)
```

and delete the `except TypeError` line. Singular path against `torch>=2.0`.

---

## Finding 4 — transformers v<5 paths: none found (clean)

The HF instrumentor (`netscope/instrument/transformers_hf.py`) targets
`"transformers.generation.utils"` / `"GenerationMixin.generate"` (lines 28-29).
That module path and the `GenerationMixin.generate` location are valid on
transformers v5 (and were already valid on v4). I searched the whole package for
version-branching and for the legacy pre-v4 module path:

- `grep -rn 'transformers.__version__|version.parse|packaging'` netscope/ → **none**.
- `grep -rn 'generation_utils|generation.utils|GenerationMixin'` → only the single
  v5-correct reference above.
- No `try: import transformers.generation_utils except: ...` legacy-path branch.

So there is **no transformers-v<5 fallback to remove** — the consumer is already
singular. The patch is done through `safe_patch` (`netscope/instrument/base.py:56`),
whose broad `try/except` is intentional load-bearing tracer-safety (it must never
break a user's `generate()` call), NOT a version fallback. Leave it.

---

## Things that look like "fallback" but are NOT version/legacy — do not touch

- `netscope/static/fx_trace.py` docstring "torch.fx **fallback** producer": an
  *architectural* fallback (static structure when no forward runs), not a version
  compat path.
- `netscope/static/dims.py:239` comment "...a layer in an unused **fallback** class
  that never executed": describes the *user's* model code, not netscope's.
- The many `try/except Exception` blocks in `torch_nn.py` (`_is_tensor`, `_dtype`,
  `_iter_tensors`, `_freeze`, weakref `except TypeError` at 323) and in
  `base.py`/`__init__.py` guard introspection of arbitrary user models / optional
  deps — load-bearing per the established grounding, not removable.
- The `weakref.ref(t)` `except TypeError` at `torch_nn.py:323` is NOT a version
  fallback — it handles tensor subclasses that don't support weakref. Keep.

---

## Out-of-scope lint noise observed (reported, not part of criterion 7)

While running ruff I incidentally saw (NOT legacy/fallback, listed only so they
aren't mistaken for in-scope): `F401 typing.List imported but unused`
(`netscope/llm/views.py:20`); two `E702` semicolon statements in
`netscope/playground.py:101,112`. These belong to other cleanup criteria.

---

## Summary table

| # | What | Where | Confidence |
|---|------|-------|-----------|
| 2 | Dead `except TypeError` for `with_kwargs` (pre-torch-2.0) | torch_nn.py:185-197 | high |
| 3 | Dead `except TypeError` for `always_call` (pre-torch-2.0) | torch_nn.py:330-333 | high |
| 1 | Redundant `from __future__ import annotations` ×38 | netscope/ | medium |
| 4 | transformers v<5 path | (none found) | n/a |

High-confidence removable items: **2** (Findings 2 and 3).
