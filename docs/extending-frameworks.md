# Adding a framework

netscope's tracing is split so that a new framework (JAX/Flax, Keras, …) plugs in
**without touching core**. This doc is the contract + a walkthrough using the
PyTorch adapter (`netscope/instrument/torch_nn.py`) as the reference.

## The shape of the system

```
your model → framework adapter → core.capture → NVGraph (IR) → sinks (HTML/JSON/…)
              (per-framework)      (neutral)      (neutral)       (neutral)
```

Everything to the right of the adapter is framework-neutral:

- **The IR** (`netscope/core/ir.py`, `NVGraph`) is plain dicts — `kind`, `name`,
  `parent`, `loc`, `meta`, `attrs`. No framework type ever reaches it. An adapter
  populates conventional `meta` keys (`in_shape`, `out_shape`, `dtype`, `device`,
  `params`, `param_bytes`, `act_bytes`, `time_ms`) with strings/lists/ints.
- **The sinks** (`netscope/sinks/*`) import no framework.
- **The capture session** (`netscope/core/capture.py`) imports no framework — the
  one inference guard it needs comes back through the registry (see below).

So adding a framework means writing *one adapter module* and registering it.

## The `Instrumentor` contract

Defined in `netscope/core/registry.py`. A session-scoped adapter is any object with:

```python
class Instrumentor(Protocol):
    def on_enter(self) -> object: ...          # install hooks; return an opaque handle
    def on_exit(self, handle, /) -> None: ...   # tear them down using that handle
```

`capture.graph()` calls `on_enter()` when a session opens and `on_exit(handle)`
when it closes — so hooks exist only *during* capture (zero overhead otherwise).
Both run inside `try/except`: a buggy adapter degrades to "no trace from this
framework", it never breaks the user's program.

**Optional** (not in the required Protocol): define `inference_context(self)`
returning a context manager (e.g. `torch.no_grad()`). The isolation re-run wraps
the re-executed module in the combined guard from `registry.inference_context()`.
Omit it if your framework needs none.

## Registration (import-order independent)

Adapters install via a wrapt post-import hook in `netscope/__init__.py`, so
`import netscope` before *or* after the framework both work:

```python
# netscope/__init__.py
wrapt.register_post_import_hook(lambda *_: _install_jax(), "jax")

def _install_jax() -> None:
    try:
        from netscope.instrument import jax_flax
        jax_flax.register()
    except Exception:
        pass
```

Your adapter's `register()` then registers the instance (idempotent per type):

```python
# netscope/instrument/jax_flax.py
from netscope.core import registry

def register() -> None:
    registry.register_session_instrumentor(JaxFlaxInstrumentor())
```

## Inside `on_enter`: emitting the graph

Open a span when a module starts, close it when it ends, via the active capture:

```python
from netscope.core import context as ctx

cap = ctx.active_capture()          # None outside a session — your hooks should no-op
handle = cap.open_span(name, kind="module", meta=meta, loc=loc)  # on start
...
cap.close_span(handle, meta_update={"out_shape": out_shape})     # on end
```

`open_span`/`close_span` (in `capture.py`) manage the parent stack, so nesting is
automatic as long as starts/ends are balanced. The torch adapter does exactly this
from a pre-hook + post-hook pair; see `pre()`/`post()` in `torch_nn.py`.

## Honest notes for non-torch frameworks

The PyTorch adapter leans on features that may not exist elsewhere — budget for these:

- **No global forward hooks.** torch has `register_module_forward_pre_hook`; most
  frameworks don't. In JAX/Flax the closest lever is wrapping `linen.Module.apply`
  (via the `safe_patch` helper in `instrument/base.py`) plus
  `flax.linen.capture_intermediates`.
- **Dataflow edges need stable value identity.** torch tracks producers by
  `id(tensor)` + weakref (`torch_nn.py`); JAX arrays are immutable/abstract under
  JIT, so a first cut may emit containment-only graphs (no dataflow edges). That's
  a valid v1 — shapes and structure still render.
- **Parameters.** `netscope/enrich/params.py` is `nn.Module`-specific
  (`module.parameters()`); a framework where params are a separate pytree needs
  its own small counter (optional — omit `params` meta and the graph still works).
- **The static path is torch-specific.** `netscope/static/*` (declared-dim
  pre-check, `module_loc`, `torch.fx`) keys off `nn.*` and `__init__`. Runtime
  tracing works without it; static fusion + click-to-source would need a parallel
  path.

## Reference files

- `netscope/instrument/torch_nn.py` — the full session-scoped adapter (hooks,
  spans, dataflow, isolation, `inference_context`).
- `netscope/instrument/transformers_hf.py` — a *non*-session adapter: an
  import-time `safe_patch` on `generate`, gated by `is_capturing`.
- `netscope/instrument/base.py` — `safe_patch` / `wrap_callable` for patching any
  callable safely.
- `netscope/core/registry.py` — the `Instrumentor` Protocol + registration.
- `tests/test_registry.py` — exercises the contract with minimal fake adapters.
