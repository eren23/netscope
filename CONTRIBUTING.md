# Contributing to netscope

Thanks for poking at netscope. It's a small, layered codebase — this is the
fast path to a working dev setup and the few rules that keep it coherent.

## Dev setup

```bash
git clone https://github.com/eren23/netscope && cd netscope
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -e ".[dev]"
# netscope shells out to YOUR torch — install it (and transformers for HF):
.venv/bin/pip install torch transformers          # CPU wheels are fine
```

`torch` / `transformers` are deliberately **not** dependencies (heavy, platform-
specific); the test suite and examples need them, so install them in your venv.

## Running the checks

```bash
.venv/bin/python -m pytest tests/                 # the engine + features (real torch fixtures)
cd extension && npm install && npm run compile     # tsc must be clean
npm run test:unit && npm run test:headless         # mergeByLoc + the webview headless mocks
```

Everything should be green (currently **190 passed / 1 skipped** — the skip is the
optional `thop` FLOPs path). The suite uses **real** PyTorch/HF models as fixtures,
not mocks — red→green TDD with a real model is the house style.

## How it fits together

One typed IR (`netscope/core/ir.py`) that every layer speaks; features are
**producers** (runtime trace, static AST, torch.fx) or **consumers** (sinks, diff,
checks, roles, timeline), fused by source `loc`. The renderer
(`netscope/web/template.html`, Cytoscape, libs inlined) is reused verbatim by the
VSCode webview. See the [Architecture](README.md#architecture) section and
[docs/API.md](docs/API.md) for the public surface.

```
netscope/
  core/        IR, capture session, context, merge, checks, diff, timeline
  instrument/  torch + transformers auto-tracing (global forward hooks)
  static/      AST analysis (declared-dim checks) + a torch.fx fallback
  enrich/      params, roles, (optional flops)
  sinks/       html / json / mermaid renderers
  llm/         provider-agnostic assistant, inference, generated views
  mcp/         JSON-RPC-over-stdio server for coding agents
  web/         the shared Cytoscape template + vendored libs
  playground.py   the local "paste a model, see it live" web app
extension/     the TypeScript VSCode/Cursor extension (a thin consumer)
```

## Two sync rules (CI-worthy, please don't break)

1. **The renderer is authored once.** Edit `netscope/web/template.html`, then copy
   it verbatim to `extension/media/template.html` — they must be **byte-identical**:
   ```bash
   cp netscope/web/template.html extension/media/template.html
   diff netscope/web/template.html extension/media/template.html   # must be empty
   ```
2. **The merge logic is mirrored.** `netscope/core/merge.py` (Python) and
   `extension/src/mergeByLoc.ts` (TypeScript) implement the *same* static⇄runtime
   fusion rules. Change one → change the other, and keep their tests in step.

## Conventions

- **Python ≥ 3.9** — start modules with `from __future__ import annotations`.
- **Tracing stays metadata-only and zero-overhead** outside a session — the
  capture-once / no-tensor-retention guarantees are tested (`tests/test_overhead.py`),
  keep them green.
- **Add an example** for a user-facing feature under `examples/` (they double as
  smoke tests and README material).
- **Versioning:** pre-1.0, bump `pyproject.toml` + `extension/package.json`
  together, once per themed release — not per change. See
  [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## Filing issues / PRs

Small, focused PRs with a test are easiest to land. For a behavior change, a
red→green test (real model fixture) makes the intent obvious. Bugs: a minimal
repro snippet (`with netscope.graph(): model(x)`) is gold.
