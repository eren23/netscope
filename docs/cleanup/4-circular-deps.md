# Criterion 4 — Untangle circular dependencies

**Verdict: clean DAG, zero cycles on both sides of the language boundary.** Nothing to untangle.
The codebase already practices the discipline this criterion is meant to enforce (lazy imports in
`core/ir.py` exist *specifically* to keep the core decoupled from rendering — see below). The only
actionable work is preventive: turn the manual checks I ran here into permanent CI gates so a future
edit cannot silently introduce a cycle.

---

## 1. TypeScript extension (`extension/src`)

Tool: `npx --yes madge --circular --extensions ts src` (run inside `extension/`).

```
Processed 8 files (292ms)
✔ No circular dependency found!
```

Full dependency graph (`madge --json`):

| module | imports (intra-`src`) |
|---|---|
| `ir.ts` | *(none — leaf / foundation)* |
| `mergeByLoc.ts` | `ir.ts` |
| `render.ts` | `ir.ts` |
| `traceStore.ts` | `ir.ts` |
| `diagnostics.ts` | `ir.ts` |
| `shapeDecorations.ts` | `ir.ts`, `traceStore.ts` |
| `extension.ts` | `diagnostics`, `ir`, `mergeByLoc`, `render`, `shapeDecorations`, `traceStore` |
| `test/mergeByLoc.test.ts` | `ir`, `mergeByLoc`, `render` |

This is a textbook layered DAG: `ir.ts` is the single foundation (zero outgoing edges),
`extension.ts` is the single root (the VS Code activation entry, `extension/src/extension.ts:12-22`),
and every other module sits in between depending only downward. There is exactly one shared internal
edge that is not directly to `ir` (`shapeDecorations.ts:10 -> traceStore.ts`), and it does not create
a cycle.

---

## 2. Python engine (`netscope/`)

No off-the-shelf run was needed for the verdict, but I derived the full intra-package import graph two
ways and cross-checked them:

1. **Grep** of every `from netscope` / `import netscope` / relative-import line (62 lines).
2. A **read-only AST script** (`ast.parse` per file, resolving relative imports, distinguishing
   module-scope imports from function/method-local ones). 44 modules, 60 edges.

Cycle detection (DFS, gray/black coloring) on **both** the full graph (including function-local
imports) and the module-scope-only graph:

```
=== CYCLES (full graph) ===                          none
=== CYCLES (top-level/module-scope imports only) ===  none
```

### Derived layering (the de-facto contract, bottom → top)

```
L0 foundation   core.context · core.registry · core.ir          (no intra-pkg deps at module load)
L1 model ops    core.checks · core.merge · core.diff · core.stage_flow · core.timeline
                enrich.{roles,params,flops} · hints.api
L2 producers    instrument.{base,torch_nn,transformers_hf} · static.{ast_producer,fx_trace,
                module_loc,dims,cli} · llm.{provider,prompts,infer,views}
L3 sinks        sinks.{json_sink,mermaid_sink,html_sink,file_sink}     (consume the IR)
L4 facade       netscope/__init__.py   (re-exports core + hints; lazily loads instrument)
L5 entry points playground · __main__ · static.__main__ · llm.__main__ · mcp.{server,__main__}
```

Imports flow strictly toward lower layers. The single edge that looks like it climbs —
`netscope.playground -> netscope` (`netscope/playground.py:25`) — is an L5 entry point importing the
L4 facade, i.e. still downward; the facade never imports `playground`, so no back-edge exists.

### The deliberate cycle-prevention already in place (do NOT "simplify" it)

`core.ir` is the universal data model — almost everything depends on it. It would naturally form a
cycle with the **sinks** (rendering reads the IR; the IR offers `.to_html()/.to_json()/.show()`
convenience methods). That cycle is avoided on purpose by **function-local imports**, and the code
says so:

- `netscope/core/ir.py:131` — comment: *"sinks (lazy imports keep core decoupled from rendering)"*
- `netscope/core/ir.py:96` `from netscope.core.checks import detect_mismatches`
- `netscope/core/ir.py:133` `from netscope.sinks.json_sink import to_json`
- `netscope/core/ir.py:138` `from netscope.sinks.mermaid_sink import to_mermaid`
- `netscope/core/ir.py:143,149` `from netscope.sinks.html_sink import {to_html,show}`

`sinks.html_sink` in turn imports `core.checks` (`netscope/sinks/html_sink.py:54`) and `enrich.roles`
(`:15`). Because `core.ir`'s references to sinks/checks are method-local, the **module-load** graph
stays acyclic even though the logical call graph closes the loop at runtime. This is exactly the right
pattern; flag it as load-bearing, not as smell.

Same pattern (intentional, load-bearing) at:
- `netscope/__init__.py:41,50` — lazy `instrument.{torch_nn,transformers_hf}` (also avoids importing
  torch unless the user actually traces), `:78` lazy `enrich.roles`.
- `netscope/core/capture.py:116-117` — lazy `core.stage_flow`, `sinks.file_sink`.
- `netscope/static/ast_producer.py:125`, `netscope/static/fx_trace.py:25,70`,
  `netscope/instrument/torch_nn.py:44`, `netscope/mcp/server.py:163,226`, `netscope/llm/views.py:234`.

---

## 3. Recommendations

### R1 (high) — Add `madge --circular` as a CI gate for the extension
The check is already green and zero-config. Make it permanent so a future `import` can't quietly
close a loop. Add a step in the `extension` job of `.github/workflows/test.yml` (after `npm install`,
before/after `npm run compile`):

```yaml
      - name: Assert no circular imports (TS)
        working-directory: extension
        run: npx --yes madge --circular --extensions ts src
```

`madge` exits non-zero when a cycle is found, so no extra scripting is needed. Optionally add
`"madge": "^8"` to `extension/package.json` devDependencies and an npm script
`"lint:cycles": "madge --circular --extensions ts src"` to drop the `--yes` network fetch and pin the
version (cleaner, but the inline `npx --yes` form already works in CI).

### R2 (high) — Add an `import-linter` contract as a CI gate for the Python engine
`import-linter` enforces a *layering* contract, which is stronger than "no cycles" and documents the
intended architecture. It is not currently a dependency (confirmed: no match in `pyproject.toml`).
Add `import-linter` to the `[project.optional-dependencies] dev` group and a `[tool.importlinter]`
config (in `pyproject.toml` or `.importlinter`), then run `lint-imports` in the `python` job of
`.github/workflows/test.yml`.

A minimal, accurate contract for the layers derived above:

```toml
[tool.importlinter]
root_package = "netscope"

[[tool.importlinter.contracts]]
name = "No import cycles anywhere"
type = "independence"  # use the "layers" type below for the real contract; this name is illustrative

[[tool.importlinter.contracts]]
name = "netscope layering"
type = "layers"
layers = [
    "netscope.playground | netscope.__main__",   # entry points (top)
    "netscope.mcp | netscope.static | netscope.llm | netscope.instrument | netscope.sinks",
    "netscope.enrich | netscope.hints",
    "netscope.core",                              # foundation (bottom)
]
# import-linter understands the function-local sink imports in core.ir as real edges, so the
# core <- sinks relationship must be expressed. Either keep sinks ABOVE core (as above) — which holds
# because core only reaches sinks via lazy method-local imports that import-linter still records — or,
# if that proves too strict, drop to a pure cycle check:

[[tool.importlinter.contracts]]
name = "No cycles"
type = "layers"
# (a single all-modules layers contract, or use the dedicated forbidden/independence contracts)
```

Caveat (why this is high-confidence on intent but the *exact contract* needs one local `lint-imports`
run to settle): import-linter, unlike my module-scope analysis, counts function-local imports as
edges. The `core.ir -> sinks` lazy imports therefore appear as real edges to it. The `layers` ordering
above (sinks ABOVE core) accommodates that; but if the maintainer prefers to assert "core never
imports sinks," they would need import-linter's `ignore_imports` to exclude the five lazy lines in
`core/ir.py`. The implementer should run `lint-imports` once locally and pick whichever framing they
want before committing — hence I do not mark the precise TOML as auto-implementable.

### R3 (medium) — Commit the read-only cycle-check script (or fold it into tests)
The AST script I used (no third-party dep, pure stdlib) could live as `tests/test_no_import_cycles.py`
so the guarantee travels with the repo even if `import-linter` is declined. Lower priority than R2
because `import-linter` is the standard tool and also covers layering; listing both would be
redundant. Marked medium because it adds a maintained file for a check R2 already provides.

### R4 (low / documentation) — Record the layering contract + the lazy-import rule in a doc
Add a short "Architecture / import layers" note (e.g. in `README` or `docs/`) stating the L0→L5 layer
order and the rule *"`core` must not import sinks/instrument/enrich at module scope; use method-local
imports (see `core/ir.py:131`)."* This makes the contract discoverable for contributors and explains
why those lazy imports must stay. Low confidence only because it is purely additive prose, not a
mechanical fix.

---

## Appendix — commands used (all read-only)
- `npx --yes madge --circular --extensions ts src` (and `--json`) in `extension/`.
- `grep -rn -E "from netscope|import netscope|from \.|..." netscope --include='*.py'`.
- Read-only AST cycle finder run with `.venv/bin/python` (DFS over the resolved import graph; checked
  both full and module-scope-only edge sets).
- Inspected `.github/workflows/test.yml`, `netscope/core/ir.py`, and the extension `import` lines.
- No source or config file was modified.
