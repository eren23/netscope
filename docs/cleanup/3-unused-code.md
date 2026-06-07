# Criterion 3 — Unused Code

Tools run (read-only):
- `npx --yes knip` in `extension/` — exit 1, flagged 5 "unused files", 2 unused exports, 1 unused exported type.
- `.venv/bin/vulture netscope tests --min-confidence 70` — exit 3, 2 hits.
- `npx --yes madge --circular --extensions ts src` — no circular dependencies.

## Verdict

The codebase is very clean. After grepping the whole repo (incl. `tests/`, `examples/`,
`netscope/mcp`, the JS inside `netscope/web/template.html`, `__all__`, and `[project.scripts]`),
**5 of the 7 knip hits are false positives** caused by dynamic file reads and an external
test-runner entry that knip is not configured for. Only **2 genuinely unreferenced TS exports**
remain (`diagnostics.clear`, `disposeShapeDecorations`), plus **1 trivial test-only dead fixture
param** (`capsys`). Total genuinely-removable LOC is tiny (~6 lines), and even those are
deliberate-looking API symmetry / lifecycle helpers, so I rate the two TS exports **medium** (not
"auto-remove safe"), not high. There is effectively nothing here that must be removed.

## Genuinely unused (candidates)

### `diagnostics.clear` — `extension/src/diagnostics.ts:74`
`export function clear(collection, uri)` clears squiggles for a stale file. The `diagnostics`
module is imported namespaced (`import * as diagnostics` at extension.ts:21); only
`diagnostics.makeCollection` (extension.ts:426) and `diagnostics.publish` (extension.ts:432,483,607)
are called. Repo-wide grep for `diagnostics.clear` / `diagCollection.clear` / `.clear(` on the
collection returns **zero** call sites. Not in `deactivate` (extension.ts:615-617, which only does
`panel?.dispose()`).
Confidence: **medium**. It is dead, but it's a coherent public counterpart to `publish` on a public
module; staleness is currently handled by `publish` re-setting the collection, so removal is
behavior-neutral. Left medium because it may be intended public API symmetry, not an accident.

### `disposeShapeDecorations` — `extension/src/shapeDecorations.ts:60`
`export function disposeShapeDecorations()` calls `decoType.dispose()`. Repo-wide grep returns the
definition only — **zero** call sites. The module's other export `refreshShapeDecorations` IS wired
(extension.ts:20 import; called at 430, 584, 594). `deactivate` (extension.ts:615) does not call it.
`decoType` is a module-singleton `TextEditorDecorationType` that VSCode reclaims on extension
unload, so leaving it undisposed is harmless in practice.
Confidence: **medium**. Genuinely uncalled, but it is the kind of cleanup hook that arguably *should*
be registered in `ctx.subscriptions`/`deactivate` rather than deleted. So this is as much a
"missing wiring" smell as a "dead code" one — flag, don't auto-delete.

### `capsys` unused fixture — `tests/test_llm_robustness.py:138`
`def test_cli_reports_missing_node_clearly(tmp_path, capsys)` — the body (lines 139-147) never
references `capsys`. Vulture reports it at 100% confidence. It is a no-op pytest fixture param;
dropping it is safe and behavior-neutral.
Confidence: **medium**. Trivial, test-only, zero functional impact. Not "high" because it is so
minor it isn't worth a churn commit on its own; bundle it if touching the file.

## Unused exported type (keep)

### `NVWarning` — `extension/src/ir.ts:30`
Knip flags it as an "unused exported type" because no *other module* imports it. But it **is** used
internally: `NVGraph.warnings?: NVWarning[]` (ir.ts:44). `ir.ts` is the intentional cross-language
mirror of `netscope/core/ir.py` (the graph emits `"warnings": detect_mismatches(self)` —
ir.py:103). Removing/inlining it would break the producer/consumer schema parity that is explicitly
out of scope. Note `diagnostics.ts:22` defines a near-duplicate local `interface Warning` instead of
importing `NVWarning`; that duplication is a possible (low-value) DRY tidy, NOT a deletion. **Keep
NVWarning as-is.**

## FALSE POSITIVES — DO NOT REMOVE

### 4 vendored JS files (knip "unused files")
- `extension/media/vendor/cytoscape.min.js`
- `extension/media/vendor/dagre.min.js`
- `extension/media/vendor/cytoscape-dagre.min.js`
- `extension/media/vendor/cytoscape-expand-collapse.min.js`

These are **load-bearing**. They are read at runtime via `fs.readFileSync` in `vendorScripts()`
(extension.ts:311-318), driven by the `VENDOR_LIBS` string-literal list (extension.ts:304-309), and
inlined into the webview replacing the `__NETSCOPE_VENDOR__` placeholder (extension.ts:399). Knip
sees no `import`/`require` so it calls them orphans, but the comment at extension.ts:300-303 is
explicit: "without this the `__NETSCOPE_VENDOR__` placeholder ships unreplaced and the graph panel
is blank." Removing them breaks the graph view. **Keep all four.**

### `src/test/mergeByLoc.test.ts` (knip "unused file")
False positive. It is compiled to `out/test/mergeByLoc.test.js` and executed by the `test:unit` npm
script (`package.json:119`: `node ./out/test/mergeByLoc.test.js`). The file is a self-contained
node test (no VSCode host) that exercises `mergeByLoc` + `render`. Knip has no config wiring this as
an entry, hence the false flag. **Keep.**

### `exc` in `__exit__(self, *exc)` — `netscope/hints/api.py:49`
False positive (vulture 100%). `*exc` is part of the Python context-manager protocol
(`__exit__(self, exc_type, exc_val, exc_tb)`); the dunder signature is required even though the
values are intentionally ignored (the span is closed in `finally`-style regardless of exception).
Removing the parameter would break `with nv.stage(...)`. **Keep.**

## Suggested config follow-up (optional, not a code change)
Adding a small `knip.json` that declares `out/test/*.js` (or the `.ts` sources) as entries and
treats `media/vendor/*.js` as ignored assets would silence the 5 false positives and make future
knip runs trustworthy. Out of scope for a read-only pass; noted for maintainers.
