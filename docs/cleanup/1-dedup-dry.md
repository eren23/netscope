# Criterion 1 — Deduplicate / DRY

Scope: find duplicated logic and apply DRY *only* where it reduces complexity.
Read-only assessment. file:line references are to the state of branch
`cleanup/code-quality`.

## Verdict

The codebase is already DRY at the level that matters. There is exactly **one
confirmed exact duplicate** — the committed copy of the entire `netscope/web/`
asset tree under `extension/media/` — and it is *structurally required* for
extension packaging, so the fix is a guard (sync-check / CI byte-equality), not a
merge. Within each language the only repetition with real payoff is the
**temp-file + subprocess + parse + cleanup boilerplate in `extension.ts`**
(5 near-identical sites). Everything else I checked is either an intentional
cross-language mirror (out of scope, must not merge), a same-named-but-different
helper, or a one-line idiom whose extraction would *add* indirection without
cutting complexity. Net: 1 high-confidence guard, 1 medium-confidence
in-language refactor, the rest explicitly *not recommended*.

## What I verified (tool output)

- `md5` of the two templates is identical: `f67737c3bd71e8783089da9ebb60a7bd`,
  773 lines each (`extension/media/template.html` ==
  `netscope/web/template.html`).
- The duplication is the **whole web tree**, not just the template. All four
  vendored libs also match byte-for-byte:
  `media/vendor/{cytoscape.min.js,dagre.min.js,cytoscape-dagre.min.js,cytoscape-expand-collapse.min.js}`
  == `netscope/web/vendor/*`.
- Both copies are git-tracked (`git ls-files` lists both); `media/` is NOT
  gitignored. `extension/.vscodeignore` explicitly *keeps* `media/` ("template +
  vendored cytoscape") because the packaged VSIX has no sibling `netscope/`.
- The extension resolves the web root at runtime
  (`extension/src/extension.ts:289` `webDir`): prefers the sibling
  `../netscope/web` in dev (F5 from repo), falls back to `ctx.extensionPath/media`
  when packaged. So the `media/` copy is the *shipped* artifact; `netscope/web` is
  the source of truth. **Neither can simply be deleted.**
- There is **no** existing test or build step asserting the two trees stay in
  sync (`grep` of `tests/`, `.github/`, `extension/test/`, `extension/scripts`
  found nothing; `extension/package.json` scripts are only
  compile/watch/test:unit/test:headless — no copy/sync step).
- `madge --circular`: "No circular dependency found" (8 TS files) — no structural
  duplication via cycles.
- `knip` flagged `media/vendor/*.js` as "unused files" — **false positive**, they
  are read at runtime by `vendorScripts()` (`extension.ts:311`) via the
  `VENDOR_LIBS` list (`extension.ts:304`). Not a dedup signal.
- The temp-file pattern `path.join(os.tmpdir(), netscope-<kind>-${process.pid}-${Date.now()}.json)`
  occurs at `extension.ts:121, 150, 240, 269, 508` (5 sites).

## Cross-language mirrors — confirmed, NOT recommended for merge

These are intentional producer/consumer parity and the grounding rules forbid
merging them. I verified they are genuine mirrors (same intent, different
language), not accidental drift:

- `extension/src/render.ts:toElements` (label/role/edge construction) <->
  `netscope/sinks/html_sink.py:to_cytoscape`. The TS `nodeRole`/`ROLE_KEYS`
  (`render.ts:14-30`) mirrors `netscope/enrich/roles.py:node_role`/`_ROLE_KEYS`.
  The TS `label()` (`render.ts:7`) mirrors `html_sink.py:_node_label` (`:45`).
- `extension/src/mergeByLoc.ts` <-> `netscope/core/merge.py`;
  `extension/src/ir.ts` <-> `netscope/core/ir.py`.
- `applyViewSpec` JS in `netscope/web/template.html` <-> `netscope/llm/views.py`.

Do not touch these.

---

## Ranked recommendations

### REC-1 (HIGH) — Replace the committed `extension/media` web-tree copy with a sync-check / CI byte-equality guard

- **Files:** `extension/media/template.html` (== `netscope/web/template.html`,
  md5 `f67737c3bd71e8783089da9ebb60a7bd`); `extension/media/vendor/*.js` (==
  `netscope/web/vendor/*.js`, all 4 byte-identical);
  consumer `extension/src/extension.ts:289` (`webDir`), `:296` (`templatePath`),
  `:304-318` (`VENDOR_LIBS`/`vendorScripts`); `extension/.vscodeignore`
  (keeps `media/`); `extension/package.json:116-121` (scripts).
- **Rationale:** Five files are committed twice with zero drift today, and there
  is no mechanism preventing future drift. Because the renderer is "written once"
  and reused verbatim (per `html_sink.py:5-6` and `extension.ts:3-4`), a silent
  divergence between the two copies would make the VSCode webview render
  differently from the standalone HTML — exactly the bug this single-source design
  is meant to prevent. The copy cannot be deleted outright: the packaged VSIX has
  no sibling `netscope/` so `webDir` falls back to `media/`.
- **Proposed change (single source of truth):** keep `netscope/web/` as the one
  source. Make `extension/media/` a *generated* artifact rather than a
  hand-committed copy. Two viable shapes (either is fine; not auto-applied):
  1. Add a `prepackage`/`compile`-time copy step in `extension/package.json`
     (e.g. a tiny `scripts/sync-web.js` that copies `../netscope/web/{template.html,vendor}`
     into `media/`), and gitignore `extension/media/` so only the source is
     committed; OR
  2. Keep both committed but add a CI test (Python `tests/` or the headless JS
     suite) that asserts byte-equality of `template.html` and every `vendor/*.js`
     between the two roots, failing the build on drift.
  Option 2 is the lower-risk, smaller change (no packaging rework); option 1 is
  the cleaner true-single-source. The md5/byte-equality assertion itself is the
  high-confidence, clearly-correct piece regardless of which is chosen.

### REC-2 (MEDIUM) — Extract the temp-file + exec + parse + cleanup boilerplate in `extension.ts`

- **Files:** `extension/src/extension.ts` — `runAndTrace` (`:120-145`),
  `runIsolate` (`:149-173`), `runLLM` (`:228-254`), `runViewSpec` (`:259-280`),
  and the `diffWithLast` command body (`:508-522`).
- **Rationale:** All five build a temp path with the identical recipe
  `path.join(os.tmpdir(), netscope-<kind>-${process.pid}-${Date.now()}.json)`,
  then follow the same lifecycle: (optionally write the graph JSON), run via
  `execAsync`, check `fs.existsSync(outPath)`, `JSON.parse(fs.readFileSync(...))`
  inside `try`, and `unlinkSync` in `finally` with `catch {/* ignore */}`. The
  `try/parse/finally-unlink` block is copied verbatim 4x. This is genuine
  mechanical repetition (not load-bearing defensiveness — the defensive `catch`es
  here guard our own temp files, not arbitrary user models).
- **Proposed change:** add two small helpers and route all five sites through
  them, e.g.:
  - `tmpJsonPath(kind: string): string` returning the `os.tmpdir()` path
    (removes the 5 duplicated path literals);
  - `readJsonOnce<T>(p: string): T | null` that does
    `try { JSON.parse(readFileSync) } catch { null } finally { unlinkSync }`
    (removes the 4 duplicated read/parse/cleanup blocks).
  Keep the per-command messaging (`explainFailure`, the distinct warning strings)
  inline — that part legitimately differs per command and should not be folded in.
  MEDIUM (not high) because the call sites differ subtly (write-then-run vs
  run-with-NETSCOPE_OUT vs run-with-`--graph-json`; different existence-vs-code
  ordering), so the extraction needs care to preserve each site's exact
  error-path semantics, and the headless suite (`extension/test/headless.js`,
  12 tests) should be re-run to confirm. The win is real but modest (~25-30
  lines), so it should be reviewed, not auto-merged.

### REC-3 (LOW) — Two `_shape` helpers share a name but are NOT duplicates (no action)

- **Files:** `netscope/core/checks.py:13` vs
  `netscope/instrument/torch_nn.py:70`.
- **Rationale / why no action:** Same name, different jobs.
  `checks.py:_shape(meta_shape)` coerces an already-serialized shape list to
  `list[int]` (the dict-IR consumer side, with a `try/except (TypeError,
  ValueError)`). `torch_nn.py:_shape(x)` extracts `list(x.shape)` from a live
  torch tensor (the producer/instrumentation side). They operate on different
  inputs and live on opposite sides of the IR boundary; unifying them would force
  a fake shared abstraction across the capture/consume split. Leave as-is. I flag
  it only to pre-empt a static tool or reviewer "mistaking" them for a dup.

### REC-4 (LOW) — `name + out_shape` label helpers differ by output format (no action)

- **Files:** `netscope/sinks/html_sink.py:45` (`_node_label`, emits
  `"{name}\n{list(out)}"` -> `[2, 3]` style),
  `netscope/sinks/mermaid_sink.py:7` (`_label`, emits
  `name + " " + "x".join(...)` -> `2x3` style, plus a Mermaid-specific
  `"`->`'` escape), and the TS mirror `render.ts:7`.
- **Rationale / why no action:** Each sink deliberately renders the shape in its
  own syntax (bracketed list for the HTML panel, `x`-joined for Mermaid/edge
  labels, Mermaid additionally must escape quotes). They are ~3 lines each and
  share no nontrivial logic beyond "read `meta.out_shape` if present." Factoring a
  common `shape_to_str(fmt=...)` would add a parameterized helper to save almost
  nothing and couple two sinks that are intentionally independent. Not worth it.

### REC-5 (LOW) — The `qualname or name` idiom recurs but should stay inline (no action)

- **Files:** `netscope/core/checks.py:27` (`_label`),
  `netscope/core/diff.py:33` (`_key`) / `:215` / `:217`,
  `netscope/llm/prompts.py:72`, `netscope/mcp/server.py:124`.
- **Rationale / why no action:** The pattern
  `(node.get("meta") or {}).get("qualname") or node["name"]` appears ~6 times, but
  each usage sits inside a function with materially different surrounding intent
  (warning labels, a diff identity *tuple* that also falls back to loc, prompt
  formatting, MCP serialization). They live in 4 different modules with no natural
  shared home (`core`, `llm`, `mcp`). Hoisting a `node_label(node)` into
  `core/ir.py` and importing it into `llm/` and `mcp/` would add cross-package
  coupling to deduplicate a single boolean-or expression — that *raises* coupling
  to lower a near-zero duplication cost. The grounding note ("apply DRY ONLY where
  it reduces complexity") argues against it. Leave inline.

## Note on the `ruff` E731/E702 hits

`ruff check netscope` reported 5 style nits (`diff.py:113` lambda-assignment,
`diff.py:195`/`playground.py:101,112` semicolons, `views.py:20` unused
`typing.List`). These are **style/lint**, not deduplication, and are out of scope
for Criterion 1 — flagged here only so they aren't mistaken for DRY findings.
