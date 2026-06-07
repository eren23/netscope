# Criterion 8 — AI slop, stubs, larp, and unhelpful comments

**Verdict: already clean. Zero high-confidence edits. Recommend no changes.**

The prior grounding ("comments are excellent, near-zero slop") holds up under a
full grep + tool sweep of both trees. Every in-motion / stub / slop pattern I
searched for resolved to either (a) third-party vendored code (do not touch),
(b) a legitimate domain term, or (c) a genuinely load-bearing caveat that is
exactly the "concise help for a new reader" this criterion wants to preserve.

## What I ran (read-only)

- In-motion phrasing grep: `now we|used to|changed to|previously|for now|NB:|NOTE:|TODO|FIXME|XXX|HACK`
  over `netscope/` + `extension/src/` + `template.html`. 96 raw hits — **84 are in
  `netscope/web/vendor/dagre.min.js`** (third-party lodash/dagre, excluded). Of the
  ~11 in our own code, none are in-motion-work residue.
- Stub/larp grep: `stub|placeholder|not implemented|coming soon|in the future|
  will be|dummy|fake|temporary|naive|cheap|simplif`. All hits are domain words
  ("cheap models", "cheap property reads") or real tokens (CSS `::placeholder`,
  the `__NETSCOPE_VENDOR__` template substitution token, VSCode `placeHolder`).
- Slop adverbs / marketing adjectives: `obviously|simply|just|clearly|leverage|
  utilize|robust|seamless|comprehensive|elegant|please note|...`. One "robust"
  (mermaid_sink.py:3) used precisely; "just"/"simply" all in well-formed sentences.
- Real stubs: scanned for bare `...` bodies (none), `raise NotImplementedError`
  (none), and bare `pass` (10 — **all inside defensive try/except guards**, the
  load-bearing optional-dependency / arbitrary-model introspection pattern).
- Commented-out code: `ruff --select ERA` → 1 hit (provider.py:34), a false
  positive (type-shape doc, see below). Manual verb-led-comment grep → 0 real hits.
  Zero commented-out code in `extension/src/`.
- Tool coverage (for completeness, not comment-specific): vulture (13 hits, all
  the known dynamic-dispatch / public-API / template-JS false positives), ruff
  (5 style issues — lambda/semicolons/unused-import — belong to other criteria,
  not slop), knip + madge (vendor inlining + DI exports; no cycles).

## Things that LOOK like findings but are NOT (do not touch)

These are the strongest candidates a careless pass would flag. Each is correct
as-is and should be left alone.

- `extension/src/extension.ts:47` `// The whole editor used to freeze on a
  synchronous execFileSync ...` — reads as in-motion ("used to") but is a precise,
  concise rationale for why exec is async + cancellable. Load-bearing. Keep.
- `extension/src/extension.ts:393` `// FUNCTION replacers: the minified libs +
  elements JSON contain `$&`/`$'` ... which corrupted the output and left the
  panel blank.` — explains a real, non-obvious `String.replace` bug. Keep.
- `netscope/instrument/torch_nn.py:264` `# NB: a ROOT module's qualname is ""
  ...` — explains the `is not None` vs truthiness invariant. Keep.
- `netscope/sinks/html_sink.py:96` `# NB: cytoscape reserves data.source/target
  ...` — explains a real namespace collision with the producer's `flow` field. Keep.
- `netscope/web/template.html:503` `// NB: compare TOTAL_NODES (captured before
  collapse), NOT cy.nodes().length ...` — explains a subtle expand-collapse trap. Keep.
- `netscope/llm/provider.py:34` `# message = {"role": ..., "content": str}`
  (ruff ERA001 "commented-out code") — **false positive**: it documents the shape
  of the `Message = Dict[str, str]` alias on the next line. Keep.
- The five design-rationale docstrings called out in the task (checks.py
  feature-axis, torch_nn.py weakref, diff.py identity, etc.) — explicitly off-limits
  and correctly so.

## Ranked recommendations

There are **no high-confidence edits**. I considered two low-confidence,
purely-cosmetic nits below for honesty/completeness; I do **not** recommend acting
on either — both comments are already clear and correct, and editing them is
churn with no reader benefit. Listed only so the record shows they were examined.

(See the recommendations array for the two low-confidence items.)
