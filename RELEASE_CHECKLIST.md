# Release checklist — netscope 0.1.4

Two artifacts ship together. The extension needs the engine installed in the
user's venv (it shells out, has no torch itself), so **publish PyPI first**.

Full step-by-step commands live in [PUBLISHING.md](PUBLISHING.md). This is the
gate list.

## Pre-flight (done locally, verified)

- [x] Version `0.1.4` in `pyproject.toml` and `extension/package.json` (in sync).
- [x] `python -m build` → `dist/netscope-0.1.4-py3-none-any.whl` + `.tar.gz`.
- [x] `twine check dist/*` → PASSED (both).
- [x] New 0.1.4 code confirmed *inside* the wheel: `core/diff.py`, `enrich/roles.py`,
      `core/timeline.py`, `playground.py`, refreshed web assets (manifest `0.1.4`).
- [x] Fresh-venv install proof: `pip install` the wheel in a clean venv →
      `diff` / `diff_view` / `roles` / `timeline` / `step` all importable → a
      profiled trace returns roles + a timeline → `g.show()` emits a
      **self-contained ~720 KB HTML with 0 external scripts**.
- [x] `vsce package` → `extension/netscope-0.1.4.vsix` (18 files, `out/`+`media/`
      only — manifest `0.1.4`, the `Diff with Last Trace` + `Run & Trace (Profiled)`
      commands and the role/cost overlays present, no `.ts`/`src` leak).
- [x] Tests: **188 passed / 1 skipped** (the skip is THOP FLOPs, an optional
      extra); `tsc` clean; headless **12/12**.

## ⚠ Blocker before `vsce publish` — set a real publisher ID

`extension/package.json` `"publisher"` is the **placeholder `"netscope"`**, which
is *not* a registered Marketplace publisher. `vsce publish` will fail until you:

1. Create a publisher at https://marketplace.visualstudio.com/manage (Microsoft
   account) and copy its exact ID (e.g. `eren23`).
2. Set that ID in `extension/package.json` → `"publisher"`.
3. **Re-run `vsce package`** so the `.vsix` carries the real publisher.

The PyPI wheel has no such blocker — it's ready to upload as-is.

## Accounts / tokens needed (you, one-time)

- [ ] **PyPI** account + API token (`__token__` / `pypi-…`).
- [ ] **VS Code Marketplace** publisher ID (see blocker above) + an Azure PAT
      (scope: Marketplace → Manage).
- [ ] **Open VSX** account + token (so Cursor / VSCodium users can install).

## Publish the engine (PyPI) — do this first

```bash
cd /Users/eren/Documents/AI/network_visualizer_ext
# (optional) TestPyPI dry-run first:
python -m twine upload --repository testpypi dist/*
#   then in a clean venv: pip install -i https://test.pypi.org/simple/ \
#   --extra-index-url https://pypi.org/simple netscope && python -c "import netscope"

python -m twine upload dist/netscope-0.1.4*       # __token__ / pypi-<token>
```
- [ ] Verify: `pip install netscope` in a clean venv works.

## Publish the extension (after the publisher fix)

```bash
cd /Users/eren/Documents/AI/network_visualizer_ext/extension
# 1. set the real publisher ID in package.json, then:
npx @vscode/vsce package                          # rebuild netscope-0.1.4.vsix
npx @vscode/vsce login <publisher-id>             # paste the PAT
npx @vscode/vsce publish                          # Marketplace
npx ovsx publish netscope-0.1.4.vsix -p <token>   # Open VSX (Cursor)
```
- [ ] Sideload-test once: `cursor --install-extension netscope-0.1.4.vsix`.

## Tag the release

```bash
git tag v0.1.4 && git push origin v0.1.4
```

## After 0.1.4

See [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md). 0.1.4 adds trace
diffing, the profiler + cost heatmap, the role lens, the generation timeline, and
the playground. Still open: the `scope=` capture API and the deeper LLM views
(attention-weight maps, KV-cache shapes) that need value / multi-forward capture.
