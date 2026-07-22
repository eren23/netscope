# Release checklist — netscope 0.1.7

**0.1.7 is the first version ever published** (0.1.0–0.1.6 were internal only —
there are no prior PyPI uploads, Marketplace listings, or pushed tags; the 0.1.6
cut was superseded on `main` before upload). 0.1.7 adds the full new batch:
`scope=` capture, the memory/OOM predictor (+ `cost: predicted mem` overlay),
`netscope fix`, the generation-timeline strip, the real-model showcase, and the
transformers-v5.12 KV-cache capture fix.

Two artifacts ship together. The extension needs the engine installed in the user's
venv (it shells out, has no torch itself), so **publish PyPI first**.

Full step-by-step commands live in [PUBLISHING.md](PUBLISHING.md). This is the
gate list.

## Pre-flight (done locally, verified for 0.1.7)

- [x] Version `0.1.7` in `pyproject.toml` and `extension/package.json` (in sync).
- [x] CHANGELOG `[Unreleased]` batch promoted to `## [0.1.7] — 2026-07-22`.
- [x] `python -m build` → `dist/netscope-0.1.7-py3-none-any.whl` + `.tar.gz`
      (stale 0.1.6 artifacts removed; stale local `v0.1.6` tag deleted).
- [x] `twine check dist/*` → **PASSED** (both).
- [x] Web assets confirmed *inside* the wheel (template + 4 vendored JS libs), and
      the new modules present (`netscope/autofix.py`, `netscope/enrich/memory.py`).
- [x] Fresh-venv install proof (isolated py3.12 venv, wheel installed, run from
      `/tmp`, torch borrowed from the dev venv): trace → `netscope.memory()` works
      from the wheel → `g.to_html()` is **self-contained, 0 external refs**, with
      the generation-timeline widget present.
- [x] Tests: **270 passed / 0 skipped** locally (torchvision + ultralytics
      installed); coverage 85%+; ruff + mypy clean (CI-identical commands).
- [x] Rebuilt the VSIX: `extension/netscope-0.1.7.vsix` (20 files, 231 KB);
      `tsc` clean; extension `test:unit` 5/5 + `test:headless` 12/12 green.
- [x] Publisher ID is **`eren23`** in `extension/package.json`.

## ⚠ Before `vsce publish` — register the publisher

The publisher ID `eren23` is baked into the package + `.vsix`, but it must be a
**registered** Marketplace publisher before `vsce publish` will accept it:

1. Create the publisher `eren23` at https://marketplace.visualstudio.com/manage
   (Microsoft account). The ID must match exactly.
2. No code change needed — `package.json` and the `.vsix` already carry `eren23`.
   (Only re-run `vsce package` if you pick a *different* ID.)

The PyPI wheel has no such blocker — it's ready to upload as-is.

## Accounts / tokens needed (you, one-time)

- [ ] **PyPI** account + API token (`__token__` / `pypi-…`).
- [ ] **VS Code Marketplace** publisher `eren23` (see above) + an Azure PAT
      (scope: Marketplace → Manage).
- [ ] **Open VSX** account + token (so Cursor / VSCodium users can install).

## Publish the engine (PyPI) — do this first

```bash
cd /Users/eren/Documents/AI/netscope-release
# (optional) TestPyPI dry-run first:
python -m twine upload --repository testpypi dist/*
#   then in a clean venv: pip install -i https://test.pypi.org/simple/ \
#   --extra-index-url https://pypi.org/simple netscope && python -c "import netscope"

python -m twine upload dist/netscope-0.1.7*       # __token__ / pypi-<token>
```
- [ ] Verify: `pip install netscope` in a clean venv works.

## Publish the extension (after registering the publisher)

```bash
cd /Users/eren/Documents/AI/netscope-release/extension
npx @vscode/vsce login eren23                     # paste the PAT
npx @vscode/vsce publish                          # Marketplace
npx ovsx publish netscope-0.1.7.vsix -p <token>   # Open VSX (Cursor)
```
- [ ] Sideload-test once: `cursor --install-extension netscope-0.1.7.vsix`.

## Tag the release

The release commit + annotated tag `v0.1.7` are created locally (the **first tag
in the repo**). Push when you're ready to ship — **after** the PyPI upload, so the
README's install badges and instructions are true the moment `main` is public:

```bash
git push origin main && git push origin v0.1.7
```

## After 0.1.7

See [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md). Next forks (parked
until 0.1.7 ships + gets feedback): click-to-focus (isolation Level 1, webview),
the extension-webview generation timeline, an MCP `apply_fix` tool, more
frameworks (JAX/Keras), and the OpenTelemetry export bridge.
