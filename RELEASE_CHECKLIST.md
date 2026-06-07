# Release checklist — netscope 0.1.5

**0.1.5 is the first version ever published** (0.1.0–0.1.4 were internal only —
there are no prior PyPI uploads, Marketplace listings, or git tags). The CHANGELOG
records the full pre-1.0 history.

Two artifacts ship together. The extension needs the engine installed in the user's
venv (it shells out, has no torch itself), so **publish PyPI first**.

Full step-by-step commands live in [PUBLISHING.md](PUBLISHING.md). This is the
gate list.

## Pre-flight (done locally, verified)

- [x] Version `0.1.5` in `pyproject.toml` and `extension/package.json` (in sync).
- [x] CHANGELOG `[Unreleased]` batch promoted to `## [0.1.5] — 2026-06-07`
      (SAM 3, multi-input playground, py3.10+/transformers v5, rank-aware checker).
- [x] `python -m build` → `dist/netscope-0.1.5-py3-none-any.whl` + `.tar.gz`.
- [x] `twine check dist/*` → **PASSED** (both).
- [x] Web assets confirmed *inside* the wheel: `netscope/web/template.html` + the
      4 vendored JS libs (cytoscape, dagre, cytoscape-dagre, expand-collapse).
- [x] Fresh-venv install proof (isolated py3.12 venv, CPU torch + the wheel, run
      from `/tmp` so the import resolves to site-packages): `import netscope` →
      a tiny traced model → `g.to_html()` emits a **self-contained 736 KB HTML,
      0 external scripts / 0 external links**.
- [x] `vsce package` → `extension/netscope-0.1.5.vsix` (18 files, manifest
      `Id=netscope Version=0.1.5 Publisher=eren23`, `out/`+`media/` only, no
      `.ts`/`src` leak).
- [x] Tests: **206 passed / 0 skipped** locally (thop installed, so the optional
      FLOPs test ran too; in a torch-only env it skips → 205 passed / 1 skipped).
      `tsc` clean; extension headless **12/12**.
- [x] Publisher ID set to **`eren23`** in `extension/package.json` (was the
      placeholder `netscope`).

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
cd /Users/eren/Documents/AI/network_visualizer_ext
# (optional) TestPyPI dry-run first:
python -m twine upload --repository testpypi dist/*
#   then in a clean venv: pip install -i https://test.pypi.org/simple/ \
#   --extra-index-url https://pypi.org/simple netscope && python -c "import netscope"

python -m twine upload dist/netscope-0.1.5*       # __token__ / pypi-<token>
```
- [ ] Verify: `pip install netscope` in a clean venv works.

## Publish the extension (after registering the publisher)

```bash
cd /Users/eren/Documents/AI/network_visualizer_ext/extension
npx @vscode/vsce login eren23                     # paste the PAT
npx @vscode/vsce publish                          # Marketplace
npx ovsx publish netscope-0.1.5.vsix -p <token>   # Open VSX (Cursor)
```
- [ ] Sideload-test once: `cursor --install-extension netscope-0.1.5.vsix`.

## Tag the release

The release commit + annotated tag `v0.1.5` are created locally (the **first tag
in the repo**). Push when you're ready to ship:

```bash
git push origin main && git push origin v0.1.5
```

## After 0.1.5

See [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md). Once 0.1.5 is live,
flip the "First PyPI release pending" note in `README.md` and the CHANGELOG header.
Next forks (parked until 0.1.5 ships + gets feedback): isolation Levels 1–2
(click-to-focus + `scope=`), attention/value capture for attention-head maps,
more frameworks (JAX/Keras), and the OpenTelemetry export bridge.
