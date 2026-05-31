# Release checklist — netscope 0.1.0

Two artifacts ship together. The extension needs the engine installed in the
user's venv (it shells out, has no torch itself), so **publish PyPI first**.

Full step-by-step commands live in [PUBLISHING.md](PUBLISHING.md). This is the
gate list.

## Pre-flight (done locally, verified)

- [x] Version bumped to `0.1.0` in `pyproject.toml` and `extension/package.json`.
- [x] `python -m build` → `dist/netscope-0.1.0-py3-none-any.whl` + `.tar.gz`.
- [x] `twine check dist/*` → PASSED (both).
- [x] Web assets + isolation + session-fix code confirmed *inside* the wheel.
- [x] Fresh-venv install proof: `pip install` the wheel → `import netscope` →
      `g.show()` emits a self-contained HTML; isolation works on resnet18.
- [x] `vsce package` → `extension/netscope-0.1.0.vsix` (15 files, out/+media/
      only, no `.ts`/`src` leak).
- [x] Tests: 76 passed / 1 skipped (JUnit-confirmed); tsc clean; headless 5/5.

## Accounts / tokens needed (you, one-time)

- [ ] **PyPI** account + API token (`__token__` / `pypi-…`).
- [ ] **VS Code Marketplace** publisher ID — set it in `extension/package.json`
      `"publisher"` (currently the placeholder `"netscope"`) — plus an Azure PAT
      (scope: Marketplace → Manage).
- [ ] **Open VSX** account + token (so Cursor / VSCodium users can install).
- [ ] **GitHub** repo created (see "Publish the repo" below).

## Publish the repo

This working tree is clean (87 tracked files, no build junk, no `.claude/`).

```bash
# create the repo on GitHub (empty), then from the project dir:
git remote add origin https://github.com/<you>/netscope.git
git push -u origin main
```

> The current local history includes the build/iteration churn. If you want a
> single clean initial commit instead, see "Squashing history" in PUBLISHING.md
> (the `git reset` is blocked by a local safety hook, so run it yourself).

## Publish the engine (PyPI)

- [ ] (recommended) TestPyPI dry-run: `twine upload --repository testpypi dist/*`
      → install from TestPyPI in a clean venv → `import netscope`.
- [ ] `twine upload dist/*`
- [ ] Verify: `pip install netscope` in a clean venv works.

## Publish the extension

- [ ] Set the real publisher ID, rerun `vsce package`.
- [ ] `vsce publish` (Marketplace).
- [ ] `ovsx publish netscope-0.1.0.vsix -p <token>` (Open VSX, for Cursor).
- [ ] Sideload-test the `.vsix` once: `cursor --install-extension netscope-0.1.0.vsix`.

## Tag the release

```bash
git tag v0.1.0 && git push origin v0.1.0
```

## After 0.1.0

See [ROADMAP.md](ROADMAP.md) — next up is the "as you write" live engine +
the LLM-augmented layer (built together).
