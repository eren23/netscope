# Publishing netscope

netscope ships as **two artifacts**:

1. **`netscope` (Python engine)** → **PyPI**. This is the core: the import-hook tracer,
   the graph IR, and the standalone HTML renderer. Usable with zero editor — `pip install
   netscope` then `import netscope`.
2. **netscope extension** → **VS Code Marketplace** + **Open VSX** (for Cursor / VSCodium).
   A thin UI shell (CodeLens, webview, click-to-source) that calls the engine running in the
   user's own Python environment.

The extension **requires** the engine to be pip-installed in the user's project venv — it
cannot trace on its own (it has no torch / no model classes). Publish the PyPI package first.

---

## A. PyPI — the `netscope` engine

### One-time setup
1. Create an account at https://pypi.org/account/register/ (and optionally
   https://test.pypi.org for a dry run).
2. Create an API token: PyPI → Account settings → API tokens → "Add API token"
   (scope: entire account for the first upload, then narrow to the project).
3. Install tooling (already in the dev venv): `pip install build twine`.

### Build
```bash
cd /Users/eren/Documents/AI/netscope-release
rm -rf dist build *.egg-info
python -m build          # produces dist/netscope-<version>-py3-none-any.whl + .tar.gz
python -m twine check dist/*
```

### (Recommended) dry-run on TestPyPI
```bash
python -m twine upload --repository testpypi dist/*
# then in a clean venv:
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple netscope
python -c "import netscope; print('ok')"
```

### Upload to real PyPI
```bash
python -m twine upload dist/*
# username: __token__
# password: pypi-<your-token>
```
After this, anyone can `pip install netscope`.

### Bumping versions
Edit `version` in `pyproject.toml` (`[project]`), rebuild, re-upload. PyPI **rejects
re-uploading an existing version**, so always bump.

---

## B. VS Code Marketplace — the extension

### One-time setup
1. Create a **publisher**: https://marketplace.visualstudio.com/manage — sign in with a
   Microsoft account, create a publisher ID (e.g. `eren23`). Put that exact ID in
   `extension/package.json` → `"publisher"`.
2. Create a Personal Access Token (PAT): https://dev.azure.com → User settings → Personal
   access tokens → New token → Organization: **All accessible**, Scope: **Marketplace →
   Manage**. Copy it.
3. `npm install -g @vscode/vsce` (or use `npx @vscode/vsce`).

### Build the .vsix
```bash
cd /Users/eren/Documents/AI/netscope-release/extension
npm install
npm run compile
npx @vscode/vsce package      # produces netscope-<version>.vsix
```
`.vscodeignore` keeps the package lean (ships `out/` + `media/`, drops `src/`, tests,
node_modules). `media/` contains the Cytoscape template + vendored JS so the webview renders
offline.

### Publish
```bash
npx @vscode/vsce login <publisher-id>     # paste the PAT
npx @vscode/vsce publish                   # or: vsce publish minor / patch to bump
```

### Open VSX (so Cursor / VSCodium users can install it too)
The MS Marketplace is not available to Cursor by default; mirror to Open VSX:
1. Account at https://open-vsx.org (sign in with GitHub), create an access token.
2. ```bash
   npx ovsx publish netscope-<version>.vsix -p <openvsx-token>
   ```

---

## C. What a brand-new user does (the out-of-the-box path)

```bash
# in their project venv
pip install netscope
```
Then either:
- **Library only:** `import netscope` → `with netscope.graph("m"): model(x)` → `g.show()`
  opens a standalone interactive graph. No editor needed.
- **With the extension:** install "netscope" from the Marketplace (VS Code) or Open VSX
  (Cursor), set `netscope.pythonPath` to the venv that has netscope installed, and use the
  `netscope: Show Graph` command / CodeLens.

---

## Release checklist
- [ ] Bump `version` in `pyproject.toml` AND `extension/package.json` (keep them in sync).
- [ ] `python -m build && python -m twine check dist/*` → clean.
- [ ] Fresh-venv smoke test: `pip install dist/*.whl` → `import netscope` + `g.show()` works.
- [ ] `vsce package` → install the `.vsix` locally (`code --install-extension *.vsix`) and
      sanity-check the webview renders.
- [ ] `twine upload dist/*`
- [ ] `vsce publish` + `ovsx publish`
- [ ] Tag the release in git.
