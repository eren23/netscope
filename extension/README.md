# netscope — VSCode extension

Live graph of your PyTorch / Hugging Face pipeline, in the editor:

- **Show Pipeline Graph** — parses the active Python file (no execution) and draws
  the *static skeleton*: stages, the branch fan-out (`for … range`), the vote
  (`Counter(...).most_common`), `@netscope.stage`-decorated functions.
- **Run & Trace** — runs the file with the netscope tracer active, captures the
  *real* runtime graph (live tensor shapes, params, dataflow), then **fuses** it
  onto the static skeleton by source location and draws the combined view.
- **Click any node** → jumps to that line in the editor.

The graph is rendered with the *same* Cytoscape template the Python library uses
for standalone `g.show()`, so the editor view and the library view are identical.

## How it works (architecture)

```
file changes ──► `python -m netscope.static <file>` ──► static graph JSON ─┐
                                                                          ├─► mergeByLoc ─► Cytoscape webview
Run & Trace ──► run file with NETSCOPE_OUT set ──► runtime graph JSON ──────┘        (click → reveal in editor)
```

The extension is a thin shell over the netscope Python library: it shells out to
the library's static CLI and to a normal `python <file>` run (the library writes
its graph to `$NETSCOPE_OUT` on session exit). All graph semantics live in Python;
the TypeScript only fuses (`src/mergeByLoc.ts`) and renders (`src/render.ts`).

## Requirements

- The `netscope` Python package importable by your interpreter, and your script
  wrapping its pipeline in `with netscope.graph("name"): …` (for **Run & Trace**).
- Set **`netscope.pythonPath`** to your venv's python (e.g. the repo's
  `.venv/bin/python`); otherwise `python3` on PATH is used.

## Run it from source (manual test)

This extension's live UI has no `code`-CLI smoke test in CI — verify interactively:

1. `cd extension && npm install && npm run compile`
2. Open the **repo root** in VSCode, press **F5** ("Run Extension") → an Extension
   Development Host window opens with `netscope` loaded.
3. In that window, set `netscope.pythonPath` to `${workspaceFolder}/.venv/bin/python`.
4. Open `examples/sfumato_cmajc.py`. Two CodeLens links appear at the top:
   - click **netscope: Show Graph** → the static skeleton (branch loop + vote) draws.
   - click **Run & Trace** → the fused graph draws: `plan → diffuse[0..4] → vote`
     with real `[32]` tensor shapes; click `vote` → editor jumps to its line.

## Verified vs. manual

- **Verified automatically:** the Python static CLI + file sink (pytest), the TS
  `mergeByLoc` + `render` logic (`npm run test:unit`), a full Python→JSON→TS fuse
  of the real sfumato demo, and a clean `tsc` compile.
- **Manual only (needs a VSCode host):** CodeLens clicks, webview rendering, and
  click-to-source navigation — follow the steps above.
