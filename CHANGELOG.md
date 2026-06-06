# Changelog

Notable changes to netscope. Pre-1.0 and iterating, so 0.1.x minors add features.
Nothing is on PyPI yet — the first published release is pending.

## [Unreleased]

_Nothing yet._

## [0.1.4]

### Added
- **Trace diffing** — `netscope.diff(before, after)` / `netscope.diff_view`, a CLI
  (`python -m netscope.core.diff`, `--graph-json` / `--html`), and a colored
  added (green) / changed (amber) / removed graph. Keyed by qualname/loc so it
  survives a layer insert. `NVGraph.from_dict` round-trips saved traces.
- **Profiler** — activation + param **memory** on every node, free (derived from
  the shapes already captured); `netscope.graph(profile=True)` / `NETSCOPE_PROFILE`
  adds per-layer **wall-time**. A cost heatmap recolors nodes by time / memory /
  params. The zero-overhead, metadata-only default path is preserved.
- **Role lens** — `netscope.roles(g)` + a "by role" graph overlay coloring nodes
  by architectural role (attention / MLP / norm / embedding).
- **Generation timeline** — `netscope.step()` marks each decode step (auto-numbered,
  timed under profile); `netscope.timeline(g)` returns per-step sequence-growth +
  latency, and the steps render as a left-to-right, cost-colorable timeline.
- **Playground** — `python -m netscope.playground`, a local split-view editor ⇄
  live graph (trace / static / profile / diff modes).
- **Extension** — `Run & Trace (Profiled)` and `Diff with Last Trace` commands.
- **Demo videos** (`docs/video/`) + a README front door and a "See it live" gallery.
- **`netscope` CLI** — a unified console command (`netscope static|playground|mcp|
  diff|views`) + a `python -m netscope` dispatcher; `torch`/`hf` install extras; a
  `py.typed` marker so downstream type-checkers see the hints.
- **CI** — GitHub Actions: pytest (3.9–3.12, CPU torch), the extension
  (tsc/unit/headless), and build + twine check on push/PR.
- **Big-model rendering** — the graph auto-folds a large model (YOLO 272, RT-DETR
  639, …) to a readable top-level pipeline by default (each block expandable), and
  uses the layered L→R layout for folded detectors — no more unreadable blob.

### Fixed
- **Mismatch checker** is rank-aware: encoder→decoder sequence-length differences
  (and Conv1d/(N,C,L) length changes) no longer false-flag — and a Conv1d *channel*
  clash, previously missed, is now caught.
- **Exception mid-forward** no longer corrupts the rest of a session — the post-hook
  registers with `always_call=True`, so a raised forward unwinds its span / timing /
  parent state instead of mis-parenting and mislabeling the next model.
- **Static analysis** no longer false-flags idiomatic CNN heads (it won't wire a
  `Conv → Linear` sibling across an implicit `flatten()`).
- **Trace diffing**: sibling ops with no qualname/loc no longer collapse in the index
  (a real add/remove is kept), and removed subtrees keep their hierarchy as ghosts.
- **`NVGraph.from_dict`** skips dangling edges (no junk auto-created nodes from a
  truncated trace).
- **Extension graph** shows mismatch warnings again — the ⚠ pill, warn list, and red
  clash edges were missing in the webview; `render.ts` is back at parity with the
  standalone sink (warnings, edge-warn, source row, inferred styling), and `merge.py`
  matches `mergeByLoc.ts` on static-edge fusion.
- **MCP `trace_file`** surfaces the script's real exit code + stderr (not a generic
  "no graph"), validates `mode`, and uses `mkstemp`. Plus smaller nits: mermaid id
  collisions, timeline sort on mixed types, an accurate threading note, dead code.
- **Detection/segmentation stack** (dogfooded on YOLOv8, RT-DETR, SAM, torchvision
  detectors): the mismatch checker now compares each edge against the tensor that
  actually flowed (multi-scale backbones like RT-DETR) and skips fan-in/merge
  consumers (concat necks in YOLO/FPN/U-Net) — eliminating ~all false alarms on
  these (RT-DETR 4→0, YOLOv8 6→0). The comet fx guards non-finite layout coords so
  a large non-linear graph can't crash the render.

### Security
- **Playground origin guard** — `python -m netscope.playground` runs the editor's
  code, so the loopback server now rejects any request whose `Host` isn't loopback
  (defeats DNS rebinding) or that carries a cross-origin `Origin` (defeats CSRF) —
  a remote page can no longer drive the code-exec endpoint.
- **HTML injection** — the standalone graph escapes `</` in its embedded data so a
  crafted node name can't close the inlined `<script>` (the editor webview already
  had a nonce CSP; this covers `g.show()` output too).
- **MCP file-read** — `explain_node` loads an untrusted trace JSON, so source-line
  reads are restricted to the project dir; a crafted `loc.file` can no longer
  exfiltrate arbitrary files into the LLM prompt.

## [0.1.3] — baseline (built; PyPI release pending)

The core product: auto-trace PyTorch / Hugging Face with zero decorators, an
interactive self-contained HTML graph (real shapes, dtype, device; folded repeated
blocks), shape-mismatch warnings, click-to-source, and isolate-a-part. Static
analysis (AST declared-dim checks + a `torch.fx` fallback). The in-editor live
experience (inline shape hints, mismatch squiggles, live-on-save). The LLM layer
(grounded assistant, augmented inference, generated views). An MCP server. Packaged
as a PyPI wheel + a VSIX.

## [0.1.0]

First cut of the trace → visualize → show-errors loop, with the VSCode / Cursor
extension.
