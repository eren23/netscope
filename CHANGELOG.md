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

### Fixed
- Mismatch checker is now rank-aware: encoder→decoder **sequence-length**
  differences no longer false-flag as shape clashes (the feature axis is the last
  axis for 2-D/3-D tensors, the channel axis for 4-D NCHW).

### Security
- **Playground origin guard** — `python -m netscope.playground` runs the editor's
  code, so the loopback server now rejects any request whose `Host` isn't loopback
  (defeats DNS rebinding) or that carries a cross-origin `Origin` (defeats CSRF) —
  a remote page can no longer drive the code-exec endpoint.
- **HTML injection** — the standalone graph escapes `</` in its embedded data so a
  crafted node name can't close the inlined `<script>` (the editor webview already
  had a nonce CSP; this covers `g.show()` output too).

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
