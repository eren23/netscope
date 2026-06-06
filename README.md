# netscope

**Trace, visualize, and sanity-check neural-network pipelines as you build them.**

`import netscope`, wrap a forward pass, and get an interactive graph of your
PyTorch / Hugging Face pipeline — real per-layer tensor shapes, the actual
dataflow, repeated blocks folded into one, and **shape mismatches flagged in
red** before they blow up at runtime. No decorators, ~zero overhead, no CDN.

![catch a shape mismatch as you type](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/bug.gif)

> *Wire an encoder's 256-dim output into a head that expects 128 — netscope flags
> it red **as you type**, before you ever run.* &nbsp;([more clips ↓](#see-it-live))

> Working name. The hero demo is [sfumato](https://github.com/eren23/sfumato) —
> a hybrid AR (Qwen) + diffusion (LLaDA) reasoning pipeline.

![sfumato cmajc pipeline](https://raw.githubusercontent.com/eren23/netscope/main/docs/img/sfumato.png)

---

## See it live

Recorded live in the editor — every keystroke re-analyzed, the graph updated as
you work. **Try it yourself:** `python -m netscope.playground` opens this split
view in your browser (paste a model, watch it analyzed — trace / static / profile
/ diff modes). Or watch the **▶ [~50-second full tour](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/netscope-reel.mp4)**.

|  |  |
|:--|:--|
| **Build a model — real shapes appear** | **Diff two versions** — 🟢 added · 🟡 changed |
| ![shapes as you write](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/shapes.gif) | ![diff two model versions](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/diff.gif) |
| **Profile by cost** — the fat layer glows red | **Color by role** — attention / MLP / norm |
| ![profile cost heatmap](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/profile.gif) | ![color a transformer by role](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/roles.gif) |

**On real models** — paste it, get the graph (big models auto-fold to a readable
top-level pipeline you can drill into):

| resnet18 | GPT-2 from config (role-colored) | MobileNetV3 |
|:--:|:--:|:--:|
| ![resnet18 traced](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/resnet.gif) | ![GPT-2 traced](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/gpt2.gif) | ![MobileNetV3 traced](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/mobilenet.gif) |

…and **detection / DETR** — YOLOv8 (272 layers) and RT-DETR (639), folded to a pipeline:

| YOLOv8 | RT-DETR |
|:--:|:--:|
| ![YOLOv8 traced](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/yolo.gif) | ![RT-DETR traced](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/rtdetr.gif) |

## Why

When you build a model or wire a multi-stage pipeline, the *structure* and the
*tensor dataflow* are invisible while you type. Existing tools are **post-hoc**
(Netron / torchview need a built+exported model), **debug-time** (tensor value
viewers), or **web dashboards** (Langfuse / Phoenix — prompts & latency, no
architecture or shapes). None fuse **static source structure** with a **runtime
trace** and render it where you write code.

netscope captures the real run and turns it into a graph you can actually read.

## Install

```bash
pip install netscope          # the engine + standalone HTML renderer (needs PyTorch)
```

> First PyPI release pending — until it lands, install from a clone with
> `pip install -e .`. The VSCode / Cursor extension lives in `extension/`; the
> fastest way to *try* netscope with no editor is `python -m netscope.playground`.

**Requirements:** Python ≥ 3.9 and **PyTorch** (install it for your platform
first — it's intentionally not a hard dependency, since torch wheels are large and
platform-specific). Add `transformers` too if you trace Hugging Face models.
netscope itself is light — just `wrapt` + `networkx`.

After install you also get a **`netscope`** command — `netscope playground`,
`netscope static model.py`, `netscope mcp`, … (see [docs/API.md](https://github.com/eren23/netscope/blob/main/docs/API.md)).

## Quickstart

```python
import torch, torch.nn as nn
import netscope

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

with netscope.graph("mlp") as g:        # auto-traces everything inside
    model(torch.randn(2, 8))

g.show()                              # opens a self-contained interactive HTML
```

That's it — no decorators on your model, no edits. The forward pass is captured
via `wrapt` post-import hooks + torch's global module hooks (capture-once, so
steady-state overhead is ~zero), and only tensor *metadata* (shape/dtype/device)
is kept, never tensors.

## What you get

- **Real tensor shapes, dtype + device** on every node and dataflow edge,
  captured from the run (dict/`ModelOutput`-returning HF modules included).
- **Hierarchy** — `pipeline → stage → model → module` as nested boxes.
- **Dataflow edges** — within-model via tensor identity, cross-stage via light
  `nv.stage` / `nv.branch` / `nv.reduce` hints. Animated comet particles show
  tensors flowing.
- **Collapse/expand** — repeated blocks (a model's N decoder layers) fold into a
  single `[+]` node, so a deep model reads as a clean left-to-right pipeline.
- **Shape-mismatch warnings** — a dataflow edge whose producer/consumer shapes
  don't line up is painted red (pulsing) with a ⚠ list — feature-dim clashes
  *and* rank/"forgot to flatten()" bugs.
- **Static analysis** — `python -m netscope.static yourfile.py` recovers the
  branch/vote structure **and** declared-dim wiring clashes from source *without
  running it* (a `torch.fx` fallback recovers real structure for traceable models).
- **Isolate a part** — re-run just one submodule on its real input, alone.
- **Click-to-source** — every node carries `file:line`.
- **Trace diffing** — `netscope.diff(before, after)`: edit a model, re-trace, and
  see exactly what changed — nodes added (green) / removed, plus shape + param
  deltas on the ones that stayed. Keyed by qualname, so it survives a layer insert.
- **Per-layer cost** — activation + param **memory** on every node (free, derived
  from shapes); `netscope.graph(profile=True)` adds wall-**time**. A cost heatmap
  recolors nodes so the fat/slow layer pops.
- **Role lens** — color a model by architectural role (attention / MLP / norm /
  embedding) so a transformer's structure reads at a glance; `netscope.roles(g)`
  returns the breakdown.
- **Generation timeline** — wrap decode steps in `netscope.step()`;
  `netscope.timeline(g)` returns the per-step sequence-growth + latency, and the
  steps render as a left-to-right timeline you can color by per-step time.

…plus three optional layers on top — the in-editor experience needs **no key**;
the LLM features are bring-your-own-key; everything works offline without them:

- **In-editor live experience** — inline shape hints + mismatch squiggles on the
  line as you write; the static skeleton + clashes refresh live on save.
- **LLM assistant** — explain a node / why it's flagged / suggest a fix, plus
  **augmented inference** (fill structure the AST can't see, drawn as provisional
  dashed nodes) and **generated views** (a prompt → a safe re-styling of the graph).
- **MCP server** — expose the live graph + real shapes to coding agents
  (Cursor / Claude Code) so they query reality instead of guessing.

## Playground

```bash
python -m netscope.playground       # opens http://localhost:8770
```

Paste a model on the left, watch the real netscope graph build on the right as you
type — `trace` / `static` / `profile` / `diff` modes. It's the clips above, live
and local. (It runs your code to trace it — the same trust as `python yourfile.py`,
bound to localhost.)

![the netscope playground — type a model, switch modes, color by cost](https://raw.githubusercontent.com/eren23/netscope/main/docs/video/playground.gif)

## Gallery

A real LLM architecture — **Qwen3** built from its Hub config (no weight
download), decoder layers folded:

![Qwen3 pipeline](https://raw.githubusercontent.com/eren23/netscope/main/docs/img/qwen3.png)

A **shape mismatch** caught while wiring an encoder into a head:

![mismatch warning](https://raw.githubusercontent.com/eren23/netscope/main/docs/img/mismatch.png)

Run them yourself:

```bash
python examples/sfumato_cmajc.py        # AR-plan -> diffuse x5 -> vote (CPU, mocked)
python examples/resnet_demo.py          # resnet18, 11.7M params, layers folded
python examples/transformer_demo.py     # a TransformerEncoderLayer
python examples/real_model_demo.py      # real Qwen3 arch, no weights (LAYERS=28 for all)
python examples/mismatch_demo.py        # Encoder(256) -> head(128): flagged red
python examples/static_dim_check_demo.py # a wiring clash caught WITHOUT running
python examples/isolate_demo.py         # re-run just resnet's layer2, alone
python examples/mcp_server_demo.py      # the MCP tools an agent would call
python examples/views_demo.py           # a prompt -> a graph re-styling spec
python examples/diff_demo.py            # two model versions -> a colored diff
python examples/profile_demo.py         # per-layer cost -> a heatmap
python examples/roles_demo.py           # color a transformer by attention/MLP/norm
python examples/generation_timeline_demo.py  # an autoregressive loop, step by step
```

## Optional hints

Auto-tracing sees *calls*, not *intent*. To name semantic regions (the stages,
the branches, the vote) add light markers — decorator or context-manager form,
both no-ops when not capturing:

```python
with netscope.stage("plan"):     plan = planner(x)
for b in range(5):
    with netscope.branch(f"diffuse[{b}]"):
        cand = refiner(plan)
with netscope.reduce("vote"):    winner = majority(cands)
```

For autoregressive generation, wrap each decode step in `with netscope.step():` —
the steps become the **generation timeline** (`netscope.timeline(g)`).

## VSCode / Cursor extension

`extension/` is a TypeScript extension that renders the graph in the editor:
**Show Graph** (static skeleton, no run) and **Run & Trace** (real graph, fused
by source location), with click-a-node → jump-to-line. After a trace you also get:

- **inline shape hints** — each layer's real tensor shape as ghost text on its line;
- **mismatch squiggles** — shape clashes underlined in red, *live on save* from the
  static pass (no run needed) and from the real trace;
- a node panel with **isolate this block**, the **LLM assistant**
  (`explain` / `why flagged` / `suggest fix`), and a **view:** box that turns a
  prompt into a graph re-styling;
- **Run & Trace (Profiled)** — captures per-layer wall-time; the graph's `cost:`
  selector then recolors nodes by time / memory / params (the fat layer glows red);
- **Diff with Last Trace** — edit your model, trace again, and this paints what
  changed (green = added, amber = changed) with shape/param deltas in the panel;
- non-blocking, cancellable runs (a big model won't freeze the editor).

```bash
cd extension && npm install && npm run compile
# then: Run ▸ Start Debugging ("Run netscope Extension"). A second window opens
# with the extension live. Set netscope.pythonPath to your venv, open a .py file,
# and click the CodeLens (or ⌘⌥T). Run `netscope: Check Setup` if anything's off.
```

> On macOS, **F5** is often a system key — use **Run ▸ Start Debugging** (or the
> Run-and-Debug sidebar) instead. Or install the packaged `.vsix`:
> `cursor --install-extension netscope-*.vsix`.

Keyboard: **⌘⌥T** / **Ctrl+Alt+T** = Run & Trace, **⌘⌥G** / **Ctrl+Alt+G** = Show Graph.

### LLM assistant (optional)

The assistant talks to **any OpenAI-compatible endpoint** — OpenRouter by default
(→ many cheap models like Gemini Flash), or OpenAI / Together / Groq / a local
server. It's entirely optional: with no key, every other feature works offline.

- **In the editor:** run **`netscope: Set LLM API Key`** from the command palette.
  Your key is stored in the **OS keychain** (VSCode SecretStorage) — never in
  `settings.json`, never synced, never in git. Pick the model / endpoint via the
  `netscope.llm.model` and `netscope.llm.baseUrl` settings (these are not secret).
- **From the library / scripts:** set an env var instead —
  `OPENROUTER_API_KEY` (or `NETSCOPE_LLM_API_KEY` / `OPENAI_API_KEY`), with
  optional `NETSCOPE_LLM_MODEL` and `NETSCOPE_LLM_BASE_URL`. Then:
  ```python
  import netscope.llm as nl
  if nl.available():
      print(nl.explain(graph, node_id, question="why_warn"))
  ```

## MCP server — ground your coding agent in the real graph

netscope ships an **MCP server** so a coding agent (Cursor, Claude Code, …) can
query your model's *real* structure instead of guessing — "what actually flows
into `model.layers.2`?", "are there wiring mismatches in this file?". It's
stdlib-only (JSON-RPC over stdio, no extra deps) and needs no LLM key for the
first three tools.

Tools: **`trace_file`** (graph of a file — static, or a real run), **`query_node`**
(a node's real shapes / dtype / neighbours / mismatch), **`list_mismatches`**
(wiring clashes as structured data + source loc), **`explain_node`** (grounded
Q&A, if an LLM key is set).

Register the command `python -m netscope.mcp` with your agent. For example, in a
`.cursor/mcp.json` (or Claude Code's MCP config):

```json
{
  "mcpServers": {
    "netscope": { "command": "/path/to/.venv/bin/python", "args": ["-m", "netscope.mcp"] }
  }
}
```

See `examples/mcp_server_demo.py` for the tools driven in-process.

## Architecture

```
import netscope ─► wrapt post-import hooks patch torch + transformers
                 │  (gated by an active capture session; capture-once)
                 ▼
   contextvars parent-stack ──► typed IR over networkx.DiGraph
       {kind, parent, source, loc{file,line}, meta{shape,dtype,device,params,bytes,time}}
                 │
   ┌─────────────┼───────────────┬──────────────┬────────────────┐
   ▼             ▼               ▼              ▼                ▼
 dataflow    stage-flow      checks         sinks            static AST
 (tensor id) (hints)      (mismatches)  HTML/JSON/mermaid   producer
                 │                              │                │
                 └──────────── merge-by-loc ────┴────────────────┘
                                   │
              shared Cytoscape renderer (web/template.html)
              ── reused verbatim by the VSCode webview ──
```

Each layer is independently usable, and the newer features are just more consumers
of that one IR — **trace diffing** compares two graphs, the **cost heatmap** and
**role lens** are enrichments over node metadata, the **playground** re-renders on
each edit. The renderer libs are vendored and **inlined into the generated HTML**,
so a graph is one self-contained file — works offline and inside locked-down webviews.

## Develop

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/          # 182 passed, 1 skipped (FLOPs: thop optional)
cd extension && npm run test:unit && npm run test:headless
```

Optional extras: `pip install -e ".[flops]"` (THOP per-layer FLOPs),
`".[otel]"` (OpenTelemetry export seam).

New here? **[CONTRIBUTING.md](https://github.com/eren23/netscope/blob/main/CONTRIBUTING.md)** has the dev workflow + the two
cross-language sync rules, and **[docs/API.md](https://github.com/eren23/netscope/blob/main/docs/API.md)** is the full API
reference. Changes are logged in **[CHANGELOG.md](https://github.com/eren23/netscope/blob/main/CHANGELOG.md)**.

## License

MIT — see [LICENSE](https://github.com/eren23/netscope/blob/main/LICENSE).
