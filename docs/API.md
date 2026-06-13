# netscope API reference

The public surface, in one place. Everything is `import netscope` unless noted.
See the [README](../README.md) for the pitch and [examples/](../examples) for
runnable demos.

## Capture

### `netscope.graph(name="", *, profile=False, capture=set()) -> contextmanager[NVGraph]`
Open a capture session. Everything that runs inside is auto-traced; the `with`
block yields the live `NVGraph`.

```python
with netscope.graph("mlp") as g:
    model(x)
g.show()
```

- `profile=True` also records per-layer **wall-time** (`meta.time_ms`); off by
  default so the trace stays metadata-only and ~zero overhead. Activation + param
  **memory** (`meta.act_bytes` / `param_bytes`) are recorded for free regardless.
- Sessions **do not nest** — a second `graph()` inside an open one raises.
- Env override: `NETSCOPE_PROFILE=1` forces `profile=True` (how the extension's
  "Run & Trace (Profiled)" turns it on without editing your code).

#### `capture=` — opt-in deeper LLM views

Pass a set of flag strings to enable richer captures on attention and KV-cache
nodes. The default (no flags) is metadata-only and unchanged — zero overhead, no
tensor retention.

```python
with netscope.graph("gpt2", capture={"attention", "kv_cache"}) as g:
    model.generate(input_ids, max_new_tokens=20)
```

| flag | what is recorded |
|---|---|
| `"kv_cache"` | KV-cache **shapes** on each module node (`meta.kv_cache = {layers, shape, seq}`). Shapes only — no tensors are retained. `netscope.timeline(g)` gains a `kv_seq` field per step so you can watch the cache grow across decode steps. |
| `"attention"` | Attention weights are captured transiently and immediately reduced to **per-head scalars** (`meta.attn_heads = [{entropy, dist, last}, ...]`): `entropy` = focus (low → sharp), `dist` = mean distance back into the sequence a head attends, `last` = fraction of mass on the final key. The HTML graph gains an `⊕ attention` overlay (nodes colored by mean entropy) and a per-head table in the node detail panel. For HF models netscope requests `output_attentions=True` automatically while capturing. |

Memory note: attention → per-head scalars only; KV → shapes only. Raw tensors
are never retained in either mode.

Env override: `NETSCOPE_CAPTURE=attention,kv_cache` (comma-separated flag names).

### Semantic markers (context manager **or** decorator; no-ops outside a session)
Auto-tracing sees *calls*, not *intent* — name regions the tracer can't infer:

| marker | use |
|---|---|
| `netscope.stage(name, *, reduce=False, **attrs)` | a pipeline stage |
| `netscope.branch(name, **attrs)` | one parallel branch |
| `netscope.reduce(name, **attrs)` | a fan-in / vote |
| `netscope.step(label=None)` | one generation/decode step (builds the timeline) |

```python
with netscope.stage("plan"):  plan = planner(x)
for b in range(5):
    with netscope.branch(f"diffuse[{b}]"):  cand = refiner(plan)
with netscope.reduce("vote"):  winner = majority(cands)
```

### `netscope.active_capture()` · `netscope.is_capturing()` · `netscope.install()`
The live `Capture` (or `None`), whether a session is open, and the (idempotent)
instrumentation installer (called on import).

## The graph (`NVGraph`)

Yielded by `graph()`; also constructible from a saved trace.

| method | returns |
|---|---|
| `g.show(path=None, open_browser=True)` | write a self-contained interactive HTML, open it; returns the path |
| `g.to_html(title=None)` | the standalone HTML string |
| `g.to_json(indent=2)` / `g.to_dict()` | the serialized IR (nodes, edges, warnings) |
| `g.to_mermaid()` | a Mermaid diagram string |
| `g.nodes()` / `g.edges()` | the node / edge dicts |
| `NVGraph.from_dict(d)` | rebuild a graph from `to_dict()` output (classmethod) |

A node: `{id, kind(pipeline|stage|model|module|op), name, parent, source, loc{file,line},
meta{out_shape,in_shape,dtype,device,params,act_bytes,param_bytes,time_ms,qualname,role}, attrs}`.

## Analysis over a trace

### `netscope.diff(before, after) -> dict`
What changed between two traces. Returns `{added, removed, changed, same, summary}`;
`changed` entries carry `fields` + `before`/`after` snapshots. Keyed by a stable
identity (qualname › loc › name) so it survives an inserted layer.

### `netscope.diff_view(before, after) -> NVGraph`
The `after` graph tagged `attrs.diff` per node (`added`/`changed`/`same`, plus
`removed` ghosts) — `.show()` it for a green/amber colored diff.

### `netscope.roles(graph) -> dict`
Architectural-role breakdown — `{"attention": n, "mlp": n, "norm": n, ...}` — from
module naming. The graph's "⊕ role" overlay colors by the same classification.

### `netscope.timeline(graph) -> list`
Ordered per-step summary of an autoregressive trace (the `step()` markers):
`[{step, label, time_ms, modules, out_shape, kv_seq}, ...]` — watch `out_shape`'s
sequence axis grow across decode steps. `kv_seq` is the KV-cache sequence length
at that step (`None` unless `capture={"kv_cache"}` was active).

## LLM layer (optional, bring-your-own-key)

`import netscope.llm as nl`. Unavailable (raises `LLMUnavailable`) with no key
configured; everything else works offline.

- `nl.available() -> bool`
- `nl.explain(graph, node_id, *, question="explain") -> str` — grounded Q&A;
  `question ∈ {"explain", "why_warn", "suggest_fix"}`.
- `nl.infer(graph, source, filename="<source>") -> NVGraph` — augment with
  LLM-inferred provisional (dashed, confidence-scored) structure.

## Command-line

After `pip install`, a unified **`netscope`** command wraps the tools:

```bash
netscope static  model.py                       # static graph (declared-dim checks, no run)
netscope playground [port]                       # local live editor ⇄ graph (default :8770)
netscope mcp                                      # MCP server (JSON-RPC over stdio) for agents
netscope diff    before.json after.json [--html out | --graph-json out]
netscope views   graph.json "highlight attention"
```

Each maps to a module entry point, which also works directly:

```bash
python -m netscope <cmd> ...                      # same dispatcher as `netscope`
python -m netscope.static  <file.py>
python -m netscope.playground [port]
python -m netscope.mcp
python -m netscope.core.diff  before.json after.json [...]
python -m netscope.llm.views  graph.json "highlight attention"
python -m netscope.llm        graph.json <node_id> <question>     # one-shot assistant
```

## Environment variables

| var | effect |
|---|---|
| `NETSCOPE_OUT` | dump the trace JSON to this path on session exit (the extension reads it) |
| `NETSCOPE_PROFILE` | `1` forces `profile=True` |
| `NETSCOPE_CAPTURE` | comma-separated flag names — forces `capture=` without editing code (e.g. `NETSCOPE_CAPTURE=attention,kv_cache`) |
| `NETSCOPE_ISOLATE` / `NETSCOPE_ISOLATE_OUT` | re-run just one submodule, dump the focused sub-trace |
| `NETSCOPE_LLM_API_KEY` › `OPENROUTER_API_KEY` › `OPENAI_API_KEY` | LLM key (first non-empty wins) |
| `NETSCOPE_LLM_MODEL` / `NETSCOPE_LLM_BASE_URL` | model + endpoint (default: OpenRouter) |
