"""Demo — generated views: a prompt becomes a declarative VIEW SPEC.

The LLM turns "highlight the attention blocks" / "color by dtype" into a small,
validated JSON spec of SAFE operations (highlight / filter / colorBy) — never
arbitrary code — which the renderer applies. Here we show the spec being applied
WITHOUT an LLM (the apply step is pure); with a key, generate_view_spec(prompt,..)
produces the spec from natural language.

    python examples/views_demo.py
"""
from __future__ import annotations

from netscope.llm.views import apply_view_spec

ELEMENTS = {
    "nodes": [
        {"data": {"id": "a", "name": "Qwen3Attention", "kind": "module",
                  "meta": {"params": 2_000_000, "dtype": "float16"}}},
        {"data": {"id": "b", "name": "Qwen3MLP", "kind": "module",
                  "meta": {"params": 500_000, "dtype": "float16"}}},
        {"data": {"id": "c", "name": "RMSNorm", "kind": "module",
                  "meta": {"params": 1024, "dtype": "float32"}}},
    ],
    "edges": [],
}


def show(label, spec):
    out = apply_view_spec(ELEMENTS, spec)
    flags = {n["data"]["name"]: {k: v for k, v in n["data"].items()
                                 if k in ("vhi", "vdim", "vcolor")}
             for n in out["nodes"]}
    print(f"\n{label}\n  spec: {spec}")
    for name, f in flags.items():
        print(f"    {name:18} {f or '(unchanged)'}")


def main() -> None:
    show("highlight attention", {"ops": [{"op": "highlight",
         "where": {"name_contains": "Attention"}}]})
    show("highlight params > 1M", {"ops": [{"op": "highlight",
         "where": {"params_gt": 1_000_000}}]})
    show("filter to MLP only", {"ops": [{"op": "filter",
         "where": {"name_contains": "MLP"}}]})
    show("color by dtype", {"ops": [{"op": "colorBy", "field": "dtype"}]})
    print("\n(with an LLM key: generate_view_spec('color by dtype', elements) -> the spec above)")


if __name__ == "__main__":
    main()
