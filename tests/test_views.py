"""Generated views: a prompt -> a declarative VIEW SPEC the renderer applies.

"group by attention vs MLP", "highlight params > 1M", "color by dtype" become a
small, validated JSON spec of SAFE operations (filter / highlight / colorBy) —
never arbitrary code. The LLM produces the spec (gated on a key); a pure function
applies it to the cytoscape elements. Tests cover the spec language + application
directly (no network), plus parsing an LLM reply via an injected transport.
"""
from __future__ import annotations

import json

import pytest

from netscope.llm.views import (
    VIEW_SPEC_SCHEMA, apply_view_spec, parse_view_spec, generate_view_spec,
)
from netscope.llm.provider import Provider


def _elements():
    """A few cytoscape-shaped nodes spanning kinds / params / dtypes."""
    return {
        "nodes": [
            {"data": {"id": "a", "name": "Qwen3Attention", "kind": "module",
                      "meta": {"params": 2_000_000, "dtype": "float16"}}},
            {"data": {"id": "b", "name": "Qwen3MLP", "kind": "module",
                      "meta": {"params": 500_000, "dtype": "float16"}}},
            {"data": {"id": "c", "name": "RMSNorm", "kind": "module",
                      "meta": {"params": 1024, "dtype": "float32"}}},
        ],
        "edges": [{"data": {"id": "e0", "source": "a", "target": "b"}}],
    }


# ---- spec schema + validation --------------------------------------------
def test_schema_is_well_formed():
    assert VIEW_SPEC_SCHEMA["type"] == "object"
    assert "ops" in VIEW_SPEC_SCHEMA["properties"]


def test_parse_strips_fences_and_validates():
    fenced = "```json\n" + json.dumps({"ops": [
        {"op": "highlight", "where": {"name_contains": "Attention"}}]}) + "\n```"
    spec = parse_view_spec(fenced)
    assert spec["ops"][0]["op"] == "highlight"


def test_parse_drops_unknown_ops():
    spec = parse_view_spec(json.dumps({"ops": [
        {"op": "rm -rf"},                       # not a known op -> dropped
        {"op": "highlight", "where": {"kind": "module"}}]}))
    assert len(spec["ops"]) == 1 and spec["ops"][0]["op"] == "highlight"


def test_parse_nonjson_yields_empty_spec():
    assert parse_view_spec("sorry, I can't") == {"ops": []}


# ---- applying ops --------------------------------------------------------
def test_highlight_marks_matching_nodes():
    spec = {"ops": [{"op": "highlight", "where": {"name_contains": "Attention"}}]}
    out = apply_view_spec(_elements(), spec)
    hi = [n for n in out["nodes"] if n["data"].get("vhi")]
    assert len(hi) == 1 and hi[0]["data"]["id"] == "a"


def test_filter_dims_nonmatching_nodes():
    # show only the MLP -> others get vdim
    spec = {"ops": [{"op": "filter", "where": {"name_contains": "MLP"}}]}
    out = apply_view_spec(_elements(), spec)
    dimmed = [n["data"]["id"] for n in out["nodes"] if n["data"].get("vdim")]
    assert "a" in dimmed and "c" in dimmed and "b" not in dimmed


def test_highlight_by_param_threshold():
    spec = {"ops": [{"op": "highlight", "where": {"params_gt": 1_000_000}}]}
    out = apply_view_spec(_elements(), spec)
    hi = {n["data"]["id"] for n in out["nodes"] if n["data"].get("vhi")}
    assert hi == {"a"}   # only the 2M-param attention


def test_color_by_dtype_assigns_stable_colors():
    spec = {"ops": [{"op": "colorBy", "field": "dtype"}]}
    out = apply_view_spec(_elements(), spec)
    colors = {n["data"]["id"]: n["data"].get("vcolor") for n in out["nodes"]}
    # same dtype -> same color; different dtype -> different color
    assert colors["a"] == colors["b"]          # both float16
    assert colors["a"] != colors["c"]          # float16 vs float32
    assert colors["a"] is not None


def test_apply_is_pure_and_ignores_bad_where():
    el = _elements()
    spec = {"ops": [{"op": "highlight", "where": {"bogus": 1}}]}
    out = apply_view_spec(el, spec)
    # nothing matched, no crash, original untouched
    assert not any(n["data"].get("vhi") for n in out["nodes"])


# ---- generate via the provider (injected transport) ----------------------
def test_generate_view_spec_from_prompt():
    def transport(url, headers, body):
        # the model should only see field names + the prompt; it returns a spec.
        payload = {"ops": [{"op": "highlight", "where": {"name_contains": "Attention"}}]}
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}
    p = Provider(base_url="http://x/v1", model="m", api_key="k", extra_headers={})
    spec = generate_view_spec("highlight the attention blocks", _elements(),
                              provider=p, _transport=transport)
    assert spec["ops"][0]["op"] == "highlight"
