"""Level 2 — `scope=` capture: record only a chosen submodule subtree.

`netscope.graph(scope=model.layers[2])` runs the full forward but records nodes
only for the scoped module and its descendants (`scope.modules()`), reusing the
qualname map the tracer already builds. The rest of the model still executes; it
just isn't recorded. Pure library, no editor.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


def _model():
    # qualnames: "0" Linear, "1" Sequential {"1.0" Linear, "1.1" ReLU}, "2" Linear
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.Sequential(nn.Linear(8, 8), nn.ReLU()),
        nn.Linear(8, 2),
    )


def _module_quals(g):
    quals = {(n.get("meta") or {}).get("qualname") for n in g.nodes() if n["kind"] == "module"}
    quals.discard(None)     # the root ("") carries no qualname in meta
    return quals


def test_scope_records_only_the_subtree():
    model = _model()
    with netscope.graph("scoped", scope=model[1]) as g:
        model(torch.randn(2, 4))
    # exactly model[1] and its two children — the [0]/[2] Linears + root are skipped
    assert _module_quals(g) == {"1", "1.0", "1.1"}


def test_no_scope_records_the_whole_model():
    # regression: the default (scope=None) path is unchanged — every module recorded
    model = _model()
    with netscope.graph("full") as g:
        model(torch.randn(2, 4))
    assert _module_quals(g) == {"0", "1", "1.0", "1.1", "2"}


def test_scope_must_be_a_module():
    # trust-boundary validation: a non-module scope fails clearly at setup, before
    # any forward runs (rather than silently tracing everything).
    import pytest
    with pytest.raises(TypeError):
        with netscope.graph("bad", scope="not a module"):
            pass


def test_scoped_subtree_keeps_real_shapes_and_nesting():
    model = _model()
    with netscope.graph("scoped", scope=model[1]) as g:
        model(torch.randn(2, 4))
    by_qual = {(n.get("meta") or {}).get("qualname"): n
               for n in g.nodes() if n["kind"] == "module"}
    lin = by_qual["1.0"]                       # inner Linear(8, 8)
    assert lin["meta"]["in_shape"] == [2, 8] and lin["meta"]["out_shape"] == [2, 8]
    # "1.0"/"1.1" nest under the scoped root "1", which itself sits at the graph root
    assert lin["parent"] == by_qual["1"]["id"]
    assert by_qual["1"]["parent"] is None
