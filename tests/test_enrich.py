"""M5: enrichers — per-module parameter counts (always on) and FLOPs (opt-in).

Params are free (`sum(p.numel())`) and framework-trivial, so the torch
instrumentor records them on every module node at capture time. We record
*own* params (recurse=False) so a container's params aren't double-counted by
its children. FLOPs are heavier (need a real forward) so they live behind an
explicit, best-effort opt-in call.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope
from netscope.enrich.params import own_params, total_params


def test_own_vs_total_params():
    lin = nn.Linear(4, 8)  # weight 4*8=32 + bias 8 = 40
    assert own_params(lin) == 40
    seq = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))  # 40 + (16+2)=18 => 58
    assert own_params(seq) == 0        # container holds no direct params
    assert total_params(seq) == 58


def test_capture_records_params_on_module_nodes():
    model = nn.Linear(4, 8)
    with netscope.graph("p") as g:
        model(torch.randn(2, 4))
    lin = next(n for n in g.nodes() if "Linear" in n["name"])
    assert lin["meta"]["params"] == 40


def test_container_node_has_zero_own_params_but_children_have_them():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    with netscope.graph("p") as g:
        model(torch.randn(2, 4))
    seq = next(n for n in g.nodes() if "Sequential" in n["name"])
    lin = next(n for n in g.nodes() if "Linear" in n["name"])
    assert seq["meta"]["params"] == 0
    assert lin["meta"]["params"] == 40


# --- FLOPs (opt-in, best-effort) --------------------------------------------
def test_count_flops_positive_for_linear():
    from netscope.enrich.flops import count_flops, flops_available

    if not flops_available():
        import pytest

        pytest.skip("thop not installed")
    macs = count_flops(nn.Linear(4, 8), torch.randn(1, 4))
    assert macs is not None and macs > 0


def test_count_flops_returns_none_on_failure():
    """Best-effort: a model that can't be profiled yields None, never raises."""
    from netscope.enrich.flops import count_flops

    class Bad(nn.Module):
        def forward(self, x):
            raise RuntimeError("nope")

    assert count_flops(Bad(), torch.randn(1, 4)) is None


def test_own_param_bytes_scales_with_dtype():
    """A half-precision layer reads as half the memory of its float32 twin, and a
    container owns 0 bytes directly (recurse=False, no double-count)."""
    from netscope.enrich.params import own_param_bytes

    lin = nn.Linear(4, 10)                 # 40 weights + 10 bias = 50 params
    assert own_param_bytes(lin) == 50 * 4          # float32 default -> 4 bytes/param
    assert own_param_bytes(nn.Linear(4, 10).half()) == 50 * 2   # float16 -> half
    assert own_param_bytes(nn.Sequential(nn.Linear(4, 10))) == 0  # container: nothing direct


def test_params_never_raise_on_broken_module():
    """The param helpers are always-on enrichment, so a module whose .parameters()
    misbehaves must degrade to 0, never take down the trace."""
    from netscope.enrich.params import own_params, own_param_bytes, total_params

    class Boom:
        def parameters(self, recurse=True):
            raise RuntimeError("no params here")

    b = Boom()
    assert own_params(b) == 0
    assert own_param_bytes(b) == 0
    assert total_params(b) == 0
