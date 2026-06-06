"""Profiler overlay — per-layer cost, captured on the trace.

Two tiers, by the zero-overhead rule:
  * activation + param BYTES are FREE — derivable from the shape/dtype already
    captured (the output tensor is right there in the post-hook) — so they ride on
    every trace, no flag.
  * wall-TIME is a measurement with real overhead, so it is opt-in:
    `netscope.graph(name, profile=True)`. The default trace stays metadata-only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


def _leaf(g, qual):
    return next(n for n in g.nodes() if (n.get("meta") or {}).get("qualname") == qual)


def test_param_bytes_present_and_correct_by_default():
    # Linear(8,16): 8*16 weight + 16 bias = 144 params; float32 -> 144*4 = 576 B
    with netscope.graph("m") as g, torch.no_grad():
        nn.Sequential(nn.Linear(8, 16)).train(False)(torch.randn(2, 8))
    lin = _leaf(g, "0")
    assert lin["meta"]["params"] == 144
    assert lin["meta"]["param_bytes"] == 576


def test_activation_bytes_present_and_correct_by_default():
    # out [2,16] float32 -> 2*16*4 = 128 bytes of activation
    with netscope.graph("m") as g, torch.no_grad():
        nn.Sequential(nn.Linear(8, 16)).train(False)(torch.randn(2, 8))
    lin = _leaf(g, "0")
    assert lin["meta"]["act_bytes"] == 128


def test_no_timing_without_profile_flag():
    # the default trace is metadata-only: NO wall-time measured anywhere.
    with netscope.graph("m") as g, torch.no_grad():
        nn.Sequential(nn.Linear(8, 16), nn.ReLU()).train(False)(torch.randn(2, 8))
    assert all("time_ms" not in (n.get("meta") or {}) for n in g.nodes())


def test_timing_present_with_profile_flag():
    with netscope.graph("m", profile=True) as g, torch.no_grad():
        nn.Sequential(nn.Linear(8, 16), nn.ReLU()).train(False)(torch.randn(2, 8))
    timed = [n for n in g.nodes() if "time_ms" in (n.get("meta") or {})]
    assert timed, "profile=True should record time_ms on traced modules"
    assert all(isinstance(n["meta"]["time_ms"], (int, float)) and n["meta"]["time_ms"] >= 0
               for n in timed)


def test_env_var_forces_profile_without_the_flag(monkeypatch):
    # the extension's "Run & Trace (profiled)" sets NETSCOPE_PROFILE — graph() must
    # honor it even though the user's code never passed profile=True.
    monkeypatch.setenv("NETSCOPE_PROFILE", "1")
    with netscope.graph("m") as g, torch.no_grad():
        nn.Linear(8, 16).train(False)(torch.randn(2, 8))
    assert any("time_ms" in (n.get("meta") or {}) for n in g.nodes())


def test_bytes_scale_with_dtype():
    # half precision halves the byte counts vs float32 for the same shapes.
    lin = nn.Linear(8, 16)
    with netscope.graph("f32") as g32, torch.no_grad():
        lin.train(False)(torch.randn(2, 8))
    with netscope.graph("f16") as g16, torch.no_grad():
        lin.half()(torch.randn(2, 8, dtype=torch.float16))
    b32 = next(n for n in g32.nodes() if n["name"] == "Linear")["meta"]["param_bytes"]
    b16 = next(n for n in g16.nodes() if n["name"] == "Linear")["meta"]["param_bytes"]
    assert b32 == 2 * b16
