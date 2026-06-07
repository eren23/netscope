"""Isolation: trace one part of a network on its own.

Two pieces:
  * every module node carries its qualified name (`model.layers.2`) in meta, so
    a node in the graph maps back to an addressable submodule.
  * setting NETSCOPE_ISOLATE=<qualname> makes a normal traced run, at finalize,
    re-run JUST that submodule on the REAL input tensor that flowed into it and
    dump that focused sub-trace to NETSCOPE_ISOLATE_OUT. No synthetic inputs.
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn

import netscope


def test_module_nodes_carry_qualname():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with netscope.graph("m") as g:
        model(torch.randn(1, 4))
    quals = {(n.get("meta") or {}).get("qualname") for n in g.nodes()}
    # Sequential submodules are named "0","1","2" by named_modules()
    assert "0" in quals
    assert "2" in quals


def test_isolate_reruns_only_target_on_real_input(tmp_path, monkeypatch):
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    iso_out = tmp_path / "iso.json"
    monkeypatch.setenv("NETSCOPE_ISOLATE", "2")          # the second Linear
    monkeypatch.setenv("NETSCOPE_ISOLATE_OUT", str(iso_out))

    with netscope.graph("full"):
        model(torch.randn(1, 4))

    assert iso_out.exists(), "isolation should dump a focused sub-trace"
    data = json.loads(iso_out.read_text())
    names = [n["name"] for n in data["nodes"]]
    # the isolated graph is JUST the target module - no ReLU, no sibling Linear
    assert names == ["Linear"]
    root = data["nodes"][0]
    # ...and it ran on the REAL tensor that reached it: ReLU's [1, 8] output
    assert root["meta"]["in_shape"] == [1, 8]
    assert root["meta"]["out_shape"] == [1, 2]


def test_no_isolation_without_env(tmp_path, monkeypatch):
    """Without NETSCOPE_ISOLATE, a normal run does no extra work / no dump."""
    monkeypatch.delenv("NETSCOPE_ISOLATE", raising=False)
    iso_out = tmp_path / "iso.json"
    monkeypatch.setenv("NETSCOPE_ISOLATE_OUT", str(iso_out))
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    with netscope.graph("m"):
        model(torch.randn(1, 4))
    assert not iso_out.exists()


def test_isolate_handles_kwargs(tmp_path, monkeypatch):
    """A submodule called with keyword args is re-runnable in isolation: the
    stashed call carries both positional and keyword tensors."""

    class TwoArg(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x, *, scale):
            return self.lin(x) * scale

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = TwoArg()

        def forward(self, x):
            return self.block(x, scale=torch.ones(1))

    iso_out = tmp_path / "iso.json"
    monkeypatch.setenv("NETSCOPE_ISOLATE", "block")
    monkeypatch.setenv("NETSCOPE_ISOLATE_OUT", str(iso_out))
    with netscope.graph("net"):
        Net().train(False)(torch.randn(1, 8))

    assert iso_out.exists()
    data = json.loads(iso_out.read_text())
    names = [n["name"] for n in data["nodes"]]
    assert "TwoArg" in names          # the isolated block re-ran successfully
    assert "Linear" in names          # including its child
