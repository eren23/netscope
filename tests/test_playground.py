"""The playground's analysis backend (`python -m netscope.playground`).

The HTTP layer is thin; the logic worth testing is `analyze()` — it must trace a
real snippet, run the static checks on source text, and raise a clear error when
the snippet doesn't define what tracing needs.
"""
from __future__ import annotations

import pytest

from netscope.playground import _trace_code, analyze

_MLP = (
    "import torch, torch.nn as nn\n"
    "model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))\n"
    "x = torch.randn(2, 8)\n"
)

_BUGGY_CLASS = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.enc = nn.Linear(64, 256)\n"
    "        self.head = nn.Linear(128, 10)\n"
    "    def forward(self, x):\n"
    "        return self.head(self.enc(x))\n"
)


def test_trace_mode_returns_html_and_real_nodes():
    out = analyze(_MLP, "trace", False)
    assert out["ok"] is True
    assert out["nodes"] >= 3
    assert "<html" in out["html"].lower()


def test_profile_mode_traces_without_error():
    out = analyze(_MLP, "trace", True)     # profile=True
    assert out["ok"] is True and out["nodes"] >= 3


def test_static_mode_flags_a_declared_dim_clash_without_running():
    out = analyze(_BUGGY_CLASS, "static", False)
    assert out["ok"] is True
    assert out["warnings"] >= 1            # 256 -> 128 caught from source, no run


def test_missing_model_raises_a_clear_error():
    with pytest.raises(ValueError):
        _trace_code("x = 1\n", False)      # no `model` defined
