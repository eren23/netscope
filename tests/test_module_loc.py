"""M0: every runtime module node carries a source `loc` (file + line).

The torch tracer records each module's qualified name (e.g. "blocks.1.attn").
This maps that qualname back to the line where the submodule was CONSTRUCTED in
the model's source (`self.attn = nn.Linear(...)`), so the editor can jump from a
graph node to the exact line — and inline shape hints / mismatch squiggles can
sit on it. Pure AST, no execution.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope
from netscope.static.module_loc import qualname_locs


# ---- the static helper, in isolation -------------------------------------
SRC = '''
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        return self.head(self.act(self.encoder(x)))
'''


def test_qualname_locs_maps_attr_assignments_to_lines():
    locs = qualname_locs(SRC, "net.py")
    # self.encoder is on line 7 (1-based, counting the leading blank line)
    assert locs["encoder"]["line"] == 7
    assert locs["act"]["line"] == 8
    assert locs["head"]["line"] == 9
    assert locs["encoder"]["file"] == "net.py"


SEQ_SRC = '''
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.body(x)
'''


def test_qualname_locs_handles_sequential_index_naming():
    locs = qualname_locs(SEQ_SRC, "m.py")
    # named_modules() names Sequential children "body.0", "body.1", "body.2"
    assert "body" in locs
    # each indexed child resolves to its own construction line inside the call
    assert locs["body.0"]["line"] == 8
    assert locs["body.1"]["line"] == 9
    assert locs["body.2"]["line"] == 10


# ---- end-to-end: runtime nodes now carry loc -----------------------------
class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        return self.head(self.encoder(x))


def test_runtime_module_nodes_get_loc_from_source():
    model = _Tiny().train(False)
    with netscope.graph("tiny") as g, torch.no_grad():
        model(torch.randn(1, 8))

    by_qual = {(n.get("meta") or {}).get("qualname"): n for n in g.nodes()}
    enc = by_qual.get("encoder")
    head = by_qual.get("head")
    assert enc is not None and head is not None
    # both now carry a loc pointing at THIS file's construction lines
    assert enc["loc"] is not None, "encoder node should have a loc"
    assert head["loc"] is not None, "head node should have a loc"
    assert enc["loc"]["file"].endswith("test_module_loc.py")
    # encoder is constructed before head -> strictly smaller line number
    assert enc["loc"]["line"] < head["loc"]["line"]


def test_loc_is_best_effort_never_raises_for_dynamic_modules():
    """A module with no resolvable source (built inline / no file) must not crash
    the trace; loc just stays None."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    with netscope.graph("seq") as g:
        model(torch.randn(1, 4))
    # the run completes and produces nodes; loc may be None, that's fine
    assert any(n["name"] == "Linear" for n in g.nodes())


def test_root_module_gets_loc_from_class_def():
    """A directly-called custom module (the trace root, qualname '') resolves to
    its class-definition line — so click-to-source works on the model itself, not
    just its submodules. resnet18 has a real source file (torchvision)."""
    from torchvision.models import resnet18
    model = resnet18(weights=None).train(False)
    with netscope.graph("r") as g, torch.no_grad():
        model(torch.randn(1, 3, 64, 64))
    root = next(n for n in g.nodes() if n["parent"] is None)
    assert root["name"].startswith("ResNet")
    assert root["loc"] is not None, "root module should carry a loc"
    assert root["loc"]["file"].endswith("resnet.py")
