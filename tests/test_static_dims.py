"""M2: declared-dim pre-check — catch a wiring clash BEFORE running.

The static AST producer reads literal layer dims (`nn.Linear(in, out)`,
`nn.Conv2d(cin, cout, ...)`) and the `forward` call order, emits dataflow edges
with the declared shapes, and the existing mismatch detector flags an obvious
clash — all without executing a single forward. Conservative: only when both
dims are literal and the wiring is direct.
"""
from __future__ import annotations

import textwrap

from netscope.static.ast_producer import analyze_source


CLASH = textwrap.dedent('''
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(64, 256)   # emits 256
            self.head = nn.Linear(128, 10)      # but expects 128

        def forward(self, x):
            return self.head(self.encoder(x))
''')


def test_clash_is_flagged_statically_with_loc():
    g = analyze_source(CLASH, filename="net.py")
    warns = g.to_dict()["warnings"]
    assert len(warns) == 1, f"expected exactly one static mismatch, got {warns}"
    w = warns[0]
    assert w["kind"] == "shape_mismatch"
    assert "256" in w["detail"] and "128" in w["detail"]
    # the message names the layers by attribute (qualname), so two Linears are
    # distinguishable — "encoder emits ... head expects ...".
    assert "encoder" in w["detail"] and "head" in w["detail"]
    # the offending consumer is `head`; its node must carry a loc so the editor
    # can squiggle the right line (the self.head = ... line, line 8).
    nodes = {n["id"]: n for n in g.nodes()}
    dst = nodes[w["dst"]]
    assert dst["loc"]["line"] == 8


def test_static_nodes_carry_declared_dims_and_qualname():
    g = analyze_source(CLASH, filename="net.py")
    by_qual = {(n.get("meta") or {}).get("qualname"): n for n in g.nodes()}
    enc = by_qual["encoder"]
    head = by_qual["head"]
    assert enc["meta"]["out_shape"] == [1, 256]
    assert head["meta"]["in_shape"] == [1, 128]
    assert enc["source"] == "static"


CORRECT = textwrap.dedent('''
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(64, 128)
            self.head = nn.Linear(128, 10)

        def forward(self, x):
            return self.head(self.encoder(x))
''')


def test_correct_wiring_has_no_static_warning():
    g = analyze_source(CORRECT, filename="net.py")
    assert g.to_dict()["warnings"] == []


AMBIGUOUS = textwrap.dedent('''
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.encoder = nn.Linear(64, d)     # non-literal -> can't check
            self.head = nn.Linear(128, 10)

        def forward(self, x):
            return self.head(self.encoder(x))
''')


def test_non_literal_dims_are_not_flagged():
    """Conservative: a runtime-variable dim is silenced, never a false alarm."""
    g = analyze_source(AMBIGUOUS, filename="net.py")
    assert g.to_dict()["warnings"] == []


# --- B1: kwargs in constructors -----------------------------------------------
KWARGS_CLASH = textwrap.dedent('''
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(in_features=64, out_features=256)
            self.head = nn.Linear(in_features=128, out_features=10)
        def forward(self, x):
            return self.head(self.encoder(x))
''')


def test_kwargs_constructors_are_dim_checked():
    """nn.Linear(in_features=.., out_features=..) — the idiomatic kwargs style —
    must still feed the dim pre-check (was silently 0 warnings)."""
    g = analyze_source(KWARGS_CLASH, filename="k.py")
    warns = g.to_dict()["warnings"]
    assert len(warns) == 1, f"kwargs clash not flagged, got {warns}"
    assert "256" in warns[0]["detail"] and "128" in warns[0]["detail"]


def test_conv_kwargs_dims():
    src = textwrap.dedent('''
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
                self.b = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1)
            def forward(self, x):
                return self.b(self.a(x))
    ''')
    g = analyze_source(src, filename="c.py")
    assert len(g.to_dict()["warnings"]) == 1   # 64 out != 32 in


# --- B2: intermediate-variable wiring -----------------------------------------
INTERMEDIATE = textwrap.dedent('''
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 256)
            self.b = nn.Linear(128, 10)
        def forward(self, x):
            h = self.a(x)
            return self.b(h)
''')


def test_intermediate_variable_wiring_is_detected():
    """`h = self.a(x); self.b(h)` — the COMMON forward() style — must wire a->b
    (was 0 edges, so the pre-check never fired on real code)."""
    g = analyze_source(INTERMEDIATE, filename="iv.py")
    edges = [e for e in g.edges() if e["kind"] == "dataflow"]
    assert len(edges) >= 1, "intermediate-var wiring produced no dataflow edge"
    warns = g.to_dict()["warnings"]
    assert len(warns) == 1, f"clash through an intermediate var not flagged: {warns}"


def test_reassigned_variable_chain():
    """`x = self.a(x); x = self.b(x)` (reassigning the same name) chains a->b."""
    src = textwrap.dedent('''
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 256)
                self.b = nn.Linear(128, 10)
            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                return x
    ''')
    g = analyze_source(src, filename="re.py")
    assert len(g.to_dict()["warnings"]) == 1


# --- B3: Sequential / ModuleList children -------------------------------------
def test_sequential_children_are_visible_and_chained():
    """nn.Sequential(a, ReLU, b) — children are named seq.0/seq.2 (mirroring
    named_modules), chained implicitly, so an internal dim clash is caught."""
    src = textwrap.dedent('''
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Linear(64, 256),
                    nn.ReLU(),
                    nn.Linear(128, 10),
                )
            def forward(self, x):
                return self.seq(x)
    ''')
    g = analyze_source(src, "seq.py")
    quals = {(n.get("meta") or {}).get("qualname") for n in g.nodes()}
    assert "seq.0" in quals and "seq.2" in quals
    warns = g.to_dict()["warnings"]
    assert len(warns) == 1
    assert "256" in warns[0]["detail"] and "128" in warns[0]["detail"]


def test_clean_sequential_has_no_warning():
    src = textwrap.dedent('''
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
            def forward(self, x):
                return self.seq(x)
    ''')
    g = analyze_source(src, "ok.py")
    assert g.to_dict()["warnings"] == []
