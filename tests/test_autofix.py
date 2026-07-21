"""Self-healing shapes — deterministically fix a declared-dim clash.

A static shape_mismatch (`nn.Linear(128, …)` fed something that emits 256) has a
mechanical fix: change the consumer's declared in-dim to what the producer emits,
at the consumer's source line. netscope proposes that edit (dry-run) and, on
apply, re-analyzes to prove the clash is gone. Deterministic, offline, no LLM key.
"""
from __future__ import annotations

import os
import tempfile
import textwrap

from netscope.autofix import apply_fixes, propose_fixes
from netscope.core.checks import detect_mismatches
from netscope.static.ast_producer import analyze_file

_SRC = textwrap.dedent(
    """
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(4, 256)
            self.head = nn.Linear(128, 10)   # expects 128, but enc emits 256

        def forward(self, x):
            return self.head(self.enc(x))
    """
)


def _write_tmp(src: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), "model.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return path


def test_propose_targets_the_consumer_in_dim():
    fixes = propose_fixes(analyze_file(_write_tmp(_SRC)))
    assert len(fixes) == 1
    fx = fixes[0]
    assert fx["qualname"] == "head"
    assert "nn.Linear(128, 10)" in fx["old"]
    assert "nn.Linear(256, 10)" in fx["new"]     # in-dim 128 -> 256, out-dim 10 untouched


def test_apply_then_reanalyze_is_clean():
    path = _write_tmp(_SRC)
    assert detect_mismatches(analyze_file(path))          # clash present first
    n = apply_fixes(propose_fixes(analyze_file(path)))
    assert n == 1
    assert detect_mismatches(analyze_file(path)) == []    # gone after the fix
    assert "nn.Linear(256, 10)" in open(path, encoding="utf-8").read()


_SRC_KW = _SRC.replace("nn.Linear(128, 10)", "nn.Linear(in_features=128, out_features=10)")


def test_fixes_keyword_in_features():
    fixes = propose_fixes(analyze_file(_write_tmp(_SRC_KW)))
    assert len(fixes) == 1
    assert "in_features=256" in fixes[0]["new"]           # only the kwarg value changes
    assert "out_features=10" in fixes[0]["new"]


def test_no_clash_proposes_nothing():
    ok = _SRC.replace("nn.Linear(128, 10)", "nn.Linear(256, 10)")   # wired correctly
    assert propose_fixes(analyze_file(_write_tmp(ok))) == []


def test_cli_dry_run_then_apply():
    from netscope.autofix import _main
    path = _write_tmp(_SRC)
    assert _main([path]) == 0                                   # dry-run
    assert "nn.Linear(128, 10)" in open(path, encoding="utf-8").read()   # untouched
    assert _main([path, "--apply"]) == 0                        # write
    assert "nn.Linear(256, 10)" in open(path, encoding="utf-8").read()
    assert _main([]) == 2                                       # usage error, no path
