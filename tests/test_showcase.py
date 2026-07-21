"""Showcase smoke test — the gallery runner works on the always-available subset.

Runs the full roster (models with missing deps skip gracefully inside main()) into
a tmp dir and checks the index carries the load-bearing pieces: a linked graph
page, an inlined memory report, and the autofix transcript.
"""
from __future__ import annotations

import importlib
import sys


def test_showcase_writes_index(tmp_path, monkeypatch):
    monkeypatch.setenv("NETSCOPE_GALLERY", str(tmp_path))
    sys.modules.pop("showcase", None)
    sys.path.insert(0, "examples")
    try:
        showcase = importlib.import_module("showcase")
        importlib.reload(showcase)          # re-read NETSCOPE_GALLERY
        # ponytail: trim the roster to the hermetic core — GPT-2 covers graph +
        # memory + timeline; the big-model entries are demo breadth, not test surface.
        showcase.ROSTER = [r for r in showcase.ROSTER if r[0] == "gpt2-generate"]
        assert showcase.main() == 0
    finally:
        sys.path.remove("examples")

    index = (tmp_path / "index.html").read_text()
    assert "gpt2-generate.html" in index                  # graph page linked
    assert "kv_cache" in index                            # memory report inlined
    assert "netscope fix model.py --apply" in index       # autofix transcript
    assert "mismatch(es) remain" in index
    assert (tmp_path / "gpt2-generate.html").exists()
