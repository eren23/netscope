"""M4 backbone (Python side): the two seams the VSCode extension talks to.

1. Static CLI: `python -m netscope.static <file.py>` emits the static graph JSON
   the extension draws as the on-type skeleton (no execution of the target).
2. File sink: a traced run, when NETSCOPE_OUT is set (the extension sets it when
   launching "Run & Trace"), writes the runtime graph JSON on session exit so
   the extension can read it back. Both are plain JSON in the SCHEMA_VERSION IR.
"""
from __future__ import annotations

import json
import os
import textwrap

import netscope
from netscope.static.cli import render_static_json

SRC = textwrap.dedent(
    '''
    def run(p, n):
        for b in range(n):
            diffuse(p, b)
        Counter(answers).most_common(1)
    '''
)


def test_static_cli_emits_schema_versioned_json(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(SRC)
    d = json.loads(render_static_json(str(f)))
    assert d["schema_version"] == netscope.SCHEMA_VERSION
    names = [n["name"] for n in d["nodes"]]
    assert any(n.startswith("branch") or "loop" in n for n in names)  # the for-range
    assert "vote" in names                                            # the most_common
    assert all(n["source"] == "static" for n in d["nodes"])


def test_static_cli_nodes_carry_loc(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(SRC)
    d = json.loads(render_static_json(str(f)))
    for n in d["nodes"]:
        assert n["loc"]["file"].endswith("m.py")
        assert isinstance(n["loc"]["line"], int)


def test_file_sink_writes_runtime_graph_when_env_set(tmp_path, monkeypatch):
    out = tmp_path / "run.netscope.json"
    monkeypatch.setenv("NETSCOPE_OUT", str(out))
    with netscope.graph("run"):
        with netscope.stage("plan"):
            pass
    assert out.exists()
    d = json.loads(out.read_text())
    assert d["schema_version"] == netscope.SCHEMA_VERSION
    assert d["name"] == "run"
    assert any(n["name"] == "plan" for n in d["nodes"])


def test_file_sink_silent_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("NETSCOPE_OUT", raising=False)
    before = set(os.listdir(tmp_path))
    with netscope.graph("run"):
        pass
    assert set(os.listdir(tmp_path)) == before  # nothing written
