from __future__ import annotations

import netscope


def test_default_capture_is_empty():
    with netscope.graph("g") as g:  # noqa: F841
        cap = netscope.active_capture()
        assert cap.capture == frozenset()
        assert cap.wants("attention") is False


def test_capture_kwarg_sets_flags():
    with netscope.graph("g", capture={"attention", "kv_cache"}):
        cap = netscope.active_capture()
        assert cap.wants("attention") and cap.wants("kv_cache")


def test_env_capture_unions_with_kwarg(monkeypatch):
    monkeypatch.setenv("NETSCOPE_CAPTURE", "kv_cache")
    with netscope.graph("g", capture={"attention"}):
        cap = netscope.active_capture()
        assert cap.wants("attention") and cap.wants("kv_cache")


def test_unknown_flag_warns_not_raises(recwarn):
    with netscope.graph("g", capture={"bogus"}):
        cap = netscope.active_capture()
        assert cap.wants("bogus") is False          # dropped
    assert any("bogus" in str(w.message) for w in recwarn.list)
