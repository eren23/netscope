"""The instrumentor registry — the framework-extensibility seam.

A framework registers a session-scoped instrumentor (any object with
``on_enter() -> handle`` / ``on_exit(handle)``); ``capture.graph()`` activates
them on entry and tears them down on exit, so e.g. torch's global forward hooks
exist only while a session is open (zero hooks => zero overhead otherwise).
Registration is idempotent per type, and a misbehaving instrumentor must never
break a session — ``enter_session`` / ``exit_session`` swallow its exceptions.
"""
from __future__ import annotations

import pytest

from netscope.core import registry


@pytest.fixture
def reg():
    """Snapshot + restore the module-global list so a test can register/clear
    freely without disturbing the torch instrumentor other tests rely on."""
    snapshot = list(registry._session_instrumentors)
    try:
        yield registry
    finally:
        registry._session_instrumentors[:] = snapshot


class _Recorder:
    """Minimal instrumentor: records the enter/exit calls it receives."""

    def __init__(self):
        self.entered = 0
        self.exited = []

    def on_enter(self):
        self.entered += 1
        return f"handle-{self.entered}"

    def on_exit(self, handle):
        self.exited.append(handle)


def test_register_is_idempotent_per_type(reg):
    reg.clear()
    a, b = _Recorder(), _Recorder()
    reg.register_session_instrumentor(a)
    reg.register_session_instrumentor(b)          # same type -> second is ignored
    assert reg._session_instrumentors == [a]


def test_enter_then_exit_roundtrip(reg):
    reg.clear()
    rec = _Recorder()
    reg.register_session_instrumentor(rec)
    handles = reg.enter_session()
    assert handles == [(rec, "handle-1")]
    assert rec.entered == 1
    reg.exit_session(handles)
    assert rec.exited == ["handle-1"]


def test_clear_empties_the_registry(reg):
    reg.register_session_instrumentor(_Recorder())
    reg.clear()
    assert reg._session_instrumentors == []


def test_enter_swallows_a_bad_instrumentor(reg):
    reg.clear()

    class _Boom:
        def on_enter(self):
            raise RuntimeError("on_enter blew up")

        def on_exit(self, handle):
            pass

    good = _Recorder()
    reg.register_session_instrumentor(_Boom())     # distinct type
    reg.register_session_instrumentor(good)        # distinct type -> also registered
    # _Boom.on_enter raises and is swallowed; it contributes no handle, while the
    # good instrumentor still activates.
    assert reg.enter_session() == [(good, "handle-1")]


def test_exit_swallows_a_bad_instrumentor(reg):
    reg.clear()

    class _BadExit:
        def on_enter(self):
            return "h"

        def on_exit(self, handle):
            raise RuntimeError("on_exit blew up")

    reg.register_session_instrumentor(_BadExit())
    handles = reg.enter_session()
    reg.exit_session(handles)        # must not propagate — reaching the next line is the assertion
