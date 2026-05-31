"""M1: the generic span wrapper that auto-instrumentation is built on.

`wrap_callable` / `safe_patch` wrap any callable so that, *while capturing*, a
call opens a span (nested by the contextvars stack) and records metadata derived
from the call. Crucially, tracing must NEVER break the wrapped call (the MLflow
`safe_patch` rule): if the span machinery throws, the original result still
returns. Outside a session it is a straight pass-through (the zero-overhead gate).
"""
from __future__ import annotations

import netscope
from netscope.instrument.base import wrap_callable


def test_wrapped_callable_creates_span_when_capturing():
    def fn(a, b=2):
        return a + b

    wrapped = wrap_callable(fn, name_fn=lambda inst, args, kwargs: "myfn", kind="op")
    with netscope.graph("g") as g:
        assert wrapped(3, b=4) == 7
    assert any(n["name"] == "myfn" for n in g.nodes())


def test_wrapped_callable_passthrough_when_inactive():
    def fn(a):
        return a * 2

    wrapped = wrap_callable(fn, name_fn=lambda *a: "x", kind="op")
    assert wrapped(5) == 10          # no session -> just calls through
    assert netscope.active_capture() is None


def test_tracing_errors_never_break_the_call():
    def fn():
        return "ok"

    def boom(*a, **k):
        raise RuntimeError("tracer bug")

    wrapped = wrap_callable(fn, name_fn=boom, kind="op")
    with netscope.graph("g"):
        assert wrapped() == "ok"     # span machinery raised, call still returns


def test_meta_fn_is_recorded_on_the_node():
    def fn(**k):
        return 1

    wrapped = wrap_callable(
        fn,
        name_fn=lambda inst, args, kwargs: "gen",
        kind="model",
        meta_fn=lambda inst, args, kwargs: {"max_new_tokens": kwargs.get("max_new_tokens")},
    )
    with netscope.graph("g") as g:
        wrapped(max_new_tokens=32)
    gen = next(n for n in g.nodes() if n["name"] == "gen")
    assert gen["meta"]["max_new_tokens"] == 32
