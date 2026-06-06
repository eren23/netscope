"""Capture state held in contextvars.

A single `ContextVar` holds the active `Capture` (None when not capturing — this
is the zero-overhead gate instrumentors check first). A second `ContextVar`
holds the parent-id stack used to nest spans.

Scope: capture is **single-flow per session**. The active-capture gate and parent
stack are contextvars, so they're correct within one flow. But a model run in a
*separate OS thread* gets a fresh (empty) context and won't be captured, and the
torch hook's nesting state isn't shared across concurrently-suspended coroutines —
so trace from one thread / one flow per `graph()`.
"""
from __future__ import annotations

import contextvars
from typing import Optional

# Forward type only; avoids an import cycle with capture.py.
_CURRENT: "contextvars.ContextVar[object]" = contextvars.ContextVar(
    "netscope_current_capture", default=None
)
_PARENT_STACK: "contextvars.ContextVar[tuple]" = contextvars.ContextVar(
    "netscope_parent_stack", default=()
)


def active_capture():
    """Return the live Capture, or None when no session is open."""
    return _CURRENT.get()


def is_capturing() -> bool:
    return _CURRENT.get() is not None


def set_capture(cap):
    return _CURRENT.set(cap)


def reset_capture(token) -> None:
    _CURRENT.reset(token)


def current_parent() -> Optional[str]:
    stack = _PARENT_STACK.get()
    return stack[-1] if stack else None


def push_parent(node_id: str):
    return _PARENT_STACK.set(_PARENT_STACK.get() + (node_id,))


def pop_parent(token) -> None:
    _PARENT_STACK.reset(token)


def push_clean_parent_scope():
    """Begin a session with an empty parent stack; returns a token to restore.

    A session must start with no inherited parent, and — crucially — must not
    leak a dangling parent to the NEXT session if its spans are abandoned mid
    forward (e.g. the traced model raised). `graph()` brackets every session with
    this + ``restore_parent_scope`` so one broken session can't poison another.
    """
    return _PARENT_STACK.set(())


def restore_parent_scope(token) -> None:
    _PARENT_STACK.reset(token)
