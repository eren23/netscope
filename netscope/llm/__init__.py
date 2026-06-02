"""netscope LLM assistant — provider-agnostic, grounded, optional.

`explain(graph, node_id, question=...)` answers a question about a node, grounded
in the IR slice + the real source lines (see prompts.build_messages). The provider
is any OpenAI-compatible endpoint configured from env (OpenRouter by default).
With no API key the layer is unavailable and raises LLMUnavailable — the rest of
netscope is untouched and works fully offline.

    import netscope.llm as nl
    if nl.available():
        print(nl.explain(graph, node_id, question="why_warn"))
"""
from __future__ import annotations

from typing import Optional

from netscope.llm.prompts import build_messages
from netscope.llm.provider import Provider, Transport


class LLMUnavailable(RuntimeError):
    """Raised when no LLM provider is configured (no API key in the env)."""


def available() -> bool:
    """True if an LLM provider can be built from the environment."""
    return Provider.from_env() is not None


def explain(
    graph,
    node_id: str,
    *,
    question: str = "explain",
    provider: Optional[Provider] = None,
    _transport: Optional[Transport] = None,
) -> str:
    """Answer a question about `node_id`, grounded in the graph + source.

    question: "explain" | "why_warn" | "suggest_fix".
    Raises LLMUnavailable if no provider is configured.
    """
    provider = provider or Provider.from_env()
    if provider is None:
        raise LLMUnavailable(
            "no LLM provider configured — set NETSCOPE_LLM_API_KEY (or "
            "OPENROUTER_API_KEY / OPENAI_API_KEY). netscope works fully without it."
        )
    messages = build_messages(graph, node_id, question=question)
    return provider.complete(messages, _transport=_transport)


def infer(graph, source: str, filename: str = "<source>"):
    """Augment `graph` with LLM-inferred provisional structure (dashed, confidence-
    scored) where the static AST couldn't recover it. No-op without a key."""
    from netscope.llm.infer import infer_structure

    return infer_structure(graph, source, filename)


__all__ = ["available", "explain", "infer", "LLMUnavailable", "Provider", "build_messages"]
