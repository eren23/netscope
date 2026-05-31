"""Parameter counting — free, framework-trivial, always on.

`own_params` counts only a module's directly-owned parameters (recurse=False) so
that a container node's params aren't the sum of its children (which would
double-count once the children are their own nodes). `total_params` is the full
recursive count, handy for a model-root summary.
"""
from __future__ import annotations


def own_params(module) -> int:
    try:
        return int(sum(p.numel() for p in module.parameters(recurse=False)))
    except Exception:
        return 0


def total_params(module) -> int:
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:
        return 0
