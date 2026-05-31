"""JSON sink — the raw IR, schema-versioned. The interchange format the VSCode
extension streams/loads and the format every other tool can consume."""
from __future__ import annotations

import json


def to_json(g, indent: int = 2) -> str:
    return json.dumps(g.to_dict(), indent=indent)
