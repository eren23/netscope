"""Auto file sink: hand a traced run's graph back to a caller via a file.

When the env var ``NETSCOPE_OUT`` is set, every `netscope.graph()` session writes its
final graph JSON to that path on exit. The VSCode extension sets ``NETSCOPE_OUT``
when it launches a script via the "Run & Trace" command, then reads the file
once the process finishes — a dependency-free alternative to live streaming.

Best-effort: a failure to write must never break the user's program.
"""
from __future__ import annotations

import os


def maybe_dump(graph) -> None:
    path = os.environ.get("NETSCOPE_OUT")
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(graph.to_json())
    except Exception:
        pass
