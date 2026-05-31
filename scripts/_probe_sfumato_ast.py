"""Probe: run the static AST producer against the real sfumato runner and print
what structure it recovers (branch fan-out + vote) without executing anything."""
from __future__ import annotations

from netscope.static.ast_producer import analyze_file

PATH = "/Users/eren/Documents/AI/sfumato/e4/runner.py"


def main() -> None:
    g = analyze_file(PATH)
    nodes = g.nodes()
    print(f"static nodes found in REAL sfumato runner.py: {len(nodes)}")
    print()
    for n in nodes:
        a = n["attrs"]
        tags = []
        if a.get("repeat"):
            tags.append(f"repeat={a['repeat']}")
        if a.get("branch"):
            tags.append("branch")
        if a.get("reduce"):
            tags.append("reduce")
        if a.get("declared"):
            tags.append("@stage")
        print(f"  L{n['loc']['line']:<4} {n['name']:<12} [{','.join(tags)}]")
    repeats = [n for n in nodes if n["attrs"].get("repeat")]
    votes = [n for n in nodes if n["attrs"].get("reduce")]
    print()
    print(f"-> {len(repeats)} branch/loop region(s), {len(votes)} vote/reduce region(s)")


if __name__ == "__main__":
    main()
