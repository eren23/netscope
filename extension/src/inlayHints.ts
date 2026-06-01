// Inline shape hints: after a Run & Trace, show each layer's real output tensor
// shape as faint end-of-line ghost text on the line where the module is defined
// (`self.encoder = nn.Linear(...)`). The "as you write" payoff — real shapes,
// in the editor, on the line. Reads from traceStore (keyed by file); every node
// already carries `loc` (M0) + `meta.out_shape` (capture).

import * as vscode from "vscode";
import { getTrace } from "./traceStore";

// Fired when a new trace lands so VSCode re-queries hints.
export const onDidChangeInlayHints = new vscode.EventEmitter<void>();

function shapeLabel(meta: Record<string, unknown>): string | null {
  const out = meta["out_shape"] as number[] | undefined;
  if (Array.isArray(out) && out.length) return `[${out.join(", ")}]`;
  return null;
}

export class ShapeHints implements vscode.InlayHintsProvider {
  onDidChangeInlayHints = onDidChangeInlayHints.event;

  provideInlayHints(
    document: vscode.TextDocument,
    range: vscode.Range
  ): vscode.InlayHint[] {
    const graph = getTrace(document.fileName);
    if (!graph) return [];

    const hints: vscode.InlayHint[] = [];
    const seen = new Set<number>(); // one hint per line (first node wins)

    for (const n of graph.nodes) {
      const loc = n.loc;
      if (!loc || loc.file !== document.fileName) continue;
      const label = shapeLabel(n.meta || {});
      if (!label) continue;

      const lineIdx = (loc.line || 1) - 1; // loc.line is 1-based
      if (lineIdx < range.start.line || lineIdx > range.end.line) continue;
      if (seen.has(lineIdx) || lineIdx >= document.lineCount) continue;
      seen.add(lineIdx);

      const eol = document.lineAt(lineIdx).range.end;
      const hint = new vscode.InlayHint(eol, `  ${label}`, vscode.InlayHintKind.Type);
      hint.paddingLeft = true;
      hint.tooltip = `netscope: ${n.name} output shape`;
      hints.push(hint);
    }
    return hints;
  }
}
