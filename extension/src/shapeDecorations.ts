// Inline shape hints via TEXT EDITOR DECORATIONS (not InlayHints).
//
// InlayHints are gated by `editor.inlayHints.enabled`, which defaults to
// "onUnlessPressed" in recent VSCode/Cursor — so our hints were invisible unless
// the user held Ctrl+Alt. Decorations render an end-of-line `after` label
// unconditionally and we fully control the styling, so the real tensor shapes are
// always visible after a trace.

import * as vscode from "vscode";
import { getTrace } from "./traceStore";
import { NVGraph } from "./ir";

const decoType = vscode.window.createTextEditorDecorationType({
  after: {
    color: new vscode.ThemeColor("editorCodeLens.foreground"),
    fontStyle: "italic",
    margin: "0 0 0 1.2em",
  },
});

function shapeLabel(meta: Record<string, unknown>): string | null {
  const out = meta["out_shape"] as number[] | undefined;
  if (Array.isArray(out) && out.length) return `[${out.join(", ")}]`;
  return null;
}

// Apply shape decorations to every visible editor whose file has a trace.
export function refreshShapeDecorations(): void {
  for (const editor of vscode.window.visibleTextEditors) {
    applyTo(editor);
  }
}

function applyTo(editor: vscode.TextEditor): void {
  const graph: NVGraph | undefined = getTrace(editor.document.fileName);
  if (!graph) {
    editor.setDecorations(decoType, []);
    return;
  }
  const seen = new Set<number>();          // one hint per line (first node wins)
  const decos: vscode.DecorationOptions[] = [];
  for (const n of graph.nodes) {
    const loc = n.loc;
    if (!loc || loc.file !== editor.document.fileName) continue;
    const label = shapeLabel(n.meta || {});
    if (!label) continue;
    const lineIdx = (loc.line || 1) - 1;     // loc.line is 1-based
    if (seen.has(lineIdx) || lineIdx < 0 || lineIdx >= editor.document.lineCount) continue;
    seen.add(lineIdx);
    const eol = editor.document.lineAt(lineIdx).range.end;
    decos.push({
      range: new vscode.Range(eol, eol),
      renderOptions: { after: { contentText: `‹${label}›` } },
      hoverMessage: `netscope: ${n.name} output shape`,
    });
  }
  editor.setDecorations(decoType, decos);
}

export function disposeShapeDecorations(): void {
  decoType.dispose();
}
