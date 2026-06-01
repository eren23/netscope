// Mismatch squiggles: render the shape/rank warnings netscope already computes
// (netscope/core/checks.py -> graph.warnings) as red underlines on the offending
// line in the editor, not only in the graph. Each warning names a producer/
// consumer node; we underline the CONSUMER's line (where the bad input lands),
// with the producer's line as a fallback.

import * as vscode from "vscode";
import { NVGraph, NVNode } from "./ir";

const COLLECTION = "netscope";

export function makeCollection(): vscode.DiagnosticCollection {
  return vscode.languages.createDiagnosticCollection(COLLECTION);
}

function lineRange(document: vscode.TextDocument, line1: number): vscode.Range {
  const lineIdx = Math.max(0, Math.min((line1 || 1) - 1, document.lineCount - 1));
  return document.lineAt(lineIdx).range;
}

// netscope warnings are {src, dst, detail, severity, kind, source?}.
interface Warning {
  src: string;
  dst: string;
  detail: string;
  severity?: string;
  kind?: string;
  source?: string;
}

/** Populate red squiggles for `document` from the file's traced graph warnings. */
export function publish(
  collection: vscode.DiagnosticCollection,
  document: vscode.TextDocument,
  graph: NVGraph
): number {
  const warnings = ((graph as unknown as { warnings?: Warning[] }).warnings) || [];
  if (!warnings.length) {
    collection.set(document.uri, []);
    return 0;
  }
  const byId = new Map<string, NVNode>();
  for (const n of graph.nodes) byId.set(n.id, n);

  const diags: vscode.Diagnostic[] = [];
  for (const w of warnings) {
    // underline the consumer's line; fall back to the producer's.
    const consumer = byId.get(w.dst);
    const producer = byId.get(w.src);
    const target =
      consumer && consumer.loc && consumer.loc.file === document.fileName
        ? consumer
        : producer && producer.loc && producer.loc.file === document.fileName
        ? producer
        : null;
    if (!target || !target.loc) continue;

    const range = lineRange(document, target.loc.line);
    const prefix = w.source === "static" ? "netscope (before run): " : "netscope: ";
    const d = new vscode.Diagnostic(
      range,
      prefix + w.detail,
      vscode.DiagnosticSeverity.Error
    );
    d.source = "netscope";
    if (w.kind) d.code = w.kind;
    diags.push(d);
  }
  collection.set(document.uri, diags);
  return diags.length;
}

/** Clear squiggles for a file (its trace went stale on edit). */
export function clear(collection: vscode.DiagnosticCollection, uri: vscode.Uri): void {
  collection.set(uri, []);
}
