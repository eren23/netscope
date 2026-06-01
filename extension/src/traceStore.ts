// Holds the most recent traced graph per source file, so the inlay-hint and
// diagnostic providers can read real shapes + warnings for the file the user is
// looking at. One small shared store keeps extension.ts, inlayHints.ts and
// diagnostics.ts decoupled — they all key off `loc.file`.

import { NVGraph } from "./ir";

const byFile = new Map<string, NVGraph>();

/** Record a freshly traced/fused graph; keyed by the file that was run. */
export function setTrace(file: string, graph: NVGraph): void {
  byFile.set(file, graph);
}

/** The last graph traced for `file`, or undefined. */
export function getTrace(file: string): NVGraph | undefined {
  return byFile.get(file);
}

/** Drop the trace for a file (e.g. it was edited → the trace is now stale). */
export function clearTrace(file: string): void {
  byFile.delete(file);
}
