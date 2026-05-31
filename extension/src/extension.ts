// netscope VSCode extension — host code.
//
// Two entry points, both rendering into one webview that reuses the library's
// shared Cytoscape template (netscope/web/template.html):
//   * Show Pipeline Graph  — runs `python -m netscope.static <file>` and draws the
//     on-type skeleton (stage/branch/vote structure, no execution).
//   * Run & Trace          — runs the file with NETSCOPE_OUT set so the library
//     dumps the real runtime graph (live tensor shapes), then fuses it onto the
//     static skeleton by source loc and redraws.
// Clicking a node posts {type:'reveal', loc} back here -> jump to that line.

import * as vscode from "vscode";
import * as cp from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { NVGraph } from "./ir";
import { mergeByLoc } from "./mergeByLoc";
import { toElements } from "./render";

function cwd(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0].uri.fsPath;
}

function pythonPath(): string {
  const cfg = vscode.workspace.getConfiguration("netscope").get<string>("pythonPath");
  if (!cfg || !cfg.trim()) return "python3";
  // VSCode does NOT expand ${workspaceFolder} in arbitrary string settings, so
  // expand it ourselves — this is the documented way to point at a venv.
  const root = cwd();
  return root ? cfg.replace(/\$\{workspaceFolder\}/g, root) : cfg;
}

function runStatic(file: string): NVGraph | null {
  try {
    const out = cp.execFileSync(pythonPath(), ["-m", "netscope.static", file], {
      encoding: "utf-8", cwd: cwd(),
    });
    return JSON.parse(out) as NVGraph;
  } catch (e: any) {
    vscode.window.showErrorMessage(`netscope static analysis failed: ${e.message}`);
    return null;
  }
}

function runAndTrace(file: string): NVGraph | null {
  const outPath = path.join(os.tmpdir(), `netscope-run-${process.pid}-${Date.now()}.json`);
  try {
    cp.execFileSync(pythonPath(), [file], {
      encoding: "utf-8", cwd: cwd(),
      env: { ...process.env, NETSCOPE_OUT: outPath },
    });
  } catch (e: any) {
    vscode.window.showWarningMessage(`netscope: script exited non-zero: ${e.message}`);
  }
  if (!fs.existsSync(outPath)) {
    vscode.window.showWarningMessage(
      'netscope: no graph captured. Wrap your pipeline in `with netscope.graph("name"):`.'
    );
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(outPath, "utf-8")) as NVGraph;
  } catch {
    return null;
  } finally {
    try { fs.unlinkSync(outPath); } catch { /* ignore */ }
  }
}

// Re-run the file with NETSCOPE_ISOLATE set so the library re-runs JUST the
// chosen submodule on its real input and dumps that focused sub-trace.
function runIsolate(file: string, qualname: string): NVGraph | null {
  const outPath = path.join(os.tmpdir(), `netscope-iso-${process.pid}-${Date.now()}.json`);
  try {
    cp.execFileSync(pythonPath(), [file], {
      encoding: "utf-8", cwd: cwd(),
      env: { ...process.env, NETSCOPE_ISOLATE: qualname, NETSCOPE_ISOLATE_OUT: outPath },
    });
  } catch (e: any) {
    vscode.window.showWarningMessage(`netscope: script exited non-zero during isolate: ${e.message}`);
  }
  if (!fs.existsSync(outPath)) {
    vscode.window.showWarningMessage(
      `netscope: couldn't isolate "${qualname}" — the module wasn't reached, or it ` +
      `needs call args that couldn't be re-run standalone.`
    );
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(outPath, "utf-8")) as NVGraph;
  } catch {
    return null;
  } finally {
    try { fs.unlinkSync(outPath); } catch { /* ignore */ }
  }
}

function nonce(): string {
  let s = "";
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) s += chars.charAt(Math.floor(Math.random() * chars.length));
  return s;
}

function webDir(ctx: vscode.ExtensionContext): string {
  // dev (F5 from repo): the library's web/ folder is a sibling of extension/.
  const sibling = path.join(ctx.extensionPath, "..", "netscope", "web");
  if (fs.existsSync(path.join(sibling, "template.html"))) return sibling;
  return path.join(ctx.extensionPath, "media"); // packaged fallback
}

function templatePath(ctx: vscode.ExtensionContext): string {
  return path.join(webDir(ctx), "template.html");
}

// Read the vendored cytoscape libs and return them as inline <script> blocks.
// Mirrors netscope/sinks/html_sink.py so the webview is self-contained (no CDN) —
// without this the __NETSCOPE_VENDOR__ placeholder ships unreplaced and the graph
// panel is blank. Vendor dir: <web>/vendor (dev) or media/vendor (packaged).
const VENDOR_LIBS = [
  "cytoscape.min.js",
  "dagre.min.js",
  "cytoscape-dagre.min.js",
  "cytoscape-expand-collapse.min.js",
];

function vendorScripts(ctx: vscode.ExtensionContext): string {
  const dir = path.join(webDir(ctx), "vendor");
  const out: string[] = [];
  for (const name of VENDOR_LIBS) {
    const src = fs.readFileSync(path.join(dir, name), "utf-8");
    out.push(`<!-- ${name} -->\n<script>\n${src}\n</script>`);
  }
  return out.join("\n");
}

let panel: vscode.WebviewPanel | undefined;
let lastTracedFile: string | undefined;   // the file behind the current graph, for isolate

function show(ctx: vscode.ExtensionContext, graph: NVGraph, title: string): void {
  if (!panel) {
    panel = vscode.window.createWebviewPanel(
      "netscopeGraph", "netscope", vscode.ViewColumn.Beside, { enableScripts: true }
    );
    panel.onDidDispose(() => (panel = undefined));
    panel.webview.onDidReceiveMessage((msg) => {
      if (msg?.type === "reveal" && msg.loc?.file) {
        const pos = new vscode.Position(Math.max(0, (msg.loc.line || 1) - 1), 0);
        vscode.workspace.openTextDocument(msg.loc.file).then((doc) => {
          vscode.window.showTextDocument(doc).then((ed) => {
            ed.selection = new vscode.Selection(pos, pos);
            ed.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
          });
        });
      } else if (msg?.type === "isolate" && msg.qualname && lastTracedFile) {
        vscode.window.withProgress(
          { location: vscode.ProgressLocation.Notification, title: `netscope: isolating ${msg.qualname}…` },
          async () => {
            const g = runIsolate(lastTracedFile as string, msg.qualname);
            if (g) show(ctx, g, `isolate: ${msg.qualname}`);
          }
        );
      }
    });
  }
  const tpl = fs.readFileSync(templatePath(ctx), "utf-8");
  const elements = JSON.stringify(toElements(graph));
  const n = nonce();
  // FUNCTION replacers: the minified libs + elements JSON contain `$&`/`$'`/`$\``
  // sequences that String.replace interprets specially in a string replacement
  // (re-injecting the matched placeholder), which corrupted the output and left
  // the panel blank. A function replacement is always literal.
  const vendor = vendorScripts(ctx);
  const html = tpl
    .replace("__NETSCOPE_VENDOR__", () => vendor)   // inline cytoscape etc.
    .replace("__NETSCOPE_ELEMENTS__", () => elements)
    .replace(/__NETSCOPE_TITLE__/g, () => title.replace(/[<>&]/g, ""))
    // all libs are inlined -> webview needs no remote origin; nonce-only CSP.
    .replace(
      "<head>",
      `<head>\n<meta http-equiv="Content-Security-Policy" content="default-src 'none'; ` +
        `img-src ${panel.webview.cspSource} data:; style-src 'unsafe-inline'; ` +
        `script-src 'nonce-${n}';">`
    )
    .replace(/<script(?![^>]*src=)/g, `<script nonce="${n}"`);
  panel.webview.html = html;
  panel.reveal(vscode.ViewColumn.Beside);
}

class Lenses implements vscode.CodeLensProvider {
  provideCodeLenses(): vscode.CodeLens[] {
    const top = new vscode.Range(0, 0, 0, 0);
    return [
      new vscode.CodeLens(top, { title: "$(graph) netscope: Show Graph", command: "netscope.showGraph" }),
      new vscode.CodeLens(top, { title: "$(play) Run & Trace", command: "netscope.runAndTrace" }),
    ];
  }
}

export function activate(ctx: vscode.ExtensionContext): void {
  ctx.subscriptions.push(
    vscode.languages.registerCodeLensProvider({ language: "python" }, new Lenses()),
    vscode.commands.registerCommand("netscope.showGraph", () => {
      const ed = vscode.window.activeTextEditor;
      if (!ed) return;
      const g = runStatic(ed.document.fileName);
      if (g) show(ctx, g, `${path.basename(ed.document.fileName)} (static)`);
    }),
    vscode.commands.registerCommand("netscope.runAndTrace", () => {
      const ed = vscode.window.activeTextEditor;
      if (!ed) return;
      const file = ed.document.fileName;
      lastTracedFile = file;   // enable "isolate this part" on the resulting graph
      const staticG = runStatic(file);
      const runtimeG = runAndTrace(file);
      if (runtimeG && staticG) show(ctx, mergeByLoc(runtimeG, staticG), `${path.basename(file)} (fused)`);
      else if (runtimeG) show(ctx, runtimeG, `${path.basename(file)} (runtime)`);
      else if (staticG) show(ctx, staticG, `${path.basename(file)} (static)`);
    })
  );
}

export function deactivate(): void {
  panel?.dispose();
}
