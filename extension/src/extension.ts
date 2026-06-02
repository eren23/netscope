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
import { ShapeHints, onDidChangeInlayHints } from "./inlayHints";
import { refreshShapeDecorations } from "./shapeDecorations";
import * as diagnostics from "./diagnostics";
import { setTrace, clearTrace, getTrace } from "./traceStore";

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

// A shared netscope output channel for diagnostics/logging the user can read.
let output: vscode.OutputChannel | undefined;
function log(msg: string): void {
  if (!output) output = vscode.window.createOutputChannel("netscope");
  output.appendLine(`[${new Date().toISOString()}] ${msg}`);
}

interface ExecResult { code: number; stdout: string; stderr: string; failed?: string; }

// Async, NON-BLOCKING process exec wrapped in a cancellable progress notification.
// The whole editor used to freeze on a synchronous execFileSync while a real
// model's forward ran; this runs the child off the UI thread, shows progress,
// and SIGTERMs the child if the user cancels. Returns a structured result
// (never throws) so callers can give actionable errors.
function execAsync(
  args: string[], opts: { env?: NodeJS.ProcessEnv; title: string }
): Promise<ExecResult> {
  return Promise.resolve(vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: opts.title, cancellable: true },
    (_progress, token) =>
      new Promise<ExecResult>((resolve) => {
        const child = cp.execFile(
          pythonPath(), args,
          { cwd: cwd(), env: opts.env || process.env, maxBuffer: 64 * 1024 * 1024 },
          (err: any, stdout: string, stderr: string) => {
            if (err && err.code === "ENOENT") {
              resolve({ code: -1, stdout: "", stderr: "", failed: "ENOENT" });
            } else {
              resolve({ code: err ? (err.code ?? 1) : 0, stdout: stdout || "", stderr: stderr || "" });
            }
          }
        );
        token.onCancellationRequested(() => {
          try { child.kill("SIGTERM"); } catch { /* ignore */ }
          log("run cancelled by user");
        });
      })
  ));
}

// Map a failed exec to one clear, actionable message (python missing vs netscope
// not installed vs the user's script erroring) instead of a raw stack.
function explainFailure(r: ExecResult): string {
  if (r.failed === "ENOENT") {
    return "netscope: Python interpreter not found. Set `netscope.pythonPath` to your venv's python.";
  }
  const err = (r.stderr || "").trim();
  if (/No module named ['"]?netscope/.test(err)) {
    return "netscope: the `netscope` package isn't installed in this interpreter. Run `pip install netscope` (or `pip install -e .`).";
  }
  return `netscope: ${err.split("\n").slice(-3).join(" ").slice(0, 240) || "the command failed."}`;
}

async function runStatic(file: string): Promise<NVGraph | null> {
  const r = await execAsync(["-m", "netscope.static", file], {
    title: "netscope: analyzing…",
  });
  if (r.code !== 0) {
    vscode.window.showErrorMessage(explainFailure(r));
    log(`static failed: code=${r.code} ${r.stderr}`);
    return null;
  }
  try {
    return JSON.parse(r.stdout) as NVGraph;
  } catch {
    return null;
  }
}

// Static analysis WITHOUT a progress notification — for the live-on-save/type
// refresh, which fires often and must stay invisible (no popup spam, no errors;
// a parse failure just yields null and the previous overlays linger).
function runStaticQuiet(file: string): Promise<NVGraph | null> {
  return new Promise((resolve) => {
    cp.execFile(pythonPath(), ["-m", "netscope.static", file],
      { cwd: cwd(), env: process.env, maxBuffer: 32 * 1024 * 1024 },
      (err: any, stdout: string) => {
        if (err) { resolve(null); return; }
        try { resolve(JSON.parse(stdout) as NVGraph); } catch { resolve(null); }
      });
  });
}

async function runAndTrace(file: string): Promise<NVGraph | null> {
  const outPath = path.join(os.tmpdir(), `netscope-run-${process.pid}-${Date.now()}.json`);
  const r = await execAsync([file], {
    title: "netscope: running & tracing…",
    env: { ...process.env, NETSCOPE_OUT: outPath },
  });
  if (r.code !== 0 && !fs.existsSync(outPath)) {
    // the script itself failed AND produced no graph -> surface the real error.
    vscode.window.showWarningMessage(explainFailure(r));
    log(`trace failed: code=${r.code} ${r.stderr}`);
    return null;
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
async function runIsolate(file: string, qualname: string): Promise<NVGraph | null> {
  const outPath = path.join(os.tmpdir(), `netscope-iso-${process.pid}-${Date.now()}.json`);
  await execAsync([file], {
    title: `netscope: isolating ${qualname}…`,
    env: { ...process.env, NETSCOPE_ISOLATE: qualname, NETSCOPE_ISOLATE_OUT: outPath },
  });
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

const SECRET_KEY = "netscope.llm.apiKey";   // SecretStorage handle (OS keychain)

// Resolve the LLM API key: VSCode SecretStorage (OS keychain — encrypted, never
// in settings.json or git) first, then an env var as a power-user fallback.
async function resolveLlmKey(ctx: vscode.ExtensionContext): Promise<string | undefined> {
  const stored = await ctx.secrets.get(SECRET_KEY);
  if (stored) return stored;
  return (
    process.env.NETSCOPE_LLM_API_KEY ||
    process.env.OPENROUTER_API_KEY ||
    process.env.OPENAI_API_KEY ||
    undefined
  );
}

// Pure env builder (no vscode globals) so it's unit-testable: resolved key +
// non-secret model/baseUrl -> the NETSCOPE_LLM_* the CLI reads. Settings are NOT
// secret, so they live in normal config; only the key comes from the keychain.
function buildLlmEnv(
  key: string | undefined,
  cfg: { model?: string; baseUrl?: string },
  base: NodeJS.ProcessEnv
): NodeJS.ProcessEnv | null {
  if (!key) return null;
  const env: NodeJS.ProcessEnv = { ...base, NETSCOPE_LLM_API_KEY: key };
  if (cfg.model && cfg.model.trim()) env.NETSCOPE_LLM_MODEL = cfg.model.trim();
  if (cfg.baseUrl && cfg.baseUrl.trim()) env.NETSCOPE_LLM_BASE_URL = cfg.baseUrl.trim();
  return env;
}

// The subprocess env for the LLM CLI: the resolved key + settings.
async function llmEnv(ctx: vscode.ExtensionContext): Promise<NodeJS.ProcessEnv | null> {
  const key = await resolveLlmKey(ctx);
  const cfg = vscode.workspace.getConfiguration("netscope.llm");
  return buildLlmEnv(key, { model: cfg.get<string>("model"), baseUrl: cfg.get<string>("baseUrl") }, process.env);
}

// test-only seam: build the env from an explicit secrets store + config (the
// headless suite asserts the keychain key + settings become the right env vars).
export async function _buildLlmEnvForTest(
  ctx: { secrets: { get(k: string): Promise<string | undefined> } },
  cfg: { model?: string; baseUrl?: string },
  base: NodeJS.ProcessEnv
): Promise<NodeJS.ProcessEnv | null> {
  const key = (await ctx.secrets.get(SECRET_KEY)) ||
    base.NETSCOPE_LLM_API_KEY || base.OPENROUTER_API_KEY || base.OPENAI_API_KEY;
  return buildLlmEnv(key, cfg, base);
}

// Ask the LLM about a node: write the current graph to a temp JSON, shell out to
// `python -m netscope.llm <graph.json> <nodeId> <question>` with the resolved
// key/model/baseUrl in env. Returns the answer text, or null (+ an actionable
// prompt) when no key is set or the call fails — the graph itself keeps working.
async function runLLM(
  ctx: vscode.ExtensionContext, graph: NVGraph, nodeId: string, question: string
): Promise<string | null> {
  const env = await llmEnv(ctx);
  if (!env) {
    const pick = await vscode.window.showWarningMessage(
      "netscope: no LLM API key set. Add one to use the assistant.",
      "Set API Key"
    );
    if (pick === "Set API Key") await vscode.commands.executeCommand("netscope.setLlmKey");
    return null;
  }
  const gpath = path.join(os.tmpdir(), `netscope-llm-${process.pid}-${Date.now()}.json`);
  try {
    fs.writeFileSync(gpath, JSON.stringify(graph));
    const r = await execAsync(["-m", "netscope.llm", gpath, nodeId, question], {
      title: "netscope: asking the assistant…", env,
    });
    if (r.code !== 0) {
      vscode.window.showWarningMessage(explainFailure(r));
      return null;
    }
    return r.stdout;
  } finally {
    try { fs.unlinkSync(gpath); } catch { /* ignore */ }
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
let currentGraph: NVGraph | undefined;     // the graph in the panel, for the LLM assistant
let currentGraphFile: string | undefined;  // which file the panel is showing
let currentGraphTag: string | undefined;   // "static" | "runtime" | "fused" | "isolate:…"

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
          // Open the source in the MAIN editor column (One), never on top of the
          // graph panel (which lives in column Beside). Without an explicit column
          // the source replaces whatever's active — and you just clicked the
          // graph, so it'd replace the graph. viewColumn One keeps them side-by-side.
          vscode.window.showTextDocument(doc, {
            viewColumn: vscode.ViewColumn.One,
            preserveFocus: false,
            preview: true,   // reuse one tab instead of spawning a tab per click
          }).then((ed) => {
            ed.selection = new vscode.Selection(pos, pos);
            ed.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
          });
        });
      } else if (msg?.type === "isolate" && msg.qualname && lastTracedFile) {
        // runIsolate self-wraps in a cancellable progress notification.
        runIsolate(lastTracedFile, msg.qualname).then((g) => {
          if (g) show(ctx, g, `isolate: ${msg.qualname}`);
        });
      } else if (msg?.type === "ask" && msg.nodeId && currentGraph) {
        const question = msg.question || "explain";
        runLLM(ctx, currentGraph, msg.nodeId, question).then((answer) => {
          panel?.webview.postMessage({ type: "answer", nodeId: msg.nodeId, text: answer || "" });
        });
      }
    });
  }
  currentGraph = graph;   // the LLM assistant asks about THIS graph
  // remember what the panel shows (title is "<file> (<tag>)") so live-static can
  // refresh a static skeleton without overwriting a richer fused/runtime view.
  const _m = /^(.*) \(([^)]+)\)$/.exec(title);
  currentGraphTag = _m ? _m[2] : undefined;
  currentGraphFile = graph.nodes.find((n) => n.loc && n.loc.file)?.loc?.file;
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

// The editor-live layer (built on every node now carrying a `loc`, M0): inline
// shape hints + mismatch squiggles, fed from the file's last traced graph.
const diagCollection = diagnostics.makeCollection();

function applyEditorOverlays(file: string, graph: NVGraph): void {
  setTrace(file, graph);                       // hints + decorations read this
  onDidChangeInlayHints.fire();                // re-query inlay hints (if enabled)
  refreshShapeDecorations();                   // ...and decorations (always visible)
  for (const doc of vscode.workspace.textDocuments) {
    if (doc.fileName === file) diagnostics.publish(diagCollection, doc, graph);
  }
}

export function activate(ctx: vscode.ExtensionContext): void {
  ctx.subscriptions.push(
    diagCollection,
    vscode.languages.registerCodeLensProvider({ language: "python" }, new Lenses()),
    vscode.languages.registerInlayHintsProvider({ language: "python" }, new ShapeHints()),
    vscode.commands.registerCommand("netscope.showGraph", async () => {
      const ed = vscode.window.activeTextEditor;
      if (!ed) return;
      const g = await runStatic(ed.document.fileName);
      if (!g) return;
      // Static analysis only sees structure it can read from SOURCE: `for` loops,
      // votes, @stage decorators, and literal `self.x = nn.Linear(...)` layers.
      // A model built dynamically (from_config, factory fns) has little to show —
      // so a near-empty static graph is expected. Steer the user to Run & Trace
      // (the REAL graph) instead of rendering a lonely node.
      if (g.nodes.length <= 1) {
        const pick = await vscode.window.showInformationMessage(
          "netscope: not much to show statically (this model's layers aren't " +
          "literal in the source). Run & Trace captures the real graph.",
          "Run & Trace"
        );
        if (pick === "Run & Trace") vscode.commands.executeCommand("netscope.runAndTrace");
        return;
      }
      show(ctx, g, `${path.basename(ed.document.fileName)} (static)`);
      // M2: the static graph carries declared-dim warnings -> squiggle wiring
      // clashes WITHOUT running. (Diagnostics only; real shape hints need a run.)
      diagnostics.publish(diagCollection, ed.document, g);
    }),
    vscode.commands.registerCommand("netscope.runAndTrace", async () => {
      const ed = vscode.window.activeTextEditor;
      if (!ed) return;
      const file = ed.document.fileName;
      lastTracedFile = file;   // enable "isolate this part" on the resulting graph
      // static analysis and the traced run are independent — run them
      // concurrently so the static skeleton doesn't add latency to the trace.
      const [staticG, runtimeG] = await Promise.all([runStatic(file), runAndTrace(file)]);
      const fused =
        runtimeG && staticG ? mergeByLoc(runtimeG, staticG) : runtimeG || staticG;
      if (!fused) return;
      const tag = runtimeG && staticG ? "fused" : runtimeG ? "runtime" : "static";
      show(ctx, fused, `${path.basename(file)} (${tag})`);
      applyEditorOverlays(file, fused);   // inline shapes + squiggles on the lines
    }),
    // validate the setup: is the python interpreter resolvable and is netscope
    // importable in it? Gives a store user an actionable answer instead of a
    // cryptic ENOENT / ModuleNotFoundError the first time they run a command.
    vscode.commands.registerCommand("netscope.checkSetup", async () => {
      const r = await execAsync(["-c", "import netscope, sys; print(netscope.__version__ if hasattr(netscope,'__version__') else 'ok'); print(sys.version.split()[0])"], {
        title: "netscope: checking setup…",
      });
      if (r.failed === "ENOENT") {
        const pick = await vscode.window.showErrorMessage(
          `netscope: Python not found at "${pythonPath()}". Set netscope.pythonPath to your venv's python.`,
          "Open Settings"
        );
        if (pick === "Open Settings") {
          vscode.commands.executeCommand("workbench.action.openSettings", "netscope.pythonPath");
        }
      } else if (r.code !== 0 || /No module named/.test(r.stderr)) {
        vscode.window.showErrorMessage(
          "netscope: Python works, but the `netscope` package isn't installed in it. Run `pip install netscope` (or `pip install -e .`)."
        );
      } else {
        vscode.window.showInformationMessage(`netscope: ready ✓ (python ${r.stdout.trim().split("\n").pop()})`);
      }
    }),
    // store the LLM API key in the OS keychain (SecretStorage) — never in
    // settings.json, never synced, never in git.
    vscode.commands.registerCommand("netscope.setLlmKey", async () => {
      const key = await vscode.window.showInputBox({
        title: "netscope: LLM API key",
        prompt: "OpenRouter / OpenAI / any OpenAI-compatible key. Stored in your OS keychain.",
        password: true,
        ignoreFocusOut: true,
        placeHolder: "sk-or-… or sk-…",
      });
      if (key && key.trim()) {
        await ctx.secrets.store(SECRET_KEY, key.trim());
        vscode.window.showInformationMessage("netscope: LLM API key saved to your keychain.");
      }
    }),
    vscode.commands.registerCommand("netscope.clearLlmKey", async () => {
      await ctx.secrets.delete(SECRET_KEY);
      vscode.window.showInformationMessage("netscope: stored LLM API key cleared.");
    }),
    // LIVE STATIC: as you edit, the REAL tensor shapes go stale (they only exist
    // after a run) — but the static structure + dim-mismatch squiggles DON'T need
    // execution, so we can refresh them live. Debounced ~600ms after you stop
    // typing: drop the stale runtime shape hints, then re-run static analysis and
    // re-publish the wiring-clash squiggles. (Toggle: netscope.liveStatic.)
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document.languageId !== "python") return;
      const doc = e.document;
      const file = doc.fileName;
      const prev = staleTimers.get(file);
      if (prev) clearTimeout(prev);
      staleTimers.set(file, setTimeout(() => {
        staleTimers.delete(file);
        // the runtime shapes are now stale -> drop the shape decorations/hints.
        const stale = getTrace(file);
        const hadRuntimeShapes = !!stale && stale.nodes.some((n) => (n.meta || {}).out_shape);
        if (hadRuntimeShapes) {
          clearTrace(file);
          onDidChangeInlayHints.fire();
          refreshShapeDecorations();
        }
        liveStaticRefresh(ctx, doc);   // re-check structure + squiggles, no run
      }, 600));
    }),
    // also refresh static on SAVE (immediate, not debounced) — saving is an
    // explicit "I'm done with this edit" signal.
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (doc.languageId === "python") liveStaticRefresh(ctx, doc);
    }),
    vscode.window.onDidChangeVisibleTextEditors(() => refreshShapeDecorations())
  );
}

const staleTimers = new Map<string, ReturnType<typeof setTimeout>>();

// Re-run static analysis for `doc` (no execution) and update its mismatch
// squiggles + (if the panel is showing this file's static graph) the skeleton.
// Off when netscope.liveStatic is disabled.
async function liveStaticRefresh(ctx: vscode.ExtensionContext, doc: vscode.TextDocument): Promise<void> {
  if (!vscode.workspace.getConfiguration("netscope").get<boolean>("liveStatic", true)) return;
  const g = await runStaticQuiet(doc.fileName);
  if (!g) return;
  diagnostics.publish(diagCollection, doc, g);   // wiring-clash squiggles, live
  // if the panel is currently showing THIS file's static skeleton, redraw it so
  // it tracks the source. Don't clobber a richer runtime/fused view of the file.
  if (panel && currentGraphFile === doc.fileName && currentGraphTag === "static" && g.nodes.length > 1) {
    show(ctx, g, `${path.basename(doc.fileName)} (static)`);
  }
}

export function deactivate(): void {
  panel?.dispose();
}
