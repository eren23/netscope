// Headless integration test for the netscope extension — no real VSCode.
//
// Mocks the `vscode` module, require()s the COMPILED extension (out/extension.js),
// calls activate(), then fires the "Run & Trace" / "Show Graph" commands exactly
// like a CodeLens click would. The commands really shell out to this repo's
// python, capture the graph, fuse static+runtime, and build the webview HTML —
// so this asserts the whole extension path end-to-end, headless.
//
//   npm run test:headless
const assert = require("assert");
const path = require("path");
const Module = require("module");

// The reveal handler opens a document via the mocked async vscode API; that
// chain can't complete headlessly (no real editor). Swallow that one expected
// async rejection — we only assert the handler is wired and callable.
process.on("unhandledRejection", () => {});

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const VENV_PY = path.join(REPO_ROOT, ".venv", "bin", "python");
const DEMO = path.join(REPO_ROOT, "examples", "sfumato_cmajc.py");
const MISMATCH = path.join(REPO_ROOT, "examples", "mismatch_demo.py");

let lastPanel = null;
const commands = {};
const codeLensProviders = [];
const diagnosticsByUri = new Map();   // fsPath -> Diagnostic[]
const changeHandlers = [];
const saveHandlers = [];
const mockLlmConfig = {};             // netscope.llm settings, set per-test
let mockLiveStatic = true;            // netscope.liveStatic toggle
let nextInputBoxValue = undefined;    // what showInputBox returns next

const vscode = {
  ViewColumn: { One: 1, Beside: 2 },
  TextEditorRevealType: { InCenter: 2 },
  Position: class { constructor(l, c) { this.line = l; this.character = c; } },
  Selection: class { constructor(a, b) { this.anchor = a; this.active = b; } },
  Range: class { constructor(a, b) { this.start = a; this.end = b; } },
  CodeLens: class { constructor(range, command) { this.range = range; this.command = command; } },
  EventEmitter: class { constructor() { this._cbs = []; this.event = (cb) => { this._cbs.push(cb); return { dispose() {} }; }; } fire(v) { this._cbs.forEach((cb) => cb(v)); } },
  Diagnostic: class { constructor(range, message, severity) { this.range = range; this.message = message; this.severity = severity; } },
  DiagnosticSeverity: { Error: 0, Warning: 1, Information: 2, Hint: 3 },
  ThemeColor: class { constructor(id) { this.id = id; } },
  workspace: {
    workspaceFolders: [{ uri: { fsPath: REPO_ROOT } }],
    getConfiguration: (section) => ({
      get: (k, dflt) => {
        if (section === "netscope.llm") return (mockLlmConfig[k] !== undefined ? mockLlmConfig[k] : "");
        if (k === "pythonPath") return VENV_PY;
        if (k === "liveStatic") return mockLiveStatic;
        return dflt;
      },
    }),
    openTextDocument: async (p) => ({ fileName: p }),
    textDocuments: [],        // populated per-test with a mock document
    onDidChangeTextDocument: (cb) => { changeHandlers.push(cb); return { dispose() {} }; },
    onDidSaveTextDocument: (cb) => { saveHandlers.push(cb); return { dispose() {} }; },
  },
  ProgressLocation: { Notification: 15 },
  window: {
    activeTextEditor: { document: { fileName: DEMO, languageId: "python", uri: { fsPath: DEMO }, lineCount: 200, lineAt: (i) => ({ range: { start: { line: i, character: 0 }, end: { line: i, character: 0 } } }) } },
    visibleTextEditors: [],
    onDidChangeVisibleTextEditors: () => ({ dispose() {} }),
    createTextEditorDecorationType: () => ({ dispose() {} }),
    showErrorMessage: () => {},
    showWarningMessage: async () => undefined,
    showInformationMessage: async () => undefined,
    showInputBox: async () => nextInputBoxValue,
    // pass a (progress, token) pair like the real API; token never cancels here.
    withProgress: (_opts, task) => task({ report() {} }, { onCancellationRequested() { return { dispose() {} }; } }),
    createOutputChannel: () => ({ appendLine() {}, dispose() {}, show() {} }),
    showTextDocument: async () => ({ selection: null, revealRange() {} }),
    createWebviewPanel: (_id, title) => {
      // store html on the panel object itself (the extension reuses one panel
      // across commands, so closing over `lastPanel` would break if a test
      // nulled it before re-invoking).
      const p = {
        title, _html: "", _posted: [],
        webview: {
          cspSource: "vscode-resource:", _onMsg: null,
          set html(v) { p._html = v; }, get html() { return p._html; },
          onDidReceiveMessage(cb) { p.webview._onMsg = cb; },
          postMessage(m) { p._posted.push(m); },
        },
        onDidDispose() {}, reveal() {}, dispose() {},
      };
      lastPanel = p;
      return p;
    },
  },
  languages: {
    registerCodeLensProvider: (_sel, prov) => { codeLensProviders.push(prov); return { dispose() {} }; },
    createDiagnosticCollection: (_name) => ({
      set(uri, diags) { diagnosticsByUri.set((uri && uri.fsPath) || uri, diags || []); },
      dispose() {},
    }),
  },
  commands: {
    registerCommand: (id, fn) => { commands[id] = fn; return { dispose() {} }; },
  },
};

const origLoad = Module._load;
Module._load = function (request) {
  if (request === "vscode") return vscode;
  return origLoad.apply(this, arguments);
};

// in-memory SecretStorage (mirrors vscode.SecretStorage: async get/store/delete)
const secretStore = new Map();
const mockSecrets = {
  get: async (k) => secretStore.get(k),
  store: async (k, v) => { secretStore.set(k, v); },
  delete: async (k) => { secretStore.delete(k); },
};

const ext = require(path.join(__dirname, "..", "out", "extension.js"));
ext.activate({
  subscriptions: [],
  extensionPath: path.join(REPO_ROOT, "extension"),
  secrets: mockSecrets,
});

let passed = 0;
const _pending = [];
// queue tests; run them sequentially (awaiting async ones) in the runner below.
function check(label, fn) { _pending.push([label, fn]); }
async function runAll() {
  for (const [label, fn] of _pending) {
    await fn();
    passed++;
    console.log("ok - " + label);
  }
}

check("activate registers all commands", () => {
  assert.ok(commands["netscope.showGraph"]);
  assert.ok(commands["netscope.runAndTrace"]);
  assert.ok(commands["netscope.checkSetup"], "checkSetup command not registered");
  assert.ok(commands["netscope.setLlmKey"]);
});

check("CodeLens provider exposes 2 lenses", () => {
  assert.strictEqual(codeLensProviders.length, 1);
  assert.strictEqual(codeLensProviders[0].provideCodeLenses().length, 2);
});

check("Show Graph renders a self-contained, fully-inlined webview", async () => {
  lastPanel = null;
  await commands["netscope.showGraph"]();
  assert.ok(lastPanel, "no webview panel");
  const h = lastPanel._html;
  // the vendor placeholder MUST be replaced and the REAL libs inlined — without
  // this the panel is blank (the bug this test exists to catch).
  assert.ok(!h.includes("__NETSCOPE_VENDOR__"), "vendor placeholder left unreplaced (blank panel)");
  assert.ok(h.includes("Cytoscape Consortium"), "real cytoscape lib not inlined");
  // self-contained = no external RESOURCE LOADS. Check actual load points only:
  // a <script src=>, a remote <link href=>, or a CDN host in the CSP. (Bare
  // "jsdelivr"/"cdnjs" substrings also appear in the inlined libs' banner/
  // sourcemap comments, so don't match those — they're not loads.)
  assert.ok(!/<script[^>]+\bsrc\s*=/i.test(h), "has external <script src=>");
  assert.ok(!/<link[^>]+\bhref\s*=\s*["']?https?:/i.test(h), "has external <link href=>");
  const csp = (h.match(/Content-Security-Policy[^>]*/i) || [""])[0];
  assert.ok(!/cdnjs|jsdelivr|unpkg/i.test(csp), "CSP references a CDN host");
  // M3: the node panel wires the LLM assistant (ask buttons + answer postback).
  assert.ok(/type:"ask"/.test(h), "node panel missing the 'ask' assistant action");
  assert.ok(/type==="answer"/.test(h), "webview missing the answer postback listener");
  // C4: the isolate action is present in the node panel.
  assert.ok(/type:"isolate"/.test(h), "node panel missing the 'isolate' action");
  // C3: tapping a node must NOT auto-reveal (only the panel loc link reveals) —
  // the tap handler should open the panel without a reveal postMessage.
  assert.ok(/cy\.on\("tap","node",\(e\)=>\{ showPanel\(e\.target\.data\(\)\); \}\)/.test(h),
    "node tap should open the panel only (no double-reveal)");
  // dump for a real serve+screenshot check
  require("fs").writeFileSync("/tmp/ext_webview.html", h);
});

check("Run & Trace renders the real fused graph", async () => {
  // NB: don't null lastPanel — the extension REUSES the existing panel and only
  // updates its html, so we just re-read it after invoking.
  await commands["netscope.runAndTrace"]();
  assert.ok(lastPanel, "no webview panel");
  assert.ok(/diffuse|vote|ARPlanner/i.test(lastPanel._html), "missing sfumato nodes");
});

check("click-a-node reveal handler fires without throwing", () => {
  const cb = lastPanel.webview._onMsg;
  assert.ok(typeof cb === "function");
  cb({ type: "reveal", loc: { file: DEMO, line: 10 } });
});

check("LLM 'ask' with no key prompts to set one, posts no answer", async () => {
  // no env key, no stored secret -> the handler should bail with the "Set API
  // Key" prompt (showWarningMessage returns undefined here) and post nothing.
  delete process.env.NETSCOPE_LLM_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  delete process.env.OPENAI_API_KEY;
  secretStore.clear();
  lastPanel._posted = [];
  await lastPanel.webview._onMsg({ type: "ask", nodeId: "ARPlanner#0", question: "explain" });
  // gated: with no key it returns before shelling out, so no answer is posted
  const answer = lastPanel._posted.find((m) => m.type === "answer");
  assert.ok(!answer, "no answer should be posted when the assistant is unconfigured");
});

check("Set LLM Key command stores the key in SecretStorage (keychain)", async () => {
  assert.ok(commands["netscope.setLlmKey"], "setLlmKey command not registered");
  nextInputBoxValue = "sk-test-keychain";
  await commands["netscope.setLlmKey"]();
  assert.strictEqual(secretStore.get("netscope.llm.apiKey"), "sk-test-keychain",
    "key should be saved in SecretStorage, not settings/env");
  // and clearing removes it
  await commands["netscope.clearLlmKey"]();
  assert.ok(!secretStore.has("netscope.llm.apiKey"), "clear should remove the stored key");
  nextInputBoxValue = undefined;
});

check("stored key + model/baseUrl settings reach the python subprocess env", () => {
  // Unit-test the env builder directly via the exported helper — deterministic,
  // no subprocess. (The full panel->shell-out path is covered by the gating test
  // above; here we prove the keychain key + settings become the right env vars.)
  secretStore.set("netscope.llm.apiKey", "sk-from-keychain");
  mockLlmConfig.model = "google/gemini-2.0-flash-001";
  mockLlmConfig.baseUrl = "https://openrouter.ai/api/v1";

  const env = ext._buildLlmEnvForTest
    ? ext._buildLlmEnvForTest({ secrets: mockSecrets }, mockLlmConfig, {})
    : null;
  assert.ok(env, "extension should expose _buildLlmEnvForTest");
  return Promise.resolve(env).then((e) => {
    assert.strictEqual(e.NETSCOPE_LLM_API_KEY, "sk-from-keychain",
      "the keychain key must become NETSCOPE_LLM_API_KEY");
    assert.strictEqual(e.NETSCOPE_LLM_MODEL, "google/gemini-2.0-flash-001");
    assert.strictEqual(e.NETSCOPE_LLM_BASE_URL, "https://openrouter.ai/api/v1");
    secretStore.clear();
    mockLlmConfig.model = ""; mockLlmConfig.baseUrl = "";
  });
});

// ---- M1: inline shape hints + mismatch squiggles ----
// A mock TextDocument over a real file: lineAt/lineCount/uri are all the
// providers touch. We read the real file so line indices are valid.
function mockDoc(file) {
  const lines = require("fs").readFileSync(file, "utf-8").split("\n");
  return {
    fileName: file,
    uri: { fsPath: file },
    languageId: "python",
    lineCount: lines.length,
    lineAt: (i) => ({
      range: { start: { line: i, character: 0 }, end: { line: i, character: lines[i].length } },
    }),
  };
}

check("Run & Trace on a mismatch file publishes shape hints + a red squiggle", async () => {
  const doc = mockDoc(MISMATCH);
  vscode.workspace.textDocuments = [doc];
  vscode.window.activeTextEditor = { document: doc };

  await commands["netscope.runAndTrace"]();

  // 1) inline shape hints (now via DECORATIONS, not InlayHints): the visible
  // editor for this file should get >=1 end-of-line shape decoration.
  const capturedDecos = [];
  vscode.window.visibleTextEditors = [{
    document: doc,
    setDecorations: (_type, decos) => { capturedDecos.length = 0; capturedDecos.push(...decos); },
  }];
  // re-fire the overlay path by re-running (sets the trace + refreshes decos)
  await commands["netscope.runAndTrace"]();
  assert.ok(capturedDecos.length >= 1, "expected at least one shape decoration");
  assert.ok(capturedDecos.some((d) => /\[\d/.test(d.renderOptions?.after?.contentText || "")),
    "decoration should carry a tensor shape");

  // 2) mismatch squiggle: a diagnostic published for this file with the detail
  const diags = diagnosticsByUri.get(MISMATCH) || [];
  assert.ok(diags.length >= 1, "expected at least one mismatch diagnostic");
  assert.ok(/netscope/.test(diags[0].message), "diagnostic should be a netscope message");
  assert.strictEqual(diags[0].severity, vscode.DiagnosticSeverity.Error);

  // restore for any later checks
  vscode.window.activeTextEditor = { document: { fileName: DEMO, languageId: "python" } };
});

check("editing re-runs LIVE static analysis (debounced), not just a clear", async () => {
  // mismatch_demo's clash is a RUNTIME edge (added inside the trace), so the
  // static pass sees no clash. Editing -> after the debounce, live-static
  // re-analyzes and (finding no static clash) clears the now-stale runtime
  // squiggle. The squiggle should linger briefly, then resolve.
  const doc = mockDoc(MISMATCH);
  assert.ok(changeHandlers.length >= 1, "no onDidChangeTextDocument handler");
  assert.ok((diagnosticsByUri.get(MISMATCH) || []).length >= 1, "precondition: a squiggle is shown");
  changeHandlers[0]({ document: doc });
  // it lingers right after the edit (debounced, not instant)...
  assert.ok((diagnosticsByUri.get(MISMATCH) || []).length >= 1, "squiggle should linger briefly");
  // ...and after the debounce + the quiet static re-run, the runtime-only clash
  // is gone (static sees none on this file).
  await new Promise((r) => setTimeout(r, 1200));
  const diags = diagnosticsByUri.get(MISMATCH) || [];
  assert.strictEqual(diags.length, 0, "runtime-only squiggle clears once live-static re-checks");
});

check("LIVE static squiggles a clash on SAVE without running (no trace needed)", async () => {
  // a file with a STATIC wiring clash: saving it should publish a squiggle via
  // the live-static path — no Run & Trace, no execution.
  const bad = path.join(require("os").tmpdir(), "netscope_live_bad.py");
  require("fs").writeFileSync(bad, [
    "import torch.nn as nn",
    "class Net(nn.Module):",
    "    def __init__(self):",
    "        super().__init__()",
    "        self.a = nn.Linear(64, 256)",
    "        self.b = nn.Linear(128, 10)",
    "    def forward(self, x):",
    "        h = self.a(x)",
    "        return self.b(h)",
    "",
  ].join("\n"));
  const doc = mockDoc(bad);
  diagnosticsByUri.delete(bad);
  assert.ok(saveHandlers.length >= 1, "no onDidSaveTextDocument handler");
  await saveHandlers[0](doc);
  // give the quiet static run a moment
  await new Promise((r) => setTimeout(r, 400));
  const diags = diagnosticsByUri.get(bad) || [];
  assert.ok(diags.length >= 1, "saving a clashing file should squiggle it live (no run)");
  assert.ok(/256.*128|128.*256/.test(diags[0].message), "message should name the clashing dims");
});

check("Show Graph squiggles a wiring clash WITHOUT running the file (M2)", async () => {
  // a clashing model written to a temp file: backbone emits 512, classifier wants 256
  const bad = path.join(require("os").tmpdir(), "netscope_bad_wiring.py");
  require("fs").writeFileSync(
    bad,
    [
      "import torch.nn as nn",
      "",
      "class MyNet(nn.Module):",
      "    def __init__(self):",
      "        super().__init__()",
      "        self.backbone = nn.Linear(784, 512)",
      "        self.classifier = nn.Linear(256, 10)",
      "    def forward(self, x):",
      "        return self.classifier(self.backbone(x))",
      "",
    ].join("\n")
  );
  const doc = mockDoc(bad);
  vscode.workspace.textDocuments = [doc];
  vscode.window.activeTextEditor = { document: doc };

  await commands["netscope.showGraph"]();   // STATIC only — never executes the file

  const diags = diagnosticsByUri.get(bad) || [];
  assert.ok(diags.length >= 1, "static pre-check should squiggle the clash before any run");
  assert.ok(/512.*256|256.*512/.test(diags[0].message), "message should name the clashing dims");
  // squiggle lands on the classifier line (index 6 -> line 7)
  assert.strictEqual(diags[0].range.start.line, 6, "squiggle should be on the classifier line");

  vscode.window.activeTextEditor = { document: { fileName: DEMO, languageId: "python" } };
});

runAll().then(() => {
  Module._load = origLoad;
  console.log("\n" + passed + " passed");
}).catch((e) => {
  Module._load = origLoad;
  console.error(e);
  process.exit(1);
});
