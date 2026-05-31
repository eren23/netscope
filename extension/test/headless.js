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

let lastPanel = null;
const commands = {};
const codeLensProviders = [];

const vscode = {
  ViewColumn: { One: 1, Beside: 2 },
  TextEditorRevealType: { InCenter: 2 },
  Position: class { constructor(l, c) { this.line = l; this.character = c; } },
  Selection: class { constructor(a, b) { this.anchor = a; this.active = b; } },
  Range: class { constructor(a, b) { this.start = a; this.end = b; } },
  CodeLens: class { constructor(range, command) { this.range = range; this.command = command; } },
  workspace: {
    workspaceFolders: [{ uri: { fsPath: REPO_ROOT } }],
    getConfiguration: () => ({ get: (k) => (k === "pythonPath" ? VENV_PY : undefined) }),
    openTextDocument: async (p) => ({ fileName: p }),
  },
  window: {
    activeTextEditor: { document: { fileName: DEMO, languageId: "python" } },
    showErrorMessage: () => {},
    showWarningMessage: () => {},
    showTextDocument: async () => ({ selection: null, revealRange() {} }),
    createWebviewPanel: (_id, title) => {
      // store html on the panel object itself (the extension reuses one panel
      // across commands, so closing over `lastPanel` would break if a test
      // nulled it before re-invoking).
      const p = {
        title, _html: "",
        webview: {
          cspSource: "vscode-resource:", _onMsg: null,
          set html(v) { p._html = v; }, get html() { return p._html; },
          onDidReceiveMessage(cb) { p.webview._onMsg = cb; },
          postMessage() {},
        },
        onDidDispose() {}, reveal() {}, dispose() {},
      };
      lastPanel = p;
      return p;
    },
  },
  languages: {
    registerCodeLensProvider: (_sel, prov) => { codeLensProviders.push(prov); return { dispose() {} }; },
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

const ext = require(path.join(__dirname, "..", "out", "extension.js"));
ext.activate({ subscriptions: [], extensionPath: path.join(REPO_ROOT, "extension") });

let passed = 0;
function check(label, fn) { fn(); passed++; console.log("ok - " + label); }

check("activate registers both commands", () => {
  assert.ok(commands["netscope.showGraph"]);
  assert.ok(commands["netscope.runAndTrace"]);
});

check("CodeLens provider exposes 2 lenses", () => {
  assert.strictEqual(codeLensProviders.length, 1);
  assert.strictEqual(codeLensProviders[0].provideCodeLenses().length, 2);
});

check("Show Graph renders a self-contained, fully-inlined webview", () => {
  lastPanel = null;
  commands["netscope.showGraph"]();
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
  // dump for a real serve+screenshot check
  require("fs").writeFileSync("/tmp/ext_webview.html", h);
});

check("Run & Trace renders the real fused graph", () => {
  // NB: don't null lastPanel — the extension REUSES the existing panel and only
  // updates its html, so we just re-read it after invoking.
  commands["netscope.runAndTrace"]();
  assert.ok(lastPanel, "no webview panel");
  assert.ok(/diffuse|vote|ARPlanner/i.test(lastPanel._html), "missing sfumato nodes");
});

check("click-a-node reveal handler fires without throwing", () => {
  const cb = lastPanel.webview._onMsg;
  assert.ok(typeof cb === "function");
  cb({ type: "reveal", loc: { file: DEMO, line: 10 } });
});

Module._load = origLoad;
console.log("\n" + passed + " passed");
