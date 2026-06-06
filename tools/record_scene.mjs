// netscope playground video driver.
// Drives the live playground, "types" a scripted scene, records 1280x720 video.
//   1. python -m netscope.playground 8770 --no-open
//   2. PW_PATH=<playwright dir> node tools/record_scene.mjs <scene>
// scenes: bug | shapes | diff | profile
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW_PATH || 'playwright');

const URL = process.env.HARNESS_URL || 'http://localhost:8770/';
const OUT = process.env.OUT_DIR || '/tmp/netscope-vids';
const scene = process.argv[2] || 'bug';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- page driver hooks (defined in tools/live_harness.py) ---
const type_ = (p, t, per = 36) => p.evaluate(([t, per]) => window.nsType(t, per), [t, per]);
const set_ = (p, t) => p.evaluate((t) => window.nsSet(t), t);
const replace_ = (p, a, b) => p.evaluate(([a, b]) => window.nsReplace(a, b), [a, b]);
const mode_ = (p, m, pr = false) => p.evaluate(([m, pr]) => window.nsSetMode(m, pr), [m, pr]);
const cap_ = (p, t) => p.evaluate((t) => window.nsCaption(t), t);

// the rendered graph lives in a srcdoc child frame; grab it (it swaps per edit).
async function graphFrame(p) {
  for (let i = 0; i < 25; i++) {
    const fr = p.frames().find((f) => f !== p.mainFrame());
    if (fr) return fr;
    await sleep(120);
  }
  return null;
}

const BUGGY = [
  'class Net(nn.Module):',
  '    def __init__(self):',
  '        super().__init__()',
  '        self.enc  = nn.Linear(64, 256)',
  '        self.head = nn.Linear(128, 10)',
  '',
  '    def forward(self, x):',
  '        h = self.enc(x)',
  '        return self.head(h)',
  '',
].join('\n');

const mlp = (hidden) =>
  `model = nn.Sequential(\n    nn.Linear(16, ${hidden}),\n    nn.ReLU(),\n    nn.Linear(${hidden}, 4),\n)\nx = torch.randn(8, 16)\n`;

const DIFF_V1 = `model = nn.Sequential(\n    nn.Linear(64, 128),\n    nn.ReLU(),\n    nn.Linear(128, 10),\n)\nx = torch.randn(4, 64)\n`;
const DIFF_V2 = `model = nn.Sequential(\n    nn.Linear(64, 256),\n    nn.ReLU(),\n    nn.Linear(256, 128),\n    nn.ReLU(),\n    nn.Linear(128, 10),\n)\nx = torch.randn(4, 64)\n`;

const FAT = `model = nn.Sequential(\n    nn.Linear(64, 1024),\n    nn.ReLU(),\n    nn.Linear(1024, 1024),\n    nn.ReLU(),\n    nn.Linear(1024, 10),\n)\nx = torch.randn(8, 64)\n`;

const ENCODER = `import torch, torch.nn as nn\n\nlayer = nn.TransformerEncoderLayer(64, 8, batch_first=True)\nmodel = nn.TransformerEncoder(layer, num_layers=2)\nx = torch.randn(2, 10, 64)\n`;

async function sceneBug(p) {
  await mode_(p, 'static');
  await cap_(p, 'Wiring an encoder into a classifier head — as you type.');
  await sleep(900);
  await type_(p, BUGGY, 30);
  await sleep(800);
  await cap_(p, 'The encoder emits <b>256</b>, the head expects <b>128</b> — flagged red, <b>before you run.</b>');
  await sleep(3000);
  await replace_(p, 'nn.Linear(128, 10)', 'nn.Linear(256, 10)');
  await sleep(500);
  await cap_(p, 'Fix the dim → clears instantly. No execution, no stack trace.');
  await sleep(3000);
}

async function sceneShapes(p) {
  await mode_(p, 'trace');
  await cap_(p, 'Build a model — watch the real tensor shapes appear.');
  await sleep(800);
  await set_(p, mlp(32));
  await sleep(2200);
  await cap_(p, 'Widen the hidden layer 32 → 128 — shapes update live.');
  await replace_(p, '16, 32', '16, 128');
  await replace_(p, '32, 4', '128, 4');
  await sleep(2600);
}

async function sceneDiff(p) {
  await mode_(p, 'trace');
  await cap_(p, 'Trace a baseline model…');
  await sleep(800);
  await set_(p, DIFF_V1);
  await sleep(2200);
  await mode_(p, 'diff');
  await cap_(p, 'Widen a layer + insert a block, re-trace → <b>green = added, amber = changed.</b>');
  await set_(p, DIFF_V2);
  await sleep(3200);
}

async function sceneProfile(p) {
  await mode_(p, 'trace', true);
  await cap_(p, 'Trace with profiling on — one layer is far heavier than the rest.');
  await sleep(800);
  await set_(p, FAT);
  await sleep(2400);
  await cap_(p, 'Color nodes by cost → the fat layer glows red.');
  const fr = await graphFrame(p);
  if (fr) {
    await fr.evaluate(() => {
      const s = document.getElementById('cost-by');
      if (s) { s.value = 'param_bytes'; }
      if (typeof window.applyCost === 'function') window.applyCost('param_bytes');
    });
  }
  await sleep(3200);
}

const PG_MODEL = `import torch, torch.nn as nn\n\nmodel = nn.Sequential(\n    nn.Linear(64, 128),\n    nn.ReLU(),\n    nn.Linear(128, 10),\n)\nx = torch.randn(8, 64)\n`;

async function scenePlayground(p) {
  await mode_(p, 'trace');
  await p.evaluate(() => window.nsSet(''));     // clear the starter model
  await sleep(400);
  await cap_(p, '<b>netscope playground</b> — type a model, watch it build live.');
  await sleep(700);
  await type_(p, PG_MODEL, 26);
  await sleep(1900);
  await cap_(p, 'Widen a layer — the shapes update as you edit.');
  await replace_(p, '64, 128', '64, 512');
  await replace_(p, '128, 10', '512, 10');
  await sleep(2100);
  await cap_(p, 'Flip the mode → <b>profile</b>, then color by cost.');
  await p.selectOption('#mode-sel', 'profile');   // the playground's mode selector
  await sleep(1500);
  const fr = await graphFrame(p);
  if (fr) await fr.evaluate(() => {
    const s = document.getElementById('cost-by'); if (s) s.value = 'param_bytes';
    if (typeof window.applyCost === 'function') window.applyCost('param_bytes');
  });
  await sleep(2600);
}

async function sceneRoles(p) {
  await mode_(p, 'trace');
  await cap_(p, 'A transformer — attention, norm and feed-forward, stacked.');
  await sleep(800);
  await set_(p, ENCODER);
  await sleep(2600);
  const fr = await graphFrame(p);
  if (fr) {
    // unfold the repeated encoder layers so the per-block leaves are visible
    await fr.evaluate(() => {
      if (window.ecApi) cy.nodes().forEach((n) => { if (n.isParent()) window.ecApi.expand(n); });
    });
    await sleep(600);
  }
  await cap_(p, 'Color by <b>role</b> — attention, norm and feed-forward pop out.');
  if (fr) await fr.evaluate(() => { const b = document.getElementById('btn-role'); if (b) b.click(); });
  await sleep(3200);
}

const RESNET = `import torch, torchvision.models as M\n\nmodel = M.resnet18()\nx = torch.randn(1, 3, 224, 224)\n`;
const GPT2 = `import torch\nfrom transformers import GPT2Config, GPT2Model\n\ncfg = GPT2Config(n_layer=3, n_head=6, n_embd=192, vocab_size=512)\nmodel = GPT2Model(cfg)\nx = torch.randint(0, 512, (1, 16))\n`;
const MOBILENET = `import torch, torchvision.models as M\n\nmodel = M.mobilenet_v3_small()\nx = torch.randn(1, 3, 224, 224)\n`;

async function sceneResnet(p) {
  await mode_(p, 'trace');
  await p.evaluate(() => window.nsSet(''));   // clear the starter model first
  await sleep(250);
  await cap_(p, 'A real vision model — <b>resnet18</b>, 11.7M params — in one line.');
  await sleep(700);
  await type_(p, RESNET, 22);
  await sleep(3200);
  await cap_(p, '76 layers, but the repeated BasicBlocks fold — it reads as a clean pipeline.');
  await sleep(3200);
}

async function sceneGpt2(p) {
  await mode_(p, 'trace');
  await p.evaluate(() => window.nsSet(''));   // clear the starter model first
  await sleep(250);
  await cap_(p, '<b>GPT-2</b>, built from its config — no weights downloaded.');
  await sleep(700);
  await type_(p, GPT2, 20);
  await sleep(3400);
  await cap_(p, 'Unfold the blocks, color by <b>role</b> — attention · MLP · norm.');
  const fr = await graphFrame(p);
  if (fr) {
    await fr.evaluate(() => {
      if (!window.ecApi || !window.ecApi.expandableNodes) return;
      // collapsed blocks aren't isParent(); they're "expandable". Unfold them to
      // reveal the attention / MLP / norm internals, then color by role.
      const exp = window.ecApi.expandableNodes();
      if (exp && exp.nonempty()) window.ecApi.expand(exp);
    });
    await sleep(800);
    await fr.evaluate(() => { const b = document.getElementById('btn-role'); if (b) b.click(); });
  }
  await sleep(3400);
}

async function sceneMobilenet(p) {
  await mode_(p, 'trace');
  await p.evaluate(() => window.nsSet(''));   // clear the starter model first
  await sleep(250);
  await cap_(p, '<b>MobileNetV3</b> — 200+ layers, traced from one forward pass.');
  await sleep(700);
  await type_(p, MOBILENET, 20);
  await sleep(3600);
  await cap_(p, 'Even a deep modern CNN folds into a graph you can actually read.');
  await sleep(2800);
}

const SCENES = { bug: sceneBug, shapes: sceneShapes, diff: sceneDiff, profile: sceneProfile, roles: sceneRoles, playground: scenePlayground, resnet: sceneResnet, gpt2: sceneGpt2, mobilenet: sceneMobilenet };

(async () => {
  const run = SCENES[scene];
  if (!run) { console.error('unknown scene ' + scene); process.exit(2); }
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    recordVideo: { dir: OUT, size: { width: 1280, height: 720 } },
    deviceScaleFactor: 2,
  });
  const page = await ctx.newPage();
  await page.goto(URL);
  await page.waitForFunction(() => window.nsReady === true, { timeout: 10000 });
  await page.evaluate(() => window.nsReset && window.nsReset());
  await sleep(500);
  await run(page);
  await sleep(700);
  const vid = page.video();
  await ctx.close();
  await browser.close();
  const out = await vid.path();
  console.log('VIDEO ' + out);
})();
