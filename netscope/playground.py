"""netscope playground — a local "paste a model, watch it analyzed" web app.

    python -m netscope.playground            # opens http://localhost:8770

A split view: edit a model on the left, see the real netscope graph on the right,
updated as you type. Pick a mode in the header:

    trace    exec the snippet (define `model` + an example input `x`) and trace
             model(x) — REAL per-layer tensor shapes
    static   netscope.static on the source text — structure + declared-dim wiring
             clashes, no run (a mismatch shows without torch crashing)
    profile  trace with profiling on, then the graph's "cost:" selector recolors
             nodes by time / memory / params
    diff     trace, then diff against the previous trace — green added, amber changed

LOCAL & TRUST: this binds to 127.0.0.1 and **executes the code in the editor** to
trace it — exactly the trust level of running `python yourfile.py` yourself. Don't
expose it to a network or paste code you wouldn't run.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import netscope
from netscope.core.checks import detect_mismatches
from netscope.core.diff import annotate_diff
from netscope.static.ast_producer import analyze_source

_LAST = {"graph": None}


def _trace_code(code: str, profile: bool):
    import torch
    ns = {"torch": torch, "nn": torch.nn, "netscope": netscope}
    exec(compile(code, "<playground>", "exec"), ns)    # local: your own code, like python
    model = ns.get("model")
    if model is None:
        raise ValueError("define `model` in the snippet")
    # most snippets define a single example input `x` -> model(x). Multi-input
    # models (SAM3 wants pixel_values + input_ids; a BERT wants input_ids +
    # attention_mask) instead define an `inputs` dict -> model(**inputs).
    inputs = ns.get("inputs")
    x = ns.get("x")
    if inputs is None and x is None:
        raise ValueError("define an example input `x` (or an `inputs` dict for multi-input models)")
    with netscope.graph("playground", profile=profile) as g:
        with torch.no_grad():
            model(**inputs) if inputs is not None else model(x)
    return g


def analyze(code: str, mode: str, profile: bool) -> dict:
    """Run netscope on `code` in the chosen mode; return rendered HTML + counts."""
    if mode == "static":
        g = analyze_source(code, "<playground>")
        return {"ok": True, "html": g.to_html(),
                "nodes": len(g.nodes()), "warnings": len(detect_mismatches(g))}
    g = _trace_code(code, profile)
    if mode == "diff" and _LAST["graph"] is not None:
        html = annotate_diff(_LAST["graph"], g).to_html()
    else:
        html = g.to_html()
    _LAST["graph"] = g
    return {"ok": True, "html": html,
            "nodes": len(g.nodes()), "warnings": len(detect_mismatches(g))}


def _origin_ok(host, origin):
    """Reject anything that isn't a same-machine request, BEFORE we exec the body.

    The endpoint runs the editor's code, so binding to loopback isn't enough: a
    page you visit could POST to http://127.0.0.1:<port>/analyze (CSRF), or rebind
    a domain to 127.0.0.1 (DNS rebinding). Defenses:
      * Host must be loopback   -> a rebound domain sends its own Host, so it fails
      * no cross-origin Origin  -> a browser always sends Origin on a cross-site POST
    Together these stop a remote page from driving code execution; a local client
    (the playground page itself, or curl with no Origin) passes.
    """
    h = (host or "").rsplit(":", 1)[0].strip("[]").lower()
    if h not in ("localhost", "127.0.0.1", "::1", ""):
        return False
    if origin:
        from urllib.parse import urlparse
        if (urlparse(origin).hostname or "").lower() not in ("localhost", "127.0.0.1", "::1"):
            return False
    return True


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if not _origin_ok(self.headers.get("Host"), self.headers.get("Origin")):
            self._send(403, json.dumps({"ok": False, "error": "forbidden"})); return
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"ok": False, "error": "not found"}))

    def do_POST(self):
        if not _origin_ok(self.headers.get("Host"), self.headers.get("Origin")):
            self._send(403, json.dumps({"ok": False, "error": "forbidden (non-local origin)"})); return
        if self.path != "/analyze":
            self._send(404, json.dumps({"ok": False}))
            return
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if req.get("reset"):
            _LAST["graph"] = None
        try:
            out = analyze(req.get("code", ""), req.get("mode", "trace"), bool(req.get("profile")))
        except Exception as e:
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(out))

    def log_message(self, *a):
        pass


def serve(port: int = 8770, open_browser: bool = True) -> None:
    url = f"http://localhost:{port}"
    print(f"netscope playground -> {url}  (Ctrl-C to stop)")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        HTTPServer(("127.0.0.1", port), _Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>netscope · playground</title>
<style>
  :root{--bg:#0b0e14;--panel:#0e131c;--hair:#1c2333;--ink:#c7d2e0;--dim:#5b6678;
    --cyan:#22d3ee;--teal:#34f5a8;--amber:#ffc233;--purple:#cf8bff;--red:#ff5a5f;
    --mono:'SF Mono',ui-monospace,Menlo,Consolas,monospace;}
  *{box-sizing:border-box} html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--ink);font-family:var(--mono);overflow:hidden}
  #app{display:grid;grid-template-columns:42% 58%;height:100vh}
  #left{display:flex;flex-direction:column;border-right:1px solid var(--hair);min-width:0}
  #tabbar{height:42px;display:flex;align-items:center;gap:11px;padding:0 14px;
    border-bottom:1px solid var(--hair);background:var(--panel)}
  .brand{font-weight:700;letter-spacing:.04em}.brand b{color:var(--cyan)}
  .tab{font-size:12px;color:#9fb0c8;background:rgba(255,255,255,.03);padding:3px 11px;
    border-radius:6px;border:1px solid var(--hair)}
  .spacer{flex:1}
  select.mode{font-family:var(--mono);font-size:11px;color:var(--ink);background:rgba(255,255,255,.04);
    border:1px solid var(--hair);border-radius:7px;padding:5px 8px;outline:none;cursor:pointer}
  select.mode:hover{border-color:var(--stage,var(--cyan))}
  #wrap{position:relative;flex:1;display:flex;overflow:hidden}
  #gutter{padding:16px 8px 16px 15px;text-align:right;color:#323d57;font-size:13px;
    line-height:22px;user-select:none;white-space:pre}
  #stack{position:relative;flex:1}
  #hl,#code{margin:0;padding:16px 16px 16px 6px;font-family:var(--mono);font-size:13px;
    line-height:22px;white-space:pre;border:0;tab-size:4}
  #hl{position:absolute;inset:0;pointer-events:none;overflow:auto;color:var(--ink)}
  #code{position:absolute;inset:0;background:transparent;color:transparent;
    caret-color:var(--cyan);resize:none;outline:none;overflow:auto}
  .k{color:var(--cyan)}.s{color:var(--teal)}.n{color:var(--amber)}
  .c{color:#414c68;font-style:italic}.t{color:var(--purple)}.f{color:#7fb0ff}
  #right{position:relative;min-width:0;background:var(--bg)}
  #g{width:100%;height:100%;border:0;display:block}
  #status{position:absolute;top:13px;right:17px;font-size:11px;color:var(--dim);
    background:rgba(8,11,18,.72);padding:5px 11px;border-radius:7px;border:1px solid var(--hair);
    letter-spacing:.04em}
  #caption{position:absolute;left:18px;right:18px;bottom:70px;padding:13px 18px;
    background:rgba(6,9,15,.86);border:1px solid var(--hair);border-radius:11px;
    box-shadow:0 12px 32px rgba(0,0,0,.5);font-size:15px;color:#eaf2ff;
    letter-spacing:.01em;opacity:0;transition:opacity .35s;line-height:1.4}
  #caption.show{opacity:1}#caption b{color:var(--cyan);font-weight:600}
</style></head>
<body><div id="app">
  <div id="left">
    <div id="tabbar"><span class="brand"><b>net</b>scope</span><span class="tab">model.py</span>
      <span class="spacer"></span>
      <select class="mode" id="mode-sel" title="analysis mode">
        <option value="trace">mode: trace</option>
        <option value="static">mode: static (no run)</option>
        <option value="profile">mode: profile</option>
        <option value="diff">mode: diff vs last</option>
      </select>
    </div>
    <div id="wrap"><div id="gutter">1</div><div id="stack">
      <pre id="hl"></pre><textarea id="code" spellcheck="false" autocomplete="off" autocapitalize="off"></textarea>
    </div></div>
  </div>
  <div id="right"><iframe id="g"></iframe><div id="status">idle</div><div id="caption"></div></div>
</div>
<script>
  const code=document.getElementById('code'),hl=document.getElementById('hl'),
        gutter=document.getElementById('gutter'),frame=document.getElementById('g'),
        statusEl=document.getElementById('status'),cap=document.getElementById('caption'),
        modeSel=document.getElementById('mode-sel');
  let MODE='trace',PROFILE=false;
  function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
  function highlight(src){
    const re=/(#[^\n]*)|('[^']*'|"[^"]*")|\b(\d+(?:\.\d+)?)\b|\b(def|class|return|import|from|as|with|for|in|if|else|elif|self|super|None|True|False|lambda|and|or|not|pass)\b|\b(nn|torch|F)\b/g;
    let out='',last=0,m;
    while((m=re.exec(src))){
      out+=esc(src.slice(last,m.index));
      const cls=m[1]?'c':m[2]?'s':m[3]?'n':m[4]?'k':'t';
      out+='<span class="'+cls+'">'+esc(m[0])+'</span>';
      last=re.lastIndex;
    }
    return out+esc(src.slice(last));
  }
  function sync(){
    hl.innerHTML=highlight(code.value)+'\n';
    const n=Math.max(1,code.value.split('\n').length);
    let g=''; for(let i=1;i<=n;i++) g+=i+'\n'; gutter.textContent=g;
    hl.scrollTop=code.scrollTop; hl.scrollLeft=code.scrollLeft;
  }
  code.addEventListener('input',function(){sync();schedule()});
  code.addEventListener('scroll',function(){hl.scrollTop=code.scrollTop;hl.scrollLeft=code.scrollLeft});
  let timer=null; function schedule(){clearTimeout(timer);timer=setTimeout(analyze,320)}
  let busy=false,pending=false;
  async function analyze(){
    if(busy){pending=true;return} busy=true; statusEl.textContent='analyzing…';
    try{
      const r=await fetch('/analyze',{method:'POST',headers:{'content-type':'application/json'},
        body:JSON.stringify({code:code.value,mode:MODE,profile:PROFILE})});
      const j=await r.json();
      if(j.ok){ frame.srcdoc=j.html;
        statusEl.textContent=(j.nodes||0)+' nodes · '+(j.warnings||0)+' ⚠';
        statusEl.style.color=j.warnings?'var(--red)':'var(--dim)'; }
      else{ statusEl.textContent='⌁ '+(j.error||'error').slice(0,42); statusEl.style.color='var(--amber)'; }
    }catch(e){ statusEl.textContent='⌁ server'; }
    busy=false; if(pending){pending=false;analyze()}
  }
  if(modeSel) modeSel.onchange=function(){ const v=modeSel.value;
    if(v==='profile'){MODE='trace';PROFILE=true;} else {MODE=v;PROFILE=false;} analyze(); };
  // driver hooks (also used by the recording rig)
  window.nsSetMode=function(m,p){MODE=m;PROFILE=!!p; if(modeSel) modeSel.value=(p?'profile':m);};
  window.nsCaption=function(t){ if(t){cap.innerHTML=t;cap.classList.add('show')} else cap.classList.remove('show') };
  window.nsSet=function(t){ code.value=t; sync(); return analyze(); };
  window.nsReplace=function(a,b){ code.value=code.value.replace(a,b); sync(); return analyze(); };
  window.nsType=function(t,per){ per=per||42; return new Promise(function(res){ let i=0;
    (function step(){ if(i>=t.length){ analyze().then(res); return; }
      code.value+=t[i++]; sync(); code.scrollTop=code.scrollHeight; setTimeout(step,per); })(); }); };
  window.nsReset=function(){ return fetch('/analyze',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({reset:true,code:'',mode:'static'})}); };
  const STARTER='import torch, torch.nn as nn\n\nmodel = nn.Sequential(\n    nn.Linear(64, 128),\n    nn.ReLU(),\n    nn.Linear(128, 10),\n)\nx = torch.randn(8, 64)\n';
  window.nsReady=true; code.value=STARTER; sync(); analyze();
</script></body></html>'''


def main(argv=None) -> int:
    """CLI entry: `netscope playground [port] [--no-open]` (and `python -m netscope.playground`)."""
    import argparse
    ap = argparse.ArgumentParser(prog="netscope.playground",
                                 description="A local paste-a-model live netscope playground.")
    ap.add_argument("port", nargs="?", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    a = ap.parse_args(argv)
    serve(a.port, open_browser=not a.no_open)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
