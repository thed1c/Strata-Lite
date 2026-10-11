// Strata-Lite: the chat page strata-lite-server serves at GET / (the setup opens it when the model is ready). One
// self-contained HTML file: text, images, thinking on or off, the answer's speed. It talks to /v1/chat/completions.
// Split into pieces under MSVC's 16 KB string-literal limit.
#pragma once
#include <string>

namespace lite {

inline const std::string & chat_page() {
    static const std::string page = std::string(R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strata-Lite</title>
<style>
:root{--bg:#f6f6f4;--fg:#1c1c1a;--muted:#6b6b66;--card:#fff;--line:#e2e1dc;--acc:#2f6f5e;--user:#e9f1ee}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#e8e7e2;--muted:#9a9993;--card:#1f1f1d;--line:#2f2e2b;--acc:#7cc0aa;--user:#21302b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
header b{font-size:16px}header span{color:var(--muted);font-size:13px}header .sp{flex:1}
button,select{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
main{max-width:860px;margin:0 auto;padding:16px 16px 180px}
.msg{margin:12px 0;padding:10px 14px;border-radius:12px;background:var(--card);border:1px solid var(--line);white-space:pre-wrap;word-wrap:break-word}
.msg.user{background:var(--user)}.msg img{max-width:240px;border-radius:8px;display:block;margin-top:6px}
.meta{color:var(--muted);font-size:12px;margin-top:6px}
details{color:var(--muted);font-size:13px;margin-bottom:6px}details pre{white-space:pre-wrap;margin:6px 0 0}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 16px}
.box{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:8px}
textarea{width:100%;min-height:64px;max-height:40vh;resize:vertical;font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.row .sp{flex:1}
.chip{font-size:12px;color:var(--muted)}
</style></head><body>
<header><b>Strata-Lite</b><span id="model">connecting...</span><span class="sp"></span>
<button id="clear" title="Start a new conversation">New chat</button></header>
<main id="log"></main>
<footer><div class="box">
<textarea id="input" placeholder="Message (Enter to send, Shift+Enter for a new line)"></textarea>
<div class="row">
<label class="chip"><input type="checkbox" id="think" checked> thinking</label>
<label class="chip">max tokens <select id="maxtok"><option>1024</option><option selected>4096</option><option>16384</option></select></label>
<input type="file" id="file" accept="image/*" multiple hidden>
<button id="attach" title="Attach images">Image</button><span class="chip" id="files"></span>
<span class="sp"></span><button id="stop" hidden>Stop</button><button class="primary" id="send">Send</button>
</div></div></footer>
)HTML") + std::string(R"HTML(<script>
const $ = id => document.getElementById(id);
let history = [], pending = [], busy = false, ctrl = null;
fetch("v1/models").then(r => r.json()).then(j => {
  const m = j.data && j.data[0]; $("model").textContent = m ? m.id : "";
}).catch(() => { $("model").textContent = "not reachable"; });
fetch("props").then(r => r.json()).then(p => {
  if (!(p.modalities && p.modalities.vision)) $("attach").hidden = true;
  const s = p.strata || {};
  if (s.slots !== undefined) $("model").textContent += "  -  " + s.slots + " expert slots per layer, " + (p.default_generation_settings || {}).n_ctx + " context";
}).catch(() => {});
function add(role, text, imgs) {
  const d = document.createElement("div"); d.className = "msg " + role;
  if (text) d.appendChild(document.createTextNode(text));
  (imgs || []).forEach(u => { const i = document.createElement("img"); i.src = u; d.appendChild(i); });
  $("log").appendChild(d); window.scrollTo(0, document.body.scrollHeight); return d;
}
$("attach").onclick = () => $("file").click();
$("file").onchange = async () => {
  for (const f of $("file").files) pending.push(await new Promise(ok => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(f); }));
  $("files").textContent = pending.length ? pending.length + " image(s)" : ""; $("file").value = "";
};
$("clear").onclick = () => { if (busy) return; history = []; $("log").innerHTML = ""; };
$("stop").onclick = () => { if (ctrl) ctrl.abort(); };
$("input").onkeydown = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
$("send").onclick = send;
async function send() {
  const text = $("input").value.trim();
  if (busy || (!text && !pending.length)) return;
  const imgs = pending; pending = []; $("files").textContent = ""; $("input").value = "";
  add("user", text, imgs);
  const content = imgs.length ? [{type: "text", text: text}].concat(imgs.map(u => ({type: "image_url", image_url: {url: u}}))) : text;
  history.push({role: "user", content: content});
  const out = add("assistant", "..."); busy = true; $("send").disabled = true; $("stop").hidden = false;
  ctrl = new AbortController();
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", signal: ctrl.signal, headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: history, max_tokens: +$("maxtok").value, chat_template_kwargs: {enable_thinking: $("think").checked}})});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    const m = j.choices[0].message; out.textContent = "";
    if (m.reasoning_content) { const d = document.createElement("details"); d.innerHTML = "<summary>thinking</summary>";
      const p = document.createElement("pre"); p.textContent = m.reasoning_content; d.appendChild(p); out.appendChild(d); }
    out.appendChild(document.createTextNode(m.content || ""));
    const t = j.timings || {}, meta = document.createElement("div"); meta.className = "meta";
    meta.textContent = (t.predicted_n || 0) + " tokens, " + (t.predicted_per_second || 0).toFixed(1) + " tok/s; prompt " +
      (t.prompt_n || 0) + " read at " + (t.prompt_per_second || 0).toFixed(0) + " tok/s" + (t.cache_n ? ", " + t.cache_n + " cached" : "") +
      (t.draft_n ? "; drafts " + t.draft_n_accepted + "/" + t.draft_n : "");
    out.appendChild(meta);
    history.push({role: "assistant", content: m.content || ""});
  } catch (e) { out.textContent = "Error: " + e.message; history.pop(); }
  busy = false; $("send").disabled = false; $("stop").hidden = true; window.scrollTo(0, document.body.scrollHeight);
}
</script></body></html>
)HTML");
    return page;
}

}  // namespace lite
