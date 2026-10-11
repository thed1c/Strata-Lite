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
.msg.old{opacity:.55}details.compact{margin:16px 0;padding:8px 12px;border:1px dashed var(--line);border-radius:10px}
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
// history: what the model sees; each entry may carry tok (an estimate of its tokens) for compaction.
// used: tokens in the context after the last reply. Compaction: when used passes 3/4 of the context (192K of 256K), or
// a new message does not fit, the whole conversation is summarized (one request: its prefix is already cached) and the
// chat goes on from the summary plus the last turns word for word (with a too-big message, the summary alone).
let history = [], pending = [], busy = false, ctrl = null, nctx = 0, used = 0;
const limit = () => Math.floor(nctx * 0.75), keepTok = () => Math.min(16384, Math.floor(nctx * 0.08));
const SUMMARY_ASK = "Write a summary of our whole conversation so far, for you to continue from after the earlier messages " +
  "are removed to free context. Keep every fact, name, number, decision, instruction I gave you, piece of code or text we " +
  "are working on, and open question, in as much detail as they need (up to about 1500 words). Write only the summary.";
fetch("v1/models").then(r => r.json()).then(j => {
  const m = j.data && j.data[0]; $("model").textContent = m ? m.id : "";
}).catch(() => { $("model").textContent = "not reachable"; });
fetch("props").then(r => r.json()).then(p => {
  if (!(p.modalities && p.modalities.vision)) $("attach").hidden = true;
  nctx = (p.default_generation_settings || {}).n_ctx || 0;
  const s = p.strata || {};
  if (s.slots !== undefined) $("model").textContent += "  -  " + s.slots + " expert slots per layer, " + nctx + " context";
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
$("clear").onclick = () => { if (busy) return; history = []; used = 0; $("log").innerHTML = ""; };
$("stop").onclick = () => { if (ctrl) ctrl.abort(); };
$("input").onkeydown = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
$("send").onclick = send;
const strip = h => h.map(m => ({role: m.role, content: m.content}));
async function chat(messages, max_tokens, think) {   // one request; a context overflow comes back as {overflow: n_prompt_tokens}
  const r = await fetch("v1/chat/completions", {method: "POST", signal: ctrl.signal, headers: {"Content-Type": "application/json"},
    body: JSON.stringify({messages: strip(messages), max_tokens: max_tokens, chat_template_kwargs: {enable_thinking: think}})});
  const j = await r.json();
  if (!r.ok) {
    const e = j.error || {};
    if (e.type === "exceed_context_size_error" || j.n_prompt_tokens || e.n_prompt_tokens) return {overflow: j.n_prompt_tokens || e.n_prompt_tokens || 0};
    throw new Error(e.message || r.statusText);
  }
  return j;
}
const total = j => { const u = j.usage || {}, t = j.timings || {};
  return (u.prompt_tokens || ((t.cache_n || 0) + (t.prompt_n || 0))) + (u.completion_tokens || t.predicted_n || 0); };
// Summarize the whole conversation; keep the last turns that fit keepTok() word for word (none when bare).
async function compact(bare, status, anchor) {   // anchor: the new message's element; the marker goes before it
  if (!history.length) return false;
  status.textContent = "Summarizing the earlier messages to free context...";
  const j = await chat(history.concat([{role: "user", content: SUMMARY_ASK}]), 3072, false);
  if (j.overflow !== undefined) return false;
  const s = (j.choices[0].message.content || "").trim();
  if (!s) return false;
  let keep = [], k = 0;
  if (!bare) for (let i = history.length - 1; i >= 0; --i) {
    if (history[i].role === "system") break;
    k += history[i].tok || 0; if (k > keepTok()) break; keep.unshift(history[i]);
  }
  while (keep.length && keep[0].role !== "user") keep.shift();
  const before = used, stok = (j.usage || {}).completion_tokens || 0;
  history = [{role: "system", content: "Summary of the earlier part of this conversation (its messages were removed to free " +
    "context):\n\n" + s, tok: stok + 40}].concat(keep);
  used = history.reduce((a, m) => a + (m.tok || 0), 0);
  for (let e = anchor.previousElementSibling; e; e = e.previousElementSibling) if (e.classList.contains("msg")) e.classList.add("old");
  const d = document.createElement("details"); d.className = "compact";
  d.innerHTML = "<summary></summary>"; d.firstChild.textContent = "Earlier messages summarized to free context (about " +
    Math.round(before / 1000) + "K tokens to " + Math.round(used / 1000) + "K" + (keep.length ? ", the last " + keep.length + " messages kept word for word" : "") + "). Show the summary";
  const p = document.createElement("pre"); p.textContent = s; d.appendChild(p);
  $("log").insertBefore(d, anchor);
  return true;
}
async function send() {
  const text = $("input").value.trim();
  if (busy || (!text && !pending.length)) return;
  const imgs = pending; pending = []; $("files").textContent = ""; $("input").value = "";
  const udiv = add("user", text, imgs);
  const content = imgs.length ? [{type: "text", text: text}].concat(imgs.map(u => ({type: "image_url", image_url: {url: u}}))) : text;
  const msg = {role: "user", content: content};
  const out = add("assistant", "..."); busy = true; $("send").disabled = true; $("stop").hidden = false;
  ctrl = new AbortController();
  const maxtok = +$("maxtok").value;
  try {
    if (nctx && used + maxtok > limit()) await compact(false, out, udiv);
    out.textContent = "...";
    let j = await chat(history.concat([msg]), maxtok, $("think").checked);
    // the new message does not fit beside the conversation: summarize to make room, unless it could never fit
    // (its own tokens: the prompt's count minus the conversation's), which leaves the conversation as it is
    const alone = j.overflow !== undefined ? j.overflow - history.reduce((a, m) => a + (m.tok || 0), 0) : 0;
    if (j.overflow !== undefined && history.length && alone + maxtok < nctx) {
      if (!history.some(m => m.role === "system") || history.length > 1) await compact(true, out, udiv);
      out.textContent = "...";
      j = await chat(history.concat([msg]), maxtok, $("think").checked);
    }
    if (j.overflow !== undefined) throw new Error("this message is about " + (alone > 0 ? alone : j.overflow) + " tokens; the context holds " + nctx +
      (maxtok ? " (minus room for the answer)" : "") + ". Send it in parts or shorten it.");
    const prev = used; used = total(j);
    msg.tok = Math.max(1, ((j.usage || {}).prompt_tokens || used) - prev);
    history.push(msg);
    const m = j.choices[0].message; out.textContent = "";
    if (m.reasoning_content) { const d = document.createElement("details"); d.innerHTML = "<summary>thinking</summary>";
      const p = document.createElement("pre"); p.textContent = m.reasoning_content; d.appendChild(p); out.appendChild(d); }
    out.appendChild(document.createTextNode(m.content || ""));
    const t = j.timings || {}, meta = document.createElement("div"); meta.className = "meta";
    meta.textContent = (t.predicted_n || 0) + " tokens, " + (t.predicted_per_second || 0).toFixed(1) + " tok/s; prompt " +
      (t.prompt_n || 0) + " read at " + (t.prompt_per_second || 0).toFixed(0) + " tok/s" + (t.cache_n ? ", " + t.cache_n + " cached" : "") +
      (t.draft_n ? "; drafts " + t.draft_n_accepted + "/" + t.draft_n : "");
    out.appendChild(meta);
    history.push({role: "assistant", content: m.content || "", tok: (j.usage || {}).completion_tokens || 0});
  } catch (e) { out.textContent = "Error: " + e.message; }   // the message was not added; a finished compaction stays
  busy = false; $("send").disabled = false; $("stop").hidden = true; window.scrollTo(0, document.body.scrollHeight);
}
</script></body></html>
)HTML");
    return page;
}

}  // namespace lite
