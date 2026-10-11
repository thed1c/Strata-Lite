#!/usr/bin/env python3
"""Strata-Lite API parity: the reference server's clients' request shapes against a server, checking what each client reads.

    api_parity.py BASE_URL [--ref REF_BASE_URL] [--out results.json]

The request set and the checks come from a read-only survey of the reference server's clients (2026-10-10, docs/API.md): a router's health check (/v1/models, /props model_path), a job runner (/props modalities.vision), the
watchdog (/slots, a 32x32 PNG with max_tokens 1 inside 12 s), an agent (tools, no sampling fields), the dashboard
(max_tokens), a studio client (temperature 0.4, JPEG image_url), the mode owner (cache_prompt, enable_thinking
false, min_p, timings), the coordinator (response_format json_schema strict, no chat_template_kwargs), and a context
overflow (must be a 4xx). All content is synthetic. With --ref, the same requests go to a reference llama-server and
the response *shapes* (keys and types of what clients read) are compared too. Stdlib only.
"""
import argparse, base64, io, json, struct, sys, time, urllib.error, urllib.request, zlib

EXPECTED_MODEL_FILE = "Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
ALIAS = "reference"


def png(w, h, rgb=(200, 80, 40)):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def data_uri(b, mime): return f"data:{mime};base64," + base64.b64encode(b).decode()


def call(base, path, body=None, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer ignored-by-server"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode(); code = r.status; ctype = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace"); code = e.code; ctype = e.headers.get("Content-Type", "")
    dt = time.time() - t0
    try: js = json.loads(txt)
    except Exception: js = None
    return code, js, dt, ctype


def shape(x, depth=0):
    """the keys and types a client can read (lists: the first element)"""
    if isinstance(x, dict): return {k: shape(v, depth + 1) for k, v in sorted(x.items())} if depth < 6 else "dict"
    if isinstance(x, list): return [shape(x[0], depth + 1)] if x else []
    return type(x).__name__


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["ok", "fix"]}, "score": {"type": "integer"}},
          "required": ["verdict", "score"], "additionalProperties": False}
LONG_DOC = "The quick brown fox jumps over the lazy dog. " * 400


def requests_set():
    jpeg_like = data_uri(png(64, 48, (30, 120, 220)), "image/png")   # a PNG stands in for Studio's JPEGs (same image_url path)
    return [
        ("agent_tools", {"model": ALIAS, "messages": [{"role": "system", "content": "You are a helpful assistant. Use tools when useful."},
                                                    {"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
                        "tools": TOOLS, "tool_choice": "auto", "chat_template_kwargs": {"enable_thinking": False}}),
        ("dashboard", {"model": ALIAS, "messages": [{"role": "system", "content": "Answer in one sentence."},
                                                   {"role": "user", "content": "Name a primary colour."}], "max_tokens": 64,
                       "chat_template_kwargs": {"enable_thinking": False}}),
        ("studio_image", {"model": ALIAS, "max_tokens": 64, "temperature": 0.4, "chat_template_kwargs": {"enable_thinking": False},
                          "messages": [{"role": "user", "content": [{"type": "text", "text": "What colour dominates this image? One word."},
                                                                     {"type": "image_url", "image_url": {"url": jpeg_like}}]}]}),
        ("owner_cache_1", {"model": ALIAS, "max_tokens": 48, "temperature": 0.7, "min_p": 0.05, "cache_prompt": True,
                           "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "system", "content": "You summarise documents."},
                                        {"role": "user", "content": LONG_DOC[:6000] + "\nSummarise in one line."}]}),
        ("owner_cache_2", {"model": ALIAS, "max_tokens": 48, "temperature": 0.7, "min_p": 0.05, "cache_prompt": True,
                           "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "system", "content": "You summarise documents."},
                                        {"role": "user", "content": LONG_DOC[:6000] + "\nGive it a title."}]}),
        ("coordinator_schema", {"model": ALIAS, "temperature": 0, "max_tokens": 1500,
                                "response_format": {"type": "json_schema", "json_schema": {"name": "review", "strict": True, "schema": SCHEMA}},
                                "messages": [{"role": "system", "content": "You review short texts and reply in JSON."},
                                             {"role": "user", "content": "Review: 'The cat sat on the mat.'"}]}),
        ("thinking_default", {"model": ALIAS, "max_tokens": 200, "messages": [{"role": "user", "content": "What is 17 * 23?"}]}),
        ("watchdog_image", {"model": ALIAS, "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                            "messages": [{"role": "user", "content": [{"type": "text", "text": "ok"},
                                                                      {"type": "image_url", "image_url": {"url": data_uri(png(32, 32), "image/png")}}]}]}),
        ("overflow", {"model": ALIAS, "max_tokens": 8, "messages": [{"role": "user", "content": LONG_DOC * 200}]}),
    ]


def check(name, code, js, dt):
    """what the surveyed clients need from each reply: a list of (ok, what)"""
    out = []
    ok = lambda c, w: out.append((bool(c), w))
    if name == "overflow":
        ok(400 <= code < 500, f"context overflow is a 4xx (got {code})"); return out
    ok(code == 200, f"HTTP 200 (got {code})")
    if code != 200 or not js: return out
    ch = (js.get("choices") or [{}])[0]; msg = ch.get("message", {})
    ok(js.get("model") == ALIAS, f"model echoed as {ALIAS} (got {js.get('model')})")
    ok(isinstance(js.get("choices"), list) and len(js["choices"]) > 0, "choices non-empty")
    ok(ch.get("finish_reason") in ("stop", "length", "tool_calls"), f"finish_reason ({ch.get('finish_reason')})")
    u = js.get("usage", {})
    ok(all(isinstance(u.get(k), int) for k in ("prompt_tokens", "completion_tokens", "total_tokens")), "usage.prompt/completion/total_tokens")
    t = js.get("timings", {})
    ok(all(k in t for k in ("prompt_n", "cache_n", "predicted_n", "prompt_per_second", "predicted_per_second")), "timings.prompt_n/cache_n/predicted_n/*_per_second")
    if name == "agent_tools":
        tc = msg.get("tool_calls") or []
        ok(len(tc) > 0, "a tool call")
        if tc:
            f = tc[0].get("function", {})
            try: args = json.loads(f.get("arguments", ""))
            except Exception: args = None
            ok(f.get("name") == "get_weather" and isinstance(args, dict), "tool_calls[0].function name + JSON-object arguments")
    if name == "coordinator_schema":
        c = (msg.get("content") or "").strip().strip("`")
        if c.startswith("json"): c = c[4:]
        try: v = json.loads(c)
        except Exception: v = None
        ok(isinstance(v, dict) and v.get("verdict") in ("ok", "fix") and isinstance(v.get("score"), int) and set(v) == {"verdict", "score"}, "content is JSON matching the strict schema")
    if name == "thinking_default":
        ok(bool(msg.get("reasoning_content")) or bool(msg.get("content")), "reasoning_content or content present")
    if name == "owner_cache_2":
        ok(isinstance(t.get("cache_n"), int) and t.get("cache_n", 0) > 0, f"cache_prompt reused the prefix (cache_n {t.get('cache_n')})")
    if name == "watchdog_image":
        ok(dt < 12.0, f"one-token image reply inside the watchdog's 12 s ({dt:.1f} s)")
    if name in ("dashboard", "studio_image"):
        ok(bool(msg.get("content")), "content non-empty")
    return out


def run(base):
    res = {}
    for path in ("/health", "/v1/models", "/props", "/slots"):
        code, js, dt, _ = call(base, path)
        res[path] = {"code": code, "json": js, "shape": shape(js), "s": round(dt, 2)}
    for name, body in requests_set():
        code, js, dt, ctype = call(base, "/v1/chat/completions", body)
        res[name] = {"code": code, "json": js, "shape": shape(js), "s": round(dt, 2), "ctype": ctype, "checks": check(name, code, js, dt)}
    # endpoint checks
    p = res["/props"]["json"] or {}
    import os
    res["/props"]["checks"] = [(os.path.basename(str(p.get("model_path", ""))) == EXPECTED_MODEL_FILE, f"props.model_path basename = {EXPECTED_MODEL_FILE}"),
                               (bool((p.get("modalities") or {}).get("vision")) or bool(p.get("vision")), "props.modalities.vision true")]
    s = res["/slots"]["json"]
    res["/slots"]["checks"] = [(isinstance(s, list) and len(s) > 0 and all("is_processing" in x for x in s), "/slots non-empty list with is_processing")]
    m = res["/v1/models"]
    res["/v1/models"]["checks"] = [(m["code"] == 200, "/v1/models 200"), (any(d.get("id") == ALIAS for d in (m["json"] or {}).get("data", [])), f"/v1/models lists {ALIAS}")]
    res["/health"]["checks"] = [(res["/health"]["code"] == 200, "/health 200")]
    return res


def shape_diff(a, b, path=""):
    d = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: d.append(f"{path}.{k}: missing here")
            elif k not in b: d.append(f"{path}.{k}: extra here")
            else: d += shape_diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if a and b: d += shape_diff(a[0], b[0], path + "[0]")
    elif a != b: d.append(f"{path}: {a} here vs {b} in the reference")
    return d


# the fields clients read (docs/API.md); shape differences elsewhere are reported but not failures
READ = {"choices", "model", "usage", "timings", "data", "model_path", "modalities", "is_processing"}


def main():
    global EXPECTED_MODEL_FILE, ALIAS
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("--ref"); ap.add_argument("--out")
    ap.add_argument("--model-file", default=EXPECTED_MODEL_FILE); ap.add_argument("--alias", default=ALIAS)   # other models
    a = ap.parse_args()
    EXPECTED_MODEL_FILE, ALIAS = a.model_file, a.alias
    res = run(a.base.rstrip("/"))
    ref = run(a.ref.rstrip("/")) if a.ref else None
    n_ok = n_all = 0
    for k, v in res.items():
        for ok, what in v.get("checks", []):
            n_all += 1; n_ok += ok
            print(f"{'PASS' if ok else 'FAIL'}  {k:20s} {what}")
        if ref:
            for d in shape_diff(v["shape"], ref[k]["shape"]):
                top = d.split(".")[1].split("[")[0] if "." in d else d
                print(f"{'DIFF' if top in READ else 'note'}  {k:20s} shape {d}")
    print(f"\n{n_ok}/{n_all} checks pass")
    if a.out:
        strip = lambda r: {k: {kk: vv for kk, vv in v.items() if kk != "json"} for k, v in r.items()}
        json.dump({"server": strip(res), "reference": strip(ref) if ref else None}, open(a.out, "w"), indent=1, default=str)
    sys.exit(0 if n_ok == n_all else 1)


if __name__ == "__main__":
    main()
