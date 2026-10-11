#!/usr/bin/env python3
"""Strata-Lite Gate 2: the same requests through any llama-server-compatible endpoint (the reference server's config on its two
3060s, llama.cpp static offload + MTP on one 3060, Strata-Lite on one 3060), numbers only.

Each request is a frozen Gate 0 input (gate0/inputs/<id>.json) sent as captured with max_tokens capped, cold, then the
same request again (the prompt cache's second turn). Then the image path: Studio-sized images (1024 px) one and eight
at a time. VRAM is read per card before and after. Nothing of the story is written out: no text, no tokens.

usage: gate2_bench.py BASE LABEL OUT.json --cards UUID[,UUID..] [--ids a,b,..] [--max-tokens 300]
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_parity import png, data_uri   # noqa: E402

E = "$WORK"
IDS = "p1,p2,p3,p3-think,p4,p5,long-a"


def vram(cards):
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used = {l.split(",")[0].strip(): int(l.split(",")[1]) for l in out.strip().splitlines()}
    return {c[:12]: used.get(c) for c in cards}


def post(base, body, timeout=3600):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: code, js = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        code = e.code
        try: js = json.load(e)
        except Exception: js = {}
    except Exception as e:
        return {"code": None, "error": type(e).__name__, "wall_s": round(time.time() - t0, 2)}
    dt = time.time() - t0
    tm = js.get("timings", {}) if isinstance(js, dict) else {}
    ch = (js.get("choices") or [{}])[0] if isinstance(js, dict) else {}
    msg = ch.get("message", {})
    r = {"code": code, "wall_s": round(dt, 2), "finish": ch.get("finish_reason"),
         "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "predicted_n": tm.get("predicted_n"),
         "prompt_tok_s": round(tm.get("prompt_per_second") or 0, 1), "gen_tok_s": round(tm.get("predicted_per_second") or 0, 1),
         "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
         "content_chars": len(msg.get("content") or ""), "reasoning_chars": len(msg.get("reasoning_content") or "")}
    if code != 200: r["error_type"] = (js.get("error") or {}).get("type") if isinstance(js, dict) else None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--cards", required=True); ap.add_argument("--ids", default=IDS); ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args(); base = a.base.rstrip("/"); cards = a.cards.split(",")
    for _ in range(240):   # up to 20 min for a 262k load
        try: urllib.request.urlopen(base + "/health", timeout=3); break
        except Exception: time.sleep(5)
    props = json.load(urllib.request.urlopen(base + "/props", timeout=30))
    res = {"label": a.label, "base": base, "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
           "build": props.get("build_info"), "vram_idle_mib": vram(cards), "requests": {}}
    for rid in a.ids.split(","):
        body = json.load(open(f"{E}/gate0/inputs/{rid}.json"))
        body["model"] = "reference"; body["max_tokens"] = min(int(body.get("max_tokens") or a.max_tokens), a.max_tokens)
        body["cache_prompt"] = True; body.pop("stream", None)
        cold = post(base, body); warm = post(base, body) if cold.get("code") == 200 else None
        res["requests"][rid] = {"cold": cold, "warm": warm, "vram_mib": vram(cards)}
        print(f"{a.label:14s} {rid:26s} cold {json.dumps(cold)}", flush=True)
        if warm: print(f"{a.label:14s} {rid:26s} warm {json.dumps(warm)}", flush=True)
    if not a.no_images:
        img = data_uri(png(1024, 768, (60, 110, 170)), "image/png")
        for n in (1, 8):
            body = {"model": "reference", "max_tokens": 64, "temperature": 0.4, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe these images in one sentence each."}] +
                                  [{"type": "image_url", "image_url": {"url": img}} for _ in range(n)]}]}
            r = post(base, body); r["vram_mib"] = vram(cards)
            res["requests"][f"image-x{n}"] = {"cold": r}
            print(f"{a.label:14s} image-x{n:<19d} {json.dumps(r)}", flush=True)
    res["vram_end_mib"] = vram(cards)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
