#!/usr/bin/env python3
"""Strata-Lite public grid: one engine at one context on one card, the same token prompts for both engines.

POST /completion (llama-server's native endpoint; strata-lite-server serves it too) with the prompt as token ids, greedy
(temperature 0, no penalties), cache_prompt on, n_predict 256. Each prompt is sent cold (read, then decode right after the
read) and again warm (the prompt cache holds it: decode only). With --images, one and eight 1024 px images through the
chat endpoint. VRAM of the card is sampled every 0.25 s: after load, and the peak over the run. Numbers only.

usage: grid_bench.py BASE OUT.json --card UUID --prompts name=path.tokens,... [--n-predict 256] [--images]
"""
import argparse, json, os, struct, subprocess, sys, threading, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_parity import png, data_uri   # noqa: E402


def read_tokens(p):
    b = open(p, "rb").read()
    n, g = struct.unpack("<II", b[:8])
    return list(struct.unpack("<%dI" % n, b[8:8 + 4 * n]))   # the prompt only


class VRAM(threading.Thread):
    def __init__(self, card):
        super().__init__(daemon=True); self.card = card; self.peak = 0; self.last = 0; self.stop = False
    def read(self):
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for l in out.splitlines():
            u, m = [x.strip() for x in l.split(",")]
            if u == self.card: return int(m)
        return 0
    def run(self):
        while not self.stop:
            self.last = self.read(); self.peak = max(self.peak, self.last); time.sleep(0.25)


def post(base, path, body, timeout=7200):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: code, js = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        code = e.code
        try: js = json.load(e)
        except Exception: js = {}
    except Exception as e:
        return {"code": None, "error": type(e).__name__, "wall_s": round(time.time() - t0, 2)}
    tm = js.get("timings", {}) if isinstance(js, dict) else {}
    r = {"code": code, "wall_s": round(time.time() - t0, 2)}
    for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second", "draft_n", "draft_n_accepted", "cache_n"):
        if k in tm: r[k] = round(tm[k], 2) if isinstance(tm[k], float) else tm[k]
    if code != 200: r["error_type"] = ((js.get("error") or {}).get("type") if isinstance(js, dict) else None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("out"); ap.add_argument("--card", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--images", action="store_true"); ap.add_argument("--label", default="")
    a = ap.parse_args(); base = a.base.rstrip("/")
    v = VRAM(a.card)
    for _ in range(480):
        try: urllib.request.urlopen(base + "/health", timeout=3); break
        except Exception: time.sleep(2.5)
    time.sleep(2)
    res = {"label": a.label, "vram_load_mib": v.read(), "requests": {}}
    try: res["props_strata"] = json.load(urllib.request.urlopen(base + "/props", timeout=30)).get("strata")
    except Exception: pass
    v.start()
    for spec in a.prompts.split(","):
        name, path = spec.split("=")
        toks = read_tokens(path)
        body = {"prompt": toks, "n_predict": a.n_predict, "temperature": 0.0, "top_k": 1, "repeat_penalty": 1.0, "cache_prompt": True,
                "ignore_eos": True}
        cold = post(base, "/completion", body)
        warm = post(base, "/completion", body) if cold.get("code") == 200 else None
        res["requests"][name] = {"prompt_tokens": len(toks), "cold": cold, "warm": warm, "vram_peak_mib": v.peak}
        print(f"{a.label:22s} {name:10s} cold {json.dumps(cold)}", flush=True)
        if warm: print(f"{a.label:22s} {name:10s} warm {json.dumps(warm)}", flush=True)
        if cold.get("code") != 200: break
    if a.images:
        img = data_uri(png(1024, 768, (60, 110, 170)), "image/png")
        for n in (1, 8):
            body = {"model": "x", "max_tokens": 32, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe these images."}] +
                                  [{"type": "image_url", "image_url": {"url": img}} for _ in range(n)]}]}
            r = post(base, "/v1/chat/completions", body)
            res["requests"][f"image-x{n}"] = {"cold": r, "vram_peak_mib": v.peak}
            print(f"{a.label:22s} image-x{n:<3d} {json.dumps(r)}", flush=True)
    try: res["props_strata_end"] = json.load(urllib.request.urlopen(base + "/props", timeout=30)).get("strata")
    except Exception: pass
    v.stop = True; time.sleep(0.3)
    res["vram_peak_mib"] = v.peak
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
