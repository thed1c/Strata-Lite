#!/usr/bin/env python3
"""Strata-Lite public grid: grid/<card>/c<ctx>/<engine>/ -> one markdown table per card (numbers only)."""
import glob, json, os, sys

G = sys.argv[1]
CTXS = [(32768, "32k"), (131072, "131k"), (262144, "262k")]
ENG = [("lite", "Strata-Lite"), ("llama", "llama.cpp")]
FIVE = ["p1", "p2", "p3", "p4", "p5"]
NB = int(__import__("os").environ.get("GRID_BLOCKS", "30"))   # the model's expert blocks (Gemma 4 26B: 30, Qwen3.6-35B: 40)
CARDS = {"5070": "RTX 5070 (12 GB, 672 GB/s; PCIe x16)", "3060": "RTX 3060 (12 GB, 360 GB/s; PCIe 3.0 x16)"}


def load(p):
    try: return json.load(open(p))
    except Exception: return None


def rd(p):
    try: return open(p).read().strip()
    except Exception: return None


def cell(D, eng):
    s = load(f"{D}/speed.json"); out = {}
    if not s: return None
    r = s["requests"]
    if eng == "lite":
        st = s.get("props_strata") or {}
        out["config"] = f"{st.get('slots', '?')} slots/layer, {rd(f'{D}/chunk')}-token chunks"
        out["chunk"] = rd(f"{D}/chunk")
    else:
        n = rd(f"{D}/ncpumoe"); out["config"] = f"{NB - int(n)} of {NB} expert blocks on the card" if n else "?"
        out["chunk"] = "512 (ubatch)"
    for k, lab in (("p5k", "read 5k"), ("p15k", "read 15k"), ("p157k", "read 157k")):
        c = (r.get(k) or {}).get("cold") or {}
        out[lab] = f"{c['prompt_per_second']:,.0f}" if c.get("code") == 200 and c.get("prompt_per_second") else "—"
    g = (r.get("p157k") or {}).get("cold") or {}
    out["TTFT 157k"] = f"{g['prompt_ms'] / 1000:.0f} s" if g.get("prompt_ms") else "—"
    w = (r.get("p5k") or {}).get("warm") or {}; c5 = (r.get("p5k") or {}).get("cold") or {}
    out["decode"] = f"{w.get('predicted_per_second', 0):.1f}" if w else "—"
    c15 = (r.get("p15k") or {}).get("cold") or {}; w15 = (r.get("p15k") or {}).get("warm") or {}
    if c15.get("predicted_per_second") and w15.get("predicted_per_second"):
        out["refill dip"] = f"{100 * (c15['predicted_per_second'] / w15['predicted_per_second'] - 1):+.0f} %"
    else: out["refill dip"] = "—"
    if w.get("draft_n"): out["MTP accepted"] = f"{100 * w['draft_n_accepted'] / w['draft_n']:.0f} %"
    else: out["MTP accepted"] = "—"
    if g and g.get("predicted_per_second"):
        gw = (r.get("p157k") or {}).get("warm") or {}
        out["decode at 157k"] = f"{gw.get('predicted_per_second', g['predicted_per_second']):.1f}"
    else: out["decode at 157k"] = "—"
    out["VRAM load / peak"] = f"{s['vram_load_mib'] / 1024:.1f} / {max(s['vram_load_mib'], s['vram_peak_mib']) / 1024:.1f} GB"
    # quality
    a = n_ = 0; kls = []; rows = bad = 0
    for id in FIVE:
        q = load(f"{D}/q-{id}.json")
        if not q: continue
        if "argmax_same" in q:
            a += q["argmax_same"]; n_ += q["steps_compared"]; kls.append(q["kl_mean"]); rows += q.get("dcheck_rows", 0); bad += q.get("dcheck_bad", 0)
        else:
            a += q["top1_agree"]; n_ += q["steps"]; kls.append(q["kl_mean"])
    out["top-1, 5 prompts"] = f"{100 * a / n_:.1f} %" if n_ else "—"
    out["KL, 5 prompts"] = f"{sum(kls) / len(kls):.1e}" if kls else "—"
    q = load(f"{D}/q-g157.json")
    if q:
        a1, n1 = (q["argmax_same"], q["steps_compared"]) if "argmax_same" in q else (q["top1_agree"], q["steps"])
        out["top-1 / KL, 157k"] = f"{100 * a1 / n1:.1f} % / {q['kl_mean']:.1e}"
        if "dcheck_rows" in q: rows += q["dcheck_rows"]; bad += q["dcheck_bad"]
    else: out["top-1 / KL, 157k"] = "—"
    out["twin rows (differ)"] = f"{rows:,} ({bad})" if eng == "lite" and rows else ("n/a" if eng == "llama" else "—")
    v = load(f"{D}/vision.json")
    if v is None: out["projector"] = "—"
    else:
        vr = v.get("requests") or {}
        if vr and all(x["cold"].get("code") == 200 for x in vr.values()):
            x1 = vr.get("image-x1", {}).get("cold", {}).get("wall_s"); x8 = vr.get("image-x8", {}).get("cold", {}).get("wall_s")
            if eng == "lite":
                sl = (v.get("props_strata_end") or v.get("props_strata") or {}).get("slots")
                out["projector"] = f"fits ({sl} slots; 1 / 8 images {x1} / {x8} s)"
            else:
                nv = rd(f"{D}/vision_ncpumoe")
                out["projector"] = f"fits ({NB - int(nv)} blocks; 1 / 8 images {x1} / {x8} s)" if nv else "fits"
        else: out["projector"] = "does not fit"
    return out


ROWS = ["config", "read 5k", "read 15k", "read 157k", "TTFT 157k", "decode", "decode at 157k", "MTP accepted", "refill dip",
        "VRAM load / peak", "projector", "top-1, 5 prompts", "KL, 5 prompts", "top-1 / KL, 157k", "twin rows (differ)"]
LABEL = {"read 5k": "prompt tok/s, 5k", "read 15k": "prompt tok/s, 15k", "read 157k": "prompt tok/s, 157k", "decode": "decode tok/s (MTP)",
         "decode at 157k": "decode tok/s at 157k", "refill dip": "decode right after a 15k read", "top-1, 5 prompts": "top-1 vs all-GPU, 5 prompts",
         "KL, 5 prompts": "mean KL vs all-GPU, 5 prompts", "top-1 / KL, 157k": "top-1 / KL vs all-GPU, 157k prompt", "projector": "image projector on the card"}
for tag in sorted(os.listdir(G)):
    if not os.path.isdir(f"{G}/{tag}"): continue
    print(f"\n### {CARDS.get(tag, tag)}\n")
    cols = [(c, cl, e, el) for c, cl in CTXS for e, el in ENG]
    cells = {(c, e): cell(f"{G}/{tag}/c{c}/{e}", e) for c, _, e, _ in cols}
    print("| | " + " | ".join(f"{cl} {el}" for c, cl, e, el in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for row in ROWS:
        print(f"| {LABEL.get(row, row)} | " + " | ".join((cells[(c, e)] or {}).get(row, "—") for c, _, e, _ in cols) + " |")
