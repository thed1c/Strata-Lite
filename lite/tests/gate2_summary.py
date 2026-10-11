#!/usr/bin/env python3
"""Strata-Lite Gate 2: the numbers from gate2.sh's output folder as markdown (numbers only; no text, no tokens)."""
import glob, json, os, sys

O = sys.argv[1]
FIVE = ["p1", "p2", "p3", "p4", "p5"]


def load(p):
    try: return json.load(open(p))
    except Exception: return None


def last_json(p):
    try:
        for l in reversed(open(p).read().splitlines()):
            if l.strip().startswith("{"): return json.loads(l)
    except Exception: pass
    return None


print("# Gate 2 on machine B (RTX 3060 1A:00.0, x16) — numbers only\n")
print("## R: the reference server's config (two 3060s), greedy reference\n")
print("| prompt | prompt tokens | generated | prefill s | decode tok/s |\n|---|---|---|---|---|")
for id in FIVE + ["g157"]:
    d = load(f"{O}/sb/{id}.json")
    if d: print(f"| {id} | {d['prompt_tokens']} | {d['gen_tokens']} | {d['prefill_s']:.1f} | {d['gen_tokens'] / max(d['decode_s'], 1e-9):.1f} |")

print("\n## B: sm_86 bit-exactness (engine vs llama.cpp on the 3060, all experts on the CPU, same split points)\n")
for p in sorted(glob.glob(f"{O}/b/*.g4.json")):
    d = load(p) or {}
    print(f"- {os.path.basename(p)[:-8]}: bit-identical {d.get('logits_bit_identical')}, max abs diff {d.get('max_abs_diff')}, steps {d.get('steps_compared')}")

print("\n## Q: against the reference server's own output (teacher-forced through its tokens)\n")
print("| prompt | port top-1 | port mean KL | twins (bad) | slots | hit rate | llama.cpp static top-1 | static mean KL |\n|---|---|---|---|---|---|---|---|")
tot = {"p_a": 0, "p_n": 0, "p_kl": [], "s_a": 0, "s_n": 0, "s_kl": [], "rows": 0, "bad": 0}
for id in FIVE + ["g157"]:
    d = load(f"{O}/q/{id}.port.json"); s = last_json(f"{O}/q/{id}.static20.cmp")
    if not d and not s: continue
    row = [id]
    if d:
        row += [f"{d['argmax_same']}/{d['steps_compared']}", f"{d['kl_mean']:.2e}", f"{d['dcheck_rows']} ({d['dcheck_bad']})", d['cache_slots'], f"{d.get('hit_rate', 0):.3f}"]
        if id != "g157":
            tot["p_a"] += d["argmax_same"]; tot["p_n"] += d["steps_compared"]; tot["p_kl"].append(d["kl_mean"])
        tot["rows"] += d["dcheck_rows"]; tot["bad"] += d["dcheck_bad"]
    else: row += ["—"] * 5
    if s:
        row += [f"{s['top1_agree']}/{s['steps']}", f"{s['kl_mean']:.2e}"]
        tot["s_a"] += s["top1_agree"]; tot["s_n"] += s["steps"]; tot["s_kl"].append(s["kl_mean"])
    else: row += ["—", "—"]
    print("| " + " | ".join(str(x) for x in row) + " |")
if tot["p_n"]:
    print(f"\n- Five prompts, port: top-1 {tot['p_a']}/{tot['p_n']} ({100 * tot['p_a'] / tot['p_n']:.1f} %), mean KL {sum(tot['p_kl']) / len(tot['p_kl']):.2e}; twins {tot['rows']} rows, {tot['bad']} differ.")
if tot["s_n"]:
    print(f"- Five prompts, llama.cpp static one-card: top-1 {tot['s_a']}/{tot['s_n']} ({100 * tot['s_a'] / tot['s_n']:.1f} %), mean KL {sum(tot['s_kl']) / len(tot['s_kl']):.2e}.")

print("\n## S: speed through the API (frozen requests as captured, max_tokens 300; cold, then the same request again)\n")
print("| server | request | prompt n | prefill tok/s | warm cache n | decode tok/s cold / warm | MTP accepted | VRAM MiB |\n|---|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(f"{O}/s/*.json")):
    if os.path.basename(p).startswith("parity"): continue
    d = load(p)
    if not d: continue
    for rid, r in d["requests"].items():
        c = r["cold"]; w = r.get("warm") or {}
        if c.get("code") != 200:
            print(f"| {d['label']} | {rid} | {c.get('prompt_n') or '—'} | HTTP {c.get('code')} {c.get('error_type') or c.get('error') or ''} | | | | |"); continue
        acc = f"{c['draft_accepted']}/{c['draft_n']}" if c.get("draft_n") else "—"
        vr = r.get("vram_mib") or c.get("vram_mib") or {}
        print(f"| {d['label']} | {rid} | {c['prompt_n']} | {c['prompt_tok_s']} | {w.get('cache_n', '')} | {c['gen_tok_s']} / {w.get('gen_tok_s', '')} | {acc} | {'+'.join(str(v) for v in vr.values())} |")
for p in sorted(glob.glob(f"{O}/s/parity-*.txt")):
    try: print(f"\n- API parity, {os.path.basename(p)[7:-4]}: {open(p).read().strip().splitlines()[-1]}")
    except Exception: pass
