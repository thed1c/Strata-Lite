#!/usr/bin/env python3
"""Summarise tests/q35_stage_g.sh (numbers only). Usage: q35_stage_g_summary.py Q35G_DIR"""
import json, sys
from pathlib import Path
O = Path(sys.argv[1])
print("| prompt | chunk | prompt tok/s | first token (s) | VRAM lent (GB) | read compute buffer (MB) | decode tok/s, first 200 | hits: steps 1-5 / 6-20 / 21+ | refill landed (s) | slots after |")
print("|---|---|---|---|---|---|---|---|---|---|")
for spec in ("p5k", "p15k", "long-a"):
    for P in ("512", "4096", "8192", "16384", "32768", "auto"):
        f = O / f"{spec}.P{P}.json"
        if not f.exists():
            log = O / f"{spec}.P{P}.log"
            if log.exists(): print(f"| {spec} | {P} | does not fit ({'out of memory' if 'out of memory' in log.read_text(errors='replace') else 'failed'}) | | | | | | | |")
            continue
        d = json.loads(f.read_text())
        st = [tuple(map(int, l.split())) for l in open(O / f"{spec}.P{P}.steps")] if (O / f"{spec}.P{P}.steps").exists() else []
        def hr(a, b):
            acc = sum(x[0] for x in st[a:b]); hit = sum(x[1] for x in st[a:b]); return f"{hit/acc:.2f}" if acc else "-"
        chunk = P
        if P == "auto":
            m = [l for l in (O / f"{spec}.P{P}.log").read_text(errors="replace").splitlines() if "-token chunks" in l]
            if m: chunk = "auto: " + m[0].split(" in ")[1].split("-token")[0]
        land = f"{d['refill_land_ms']/1000:.2f}" if d["refill_land_ms"] >= 0 else "-"
        print(f"| {spec} | {chunk} | {d['prompt']/d['prefill_s']:.0f} | {d['prefill_s']:.1f} | {d['lent_gb']:.1f} | {d['read_buf_mb']:.0f} | "
              f"{d['decode_tok_s']:.1f} | {hr(0,5)} / {hr(5,20)} / {hr(20,None)} | {land} | {d['slots_end']} |")
