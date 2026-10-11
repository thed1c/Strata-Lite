#!/usr/bin/env python3
"""Summarise tests/stage_g.sh (numbers only). Usage: stage_g_summary.py GATE1G_DIR"""
import json, sys
from pathlib import Path
O = Path(sys.argv[1])
print("| prompt | chunk | prompt tok/s | first token (s) | VRAM lent (GB) | window cells | decode tok/s, first 200 | hits: steps 1-5 / 6-20 / 21+ | refill landed (s) |")
print("|---|---|---|---|---|---|---|---|---|")
for spec in ("p5k", "p15k", "long-a"):
    for P in (512, 4096, 8192, 16384, 32768):
        f = O / f"{spec}.P{P}.json"
        if not f.exists():
            log = O / f"{spec}.P{P}.log"
            if log.exists(): print(f"| {spec} | {P} | does not fit ({'out of memory' if 'out of memory' in log.read_text(errors='replace') else 'failed'}) | | | | | | |")
            continue
        d = json.loads(f.read_text())
        st = [tuple(map(int, l.split())) for l in open(O / f"{spec}.P{P}.steps")] if (O / f"{spec}.P{P}.steps").exists() else []
        def hr(a, b):
            acc = sum(x[0] for x in st[a:b]); hit = sum(x[1] for x in st[a:b]); return f"{hit/acc:.2f}" if acc else "-"
        print(f"| {spec} | {P} | {d['prompt']/d['prefill_s']:.0f} | {d['prefill_s']:.1f} | {d['lent_gb']:.1f} | {d['swa_big_cells']} | "
              f"{d['decode_tok_s']:.1f} | {hr(0,5)} / {hr(5,20)} / {hr(20,None)} | {d['refill_land_ms']/1000:.2f} |")
