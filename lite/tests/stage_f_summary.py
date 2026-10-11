#!/usr/bin/env python3
"""Summarise tests/stage_f.sh: per prompt and overall KL / top-1 against llama.cpp all-GPU for stage e and stage f,
a paired bootstrap of f - e over the same steps, MTP acceptance, decode tok/s, greedy identity with and without MTP,
the twin check. Numbers only (shareable). Usage: stage_f_summary.py GATE1F_DIR"""
import json, random, struct, sys
from pathlib import Path
O = Path(sys.argv[1]); ids = ["p1", "p2", "p3", "p4", "p5"]
J = lambda i, r: json.loads((O / f"{i}.{r}.json").read_text())
steps = lambda i, r: [float(l.split()[0]) for l in open(O / f"{i}.{r}.steps")]
tot = {r: [0, 0, 0.0] for r in ("e-force", "f-force")}
acc = [0, 0, 0]; tps = {"e-greedy": [], "f-greedy": []}; dbad = drows = 0
for i in ids:
    line = [i]
    for r in ("e-force", "f-force"):
        d = J(i, r); tot[r][0] += d["steps_compared"]; tot[r][1] += d["argmax_same"]; tot[r][2] += d["kl_mean"] * d["steps_compared"]
        line.append(f"{r} top1 {d['argmax_same']}/{d['steps_compared']} KL {d['kl_mean']:.2e}")
    f = J(i, "f-greedy"); e = J(i, "e-greedy")
    a = open(O / f"{i}.e-greedy.tok", "rb").read(); b = open(O / f"{i}.f-greedy.tok", "rb").read()
    a = struct.unpack(f"<{len(a)//4}i", a); b = struct.unpack(f"<{len(b)//4}i", b)
    same = a == b
    diff = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
    acc[0] += f["drafted"]; acc[1] += f["accepted"]; acc[2] += f["mtp_windows"]
    tps["e-greedy"].append(e["decode_tok_s"]); tps["f-greedy"].append(f["decode_tok_s"]); dbad += f["dcheck_bad"]; drows += f["dcheck_rows"]
    line.append(f"greedy e/f {'identical' if same else f'differ at {diff}'} ({len(a)}/{len(b)} tokens); accepted {f['accepted']}/{f['drafted']} "
                f"in {f['mtp_windows']} windows; tok/s {e['decode_tok_s']:.1f} -> {f['decode_tok_s']:.1f}; hits {f['hit_rate']:.3f}")
    print(" | ".join(line))
for r, (st, t1, kl) in tot.items():
    print(f"ALL {r} vs all-GPU: top-1 {t1}/{st} = {100*t1/st:.2f}%, mean KL {kl/st:.3e}")
E = {i: steps(i, "e-force") for i in ids}; F = {i: steps(i, "f-force") for i in ids}
N = sum(len(v) for v in E.values()); random.seed(7); dd = []
for _ in range(5000):
    s = 0.0
    for i in ids:
        n = min(len(E[i]), len(F[i]))
        for _k in range(n):
            j = random.randrange(n); s += F[i][j] - E[i][j]
    dd.append(s / N)
dd.sort()
print(f"f - e mean KL: {(sum(map(sum, F.values())) - sum(map(sum, E.values()))) / N:.3e}, 95% CI [{dd[int(.025*len(dd))]:.3e}, {dd[int(.975*len(dd))]:.3e}]")
print(f"MTP: accepted {acc[1]}/{acc[0]} = {100*acc[1]/acc[0]:.1f}% of drafts, {(acc[1]+acc[2])/acc[2]:.2f} tokens per verify window")
print(f"decode tok/s without MTP {sum(tps['e-greedy'])/5:.1f}, with MTP {sum(tps['f-greedy'])/5:.1f} (mean of the five)")
print(f"twin check (layer 3, with MTP windows): {drows} rows, {dbad} differ")
