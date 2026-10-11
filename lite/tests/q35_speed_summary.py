#!/usr/bin/env python3
"""Summarise tests/q35_speed.sh against stage f (synchronous swaps): KL / top-1 vs llama.cpp all-GPU with a paired
bootstrap of async - sync over the same steps, twin check, greedy tok/s and where the time went. Numbers only.
Usage: q35_speed_summary.py Q35F_DIR Q35A_DIR"""
import json, random, sys
from pathlib import Path
F, A = Path(sys.argv[1]), Path(sys.argv[2])
ids = ["p1", "p2", "p3", "p4", "p5"]
J = lambda d, i, r: json.loads((d / f"{i}.{r}.json").read_text())
st = lambda d, i, r: [float(l.split()[0]) for l in open(d / f"{i}.{r}.steps")]
tot = {"sync": [0, 0, 0.0], "async": [0, 0, 0.0]}; tps = {"sync": [], "async": []}; rows = bad = 0
for i in ids:
    s, a = J(F, i, "f-force"), J(A, i, "a-force")
    for k, d in (("sync", s), ("async", a)):
        tot[k][0] += d["steps_compared"]; tot[k][1] += d["argmax_same"]; tot[k][2] += d["kl_mean"] * d["steps_compared"]
    sg, ag = J(F, i, "s-greedy"), J(A, i, "a-greedy")
    tps["sync"].append(sg["decode_tok_s"]); tps["async"].append(ag["decode_tok_s"]); rows += ag["dcheck_rows"]; bad += ag["dcheck_bad"]
    print(f"{i} | force top1 sync {s['argmax_same']}/{s['steps_compared']} KL {s['kl_mean']:.2e}, async {a['argmax_same']}/{a['steps_compared']} "
          f"KL {a['kl_mean']:.2e} | greedy tok/s {sg['decode_tok_s']:.1f} -> {ag['decode_tok_s']:.1f}; accepted {ag['accepted']}/{ag['drafted']}; "
          f"hits {ag['hit_rate']:.3f}; swap issue {ag['swap_ms']:.0f} ms (sync {sg['swap_ms']:.0f}); pool {ag['pool_ms']:.0f} ms, join wait "
          f"{ag['join_wait_ms']:.0f} ms, miss-free layer-steps {ag['ov_missfree']}/{ag['ov_layers']}; draft {ag['draft_ms']:.0f} ms; decode {ag['decode_s']:.2f} s")
for k, (n, t1, kl) in tot.items():
    print(f"ALL {k} vs all-GPU: top-1 {t1}/{n} = {100*t1/n:.2f}%, mean KL {kl/n:.3e}")
S = {i: st(F, i, "f-force") for i in ids}; X = {i: st(A, i, "a-force") for i in ids}
N = sum(len(v) for v in S.values()); random.seed(7); dd = []
for _ in range(5000):
    acc = 0.0
    for i in ids:
        n = min(len(S[i]), len(X[i]))
        for _k in range(n):
            j = random.randrange(n); acc += X[i][j] - S[i][j]
    dd.append(acc / N)
dd.sort()
print(f"async - sync mean KL: {(sum(map(sum, X.values())) - sum(map(sum, S.values()))) / N:.3e}, 95% CI [{dd[125]:.3e}, {dd[4875]:.3e}]")
print(f"greedy decode tok/s, mean of five: sync {sum(tps['sync'])/5:.1f}, async + overlap {sum(tps['async'])/5:.1f}")
print(f"twin check (layer 3, async + overlap, MTP windows): {rows} rows, {bad} differ")
