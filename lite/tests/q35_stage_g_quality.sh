#!/bin/bash
# Strata-Lite Qwen3.6 stage g, quality: the five prompts read in one auto-sized chunk with the arena lent (the prompt is
# shorter than any chunk), then MTP verify windows teacher-forced on llama.cpp's all-CPU greedy tokens and scored against
# llama.cpp all-GPU, twin check on layer 3. Config otherwise = tests/q35_speed.sh's a-force (drafter's own head, async +
# overlap, 64 slots), so the per-step KL pairs with q35a/<id>.a-force.steps (reads in llama.cpp's 512-token ubatches).
E=$WORK; cd $E || exit 1; O=$E/q35gq; mkdir -p $O
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  Q35_KL_STEPS=$O/$id.g-force.steps timeout 1800 numactl --cpunodebind=1 --membind=1 build-q35/strata-qwen35 -m $M --n-cpu-moe 40 -c 8192 -t 16 \
    --pool --pool-threads 16 --cache-slots 64 --profile packs/qwen36/profile-canon.txt --adapt --adapt-half-life 64 --adapt-budget 2 \
    --mtp qwen36/mtp-qwen3.6-35b-a3b-unsloth.gguf --n-draft 3 --async-swaps --overlap --prefill-ubatch auto --lend --dcheck-layer 3 \
    --tokens q35b/$id.oracle.tokens --oracle q35b/$id.allgpu.logits --mode mtp-force --out $O/$id.g-force > $O/$id.g-force.log 2>&1
  echo "$id g-force: $(cat $O/$id.g-force.json 2>/dev/null | cut -c1-200 || tail -2 $O/$id.g-force.log)"
done
python3 - <<PY
import json, random
ids = "${IDS:-p1 p2 p3 p4 p5}".split()
st = lambda p: [float(l.split()[0]) for l in open(p)]
A = {i: st(f"q35a/{i}.a-force.steps") for i in ids}; G = {i: st(f"$O/{i}.g-force.steps") for i in ids}
tot = {"512": [0, 0, 0.0], "chunked": [0, 0, 0.0]}; rows = bad = 0
for i in ids:
    a = json.load(open(f"q35a/{i}.a-force.json")); g = json.load(open(f"$O/{i}.g-force.json"))
    for k, d in (("512", a), ("chunked", g)):
        tot[k][0] += d["steps_compared"]; tot[k][1] += d["argmax_same"]; tot[k][2] += d["kl_mean"] * d["steps_compared"]
    rows += g["dcheck_rows"]; bad += g["dcheck_bad"]
    print(f"{i}: reads in 512 top1 {a['argmax_same']}/{a['steps_compared']} KL {a['kl_mean']:.2e} | one chunk ({g['prompt']} tokens, lent {g['lent_gb']:.1f} GB) top1 {g['argmax_same']}/{g['steps_compared']} KL {g['kl_mean']:.2e}, prefill {a['prefill_s']:.2f} -> {g['prefill_s']:.2f} s")
for k, (n, t1, kl) in tot.items(): print(f"ALL reads {k} vs all-GPU: top-1 {t1}/{n}, mean KL {kl/n:.3e}")
N = sum(len(v) for v in A.values()); random.seed(7); dd = []
for _ in range(5000):
    s = 0.0
    for i in ids:
        n = min(len(A[i]), len(G[i]))
        for _k in range(n): j = random.randrange(n); s += G[i][j] - A[i][j]
    dd.append(s / N)
dd.sort()
print(f"chunked - 512 mean KL: {(sum(map(sum, G.values())) - sum(map(sum, A.values()))) / N:.3e}, 95% CI [{dd[125]:.3e}, {dd[4875]:.3e}]")
print(f"twin check (layer 3, chunked reads with lending, MTP windows): {rows} rows, {bad} differ")
PY
echo QUALITY-DONE
