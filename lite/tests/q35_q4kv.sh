#!/bin/bash
# Strata-Lite Qwen3.6, q4_0 KV (llama.cpp's attn_rot_k/v Hadamard rotations, mirrored). Per prompt, teacher-forced on
# llama.cpp's all-CPU greedy tokens (f16 KV):
#   1. exactness: llama.cpp --n-cpu-moe 40 with q4_0 K/V vs the engine plain (every expert in RAM, no cache / pool),
#      the same KV types: logits must be bit-identical (stage b's bar)
#   2. scale: llama.cpp all-GPU (4070 Ti SUPER + 5070) q4_0 KV vs all-GPU f16 KV, the cost of q4 KV in llama.cpp itself
#   3. the full engine with q4 KV (64 slots, pool, MTP with the drafter's own head, async + overlap, auto chunk + lend,
#      twin check on layer 3) vs all-GPU f16 and vs all-GPU q4; per-step KL kept
E=$WORK; cd $E || exit 1; O=$E/q35kv; mkdir -p $O
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
C5070=GPU-<uuid>; C4070=GPU-<uuid>
NUMA="numactl --cpunodebind=1 --membind=1"; KV="-ctk q4_0 -ctv q4_0"
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  T=q35b/$id.oracle.tokens; P=$O/$id
  [ -s $P.q4cpu.logits ] || CUDA_VISIBLE_DEVICES=$C5070 ROUTE_LOGITS=1 ROUTE_FORCE=$T ROUTE_OUT=$P.q4cpu timeout 1800 $NUMA llama.cpp/build86/bin/route-oracle \
      -m $M -ngl 99 --n-cpu-moe 40 -fa on -c 8192 $KV -b 2048 -ub 512 -t 16 --temp 0 > $P.q4cpu.log 2>&1
  CUDA_VISIBLE_DEVICES=$C5070 timeout 1800 $NUMA build-q35/strata-qwen35 -m $M --tokens $T --oracle $P.q4cpu.logits --mode force --n-cpu-moe 40 \
      -c 8192 -t 16 $KV --out $P.plain > $P.plain.log 2>&1
  echo "$id 1 exactness (plain vs llama.cpp, q4 KV): $(python3 -c "import json;d=json.load(open('$P.plain.json'));print('bit-identical %d/%d'%(d['logits_bit_identical'],d['steps_compared']),'first diff',d['first_nonidentical_step'],'greedy first diff',d['first_token_diff'])" 2>/dev/null || tail -2 $P.plain.log)"
  [ -s $P.allgpu-q4.logits ] || CUDA_VISIBLE_DEVICES=$C4070,$C5070 ROUTE_LOGITS=1 ROUTE_FORCE=$T ROUTE_OUT=$P.allgpu-q4 timeout 1800 $NUMA llama.cpp/build86/bin/route-oracle \
      -m $M -ngl 99 --n-cpu-moe 0 -fa on -c 8192 $KV -b 2048 -ub 512 -t 16 --temp 0 > $P.allgpu-q4.log 2>&1
  build-g4-86/logit-cmp $P.allgpu-q4.logits q35b/$id.allgpu.logits 248320 > $P.q4-vs-f16.cmp 2>&1
  echo "$id 2 llama.cpp all-GPU q4 vs f16: $(tail -1 $P.q4-vs-f16.cmp)"
  for ref in allgpu:q35b/$id.allgpu.logits allgpu-q4:$P.allgpu-q4.logits; do r=${ref%%:*}; R=${ref#*:}
    Q35_KL_STEPS=$P.full-$r.steps CUDA_VISIBLE_DEVICES=$C5070 timeout 1800 $NUMA build-q35/strata-qwen35 -m $M --n-cpu-moe 40 -c 8192 -t 16 $KV \
      --pool --pool-threads 16 --cache-slots 64 --profile packs/qwen36/profile-canon.txt --adapt --adapt-half-life 64 --adapt-budget 2 \
      --mtp qwen36/mtp-qwen3.6-35b-a3b-unsloth.gguf --n-draft 3 --async-swaps --overlap --prefill-ubatch auto --lend --dcheck-layer 3 \
      --tokens $T --oracle $R --mode mtp-force --out $P.full-$r > $P.full-$r.log 2>&1
    echo "$id 3 full engine q4 KV vs $r: $(python3 -c "import json;d=json.load(open('$P.full-$r.json'));print('top1 %d/%d'%(d['argmax_same'],d['steps_compared']),'KL %.3e'%d['kl_mean'],'twins',d['dcheck_rows'],'bad',d['dcheck_bad'])" 2>/dev/null || tail -2 $P.full-$r.log)"
  done
done
echo Q4KV-DONE
