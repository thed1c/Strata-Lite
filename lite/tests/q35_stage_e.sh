#!/bin/bash
# Strata-Lite Qwen3.6 stage e: adaptation (decayed counts, half-life HL, <= BUDGET swaps per layer per step, refill after
# a read) on 64 slots, twin check on layer 3, teacher-forced through llama.cpp's all-CPU greedy tokens. Judged against
# llama.cpp all-GPU (4070 Ti SUPER + 5070, every expert on the GPUs; the maintainer's 2026-10-10 rule for Gemma: "e and f are
# judged against the all-GPU reference"); llama.cpp's own all-CPU vs all-GPU pair is written beside it for scale.
E=$WORK; cd $E || exit 1; O=$E/q35b
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
C5070=GPU-<uuid>; C4070=GPU-<uuid>
NUMA="numactl --cpunodebind=1 --membind=1"
HL=${HL:-64}; BUDGET=${BUDGET:-2}; S=${S:-64}; TAG=${TAG:-e}
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  P=$O/$id
  if [ ! -s $P.allgpu.logits ]; then
    CUDA_VISIBLE_DEVICES=$C4070,$C5070 ROUTE_LOGITS=1 ROUTE_FORCE=$P.oracle.tokens ROUTE_OUT=$P.allgpu timeout 1800 $NUMA llama.cpp/build86/bin/route-oracle \
      -m $M -ngl 99 --n-cpu-moe 0 -fa on -c 8192 -ctk f16 -ctv f16 -b 2048 -ub 512 -t 16 --temp 0 > $P.allgpu.log 2>&1
    build-g4-86/logit-cmp $P.allgpu.logits $P.oracle.logits 248320 > $P.pair.cmp 2>&1
  fi
  echo "$id llama.cpp all-CPU vs all-GPU: $(tail -1 $P.pair.cmp)"
  CUDA_VISIBLE_DEVICES=$C5070 timeout 1800 $NUMA build-q35/strata-qwen35 -m $M --tokens $P.oracle.tokens --oracle $P.allgpu.logits --mode force \
    --n-cpu-moe 40 -c 8192 -t 16 --pool --pool-threads 16 --cache-slots $S --profile packs/qwen36/profile-canon.txt --adapt \
    --adapt-half-life $HL --adapt-budget $BUDGET --dcheck-layer 3 --out $P.q35$TAG > $P.q35$TAG.log 2>&1
  echo "$id stage e (hl $HL, budget $BUDGET, $S slots) vs all-GPU: $(python3 -c "import json;d=json.load(open('$P.q35$TAG.json'));print('top1 %d/%d'%(d['argmax_same'],d['steps_compared']),'KL %.2e'%d['kl_mean'],'twins',d['dcheck_rows'],'bad',d['dcheck_bad'],'hits %.3f'%d['hit_rate'],'swaps',d['swaps'],'refill',d['refill_swaps'],'%.0f MB'%d['swap_mb'],'decode %.1fs'%d['decode_s'])" 2>/dev/null || tail -2 $P.q35$TAG.log)"
done
