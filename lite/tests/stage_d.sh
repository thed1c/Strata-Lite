#!/bin/bash
# Strata-Lite Gate 1 stage d: the GPU expert cache (static set) in front of Strata's pool.
# d1 (bit for bit): layers 20-29 wholly resident in slots, the rest in the pool, against llama.cpp --n-cpu-moe 20,
#     teacher-forced on the five prompts' reference tokens, both sides split at the same nodes (TDUMP=- TDUMP_N=-2).
# d2 (per expert): CACHE_SLOTS per layer from the starting profile; a check layer computes every routed expert both ways
#     in the same step and every merged row must equal its GPU or CPU twin; plus top-1/KL against llama.cpp ncm30.
E=$WORK; cd $E || exit 1; O=$E/gate1d; mkdir -p $O
M=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
PROF=$E/packs/gemma4-26b/profile-canon.txt
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
g4() { timeout 1800 numactl --cpunodebind=1 --membind=1 build-g4/strata-gemma4 -m $M --pool --pool-threads 16 --n-cpu-moe 30 -c 8192 -t 16 "$@"; }
oracle() { timeout 1800 numactl --cpunodebind=1 --membind=1 llama.cpp/build/bin/route-oracle -m $M -ngl 99 --n-cpu-moe $1 -fa on -c 8192 -ctk f16 -ctv f16 -b 2048 -ub 512 -t 16 --temp 0; }
short="p1 p2"; long="p3 p4 p5"
if [ "${PART:-d1}" = d1 ]; then
  # with GPU-resident experts llama.cpp's own plain output depends on execution grouping even on short prompts
  # (results/gate1d/ISOLATION.md), so every d1 comparison runs both sides split at the same nodes
  export TDUMP=- TDUMP_N=-2
  for id in ${IDS-$short $long}; do
    P=$O/$id.d1ss
    ROUTE_LOGITS=1 ROUTE_FORCE=noise/$id.ncm30.tokens ROUTE_OUT=$P.oracle oracle 20 > $P.oracle.log 2>&1
    g4 --cache-slots 128 --cache-l0 20 --tokens noise/$id.ncm30.tokens --oracle $P.oracle.logits --mode force --out $P.force > $P.force.log 2>&1
    echo "d1 $id same-split force: $(cat $P.force.json 2>/dev/null || tail -2 $P.force.log)"
    rm -f $P.oracle.logits
  done
else
  S=${CACHE_SLOTS:-64}
  for id in $short $long; do
    P=$O/$id.d2s$S
    g4 --cache-slots $S --profile $PROF --dcheck-layer ${DCHECK:-3} --tokens noise/$id.ncm30.tokens --oracle noise/$id.ncm30.logits --mode force --out $P.check > $P.check.log 2>&1
    g4 --cache-slots $S --profile $PROF --tokens noise/$id.ncm30.tokens --oracle noise/$id.ncm30.logits --mode force --out $P.force > $P.force.log 2>&1
    echo "d2 $id slots=$S check: $(cat $P.check.json 2>/dev/null || tail -2 $P.check.log)"
    echo "d2 $id slots=$S force: $(cat $P.force.json 2>/dev/null || tail -2 $P.force.log)"
  done
fi
