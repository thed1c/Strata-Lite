#!/bin/bash
# Strata-Lite Gate 1 stage c: the RAM-resident experts of small ubatches (decode, prompt tails < 32 tokens) through
# Strata's CPU expert pool (strata-gemma4 --pool) must match llama.cpp bit for bit, like stage b.
# Short prompts: plain, against llama.cpp --n-cpu-moe 30 (noise/<id>.ncm30.logits from noise_floor.sh).
# Long prompts: both sides split at the same named nodes (TDUMP=- TDUMP_N=-2; see results/gate1b/BISECT.md).
E=$WORK; cd $E || exit 1; O=$E/gate1c; mkdir -p $O
M=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
NCM=${NCM:-30}
g4() { timeout 1800 numactl --cpunodebind=1 --membind=1 build-g4/strata-gemma4 -m $M --pool --pool-threads 16 --n-cpu-moe $NCM -c 8192 -t 16 "$@"; }
for id in ${SHORT-p1 p2}; do
  for mode in force greedy; do
    g4 --tokens noise/$id.ncm30.tokens --oracle noise/$id.ncm30.logits --mode $mode --out $O/$id.$mode > $O/$id.$mode.log 2>&1
    echo "$id plain $mode: $(cat $O/$id.$mode.json 2>/dev/null || tail -2 $O/$id.$mode.log)"
  done
done
export TDUMP=- TDUMP_N=-2
for id in ${LONG-p3 p4 p5}; do
  P=$O/$id.ss
  ROUTE_LOGITS=1 ROUTE_FORCE=$E/gate1b/samesplit/$id.ncm30.prompt.tokens ROUTE_OUT=$P.oracle ROUTE_NGEN=200 timeout 1800 numactl --cpunodebind=1 --membind=1 \
    llama.cpp/build/bin/route-oracle -m $M -ngl 99 --n-cpu-moe $NCM -fa on -c 8192 -ctk f16 -ctv f16 -b 2048 -ub 512 -t 16 --temp 0 > $P.oracle.log 2>&1
  for mode in force greedy; do
    g4 --tokens $P.oracle.tokens --oracle $P.oracle.logits --mode $mode --out $P.$mode > $P.$mode.log 2>&1
    echo "$id same-split $mode: $(cat $P.$mode.json 2>/dev/null || tail -2 $P.$mode.log)"
  done
  rm -f $P.oracle.logits
done
