#!/bin/bash
# Strata-Lite Gate 1 stage b: the gemma4 engine against llama.cpp, same placement, f16 KV, bit for bit.
# For each prompt: llama.cpp (route-oracle, greedy, logits per step) then strata-gemma4 teacher-forced (every step's
# logits compared byte for byte) and greedy (its own argmax tokens compared). Card: the 5070. Usage: stage_b.sh [NCM] [prompts...]
E=$WORK; cd $E || exit 1; O=$E/gate1b; mkdir -p $O
M=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
NCM=${1:-30}; shift
for id in ${*:-p1 p2 p3 p4 p5}; do
  P=$O/$id.ncm$NCM
  ROUTE_LOGITS=1 ROUTE_REQ=gate0/inputs/$id.json ROUTE_OUT=$P.oracle ROUTE_NGEN=200 timeout 1800 numactl --cpunodebind=1 --membind=1 \
    llama.cpp/build/bin/route-oracle -m $M -ngl 99 --n-cpu-moe $NCM -fa on -c 8192 -ctk f16 -ctv f16 -b 2048 -ub 512 -t 16 --temp 0 --seed 42 > $P.oracle.log 2>&1
  for mode in force greedy; do
    timeout 1800 numactl --cpunodebind=1 --membind=1 build-g4/strata-gemma4 -m $M --tokens $P.oracle.tokens --oracle $P.oracle.logits \
      --mode $mode --n-cpu-moe $NCM -c 8192 -t 16 --out $P.g4-$mode > $P.g4-$mode.log 2>&1
    echo "$id ncm=$NCM $mode: $(cat $P.g4-$mode.json 2>/dev/null || tail -2 $P.g4-$mode.log)"
  done
  rm -f $P.oracle.logits     # 200 MB each; the verdicts are in the json
done
