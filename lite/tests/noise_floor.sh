#!/bin/bash
# Strata-Lite: llama.cpp's own placement-to-placement difference (the bar for Gate 1 stages e/f, the maintainer 2026-10-09).
# Per prompt: llama.cpp --n-cpu-moe 30 (every expert in RAM: the stage-b reference) greedy 200 tokens with logits; then
# llama.cpp --n-cpu-moe 14 (16 of 30 expert blocks on the GPU) teacher-forced on the same tokens; logit-cmp -> top-1 + KL.
# Plain execution (no eval-callback splits), f16 KV, -c 8192, the 5070.
E=$WORK; cd $E || exit 1; O=$E/noise; mkdir -p $O
M=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
A=${A:-30}; B=${B:-14}
run() { timeout 1800 numactl --cpunodebind=1 --membind=1 llama.cpp/build/bin/route-oracle -m $M -ngl 99 --n-cpu-moe $1 -fa on -c 8192 -ctk f16 -ctv f16 -b 2048 -ub 512 -t 16 --temp 0 --seed 42; }
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  P=$O/$id
  [ -s $P.ncm$A.logits ] || ROUTE_LOGITS=1 ROUTE_REQ=gate0/inputs/$id.json ROUTE_OUT=$P.ncm$A ROUTE_NGEN=200 run $A > $P.ncm$A.log 2>&1
  ROUTE_LOGITS=1 ROUTE_FORCE=$P.ncm$A.tokens ROUTE_OUT=$P.ncm$B run $B > $P.ncm$B.log 2>&1
  echo "$id ncm$A-vs-ncm$B $(build-g4/logit-cmp $P.ncm$A.logits $P.ncm$B.logits 262144)"
  rm -f $P.ncm$B.logits
done
