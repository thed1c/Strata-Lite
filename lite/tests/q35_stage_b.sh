#!/bin/bash
# Strata-Lite Qwen3.6 stage b: every expert in RAM (llama.cpp --n-cpu-moe 40, f16 KV, 8k context) on the five Gate 1
# prompts: llama.cpp greedy 200 tokens with logits (route-oracle), then the engine teacher-forced (force) and greedy, plain;
# where plain differs, again with both sides split at the same named nodes (TDUMP=- TDUMP_N=-2).
E=$WORK; cd $E || exit 1; O=$E/q35b; mkdir -p $O
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
export CUDA_VISIBLE_DEVICES=${CARD:-GPU-<uuid>}
NUMA=${NUMA-numactl --cpunodebind=1 --membind=1}
ORA="-m $M -ngl 99 --n-cpu-moe 40 -fa on -c 8192 -ctk f16 -ctv f16 -b 2048 -ub 512 -t 16 --temp 0"
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  P=$O/$id
  [ -s $P.oracle.logits ] || ROUTE_LOGITS=1 ROUTE_REQ=gate0/inputs/$id.json ROUTE_OUT=$P.oracle ROUTE_NGEN=200 timeout 1800 $NUMA \
    llama.cpp/build86/bin/route-oracle $ORA > $P.oracle.log 2>&1
  for mode in force greedy; do
    timeout 1800 $NUMA build-q35/strata-qwen35 -m $M --tokens $P.oracle.tokens --oracle $P.oracle.logits --mode $mode --n-cpu-moe 40 \
      -c 8192 -t 16 --out $P.q35-$mode > $P.q35-$mode.log 2>&1
    echo "$id plain $mode: $(cat $P.q35-$mode.json 2>/dev/null | cut -c1-330 || tail -2 $P.q35-$mode.log)"
  done
  if ! /bin/grep -q '"first_nonidentical_step":-1' $P.q35-force.json 2>/dev/null; then
    export TDUMP=- TDUMP_N=-2
    ROUTE_LOGITS=1 ROUTE_FORCE=$P.oracle.tokens ROUTE_OUT=$P.ss timeout 1800 $NUMA llama.cpp/build86/bin/route-oracle $ORA > $P.ss.log 2>&1
    timeout 1800 $NUMA build-q35/strata-qwen35 -m $M --tokens $P.oracle.tokens --oracle $P.ss.logits --mode force --n-cpu-moe 40 \
      -c 8192 -t 16 --out $P.q35-ss > $P.q35-ss.log 2>&1
    echo "$id same-split force: $(cat $P.q35-ss.json 2>/dev/null | cut -c1-330 || tail -2 $P.q35-ss.log)"
    unset TDUMP TDUMP_N; rm -f $P.ss.logits
  fi
done
