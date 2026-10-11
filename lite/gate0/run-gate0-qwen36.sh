#!/bin/bash
# Strata-Lite Gate 0 for Qwen3.6-35B-A3B (HauhauCS Aggressive Q4_K_P): routing logs on the same frozen inputs as Gemma's
# Gate 0 (the maintainer, 2026-10-10: "reuse the same test from Gemma. We will just remove it from the notes before
# publishing"). Card: the RTX 5070 on machine A. Sampling: Qwen's recommended (non-thinking: temp 0.7, top-p 0.8, top-k 20,
# presence 1.5; thinking: temp 1.0, top-p 0.95, top-k 20, presence 1.5). Routes stay on machine A; numbers only in the repo.
E=$WORK; G=$E/gate0; O=$E/gate0-qwen36; mkdir -p $O/runs; LOG=$O/run.log
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid> CUDA_DEVICE_ORDER=PCI_BUS_ID
say(){ echo "$(date +%T) [gate0-qwen36] $*" | tee -a $LOG; }
LIST=${*:-"p1 p2 p1-think p3 p3-think p4 p5 long-d long-a long-b long-c long-e"}
for id in $LIST; do
  [ -s $O/runs/$id.json ] && { say "skip $id (done)"; continue; }
  case $id in *-think) S="--temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5";; *) S="--temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5";; esac
  say "start $id"
  ROUTE_REQ=$G/inputs/$id.json ROUTE_OUT=$O/runs/$id ROUTE_NGEN=${NGEN:-768} timeout 5400 \
    numactl --cpunodebind=1 --membind=1 $E/llama.cpp/build86/bin/route-log -m $M -ngl 99 --n-cpu-moe ${NCM:-30} -fa on \
    -c 200000 -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 -t 16 $S --seed 42 > $O/runs/$id.log 2>&1
  say "end $id rc=$? $(tail -1 $O/runs/$id.log | cut -c1-200)"
done
say "GATE0 QWEN36 RUNS DONE"
