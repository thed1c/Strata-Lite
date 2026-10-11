#!/bin/bash
# Strata-Lite Gate 0: routing logs on real inputs. Card: the 4070 Ti SUPER on machine A;
# the 5070 stays free for the baseline sweep. Order: short inputs first, the ~150k-token reads last.
E=$WORK; G=$E/gate0; LOG=$G/run.log
M=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid> CUDA_DEVICE_ORDER=PCI_BUS_ID
say(){ echo "$(date +%T) [gate0] $*" | tee -a $LOG; }
LIST=${*:-"p1 p2 p1-think p3 p3-think p4 p5 long-d long-a long-b long-c long-e"}
for id in $LIST; do
  [ -s $G/runs/$id.json ] && { say "skip $id (done)"; continue; }
  say "start $id"
  ROUTE_REQ=$G/inputs/$id.json ROUTE_OUT=$G/runs/$id ROUTE_NGEN=${NGEN:-768} timeout 5400 \
    numactl --cpunodebind=0 --membind=0 $E/llama.cpp/build/bin/route-log -m $M -ngl 99 --n-cpu-moe ${NCM:-10} -fa on \
    -c 200000 -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 -t 18 --temp 0.6 --top-k 64 --top-p 0.9 --min-p 0.05 --seed 42 \
    > $G/runs/$id.log 2>&1
  say "end $id rc=$? $(tail -1 $G/runs/$id.log)"
done
say "GATE0 RUNS DONE"
