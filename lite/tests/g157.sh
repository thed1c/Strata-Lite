#!/bin/bash
# Strata-Lite: the 157k prompt re-measure before Gate 2 (the maintainer, 2026-10-10: top-1 and KL with 500 generated tokens).
# Reference: llama.cpp all-GPU (4070 Ti SUPER + 5070), q4 KV, greedy 500 (g157/allgpu-q4.*). The port (5070, q4 KV,
# 64 adaptive slots, chunked read with lending, copy stream, overlap, MTP) teacher-forced through the 500 steps with
# the twin check on layer 3; for scale, llama.cpp's own all-CPU-experts run on the 5070 at q4 KV against the same reference.
E=$WORK; cd $E || exit 1; O=$E/g157
MD=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
S=${S:-64}
G4_KL_STEPS=$O/port.steps timeout 7200 numactl --cpunodebind=1 --membind=1 build-g4/strata-gemma4 -m $MD/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf \
  --pool --pool-threads 16 --n-cpu-moe 30 -c 160256 -t 16 -ctk q4_0 -ctv q4_0 --cache-slots $S --profile packs/gemma4-26b/profile-canon.txt --adapt \
  --async-swaps --overlap --prefill-ubatch 4096 --lend --mtp $MD/mtp-gemma-4-26B-A4B-it.gguf --dcheck-layer 3 \
  --tokens $O/allgpu-q4.tokens --oracle $O/allgpu-q4.logits --mode mtp-force --out $O/port > $O/port.log 2>&1
echo "port rc=$?: $(cat $O/port.json 2>/dev/null | cut -c1-200 || tail -2 $O/port.log)"
ROUTE_LOGITS=1 ROUTE_FORCE=$O/allgpu-q4.tokens ROUTE_OUT=$O/ncm30-q4 timeout 7200 numactl --cpunodebind=1 --membind=1 llama.cpp/build/bin/route-oracle \
  -m $MD/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf -ngl 99 --n-cpu-moe 30 -fa on -c 160256 -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 -t 16 --temp 0 > $O/ncm30-q4.log 2>&1
echo "llama.cpp all-CPU-experts q4 vs all-GPU q4: $(build-g4/logit-cmp $O/allgpu-q4.logits $O/ncm30-q4.logits 262144 1000000 $O/ncm30.steps)"
