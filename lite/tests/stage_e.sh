#!/bin/bash
# Strata-Lite Gate 1 stage e: adaptation (decayed counts, half-life HL tokens, <= BUDGET swaps/layer/step, refill to
# top-S after a prompt read). Bar (the maintainer, 2026-10-10): mean KL <= 1.6e-3 and top-1 >= 98.5 % over the five
# prompts' 993 teacher-forced steps against llama.cpp (gate reference: all experts in RAM, noise/<id>.ncm30.*; also
# reported against --n-cpu-moe 14), plus the per-row twin check (every merged row = its GPU or CPU twin, bit for bit).
E=$WORK; cd $E || exit 1; O=$E/gate1e; mkdir -p $O
M=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
PROF=$E/packs/gemma4-26b/profile-canon.txt
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
S=${CACHE_SLOTS:-64}; HL=${HL:-32}; B=${BUDGET:-2}; TAG=s${S}hl${HL}b${B}
g4() { timeout 1800 numactl --cpunodebind=1 --membind=1 build-g4/strata-gemma4 -m $M --pool --pool-threads 16 --n-cpu-moe 30 -c 8192 -t 16 \
         --cache-slots $S --profile $PROF --adapt --adapt-half-life $HL --adapt-budget $B "$@"; }
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  P=$O/$id.$TAG
  [ -n "$NOCHECK" ] || { g4 --dcheck-layer ${DCHECK:-3} --tokens noise/$id.ncm30.tokens --oracle noise/$id.ncm30.logits --mode force --out $P.check > $P.check.log 2>&1
    echo "e $id $TAG check: $(cat $P.check.json 2>/dev/null || tail -2 $P.check.log)"; }
  for ref in ${REFS:-ncm30 ncm14}; do
    g4 --tokens noise/$id.ncm30.tokens --oracle noise/$id.$ref.logits --mode force --out $P.vs$ref > $P.vs$ref.log 2>&1
    echo "e $id $TAG vs$ref: $(cat $P.vs$ref.json 2>/dev/null || tail -2 $P.vs$ref.log)"
  done
done
