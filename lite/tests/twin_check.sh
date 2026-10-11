#!/bin/bash
# Strata-Lite: the row-twin check after a concurrency change (the maintainer, 2026-10-10: run it on the five prompts before
# writing any speed number down). Full engine (64 adaptive slots + pool + MTP, teacher-forced via mtp-force), check layer
# DCHECK: every merged row must equal its GPU or CPU twin. EXTRA carries the change under test (e.g. --async-swaps).
E=$WORK; cd $E || exit 1; O=$E/twin; mkdir -p $O
MD=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
TAG=${TAG:-check}; rows=0; bad=0
for id in p1 p2 p3 p4 p5; do
  timeout 1800 numactl --cpunodebind=1 --membind=1 ${BUILD:-build-g4}/strata-gemma4 -m $MD/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf \
    --pool --pool-threads 16 --n-cpu-moe 30 -c 8192 -t 16 --cache-slots 64 --profile packs/gemma4-26b/profile-canon.txt --adapt \
    --mtp $MD/mtp-gemma-4-26B-A4B-it.gguf --tokens noise/$id.ncm30.tokens --oracle noise/$id.${OSUF:-allgpu}.logits --mode mtp-force \
    --dcheck-layer ${DCHECK:-3} $EXTRA --out $O/$id.$TAG > $O/$id.$TAG.log 2>&1
  python3 -c "import json; d=json.load(open('$O/$id.$TAG.json')); print('$id', 'rows', d['dcheck_rows'], 'gpu', d['dcheck_gpu_rows'], 'bad', d['dcheck_bad'], 'KL vs all-GPU %.2e' % d['kl_mean'], 'top1 %d/%d' % (d['argmax_same'], d['steps_compared']), 'commits', d.get('commits'))" \
    || { echo "$id FAILED: $(tail -2 $O/$id.$TAG.log)"; }
done
