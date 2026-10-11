#!/bin/bash
# Strata-Lite Qwen3.6, q4_0 KV at long context on the 5070: the 157k prompt (q35g/long-a.tokens) read with the auto
# chunk and the arena lent, then 200 greedy tokens with MTP (shared head), async + overlap; at 160k and 262k context, with
# the slot counts given. Against stage g's f16 run (160k, 40 slots: 8k chunks, 147 s to the first token).
E=$WORK; cd $E || exit 1; O=$E/q35kv; mkdir -p $O
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
for spec in ${SPECS:-160256:64 160256:80 262144:64 262144:48}; do IFS=: read ctx S <<< "$spec"
  ( while sleep 2; do nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits; done ) > $O/long-c$ctx-S$S.vram & W=$!
  Q35_STEP_HITS=$O/long-c$ctx-S$S.steps timeout 3600 numactl --cpunodebind=1 --membind=1 build-q35/strata-qwen35 -m $M --n-cpu-moe 40 -c $ctx -t 16 \
    -ctk q4_0 -ctv q4_0 --pool --pool-threads 16 --cache-slots $S --profile packs/qwen36/profile-canon.txt --adapt --async-swaps --overlap \
    --mtp qwen36/mtp-qwen3.6-35b-a3b-unsloth.gguf --n-draft 3 --mtp-share-head --tokens q35g/long-a.tokens --mode mtp-greedy --n-gen 200 \
    --prefill-ubatch auto --lend --out $O/long-c$ctx-S$S > $O/long-c$ctx-S$S.log 2>&1
  rc=$?; kill $W
  echo "c=$ctx S=$S rc=$rc $(grep -m1 'reading' $O/long-c$ctx-S$S.log) | $(python3 -c "import json;d=json.load(open('$O/long-c$ctx-S$S.json'));print('first token %.1fs'%d['prefill_s'],'%.0f tok/s'%(d['prompt']/d['prefill_s']),'decode %.1f tok/s'%d['decode_tok_s'],'hits %.3f'%d['hit_rate'],'read buf %.0f MB'%d['read_buf_mb'],'lent %.1f GB'%d['lent_gb'],'refill %.2fs'%(d['refill_land_ms']/1000),'slots after',d['slots_end'],'acc %d/%d'%(d['accepted'],d['drafted']))" 2>/dev/null || grep -m1 -i 'out of memory\|fail\|abort' $O/long-c$ctx-S$S.log) | peak VRAM $(sort -n $O/long-c$ctx-S$S.vram | tail -1) MiB"
done
echo LONG-DONE
