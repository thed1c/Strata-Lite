#!/bin/bash
# Strata-Lite Qwen3.6 stage g, the chunk test (Gemma's tests/stage_g.sh on Qwen). The full engine (adaptive cache, pool,
# MTP with the shared head, copy stream, overlap) reads a prompt in chunks of P tokens and then writes 200 tokens greedily.
# P=512 without lending is llama.cpp's path (each layer's experts offloaded per 512-token ubatch); larger P (and auto)
# lends the slot arena's VRAM to the read and refills it in the background after. Prompts: Gemma's stage g set, tokenized
# for Qwen (q35g/: p5k = p3; p15k = the first quarter + the tail of long-a; long-a = 157k).
# Measures: prompt tok/s and time to first token, the chunk that fits, VRAM lent, the read's compute buffer, decode tok/s
# and hit rate in the 200 tokens after the read, how long the refill takes to land. OOM = this card cannot fund the chunk.
E=$WORK; cd $E || exit 1; O=$E/q35g; mkdir -p $O
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
run() {   # prompt ctx slots P
  local id=$1 ctx=$2 S=$3 P=$4 extra=""
  [ "$P" != 512 ] && extra="--prefill-ubatch $P --lend"
  Q35_STEP_HITS=$O/$id.P$P.steps timeout 3600 numactl --cpunodebind=1 --membind=1 build-q35/strata-qwen35 -m $M --n-cpu-moe 40 -c $ctx -t 16 \
    --pool --pool-threads 16 --cache-slots $S --profile packs/qwen36/profile-canon.txt --adapt --async-swaps --overlap \
    --mtp qwen36/mtp-qwen3.6-35b-a3b-unsloth.gguf --n-draft 3 --mtp-share-head --tokens $O/$id.tokens --mode mtp-greedy --n-gen 200 \
    $extra --out $O/$id.P$P > $O/$id.P$P.log 2>&1
  echo "$id P=$P rc=$? $(tail -1 $O/$id.P$P.log | cut -c1-150)"
}
for P in ${PS:-512 4096 8192 16384 32768 auto}; do
  for spec in ${SPECS:-p5k:20480:64 p15k:20480:64 long-a:160256:40}; do IFS=: read id ctx S <<< "$spec"; run $id $ctx $S $P; done
done
echo STAGE-G-DONE
