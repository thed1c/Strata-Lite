#!/bin/bash
# Strata-Lite stage g, the chunk test (NEXT-ORDER.md §2). The full engine (adaptive cache, pool, MTP, copy stream,
# overlap) reads a real prompt in chunks of P tokens and then writes 200 tokens greedily. P=512 without lending is
# llama.cpp's path (each layer's experts offloaded per 512-token ubatch); larger P lends the slot arena's VRAM to the read
# (compute buffers + the window cache, which must hold n_swa + P cells) and refills the arena in the background after.
# Measures: prompt tok/s and time to first token, the chunk that fits, VRAM lent, decode tok/s and hit rate in the first
# 200 tokens after the read, and how long the refill takes to land. An OOM means this card cannot fund that chunk.
E=$WORK; cd $E || exit 1; O=$E/gate1g; mkdir -p $O
MD=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
run() {   # prompt ctx slots P
  local id=$1 ctx=$2 S=$3 P=$4 extra=""
  [ "$P" != 512 ] && extra="--prefill-ubatch $P --lend"
  G4_STEP_HITS=$O/$id.P$P.steps timeout 3600 numactl --cpunodebind=1 --membind=1 build-g4/strata-gemma4 -m $MD/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf \
    --pool --pool-threads 16 --n-cpu-moe 30 -c $ctx -t 16 --cache-slots $S --profile packs/gemma4-26b/profile-canon.txt --adapt --async-swaps --overlap \
    --mtp $MD/mtp-gemma-4-26B-A4B-it.gguf --tokens $O/$id.tokens --mode mtp-greedy --n-gen 200 $extra --out $O/$id.P$P > $O/$id.P$P.log 2>&1
  echo "$id P=$P rc=$? $(tail -1 $O/$id.P$P.log | cut -c1-150)"
}
for P in ${PS:-512 4096 8192 16384 32768}; do
  for spec in ${SPECS:-p5k:20480:64 p15k:20480:64 long-a:160256:40}; do IFS=: read id ctx S <<< "$spec"; run $id $ctx $S $P; done
done
