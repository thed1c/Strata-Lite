#!/bin/bash
# Strata-Lite Qwen3.6, the speed work ported from Gemma: async swaps (a copy stream from a pinned expert arena, newcomers
# committed when they land) + the overlap protocol (the CPU misses run beside the GPU hits), on top of stage f (MTP,
# n_draft 3) on the 5070, 64 adaptive slots. Per prompt:
#   a-force  : stage f's f-force config (the drafter's own head) + async + overlap, vs llama.cpp all-GPU, per-step KL kept
#              (paired against q35f/<id>.f-force.steps, the same run with synchronous swaps)
#   a-greedy : the proposed default (--mtp-share-head) + async + overlap, greedy, twin check on layer 3; tok/s against
#              q35f/<id>.s-greedy.json (the same with synchronous swaps)
# Summary: tests/q35_speed_summary.py $E/q35f $E/q35a
E=$WORK; cd $E || exit 1; O=$E/q35a; mkdir -p $O
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
q35() { timeout 1800 numactl --cpunodebind=1 --membind=1 build-q35/strata-qwen35 -m $M --n-cpu-moe 40 -c 8192 -t 16 --pool --pool-threads 16 \
          --cache-slots ${S:-64} --profile packs/qwen36/profile-canon.txt --adapt --adapt-half-life 64 --adapt-budget 2 \
          --mtp qwen36/mtp-qwen3.6-35b-a3b-unsloth.gguf --n-draft 3 --async-swaps --overlap "$@"; }
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  T=q35b/$id.oracle.tokens; R=q35b/$id.allgpu.logits
  Q35_KL_STEPS=$O/$id.a-force.steps q35 --tokens $T --oracle $R --mode mtp-force --out $O/$id.a-force > $O/$id.a-force.log 2>&1
  q35 --mtp-share-head --tokens $T --oracle $R --mode mtp-greedy --dcheck-layer 3 --out $O/$id.a-greedy > $O/$id.a-greedy.log 2>&1
  for r in a-force a-greedy; do echo "$id $r: $(cat $O/$id.$r.json 2>/dev/null || tail -2 $O/$id.$r.log)"; done
done
echo SPEED-DONE
