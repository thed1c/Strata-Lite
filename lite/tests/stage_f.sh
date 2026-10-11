#!/bin/bash
# Strata-Lite Gate 1 stage f: the MTP drafter (mtp-gemma-4-26B-A4B-it.gguf, llama.cpp draft-mtp semantics, n_draft 3) on
# top of stage e (64 adaptive slots per layer). Per prompt, teacher-forced on the reference tokens where scored:
#   e-force  : stage e without MTP, vs llama.cpp all-GPU (per-step KL kept for the paired comparison)
#   f-force  : MTP drafts + verify windows, acceptance follows the reference tokens, vs llama.cpp all-GPU
#   e-greedy / f-greedy : the port's own greedy text without and with MTP (must be identical), acceptance, tok/s;
#              f-greedy also runs the per-row twin check on layer 3 (multi-token verify windows included)
# Judged (the maintainer, 2026-10-10) against the all-GPU reference; the all-CPU KL is reported, not gated.
E=$WORK; cd $E || exit 1; O=$E/gate1f; mkdir -p $O
MD=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
g4() { timeout 1800 numactl --cpunodebind=1 --membind=1 build-g4/strata-gemma4 -m $MD/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf \
         --pool --pool-threads 16 --n-cpu-moe 30 -c 8192 -t 16 --cache-slots 64 --profile packs/gemma4-26b/profile-canon.txt --adapt "$@"; }
MTP="--mtp $MD/mtp-gemma-4-26B-A4B-it.gguf --n-draft ${NDRAFT:-3}"
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  T=noise/$id.ncm30.tokens; R=noise/$id.allgpu.logits
  G4_KL_STEPS=$O/$id.e-force.steps g4 --tokens $T --oracle $R --mode force --out $O/$id.e-force > $O/$id.e-force.log 2>&1
  G4_KL_STEPS=$O/$id.f-force.steps g4 $MTP --tokens $T --oracle $R --mode mtp-force --out $O/$id.f-force > $O/$id.f-force.log 2>&1
  g4 --tokens $T --oracle $R --mode greedy --dump-tokens $O/$id.e-greedy.tok --out $O/$id.e-greedy > $O/$id.e-greedy.log 2>&1
  g4 $MTP --tokens $T --oracle $R --mode mtp-greedy --dcheck-layer 3 --dump-tokens $O/$id.f-greedy.tok --out $O/$id.f-greedy > $O/$id.f-greedy.log 2>&1
  for r in e-force f-force e-greedy f-greedy; do echo "$id $r: $(cat $O/$id.$r.json 2>/dev/null || tail -2 $O/$id.$r.log)"; done
done
