#!/bin/bash
# Strata-Lite Qwen3.6 stage f: the MTP drafter (unsloth's MTP head extracted by strata/tools/qwen35_mtp_extract.py,
# llama.cpp draft-mtp semantics, n_draft 3, n_rs_seq 3 recurrent snapshots for the rollback) on top of stage e (64
# adaptive slots, half-life 64, budget 2) on the 5070. Per prompt, teacher-forced on llama.cpp's all-CPU greedy tokens:
#   e-force  : stage e without MTP, vs llama.cpp all-GPU (per-step KL kept for the paired comparison)
#   f-force  : MTP drafts + verify windows, acceptance follows the reference tokens, vs llama.cpp all-GPU
#   e-greedy / f-greedy : the port's own greedy text without and with MTP, acceptance, tok/s; f-greedy also runs the
#              per-row twin check on layer 3 (multi-token verify windows included)
#   s-greedy : f-greedy with the target's token_embd / output instead of the drafter file's own (the VRAM option)
# Summary: tests/stage_f_summary.py $E/q35f (the same file layout as Gemma's stage f).
E=$WORK; cd $E || exit 1; O=$E/q35f; mkdir -p $O
M=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
q35() { timeout 1800 numactl --cpunodebind=1 --membind=1 build-q35/strata-qwen35 -m $M --n-cpu-moe 40 -c 8192 -t 16 --pool --pool-threads 16 \
          --cache-slots 64 --profile packs/qwen36/profile-canon.txt --adapt --adapt-half-life 64 --adapt-budget 2 "$@"; }
MTP="--mtp qwen36/mtp-qwen3.6-35b-a3b-unsloth.gguf --n-draft ${NDRAFT:-3}"
for id in ${IDS:-p1 p2 p3 p4 p5}; do
  T=q35b/$id.oracle.tokens; R=q35b/$id.allgpu.logits
  Q35_KL_STEPS=$O/$id.e-force.steps q35 --tokens $T --oracle $R --mode force --out $O/$id.e-force > $O/$id.e-force.log 2>&1
  Q35_KL_STEPS=$O/$id.f-force.steps q35 $MTP --tokens $T --oracle $R --mode mtp-force --out $O/$id.f-force > $O/$id.f-force.log 2>&1
  q35 --tokens $T --oracle $R --mode greedy --dump-tokens $O/$id.e-greedy.tok --out $O/$id.e-greedy > $O/$id.e-greedy.log 2>&1
  q35 $MTP --tokens $T --oracle $R --mode mtp-greedy --dcheck-layer 3 --dump-tokens $O/$id.f-greedy.tok --out $O/$id.f-greedy > $O/$id.f-greedy.log 2>&1
  q35 $MTP --mtp-share-head --tokens $T --oracle $R --mode mtp-greedy --dump-tokens $O/$id.s-greedy.tok --out $O/$id.s-greedy > $O/$id.s-greedy.log 2>&1
  for r in e-force f-force e-greedy f-greedy s-greedy; do echo "$id $r: $(cat $O/$id.$r.json 2>/dev/null || tail -2 $O/$id.$r.log)"; done
done
echo STAGE-F-DONE
