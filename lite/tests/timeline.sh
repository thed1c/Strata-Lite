#!/bin/bash
# Strata-Lite: the per-token timeline of the full engine (stage f: 64 adaptive slots + pool + MTP) on cached prompts.
# Host wall clock per component (engine timers), summed over the decode, per verify window and as a share of the decode.
E=$WORK; cd $E || exit 1; O=$E/timeline; mkdir -p $O
MD=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
TAG=${TAG:-base}
for id in ${IDS:-p3 p1}; do
  timeout 1800 numactl --cpunodebind=1 --membind=1 ${PROFILER} build-g4/strata-gemma4 -m $MD/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf \
    --pool --pool-threads 16 --n-cpu-moe 30 -c 8192 -t 16 --cache-slots 64 --profile packs/gemma4-26b/profile-canon.txt --adapt \
    --mtp $MD/mtp-gemma-4-26B-A4B-it.gguf --tokens noise/$id.ncm30.tokens --mode mtp-greedy $EXTRA --out $O/$id.$TAG > $O/$id.$TAG.log 2>&1
  python3 - $O/$id.$TAG.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); t = d["timeline"]; w = d["mtp_windows"]; tot = d["decode_s"] * 1000
sw, rf = d["swap_ms"], d["refill_ms"]
print("%s: %d tokens, %d windows, %.1f tok/s, hits %.3f, accepted %d/%d" % (sys.argv[1].split('/')[-1], d["tokens"], w, d["decode_tok_s"], d["hit_rate"], d["accepted"], d["drafted"]))
ov = d.get("ov_layers", 0) > 0
jw = d.get("join_wait_ms", 0.0)
parts = [("drafts (%d drafter graphs)" % t["drafts"], t["draft_ms"]),
         ("verify: graph build + alloc + inputs", t["build_ms"])]
if ov:
    parts += [("verify: compute, host blocked in join (pool slower than GPU hits)", jw),
              ("verify: compute, the rest (GPU work, transfers, syncs, start ops)", t["compute_ms"] - jw)]
else:
    parts += [("verify: compute, CPU pool on the misses (GPU idle)", t["pool_ms"]),
              ("verify: compute, the rest (GPU work, transfers, syncs)", t["compute_ms"] - t["pool_ms"])]
parts += [("verify: readbacks (logits, hidden rows, routing)", t["readout_ms"]),
          ("adapt: score updates", t["adapt_ms"] - sw - rf),
          ("adapt: swap copies (issue cost if async)", sw),
          ("adapt: refill after the read", rf)]
if ov:
    print("  overlap: %d layer-steps, %d with no miss (%.0f%%); pool busy %.1f ms (concurrent with the GPU)" % (d["ov_layers"], d["ov_missfree"], 100.0 * d["ov_missfree"] / d["ov_layers"], t["pool_ms"]))
for n, v in parts: print("  %-56s %8.1f ms %6.2f ms/window %5.1f%%" % (n, v, v / w, 100 * v / tot))
print("  %-56s %8.1f / %.1f ms; %.1f graph splits per verify ubatch" % ("accounted / decode wall", sum(v for _, v in parts), tot, t["splits"] / max(1, t["ubatches"])))
PY
done
