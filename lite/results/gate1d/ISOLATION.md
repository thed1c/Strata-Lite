# Stage d: why every d1 comparison runs with the same execution grouping (2026-10-10, Opus AC)

p1 (603 + 195 tokens), teacher-forced, the 5070, f16 KV:
- **llama.cpp `--n-cpu-moe 20`, plain, against:**
  - strata-gemma4 `--n-cpu-moe 20` (stage b, no pool, no cache): 0/195 bit-identical, first difference at step 0;
  - with `--pool`, and with `--cache-slots 128 --cache-l0 20`: the same 0/195.
- **The two graphs:** identical for a decode step, all 2,644 nodes (op, type, shape, op_params, srcs).
- **llama.cpp alone:** `--n-cpu-moe 20` split at the debug nodes vs plain: **its logits differ.**
- **Same grouping on both sides** (`TDUMP=- TDUMP_N=-2`): strata-gemma4 `--n-cpu-moe 20` 195/195 bit-identical.

So once expert layers sit on the GPU, llama.cpp's plain output depends on execution grouping even for a short prompt. With all experts in RAM it does not (stage b/c plain passes). This is the effect already found past ~4.6k tokens of context (results/gate1b/BISECT.md). d1 therefore runs every prompt with both sides split at the same nodes.
