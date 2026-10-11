# Stage b: where the engine and llama.cpp first differ, and why (2026-10-09, Opus AC)

Setup: the RTX 5070, `--n-cpu-moe 30` (dense layers on the GPU, every expert in RAM), f16 KV, `-c 8192 -b 2048 -ub 512`. The tokens are `p3`'s prompt cut to length L plus 3 continuation tokens, teacher-forced on both sides. Tools: `route-oracle` (llama.cpp b6b003d + an eval callback; `ROUTE_FORCE`) and `strata-gemma4 --mode force`.

| L (tokens) | final prompt ubatch | logits bit-identical (of 3) |
|---:|---:|---:|
| 1530, 1540, 2040, 2050, 2100, 2600, 3100, 4100 | various | 3 |
| 4608 | 512 | 3 |
| 4672, 4700, 4736, 5376 | 64 to 256 | 0 |
| 5120, 6656 (whole ubatches) | 512 | 0 (`stage_b_aligned.out`) |

## What was ruled out at L = 4700
- **KV cells:** identical. The window-cache heads are 0 512 1024 1536 512 1024 1536 512 1024 1536 92 93 on both sides, the global heads match, and llama.cpp's `LLAMA_KV_CACHE_DEBUG` agrees with `G4_KV_DEBUG`.
- **The graph:** identical. For the 92-token ubatch, all 2,643 nodes match on op, type, shape, op_params and flags (`GDUMP`). The decode graph matches too (`*-decode.graph.tsv.gz`): 2,644 nodes, differing only in one node name.
- **CUDA graphs:** `GGML_CUDA_DISABLE_GRAPHS=1` on both sides changes nothing.
- **Determinism:** each side gives the same bytes run after run.

## What it is
- **llama.cpp's own output at L = 4700 changes with execution grouping.** The same graph and inputs give different logits when only the scheduler's grouping changes: the eval callback asks for 330 named per-layer tensors, which splits the graph at those nodes. That is "llama.cpp split vs unsplit: DIFFER".
- **When both sides run with the same grouping, they match bit for bit:**
  - At L = 4700, all 330 intermediate tensors and the logits match.
  - Over the full long prompts with 200 generated tokens (`stage_b_samesplit.out`): p3 (5,402), p4 (6,770) and p5 (6,773), plus p3 at `--n-cpu-moe 20`. Every one gives 200/200 bit-identical steps and 200/200 identical greedy tokens.
- **Conclusion:** past about 4.6k tokens of context, llama.cpp's result depends on execution details outside the graph's arithmetic. Unsplit, the gap between the two engines is of the same kind as llama.cpp's gap with itself.
- **Leads, not followed up:** the CUDA flash-attention path for long KV (stream-k / parallel-block fixup), and `flash_attn_mask_to_KV_max`. The latter reads mask rows up to the next query tile without a bounds check (`ggml-cuda/fattn-common.cuh:691`). Not needed for the port.
