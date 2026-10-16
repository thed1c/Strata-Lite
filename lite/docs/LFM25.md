# LFM2.5-24B-A2B

Strata-Lite runs Liquid AI's LFM2.5-24B-A2B (`lfm2moe`) on the same engine as Qwen3.6: experts in system RAM, an
adaptive expert cache on the card, the CPU expert pool, async swaps, CPU/GPU overlap, lent prompt reads and the KV
pager. The model weights are unchanged. The tokenizer and chat template come from the GGUF through llama.cpp.

The port is Codex's (validation below); the speed numbers in the README are measured with the standard benchmark
(Wikipedia text, the same scripts and cards as Gemma 4 and Qwen3.6).

## Checkpoints the setup offers

| | File | Size | SHA256 |
|---|---|---|---|
| Default (uncensored) | [AtomicChat LFM2.5-24B-A2B-Uncensored-AD-Q4_K_M](https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF/blob/1acb8f6e/LFM2.5-24B-A2B-Uncensored-AD-Q4_K_M.gguf), rev `1acb8f6e` | 15.4 GB | `17a5de7f…` |
| Official | [LiquidAI LFM2.5-24B-A2B-QAD-Q4_0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/blob/0912d33141f77e10705345a6732bfd83f0252fdc/LFM2.5-24B-A2B-QAD-Q4_0.gguf), rev `0912d331` | 13.6 GB | `d1c03f73…` |

The full hashes are in `setup_lite.py`; the setup checks them after the download.

## The architecture

Of 40 blocks, 30 are causal short convolutions and 10 are GQA attention with Q/K RMSNorm and NeoX RoPE. The first two
FFNs are dense; the other 38 have 64 experts, 4 chosen per token, 1536 wide. Selection adds an expert bias to sigmoid
scores; the mixture weights are the unbiased sigmoid values, normalized as llama.cpp does. No shared expert. The tied
Q6_K token embedding is also the output projection. The short-convolution state, its snapshots and the prompt-cache
checkpoints are kept apart from the attention KV, as Qwen3.6's delta-net state is. The graph mirrors llama.cpp
`b6b003d` [`src/models/lfm2.cpp`](https://github.com/ggml-org/llama.cpp/blob/b6b003d2cb29647d968302eb2db8da6f66303b3e/src/models/lfm2.cpp).

## Run

`strata-lite-server` detects LFM from the GGUF and starts the hybrid engine:

```bash
strata-lite-server -m LFM2.5-24B-A2B-Uncensored-AD-Q4_K_M.gguf
```

Defaults for LFM: context 131072 (the checkpoint's limit; more is refused), f16 KV, the KV pager (32K cells first,
16K steps), every MoE block on the CPU side with the adaptive cache, auto slots and chunks. Sampling defaults are the
model card's (temperature 0.1, top-k 50, repetition penalty 1.05); flags and request fields still win. There is no MTP
head and no image projector: `--mtp` and `--mmproj` are refused.

`lite/profiles/lfm25-qad-q4_0.txt` is an optional starting order for the cache, built only from the four synthetic
inputs in `lite/tests/lfm25/` (`heldout.json` left out); the cache keeps learning from each conversation without it.

### Rank-one output adapter (optional)

`--lora FILE --lora-scale S` (or `--lora-scaled FILE:S`) loads an F32 rank-one LoRA on the mixer output and expert
down projections, fixed for the life of the server and shown in `/props`. A request that asks for another adapter gets
HTTP 400, so prompt caches never mix adapters. Code: `src/lite/lfm_lora.*`.

## The second llama.cpp hook

LFM's chat parser in llama.cpp `b6b003d` turns a request with `tool_choice: auto` plus a JSON response format into a
mandatory tool call. `lite/llama-hooks/lfm-json-tools.cmake` writes a patched copy of `common/parsers/lfm2.cpp` into
the build folder (the fetched llama.cpp stays untouched) so the answer may be either a tool call or the requested
JSON. The build stops if the pinned source no longer matches. It sits beside the sampler hook; see [BUILD.md](BUILD.md).

## Validation

Same method as Gemma 4 and Qwen3.6 ([CORRECTNESS.md](CORRECTNESS.md)), on the official QAD file:

- Every expert on the CPU pool: 960/960 logit vectors bit-identical to llama.cpp with the experts on the CPU.
- Cached rows vs their CPU twins: 3,820 rows (fixed cache) and 3,820 more (adaptive), 0 differ.
- Adaptive cache vs all-GPU (a 16 GB card): top-1 948/960, mean KL 0.0019; llama.cpp's static offload on the same
  prompts 947/960, KL 0.0021.
- Prompt-cache checkpoints, reset, KV page growth and replay after paging: pass.
- Qwen3.6 with the shared changes: bit for bit against the released engine; Gemma 4 and Qwen3.6 API parity 63/63.
