# Qwen3.6-35B-A3B in Strata-Lite

Second architecture after Gemma 4, built and checked the same way: every stage against llama.cpp on the same card before the
next began. Status: **the engine passes every stage, the server passes API parity (63/63), and the grid is in the README.**

## The model
llama.cpp calls the architecture `qwen35moe` (the Qwen3.5 / 3.6 MoE family).

| | Qwen3.6-35B-A3B | Gemma 4 26B-A4B |
|---|---|---|
| Blocks | 40: 30 Gated DeltaNet + 10 gated attention (every 4th) | 30, sliding-window and full attention |
| Experts | 256 per block, 8 routed + 1 shared (gated), width 512 | 128 per block, 8 routed, width 704, plus a small dense FFN |
| Expert activation | SwiGLU, separate gate / up / down tensors | GeGLU |
| Hidden size, heads | 2048; attention 16 Q / 2 KV heads of 256, IMROPE (sections 11, 11, 10, 0, base 1e7) | see `lite/docs/HOW-IT-WORKS.md` |
| Recurrent state | delta-net: conv 4, state 128, 16 groups, 32 V heads, inner 4096 | none |
| Experts' weight (Q4_K_P) | about 21.7 GB, 2.12 MB per expert | 15.1 GB |
| Context | 262,144 native | 262,144 |
| MTP | one head, trained with the model; not in the HauhauCS file (below) | separate drafter GGUF |

A 12 GB card holds about 25 to 37 % of Qwen's experts (Gemma: 37 to 61 %), so the cache works harder here.

## Files
Tested set (all downloads under their own licences, never in this repository):
- **Model:** HauhauCS `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`, `Q4_K_P` (23 GB). Any GGUF with architecture `qwen35moe`
  is meant to load (the official `Qwen/Qwen3.6-35B-A3B` quants, unsloth's).
- **Projector:** the `mmproj-...-f16.gguf` beside the model (0.9 GB).
- **MTP drafter:** HauhauCS's file carries no MTP head. unsloth's `Qwen3.6-35B-A3B-MTP-GGUF` carries one inside its model GGUF (block
  40, plus `nextn` tensors). `tools/qwen35_mtp_extract.py` copies that block, the embedding and the output head into a 1.62 GB
  GGUF that both llama.cpp (`-md ... --spec-type draft-mtp`) and Strata-Lite (`--mtp`) load beside the HauhauCS model:
  ```bash
  python3 tools/qwen35_mtp_extract.py Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf mtp-qwen3.6-35b-a3b.gguf
  ```
  In llama.cpp the pairing accepts 67.4 % of drafts (unsloth's own model with its own head: 64.1 %): the head was trained on the base
  model and drafts as well for the uncensored one (`lite/results/qwen36/mtp-pairing.md`). Strata-Lite uses the target's embedding and
  output head by default instead of the file's copies (`--mtp-own-head` for those): it accepts as many drafts and keeps 0.5 GB of the
  card for experts.

## Gate 0: will a small cache hit? (pass)
Routing logged by llama.cpp, replayed through the cache policies (`lite/gate0/run-gate0-qwen36.sh`, `lite/gate0/analyze.py`,
`lite/results/gate0-qwen36/`). Same 12 inputs as Gemma's Gate 0, which are the author's own and not published.

| share of experts cached | adaptive, half-life 32 | half-life 64 | LRU | best fixed set (in hindsight) | fixed profile from other prompts |
|---|---|---|---|---|---|
| 25 % | 77.2 % | 76.3 % | 75.3 % | 77.0 % | 55.6 % |
| 37 % | 87.1 % | 87.1 % | 85.3 % | 88.2 % | 69.4 % |
| 50 % (the gate: 80 %) | 92.9 % | 93.2 % | 92.1 % | 94.6 % | 79.9 % |

Routing is more concentrated than Gemma's and the cache swaps more: half-life 64 keeps the hits of 32 with half the swaps, so **64
is the Qwen default.**

## Engine stages (all against llama.cpp, same card)
Code: `src/qwen35/` (`q35_engine.*`, the test driver `q35_cli.cpp`, the server `q35_server.cpp`). The block rebuilds
llama.cpp's `qwen35moe` graph op for op in ggml. Scripts `lite/tests/q35_*.sh`, numbers `lite/results/qwen36-*/`. Five prompts of 0.6k to
6.5k tokens, 200 greedy steps each (968 steps), on an RTX 5070.

| stage | what | result |
|---|---|---|
| b | full forward, every expert in RAM, f16 KV | bit-identical to llama.cpp `--n-cpu-moe 40`, 968/968 steps, greedy tokens identical |
| c | Strata's CPU expert pool | bit-identical, 968/968 (new `SwiGLU_ggml` kernel mode, below) |
| d1 | layers 30 to 39 whole on the GPU | bit-identical to llama.cpp `--n-cpu-moe 30`, 968/968 |
| d2 | 64 of 256 slots per layer from a static profile | twin check 7,704 rows, 0 differ; hits 38 to 60 % |
| e | adaptation (half-life 64, at most 2 swaps per layer per step) | top-1 vs all-GPU 944/968, KL 4.5e-3; llama.cpp's own all-CPU vs all-GPU on the same prompts: 945/968, 4.6e-3. Hits 72 to 80 % at a 25 % cache. Twins 7,704, 0 differ |
| f | MTP: the extracted head as the drafter, draft-mtp, 3 drafts | top-1 948/968, KL 4.4e-3 (f - e: -1.0e-4, 95 % CI [-4.7e-4, 2.4e-4]); 64.6 % of drafts accepted; twins 10,520, 0 differ |
| speed | background swap stream + CPU/GPU overlap | every scored step bit-identical to synchronous swaps; decode 26.8 to 46.9 tok/s; twins 10,376, 0 differ |
| g | chunked prompt reads with lent VRAM | first token 5k 14.0 to 3.2 s, 15k 43.6 to 7.2 s, 157k 478 to 147 s; chunked - 512 KL +5.3e-4, CI [-1.6e-4, 1.3e-3]; twins 10,192, 0 differ |
| q4 KV | llama.cpp's Hadamard KV rotation mirrored | bit-identical to llama.cpp at q4_0, 968/968; full engine 940/968, KL 1.01e-2 against all-GPU f16, where llama.cpp's own all-GPU q4 scores 940/968, 1.07e-2 |
| server | `strata-lite-server-qwen35` | API parity 63/63 against llama-server on the same files (`lite/results/api/parity-qwen36-5070.txt`) |

What it took to match llama.cpp bit for bit, in case you port another model:
- **Node order.** llama.cpp builds V before K in the attention block; the results depend on it through the allocator.
- **`build_rs`'s empty nodes.** The recurrent-state helper adds clear and copy nodes that sit between nodes ggml-cuda would otherwise
  fuse. Leave them out and a fusion llama.cpp does not do appears.
- **The reserve sequence.** llama.cpp reserves its graphs as prompt (512 tokens, 512 outputs), decode, prompt again over the full KV
  cache. ggml's allocator reuses the last plan, and the plan decides which tensors share memory, which decides whether ggml-cuda's
  fused top-k MoE kernel passes its overlap check. The engine runs the same sequence at load and again after every chunked read.
- **Outputs flagged in the reserve too.** A tensor flagged as an output only in the real graph is not protected by the reserved plan:
  the allocator reuses its memory. The router ids the cache reads are flagged in the reserve graphs as well.
- **SwiGLU on the CPU.** ggml-cpu's dot products plus its exported `ggml_vec_swiglu_f32` on 16-element blocks. A scalar silu rounds
  differently from the SIMD one, so the pool borrows ggml's.

## What is different from Gemma inside
- **Rolling back a recurrent state.** A rejected MTP draft cannot be erased from a delta-net state, so the delta-net layers keep
  n_draft + 1 snapshots (slot s = the state s tokens back, written by `ggml_gated_delta_net` and the conv-window copies), as llama.cpp's
  `n_rs_seq` does. A rejected tail makes the next step read the right snapshot.
- **The drafter has its own KV cache.** After every target batch it re-reads the batch, each token paired with the target's hidden
  row of the position before (llama.cpp's draft-mtp `process()`). Only its K/V cells matter there, so that pass stops at the cache
  writes.
- **The prompt cache.** A delta-net state cannot return to an arbitrary earlier position. The server keeps checkpoints of it (66 MB
  each, on the GPU) where the next request is likely to resume: before the assistant header (the next chat turn re-renders the
  conversation up to there the same way) and before the prompt's last token (the same prompt again). Reads of 16k tokens or more
  also checkpoint the start of the last user message and 512 tokens from the end, as llama-server does
  (`--checkpoint-min-tokens`); each such cut costs one more pass of the experts over PCIe, so short reads skip them.
- **Images.** Qwen's projector gives image tokens M-RoPE positions (time, row, column); the KV cell keeps the time position, and the
  position advances by the chunk's position count, not its token count.

## What is not here yet
- Qwen's recommended sampling (thinking: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5; non-thinking: 0.7, 0.8, 20,
  1.5) is not a server default; send it per request. The model card asks for at least 128k context to keep thinking intact.
- A shared model descriptor: Qwen and Gemma are two engines side by side behind one launcher (`lite/docs/MODEL-DESCRIPTOR.md`).
