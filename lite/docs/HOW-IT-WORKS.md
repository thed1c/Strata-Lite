# How Strata-Lite works

Gemma 4 26B-A4B is 30 blocks, each with 128 experts of which 8 run per token, plus a small dense FFN. The experts weigh 15.1 GB; the
rest of the model 1.65 GB. A 12 GB card cannot hold the experts, and every token needs 240 of them.

**Static offload (llama.cpp's `--n-cpu-moe`)** keeps whole blocks' experts in RAM and computes them on the CPU for every token,
whichever experts the token wanted. On a 12 GB card that is 11 to 18 of 30 blocks on the CPU, and prompt reads crawl because the
CPU computes those blocks for every chunk.

**Strata-Lite** does what Strata does for Qwen3.8:
1. **The expert arena.** Every expert stays in RAM, mapped from the GGUF and pinned.
2. **A GPU expert cache per layer.** A slot table holds the experts the conversation is actually using. Routing is skewed (about 70
   of 128 experts per layer see use within one conversation, and the set moves between conversations), so a cache of half the
   experts hits more than 90 % of the time (README, Gate 0).
3. **Adaptation.** Usage counts decay with a half-life of 32 tokens; between steps, up to 2 experts per layer are swapped in on a
   background copy stream. The cache follows the text.
4. **Misses on the CPU pool, in the same step.** The 8 routed experts split into hits (the GPU computes them with `mul_mat_id` over
   the slots) and misses (Strata's CPU pool computes them from RAM). The two run overlapped; a layer with no miss skips the CPU
   entirely, which at 92 % hits is most layers.
5. **Chunked prompt reads with lent VRAM.** A long prompt is read in chunks of up to 32k tokens; the chunk borrows the slot
   arena's VRAM for its buffers and streams the experts it touches over PCIe, then gives the slots back. The cache refills over the
   first tokens of the reply (the one visible cost: the first reply after a long read is 13–27 % slower).
6. **MTP drafting.** Gemma 4's drafter proposes up to 3 tokens per step; the model verifies them in one pass. 70–80 % accepted.
7. **The block itself** is built from ggml's CUDA ops, mirroring llama.cpp's `gemma4` graph op for op. That is why every stage could
   be checked bit for bit against llama.cpp: attention, norms, rope, the dense FFN, the router and the head are the same kernels in
   the same order; only the expert layer is Strata's.

**Qwen3.6-35B-A3B** goes through the same machinery with its own block (`src/qwen35/`, `lite/docs/QWEN36.md`): 40 layers, 30 of
them Gated DeltaNet with a recurrent state, 256 experts per layer of which a 12 GB card holds a quarter to a third. The differences
the engine handles: a recurrent state that MTP rejections roll back from snapshots, a drafter with its own KV cache, image tokens
with M-RoPE positions, and a prompt cache that resumes at checkpoints of the recurrent state.

What is kept from Strata: the arena, the cache and its adaptation, the CPU pool, prompt-chunk lending, the prompt and
conversation caches, the server layer. What is new: the gemma4 block, the pack reader for its quant formats, the MTP drafter as a
separate small model, q4 KV mirrored from llama.cpp (its Hadamard rotation), the image projector on demand.
