# Roadmap

Done (2026-10-10): Gemma 4 26B-A4B on one 12 GB card, the grid on RTX 3060 and RTX 5070 at 32k / 131k / 262k, MTP, chunked reads,
q4 KV, the image projector on demand, llama-server API parity, sm_86 / sm_89 / sm_120 builds.

Done (2026-10-11): Qwen3.6-35B-A3B (HauhauCS Uncensored Aggressive Q4_K_P with the MTP head from unsloth's release), every stage
checked against llama.cpp, its server behind the same launcher, the grid on both cards (`lite/docs/QWEN36.md`).

Next, in order:
1. **A model descriptor** (hparams, tensor map, expert shapes and quants, activation, norm order, mixer pattern) so a third
   architecture is a descriptor plus a block file rather than a fork; Gemma moves onto the model-neutral slot cache Qwen uses,
   behind a full re-validation run (`lite/docs/MODEL-DESCRIPTOR.md`).
2. **An installer**: pick a model (Qwen3.6-35B, Gemma 4 26B, or Strata's own Qwen3.8-Flash-Next), then the uncensored release
   tested here, the official quants, or a file of your own.
3. **KV streaming** (Strata's `--kv-resident`) so a 262k context stops costing the cache half its slots on 12 GB.
4. **8 GB cards** (tested target: Quadro RTX 4000, Turing): about 30 % of the experts resident, expected 76 % hits at 32k.
5. **Predicted-miss prefetch (speculative routing)**: run block L+1's router on block L's output a layer early, start
   copying the predicted non-resident experts over PCIe while block L computes, and warm the next step from the MTP drafts' routes.
   Today's adaptation is reactive (decayed counts, swaps between steps); this converts misses into hits within the step. Measured
   by misses converted, prefetch accuracy, and the cost to the adaptation stream; it matters most on slower cards and x16 links, and
   cannot keep up on x4.
6. Batched slots (several requests at once), as Strata has for Qwen3.8.
7. Smaller quants from the same releases (IQ3) for more residency on 12 GB, measured in the same grid.
