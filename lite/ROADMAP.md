# Roadmap

Done (2026-10-10): Gemma 4 26B-A4B on one 12 GB card, the grid on RTX 3060 and RTX 5070 at 32k / 131k / 262k, MTP, chunked reads,
q4 KV, the image projector on demand, llama-server API parity, sm_86 / sm_89 / sm_120 builds.

Done (2026-10-11): Qwen3.6-35B-A3B (HauhauCS Uncensored Aggressive Q4_K_P with the MTP head from unsloth's release), every stage
checked against llama.cpp, its server behind the same launcher, the grid on both cards (`lite/docs/QWEN36.md`).

Done (2026-10-10, v0.2.0): the installer (`START-HERE.bat` / `setup.sh`: Gemma 4 or Qwen3.6, the measured uncensored release, the
official quants, or a file of your own; low-RAM mode under 24 / 32 GB) and Windows + Linux release builds.

Done (v0.2.1): **the KV pager** (the card holds only the context the conversation has reached: 32K at load, 16K steps, the expert
cache gives way and takes the room back on a new chat; bit for bit the same output) with **chat-page compaction at 192K**, which
together made **q8 KV affordable for Gemma** (64K, Wikipedia text: top-1 87.1 % against q4's 80.5 %, at the f16 noise floor). This
was the goal of the old item "KV streaming so a 262k context stops costing the cache half its slots": met without streaming.

Next, in order:
1. **A model descriptor** (hparams, tensor map, expert shapes and quants, activation, norm order, mixer pattern) so a third
   architecture is a descriptor plus a block file rather than a fork; Gemma moves onto the model-neutral slot cache Qwen uses,
   behind a full re-validation run (`lite/docs/MODEL-DESCRIPTOR.md`).
2. **8 GB cards** (tested target: Quadro RTX 4000, Turing): about 30 % of the experts resident, expected 76 % hits at 32k. If the
   KV cannot fit there, the fallback is streaming old KV pages from RAM, measured and set aside for 12 GB cards (v0.2.1): these
   models attend to every past token, so the streamed part crosses PCIe once per token (~1 GB for Gemma q8 between 96K and 192K,
   ~85 ms a token on PCIe 3.0 at 12.3 GB/s), against a few expert slots for holding it on the card.
3. **Predicted-miss prefetch (speculative routing)**: run block L+1's router on block L's output a layer early, start
   copying the predicted non-resident experts over PCIe while block L computes, and warm the next step from the MTP drafts' routes.
   Today's adaptation is reactive (decayed counts, swaps between steps); this converts misses into hits within the step. Measured
   by misses converted, prefetch accuracy, and the cost to the adaptation stream; it matters most on slower cards and x16 links, and
   cannot keep up on x4.
4. Batched slots (several requests at once), as Strata has for Qwen3.8.
5. Smaller quants from the same releases (IQ3) for more residency on 12 GB, measured in the same grid.
