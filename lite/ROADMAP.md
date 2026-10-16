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

Done (v0.2.2): **LFM2.5-24B-A2B** on the hybrid engine (a third architecture) and in the installer; **the RAM tier** (lock the
experts the PC can spare, read the rest from the model file: one setting between locked and low-RAM mode); the speed defaults
(pool spin, MTP depth by model and context); the easier START-HERE (menu, shortcut, plain-language failures with fixes).

Next, in order:
0. **Qwen3.6 on Strata's own engine** once upstream releases it (0.1.45): the fork's Strata base moves up from 0.1.42 and the
   setup's Qwen choice switches when Strata's engine measures at least as well on the standard benchmark.
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
5. **Models that fit the card whole** (LFM2.5-8B on 12 GB; any of these on a 24 GB card): llama.cpp's own offload is faster
   there today; the engine's full-GPU path, or the setup handing such a model to llama-server.
5. Smaller quants from the same releases (IQ3) for more residency on 12 GB, measured in the same grid.
