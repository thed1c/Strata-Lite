# Changelog

## 0.2.1 (unreleased), experimental
- **The KV pager** (both engines): the full-attention KV starts at 32K cells on the card and grows in 16K steps as the conversation
  needs them; the expert cache gives way, and takes the room back when a new conversation starts (`--kv-initial`, `--kv-step`;
  `--kv-initial 0` reserves the whole context at load as before). Bit for bit the same output as the whole-context engine (both
  models, growth during a read, during decode and with MTP). More expert slots for short chats: Qwen 89 vs 73, Gemma 81 vs 70 at 262K.
- **Chat-page compaction**: at 3/4 of the context (192K of 256K) the page summarizes the conversation and goes on from the summary
  plus the last turns, with a marker that opens the summary; a big paste makes room the same way; one that can never fit is refused.
- **Gemma defaults to q8_0 KV** (Qwen keeps q4_0): at 64K, top-1 87.1 % against q4's 80.5 % and an f16 noise floor of 87.1 %.
- KV streaming from RAM measured and set aside for 12 GB cards (`lite/ROADMAP.md`).
- Fixes: the projector warm-up on a second GPU or the card (an unset `text_len`; it threw and was skipped); an answer with an invalid
  UTF-8 byte made the chat parser fail and the server answer 500 (`src/lite/utf8.hpp`).

## 0.2.0 (2026-10-10), experimental
- **Qwen3.6-35B-A3B** (`qwen35moe`) engine and server in `src/qwen35/` (`lite/docs/QWEN36.md`): llama.cpp's graph rebuilt op for op
  (bit-identical through the full forward, the CPU pool, whole GPU layers and q4 KV), Strata's adaptive GPU expert cache, CPU pool,
  background swaps and overlap, MTP from the extracted head with recurrent-state rollback, chunked reads with lent VRAM, the image
  projector with M-RoPE positions, a prompt cache with recurrent-state checkpoints. API parity 63/63. The grid on RTX 3060 and 5070.
- `strata-lite-server` reads the GGUF's architecture and starts the matching engine's server.
- `tools/qwen35_mtp_extract.py`: the MTP head of a Qwen3.5/3.6 GGUF into its own drafter GGUF.
- `lite/patches/llama.cpp-b6b003d-strata-lite.patch`: the build is a patched llama.cpp b6b003d (a sampler-on-logits helper and the
  route-oracle tools), no longer a private copy.
- Cache behaviour flags: `--adapt-half-life`, `--adapt-budget`, `--no-adapt`, `--no-adapt-refill`, `--no-async-swaps`, `--no-overlap`.

## 0.1.0 (2026-10-10), experimental
- gemma4 engine: the Gemma 4 26B-A4B block from ggml ops, Strata's expert arena, GPU cache, adaptation, CPU pool, chunked reads with
  lent VRAM, background swaps, CPU/GPU overlap, MTP drafting, q4 KV, the image projector on demand.
- `strata-lite-server`: llama-server-compatible API (63/63 parity checks; `lite/docs/API.md`).
- Builds for sm_86, sm_89, sm_120.
- The grid: RTX 3060 and RTX 5070 at 32k / 131k / 262k against llama.cpp static offload with MTP (README).
