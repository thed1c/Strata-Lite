# Changelog

## 0.2.0 (2026-10-11), experimental
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
