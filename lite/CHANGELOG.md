# Changelog

## 0.2.2 (2026-10-16), experimental
- **LFM2.5-24B-A2B** (`lfm2moe`) on the hybrid engine (Codex's port, `lite/docs/LFM25.md`): short-convolution blocks with their own
  state and checkpoints, sigmoid + bias routing, the expert cache, CPU pool, swaps, overlap, lent reads and the KV pager; an optional
  rank-one output LoRA; 128K context. The setup offers AtomicChat's uncensored Q4_K_M (default) and Liquid AI's QAD Q4_0.
- **The RAM tier** (both engines, `--ram-budget GIB`): the experts the card does not hold are locked in RAM up to a budget and the rest
  are read from the model file (the OS file cache, else the disk); by default the engine locks what the PC can spare when it starts
  (available RAM less a headroom: 4 GiB / 15 % on Linux, 8 GiB / 25 % on Windows) instead of all or nothing, so a 32 GB PC no longer
  locks so much that Windows runs out. Bit for bit the same output whatever the budget (only where the bytes come from changes).
  `--ram-budget 0` is low-RAM mode.
- **Speed defaults:** the CPU pool's workers spin 100 us before sleeping, not 20 ms (the same speed, 57 % less CPU time); Qwen3.6 drafts
  2 tokens per step, not 3 (+21 % writing on a short chat, RTX 3060); Gemma 4 drafts 3, and 2 once the context passes 32K
  (`--n-draft-long`, `--n-draft-long-from`).
- **Easier START-HERE:** a short menu on every run (Enter starts the model; change the model or its settings; update Strata-Lite), a
  desktop shortcut on the first install, startup failures said in plain words with the fixes to pick (not enough RAM or VRAM, a port in
  use, a CPU the ready-made engine does not suit, missing CUDA libraries, an old driver, a missing model file), the window titled
  Strata-Lite. UPDATE.bat / update.sh update a Strata-Lite install.
- Fixes: on a 12 GB card at 262K, Gemma read a 17K-32K-token prompt as one chunk that did not fit the card and the server stopped
  (the read's estimate now uses measured sizes and counts the window cache's growth; such prompts read in 16K chunks); upstream
  Strata fixes: no SIMD code before `main` in the AVX-512 / AVX2 / AVX files (#2324), IQ4_NL down rows declined in the AVX1 kernel
  (#2638), a CPU without a `flags` line (#2551), metadata-only GGUF shards (#2491), host compilers for `--build` (#1645), a failed
  build no longer leaves the old version mark (#2054), `.bat` line endings (#1939), UTF-8 file reads (#2648).

## 0.2.1 (2026-10-11), experimental
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
