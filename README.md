# Strata-Lite

*A fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221: Strata's adaptive expert cache for Gemma 4 26B-A4B,
Qwen3.6-35B-A3B and LFM2.5-24B-A2B on one 12 GB card. Strata's own README is [`STRATA.md`](STRATA.md).*

**Experimental.** A port of [Strata](https://github.com/Niko1221/Strata)'s adaptive expert-cache engine to Gemma 4 26B-A4B,
Qwen3.6-35B-A3B and LFM2.5-24B-A2B, so each runs on one 12 GB card at near-GPU speed, with MTP drafting, chunked prompt reads and
the image projector, behind a llama-server-compatible API. Built and measured in about a week; read the numbers as what they are,
two cards and three models.

| | |
|---|---|
| Models | Gemma 4 26B-A4B (HauhauCS QAT Uncensored Balanced, Q4_K_M, with its MTP drafter and projector); Qwen3.6-35B-A3B (HauhauCS Uncensored Aggressive, Q4_K_P, with its projector and the MTP head from unsloth's release, `lite/docs/QWEN36.md`); LFM2.5-24B-A2B (AtomicChat Uncensored Q4_K_M or Liquid AI's QAD Q4_0, text only, 128K, `lite/docs/LFM25.md`) |
| Cards | 12 GB: RTX 3060 (sm_86), RTX 5070 (sm_120); built for sm_89 too. 8 GB is on the roadmap |
| Needs | Windows or Linux, an NVIDIA RTX 30 / 40 / 50 card, driver 580+, 24 GB of RAM for Gemma or LFM2.5 and 32 GB for Qwen (with less, the engine keeps what the PC can spare in RAM and reads the rest from the model file: 16 GB runs, slower), a CPU with AVX2. The setup installs everything else (Install, below); to compile it yourself: `lite/docs/BUILD.md` |
| Run | `strata-lite-server -m model.gguf --mtp mtp.gguf --mmproj mmproj.gguf`; it picks the engine from the GGUF (`lite/docs/RUN.md`) |
| API | llama-server's and OpenAI chat; 63/63 parity checks for each model (`lite/docs/API.md`) |
| Images | The vision projector does not sit on the card beside the model: on one GPU it waits in RAM and takes GPU memory only while it is in use. Small images encode on the CPU, big ones move it to the card and it leaves after 15 s idle; prompts over 64K keep images on the CPU; a second GPU takes it automatically; under 32 GB of RAM it loads only when an image arrives (`--mmproj-place`, see "The image projector, on demand") |
| Docs | `lite/docs/HOW-IT-WORKS.md`, `lite/docs/CORRECTNESS.md`, `lite/docs/QWEN36.md`, `lite/docs/MODEL-DESCRIPTOR.md`, `lite/ROADMAP.md`, `lite/CONTRIBUTING.md`, `lite/CHANGELOG.md` |
| Licence | MIT (`LICENSE`); weights are yours to download under their own terms |

Gemma 4 26B-A4B is a mixture-of-experts model: 30 blocks with 128 experts each, 8 used per token. Its experts weigh
15.1 GB, more than a 12 GB card holds. llama.cpp's answer is static offload: whole blocks of experts stay in RAM and are
computed on the CPU (`--n-cpu-moe`). Strata's answer is a per-layer GPU cache of the experts the conversation is actually
using. The cache adapts as the conversation moves, misses go to a CPU expert pool in the same step, and swaps run on a
background copy stream.

## New in v0.2.2: LFM2.5-24B, a RAM tier, an easier start

- **A third model, LFM2.5-24B-A2B** (Liquid AI's hybrid MoE: 24B, 2B active, text only, 128K), on the same engine as Qwen3.6, with
  the uncensored AtomicChat release and Liquid AI's own QAD Q4_0 in the installer (`lite/docs/LFM25.md`). It reads fastest of the
  three (3,300 tokens a second on a 15K prompt on an RTX 5070, a 120K prompt in about a minute) and writes 97-116 tokens a second.
- **The RAM tier.** v0.2.1 locked all the experts the card does not hold in RAM, or none (low-RAM mode). v0.2.2 locks what the PC
  can spare when it starts, leaving room for the OS and other programs, and reads the rest from the model file (`--ram-budget`).
  The output is the same bit for bit whatever is locked (checked on Gemma and Qwen: all, part, none).
- **Lighter on the CPU, faster Qwen writing.** The expert pool's workers no longer spin 20 ms between steps: the same speed with
  about half the CPU time over the same session (Gemma 349 s against 782, Qwen 472 against 1,111 on the RTX 3060). Qwen3.6 drafts
  2 tokens per step instead of 3: +18 % writing on the RTX 3060, +7 % on the RTX 5070. Gemma drafts 2 instead of 3 once the context
  passes 32K.
- **An easier START-HERE**: a short menu on every run (Enter starts the model), a desktop shortcut, and startup failures said in
  plain words with the fix to pick.
- **Fixed:** on a 12 GB card at 262K, Gemma read a 17K-32K-token prompt in one piece that did not fit the card, and the server
  stopped; such prompts now read in 16K pieces. Plus the upstream Strata fixes for older CPUs and the installer
  (`lite/CHANGELOG.md`).

Every column below was measured the same way on 2026-10-16, at each engine's own defaults with MTP drafting on (Gemma and Qwen;
LFM2.5 has no MTP head), on the grid's two cards. **The RTX 3060 now sits in a PCIe 3.0 x8 slot** (x16 in the earlier tables), so
every 3060 column here, v0.2.1 and llama.cpp included, was measured again in that slot; long prompts read 8-18 % slower than
at x16 (v0.2.1, 157K: Gemma 252 s against 234, Qwen 260 against 221). The 5070 is unchanged; its v0.2.1 and llama.cpp columns are the 2026-10-11 runs, and LFM2.5's llama.cpp columns the
2026-10-16 night runs, on the same text. llama.cpp is run as in the grid: the expert blocks that fit on the card (the most that
held through the whole session), the rest in RAM, q4 KV for Gemma and Qwen, f16 for LFM2.5 on both engines. The text is
Wikipedia (the CC BY-SA corpus published with [llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)).
Prefills are fresh conversations; "writing after a 4K prompt" is the mean of the last three of four sends of two different
prompts; the long prompt (157K; LFM2.5 120K, its context being 128K) is read once and answered twice. Both engines run with the
same sampling: LFM2.5's model card asks for a 1.05 repeat penalty, which Strata-Lite's LFM2.5 server applies by default and
llama-server does not, so these requests set it to 1.0 for both (with 1.05, Strata-Lite writes 8-16 % slower on a short chat:
the penalty scans every token of the vocabulary each step). The accuracy rows are the
KV type's cost against f16 KV at 64K (`lite/docs/RELEASES.md`, v0.2.1). The ratio in brackets is v0.2.2 over the other column
(for the first token, the other column's time over v0.2.2's).

**Writing at the long prompt swings from turn to turn** with how well MTP predicts the text (on Wikipedia 45-80 % of drafts are
accepted): read differences in those rows under ~25 % as level. v0.2.2 against v0.2.1 is level everywhere except Qwen's writing.

### v0.2.2 vs llama.cpp

**Gemma 4 26B-A4B, RTX 3060 (PCIe 3.0 x8), 262K context**

| | v0.2.2 | llama.cpp |
|---|---|---|
| KV cache | q8_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 87.1 %) | 87.1 % | 80.5 % |
| mean KL against f16 KV, 64K (floor 0.073) | 0.106 | 0.216 |
| experts on the card at load | 79 slots/layer (the KV grows from 32K) | 11 of 30 expert blocks on the card (`--n-cpu-moe 19`) |
| experts on the card after the 157K prompt | 56 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 1,164 (3.48×) | 334 |
| prefill, 15K prompt, tok/s | 1,238 (3.74×) | 331 |
| prefill, 157K prompt, tok/s | 613 (2.19×) | 280 |
| first token after the 157K prefill, s | 256 (2.19×) | 560 |
| writing after a 4K prompt, tok/s | 72.0 (1.33×) | 54.1 |
| writing at 157K, first turn, tok/s | 31.4 (1.24×) | 25.3 |
| writing at 157K, second turn, tok/s | 36.8 (1.30×) | 28.3 |

**Gemma 4 26B-A4B, RTX 5070, 262K context**

| | v0.2.2 | llama.cpp |
|---|---|---|
| KV cache | q8_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 87.1 %) | 87.1 % | 80.5 % |
| mean KL against f16 KV, 64K (floor 0.073) | 0.106 | 0.216 |
| experts on the card at load | 78 slots/layer (the KV grows from 32K) | 11 of 30 expert blocks on the card (`--n-cpu-moe 19`) |
| experts on the card after the 157K prompt | 54 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 2,230 (8.37×) | 266 |
| prefill, 15K prompt, tok/s | 2,263 (8.43×) | 269 |
| prefill, 157K prompt, tok/s | 987 (3.83×) | 257 |
| first token after the 157K prefill, s | 159 (3.83×) | 610 |
| writing after a 4K prompt, tok/s | 112.9 (2.32×) | 48.6 |
| writing at 157K, first turn, tok/s | 47.9 (1.23×) | 38.9 |
| writing at 157K, second turn, tok/s | 61.2 (1.65×) | 37.1 |

**Qwen3.6-35B-A3B, RTX 3060 (PCIe 3.0 x8), 262K context**

| | v0.2.2 | llama.cpp |
|---|---|---|
| KV cache | q4_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 94.1 %) | 92.2 % | 92.2 % |
| mean KL against f16 KV, 64K (floor 0.024) | 0.020 | 0.020 |
| experts on the card at load | 90 slots/layer (the KV grows from 32K) | 9 of 40 expert blocks on the card (`--n-cpu-moe 31`) |
| experts on the card after the 157K prompt | 78 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 706 (3.23×) | 219 |
| prefill, 15K prompt, tok/s | 1,076 (4.88×) | 220 |
| prefill, 157K prompt, tok/s | 618 (3.06×) | 202 |
| first token after the 157K prefill, s | 254 (3.06×) | 779 |
| writing after a 4K prompt, tok/s | 58.0 (1.36×) | 42.8 |
| writing at 157K, first turn, tok/s | 30.5 (1.25×) | 24.4 |
| writing at 157K, second turn, tok/s | 28.7 (1.18×) | 24.4 |

**Qwen3.6-35B-A3B, RTX 5070, 262K context**

| | v0.2.2 | llama.cpp |
|---|---|---|
| KV cache | q4_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 94.1 %) | 92.2 % | 92.2 % |
| mean KL against f16 KV, 64K (floor 0.024) | 0.020 | 0.020 |
| experts on the card at load | 88 slots/layer (the KV grows from 32K) | 9 of 40 expert blocks on the card (`--n-cpu-moe 31`) |
| experts on the card after the 157K prompt | 75 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 1,353 (7.76×) | 174 |
| prefill, 15K prompt, tok/s | 1,982 (11.26×) | 176 |
| prefill, 157K prompt, tok/s | 1,066 (6.18×) | 173 |
| first token after the 157K prefill, s | 147 (6.18×) | 910 |
| writing after a 4K prompt, tok/s | 78.9 (1.65×) | 47.8 |
| writing at 157K, first turn, tok/s | 43.1 (1.43×) | 30.2 |
| writing at 157K, second turn, tok/s | 47.0 (1.56×) | 30.2 |

**LFM2.5-24B-A2B, AtomicChat Uncensored Q4_K_M, RTX 3060 (PCIe 3.0 x8), 128K context**

| | v0.2.2 | llama.cpp |
|---|---|---|
| KV cache | f16 | f16 |
| experts on the card at load | 42 slots/layer (the KV grows from 32K) | 16 of 38 expert blocks on the card (`--n-cpu-moe 22`) |
| experts on the card after the 120K prompt | 33 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 1,041 (2.60×) | 400 |
| prefill, 15K prompt, tok/s | 1,870 (4.73×) | 395 |
| prefill, 120K prompt, tok/s | 1,132 (3.01×) | 376 |
| first token after the 120K prefill, s | 106 (3.01×) | 320 |
| writing after a 4K prompt, tok/s | 64.6 (1.40×) | 46.0 |
| writing at 120K, first turn, tok/s | 37.2 (1.06×) | 35.1 |
| writing at 120K, second turn, tok/s | 39.8 (1.15×) | 34.5 |

**LFM2.5-24B-A2B, AtomicChat Uncensored Q4_K_M, RTX 5070, 128K context**

| | v0.2.2 | llama.cpp |
|---|---|---|
| KV cache | f16 | f16 |
| experts on the card at load | 42 slots/layer (the KV grows from 32K) | 19 of 38 expert blocks on the card (`--n-cpu-moe 19`) |
| experts on the card after the 120K prompt | 32 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 2,021 (5.68×) | 356 |
| prefill, 15K prompt, tok/s | 3,307 (9.38×) | 353 |
| prefill, 120K prompt, tok/s | 1,795 (5.10×) | 352 |
| first token after the 120K prefill, s | 67 (5.10×) | 341 |
| writing after a 4K prompt, tok/s | 97.1 (1.99×) | 48.8 |
| writing at 120K, first turn, tok/s | 55.2 (1.36×) | 40.6 |
| writing at 120K, second turn, tok/s | 59.3 (1.46×) | 40.6 |

**LFM2.5-24B-A2B, Liquid AI QAD Q4_0, RTX 3060 (PCIe 3.0 x8), 128K context**

| | v0.2.2 | llama.cpp |
|---|---|---|
| KV cache | f16 | f16 |
| experts on the card at load | 49 slots/layer (the KV grows from 32K) | 15 of 38 expert blocks on the card (`--n-cpu-moe 23`) |
| experts on the card after the 120K prompt | 38 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 1,066 (2.57×) | 415 |
| prefill, 15K prompt, tok/s | 1,906 (4.66×) | 409 |
| prefill, 120K prompt, tok/s | 1,281 (3.30×) | 388 |
| first token after the 120K prefill, s | 94 (3.30×) | 309 |
| writing after a 4K prompt, tok/s | 78.8 (1.51×) | 52.2 |
| writing at 120K, first turn, tok/s | 43.0 (1.14×) | 37.6 |
| writing at 120K, second turn, tok/s | 41.1 (1.09×) | 37.6 |

**LFM2.5-24B-A2B, Liquid AI QAD Q4_0, RTX 5070, 128K context**

| | v0.2.2 | llama.cpp |
|---|---|---|
| KV cache | f16 | f16 |
| experts on the card at load | 49 slots/layer (the KV grows from 32K) | 23 of 38 expert blocks on the card (`--n-cpu-moe 15`) |
| experts on the card after the 120K prompt | 37 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 2,056 (4.31×) | 478 |
| prefill, 15K prompt, tok/s | 3,359 (7.08×) | 475 |
| prefill, 120K prompt, tok/s | 1,938 (4.14×) | 468 |
| first token after the 120K prefill, s | 62 (4.14×) | 256 |
| writing after a 4K prompt, tok/s | 115.5 (1.98×) | 58.2 |
| writing at 120K, first turn, tok/s | 58.6 (1.25×) | 46.8 |
| writing at 120K, second turn, tok/s | 64.6 (1.38×) | 46.7 |

### v0.2.2 vs v0.2.1

**Gemma 4 26B-A4B, RTX 3060 (PCIe 3.0 x8), 262K context**

| | v0.2.2 | v0.2.1 |
|---|---|---|
| KV cache | q8_0 | q8_0 |
| top-1 agreement with f16 KV, 64K (floor 87.1 %) | 87.1 % | 87.1 % |
| mean KL against f16 KV, 64K (floor 0.073) | 0.106 | 0.106 |
| experts on the card at load | 79 slots/layer (the KV grows from 32K) | 79 slots/layer (the KV grows from 32K) |
| experts on the card after the 157K prompt | 56 slots/layer | 56 slots/layer |
| prefill, 5K prompt, tok/s | 1,164 (1.00×) | 1,165 |
| prefill, 15K prompt, tok/s | 1,238 (0.99×) | 1,248 |
| prefill, 157K prompt, tok/s | 613 (0.98×) | 623 |
| first token after the 157K prefill, s | 256 (0.98×) | 252 |
| writing after a 4K prompt, tok/s | 72.0 (1.01×) | 71.2 |
| writing at 157K, first turn, tok/s | 31.4 (0.93×) | 33.9 |
| writing at 157K, second turn, tok/s | 36.8 (1.38×) | 26.6 |

**Gemma 4 26B-A4B, RTX 5070, 262K context**

| | v0.2.2 | v0.2.1 |
|---|---|---|
| KV cache | q8_0 | q8_0 |
| top-1 agreement with f16 KV, 64K (floor 87.1 %) | 87.1 % | 87.1 % |
| mean KL against f16 KV, 64K (floor 0.073) | 0.106 | 0.106 |
| experts on the card at load | 78 slots/layer (the KV grows from 32K) | 78 slots/layer (the KV grows from 32K) |
| experts on the card after the 157K prompt | 54 slots/layer | 54 slots/layer |
| prefill, 5K prompt, tok/s | 2,230 (1.00×) | 2,234 |
| prefill, 15K prompt, tok/s | 2,263 (1.00×) | 2,268 |
| prefill, 157K prompt, tok/s | 987 (0.99×) | 1,000 |
| first token after the 157K prefill, s | 159 (0.99×) | 157 |
| writing after a 4K prompt, tok/s | 112.9 (1.01×) | 111.5 |
| writing at 157K, first turn, tok/s | 47.9 (0.91×) | 52.5 |
| writing at 157K, second turn, tok/s | 61.2 (0.92×) | 66.7 |

**Qwen3.6-35B-A3B, RTX 3060 (PCIe 3.0 x8), 262K context**

| | v0.2.2 | v0.2.1 |
|---|---|---|
| KV cache | q4_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 94.1 %) | 92.2 % | 92.2 % |
| mean KL against f16 KV, 64K (floor 0.024) | 0.020 | 0.020 |
| experts on the card at load | 90 slots/layer (the KV grows from 32K) | 89 slots/layer (the KV grows from 32K) |
| experts on the card after the 157K prompt | 78 slots/layer | 77 slots/layer |
| prefill, 5K prompt, tok/s | 706 (1.00×) | 707 |
| prefill, 15K prompt, tok/s | 1,076 (1.00×) | 1,074 |
| prefill, 157K prompt, tok/s | 618 (1.02×) | 603 |
| first token after the 157K prefill, s | 254 (1.02×) | 260 |
| writing after a 4K prompt, tok/s | 58.0 (1.18×) | 49.0 |
| writing at 157K, first turn, tok/s | 30.5 (1.19×) | 25.7 |
| writing at 157K, second turn, tok/s | 28.7 (1.10×) | 26.1 |

**Qwen3.6-35B-A3B, RTX 5070, 262K context**

| | v0.2.2 | v0.2.1 |
|---|---|---|
| KV cache | q4_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 94.1 %) | 92.2 % | 92.2 % |
| mean KL against f16 KV, 64K (floor 0.024) | 0.020 | 0.020 |
| experts on the card at load | 88 slots/layer (the KV grows from 32K) | 87 slots/layer (the KV grows from 32K) |
| experts on the card after the 157K prompt | 75 slots/layer | 75 slots/layer |
| prefill, 5K prompt, tok/s | 1,353 (1.00×) | 1,347 |
| prefill, 15K prompt, tok/s | 1,982 (1.00×) | 1,978 |
| prefill, 157K prompt, tok/s | 1,066 (1.00×) | 1,064 |
| first token after the 157K prefill, s | 147 (1.00×) | 148 |
| writing after a 4K prompt, tok/s | 78.9 (1.07×) | 73.9 |
| writing at 157K, first turn, tok/s | 43.1 (0.95×) | 45.6 |
| writing at 157K, second turn, tok/s | 47.0 (0.96×) | 48.8 |


Earlier releases' tables (v0.2.1 vs llama.cpp and vs v0.2.0, with the KV accuracy measurements and the KV-streaming study):
[`lite/docs/RELEASES.md`](lite/docs/RELEASES.md).

## Install

[Download Strata-Lite](https://github.com/thed1c/Strata-Lite/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the folder.

It is Strata's installer. In "Which model?" choose **Gemma 4 26B-A4B**, **Qwen3.6-35B-A3B** or **LFM2.5-24B-A2B** (Strata's own
Qwen3.8-Flash-Next models are there too). Then it asks:
- **which files:** *uncensored* (the releases every number here was measured with), *official* (Gemma and Qwen: unsloth's
  quantizations, experimental, not measured with this engine; LFM2.5: Liquid AI's QAD Q4_0, measured), or GGUF files you already have;
- **how much context:** 32K, 128K or 256K tokens (LFM2.5: up to 128K);
- **images:** the image encoder loads only when an image arrives;
- **a Hugging Face token:** optional, press Enter (none of the files needs one).

Press Enter for the recommended answers. It downloads the model (Qwen ~24 GB, Gemma ~18 GB, LFM2.5 ~15 GB, eight connections
at a time, resumable) and NVIDIA's CUDA libraries, gets the ready-made engine, puts a Strata-Lite shortcut on the desktop, and
starts the model. Your browser opens a chat page at `http://127.0.0.1:8080`; the same address serves the OpenAI and llama-server
APIs (`/v1/chat/completions`).

What it needs: an NVIDIA RTX 30, 40 or 50 card with 12 GB (8-11 GB runs with fewer experts on the card), driver 580 or
newer, 24 GB of RAM for Gemma or LFM2.5 and 32 GB for Qwen, and a CPU with AVX2. The first start loads ~13-22 GB into locked
RAM: the PC can be slow for a minute. With less RAM the engine locks what the PC can spare (leaving room for the OS and your
other programs) and reads the rest from the model file as needed, so keep it on an SSD; with 16 GB nearly nothing is locked.
Measured with 16 GB on an RTX 3060: Gemma writes about 10 tokens a second and reads an image in 11 s; Qwen about 4 tokens a
second (Gemma with enough RAM, same test: about 43).

Next time, double-click the shortcut (or run `START-HERE.bat` / `./setup.sh`): a short menu comes up, and Enter starts the
model. The menu also changes the model or its settings and updates Strata-Lite. If the model cannot start, it says why in plain
words (not enough RAM or graphics memory, a port in use, a missing file...) and offers the fix.

## Results: Gemma 4 26B-A4B

**On the same 12 GB card, at the same context, Strata-Lite reads prompts 2–7× faster than llama.cpp and writes 1.2–2.2× faster. Its top-token agreement with an all-GPU run is within about one point of llama.cpp's.**

| per card, all three contexts | RTX 3060 | RTX 5070 |
|---|---|---|
| reading a 5k–15k prompt | 2.5–3.6× | 5.1–8.0× |
| reading the 157k prompt (time to first token) | 234 s vs 474 s | 159 s vs 596 s |
| writing (decode, MTP on in both) | 1.19–1.41× | 1.68–2.16× |
| writing at 157k of context | 32 vs 25 tok/s | 46 vs 29 tok/s |
| top-1 agreement with all-GPU, 5 prompts | 96.1–96.4 % vs 95.9–96.6 % | 96.7–96.8 % vs 96.3–97.2 % |

**The costs:**
- **The first reply after a long read writes 13–27 % slower than the second,** while the cache settles after the read; llama.cpp has no such dip.
- **Mean KL against all-GPU is slightly higher than llama.cpp's** (1.5–2.0e-2 vs 1.3–1.7e-2). The twin check shows every cached row computes exactly; the difference is which experts run on the GPU and which on the CPU.
- **Both engines fit the image projector beside the model at every context**, when it is forced onto the card for this test (`--mmproj-place gpu`). Strata-Lite gives up 16–18 expert slots for it; llama.cpp moves 3–4 more blocks of experts to RAM. Strata-Lite's default doesn't do that: the projector waits in RAM and visits the card only for an image, so text keeps every slot ("The image projector, on demand").

The grid below is the headline. Each column puts the same 12 GB card, the same context, the same prompts and the same greedy
sampling through both engines, with MTP drafting on in both:
- **Strata-Lite:** the most expert slots the card holds, sized automatically.
- **llama.cpp:** static expert offload at the most it can hold, i.e. the fewest `--n-cpu-moe` blocks that load and run.

Both use q4_0 KV at every context.

<!-- GRID:START -->
### RTX 3060 (12 GB, 360 GB/s; PCIe 3.0 x16)

| | 32k Strata-Lite | 32k llama.cpp | 131k Strata-Lite | 131k llama.cpp | 262k Strata-Lite | 262k llama.cpp |
|---|---|---|---|---|---|---|
| config | 81 slots/layer, 16384-token chunks | 18 of 30 expert blocks on the card | 77 slots/layer, 16384-token chunks | 15 of 30 expert blocks on the card | 70 slots/layer, 8192-token chunks | 11 of 30 expert blocks on the card |
| prompt tok/s, 5k | 1,385 | 558 | 1,395 | 485 | 1,404 | 386 |
| prompt tok/s, 15k | 1,414 | 559 | 1,413 | 492 | 1,445 | 404 |
| prompt tok/s, 157k | — | — | — | — | 672 | 333 |
| TTFT 157k | — | — | — | — | 234 s | 474 s |
| decode tok/s (MTP) | 72.0 | 60.4 | 68.6 | 53.5 | 66.6 | 47.1 |
| decode tok/s at 157k | — | — | — | — | 32.1 | 24.6 |
| MTP accepted | 76 % | 77 % | 75 % | 73 % | 77 % | 78 % |
| decode right after a 15k read | -13 % | +6 % | -27 % | +5 % | -22 % | -0 % |
| VRAM load / peak | 11.2 / 11.5 GB | 11.3 / 11.3 GB | 11.3 / 11.3 GB | 11.2 / 11.3 GB | 11.2 / 11.2 GB | 11.4 / 11.4 GB |
| image projector on the card | fits (65 slots; 1 / 8 images 2.79 / 11.69 s) | fits (15 blocks; 1 / 8 images 2.34 / 10.31 s) | fits (60 slots; 1 / 8 images 2.81 / 11.63 s) | fits (12 blocks; 1 / 8 images 2.78 / 12.15 s) | fits (53 slots; 1 / 8 images 3.41 / 12.94 s) | fits (7 blocks; 1 / 8 images 3.17 / 14.86 s) |
| top-1 vs all-GPU, 5 prompts | 96.3 % | 96.4 % | 96.4 % | 96.6 % | 96.1 % | 95.9 % |
| mean KL vs all-GPU, 5 prompts | 2.0e-02 | 1.7e-02 | 2.0e-02 | 1.7e-02 | 2.0e-02 | 1.6e-02 |
| top-1 / KL vs all-GPU, 157k prompt | — | — | — | — | 95.6 % / 1.8e-02 | 94.6 % / 1.9e-02 |
| twin rows (differ) | 9,928 (0) | n/a | 9,936 (0) | n/a | 15,304 (0) | n/a |

### RTX 5070 (12 GB, 672 GB/s; PCIe x16)

| | 32k Strata-Lite | 32k llama.cpp | 131k Strata-Lite | 131k llama.cpp | 262k Strata-Lite | 262k llama.cpp |
|---|---|---|---|---|---|---|
| config | 80 slots/layer, 16384-token chunks | 18 of 30 expert blocks on the card | 75 slots/layer, 16384-token chunks | 15 of 30 expert blocks on the card | 69 slots/layer, 8192-token chunks | 11 of 30 expert blocks on the card |
| prompt tok/s, 5k | 1,949 | 380 | 1,961 | 323 | 1,943 | 259 |
| prompt tok/s, 15k | 2,079 | 402 | 2,083 | 341 | 2,161 | 271 |
| prompt tok/s, 157k | — | — | — | — | 993 | 264 |
| TTFT 157k | — | — | — | — | 159 s | 596 s |
| decode tok/s (MTP) | 111.3 | 60.5 | 94.9 | 56.4 | 100.2 | 46.3 |
| decode tok/s at 157k | — | — | — | — | 45.8 | 29.1 |
| MTP accepted | 76 % | 72 % | 70 % | 77 % | 82 % | 73 % |
| decode right after a 15k read | -24 % | -0 % | -24 % | +6 % | -13 % | -6 % |
| VRAM load / peak | 11.1 / 11.5 GB | 11.3 / 11.4 GB | 11.1 / 11.1 GB | 11.3 / 11.4 GB | 11.2 / 11.2 GB | 11.4 / 11.6 GB |
| image projector on the card | fits (62 slots; 1 / 8 images 2.2 / 8.14 s) | fits (14 blocks; 1 / 8 images 2.84 / 11.54 s) | fits (57 slots; 1 / 8 images 2.24 / 8.15 s) | fits (11 blocks; 1 / 8 images 3.17 / 14.27 s) | fits (51 slots; 1 / 8 images 2.33 / 8.15 s) | fits (7 blocks; 1 / 8 images 4.06 / 18.6 s) |
| top-1 vs all-GPU, 5 prompts | 96.7 % | 97.2 % | 96.8 % | 96.4 % | 96.8 % | 96.3 % |
| mean KL vs all-GPU, 5 prompts | 1.6e-02 | 1.3e-02 | 1.6e-02 | 1.3e-02 | 1.5e-02 | 1.3e-02 |
| top-1 / KL vs all-GPU, 157k prompt | — | — | — | — | 95.6 % / 1.8e-02 | 95.2 % / 1.7e-02 |
| twin rows (differ) | 9,848 (0) | n/a | 9,784 (0) | n/a | 15,392 (0) | n/a |
<!-- GRID:END -->

**How each cell was measured** (`lite/tests/grid.sh`, `lite/tests/grid_bench.py`, `lite/tests/grid_summary.py`):
- **Prompt speed:** three real long-form prompts of 5.4k, 15.4k and 157k tokens, sent as token ids to `/completion`; the 157k one only at 262k.
- **Decode:** 256 greedy tokens per prompt. "Decode" is the second turn on the same prompt, where the prompt cache holds it and the cache has settled. "Right after a read" is the first turn, where Strata-Lite's cache is still settling.
- **MTP accepted:** drafts accepted over drafts proposed.
- **VRAM:** sampled every 0.25 s on the card.
- **Strata-Lite configuration:** slots per layer chosen at load from free VRAM. It reads in the largest chunk the card funds. The grid found 16k up to 131k of context and 8k at 262k with the 157k prompt; the server's automatic chunk choice (`--prefill-ubatch auto`, the default) picks the same.
- **llama.cpp configuration:** `-b 2048 -ub 512`, `--spec-type draft-mtp`.
- **Image projector:** the same configuration plus the BF16 mmproj kept on the same card (`--mmproj-place gpu`; not the default, which keeps it in RAM until an image needs it). Does it load and answer one and eight 1024 px images? Strata-Lite gives up slots to fit it; llama.cpp moves more blocks to RAM.
- **Quality:** teacher-forced through the greedy tokens of llama.cpp with every expert on the GPU (q4 KV, two cards). Top-1 agreement and mean KL(ref ‖ engine) over 200 steps of five prompts (0.6k–6.8k tokens), and 500 steps of the 157k prompt at 262k.
  - The all-GPU run is the arithmetic a cache approaches as it holds more of the model.
  - llama.cpp's own placements score in the same range against it: all-CPU vs all-GPU is 96.3 % / 1.35e-2 at q4 on these prompts.
- **Twin check:** in Strata-Lite's quality runs, every routed expert of one cached layer is computed both on the GPU slot path and on the CPU pool. Every merged row must equal its twin bit for bit; "0 differ" is the result each time.

The prompts are the author's own creative-writing workloads and are not published; only numbers are. The model is
HauhauCS's Gemma 4 26B-A4B QAT Uncensored Balanced Q4_K_M GGUF with its MTP drafter and mmproj.

## Results: Qwen3.6-35B-A3B

<!-- QWEN-HEADLINE:START -->
**On the same 12 GB card, at the same context, Strata-Lite reads Qwen3.6 prompts 3–11× faster than llama.cpp. On the RTX 5070
it writes 1.6× faster; on the RTX 3060 the two write at the same speed. Top-token agreement with an all-GPU run is within about
a point of llama.cpp's.**

| per card, all three contexts | RTX 3060 | RTX 5070 |
|---|---|---|
| reading a 5k–15k prompt | 3.0–4.6× | 6.2–10.9× |
| reading the 157k prompt (time to first token) | 223 s vs 670 s | 151 s vs 971 s |
| writing (decode, MTP on in both) | 0.98–1.07× | 1.58–1.65× |
| writing at 157k of context | 34 vs 33 tok/s | 50 vs 37 tok/s |
| top-1 agreement with all-GPU, 5 prompts | 96.4–97.3 % vs 97.3–97.6 % | 96.9–97.7 % vs 96.6–97.3 % |

- **Why the 3060 writes no faster:** a Qwen3.6 token touches 3B parameters, so llama.cpp's experts on the CPU keep up with a
  3060; the card's cache wins where the card is fast (the 5070) and where many tokens go through at once (reading).
- **Twin check:** every cached row equal to its CPU twin, bit for bit, in every cell (10k–16k rows each, 0 differ).
- **The first reply after a long read** writes within a few percent of the next one (-7 % to +3 %; Gemma's dipped 13–27 %).
- **Both engines fit the image projector beside the model at every context**, when it is forced onto the card for this test (`--mmproj-place gpu`); Strata-Lite gives up 18–20 expert slots for it. By default it waits in RAM instead and visits the card only for an image ("The image projector, on demand").
<!-- QWEN-HEADLINE:END -->

The same grid, cell for cell (`lite/tests/grid_qwen.sh`): Qwen's 40 expert blocks, the same prompts tokenized for Qwen, its own
all-GPU q4 reference (llama.cpp with every expert on a 4070 Ti SUPER + 5070) and MTP on in both engines (llama.cpp with the
extracted head as `-md`, Strata-Lite with the model's own embedding and head).

<!-- GRID-QWEN:START -->
### RTX 3060 (12 GB, 360 GB/s; PCIe 3.0 x16)

| | 32k Strata-Lite | 32k llama.cpp | 131k Strata-Lite | 131k llama.cpp | 262k Strata-Lite | 262k llama.cpp |
|---|---|---|---|---|---|---|
| config | 98 slots/layer, 16384-token chunks | 14 of 40 expert blocks on the card | 89 slots/layer, 16384-token chunks | 12 of 40 expert blocks on the card | 73 slots/layer, 8192-token chunks | 9 of 40 expert blocks on the card |
| prompt tok/s, 5k | 907 | 302 | 910 | 283 | 926 | 253 |
| prompt tok/s, 15k | 1,144 | 297 | 1,241 | 280 | 1,170 | 253 |
| prompt tok/s, 157k | — | — | — | — | 707 | 235 |
| TTFT 157k | — | — | — | — | 223 s | 670 s |
| decode tok/s (MTP) | 58.3 | 54.3 | 55.3 | 53.3 | 54.5 | 55.4 |
| decode tok/s at 157k | — | — | — | — | 34.4 | 32.7 |
| MTP accepted | 66 % | 66 % | 67 % | 68 % | 64 % | 80 % |
| decode right after a 15k read | -7 % | -1 % | -1 % | -0 % | +3 % | +1 % |
| VRAM load / peak | 11.3 / 11.6 GB | 11.6 / 11.7 GB | 11.3 / 11.6 GB | 11.6 / 11.6 GB | 11.2 / 11.4 GB | 11.3 / 11.4 GB |
| image projector on the card | fits (80 slots; 1 / 8 images 4.18 / 21.34 s) | fits (11 blocks; 1 / 8 images 2.81 / 15.29 s) | fits (71 slots; 1 / 8 images 3.98 / 20.92 s) | fits (10 blocks; 1 / 8 images 2.86 / 16.08 s) | fits (55 slots; 1 / 8 images 3.92 / 20.23 s) | fits (7 blocks; 1 / 8 images 3.41 / 17.24 s) |
| top-1 vs all-GPU, 5 prompts | 97.0 % | 97.6 % | 96.4 % | 97.3 % | 97.3 % | 97.4 % |
| mean KL vs all-GPU, 5 prompts | 1.2e-02 | 1.1e-02 | 1.2e-02 | 1.1e-02 | 1.2e-02 | 1.1e-02 |
| top-1 / KL vs all-GPU, 157k prompt | — | — | — | — | 95.8 % / 1.1e-02 | 97.0 % / 1.2e-02 |
| twin rows (differ) | 10,440 (0) | n/a | 10,480 (0) | n/a | 15,736 (0) | n/a |

### RTX 5070 (12 GB, 672 GB/s; PCIe x16)

| | 32k Strata-Lite | 32k llama.cpp | 131k Strata-Lite | 131k llama.cpp | 262k Strata-Lite | 262k llama.cpp |
|---|---|---|---|---|---|---|
| config | 96 slots/layer, 16384-token chunks | 13 of 40 expert blocks on the card | 87 slots/layer, 16384-token chunks | 11 of 40 expert blocks on the card | 72 slots/layer, 8192-token chunks | 9 of 40 expert blocks on the card |
| prompt tok/s, 5k | 1,184 | 192 | 1,185 | 179 | 1,196 | 163 |
| prompt tok/s, 15k | 1,957 | 191 | 1,931 | 177 | 1,740 | 162 |
| prompt tok/s, 157k | — | — | — | — | 1,044 | 162 |
| TTFT 157k | — | — | — | — | 151 s | 971 s |
| decode tok/s (MTP) | 81.7 | 49.6 | 76.8 | 48.6 | 71.6 | 45.2 |
| decode tok/s at 157k | — | — | — | — | 50.3 | 36.9 |
| MTP accepted | 67 % | 64 % | 64 % | 71 % | 61 % | 66 % |
| decode right after a 15k read | +2 % | -1 % | -5 % | +1 % | -2 % | -1 % |
| VRAM load / peak | 11.1 / 11.6 GB | 11.3 / 11.4 GB | 11.1 / 11.6 GB | 11.1 / 11.2 GB | 11.2 / 11.5 GB | 11.5 / 11.6 GB |
| image projector on the card | fits (77 slots; 1 / 8 images 3.4 / 16.66 s) | fits (11 blocks; 1 / 8 images 3.84 / 22.76 s) | fits (68 slots; 1 / 8 images 3.47 / 16.84 s) | fits (9 blocks; 1 / 8 images 4.09 / 24.71 s) | fits (52 slots; 1 / 8 images 3.49 / 17.07 s) | fits (6 blocks; 1 / 8 images 4.62 / 27.95 s) |
| top-1 vs all-GPU, 5 prompts | 97.1 % | 96.6 % | 96.9 % | 97.3 % | 97.7 % | 97.2 % |
| mean KL vs all-GPU, 5 prompts | 1.1e-02 | 1.3e-02 | 1.2e-02 | 1.2e-02 | 1.2e-02 | 1.2e-02 |
| top-1 / KL vs all-GPU, 157k prompt | — | — | — | — | 96.2 % / 1.2e-02 | 97.4 % / 1.1e-02 |
| twin rows (differ) | 10,416 (0) | n/a | 10,296 (0) | n/a | 15,560 (0) | n/a |
<!-- GRID-QWEN:END -->

The engine's own stage-by-stage checks against llama.cpp (bit for bit where the arithmetic is the same) are in `lite/docs/QWEN36.md`.

### Why a cache works here: routing skew (Gate 0, Gemma 4; Qwen's is in `lite/docs/QWEN36.md`)

Twelve real runs (790k prompt tokens, 6.5k decoded) were replayed with every token's 8 experts per layer logged. The table gives simulated decode hit rates, with the cache warmed by the prompt.

| cache size | adaptive | LRU | best fixed set (hindsight) | fixed profile from other prompts |
|---|---|---|---|---|
| 25 % of experts | 72.3 % | 71.2 % | 72.8 % | 50.8 % |
| 37 % | 84.1 % | 81.8 % | 85.0 % | 64.6 % |
| 50 % | 92.3 % | 90.6 % | 93.3 % | 76.8 % |
| 61 % | 96.2 % | 95.5 % | 97.2 % | 84.8 % |

Routing concentrates within a conversation (about 70 of 128 experts per layer in use) but moves between conversations. An adaptive cache gets within a point of the best set chosen in hindsight; a fixed profile does not.

### Correctness, stage by stage (Gate 1, Gemma 4; Qwen's is in `lite/docs/QWEN36.md`)

Each stage was checked against llama.cpp on the same card before the next began (`lite/tests/stage_*.sh`, `lite/results/gate1*/`).

| stage | what | check | result |
|---|---|---|---|
| a | the GGUF read into the engine's layout (658 tensors, 5 quant types) | every tensor's role, type and bytes accounted for | `lite/results/gate1a-pack/check.json` |
| b | full forward, every expert on the CPU | logits vs llama.cpp, 5 prompts + 200 steps | bit-identical, 993/993 steps; greedy tokens identical; graphs equal node for node (2,644 nodes) |
| c | Strata's CPU expert pool (GeGLU, Gemma's quant formats) | same | bit-identical, 993/993 |
| d | GPU expert slots in front of the pool | whole layers resident vs `--n-cpu-moe 20`; mixed slots via the twin check | 993/993 bit-identical; 7,904 twin rows, 0 differ |
| e | adaptation (≤ 2 swaps/layer/step) | top-1 / KL vs llama.cpp's own placement noise | 98.89 % / 3.2e-3, against llama.cpp all-GPU vs all-CPU at 98.19 % / 3.6e-3 |
| f | MTP drafter (gemma4-assistant), verify windows, rollback | KL vs all-GPU with and without MTP | KL unchanged (within ±10 % per prompt); greedy output identical to e on 3 of 5 prompts, the other two part where a 4-token verify window rounds differently, as llama.cpp's MTP does; 48–97 % of drafts accepted |
| g | chunked reads with lent VRAM, copy stream, CPU/GPU overlap | twin check, five prompts | 9,864 rows, 0 differ |
| q4 KV | llama.cpp's Hadamard KV rotation mirrored | logits vs llama.cpp at q4_0 | bit-identical (195/195, 198/198, 200/200) |
| sm_86 | the RTX 3060 build | logits vs llama.cpp on the 3060 | bit-identical (195/195, 200/200) |

Where results are not bit-identical by design (e, f, g), the reason is that a cached expert computes on the GPU what llama.cpp would compute on the CPU. The bar is llama.cpp's own disagreement with itself between placements, and the twin check proves no row is wrong.

## Launch

```bash
strata-lite-server -m Gemma4-26B-A4B-...-Q4_K_M.gguf --mtp mtp-gemma-4-26B-A4B-it.gguf --mmproj mmproj-...-BF16.gguf
strata-lite-server -m Qwen3.6-35B-A3B-...-Q4_K_P.gguf --mtp mtp-qwen3.6-35b-a3b.gguf --mmproj mmproj-...-f16.gguf
```

The defaults (Qwen's differences are in `lite/docs/RUN.md`) (every one can be overridden):

| what | default |
|---|---|
| API | OpenAI `/v1/chat/completions` and llama-server's `/completion`, `/health`, `/props`, `/slots`, `/v1/models`; alias `strata-lite`, `127.0.0.1:8080`, one request at a time, prompt cache on |
| context, KV | 262,144 tokens; q8_0 K and V (Qwen: q4_0, where q8 measured no better); the card holds only the context the conversation has reached (`--kv-initial`) |
| experts | all in RAM (`--n-cpu-moe 30`), with a GPU cache in front sized from free VRAM (`--cache-slots auto`). It shrinks instead of failing when VRAM gets tight and grows back when VRAM frees. In RAM, as much is locked as the PC can spare; the rest is read from the model file (`--ram-budget`). |
| starting profile | none needed: the cache fills from the first prompt's routing (`--profile` to start from a ranked list) |
| prompt reads | in the largest chunk that fits, from 32k / 16k / 8k / 4k (`--prefill-ubatch auto`) |
| threads | the physical cores the process may run on (`-t`, `--pool-threads`) |
| sampling | llama-server's: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, no repeat penalty |
| MTP drafting | with `--mtp`: Gemma 3 drafts per step, 2 once the context passes 32K (`--n-draft`, `--n-draft-long`); Qwen 2 |
| cache behaviour | adaptive, at most 2 swaps per layer per step (`--adapt-budget`), swaps copied on a background stream, CPU misses overlapped with the GPU hits. Every switch: `lite/docs/RUN.md`. |
| image projector | `--mmproj-place`: `second` on a second GPU with room, else `auto` with 32 GB of RAM or more, else `lazy`. Images under 128 px, and every image in a prompt over 65,536 tokens, encode on the CPU. The projector leaves the card after 15 s idle. Up to 384 tokens per image. |

## The image projector, on demand

The grid's projector row keeps the projector on the card. That is what llama.cpp does, and it costs Strata-Lite 16–18 expert slots: 7–16 % of its decode speed on text-only requests. The server's default places it on demand instead (`--mmproj-place`):

| mode | where the projector lives | text-only requests |
|---|---|---|
| `second` (default when a second GPU with room is visible) | on the second GPU, whole: no slots given up, no moves, no CPU encodes. RTX 3060 + Quadro RTX 4000 at 262k: 70 slots kept, decode 64.7 tok/s (66.6 with no projector at all); a 32 px image 2.9 s, one 1024 px 3.7 s, eight 22 s | full arena |
| `auto` (default with 32 GB of RAM or more and one GPU) | in RAM. Images under `--mmproj-min-size` (128 px), and every image in a prompt over `--mmproj-gpu-max-ctx` (65,536), encode on the CPU. Bigger images move it to the card (about 0.6–0.9 s; slots give way), and it leaves after `--mmproj-idle` (15 s) | full arena |
| `lazy` (default under 32 GB of RAM and one GPU) | nowhere until an image arrives; then on the card or in RAM by the same rules; unloaded when idle | full arena, no RAM held |
| `gpu` / `cpu` | always on the card / always in RAM | fewer slots / full arena |

On an RTX 3060 at 131k: a 32 px image (a watchdog probe) encodes on the CPU in about 8 s without touching the card. One 1024 px image takes 7 s with the move, eight take 26 s. On the CPU those two would take 22 s and 150 s.

## Footnote: the author's two-card setup

Strata-Lite was built to replace a llama.cpp server on two RTX 3060s plus a Quadro for images: all experts on GPU, no MTP, 262k context. On the same requests, Strata-Lite on one 3060 at 262k with the projector on the card ran as follows (`lite/results/gate2/`):
- decode at 74–116 % of that server's speed;
- long prompts read at about half its speed;
- faster on images;
- all 63 API-parity checks passed.

Two invented test turns (`lite/tests/sameness-inputs/`) compared the two setups at greedy decoding:
- **A tool-calling assistant turn:** the same tool call with the same arguments.
- **A shot-notes turn with an image:** both returned valid JSON with 6 shots and every required field. Five of six lenses and the first framing matched; the wording differs.

## Limitations, plainly
- One request at a time; no batched slots yet.
- The first reply after a long read is 13–27 % slower while the cache settles (about half a second).
- Mean KL against an all-GPU run is slightly above llama.cpp's static offload (which experts run where), with every cached row
  proven bit-exact; see `lite/docs/CORRECTNESS.md`.
- On 12 GB a long conversation costs the cache slots as its KV grows (Gemma at 157K: 79 slots at load, 56 after); streaming the KV
  from RAM was measured and set aside (above).
- Qwen's prompt cache resumes only at checkpoints of its recurrent state (the next chat turn and a repeated prompt resume; a prompt
  that changes earlier is read again), see `lite/docs/RUN.md`.
- LFM2.5-8B and other models small enough to sit on the card whole run faster on llama.cpp's own server today.
- The setup's Windows paths (the desktop shortcut, the menu in `START-HERE.bat`) were written for Windows and tested on Linux only.
- Tested on two cards and three models, by one person, in about a week.

## Credits
- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT): the engine this is adapted from.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml: the kernels, the quant formats, the projector path and the server layer; and the reference every stage was checked against.
- HauhauCS for the Gemma 4 release and the MTP drafter (credited there to Unsloth); Google for Gemma 4.
- HauhauCS for the Qwen3.6 release; Unsloth for the Qwen3.6 MTP GGUF the drafter head is taken from; the Qwen team for Qwen3.6.
- Liquid AI for LFM2.5 and its QAD quantization; AtomicChat for the uncensored LFM2.5-24B release.

## Support
Strata-Lite exists because of Niko's [Strata](https://github.com/Niko1221/Strata). If it runs well for you, please support Niko first: [buymeacoffee.com/strataengine](https://buymeacoffee.com/strataengine). If you'd like to help with this fork's testing and builds too: [buymeacoffee.com/thed1c](https://buymeacoffee.com/thed1c).

## Layout

Strata-Lite is a fork of Strata: everything of Strata is here and works as it does upstream (its engine for Qwen3.8-Flash-Next,
its setup, server and tools; its README is `STRATA.md`). What the fork adds:
- `src/gemma4/`: the Gemma 4 engine (`g4_engine.*`), its test driver (`g4_cli.cpp`), the server (`g4_server.cpp`: the
  OpenAI / llama-server-compatible server, and the launcher that hands a Qwen GGUF to the Qwen server), `logit_cmp.cpp`;
- `src/qwen35/`: the Qwen3.6 engine, test driver and server (`strata-lite-server-qwen35`);
- `tools/qwen35_mtp_extract.py`: the Qwen MTP head into its own GGUF;
- `src/kernels/cpu/`: the CPU expert pool, extended for these models' widths and activations;
- `lite/`: the docs, the llama.cpp patch, the check and benchmark scripts (`lite/tests/`), the profile builder and the Gate 0
  simulator, and `lite/results/`: their numbers.
