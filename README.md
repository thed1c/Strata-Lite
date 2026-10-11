# Strata-Lite

*A fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221: Strata's adaptive expert cache for Gemma 4 26B-A4B and
Qwen3.6-35B-A3B on one 12 GB card. Strata's own README is [`STRATA.md`](STRATA.md).*

**Experimental.** A port of [Strata](https://github.com/Niko1221/Strata)'s adaptive expert-cache engine to Gemma 4 26B-A4B and
Qwen3.6-35B-A3B, so either model runs on one 12 GB card at near-GPU speed, with MTP drafting, chunked prompt reads and the image
projector, behind a llama-server-compatible API. Built and measured in about a week; read the numbers as what they are, two cards
and two models.

| | |
|---|---|
| Models | Gemma 4 26B-A4B (HauhauCS QAT Uncensored Balanced, Q4_K_M, with its MTP drafter and projector); Qwen3.6-35B-A3B (HauhauCS Uncensored Aggressive, Q4_K_P, with its projector and the MTP head from unsloth's release, `lite/docs/QWEN36.md`) |
| Cards | 12 GB: RTX 3060 (sm_86), RTX 5070 (sm_120); built for sm_89 too. 8 GB is on the roadmap |
| Needs | Linux, NVIDIA driver 580+, CUDA 12.8/13, 32 GB RAM for Gemma, 48 GB for Qwen (64 for 262k + images), a patched llama.cpp build (`lite/docs/BUILD.md`) |
| Run | `strata-lite-server -m model.gguf --mtp mtp.gguf --mmproj mmproj.gguf`; it picks the engine from the GGUF (`lite/docs/RUN.md`) |
| API | llama-server's and OpenAI chat; 63/63 parity checks for each model (`lite/docs/API.md`) |
| Docs | `lite/docs/HOW-IT-WORKS.md`, `lite/docs/CORRECTNESS.md`, `lite/docs/QWEN36.md`, `lite/docs/MODEL-DESCRIPTOR.md`, `lite/ROADMAP.md`, `lite/CONTRIBUTING.md`, `lite/CHANGELOG.md` |
| Licence | MIT (`LICENSE`); weights are yours to download under their own terms |

Gemma 4 26B-A4B is a mixture-of-experts model: 30 blocks with 128 experts each, 8 used per token. Its experts weigh
15.1 GB, more than a 12 GB card holds. llama.cpp's answer is static offload: whole blocks of experts stay in RAM and are
computed on the CPU (`--n-cpu-moe`). Strata's answer is a per-layer GPU cache of the experts the conversation is actually
using. The cache adapts as the conversation moves, misses go to a CPU expert pool in the same step, and swaps run on a
background copy stream.

## Install

[Download Strata-Lite](https://github.com/thed1c/Strata-Lite/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the folder.

It is Strata's installer. In "Which model?" choose **Qwen3.6-35B-A3B** or **Gemma 4 26B-A4B** (Strata's own Qwen3.8-Flash-Next
models are there too). Then it asks:
- **which files:** *uncensored* (the HauhauCS releases every number here was measured with), *official* (unsloth's
  quantizations; experimental, not measured with this engine), or GGUF files you already have;
- **how much context:** 32K, 128K or 256K tokens;
- **images:** the image encoder loads only when an image arrives;
- **a Hugging Face token:** optional, press Enter (none of the files needs one).

Press Enter for the recommended answers. It downloads the model (Qwen ~24 GB, Gemma ~18 GB, eight connections at a time,
resumable) and NVIDIA's CUDA libraries, gets the ready-made engine, and starts the model. Your browser opens a chat page at
`http://127.0.0.1:8080`; the same address serves the OpenAI and llama-server APIs (`/v1/chat/completions`).

What it needs: an NVIDIA RTX 30, 40 or 50 card with 12 GB (8-11 GB runs with fewer experts on the card), driver 580 or
newer, 32 GB of RAM for Gemma and 48 GB for Qwen, and a CPU with AVX2. The first start loads ~15-22 GB into locked RAM:
the PC can be slow for a minute. Next time, run `START-HERE.bat` (or `./setup.sh`) again: it starts right away.

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
- **The first reply after a long read writes 13–27 % slower than the second.** The read lent the cache's VRAM and the cache refills while it writes; llama.cpp has no such dip.
- **Mean KL against all-GPU is slightly higher than llama.cpp's** (1.5–2.0e-2 vs 1.3–1.7e-2). The twin check shows every cached row computes exactly; the difference is which experts run on the GPU and which on the CPU.
- **Both engines fit the image projector beside the model at every context.** Strata-Lite gives up 16–18 expert slots for it; llama.cpp moves 3–4 more blocks of experts to RAM.

The grid below is the headline. Each column puts the same 12 GB card, the same context, the same prompts and the same greedy
sampling through both engines, with MTP drafting on in both:
- **Strata-Lite:** the most expert slots the card holds, sized automatically.
- **llama.cpp:** static expert offload at the most it can hold, i.e. the fewest `--n-cpu-moe` blocks that load and run.

Both use q4_0 KV at every context.

<!-- GRID:START -->
### RTX 3060 (12 GB, 360 GB/s; machine B, x16 Gen3)

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

### RTX 5070 (12 GB, 672 GB/s; machine A, x16)

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
- **Decode:** 256 greedy tokens per prompt. "Decode" is the second turn on the same prompt, where the prompt cache holds it and the cache has settled. "Right after a read" is the first turn, where Strata-Lite's arena is refilling.
- **MTP accepted:** drafts accepted over drafts proposed.
- **VRAM:** sampled every 0.25 s on the card.
- **Strata-Lite configuration:** slots per layer chosen at load from free VRAM. It reads in the largest chunk the card funds, lending the slot arena's VRAM to the read. The grid found 16k up to 131k of context and 8k at 262k with the 157k prompt; the server's automatic chunk choice (`--prefill-ubatch auto`, the default) picks the same.
- **llama.cpp configuration:** `-b 2048 -ub 512`, `--spec-type draft-mtp`.
- **Image projector:** the same configuration plus the BF16 mmproj kept on the same card (`--mmproj-place gpu`). Does it load and answer one and eight 1024 px images? Strata-Lite gives up slots to fit it; llama.cpp moves more blocks to RAM.
- **Quality:** teacher-forced through the greedy tokens of llama.cpp with every expert on the GPU (q4 KV, two cards). Top-1 agreement and mean KL(ref ‖ engine) over 200 steps of five prompts (0.6k–6.8k tokens), and 500 steps of the 157k prompt at 262k.
  - The all-GPU run is the arithmetic a cache approaches as it holds more of the model.
  - llama.cpp's own placements score in the same range against it: all-CPU vs all-GPU is 96.3 % / 1.35e-2 at q4 on these prompts.
- **Twin check:** in Strata-Lite's quality runs, every routed expert of one cached layer is computed both on the GPU slot path and on the CPU pool. Every merged row must equal its twin bit for bit; "0 differ" is the result each time.

The prompts are the author's own creative-writing workloads and are not published; only numbers are. The model is
HauhauCS's Gemma 4 26B-A4B QAT Uncensored Balanced Q4_K_M GGUF with its MTP drafter and mmproj.

## Results: Qwen3.6-35B-A3B

<!-- QWEN-HEADLINE:START -->
<!-- QWEN-HEADLINE:END -->

The same grid, cell for cell (`lite/tests/grid_qwen.sh`): Qwen's 40 expert blocks, the same prompts tokenized for Qwen, its own
all-GPU q4 reference (llama.cpp with every expert on a 4070 Ti SUPER + 5070) and MTP on in both engines (llama.cpp with the
extracted head as `-md`, Strata-Lite with the model's own embedding and head).

<!-- GRID-QWEN:START -->
<!-- GRID-QWEN:END -->

The engine's own stage-by-stage checks against llama.cpp (bit for bit where the arithmetic is the same) are in `lite/docs/QWEN36.md`.

### Why a cache works here: routing skew (Gate 0, Gemma 4; Qwen's is in `lite/docs/QWEN36.md`)

Twelve real runs (790k prompt tokens, 6.5k decoded) were replayed with every token's 8 experts per layer logged. The table gives simulated decode hit rates, with the cache warmed by the prompt.

| cache size | adaptive (decayed counts, half-life 32) | LRU | best fixed set (hindsight) | fixed profile from other prompts |
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
| e | adaptation (decayed counts, ≤ 2 swaps/layer/step) | top-1 / KL vs llama.cpp's own placement noise | 98.89 % / 3.2e-3, against llama.cpp all-GPU vs all-CPU at 98.19 % / 3.6e-3 |
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
| context, KV | 262,144 tokens, q4_0 K and V |
| experts | all in RAM (`--n-cpu-moe 30`), with a GPU cache in front sized from free VRAM (`--cache-slots auto`). It shrinks instead of failing when VRAM gets tight and grows back when VRAM frees. |
| starting profile | none needed: the cache fills from the first prompt's routing (`--profile` to start from a ranked list) |
| prompt reads | in the largest chunk that fits, from 32k / 16k / 8k / 4k (`--prefill-ubatch auto`), lending the cache's VRAM to the read |
| threads | the physical cores the process may run on (`-t`, `--pool-threads`) |
| sampling | llama-server's: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, no repeat penalty |
| MTP drafting | with `--mtp`, 3 drafts per step (`--n-draft`) |
| cache behaviour | adaptive: usage counts halve every 32 tokens (`--adapt-half-life`), at most 2 swaps per layer per step (`--adapt-budget`), refill after a prompt read, swaps copied on a background stream, CPU misses overlapped with the GPU hits. `--no-adapt`, `--no-adapt-refill`, `--no-async-swaps`, `--no-overlap` and `--no-lend` turn each off; `--no-adapt` also stops lending, since a fixed cache would not refill. |
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
- The first reply after a long read is 13–27 % slower while the cache refills (about half a second).
- Mean KL against an all-GPU run is slightly above llama.cpp's static offload (which experts run where), with every cached row
  proven bit-exact; see `lite/docs/CORRECTNESS.md`.
- On 12 GB a 262k context halves the cache's slots; KV streaming is next on the roadmap.
- Qwen's prompt cache resumes only at checkpoints of its recurrent state (the next chat turn and a repeated prompt resume; a prompt
  that changes earlier is read again), see `lite/docs/RUN.md`.
- Tested on two cards and two models, by one person, in about a week.

## Credits
- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT): the engine this is adapted from.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml: the kernels, the quant formats, the projector path and the server layer; and the reference every stage was checked against.
- HauhauCS for the Gemma 4 release and the MTP drafter (credited there to Unsloth); Google for Gemma 4.
- HauhauCS for the Qwen3.6 release; Unsloth for the Qwen3.6 MTP GGUF the drafter head is taken from; the Qwen team for Qwen3.6.

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
