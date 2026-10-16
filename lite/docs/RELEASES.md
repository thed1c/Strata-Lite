# Earlier releases

The README carries the current release against llama.cpp and against the previous release. Earlier releases' sections are kept
here as they were published.

## v0.2.1 (2026-10-11): VRAM only for the conversation you are having

A 12 GB card is shared between the model's expert cache and the attention's KV cache. v0.2.0, like llama.cpp, reserved the KV
for the whole 262,144-token context at load, even for a two-line chat. v0.2.1 gives the card only what the conversation uses:

- **The KV grows with the conversation.** It starts at 32K tokens on the card and grows in 16K steps; the expert cache gives way
  as it grows and takes the room back when a new chat starts (`--kv-initial`, `--kv-step`). Same output, bit for bit, as
  reserving it all up front (checked on both models, with MTP drafting).
- **The chat page compacts at 192K.** When a conversation reaches three quarters of the context, the page has the model
  summarize it and carries on from the summary plus the last turns, with a marker you can open to read the summary. A paste of
  up to 256K still goes through in one message. So in normal use the KV tops out near 192K, not 262K.

**What that bought: Gemma back on q8 KV.** At 64K tokens of Wikipedia text, against f16 KV and a noise floor of the same run
batched differently:

| | Gemma 4 top-1 | Gemma 4 KL | Qwen3.6 top-1 | Qwen3.6 KL |
|---|---|---|---|---|
| noise floor (f16, other batching) | 87.1 % | 0.073 | 94.1 % | 0.024 |
| q8_0 KV | **87.1 %** | 0.106 | 93.4 % | 0.014 |
| q4_0 KV (v0.2.0) | 80.5 % | 0.216 | 92.2 % | 0.020 |

q4 KV cost Gemma 6.6 points of top-token agreement; q8 is back at the floor. Qwen's q4 cost is already inside its floor, so Qwen
stays on q4. Gemma's q8 KV takes twice the VRAM per token, and the growing KV is what makes that affordable:

Every column below was measured the same way on 2026-10-11, on the grid's two cards (RTX 3060 on PCIe 3.0 x16, RTX 5070 on PCIe x16), at 262K context with MTP drafting on and each engine at its own defaults (llama.cpp as in the grid: the expert blocks that fit on the card, the rest in RAM, q4 KV). The text is Wikipedia (the CC BY-SA corpus published with [llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)), so these numbers differ from the grid's further down, which used the project's own prompts. Prefills are fresh conversations (nothing cached); "writing after a 4K prompt" is the mean of the last three of four sends of two different prompts; the 157K prompt is read once and answered twice. The two accuracy rows are the KV type's cost against f16 KV at 64K (the table under "What that bought"; the engines match llama.cpp bit for bit at the same KV type). The ratio in brackets is v0.2.1 over the other column (for the first token, the other column's time over v0.2.1's).

Two things to read the tables by. **Writing at 157K swings from turn to turn** with how well MTP predicts the text (on Wikipedia about 45 % of drafts are accepted): the same engine gave 37.6 and then 28.4 tok/s on two turns, so read differences in those rows under ~25 % as level. **v0.2.0's short reads had two faults that v0.2.1 fixes:** after a previous read it sometimes fell back to 512-token chunks (Gemma's 5K mean includes one such read on each card: 499 and 382 tok/s against ~1,500 and ~2,200), and on the RTX 5070 its Qwen server ran out of VRAM on a 15K read that followed a 5K one.

### v0.2.1 vs llama.cpp

**Gemma 4 26B-A4B, RTX 3060, 262K context**

| | v0.2.1 | llama.cpp |
|---|---|---|
| KV cache | q8_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 87.1 %) | 87.1 % | 80.5 % |
| mean KL against f16 KV, 64K (floor 0.073) | 0.106 | 0.216 |
| experts on the card at load | 79 slots/layer (the KV grows from 32K) | 11 of 30 expert blocks on the card (`--n-cpu-moe 19`) |
| experts on the card after the 157K prompt | 56 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 1,510 (3.70×) | 408 |
| prefill, 15K prompt, tok/s | 1,495 (3.70×) | 404 |
| prefill, 157K prompt, tok/s | 670 (2.04×) | 329 |
| first token after the 157K prefill, s | 234 (2.04×) | 477 |
| writing after a 4K prompt, tok/s | 77.8 (1.56×) | 50.0 |
| writing at 157K, first turn, tok/s | 37.6 (1.56×) | 24.0 |
| writing at 157K, second turn, tok/s | 28.4 (1.02×) | 27.7 |

**Gemma 4 26B-A4B, RTX 5070, 262K context**

| | v0.2.1 | llama.cpp |
|---|---|---|
| KV cache | q8_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 87.1 %) | 87.1 % | 80.5 % |
| mean KL against f16 KV, 64K (floor 0.073) | 0.106 | 0.216 |
| experts on the card at load | 78 slots/layer (the KV grows from 32K) | 11 of 30 expert blocks on the card (`--n-cpu-moe 19`) |
| experts on the card after the 157K prompt | 54 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 2,234 (8.39×) | 266 |
| prefill, 15K prompt, tok/s | 2,268 (8.44×) | 269 |
| prefill, 157K prompt, tok/s | 1,000 (3.88×) | 257 |
| first token after the 157K prefill, s | 157 (3.88×) | 610 |
| writing after a 4K prompt, tok/s | 111.5 (2.29×) | 48.6 |
| writing at 157K, first turn, tok/s | 52.5 (1.35×) | 38.9 |
| writing at 157K, second turn, tok/s | 66.7 (1.80×) | 37.1 |

**Qwen3.6-35B-A3B, RTX 3060, 262K context**

| | v0.2.1 | llama.cpp |
|---|---|---|
| KV cache | q4_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 94.1 %) | 92.2 % | 92.2 % |
| mean KL against f16 KV, 64K (floor 0.024) | 0.020 | 0.020 |
| experts on the card at load | 89 slots/layer (the KV grows from 32K) | 9 of 40 expert blocks on the card (`--n-cpu-moe 31`) |
| experts on the card after the 157K prompt | 77 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 956 (3.59×) | 266 |
| prefill, 15K prompt, tok/s | 1,242 (4.63×) | 268 |
| prefill, 157K prompt, tok/s | 710 (2.92×) | 243 |
| first token after the 157K prefill, s | 221 (2.92×) | 645 |
| writing after a 4K prompt, tok/s | 58.5 (1.24×) | 47.3 |
| writing at 157K, first turn, tok/s | 28.3 (1.12×) | 25.2 |
| writing at 157K, second turn, tok/s | 29.4 (1.16×) | 25.4 |

**Qwen3.6-35B-A3B, RTX 5070, 262K context**

| | v0.2.1 | llama.cpp |
|---|---|---|
| KV cache | q4_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 94.1 %) | 92.2 % | 92.2 % |
| mean KL against f16 KV, 64K (floor 0.024) | 0.020 | 0.020 |
| experts on the card at load | 87 slots/layer (the KV grows from 32K) | 9 of 40 expert blocks on the card (`--n-cpu-moe 31`) |
| experts on the card after the 157K prompt | 75 slots/layer | (fixed) |
| prefill, 5K prompt, tok/s | 1,347 (7.73×) | 174 |
| prefill, 15K prompt, tok/s | 1,978 (11.23×) | 176 |
| prefill, 157K prompt, tok/s | 1,064 (6.17×) | 173 |
| first token after the 157K prefill, s | 148 (6.17×) | 910 |
| writing after a 4K prompt, tok/s | 73.9 (1.55×) | 47.8 |
| writing at 157K, first turn, tok/s | 45.6 (1.51×) | 30.2 |
| writing at 157K, second turn, tok/s | 48.8 (1.62×) | 30.2 |

### v0.2.1 vs v0.2.0

**Gemma 4 26B-A4B, RTX 3060, 262K context**

| | v0.2.1 | v0.2.0 |
|---|---|---|
| KV cache | q8_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 87.1 %) | 87.1 % | 80.5 % |
| mean KL against f16 KV, 64K (floor 0.073) | 0.106 | 0.216 |
| experts on the card at load | 79 slots/layer (the KV grows from 32K) | 70 slots/layer (the KV reserved for 262K) |
| experts on the card after the 157K prompt | 56 slots/layer | 60 slots/layer |
| prefill, 5K prompt, tok/s | 1,510 (1.60×) | 945 |
| prefill, 15K prompt, tok/s | 1,495 (1.01×) | 1,482 |
| prefill, 157K prompt, tok/s | 670 (1.01×) | 664 |
| first token after the 157K prefill, s | 234 (1.01×) | 236 |
| writing after a 4K prompt, tok/s | 77.8 (1.05×) | 73.9 |
| writing at 157K, first turn, tok/s | 37.6 (1.19×) | 31.5 |
| writing at 157K, second turn, tok/s | 28.4 (0.73×) | 39.0 |

**Gemma 4 26B-A4B, RTX 5070, 262K context**

| | v0.2.1 | v0.2.0 |
|---|---|---|
| KV cache | q8_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 87.1 %) | 87.1 % | 80.5 % |
| mean KL against f16 KV, 64K (floor 0.073) | 0.106 | 0.216 |
| experts on the card at load | 78 slots/layer (the KV grows from 32K) | 69 slots/layer (the KV reserved for 262K) |
| experts on the card after the 157K prompt | 54 slots/layer | 58 slots/layer |
| prefill, 5K prompt, tok/s | 2,234 (1.67×) | 1,339 |
| prefill, 15K prompt, tok/s | 2,268 (1.02×) | 2,233 |
| prefill, 157K prompt, tok/s | 1,000 (1.01×) | 986 |
| first token after the 157K prefill, s | 157 (1.01×) | 159 |
| writing after a 4K prompt, tok/s | 111.5 (0.98×) | 113.2 |
| writing at 157K, first turn, tok/s | 52.5 (0.95×) | 55.1 |
| writing at 157K, second turn, tok/s | 66.7 (0.98×) | 68.2 |

**Qwen3.6-35B-A3B, RTX 3060, 262K context**

| | v0.2.1 | v0.2.0 |
|---|---|---|
| KV cache | q4_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 94.1 %) | 92.2 % | 92.2 % |
| mean KL against f16 KV, 64K (floor 0.024) | 0.020 | 0.020 |
| experts on the card at load | 89 slots/layer (the KV grows from 32K) | 73 slots/layer (the KV reserved for 262K) |
| experts on the card after the 157K prompt | 77 slots/layer | 73 slots/layer |
| prefill, 5K prompt, tok/s | 956 (1.02×) | 937 |
| prefill, 15K prompt, tok/s | 1,242 (0.99×) | 1,250 |
| prefill, 157K prompt, tok/s | 710 (1.01×) | 700 |
| first token after the 157K prefill, s | 221 (1.01×) | 224 |
| writing after a 4K prompt, tok/s | 58.5 (1.03×) | 56.9 |
| writing at 157K, first turn, tok/s | 28.3 (0.94×) | 30.0 |
| writing at 157K, second turn, tok/s | 29.4 (1.04×) | 28.2 |

**Qwen3.6-35B-A3B, RTX 5070, 262K context**

| | v0.2.1 | v0.2.0 |
|---|---|---|
| KV cache | q4_0 | q4_0 |
| top-1 agreement with f16 KV, 64K (floor 94.1 %) | 92.2 % | 92.2 % |
| mean KL against f16 KV, 64K (floor 0.024) | 0.020 | 0.020 |
| experts on the card at load | 87 slots/layer (the KV grows from 32K) | 72 slots/layer (the KV reserved for 262K) |
| experts on the card after the 157K prompt | 75 slots/layer | 72 slots/layer |
| prefill, 5K prompt, tok/s | 1,347 | crashed: out of VRAM on a 15K read after a 5K one (fixed in v0.2.1) |
| prefill, 15K prompt, tok/s | 1,978 | crashed: out of VRAM on a 15K read after a 5K one (fixed in v0.2.1) |
| prefill, 157K prompt, tok/s | 1,064 (1.03×) | 1,034 |
| first token after the 157K prefill, s | 148 (1.03×) | 152 |
| writing after a 4K prompt, tok/s | 73.9 (1.09×) | 67.8 |
| writing at 157K, first turn, tok/s | 45.6 (1.02×) | 44.7 |
| writing at 157K, second turn, tok/s | 48.8 (0.93×) | 52.6 |

**Streaming the KV from RAM was measured and set aside.** It was on the roadmap: keep the KV in RAM past 96K and stream it to the
card. These models attend to every past token for every new one, so everything streamed crosses PCIe once per token: about 1 GB
per token for Gemma's q8 KV between 96K and 192K. Pinned RAM reaches the RTX 3060 below at 12.3 GB/s (PCIe 3.0 x16), so that is
~85 ms a token against ~40 ms for the whole step today: long chats would drop from ~24 to ~8 tok/s. Holding the KV on the card
costs a few expert slots instead (~0.7 % of decode speed each on the 3060). Streaming stays a fallback for cards where the KV
cannot fit at all (8 GB, `lite/ROADMAP.md`).

Also in v0.2.1: the projector's warm-up works on a second GPU (it was skipped), and an answer with a broken UTF-8 byte no longer
comes back as an HTTP 500.

