# Running Strata-Lite

```bash
strata-lite-server -m Gemma4-26B-A4B-...-Q4_K_M.gguf --mtp mtp-gemma-4-26B-A4B-it.gguf --mmproj mmproj-...-BF16.gguf
```
That is the whole launch: 262k context, q4 KV, every expert in RAM with a GPU cache in front sized from free VRAM, chunked prompt
reads, MTP drafting, the projector on demand, llama-server's API on `127.0.0.1:8080`.

Qwen3.6-35B-A3B is the same command with its files; the server reads the GGUF's architecture and starts the Qwen engine:
```bash
strata-lite-server -m Qwen3.6-35B-A3B-...-Q4_K_P.gguf --mtp mtp-qwen3.6-35b-a3b.gguf --mmproj mmproj-...-f16.gguf
```
Where Qwen's defaults differ: `-ctk q4_0 -ctv q4_0`, `--n-cpu-moe 40`, `--adapt-half-life 64`, the drafter uses the model's own embedding and output head
(`--mtp-own-head` to load the drafter file's), and the prompt cache keeps checkpoints of the recurrent state (below).

## Flags
| flag | default | what |
|---|---|---|
| `-m`, `--model` | | the text GGUF (required) |
| `--mtp` | off | the MTP drafter GGUF; `--n-draft N` drafts per step (3) |
| `--mmproj` | off | the image projector; `--mmproj-place second|auto|lazy|gpu|cpu`, `--mmproj-idle S` (15), `--mmproj-min-size PX` (128), `--mmproj-gpu-max-ctx N` (65536), `--image-max-tokens N` |
| `--alias` | `strata-lite` | the model name the API reports |
| `--host`, `--port` | `127.0.0.1`, `8080` | bind; keep it loopback or put a proxy with auth in front |
| `-c`, `--ctx-size` | 262144 | context; KV is q4_0 |
| `--cache-slots` | `auto` | expert slots per layer in VRAM; `auto` = what free VRAM allows, shrinking when VRAM tightens and growing back |
| `--vram-reserve-mb` | sized automatically | VRAM kept free beside the cache |
| `--n-cpu-moe` | 30 (Qwen: 40) | blocks whose experts live in RAM (all of them: the cache fronts them) |
| `--profile` | none | a ranked expert list per layer to start from |
| `--prefill-ubatch` | `auto` | prompt chunk: the largest of 32k / 16k / 8k / 4k the card funds; `--no-lend` stops the read borrowing the cache's VRAM |
| `--kv-initial`, `--kv-step` | 32768, 16384 | the KV pager: the full-attention KV (Qwen's 10 attention layers, Gemma's 5 global layers; Gemma's window layers keep their own small cache) starts with this many cells on the card and grows in steps as the conversation needs them (the expert cache gives way, and takes the room back when a new conversation starts); `--kv-initial 0` reserves the whole context at load. Same output bit for bit |
| `-ctk`, `-ctv` | `q8_0` (Gemma), `q4_0` (Qwen) | the KV cache types (`f16`, `q8_0`, `q4_0`). At 64k, against f16 KV and a noise floor of top-1 87.1 % (Gemma) / 94.1 % (Qwen): Gemma q8_0 87.1 %, q4_0 80.5 %; Qwen q8_0 93.4 %, q4_0 92.2 % (both inside its floor) |
| `--adapt-half-life` | 32 (Qwen: 64) | tokens over which usage counts halve |
| `--adapt-budget` | 2 | swaps per layer per step |
| `--no-adapt`, `--no-adapt-refill`, `--no-async-swaps`, `--no-overlap` | | the ablation switches (fixed cache; no refill after a read; swaps in the step; no CPU/GPU overlap) |
| `--no-async-swaps` | | also low-RAM mode: the experts are not copied into locked RAM but read from the model file as needed (16 GB of RAM runs, slower; keep the model on an SSD). Turned on by itself when locking fails |
| `-t`, `--pool-threads` | the cores the process may use | GPU-side threads / the CPU expert pool |
| `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty` | 0.8 / 40 / 0.95 / 0.05 / off | llama-server's sampling defaults; every request can override them |
| `--mtp-own-head` | off | Qwen: load the drafter file's embedding and output head instead of sharing the model's (0.5 GB more VRAM) |
| `--checkpoint-min-tokens` | 16384 | Qwen: prompt reads this long also checkpoint the last user message's start and 512 tokens from the end |

## What to expect
- The first prompt is a cold read; the cache fills from its routing. Later turns on the same conversation hit the prompt cache.
- The first reply after a long read writes 13–27 % slower than the next one: the read lent the cache's VRAM and the cache refills
  for about half a second. See the README's grid, row "decode right after a read".
- One request at a time. Batched slots are not in this engine yet.
- Qwen's prompt cache: a repeated prompt, or the next turn of a chat, resumes where the server left a checkpoint of the recurrent
  state (before the assistant header and before the last token; for reads of 16k tokens or more also at the last user message and
  512 tokens from the end). A prompt that differs earlier than that is read again from the start.
- Memory: the process maps the whole GGUF and pins the expert arena. Check `free -g` before raising the context on a small machine.

## Picking the context on a 12 GB card
From the grid (README): at 32k the cache holds about 60 % of the experts (111 tok/s on a 5070), at 131k about 55 %, at 262k about
50 % with 8k read chunks (100 tok/s on a 5070, 67 on a 3060). The full 262k fits with the projector on the card on both cards.
