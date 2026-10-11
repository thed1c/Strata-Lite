# The model descriptor (design draft, not built)

Today each architecture is its own engine: `src/gemma4/g4_engine.cpp` and `src/qwen35/q35_engine.cpp`, each with its
own hyper-parameter struct, tensor lookup, expert shapes and block graph. What they share is Strata's expert machinery (arena, slot
cache, adaptation, CPU pool, chunk lending). Stage d on Qwen already moved that machinery behind a model-neutral interface: an expert
is a list of tensors, each with its own slot tensor. The descriptor finishes the job, so a third architecture is a descriptor plus a
block file, not a fork.

## What a descriptor says
Everything the shared code needs to know about a model, read from the GGUF where the GGUF has it:

| field | Gemma 4 26B-A4B | Qwen3.6-35B-A3B |
|---|---|---|
| GGUF architecture string | `gemma4` (drafter: `gemma4-assistant`) | `qwen35moe` |
| Layers, hidden size, vocab | from `gemma4.*` keys | from `qwen35moe.*` keys |
| Mixer per layer | sliding-window or full attention (per-layer KV heads) | Gated DeltaNet, or gated attention every 4th layer |
| Per-layer state | KV cells | KV cells for attention layers, conv + delta-rule state for the others |
| Routed experts: count, used, width | 128, 8, 704 | 256, 8, 512 |
| Expert tensors and quant types | fused `gate_up` and `down`, plus a per-expert `down` scale (F32); Gemma's quant formats | separate gate, up, down; Q4_K / Q6_K |
| Expert activation | GeGLU | SwiGLU |
| Shared / dense branch | a dense FFN beside the experts | one shared expert with a sigmoid gate, added to the sum |
| Norm order | Gemma's pre and post norms | Qwen's |
| Drafter | separate GGUF, own architecture | block 40 of unsloth's GGUF, extracted |
| Image projector | mmproj GGUF | mmproj GGUF |
| Defaults | adaptation half-life 32 | half-life 64 (Gate 0) |

## What stays in a block file
The graph itself. Each block mirrors llama.cpp's graph for that architecture op for op, because the bit-for-bit checks depend on it
(`lite/docs/CORRECTNESS.md`, and the three Qwen findings in `lite/docs/QWEN36.md`). A descriptor cannot express that, and should not try.

## What moves into the descriptor (in this order)
1. The `HParams` structs and their GGUF key reads, one per architecture today.
2. Expert layout: how many tensors make an expert, their types, strides and per-expert byte sizes, so the arena, slot tables, swap
   copies and CPU pool take the layout from the descriptor. The pool's kernel mode (`GeGLU` or `SwiGLU_ggml`) is a field.
3. The options the engines duplicate (`Options`, `Stats`, cache and chunk settings) as one shared header.
4. The server: one binary for every engine. Today `strata-lite-server` reads the GGUF's architecture and starts the matching
   engine's own server with the same arguments; the two servers share their request handling by copy.
5. Pack and profile tools (`lite/tools/g4_profile.py` and the Gemma pack reader), which take their expert layout from the same place.

## How it is checked
No behaviour may change. After each step, rerun the Gemma gates that touch it (`lite/tests/stage_b.sh` bit for bit, `lite/tests/twin_check.sh`,
the API parity run) and the Qwen stages b to d. A step that changes one bit of either is wrong. Gemma moves onto the neutral slot
cache only behind a full re-validation run, as the report says.

## Open questions
- Whether the second GPU projector placement and the lend/restore logic are architecture-free enough to leave the engines.
- The MTP drafter's two shapes (own GGUF versus extracted block) behind one interface.
- Where Strata's own Qwen3.8-Flash-Next engine sits: a third descriptor, or a separate engine the launcher selects.
