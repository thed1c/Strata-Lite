# Contributing

- **No prompts, tokens, logits, routes or generated text in the repo.** Results are numbers. The benchmark scripts take your own
  prompt files; keep them out of commits (`.gitignore` blocks the usual types; check `git diff` before every commit).
- **A number needs a run.** Every speed or quality figure in a pull request points at the script that produced it and the card,
  context and model it ran on. Twin-check (`lite/tests/twin_check.sh`) any change that touches the expert path before writing a speed down.
- **Correctness before speed.** A change to the block is checked bit for bit against llama.cpp (`lite/tests/stage_b.sh` pattern); a change
  to the cache, pool, swaps or prefill is twin-checked and scored against the all-GPU reference (`lite/docs/CORRECTNESS.md`).
- **Per-architecture builds are re-proved per card.** Say which card you ran on.
- **Adding a model:** read its GGUF (architecture, tensor names, expert layout, quant types), write its descriptor, mirror llama.cpp's
  graph for its block, then walk the same gates: pack cross-check, dense forward bit for bit, all-CPU experts bit for bit, cache with
  a static profile, adaptation, MTP, chunked reads, the grid.
- Licence: contributions are MIT, like the rest.
