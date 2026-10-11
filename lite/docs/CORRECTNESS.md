# Correctness: how every number was earned

Every stage of the port was checked against llama.cpp on the same card before the next stage began. The README's "Correctness,
stage by stage" table has the results; this page is the method.

- **Bit for bit** (stages a–d, q4 KV, the sm_86 build): the engine's logits equal llama.cpp's byte for byte over 200 greedy steps on
  five prompts (0.6k–6.8k tokens), and the graphs equal node for node. One finding on the way: past about 4.6k tokens, llama.cpp's
  own output depends on how its graph is grouped for execution; both engines were run with the same grouping to compare.
- **The twin check** (stages d–g and every grid cell): in a cached layer, every routed expert is computed both on the GPU slot path
  and on the CPU pool, and every merged row must equal its twin bit for bit. Thousands of rows per run, zero differences, every time.
  It proves the cache never computes a row differently from the pool.
- **Placement noise as the bar** (stages e–g): when experts move between CPU and GPU, llama.cpp itself changes its output, since the
  two paths round differently. The bar for a cache that moves experts is therefore llama.cpp's own disagreement with itself between
  placements: top-1 agreement and KL(reference ‖ engine), teacher-forced on the same tokens. The reference is llama.cpp with every
  expert on the GPU, the arithmetic a cache approaches as it holds more of the model. The grid reports both engines against it.
- **What is not claimed:** bit-identical output in production. A cache that learns the prompt puts that prompt's experts on the GPU,
  so its output moves toward the all-GPU run and away from an all-CPU one. That is placement, not error, and the twin check is what
  separates the two.

Scripts: `lite/tests/stage_b.sh` … `lite/tests/stage_g.sh`, `lite/tests/twin_check.sh`, `lite/tests/noise_floor.sh`, `lite/tests/grid.sh`; for Qwen3.6
`lite/tests/q35_stage_b.sh` … `lite/tests/q35_stage_g.sh`, `lite/tests/q35_speed.sh`, `lite/tests/q35_q4kv.sh`, `lite/tests/grid_qwen.sh` (`lite/docs/QWEN36.md`). Numbers:
`lite/results/`. Prompts are the author's own and are not published; the scripts take any prompt files you give them.
