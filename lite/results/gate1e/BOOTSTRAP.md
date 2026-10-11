# Stage e vs the ~90 % same-share pair (2026-10-10, Opus AC)

**The pair:** on the RTX 4070 Ti SUPER alone (one architecture), llama.cpp `--n-cpu-moe 3` (experts of layers 3–29 on the GPU) vs `-ot "blk\.(2[7-9])\.ffn_(up|down|gate|gate_up)_exps=CPU"` (layers 0–26 on the GPU). Each has 27 of 30 layers' experts on the GPU, 90 %. Both are teacher-forced on the five prompts' reference tokens, f16 KV, `-c 8192`. `results/noise-floor/same_share_90_4070.out`.

| | top-1 | mean KL |
|---|---|---|
| the pair (the gate for e and f, the maintainer 2026-10-10) | 98.49 % (978/993) | **2.57e-3** |
| stage e (5070, vs llama.cpp all-CPU) | 98.89 % (982/993) | 3.21e-3 |

**Paired bootstrap** over the 993 steps (resampled within each prompt, 10,000 draws):

| | mean KL | 95 % CI |
|---|---|---|
| stage e | 3.21e-3 | [2.49e-3, 4.01e-3] |
| the pair | 2.57e-3 | [1.85e-3, 3.54e-3] |
| stage e − the pair | 6.4e-4 | [−0.9e-4, 1.34e-3]; P(stage e ≤ pair) = 0.043 |

**All of the gap is one prompt.** Without p4, stage e is 2.015e-3 and the pair 2.000e-3 (793 steps). On p4, stage e is 7.92e-3 and the pair 4.81e-3. Stage e's per-step median there is 1.77e-3 against the pair's 1.09e-3, so it is broadly higher on that prompt, not a few outlier steps (its top 10 steps carry 44 % of its KL; the pair's top 10, 35 %).

In passing: llama.cpp at 90 % resident on the 16 GB card decodes at ~100 tok/s (no MTP, f16 KV, 8k).
