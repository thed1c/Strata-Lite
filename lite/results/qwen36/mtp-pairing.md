# Qwen3.6-35B-A3B: the MTP drafter pairing (2026-10-10, RTX 5070 on machine A, llama.cpp `draft-mtp`, n-max 2)

llama-server, `-ngl 99 --n-cpu-moe 30 -fa on -c 32768 -ctk q4_0 -ctv q4_0 -t 16`, greedy, thinking off, three invented prompts
(an explanation, a Python function, a short description), 300 tokens each.

| target | drafter | drafts accepted | decode tok/s (mean) |
|---|---|---|---|
| HauhauCS Aggressive Q4_K_P (no MTP head of its own) | unsloth UD-Q4_K_XL's head, extracted (`strata/tools/qwen35_mtp_extract.py`, 1.62 GB) | 514/763 = 67.4 % | 47.2 |
| unsloth UD-Q4_K_XL | its own head | 503/785 = 64.1 % | 46.0 |
| HauhauCS Aggressive Q4_K_P | none | — | 30.8 |

The base-trained head drafts as well for the abliterated target as for its own model: the maintainer's first option holds.
