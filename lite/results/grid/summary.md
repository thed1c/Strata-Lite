
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
