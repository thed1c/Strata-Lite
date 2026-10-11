
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
