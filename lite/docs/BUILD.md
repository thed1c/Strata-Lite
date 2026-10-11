# Building Strata-Lite

Most people never build it: the setup (`START-HERE.bat` on Windows, `./setup.sh` on Linux) downloads the ready-made
`strata-lite-server` from this repository's release. This page is for building it yourself.

`strata-lite-server` is one program for both models. It links the parts of llama.cpp it uses (ggml with CUDA, llama, common,
the server's request handling, mtmd for images) statically, from llama.cpp at commit b6b003d, the commit every number in
`lite/` was checked against. The build fetches that commit by itself.

## Requirements
- Windows 10/11 x64 or Linux x86-64. A CPU with AVX2 (AVX-512 is used when present).
- An NVIDIA RTX 30, 40 or 50 card (sm_86 / sm_89 / sm_120), driver 580 or newer.
- The CUDA toolkit 13.0 (`nvcc`). Avoid 13.2.0 / 13.2.1 for RTX 50 cards (a known sm_120 miscompile; 13.2.2 is fixed).
- CMake 3.24+ and Ninja; on Windows, Visual Studio 2022's C++ build tools (run from an "x64 Native Tools" prompt).
- RAM to run it: Gemma 4 keeps about 17 GB of experts in locked RAM, Qwen3.6 about 24 GB.

## Build
```bash
cmake -S lite -B build-lite -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-lite --target strata-lite-server
```
The program is `build-lite/bin/strata-lite-server` (`.exe` on Windows). Options:

| option | default | what |
|---|---|---|
| `-DCMAKE_CUDA_ARCHITECTURES=...` | `86-real;89-real;120a-real` | the cards to build for (RTX 30 / 40 / 50); fewer builds faster |
| `-DSTRATA_LITE_PORTABLE=ON` | off | ggml's CPU code at an AVX2 baseline instead of this CPU's features (the release builds) |
| `-DSTRATA_LITE_LLAMA_DIR=<path>` | download | a llama.cpp b6b003d source tree already on disk |
| `-DSTRATA_LITE_TOOLS=ON` | off | also the test drivers (`strata-gemma4`, `strata-qwen35`), `logit-cmp` and `route-oracle` / `route-log` |

What the build adds to llama.cpp: one function, `common_sampler_sample_logits` (llama.cpp's sampler run on a logits vector;
the server needs it because the forward pass is the engine's), appended to `common/sampling.cpp` and `sampling.h` from
`lite/llama-hooks/`. Nothing else in llama.cpp changes. `lite/patches/` has the same change, plus the comparison tools, as
a patch for a llama.cpp checkout of your own.

A build is per card architecture: the bit-for-bit checks in `lite/tests/` are re-run on each card you build for, because
CUDA kernels differ by architecture.

## The model files
The setup downloads them. By hand:
- **Gemma 4 26B-A4B:** HauhauCS `Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP`: `...-Q4_K_M.gguf` (16.8 GB),
  `mtp-gemma-4-26B-A4B-it.gguf` (252 MB), `mmproj-...-BF16.gguf` (1.2 GB, optional).
- **Qwen3.6-35B-A3B:** HauhauCS `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`: `...-Q4_K_P.gguf` (23 GB) and its
  `mmproj-...-f16.gguf` (0.9 GB, optional); the MTP head read out of unsloth's `Qwen3.6-35B-A3B-MTP-GGUF`:
  ```bash
  python3 tools/qwen35_mtp_extract.py --no-head \
    https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf mtp-qwen3.6.gguf
  ```
  (0.5 GB read with range requests, not the 23 GB file.)

Weights are never part of this repository; they are yours to download under their own licences.

## Run
```bash
strata-lite-server -m model.gguf --mtp mtp.gguf --mmproj mmproj.gguf
```
Then open `http://127.0.0.1:8080/`. Every flag: `lite/docs/RUN.md`. NVIDIA's cuBLAS and CUDA runtime must be findable (the
toolkit's, or the `nvidia-cublas` / `nvidia-cuda-runtime` pip wheels the setup installs).

## Optional: a starting profile
The cache fills itself from the first prompt. A ranked expert list per layer (`--profile`) only shortens the first seconds:
```bash
python3 lite/tools/g4_profile.py profile.txt run1.routes run2.routes   # from route-log logs of your own prompts
```
