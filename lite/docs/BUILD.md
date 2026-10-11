# Building Strata-Lite

Strata-Lite is a C++/CUDA engine plus a server. It links against a llama.cpp build for ggml (the CUDA kernels, the quant formats,
the vision projector path and the server's HTTP layer), so a llama.cpp checkout is built first, then the engine.

## Requirements
- Linux x86-64 (tested: Ubuntu 24.04). CPU with AVX2; AVX-512 is used when present.
- NVIDIA GPU, 12 GB or more, driver 580 or newer, CUDA toolkit 12.8 or 13 (`nvcc` on `PATH`). Tested on RTX 3060 (sm_86), RTX 4070 Ti SUPER (sm_89), RTX 5070 (sm_120).
- RAM: the model's experts live in RAM, pinned. Gemma 4 26B-A4B at Q4_K_M needs about 17 GB plus the KV cache you choose (32 GB is
  comfortable, 64 GB lets the projector and a 262k context sit in RAM too). Qwen3.6-35B-A3B at Q4_K_P needs about 24 GB (48 GB
  comfortable, 64 GB with the projector and a 262k context).
- CMake 3.24+, Ninja, a C++20 compiler, Python 3.10+ (pack and profile tools; stdlib only).

## 1. llama.cpp (the dependency)
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b6b003d        # the commit Strata-Lite was built and checked against
git apply /path/to/strata-lite/patches/llama.cpp-b6b003d-strata-lite.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  "-DCMAKE_CUDA_ARCHITECTURES=86-real;89-real;120a-real" \
  "-DGGML_CUDA_FA_QUANTS=q8_0-q8_0;q4_0-q4_0;f16-f16"
cmake --build build -j
```
The patch adds two things to that commit: `common_sampler_sample_logits` (llama.cpp's sampler on a logits vector, which the
server uses for its own forward pass) and the `route-oracle` / `route-log` examples the correctness scripts compare against
(teacher-forced logits and routing logs). Nothing in llama.cpp's own behaviour changes.

Pick the architectures you have: `86-real` RTX 30, `89-real` RTX 40, `120a-real` RTX 50. The `GGML_CUDA_FA_QUANTS` list is what
the q4 KV path needs. Keep `build/bin` and `build/tools/server` where the engine's CMake can find them (`LLAMA_BUILD`).

## 2. The engine and server
```bash
cd src/gemma4
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=/path/to/llama.cpp -DLLAMA_BUILD=/path/to/llama.cpp/build \
  "-DCMAKE_CUDA_ARCHITECTURES=86-real;89-real;120a-real"
cmake --build build -j
```
Targets: `strata-lite-server` (the server), `strata-gemma4` (the test driver used by the correctness stages), `logit-cmp`
(compares logits dumps against llama.cpp's).

The Qwen3.6 engine builds the same way from its own folder:
```bash
cd src/qwen35
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=/path/to/llama.cpp -DLLAMA_BUILD=/path/to/llama.cpp/build \
  "-DCMAKE_CUDA_ARCHITECTURES=86-real;89-real;120a-real"
cmake --build build -j
cp build/strata-lite-server-qwen35 ../gemma4/build/    # beside strata-lite-server
```
Targets: `strata-lite-server-qwen35` and `strata-qwen35` (its test driver). `strata-lite-server` is the one entry point: it reads the
GGUF's architecture and, for `qwen35moe`, starts `strata-lite-server-qwen35` from its own folder with the same arguments (or the
binary named in `STRATA_LITE_QWEN35_SERVER`).

A build is per card architecture: the bit-for-bit checks in `lite/tests/` are re-run on each card you build for, because CUDA kernels
differ by architecture.

## 3. The model files
Three GGUFs from the same release, in one folder:
- the text model (tested: `HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP`, `...-Q4_K_M.gguf`, 16.8 GB);
- the MTP drafter (`mtp-gemma-4-26B-A4B-it.gguf`, 252 MB);
- the image projector (`mmproj-...-BF16.gguf`, 1.2 GB), optional.
Weights are never part of this repository; download them under their own licence (Google's Gemma terms and the release's).

For Qwen3.6-35B-A3B: the text model (tested: HauhauCS `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`, `Q4_K_P`, 23 GB), its
`mmproj-...-f16.gguf` (0.9 GB, optional), and the MTP head extracted from unsloth's `Qwen3.6-35B-A3B-MTP-GGUF`
(`lite/docs/QWEN36.md`, "Files").

## 4. Optional: a starting profile
The cache fills itself from the first prompt. A ranked expert list per layer (`--profile`) only shortens the first seconds:
```bash
python3 lite/tools/g4_profile.py profile.txt run1.routes run2.routes   # from route logs of your own prompts (lite/tests/, Gate 0)
```
Profiles are aggregate counts of whoever made them; the repository ships none.

## 5. Check the build
```bash
lite/tests/twin_check.sh      # every cached row equals its CPU twin, bit for bit (needs the model files)
lite/tests/api_parity.py      # the server against a llama-server on the same model: 63 checks
```
