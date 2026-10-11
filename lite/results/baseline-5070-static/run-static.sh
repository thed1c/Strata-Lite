#!/bin/bash
# Gemma 4 26B-A4B on the RTX 5070 alone: static expert offload + MTP (baseline for the adaptive-cache experiment)
cd $SERVING || exit 1
M=models/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366; N=${1:-20}; CTX=${2:-32768}
export CUDA_VISIBLE_DEVICES=GPU-<uuid> CUDA_DEVICE_ORDER=PCI_BUS_ID
exec numactl --cpunodebind=0 --membind=0 runtimes/llama.cpp-b6b003d2cb29-sm89-sm120a/build/bin/llama-server -m $M/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf -md $M/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp --alias gemma26-5070 --host 127.0.0.1 --port 18190 -ngl 99 -fa on -c $CTX --n-cpu-moe $N -t 18 --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1 --metrics
