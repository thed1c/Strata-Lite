// Startup-only rank-one LFM output adapters. Owns validation, storage and arithmetic;
// the base GGUF and the expert-cache copies remain unchanged.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <string>
#include <unordered_map>

namespace lite {
class LfmLora {
public:
    struct Pair {
        ggml_tensor * a = nullptr, * b = nullptr;
        float scale = 0;
    };
    ~LfmLora();
    std::string load(const std::string & path, float scale, ggml_context * base, ggml_backend_t gpu);
    const Pair * find(const ggml_tensor * weight) const;
    ggml_tensor * add(ggml_context * ctx, const ggml_tensor * weight, ggml_tensor * input,
                      ggml_tensor * result, ggml_tensor * expert_ids = nullptr) const;
    size_t size() const { return pairs_.size(); }
private:
    gguf_context * file_ = nullptr;
    ggml_context * cpu_ = nullptr, * gpu_ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<std::string, Pair> pairs_;
};
}
