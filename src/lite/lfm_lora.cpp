#include "lfm_lora.hpp"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace lite {
LfmLora::~LfmLora() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (gpu_ctx_) ggml_free(gpu_ctx_);
    if (cpu_) ggml_free(cpu_);
    if (file_) gguf_free(file_);
}

std::string LfmLora::load(const std::string & path, float scale, ggml_context * base, ggml_backend_t gpu) {
    if (!std::isfinite(scale)) return "LoRA scale must be finite";
    gguf_init_params p = { false, &cpu_ };
    file_ = gguf_init_from_file(path.c_str(), p);
    if (!file_ || !cpu_) return "cannot read LoRA " + path;
    auto str = [&](const char * key) -> std::string {
        int64_t i = gguf_find_key(file_, key);
        return i >= 0 && gguf_get_kv_type(file_, i) == GGUF_TYPE_STRING ? gguf_get_val_str(file_, i) : "";
    };
    if (str("general.architecture") != "lfm2moe" || str("general.type") != "adapter" || str("adapter.type") != "lora")
        return "expected an lfm2moe LoRA adapter";
    float alpha = 0;
    int64_t ai = gguf_find_key(file_, "adapter.lora_alpha");
    if (ai >= 0) {
        if (gguf_get_kv_type(file_, ai) != GGUF_TYPE_FLOAT32) return "LoRA alpha must be f32";
        alpha = gguf_get_val_f32(file_, ai);
    }
    if (!std::isfinite(alpha) || !std::isfinite(scale * alpha)) return "LoRA alpha/scale must be finite";
    const int64_t nt = gguf_get_n_tensors(file_);
    if (nt <= 0 || nt > 1024 || nt % 2) return "invalid LoRA tensor count";
    ggml_init_params gp = { (size_t)(nt + 4) * ggml_tensor_overhead(), nullptr, true };
    gpu_ctx_ = ggml_init(gp);
    if (!gpu_ctx_) return "LoRA tensor allocation failed";
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copies;
    for (int64_t i = 0; i < nt; ++i) {
        std::string name = gguf_get_tensor_name(file_, i);
        if (name.size() < 7 || (name.substr(name.size()-7) != ".lora_a" && name.substr(name.size()-7) != ".lora_b"))
            return "unsupported LoRA tensor " + name;
        std::string target = name.substr(0, name.size()-7);
        auto * a = ggml_get_tensor(cpu_, (target + ".lora_a").c_str());
        auto * b = ggml_get_tensor(cpu_, (target + ".lora_b").c_str());
        if (!a || !b) return "incomplete LoRA pair for " + target;
        if (name.back() == 'b') continue;
        auto * w = ggml_get_tensor(base, target.c_str());
        if (!w) return "LoRA target absent from base: " + target;
        int block = -1, prefix = 0;
        if (sscanf(target.c_str(), "blk.%d.%n", &block, &prefix) != 1 || prefix == 0 || block < 0)
            return "unsupported LoRA target " + target;
        std::string leaf = target.substr(prefix);
        const bool experts = leaf == "ffn_down_exps.weight";
        if (!experts && leaf != "attn_output.weight" && leaf != "shortconv.out_proj.weight")
            return "only LFM mixer-output and expert-down LoRA targets are supported: " + target;
        if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || a->ne[1] != 1 || b->ne[0] != 1)
            return "only f32 rank-one LFM adapters are supported: " + target;
        if (a->ne[0] != w->ne[0] || b->ne[1] != w->ne[1] || a->ne[2] != w->ne[2] || b->ne[2] != w->ne[2] ||
            a->ne[3] != 1 || b->ne[3] != 1 || w->ne[3] != 1 || (!experts && w->ne[2] != 1))
            return "LoRA/base shape mismatch: " + target;
        for (auto * t : {a, b}) {
            const float * values = (const float *) t->data;
            for (int64_t j = 0; j < ggml_nelements(t); ++j) if (!std::isfinite(values[j])) return "non-finite LoRA tensor " + target;
        }
        Pair pair;
        pair.a = ggml_dup_tensor(gpu_ctx_, a); pair.b = ggml_dup_tensor(gpu_ctx_, b);
        ggml_set_name(pair.a, ggml_get_name(a)); ggml_set_name(pair.b, ggml_get_name(b));
        pair.scale = alpha ? scale * alpha : scale; // rank = 1, matching llama.cpp
        copies.emplace_back(pair.a, a); copies.emplace_back(pair.b, b);
        if (!pairs_.emplace(target, pair).second) return "duplicate LoRA target " + target;
    }
    if (pairs_.size() * 2 != (size_t)nt) return "LoRA pair count mismatch";
    buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(gpu_ctx_, ggml_backend_get_default_buffer_type(gpu));
    if (!buffer_) return "LoRA GPU allocation failed";
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto & c : copies) ggml_backend_tensor_set(c.first, c.second->data, 0, ggml_nbytes(c.first));
    fprintf(stderr, "strata-lite: LoRA %zu output sites, scale %.6g, %.2f MiB on GPU\n", pairs_.size(), scale, ggml_backend_buffer_get_size(buffer_) / 1048576.0);
    return {};
}

const LfmLora::Pair * LfmLora::find(const ggml_tensor * weight) const {
    auto i = pairs_.find(ggml_get_name(weight));
    return i == pairs_.end() || i->second.scale == 0 ? nullptr : &i->second;
}

ggml_tensor * LfmLora::add(ggml_context * ctx, const ggml_tensor * weight, ggml_tensor * input,
                          ggml_tensor * result, ggml_tensor * ids) const {
    const Pair * p = find(weight);
    if (!p) return result;
    auto * delta = ids ? ggml_mul_mat_id(ctx, p->b, ggml_mul_mat_id(ctx, p->a, input, ids), ids)
                      : ggml_mul_mat(ctx, p->b, ggml_mul_mat(ctx, p->a, input));
    return ggml_add(ctx, result, ggml_scale(ctx, delta, p->scale));
}

}
