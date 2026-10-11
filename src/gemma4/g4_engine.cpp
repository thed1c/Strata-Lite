// Strata-Lite gemma4 engine, Gate 1 stage b. See g4_engine.hpp. Every graph-building step below names the llama.cpp
// function it mirrors (b6b003d: src/models/gemma4.cpp, src/llama-graph.cpp, src/llama-kv-cache.cpp); the order of the
// ggml calls is kept, because the CUDA backend fuses adjacent nodes and the scheduler splits by node order.
#include "g4_engine.hpp"

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml-cpu.h"

#include <cuda_runtime.h>
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../lite/platform.hpp"

namespace g4 {

static constexpr int   kMaxNodes   = 8192;
// rope parameters as llama.cpp's cparams resolve them for this model (checked against its graph's op_params)
static constexpr int   kRopeType   = GGML_ROPE_TYPE_NEOX;
static constexpr float kExtFactor  = 0.0f;
static constexpr float kAttnFactor = 1.0f;
static constexpr float kBetaFast   = 32.0f;
static constexpr float kBetaSlow   = 1.0f;
// ggml-cuda offloads an op on CPU-resident weights to the GPU from this many tokens (ggml_backend_cuda_device_offload_op):
// below it llama.cpp computes the experts on the CPU, and so does the pool; above it both keep ggml's (offloaded) path
static constexpr int   kOffloadMin = 32;
// the slot path's misses all point at one zero slot, so a token's ids repeat; ggml-cuda's mat-vec MoE kernel (<= 8 tokens)
// takes that, its batched MMQ kernel (more tokens) assumes distinct ids per token and overruns. Ubatches of 9-31 tokens
// (short prompts, prompt tails) therefore go through the pool for every expert (stage c), exactly as llama.cpp's CPU MoE.
static constexpr int   kSlotMaxTokens = 8;
static constexpr int   kMinSlots = 8;          // below this the arena is not worth its VRAM
// A small step's compute buffer grows with the context (a 4-token verify window at 157k cells took 620 MiB on sm_86/sm_120,
// ~4.1 KB per cell); ggml's allocator crashes rather than fails when it cannot get it, so the room is made beforehand.
static constexpr size_t kComputePerCell = 4608;
static constexpr int    kHeadroomStep   = 8192;   // cells of context growth covered per check
// Auto chunk: a read's compute buffers ~ fixed + per token + the KQ mask (chunk x context, f16). Calibrated on the grid
// (RTX 5070 and 3060, q4 KV): 16k chunks fit at 32k/131k contexts, at 262k a 157k prompt takes 8k, not 16k.
static constexpr size_t kChunkFixed  = (size_t) 768 << 20;
static constexpr size_t kChunkPerTok = 200 * 1024;
static constexpr int    kChunkMin    = 4096;

namespace scpu = strata::kernels::cpu;

// llama-kv-cache.cpp ggml_gen_hadamard: the orthonormal Walsh-Hadamard matrix (H*H = I), n a power of two
static std::vector<float> gen_hadamard(int n) {
    std::vector<float> d((size_t) n * n);
    d[0] = 1.0 / sqrtf(n);
    for (int s = 1; s < n; s *= 2)
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++) {
                const float v = d[i * n + j];
                d[(i + s) * n + j] = v; d[i * n + (j + s)] = v; d[(i + s) * n + (j + s)] = -v;
            }
    return d;
}
// llama-impl.h llama_mul_mat_hadamard
static ggml_tensor * mul_mat_hadamard(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const int64_t n = rot->ne[0];
    ggml_tensor * res = ggml_is_contiguous(cur) ? ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n)
                                                : ggml_cont_2d(ctx, cur, n, ggml_nelements(cur) / n);
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}
// build_input_k_rot: the largest power of two (from 64) dividing the cache's head size; V always 64
static int k_rot_size(int head) { int r = 64; do { r *= 2; } while (head % r == 0); return r / 2; }

// Strata's ExpertPool is not reentrant: the overlap driver thread and a pool op on the host thread (the twin check's
// all-CPU twin) take turns on it
static std::mutex g_pool_mu;

// ---------------------------------------------------------------- stage c: the MoE layer through Strata's pool
struct Engine::PoolLayer {
    scpu::NativeFmt f;
    const uint8_t * gu = nullptr;  size_t gu_stride = 0;   // expert e's gate+up rows: gu + e*gu_stride (fused tensor)
    const uint8_t * dn = nullptr;  size_t d_stride = 0;    // its down rows
    const float * scale = nullptr;                         // ffn_down_exps.scale, one F32 per expert
    scpu::ExpertPool * pool = nullptr;
    const int * res = nullptr;                             // stage d: slot of each expert, -1 = not resident (nullptr: none)
    Stats * stats = nullptr;
    // scratch, reused every call
    std::vector<uint8_t> act;                              // n_tokens quantized activations
    std::vector<scpu::ExpertJobMulti> jobs;
    // overlap: the misses' rows, written by the pool on the driver thread, copied out by the join op
    std::vector<float> out;
    std::mutex mu; std::condition_variable cv; bool done = true;
    struct Engine::Driver * driver = nullptr;
};

// overlap: one thread that runs a layer's miss jobs through the pool, so the host thread can launch the GPU hits
struct Engine::Driver {
    std::mutex mu; std::condition_variable cv; Engine::PoolLayer * task = nullptr; bool stop = false;
    std::thread th;
    Driver() : th([this] { loop(); }) {}
    ~Driver() { { std::lock_guard<std::mutex> g(mu); stop = true; } cv.notify_all(); th.join(); }
    void submit(Engine::PoolLayer * L) { { std::lock_guard<std::mutex> g(mu); task = L; } cv.notify_all(); }
    void loop() {
        for (;;) {
            Engine::PoolLayer * L;
            { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return stop || task; }); if (stop) return; L = task; task = nullptr; }
            const auto tp0 = std::chrono::steady_clock::now();
            { std::lock_guard<std::mutex> g(g_pool_mu); L->pool->run_split_multi_native(L->f, L->jobs.data(), (int) L->jobs.size()); }
            if (L->stats) L->stats->t_pool += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
            { std::lock_guard<std::mutex> g(L->mu); L->done = true; }
            L->cv.notify_all();
        }
    }
};

// overlap, start: the routing and the activations are here (the scheduler synced the router for them); quantize, plan
// the misses into L->out, hand them to the driver thread and return at once. dst is a 1-element token for ordering.
static void moe_start_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth; (void) dst;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1];
    const int64_t n = x->ne[1], k = ids->ne[0], ne = x->ne[0];
    const size_t ab = L->f.act_bytes;
    L->act.resize((size_t) n * ab);
    L->out.assign((size_t) n * k * ne, 0.0f);
    auto id_at = [&](int64_t t, int64_t j) { return *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]); };
    L->jobs.clear();
    std::vector<int> job_of(1024, -1);
    bool quantized = false;
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            const int e = id_at(t, j);
            L->stats->accesses++;
            if (L->res[e] >= 0) { L->stats->hits++; continue; }
            if (!quantized) {
                for (int64_t u = 0; u < n; ++u)
                    scpu::native_quant_act(L->f, (const float *) ((const char *) x->data + u * x->nb[1]), L->act.data() + u * ab);
                quantized = true;
            }
            int ji = job_of[e];
            if (ji < 0 || L->jobs[ji].nt == scpu::MAXT) {
                scpu::ExpertJobMulti jb;
                jb.blob = L->gu + (size_t) e * L->gu_stride;
                jb.down = L->dn + (size_t) e * L->d_stride;
                L->jobs.push_back(jb);
                ji = job_of[e] = (int) L->jobs.size() - 1;
            }
            auto & jb = L->jobs[ji];
            jb.nact[jb.nt] = L->act.data() + t * ab;
            jb.out[jb.nt] = L->out.data() + ((size_t) t * k + j) * ne;
            jb.nt++;
        }
    }
    L->stats->ov_layers++;
    if (L->jobs.empty()) { L->stats->ov_missfree++; std::lock_guard<std::mutex> g(L->mu); L->done = true; return; }
    { std::lock_guard<std::mutex> g(L->mu); L->done = false; }
    L->driver->submit(L);
}

// overlap, join: wait for the driver (the GPU has been computing the hits meanwhile), then hand over the miss rows
static void moe_join_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    const auto t0 = std::chrono::steady_clock::now();
    { std::unique_lock<std::mutex> lk(L->mu); L->cv.wait(lk, [&] { return L->done; }); }
    L->stats->t_join_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    memcpy(dst->data, L->out.data(), ggml_nbytes(dst));
}

// x [n_embd, n] f32, ids [k, n] i32 (argsort_top_k view) -> dst [n_embd, k, n]: exactly what build_moe_ffn's
// mul_mat_id (gate_up) -> geglu -> mul_mat_id (down) gives on ggml-cpu (the part llama.cpp runs on the CPU for a
// RAM-resident layer below the offload size), with the same quantizers, dot products and GELU table. The per-expert
// scale, the router weights and the sum over k stay ggml ops after it, where the scheduler put them in stage b (GPU).
static void moe_pool_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1];
    const int64_t n = x->ne[1], k = ids->ne[0];
    const size_t ab = L->f.act_bytes;
    L->act.resize((size_t) n * ab);
    for (int64_t t = 0; t < n; ++t)
        scpu::native_quant_act(L->f, (const float *) ((const char *) x->data + t * x->nb[1]), L->act.data() + t * ab);
    auto id_at = [&](int64_t t, int64_t j) { return *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]); };
    // one job per (expert, up to MAXT of its tokens), experts in order of first use
    L->jobs.clear();
    std::vector<int> job_of(1024, -1);
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            const int e = id_at(t, j);
            if (L->res) {   // stage d: a resident expert's row comes from the GPU; this side adds zeros
                L->stats->accesses++;
                if (L->res[e] >= 0) {
                    L->stats->hits++;
                    memset((char *) dst->data + t * dst->nb[2] + j * dst->nb[1], 0, dst->ne[0] * sizeof(float));
                    continue;
                }
            }
            int ji = job_of[e];
            if (ji < 0 || L->jobs[ji].nt == scpu::MAXT) {
                scpu::ExpertJobMulti jb;
                jb.blob = L->gu + (size_t) e * L->gu_stride;
                jb.down = L->dn + (size_t) e * L->d_stride;
                L->jobs.push_back(jb);
                ji = job_of[e] = (int) L->jobs.size() - 1;
            }
            auto & jb = L->jobs[ji];
            jb.nact[jb.nt] = L->act.data() + t * ab;
            jb.out[jb.nt] = (float *) ((char *) dst->data + t * dst->nb[2] + j * dst->nb[1]);
            jb.nt++;
        }
    }
    const auto tp0 = std::chrono::steady_clock::now();
    if (!L->jobs.empty()) { std::lock_guard<std::mutex> g(g_pool_mu); L->pool->run_split_multi_native(L->f, L->jobs.data(), (int) L->jobs.size()); }
    if (L->stats) L->stats->t_pool += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
}

struct Engine::CacheLayer {
    bool full = false;                  // every expert resident, identity order: exactly llama.cpp's GPU-layer graph
    int  S = 0;                         // resident slots (the arena has S + 1: slot S is all zeros, where misses point)
    ggml_tensor * gu = nullptr, * dn = nullptr, * res_f = nullptr, * scale = nullptr;
    std::vector<int> res;               // expert -> slot, -1 = in RAM only
    std::vector<int> owner;             // slot -> expert (stage e)
    std::vector<double> score;          // decayed usage counts (stage e)
    std::vector<char> pending;          // expert with a copy in flight (async swaps)
    std::unique_ptr<PoolLayer> miss;    // the pool for the misses (res set)
    ggml_tensor * gu_all = nullptr, * dn_all = nullptr;   // dcheck twin: the whole layer on the GPU
};


// ---------------------------------------------------------------- GGUF helpers
static int64_t kv_int(gguf_context * g, const std::string & key, int64_t def = -1) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(g, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(g, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(g, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(g, id);
        default: return def;
    }
}
static float kv_f32(gguf_context * g, const std::string & key, float def = 0.0f) {
    const int64_t id = gguf_find_key(g, key.c_str());
    return id < 0 ? def : gguf_get_val_f32(g, id);
}
static std::vector<int64_t> kv_arr(gguf_context * g, const std::string & key) {
    std::vector<int64_t> r;
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return r;
    const size_t n = gguf_get_arr_n(g, id);
    const void * d = gguf_get_arr_data(g, id);
    switch (gguf_get_arr_type(g, id)) {
        case GGUF_TYPE_BOOL: case GGUF_TYPE_INT8: case GGUF_TYPE_UINT8:
            for (size_t i = 0; i < n; ++i) r.push_back(((const int8_t *) d)[i]); break;
        case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; ++i) r.push_back(((const int32_t *) d)[i]); break;
        case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; ++i) r.push_back(((const uint32_t *) d)[i]); break;
        default: break;
    }
    return r;
}

// ---------------------------------------------------------------- KV cells (llama_kv_cache, one sequence)
uint32_t Engine::Cache::used() const {
    return (uint32_t) std::count_if(pos.begin(), pos.end(), [](int32_t p) { return p >= 0; });
}
uint32_t Engine::Cache::used_max_p1() const {
    for (int64_t i = (int64_t) pos.size() - 1; i >= 0; --i) if (pos[i] >= 0) return (uint32_t) i + 1;
    return 0;
}
int32_t Engine::Cache::pos_min() const {
    int32_t m = INT32_MAX; bool any = false;
    for (int32_t p : pos) if (p >= 0) { m = std::min(m, p); any = true; }
    return any ? m : -1;
}
int32_t Engine::Cache::pos_max() const {
    int32_t m = -1;
    for (int32_t p : pos) m = std::max(m, p);
    return m;
}
// llama_kv_cache::find_slot(ubatch, cont = false)
bool Engine::Cache::find_slot(uint32_t n_tokens, int n_swa) {
    idxs.clear();
    uint32_t head_cur = head;
    if (head_cur > used() + 2 * n_tokens) head_cur = 0;
    if (n_tokens > size) return false;
    const int32_t seq_max = pos_max();
    uint32_t n_tested = 0;
    while (true) {
        if (head_cur + 1 > size) { n_tested += size - head_cur; head_cur = 0; continue; }
        const uint32_t idx = head_cur++;
        n_tested++;
        bool can_use = pos[idx] < 0;
        if (!can_use && swa && (seq_max + 1) - pos[idx] >= n_swa) can_use = true;   // is_masked_swa(STANDARD)
        if (can_use) idxs.push_back(idx);
        if (idxs.size() == n_tokens) break;
        if (n_tested >= size) return false;
    }
    return true;
}
// llama_kv_cache::seq_rm(seq, p0, -1): free every cell at p0 or later; the first freed cell becomes the head if lower
void Engine::Cache::seq_rm(int32_t p0) {
    uint32_t new_head = size;
    for (uint32_t i = 0; i < size; ++i) {
        if (pos[i] >= p0) { pos[i] = -1; if (new_head == size) new_head = i; }
    }
    if (new_head != size && new_head < head) head = new_head;
}
uint32_t Engine::Cache::calc_n_kv() const {
    const uint32_t pad = 256, um = used_max_p1();
    return std::min(size, std::max(pad, (um + pad - 1) / pad * pad));
}

// llama_kv_cache::apply_ubatch + get_n_kv
void Engine::Cache::apply(const std::vector<int32_t> & positions) {
    int32_t max_rm = -1;
    for (size_t i = 0; i < idxs.size(); ++i) {
        const int64_t idx = idxs[i];
        if (pos[idx] >= 0) { max_rm = std::max(max_rm, pos[idx]); pos[idx] = -1; }
        pos[idx] = positions[i];
    }
    if (max_rm >= 0) {
        const int32_t pmin = pos_min();
        if (pmin <= max_rm) for (auto & p : pos) if (p >= pmin && p < max_rm + 1) p = -1;
    }
    head = (uint32_t) idxs.back() + 1;
    const uint32_t pad = 256;
    const uint32_t um = used_max_p1();
    n_kv = std::min(size, std::max(pad, (um + pad - 1) / pad * pad));
}

// ---------------------------------------------------------------- load
Engine::Engine() = default;

Engine::~Engine() {
    driver_.reset();
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); cudaStreamDestroy((cudaStream_t) copy_stream_); }
    for (auto & b : pending_) cudaEventDestroy((cudaEvent_t) b.ev);
    if (buf_arena_) ggml_backend_buffer_free(buf_arena_);
    if (arena_) cudaFreeHost(arena_);
    if (sched_) ggml_backend_sched_free(sched_);
    if (buf_kv_) ggml_backend_buffer_free(buf_kv_);
    for (KvLayer & kl : kvl_) { if (kl.buf) ggml_backend_buffer_free(kl.buf); if (kl.ctx) ggml_free(kl.ctx); }
    if (buf_gpu_) ggml_backend_buffer_free(buf_gpu_);
    if (buf_map_) ggml_backend_buffer_free(buf_map_);
    if (ctx_kv_) ggml_free(ctx_kv_);
    if (ctx_w_) ggml_free(ctx_w_);
    if (meta_) ggml_free(meta_);
    if (gguf_) gguf_free(gguf_);
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
    lite::unmap_file(map_, map_size_);
}

std::string Engine::load(const Options & o) {
    o_ = o;
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta_ };
    gguf_ = gguf_init_from_file(o.gguf.c_str(), gp);
    if (!gguf_) return "cannot read " + o.gguf;
    const std::string A = "gemma4.";
    hp_.n_layer       = (int) kv_int(gguf_, A + "block_count");
    hp_.n_embd        = (int) kv_int(gguf_, A + "embedding_length");
    hp_.n_head        = (int) kv_int(gguf_, A + "attention.head_count");
    hp_.n_ff          = (int) kv_int(gguf_, A + "feed_forward_length");
    hp_.n_ff_exp      = (int) kv_int(gguf_, A + "expert_feed_forward_length");
    hp_.n_expert      = (int) kv_int(gguf_, A + "expert_count");
    hp_.n_expert_used = (int) kv_int(gguf_, A + "expert_used_count");
    hp_.n_swa         = (int) kv_int(gguf_, A + "attention.sliding_window");
    hp_.head_dim      = (int) kv_int(gguf_, A + "attention.key_length");
    hp_.head_dim_swa  = (int) kv_int(gguf_, A + "attention.key_length_swa");
    hp_.n_rot         = (int) kv_int(gguf_, A + "rope.dimension_count");
    hp_.n_rot_swa     = (int) kv_int(gguf_, A + "rope.dimension_count_swa");
    hp_.n_ctx_train   = (int) kv_int(gguf_, A + "context_length");
    hp_.rope_base     = kv_f32(gguf_, A + "rope.freq_base");
    hp_.rope_base_swa = kv_f32(gguf_, A + "rope.freq_base_swa");
    hp_.eps           = kv_f32(gguf_, A + "attention.layer_norm_rms_epsilon");
    hp_.softcap       = kv_f32(gguf_, A + "final_logit_softcapping");
    for (auto v : kv_arr(gguf_, A + "attention.head_count_kv")) hp_.n_head_kv.push_back((int) v);
    for (auto v : kv_arr(gguf_, A + "attention.sliding_window_pattern")) hp_.is_swa.push_back(v != 0);
    {
        const int64_t id = gguf_find_key(gguf_, "tokenizer.ggml.tokens");
        hp_.n_vocab = id < 0 ? 0 : (int) gguf_get_arr_n(gguf_, id);
    }
    if (hp_.n_layer <= 0 || (int) hp_.n_head_kv.size() != hp_.n_layer || (int) hp_.is_swa.size() != hp_.n_layer)
        return "unexpected gemma4 hparams";

    // backends: [CUDA0, CPU] and the CPU's compute buffers in the GPU's pinned host memory, as llama_context does
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) return "no GPU device";
    gpu_ = ggml_backend_dev_init(gpu_dev, nullptr);
    cudaGetDevice(&cuda_dev_);   // the first GPU device = CUDA device 0 of the visible set
    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu_ || !cpu_) return "backend init failed";
    ggml_backend_cpu_set_n_threads(cpu_, o.n_threads);

    // the file, mapped: the CPU-resident weights are used in place (llama.cpp with mmap: no repacking)
    map_ = lite::map_file(o.gguf, &map_size_);
    if (!map_) return "cannot map " + o.gguf;
    const size_t data_off = gguf_get_data_offset(gguf_);

    // which tensors live where: `-ngl 99 --n-cpu-moe N` = token_embd and the first N layers' experts in RAM, the rest
    // (including the tied output copy of token_embd) on the GPU
    const int n_tensors = (int) gguf_get_n_tensors(gguf_);
    ggml_init_params ip = { (size_t) (n_tensors + 4) * ggml_tensor_overhead(), nullptr, true };
    ctx_w_ = ggml_init(ip);
    ggml_context * ctx_gpu = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, size_t>> on_cpu, on_gpu;   // tensor, file offset
    auto make = [&](const std::string & name, bool cpu, const char * as = nullptr) -> ggml_tensor * {
        ggml_tensor * m = ggml_get_tensor(meta_, name.c_str());
        if (!m) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(cpu ? ctx_w_ : ctx_gpu, m);
        ggml_set_name(t, as ? as : name.c_str());
        const size_t off = data_off + gguf_get_tensor_offset(gguf_, gguf_find_tensor(gguf_, name.c_str()));
        (cpu ? on_cpu : on_gpu).push_back({ t, off });
        return t;
    };
    auto is_exps = [&](int il) { return il < o.n_cpu_moe; };
    tok_embd_    = make("token_embd.weight", true);
    output_      = make("token_embd.weight", false);
    output_norm_ = make("output_norm.weight", false);
    rope_freqs_  = make("rope_freqs.weight", false);
    L_.resize(hp_.n_layer);
    for (int il = 0; il < hp_.n_layer; ++il) {
        auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
        Layer & l = L_[il];
        l.attn_norm      = make(b("attn_norm.weight"), false);
        l.wq             = make(b("attn_q.weight"), false);
        l.wk             = make(b("attn_k.weight"), false);
        l.wv             = make(b("attn_v.weight"), false);          // absent on global layers: V = K
        l.wo             = make(b("attn_output.weight"), false);
        l.q_norm         = make(b("attn_q_norm.weight"), false);
        l.k_norm         = make(b("attn_k_norm.weight"), false);
        l.attn_post_norm = make(b("post_attention_norm.weight"), false);
        l.out_scale      = make(b("layer_output_scale.weight"), false);
        l.ffn_norm       = make(b("ffn_norm.weight"), false);
        l.ffn_gate       = make(b("ffn_gate.weight"), false);
        l.ffn_up         = make(b("ffn_up.weight"), false);
        l.ffn_down       = make(b("ffn_down.weight"), false);
        l.ffn_post_norm  = make(b("post_ffw_norm.weight"), false);
        l.gate_inp       = make(b("ffn_gate_inp.weight"), false);
        l.gate_inp_s     = make(b("ffn_gate_inp.scale"), false);
        l.pre_norm_2     = make(b("pre_ffw_norm_2.weight"), false);
        l.post_norm_1    = make(b("post_ffw_norm_1.weight"), false);
        l.post_norm_2    = make(b("post_ffw_norm_2.weight"), false);
        l.gate_up_exps   = make(b("ffn_gate_up_exps.weight"), is_exps(il));
        l.down_exps      = make(b("ffn_down_exps.weight"), is_exps(il));
        l.down_exps_s    = make(b("ffn_down_exps.scale"), is_exps(il));
        if (!l.attn_norm || !l.wq || !l.wk || !l.wo || !l.gate_inp || !l.gate_up_exps || !l.down_exps)
            return "layer " + std::to_string(il) + ": missing tensors";
    }
    // GPU weights: one buffer, filled from the mapping
    buf_gpu_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gpu, ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_gpu_) return "GPU weight allocation failed";
    ggml_backend_buffer_set_usage(buf_gpu_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // the scheduler places ops by their weights
    for (auto & [t, off] : on_gpu) ggml_backend_tensor_set(t, (const char *) map_ + off, 0, ggml_nbytes(t));
    // CPU weights: views into the mapping
    buf_map_ = ggml_backend_cpu_buffer_from_ptr(map_, map_size_);
    ggml_backend_buffer_set_usage(buf_map_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // with async swaps the RAM-resident experts live in a page-locked arena (Strata's expert arena): the copy stream
    // needs pinned sources, and a read-only file mapping cannot be registered. Same bytes, so the arithmetic is unchanged.
    // Low-RAM mode: when the OS will not lock that much (too little free RAM), the experts stay in the file mapping and the
    // OS reads them in as needed; swaps are then copied in the step (slower, but it runs). --no-async-swaps chooses it up front.
    auto in_arena = [&](ggml_tensor * t) { return o_.async_swaps && strstr(ggml_get_name(t), "_exps.") != nullptr; };
    if (o_.async_swaps) {
        for (auto & [t, off] : on_cpu) if (in_arena(t)) arena_size_ += GGML_PAD(ggml_nbytes(t), 4096);
        if (cudaHostAlloc(&arena_, arena_size_, cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            fprintf(stderr, "strata-lite: not enough free RAM to lock the experts (%.1f GB): low-RAM mode - they stay in the model file and "
                            "are read in as needed (slower)\n", arena_size_ / 1e9);
            arena_ = nullptr; arena_size_ = 0; o_.async_swaps = false;
        }
    }
    if (o_.async_swaps) {
        buf_arena_ = ggml_backend_cpu_buffer_from_ptr(arena_, arena_size_);
        ggml_backend_buffer_set_usage(buf_arena_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        pinned_ = true;
    }
    size_t aoff = 0;
    for (auto & [t, off] : on_cpu) {
        if (in_arena(t)) {
            char * dst = (char *) arena_ + aoff;
            memcpy(dst, (const char *) map_ + off, ggml_nbytes(t));
            if (ggml_backend_tensor_alloc(buf_arena_, t, dst) != GGML_STATUS_SUCCESS) return "arena alloc failed";
            aoff += GGML_PAD(ggml_nbytes(t), 4096);
        } else if (ggml_backend_tensor_alloc(buf_map_, t, (char *) map_ + off) != GGML_STATUS_SUCCESS) return "map alloc failed";
    }
    // ctx_gpu's tensors stay alive with the context; keep it by chaining (freed with ctx_w_ at exit is fine here)
    (void) ctx_gpu;

    // KV caches (llama_kv_cache_iswa): base = the global layers, n_ctx cells; swa = the window layers,
    // GGML_PAD(min(n_ctx, n_swa + n_ubatch), 256) cells; K and V [n_embd_gqa, cells], V not transposed (flash attention)
    base_.swa = false; swa_.swa = true;
    base_.size = (uint32_t) ((o.n_ctx + 255) / 256 * 256);
    swa_.size  = (uint32_t) ((std::min<uint32_t>(base_.size, hp_.n_swa + o.n_ubatch) + 255) / 256 * 256);
    base_.pos.assign(base_.size, -1); swa_.pos.assign(swa_.size, -1);
    base_.k.assign(hp_.n_layer, nullptr); base_.v.assign(hp_.n_layer, nullptr);
    swa_.k.assign(hp_.n_layer, nullptr);  swa_.v.assign(hp_.n_layer, nullptr);
    ggml_init_params kp = { (size_t) (2 * hp_.n_layer + 4) * ggml_tensor_overhead(), nullptr, true };
    ctx_kv_ = ggml_init(kp);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_swa[il]) continue;   // the global layers: the KV pager below
        const int64_t n_gqa = (int64_t) hp_.head_dim_swa * hp_.n_head_kv[il];
        swa_.k[il] = ggml_new_tensor_3d(ctx_kv_, o.type_k, n_gqa, swa_.size, 1);
        swa_.v[il] = ggml_new_tensor_3d(ctx_kv_, o.type_v, n_gqa, swa_.size, 1);
        ggml_format_name(swa_.k[il], "cache_k_l%d", il);
        ggml_format_name(swa_.v[il], "cache_v_l%d", il);
    }
    buf_kv_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_kv_, ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_kv_) return "KV allocation failed";
    ggml_backend_buffer_clear(buf_kv_, 0);
    // the KV pager: each global layer's K/V in its own buffer of kv_cap_ cells
    kv_cap_ = o.kv_initial > 0 ? std::min<uint32_t>(base_.size, (uint32_t) ((o.kv_initial + 255) / 256 * 256)) : base_.size;
    kvl_.assign(hp_.n_layer, {});
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) continue;
        const std::string e = kv_alloc_layer(il, kv_cap_);
        if (!e.empty()) return e;
    }

    if (o.pool) {   // stage c: Strata's CPU expert pool for the RAM-resident layers
        pool_ = std::make_unique<scpu::ExpertPool>(o.pool_threads, /*pin*/ false);   // never pin: machine A runs production
        pool_layers_.resize(hp_.n_layer);
        for (int il = 0; il < hp_.n_layer && il < o.n_cpu_moe; ++il) {
            const Layer & l = L_[il];
            auto pl = std::make_unique<PoolLayer>();
            std::string e;
            if (!scpu::native_fmt(l.gate_up_exps->type, l.down_exps->type, hp_.n_embd, hp_.n_ff_exp, pl->f, e,
                                  scpu::NativeAct::GeGLU, /*separate_down*/ true))
                return "layer " + std::to_string(il) + ": " + e;
            pl->gu = (const uint8_t *) l.gate_up_exps->data; pl->gu_stride = l.gate_up_exps->nb[2];
            pl->dn = (const uint8_t *) l.down_exps->data;    pl->d_stride = l.down_exps->nb[2];
            pl->scale = l.down_exps_s ? (const float *) l.down_exps_s->data : nullptr;
            pl->pool = pool_.get();
            pl->stats = &stats_;
            if (pl->gu_stride != 2 * pl->f.up_off || pl->d_stride != pl->f.d_row * (size_t) hp_.n_embd)
                return "layer " + std::to_string(il) + ": expert strides do not match the pool's format";
            pool_layers_[il] = std::move(pl);
        }
    }

    if (o.cache_slots != 0) {   // stage d
        if (!o.pool) return "--cache-slots needs --pool (the misses go to the pool)";
        rank_.assign(hp_.n_layer, {});
        if (o.cache_slots < hp_.n_expert && o.profile.empty()) {   // no profile: plain order; the first prompt's routing refills it
            for (auto & r : rank_) for (int e = 0; e < hp_.n_expert; ++e) r.push_back(e);
        } else if (o.cache_slots < hp_.n_expert) {
            FILE * pf = fopen(o.profile.c_str(), "r");
            if (!pf) return "cannot read the profile " + o.profile;
            char line[8192];
            while (fgets(line, sizeof line, pf)) {
                char * q = line; int il = (int) strtol(q, &q, 10);
                if (*q == ':') ++q;
                if (il < 0 || il >= hp_.n_layer) continue;
                for (;;) { char * e2; long v = strtol(q, &e2, 10); if (e2 == q) break; rank_[il].push_back((int) v); q = e2; }
            }
            fclose(pf);
        }
        cache_.resize(hp_.n_layer);
        for (int il = std::max(0, o.cache_l0); il < std::min(o.n_cpu_moe, hp_.n_layer); ++il) {
            auto C = std::make_unique<CacheLayer>();
            C->full = o.cache_slots >= hp_.n_expert;
            C->res.assign(hp_.n_expert, -1);
            if (!C->full) { C->score.assign(hp_.n_expert, 0.0); C->pending.assign(hp_.n_expert, 0); }
            cache_[il] = std::move(C);
        }
        int S = o.cache_slots;
        const bool autos = S < 0;
        if (autos) {   // the most slots the free VRAM holds after the reserve
            size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
            const size_t reserve = (size_t) o.vram_reserve_mb << 20, per = slot_bytes();
            S = fr > reserve + per ? (int) ((fr - reserve) / per) - 1 : 0;
            S = std::min(S, hp_.n_expert - 1);
            if (S < kMinSlots) return "auto slots: the card has no room for an expert arena (" + std::to_string(fr >> 20) + " MiB free)";
        }
        for (int il = 0; il < hp_.n_layer; ++il)
            if (cache_[il] && !cache_[il]->full && (int) rank_[il].size() < S) return "profile: layer " + std::to_string(il) + " ranks too few experts";
        std::string aerr;
        while (!(aerr = arena_build(S, true)).empty() && autos && S > kMinSlots) S = std::max(kMinSlots, S - 2);
        if (!aerr.empty()) return aerr;
        slots_load_ = slots_now_;
        for (int il = 0; il < hp_.n_layer; ++il) {
            CacheLayer * C = cache_[il].get();
            if (!C || C->full) continue;
            C->miss = std::make_unique<PoolLayer>();
            { const PoolLayer & P = *pool_layers_[il]; PoolLayer & M = *C->miss;
              M.f = P.f; M.gu = P.gu; M.gu_stride = P.gu_stride; M.dn = P.dn; M.d_stride = P.d_stride; M.scale = P.scale; M.pool = P.pool; }
            C->miss->res = C->res.data();
            C->miss->stats = &stats_;
        }
    }

    if (o.overlap) {
        if (cache_.empty()) return "--overlap needs --cache-slots";
        driver_ = std::make_unique<Driver>();
        for (auto & C : cache_) if (C && C->miss) C->miss->driver = driver_.get();
    }
    if (o_.async_swaps) {   // the copy stream: expert bytes from the page-locked mapping, beside the compute stream
        if (!o.adapt) return "--async-swaps needs --adapt";
        cudaStream_t cs;
        if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) return "cannot create the copy stream";
        copy_stream_ = cs;

    }

    ggml_backend_t backs[2] = { gpu_, cpu_ };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(gpu_),
                                            ggml_backend_dev_host_buffer_type(gpu_dev) };
    if (!bufts[1]) bufts[1] = ggml_backend_get_default_buffer_type(cpu_);
    sched_ = ggml_backend_sched_new(backs, bufts, 2, kMaxNodes, /*parallel*/ false, /*op_offload*/ true);
    graph_buf_.resize(ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false));
    return "";
}

// ---------------------------------------------------------------- the graph
// llm_graph_context::build_norm (LLM_NORM_RMS, weight, no bias) + llama_context::graph_get_cb's pin of "norm" nodes to
// the layer's device (full offload: every ubatch size)
ggml_tensor * Engine::norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, int il) {
    cur = ggml_rms_norm(ctx, cur, hp_.eps);
    if (w) {
        if (il >= 0) {
            ggml_format_name(cur, "norm-%d", il);
            if (ggml_backend_supports_op(gpu_, cur)) ggml_backend_sched_set_tensor_backend(sched_, cur, gpu_);
        } else {
            ggml_set_name(cur, "norm");
        }
        cur = ggml_mul(ctx, cur, w);
    }
    return cur;
}

// TDUMP=<file> TDUMP_N=<n_tokens>: hash the named per-layer tensors of every ubatch with that many tokens (debug)
static bool tdump_want(const ggml_tensor * t) {
    static const char * pfx[] = { "attn_norm-", "Qcur_pos-", "Kcur_pos-", "kqv_out-", "attn_out-", "ffn_mlp-", "ffn_moe_logits-",
                                  "ffn_moe_weights_norm-", "ffn_moe_out-", "l_out-", "result_norm", "result_output", nullptr };
    static const int n = getenv("TDUMP_N") ? atoi(getenv("TDUMP_N")) : -1;
    if (!getenv("TDUMP")) return false;
    bool ok = false;
    for (int i = 0; pfx[i]; ++i) if (strncmp(t->name, pfx[i], strlen(pfx[i])) == 0) ok = true;
    if (!ok) return false;
    const int64_t nt = strncmp(t->name, "Qcur_pos-", 9) == 0 || strncmp(t->name, "Kcur_pos-", 9) == 0 ? t->ne[2] : t->ne[1];
    return (n == -2 || nt == n) && ggml_is_contiguous(t);   // TDUMP_N=-2: every ubatch (the same execution grouping on both sides)
}
static void tdump_put(const ggml_tensor * t) {
    static FILE * f = strcmp(getenv("TDUMP"), "-") == 0 ? nullptr : fopen(getenv("TDUMP"), "w");
    if (!f) return;
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t c : b) { h ^= c; h *= 1099511628211ULL; }
    fprintf(f, "%s %016llx %s\n", t->name, (unsigned long long) h, ggml_type_name(t->type));
    fflush(f);
}

// GDUMP=<file> GDUMP_N=<n_tokens>: write every node of the ubatch with that many tokens (name op type ne op_params srcs)
static void gdump_node(const ggml_tensor * t) {
    static const char * path = getenv("GDUMP");
    static const int n = getenv("GDUMP_N") ? atoi(getenv("GDUMP_N")) : -1;
    static bool on = false;
    static FILE * f = nullptr;
    if (!path) return;
    if (!on && strcmp(t->name, "inp_scaled") == 0 && t->ne[1] == n) { on = true; f = fopen(path, "w"); }
    if (!on) return;
    fprintf(f, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(f, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(f, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(f, "\t%d\n", (int) t->flags);
    if (strcmp(t->name, "result_output") == 0) { fclose(f); f = nullptr; on = false; path = nullptr; }
}

struct DumpState { FILE * f; };
static bool tdump_cb(ggml_tensor * t, bool ask, void *) {
    if (ask) gdump_node(t);
    if (!tdump_want(t)) return ask ? false : true;
    if (ask) return true;
    tdump_put(t);
    return true;
}
static bool dump_cb(ggml_tensor * t, bool ask, void * ud) {
    if (!ask) return true;
    FILE * f = ((DumpState *) ud)->f;
    fprintf(f, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(f, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(f, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(f, "\n");
    return false;
}

std::string Engine::ubatch(const int32_t * toks, int n, int pos0, bool output, std::vector<float> * logits) {
    if (!fatal_.empty()) return fatal_;
    if (reading_ && n < kOffloadMin) end_read();
    if (n < kOffloadMin) headroom_for(pos0 + n);
    const auto tb0 = std::chrono::steady_clock::now();
    if (copy_stream_) commit_landed();
    const HParams & H = hp_;
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = pos0 + i;
    // slots first (llama_kv_cache::prepare + apply), then the graph sees the applied cells
    static const bool kv_debug = getenv("G4_KV_DEBUG") != nullptr;
    for (Cache * c : { &base_, &swa_ }) {
        if (kv_debug) fprintf(stderr, "g4 find_slot: %s n = %5u, used = %5u, head = %5u, size = %5u, n_swa = %5d\n",
                              c->swa ? "swa " : "base", c->used_max_p1(), c->used(), c->head, c->size, c->swa ? H.n_swa : 0);
        if (!c->find_slot(n, H.n_swa)) return "KV cache full";
        c->apply(positions);
        if (kv_debug) fprintf(stderr, "g4   cells %lld..%lld n_kv %u\n", (long long) c->idxs.front(), (long long) c->idxs.back(), c->n_kv);
    }
    {   // the KV pager
        const uint32_t need = (uint32_t) *std::max_element(base_.idxs.begin(), base_.idxs.end()) + 1;
        if (need > kv_cap_) { const std::string e = kv_grow(need); if (!e.empty()) return e; }
    }

    ggml_init_params ip = { graph_buf_.size(), graph_buf_.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_backend_sched_reset(sched_);
    const int n_outputs = output ? (out_all_ ? n : 1) : 0;
    const int64_t n_embd = H.n_embd;

    // build_inp_embd
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(inp_tokens, "inp_tokens"); ggml_set_input(inp_tokens);
    ggml_tensor * inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n);
    ggml_set_name(inp_embd, "inp_embd"); ggml_set_input(inp_embd);
    ggml_tensor * inps[2] = { ggml_get_rows(ctx, tok_embd_, inp_tokens), inp_embd };
    ggml_tensor * inpL = ggml_build_forward_select(gf, inps, 2, embd_in_ ? 1 : 0);
    ggml_set_name(inpL, "embd");
    ggml_build_forward_expand(gf, inpL);

    // gemma4 graph (raw embeddings, i.e. encoded images, are not scaled)
    inpL = ggml_scale(ctx, inpL, embd_in_ ? 1.0f : sqrtf((float) n_embd));
    ggml_set_name(inpL, "inp_scaled");
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(inp_pos);
    // build_attn_inp_kv_iswa: base idxs + mask, then swa idxs + mask (f16 masks for flash attention)
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, base_.n_kv, n, 1, 1);
    ggml_set_input(kq_mask); ggml_set_name(kq_mask, "attn_inp_kq_mask");
    ggml_tensor * k_idxs_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs_swa);
    ggml_tensor * v_idxs_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs_swa);
    ggml_tensor * kq_mask_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, swa_.n_kv, n, 1, 1);
    ggml_set_input(kq_mask_swa); ggml_set_name(kq_mask_swa, "attn_inp_kq_mask");
    // attn_rot_k/v (quantized KV): base k_rot, base v_rot, swa k_rot, swa v_rot, as build_attn_inp_kv_iswa creates them
    const bool rot_k = ggml_is_quantized(o_.type_k), rot_v = ggml_is_quantized(o_.type_v);
    ggml_tensor * k_rot = nullptr, * v_rot = nullptr, * k_rot_swa = nullptr, * v_rot_swa = nullptr;
    if (rot_k) { const int r = k_rot_size(H.head_dim); k_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(k_rot); ggml_set_name(k_rot, "attn_inp_k_rot"); }
    if (rot_v) { v_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(v_rot); ggml_set_name(v_rot, "attn_inp_v_rot"); }
    if (rot_k) { const int r = k_rot_size(H.head_dim_swa); k_rot_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(k_rot_swa); ggml_set_name(k_rot_swa, "attn_inp_k_rot"); }
    if (rot_v) { v_rot_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(v_rot_swa); ggml_set_name(v_rot_swa, "attn_inp_v_rot"); }
    // build_inp_out_ids
    ggml_tensor * inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    ggml_set_input(inp_out_ids);

    std::vector<ggml_tensor *> adapt_ids(o_.adapt ? H.n_layer : 0, nullptr);   // stage e: each cached layer's routing
    for (int il = 0; il < H.n_layer; ++il) {
        const Layer & l = L_[il];
        const bool swa = H.is_swa[il];
        const int64_t n_embd_head = swa ? H.head_dim_swa : H.head_dim;
        const int64_t n_head = H.n_head, n_head_kv = H.n_head_kv[il];
        const float freq_base_l = swa ? H.rope_base_swa : H.rope_base;
        const int n_rot_l = swa ? H.n_rot_swa : H.n_rot;
        auto nm = [&](ggml_tensor * t, const char * s) { ggml_format_name(t, "%s-%d", s, il); };

        ggml_tensor * cur = norm(ctx, inpL, l.attn_norm, il);
        nm(cur, "attn_norm");
        ggml_tensor * freq_factors = swa ? nullptr : rope_freqs_;

        ggml_tensor * Qcur = ggml_mul_mat(ctx, l.wq, cur);
        nm(Qcur, "Qcur");
        Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n);
        Qcur = norm(ctx, Qcur, l.q_norm, il);
        nm(Qcur, "Qcur_normed");
        Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, freq_factors, n_rot_l, kRopeType, H.n_ctx_train, freq_base_l, 1.0f,
                             kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
        nm(Qcur, "Qcur_pos");

        ggml_tensor * Kcur = ggml_mul_mat(ctx, l.wk, cur);
        ggml_tensor * Vcur = l.wv ? ggml_mul_mat(ctx, l.wv, cur) : Kcur;
        nm(Kcur, "Kcur"); nm(Vcur, "Vcur");
        Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n);
        Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n);
        Kcur = norm(ctx, Kcur, l.k_norm, il);
        Vcur = ggml_rms_norm(ctx, Vcur, H.eps);
        nm(Kcur, "Kcur_normed"); nm(Vcur, "Vcur_normed");
        Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, freq_factors, n_rot_l, kRopeType, H.n_ctx_train, freq_base_l, 1.0f,
                             kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
        nm(Kcur, "Kcur_pos");

        // build_attn (iswa)
        ggml_tensor * kr = swa ? k_rot_swa : k_rot, * vr = swa ? v_rot_swa : v_rot;
        if (kr) { Qcur = mul_mat_hadamard(ctx, Qcur, kr); Kcur = mul_mat_hadamard(ctx, Kcur, kr); }
        if (vr) Vcur = mul_mat_hadamard(ctx, Vcur, vr);
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        Cache & C = swa ? swa_ : base_;
        const int64_t n_gqa = n_embd_head * n_head_kv;
        {   // cpy_k / cpy_v
            ggml_tensor * kc = ggml_view_2d(ctx, Kcur, n_gqa, n, Kcur->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.k[il], kc, swa ? k_idxs_swa : k_idxs));
            ggml_tensor * vc = ggml_view_2d(ctx, Vcur, n_gqa, n, Vcur->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, C.v[il], vc, swa ? v_idxs_swa : v_idxs));
        }
        ggml_tensor * mask = swa ? kq_mask_swa : kq_mask;
        ggml_tensor * k = ggml_view_4d(ctx, C.k[il], n_embd_head, n_head_kv, C.n_kv, 1,
                                       ggml_row_size(C.k[il]->type, n_embd_head), ggml_row_size(C.k[il]->type, n_gqa),
                                       ggml_row_size(C.k[il]->type, n_gqa * C.size), 0);
        ggml_tensor * v = ggml_view_4d(ctx, C.v[il], n_embd_head, n_head_kv, C.n_kv, 1,
                                       ggml_row_size(C.v[il]->type, n_embd_head), ggml_row_size(C.v[il]->type, n_gqa),
                                       ggml_row_size(C.v[il]->type, n_gqa * C.size), 0);
        {   // build_attn_mha, flash attention
            ggml_tensor * q = ggml_view_4d(ctx, Qcur, Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], 1,
                                           Qcur->nb[1], Qcur->nb[2], Qcur->nb[3], 0);
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
            cur = ggml_flash_attn_ext(ctx, q, kp, vp, mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(cur, nullptr);
            ggml_flash_attn_ext_set_n_kv_max(cur, 0);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        }
        nm(cur, "kqv_out");
        if (vr) cur = mul_mat_hadamard(ctx, cur, vr);
        cur = ggml_mul_mat(ctx, l.wo, cur);

        cur = norm(ctx, cur, l.attn_post_norm, il);
        nm(cur, "attn_post_norm");
        ggml_tensor * attn_out = ggml_add(ctx, cur, inpL);
        nm(attn_out, "attn_out");

        // MLP (shared) beside the experts
        ggml_tensor * cur_mlp = norm(ctx, attn_out, l.ffn_norm, il);
        nm(cur_mlp, "ffn_norm_1");
        {   // build_ffn(GELU, PAR)
            ggml_tensor * tmp = ggml_mul_mat(ctx, l.ffn_up, cur_mlp);
            nm(tmp, "ffn_up");
            ggml_tensor * g = ggml_mul_mat(ctx, l.ffn_gate, cur_mlp);
            nm(g, "ffn_gate");
            g = ggml_geglu_split(ctx, g, tmp);
            nm(g, "ffn_geglu");
            cur_mlp = ggml_mul_mat(ctx, l.ffn_down, g);
        }
        cur_mlp = norm(ctx, cur_mlp, l.post_norm_1, il);
        nm(cur_mlp, "ffn_mlp");

        // experts
        ggml_tensor * cur_moe = norm(ctx, attn_out, l.pre_norm_2, il);
        nm(cur_moe, "ffn_norm_2");
        ggml_tensor * tmp = ggml_rms_norm(ctx, attn_out, H.eps);
        tmp = ggml_scale(ctx, tmp, 1.0f / sqrtf((float) n_embd));
        tmp = ggml_mul(ctx, tmp, l.gate_inp_s);
        ggml_tensor * logits_r = ggml_mul_mat(ctx, l.gate_inp, tmp);
        nm(logits_r, "ffn_moe_logits");
        {   // build_moe_ffn(probs_in = logits, softmax gating, norm_w, fused gate_up, GELU, down scale)
            const int64_t n_expert = H.n_expert, n_used = H.n_expert_used;
            ggml_tensor * probs = ggml_soft_max(ctx, logits_r);
            nm(probs, "ffn_moe_probs");
            ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, (int) n_used);
            nm(selected->src[0], "ffn_moe_argsort");
            nm(selected, "ffn_moe_topk");
            probs = ggml_reshape_3d(ctx, probs, 1, n_expert, n);
            ggml_tensor * weights = ggml_get_rows(ctx, probs, selected);
            nm(weights, "ffn_moe_weights");
            weights = ggml_reshape_2d(ctx, weights, n_used, n);
            ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
            nm(wsum, "ffn_moe_weights_sum");
            wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
            nm(wsum, "ffn_moe_weights_sum_clamped");
            weights = ggml_div(ctx, weights, wsum);
            nm(weights, "ffn_moe_weights_norm");
            weights = ggml_reshape_3d(ctx, weights, 1, n_used, n);
            ggml_build_forward_expand(gf, weights);

            // gate_up -> geglu -> down over expert tensors `gu_w`/`dn_w` addressed by `ids` (build_moe_ffn's middle)
            auto mm_path = [&](ggml_tensor * gu_w, ggml_tensor * dn_w, ggml_tensor * ids) {
                ggml_tensor * x = ggml_reshape_3d(ctx, cur_moe, n_embd, 1, n);
                ggml_tensor * gate_up = ggml_mul_mat_id(ctx, gu_w, x, ids);
                nm(gate_up, "ffn_moe_gate_up");
                const int64_t n_ff = gate_up->ne[0] / 2;
                ggml_tensor * gt = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
                nm(gt, "ffn_moe_gate");
                ggml_tensor * up = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2],
                                                n_ff * gate_up->nb[0]);
                nm(up, "ffn_moe_up");
                gt = ggml_geglu_split(ctx, gt, up);
                nm(gt, "ffn_moe_geglu");
                return ggml_mul_mat_id(ctx, dn_w, gt, ids);
            };
            CacheLayer * C = cache_.empty() ? nullptr : cache_[il].get();
            const bool small = n < kOffloadMin;
            if (o_.adapt && C && !C->full) { ggml_set_output(selected->src[0]); adapt_ids[il] = selected->src[0]; }
            ggml_tensor * scale_t = l.down_exps_s;
            ggml_tensor * experts;
            if (C && C->full) {                                     // stage d, whole layer: llama.cpp's GPU-layer path
                experts = mm_path(C->gu, C->dn, selected);
                scale_t = C->scale;
                if (small) { stats_.accesses += (uint64_t) n * n_used; stats_.hits += (uint64_t) n * n_used; }
            } else if (C && small && n <= kSlotMaxTokens) {         // stage d: hits on the GPU slots, misses in the pool
                ggml_tensor * slot = ggml_reshape_3d(ctx, C->res_f, 1, n_expert, 1);   // per token, as the scale is gathered
                slot = ggml_repeat_4d(ctx, slot, 1, n_expert, n, 1);
                slot = ggml_get_rows(ctx, slot, selected);                          // [1, k, n]: slot, or S (zeros)
                slot = ggml_reshape_2d(ctx, slot, n_used, n);
                ggml_tensor * slot_ids = ggml_cast(ctx, slot, GGML_TYPE_I32);
                nm(slot_ids, "ffn_moe_slots");
                ggml_tensor * args[2] = { cur_moe, selected };
                ggml_tensor * eg, * ec;
                if (o_.overlap) {   // start (CPU) -> hits (GPU, launched async) -> join (CPU, waits on the pool) -> add
                    ggml_tensor * st = ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, moe_start_op, 1, C->miss.get());
                    nm(st, "ffn_moe_start");
                    ggml_build_forward_expand(gf, st);
                    eg = mm_path(C->gu, C->dn, slot_ids);
                    ggml_build_forward_expand(gf, eg);
                    ggml_tensor * jargs[1] = { st };
                    ec = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, jargs, 1, moe_join_op, 1, C->miss.get());
                } else {
                    eg = mm_path(C->gu, C->dn, slot_ids);
                    ec = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1, C->miss.get());
                }
                nm(ec, "ffn_moe_misses");
                experts = ggml_add(ctx, eg, ec);                    // each row: its GPU value + 0, or 0 + its CPU value
                scale_t = C->scale;
                if (C->gu_all) {                                    // dcheck twins: the whole layer both ways
                    ggml_tensor * tg = mm_path(C->gu_all, C->dn_all, selected);
                    ggml_tensor * tc = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1,
                                                      pool_layers_[il].get());
                    ggml_set_output(tg); ggml_set_output(tc); ggml_set_output(experts); ggml_set_output(selected->src[0]);
                    ggml_build_forward_expand(gf, tg); ggml_build_forward_expand(gf, tc);
                    dcheck_ = { experts, tg, tc, selected->src[0], C };
                }
            } else if (pool_ && pool_layers_[il] && small) {         // stage c: all of the layer's experts in the pool
                ggml_tensor * args[2] = { cur_moe, selected };
                experts = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1,
                                         pool_layers_[il].get());
                if (!cache_.empty()) stats_.accesses += (uint64_t) n * n_used;
            } else {
                experts = mm_path(l.gate_up_exps, l.down_exps, selected);
            }
            if (scale_t) {    // build_lora_mm_id's per-expert scale
                ggml_tensor * s = ggml_reshape_3d(ctx, scale_t, 1, n_expert, 1);
                s = ggml_repeat_4d(ctx, s, 1, n_expert, n, 1);
                s = ggml_get_rows(ctx, s, selected);
                experts = ggml_mul(ctx, experts, s);
            }
            nm(experts, "ffn_moe_down");
            experts = ggml_mul(ctx, experts, weights);
            nm(experts, "ffn_moe_weighted");
            ggml_build_forward_expand(gf, experts);
            ggml_tensor * parts[64];
            for (int i = 0; i < n_used; ++i) {
                parts[i] = ggml_view_2d(ctx, experts, n_embd, n, experts->nb[2], i * experts->nb[1]);
                ggml_build_forward_expand(gf, parts[i]);
            }
            ggml_tensor * moe_out = parts[0];
            for (int i = 1; i < n_used; ++i) {
                moe_out = ggml_add(ctx, moe_out, parts[i]);
                ggml_build_forward_expand(gf, moe_out);
            }
            nm(moe_out, "ffn_moe_out");
            cur_moe = moe_out;
        }
        cur_moe = norm(ctx, cur_moe, l.post_norm_2, il);
        nm(cur_moe, "ffn_moe");
        cur = ggml_add(ctx, cur_mlp, cur_moe);
        nm(cur, "ffn_moe_combined");

        cur = norm(ctx, cur, l.ffn_post_norm, -1);
        nm(cur, "ffn_post_norm");
        cur = ggml_add(ctx, cur, attn_out);
        if (l.out_scale) {
            cur = ggml_mul(ctx, cur, l.out_scale);
            nm(cur, "out_scaled");
        }
        nm(cur, "l_out");
        inpL = cur;
    }
    ggml_tensor * cur = norm(ctx, inpL, output_norm_, -1);
    ggml_set_name(cur, "h_nextn");
    ggml_tensor * t_h = (want_h_ && n_outputs) ? cur : nullptr;   // the MTP drafter's input (all rows: not masked)
    if (t_h) ggml_set_output(t_h);
    cur = ggml_get_rows(ctx, cur, inp_out_ids);
    ggml_set_name(cur, "result_norm");
    cur = ggml_mul_mat(ctx, output_, cur);
    if (H.softcap > 0) {
        cur = ggml_scale(ctx, cur, 1.0f / H.softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, H.softcap);
    }
    ggml_set_name(cur, "result_output");
    ggml_tensor * t_logits = cur;
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(sched_, gf)) { ggml_free(ctx); return "graph allocation failed"; }
    const bool timed = n < kOffloadMin;
    if (timed) { stats_.t_ubatches++; stats_.t_splits += ggml_backend_sched_get_n_splits(sched_); }
    const uint64_t acc0 = stats_.accesses, hit0 = stats_.hits;
    if (getenv("G4_SCHED_DUMP") && n == 1 && n_ubatch_graphs_ == 1) {   // which backend runs each node (one decode step)
        FILE * f = fopen(getenv("G4_SCHED_DUMP"), "w");
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched_, t);
            fprintf(f, "%s\t%s\t%s\n", t->name, ggml_op_desc(t), b ? ggml_backend_name(b) : "-");
        }
        fclose(f);
    }

    // inputs
    if (embd_in_) {
        std::vector<int32_t> zero(n, 0);
        ggml_backend_tensor_set(inp_tokens, zero.data(), 0, n * sizeof(int32_t));
        ggml_backend_tensor_set(inp_embd, embd_in_, 0, (size_t) n * n_embd * sizeof(float));
    } else {
        ggml_backend_tensor_set(inp_tokens, toks, 0, n * sizeof(int32_t));
    }
    ggml_backend_tensor_set(inp_pos, positions.data(), 0, n * sizeof(int32_t));
    auto set_cache_inputs = [&](Cache & C, ggml_tensor * ki, ggml_tensor * vi, ggml_tensor * m) {
        ggml_backend_tensor_set(ki, C.idxs.data(), 0, n * sizeof(int64_t));
        ggml_backend_tensor_set(vi, C.idxs.data(), 0, n * sizeof(int64_t));
        std::vector<ggml_fp16_t> md((size_t) C.n_kv * n);
        const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n; ++i) {
            const int32_t p1 = positions[i];
            for (uint32_t j = 0; j < C.n_kv; ++j) {
                const int32_t p0 = C.pos[j];
                // non-causal (an image chunk on Gemma, LLAMA_NON_CAUSAL_TYPE_SWA_ONLY): the window layers see the whole chunk
                const bool causal = !(non_causal_ && C.swa);
                bool masked = p0 < 0 || (causal && p0 > p1) || (C.swa && p1 - p0 >= H.n_swa);
                md[(size_t) i * C.n_kv + j] = masked ? drop : keep;
            }
        }
        ggml_backend_tensor_set(m, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
    };
    set_cache_inputs(base_, k_idxs, v_idxs, kq_mask);
    set_cache_inputs(swa_, k_idxs_swa, v_idxs_swa, kq_mask_swa);
    for (ggml_tensor * r : { k_rot, v_rot, k_rot_swa, v_rot_swa }) {
        if (!r) continue;
        const std::vector<float> h = gen_hadamard((int) r->ne[0]);
        ggml_backend_tensor_set(r, h.data(), 0, h.size() * sizeof(float));
    }
    if (n_outputs) {
        std::vector<int32_t> oid(n_outputs);
        for (int i = 0; i < n_outputs; ++i) oid[i] = n - n_outputs + i;
        ggml_backend_tensor_set(inp_out_ids, oid.data(), 0, n_outputs * sizeof(int32_t));
    }

    DumpState ds = { nullptr };
    if (n == 1) ++n_ubatch_graphs_;
    if (!o_.graph_dump.empty() && n == 1 && n_ubatch_graphs_ == 3) {   // route-log dumps its 3rd decoded token
        ds.f = fopen(o_.graph_dump.c_str(), "w");
        ggml_backend_sched_set_eval_callback(sched_, dump_cb, &ds);
    }
    if ((getenv("TDUMP") || getenv("GDUMP")) && !ds.f) ggml_backend_sched_set_eval_callback(sched_, tdump_cb, nullptr);
    const auto tc0 = std::chrono::steady_clock::now();
    if (timed) stats_.t_build += std::chrono::duration<double, std::milli>(tc0 - tb0).count();
    const ggml_status st = ggml_backend_sched_graph_compute(sched_, gf);
    const auto tc1 = std::chrono::steady_clock::now();
    if (timed) stats_.t_compute += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
    if (ds.f) { fclose(ds.f); ggml_backend_sched_set_eval_callback(sched_, nullptr, nullptr); }
    if (st != GGML_STATUS_SUCCESS) { ggml_free(ctx); return "compute failed"; }
    if (dcheck_.merged) {   // every merged row must equal its source twin bit for bit
        const DCheck d = dcheck_;
        dcheck_ = {};
        const int64_t k = d.merged->ne[1], ne = d.merged->ne[0];
        std::vector<float> m(ggml_nelements(d.merged)), g(m.size()), c(m.size());
        ggml_backend_tensor_get(d.merged, m.data(), 0, m.size() * 4);
        ggml_backend_tensor_get(d.twin_gpu, g.data(), 0, g.size() * 4);
        ggml_backend_tensor_get(d.twin_cpu, c.data(), 0, c.size() * 4);
        std::vector<int32_t> ids(ggml_nelements(d.argsort));
        ggml_backend_tensor_get(d.argsort, ids.data(), 0, ids.size() * 4);
        const int64_t ne_sort = d.argsort->ne[0];
        for (int64_t t = 0; t < n; ++t) for (int64_t j = 0; j < k; ++j) {
            const int e = ids[t * ne_sort + j];
            const bool hit = d.C->res[e] >= 0;
            const float * src = (hit ? g.data() : c.data()) + (t * k + j) * ne;
            stats_.dcheck_rows++; stats_.dcheck_gpu_rows += hit;
            if (memcmp(m.data() + (t * k + j) * ne, src, ne * 4) != 0) stats_.dcheck_bad++;
        }
    }
    const auto ta0 = std::chrono::steady_clock::now();
    if (timed) stats_.t_readout += std::chrono::duration<double, std::milli>(ta0 - tc1).count();
    if (timed && (o_.lend || o_.prefill_ubatch > 0)) {
        stats_.step_acc.push_back((uint32_t) (stats_.accesses - acc0)); stats_.step_hit.push_back((uint32_t) (stats_.hits - hit0));
    }
    if (o_.adapt) adapt_after(n, adapt_ids);
    const auto ta1 = std::chrono::steady_clock::now();
    if (timed) stats_.t_scores += std::chrono::duration<double, std::milli>(ta1 - ta0).count();   // incl. the swaps
    if (n_outputs && logits) {
        logits->resize((size_t) n_outputs * H.n_vocab);
        ggml_backend_tensor_get(t_logits, logits->data(), 0, (size_t) n_outputs * H.n_vocab * sizeof(float));
    }
    if (t_h) {   // the output rows' hidden states (the last n_outputs rows of the ubatch)
        h_rows_.resize((size_t) n_outputs * n_embd);
        ggml_backend_tensor_get(t_h, h_rows_.data(), (size_t) (n - n_outputs) * n_embd * sizeof(float),
                                h_rows_.size() * sizeof(float));
    }
    if (timed) stats_.t_readout += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta1).count();
    ggml_free(ctx);
    return "";
}

// ---------------------------------------------------------------- stage e: adaptation
// After a ubatch: every cached layer's scores decay per token and count its routed experts (Gate 0's
// lfu_decay rule). A prompt read (a big ubatch) only marks a refill; the next small ubatch first re-sets every layer
// to its top-S, then each small ubatch swaps at most adapt_budget experts per layer. Copies are synchronous here
// (the copy stream comes with the overlap work), and happen after the step's compute and its twin check.
void Engine::adapt_after(int n, const std::vector<ggml_tensor *> & argsorts) {
    const double d = pow(0.5, 1.0 / o_.adapt_half_life);
    const int k = hp_.n_expert_used;
    std::vector<int32_t> ids;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!argsorts[il]) continue;
        CacheLayer * C = cache_[il].get();
        ids.resize(ggml_nelements(argsorts[il]));
        ggml_backend_tensor_get(argsorts[il], ids.data(), 0, ids.size() * sizeof(int32_t));
        const int64_t ne = argsorts[il]->ne[0];
        for (int t = 0; t < n; ++t) {
            for (auto & v : C->score) v *= d;
            for (int j = 0; j < k; ++j) C->score[ids[t * ne + j]] += 1.0;
        }
    }
    if (n >= kOffloadMin) { refill_pending_ = o_.adapt_refill; return; }
    const bool refill = refill_pending_;
    refill_pending_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t before = stats_.swaps;
    for (int il = 0; il < hp_.n_layer; ++il)
        if (argsorts[il]) swap_layer(il, refill ? INT32_MAX : o_.adapt_budget);
    if (copy_stream_ && !pending_.empty() && !pending_.back().ev) {
        if (pending_.back().sw.empty()) pending_.pop_back();
        else {
            cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            cudaEventRecord(ev, (cudaStream_t) copy_stream_);
            pending_.back().ev = ev;
            uint64_t inflight = 0; for (auto & b : pending_) inflight += b.sw.size();
            stats_.pending_peak = std::max<uint64_t>(stats_.pending_peak, inflight);
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (refill) { stats_.refill_swaps += stats_.swaps - before; stats_.refill_ms += ms; }
    else stats_.swap_ms += ms;
}

void Engine::swap_layer(int il, int budget) {
    CacheLayer * C = cache_[il].get();
    const Layer & l = L_[il];
    const int NE = hp_.n_expert;
    std::vector<int> want, worst;
    for (int e = 0; e < NE; ++e) (C->res[e] < 0 ? want : worst).push_back(e);
    std::stable_sort(want.begin(), want.end(), [&](int a, int b) { return C->score[a] > C->score[b]; });
    std::stable_sort(worst.begin(), worst.end(), [&](int a, int b) { return C->score[a] < C->score[b]; });
    const size_t gs = l.gate_up_exps->nb[2], ds = l.down_exps->nb[2];
    int done = 0;
    if (copy_stream_) {   // async: the victim goes to the CPU now, the newcomer when its copy has landed
        want.erase(std::remove_if(want.begin(), want.end(), [&](int e) { return C->pending[e] != 0; }), want.end());
        if (pending_.empty() || pending_.back().ev) pending_.push_back({ nullptr, {} });
    }
    // stage g: slots left free (after a read the arena comes back empty) take the best newcomers first
    std::vector<int> free_slots;
    for (int j = 0; j < C->S; ++j) if (C->owner[j] == -1) free_slots.push_back(j);
    size_t wi = 0;
    for (size_t i = 0; i < want.size() && done < budget; ++i) {
        const int a = want[i];
        int slot, w = -1;
        if (!free_slots.empty()) {
            if (C->score[a] <= 0) break;
            slot = free_slots.back(); free_slots.pop_back();
        } else {
            if (wi >= worst.size()) break;
            w = worst[wi++];
            if (C->score[a] <= C->score[w]) break;
            slot = C->res[w];
        }
        if (copy_stream_) {
            cudaStream_t cs = (cudaStream_t) copy_stream_;
            cudaMemcpyAsync((char *) C->gu->data + (size_t) slot * gs, (const char *) l.gate_up_exps->data + a * gs, gs, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync((char *) C->dn->data + (size_t) slot * ds, (const char *) l.down_exps->data + a * ds, ds, cudaMemcpyHostToDevice, cs);
            if (w >= 0) C->res[w] = -1;
            C->owner[slot] = -2; C->pending[a] = 1;
            pending_.back().sw.push_back({ il, a, slot });
        } else {
            ggml_backend_tensor_set(C->gu, (const char *) l.gate_up_exps->data + a * gs, (size_t) slot * gs, gs);
            ggml_backend_tensor_set(C->dn, (const char *) l.down_exps->data + a * ds, (size_t) slot * ds, ds);
            if (w >= 0) C->res[w] = -1;
            C->res[a] = slot; C->owner[slot] = a;
        }
        stats_.swaps++; stats_.swap_bytes += gs + ds;
        done++;
    }
    if (done) {
        std::vector<float> rf(NE);
        for (int e = 0; e < NE; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
        ggml_backend_tensor_set(C->res_f, rf.data(), 0, NE * sizeof(float));
    }
}

// ---------------------------------------------------------------- stage g: chunked reads with VRAM lent from the cache
// The window cache holds n_swa + ubatch cells (llama_kv_cache_iswa); a read in big ubatches needs it that large.
void Engine::resize_swa(uint32_t cells) {
    cells = (std::min<uint32_t>(base_.size, cells) + 255) / 256 * 256;
    if (cells == swa_.size) return;
    // keep the most recent positions (all of them if they fit), in position order, from cell 0
    std::vector<std::pair<int32_t, uint32_t>> live;
    for (uint32_t i = 0; i < swa_.size; ++i) if (swa_.pos[i] >= 0) live.push_back({ swa_.pos[i], i });
    std::sort(live.begin(), live.end());
    if (live.size() > cells) live.erase(live.begin(), live.end() - cells);
    ggml_init_params kp = { (size_t) (2 * hp_.n_layer + 4) * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(kp);
    std::vector<ggml_tensor *> nk(hp_.n_layer, nullptr), nv(hp_.n_layer, nullptr);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!swa_.k[il]) continue;
        nk[il] = ggml_new_tensor_3d(ctx, swa_.k[il]->type, swa_.k[il]->ne[0], cells, 1);
        nv[il] = ggml_new_tensor_3d(ctx, swa_.v[il]->type, swa_.v[il]->ne[0], cells, 1);
        ggml_format_name(nk[il], "cache_k_l%d", il); ggml_format_name(nv[il], "cache_v_l%d", il);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(gpu_));
    ggml_backend_buffer_clear(buf, 0);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!nk[il]) continue;
        for (int pass = 0; pass < 2; ++pass) {
            ggml_tensor * src = pass ? swa_.v[il] : swa_.k[il], * dst = pass ? nv[il] : nk[il];
            const size_t row = src->nb[1];
            for (size_t j = 0; j < live.size();) {   // contiguous runs of old cells
                size_t r = 1;
                while (j + r < live.size() && live[j + r].second == live[j].second + r) ++r;
                cudaMemcpy((char *) dst->data + j * row, (const char *) src->data + (size_t) live[j].second * row, r * row, cudaMemcpyDeviceToDevice);
                j += r;
            }
        }
    }
    std::vector<int32_t> pos(cells, -1);
    for (size_t j = 0; j < live.size(); ++j) pos[j] = live[j].first;
    if (buf_swa_) ggml_backend_buffer_free(buf_swa_);
    if (ctx_swa_) ggml_free(ctx_swa_);
    ctx_swa_ = ctx; buf_swa_ = buf;
    swa_.k = nk; swa_.v = nv; swa_.pos = pos; swa_.size = cells;
    swa_.head = (uint32_t) (live.size() % cells);
    stats_.swa_big_cells = std::max<uint64_t>(stats_.swa_big_cells, cells);
}

void Engine::begin_read(int ubatch) {
    const auto t0 = std::chrono::steady_clock::now();
    reading_ = true;
    if (o_.lend && buf_cache_) {   // give the arena's VRAM to the read
        if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
        stats_.lent_bytes = ggml_backend_buffer_get_size(buf_cache_);
        ggml_backend_buffer_free(buf_cache_); buf_cache_ = nullptr;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_cache_); t; t = ggml_get_next_tensor(ctx_cache_, t)) { t->data = nullptr; t->buffer = nullptr; }
        for (auto & C : cache_) if (C) { std::fill(C->res.begin(), C->res.end(), -1); std::fill(C->owner.begin(), C->owner.end(), -1); }
        ggml_backend_sched_free(sched_);   // its compute buffers re-grow for the big ubatches
        ggml_backend_t backs[2] = { gpu_, cpu_ };
        ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(gpu_), ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu_)) };
        sched_ = ggml_backend_sched_new(backs, bufts, 2, kMaxNodes, false, true);
    }
    resize_swa((uint32_t) (hp_.n_swa + ubatch));
    stats_.lend_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void Engine::end_read() {
    const auto t0 = std::chrono::steady_clock::now();
    reading_ = false;
    resize_swa((uint32_t) (hp_.n_swa + o_.n_ubatch));
    ggml_backend_sched_free(sched_);   // drop the big ubatches' compute buffers
    ggml_backend_t backs[2] = { gpu_, cpu_ };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(gpu_), ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu_)) };
    sched_ = ggml_backend_sched_new(backs, bufts, 2, kMaxNodes, false, true);
    if (o_.lend && !buf_cache_ && ctx_cache_) {   // the arena comes back empty; the refill copies in the background
        // The read may leave less VRAM than it lent (ggml-cuda's pool keeps what the big ubatches grew): fewer slots
        // then, never a failed step.
        int S = slots_now_;
        if (o_.cache_slots < 0) {   // auto slots: what the VRAM holds after the reserve, as at load (the KV pager may have
            // grown the KV during the read; without it this is the load-time count)
            size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
            const size_t per = slot_bytes(), reserve = (size_t) o_.vram_reserve_mb << 20;
            const int fit = fr > reserve + 2 * per ? (int) ((fr - reserve) / per) - 1 : kMinSlots;
            S = std::max(kMinSlots, std::min(S, fit));
        }
        std::string err;
        while (!(err = arena_build(S, false)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 4);
        if (!err.empty()) fatal_ = "after the read: " + err;
        headroom_kv_ = 0;   // free VRAM changed: check again before the next small step
    }
    read_end_ = std::chrono::steady_clock::now();
    refill_watch_ = true;
    stats_.restore_ms += std::chrono::duration<double, std::milli>(read_end_ - t0).count();
}

// ---------------------------------------------------------------- the KV pager (global layers)
size_t Engine::kv_bytes(uint32_t cells) const {
    size_t b = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) continue;
        const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv[il];
        b += (ggml_row_size(o_.type_k, n_gqa) + ggml_row_size(o_.type_v, n_gqa)) * cells;
    }
    return b;
}

// one global layer's K/V in a buffer of cap cells: zeroed (as the whole-context buffer was at load), the old cells
// copied over, the old buffer freed
std::string Engine::kv_alloc_layer(int il, uint32_t cap) {
    const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv[il];
    ggml_init_params p = { 4 * ggml_tensor_overhead(), nullptr, true };
    KvLayer nl; nl.ctx = ggml_init(p);
    ggml_tensor * k = ggml_new_tensor_3d(nl.ctx, o_.type_k, n_gqa, cap, 1);
    ggml_tensor * v = ggml_new_tensor_3d(nl.ctx, o_.type_v, n_gqa, cap, 1);
    ggml_format_name(k, "cache_k_l%d", il);
    ggml_format_name(v, "cache_v_l%d", il);
    nl.buf = ggml_backend_alloc_ctx_tensors_from_buft(nl.ctx, ggml_backend_get_default_buffer_type(gpu_));
    if (!nl.buf) { ggml_free(nl.ctx); return "KV allocation failed (layer " + std::to_string(il) + ", " + std::to_string(cap) + " cells)"; }
    ggml_backend_buffer_clear(nl.buf, 0);
    if (base_.k[il]) {
        const size_t rows = (size_t) std::min<int64_t>(base_.k[il]->ne[1], cap);
        cudaMemcpy(k->data, base_.k[il]->data, ggml_row_size(k->type, n_gqa) * rows, cudaMemcpyDeviceToDevice);
        cudaMemcpy(v->data, base_.v[il]->data, ggml_row_size(v->type, n_gqa) * rows, cudaMemcpyDeviceToDevice);
    }
    if (kvl_[il].buf) ggml_backend_buffer_free(kvl_[il].buf);
    if (kvl_[il].ctx) ggml_free(kvl_[il].ctx);
    kvl_[il] = nl;
    base_.k[il] = k; base_.v[il] = v;
    return "";
}

// room for cells: the capacity rounded up to kv_step (at most the context); the slot arena gives way first, one layer
// at a time is reallocated (the moment's extra VRAM is one layer's new buffer)
std::string Engine::kv_grow(uint32_t cells) {
    if (cells <= kv_cap_) return "";
    cudaSetDevice(cuda_dev_);
    const uint32_t step = (uint32_t) std::max(256, (o_.kv_step + 255) / 256 * 256);
    const uint32_t cap = std::min(base_.size, (cells + step - 1) / step * step);
    size_t one = 0;
    for (int il = 0; il < hp_.n_layer; ++il) if (!hp_.is_swa[il]) {
        const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv[il];
        one = std::max(one, (ggml_row_size(o_.type_k, n_gqa) + ggml_row_size(o_.type_v, n_gqa)) * cap);
    }
    ensure_free(kv_bytes(cap) - kv_bytes(kv_cap_) + one + ((size_t) 64 << 20));
    if (!fatal_.empty()) return fatal_;
    ggml_backend_synchronize(gpu_);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) continue;
        const std::string e = kv_alloc_layer(il, cap);
        if (!e.empty()) return e;
    }
    fprintf(stderr, "strata-lite: KV on the card %u -> %u cells (%.0f MiB)\n", kv_cap_, cap, kv_bytes(cap) / 1048576.0);
    kv_cap_ = cap;
    headroom_kv_ = 0;   // free VRAM changed
    return "";
}

void Engine::kv_shrink() {
    const uint32_t init = o_.kv_initial > 0 ? std::min<uint32_t>(base_.size, (uint32_t) ((o_.kv_initial + 255) / 256 * 256)) : base_.size;
    if (kv_cap_ <= init || kvl_.empty()) return;
    ggml_backend_synchronize(gpu_);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) continue;
        const std::string e = kv_alloc_layer(il, init);
        if (!e.empty()) { fatal_ = "new conversation: " + e; return; }
    }
    fprintf(stderr, "strata-lite: KV on the card %u -> %u cells (a new conversation)\n", kv_cap_, init);
    kv_cap_ = init;
    headroom_kv_ = 0;
    if (slots_load_ > slots_now_) grow_to(slots_load_, (size_t) o_.vram_reserve_mb << 20);
}

// ---------------------------------------------------------------- the slot arena: (re)build, size, shrink
size_t Engine::slot_bytes() const {
    size_t b = 0;
    for (int il = 0; il < hp_.n_layer; ++il)
        if (il < (int) cache_.size() && cache_[il] && !cache_[il]->full) b += L_[il].gate_up_exps->nb[2] + L_[il].down_exps->nb[2];
    return b;
}

// Every cached layer gets S slots (+ the zero slot); with fill_profile the top S of the profile are copied in, else the
// arena starts empty and the next small ubatch refills it from the usage counts (as after a read).
std::string Engine::arena_build(int S, bool fill_profile) {
    if (buf_cache_) { ggml_backend_buffer_free(buf_cache_); buf_cache_ = nullptr; }
    if (ctx_cache_) { ggml_free(ctx_cache_); ctx_cache_ = nullptr; }
    ggml_init_params cp = { (size_t) (8 * hp_.n_layer + 8) * ggml_tensor_overhead(), nullptr, true };
    ctx_cache_ = ggml_init(cp);
    for (int il = 0; il < (int) cache_.size(); ++il) {
        CacheLayer * C = cache_[il].get();
        if (!C) continue;
        const Layer & l = L_[il];
        C->S = C->full ? hp_.n_expert : S;
        const int alloc = C->full ? hp_.n_expert : C->S + 1;
        C->gu = ggml_new_tensor_3d(ctx_cache_, l.gate_up_exps->type, hp_.n_embd, 2 * hp_.n_ff_exp, alloc);
        C->dn = ggml_new_tensor_3d(ctx_cache_, l.down_exps->type, hp_.n_ff_exp, hp_.n_embd, alloc);
        ggml_format_name(C->gu, "blk.%d.ffn_gate_up_exps.slots", il);
        ggml_format_name(C->dn, "blk.%d.ffn_down_exps.slots", il);
        C->scale = nullptr;
        if (l.down_exps_s) { C->scale = ggml_dup_tensor(ctx_cache_, l.down_exps_s); ggml_format_name(C->scale, "blk.%d.ffn_down_exps.scale", il); }
        std::fill(C->res.begin(), C->res.end(), -1);
        C->res_f = nullptr; C->gu_all = C->dn_all = nullptr;
        if (C->full) { for (int e = 0; e < hp_.n_expert; ++e) C->res[e] = e; continue; }
        C->owner.assign(C->S, -1);
        if (C->pending.size()) std::fill(C->pending.begin(), C->pending.end(), 0);
        C->res_f = ggml_new_tensor_2d(ctx_cache_, GGML_TYPE_F32, 1, hp_.n_expert);
        ggml_format_name(C->res_f, "blk.%d.ffn_exps.slot_of", il);
        if (il == o_.dcheck_layer) { C->gu_all = ggml_dup_tensor(ctx_cache_, l.gate_up_exps); C->dn_all = ggml_dup_tensor(ctx_cache_, l.down_exps); }
    }
    buf_cache_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cache_, ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_cache_) return "expert cache allocation failed (" + std::to_string(S) + " slots do not fit on this card)";
    ggml_backend_buffer_clear(buf_cache_, 0);                     // slot S (the misses' target) stays all zeros
    ggml_backend_buffer_set_usage(buf_cache_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int il = 0; il < (int) cache_.size(); ++il) {
        CacheLayer * C = cache_[il].get();
        if (!C) continue;
        const Layer & l = L_[il];
        const size_t gs = l.gate_up_exps->nb[2], ds = l.down_exps->nb[2];
        if (C->full) {
            ggml_backend_tensor_set(C->gu, l.gate_up_exps->data, 0, ggml_nbytes(l.gate_up_exps));
            ggml_backend_tensor_set(C->dn, l.down_exps->data, 0, ggml_nbytes(l.down_exps));
        } else {
            if (fill_profile)
                for (int j = 0; j < C->S; ++j) {
                    const int e = rank_[il][j];
                    C->res[e] = j; C->owner[j] = e;
                    ggml_backend_tensor_set(C->gu, (const char *) l.gate_up_exps->data + e * gs, (size_t) j * gs, gs);
                    ggml_backend_tensor_set(C->dn, (const char *) l.down_exps->data + e * ds, (size_t) j * ds, ds);
                }
            std::vector<float> rf(hp_.n_expert);
            for (int e = 0; e < hp_.n_expert; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
            ggml_backend_tensor_set(C->res_f, rf.data(), 0, rf.size() * sizeof(float));
            if (C->gu_all) {
                ggml_backend_tensor_set(C->gu_all, l.gate_up_exps->data, 0, ggml_nbytes(l.gate_up_exps));
                ggml_backend_tensor_set(C->dn_all, l.down_exps->data, 0, ggml_nbytes(l.down_exps));
            }
        }
        if (C->scale) ggml_backend_tensor_set(C->scale, l.down_exps_s->data, 0, ggml_nbytes(l.down_exps_s));
    }
    headroom_kv_ = 0;   // free VRAM changed: check again before the next small step
    if (slots_now_ && S < slots_now_) {
        stats_.arena_shrinks++;
        fprintf(stderr, "strata: expert arena %d -> %d slots per layer (VRAM)\n", slots_now_, S);
    }
    slots_now_ = S; stats_.slots_now = S;
    return "";
}

int Engine::pick_chunk(int n) {
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    size_t avail = fr + ((o_.lend && buf_cache_) ? ggml_backend_buffer_get_size(buf_cache_) : 0);
    const size_t n_kv_end = (size_t) n_past_ + n, margin = (size_t) 256 << 20;
    {   // the KV pager grows during this read: its new cells come out of the same VRAM
        const uint32_t step = (uint32_t) std::max(256, (o_.kv_step + 255) / 256 * 256);
        const uint32_t cells = (uint32_t) std::min<size_t>(base_.size, (n_kv_end + step - 1) / step * step);
        const size_t kv_more = cells > kv_cap_ ? kv_bytes(cells) - kv_bytes(kv_cap_) : 0;
        avail = avail > kv_more ? avail - kv_more : 0;
    }
    for (int P : { 32768, 16384, 8192, kChunkMin }) {   // 32k only where the borrowable VRAM covers it (24-32 GB cards)
        const size_t m = (size_t) std::min(P, n);
        if (kChunkFixed + m * kChunkPerTok + m * n_kv_end * 2 + margin <= avail) return P;
    }
    // nothing fits: with a lent arena 4k is the floor (the maintainer, 2026-10-10: minimum chunk 4k); without lending the
    // read takes llama.cpp's ubatches
    return (o_.lend && buf_cache_) ? kChunkMin : o_.n_ubatch;
}

void Engine::headroom_for(int n_kv) {
    if (n_kv <= headroom_kv_ || !buf_cache_ || slot_bytes() == 0) return;
    const int target = n_kv + kHeadroomStep;
    const size_t need = ((size_t) 256 << 20) + (size_t) target * kComputePerCell;
    const size_t have = ggml_backend_sched_get_buffer_size(sched_, gpu_);
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    const size_t margin = (size_t) 128 << 20;
    if (have < need && fr + have < need + margin) ensure_free(need + margin - have);
    headroom_kv_ = target;
}

int Engine::grow_to(int S, size_t headroom) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    if (!buf_cache_ || slot_bytes() == 0 || S <= slots_now_) return slots_now_;
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    const size_t per = slot_bytes(), now = (size_t) (slots_now_ + 1) * per;   // what the current arena holds
    const int fit = fr + now > headroom + per ? (int) ((fr + now - headroom) / per) - 1 : slots_now_;
    int T = std::min(S, fit);
    if (T <= slots_now_) return slots_now_;
    const int before = slots_now_;
    std::string err;
    while (!(err = arena_build(T, false)).empty() && T > before) T = std::max(before, T - 2);
    if (!err.empty()) { fatal_ = "growing the arena: " + err; return 0; }
    fprintf(stderr, "strata: expert arena %d -> %d slots per layer (VRAM free again)\n", before, T);
    refill_pending_ = true;
    read_end_ = std::chrono::steady_clock::now(); refill_watch_ = true;
    return slots_now_;
}

int Engine::ensure_free(size_t bytes, bool keep_profile) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    if (!buf_cache_ || slot_bytes() == 0) return slots_now_;
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    if (fr >= bytes) return slots_now_;
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
    const size_t per = slot_bytes();
    int S = std::max(kMinSlots, slots_now_ - (int) ((bytes - fr + per - 1) / per) - 1);
    std::string err;
    while (!(err = arena_build(S, keep_profile)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 2);
    if (!err.empty()) { fatal_ = "making room: " + err; return 0; }
    if (keep_profile) return slots_now_;
    refill_pending_ = true;   // the next small ubatch refills every layer to its top-S by usage
    read_end_ = std::chrono::steady_clock::now(); refill_watch_ = true;
    return slots_now_;
}

// async swaps: newcomers whose copies have landed become resident (in order: batches complete in stream order)
void Engine::commit_landed() {
    std::vector<char> touched(hp_.n_layer, 0);
    size_t k = 0;
    for (; k < pending_.size(); ++k) {
        PendingBatch & b = pending_[k];
        if (!b.ev || cudaEventQuery((cudaEvent_t) b.ev) != cudaSuccess) break;
        for (const PendingSwap & p : b.sw) {
            CacheLayer * C = cache_[p.il].get();
            C->res[p.e] = p.slot; C->owner[p.slot] = p.e; C->pending[p.e] = 0;
            touched[p.il] = 1; stats_.commits++;
        }
        cudaEventDestroy((cudaEvent_t) b.ev);
        stats_.commit_batches++;
    }
    cudaGetLastError();
    pending_.erase(pending_.begin(), pending_.begin() + k);
    if (refill_watch_ && pending_.empty() && !refill_pending_) {
        stats_.refill_land_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - read_end_).count();
        refill_watch_ = false;
    }
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!touched[il]) continue;
        CacheLayer * C = cache_[il].get();
        std::vector<float> rf(hp_.n_expert);
        for (int e = 0; e < hp_.n_expert; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
        ggml_backend_tensor_set(C->res_f, rf.data(), 0, rf.size() * sizeof(float));
    }
}

// llama_decode: one batch, split into n_ubatch pieces; only the batch's last token outputs (llama_batch_get_one)
std::string Engine::decode(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h_last) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    const int n = (int) toks.size();
    const int pu = (o_.prefill_ubatch < 0 && n >= kOffloadMin) ? pick_chunk(n) : o_.prefill_ubatch;
    const int ub = (pu > 0 && n >= kOffloadMin) ? pu : o_.n_ubatch;
    if (n >= kOffloadMin) stats_.last_chunk = std::min(ub, n);   // the chunk actually used (a short prompt is one chunk)
    if (o_.prefill_ubatch < 0 && n >= kOffloadMin)
        fprintf(stderr, "strata: reading %d tokens at position %d in %d-token chunks (cap %d)\n", n, n_past_, std::min(ub, n), ub);
    const auto tr0 = std::chrono::steady_clock::now();
    if (ub != o_.n_ubatch || (o_.lend && n >= kOffloadMin)) begin_read(ub);
    for (int i = 0; i < n; i += ub) {
        const int m = std::min(ub, n - i);
        const bool out = (i + m == n);
        want_h_ = out && h_last;
        std::string err = ubatch(toks.data() + i, m, n_past_ + i, out, out ? logits : nullptr);
        want_h_ = false;
        if (!err.empty()) return err;
    }
    if (h_last) *h_last = h_rows_;
    n_past_ += n;
    if (n >= kOffloadMin) stats_.prefill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
    return "";
}

std::string Engine::decode_all(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    const int n = (int) toks.size();
    if (n > o_.n_ubatch) return "verify window larger than a ubatch";
    out_all_ = true; want_h_ = h != nullptr;
    std::string err = ubatch(toks.data(), n, n_past_, true, logits);
    out_all_ = false; want_h_ = false;
    if (!err.empty()) return err;
    if (h) *h = h_rows_;
    n_past_ += n;
    return "";
}

std::string Engine::decode_embd(const float * embd, int n, bool non_causal, std::vector<float> * logits, std::vector<float> * h_last) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    // one ubatch: the window cache must hold n_swa + n cells for it (as a chunked read)
    // an image chunk of kOffloadMin tokens or more streams whole expert layers through the card like a text read, so it
    // borrows the arena's VRAM the same way (with a projector on a second card nothing else made that room)
    const bool big = n > o_.n_ubatch || (o_.lend && n >= kOffloadMin);
    if (big) begin_read(std::max(n, o_.n_ubatch));
    embd_in_ = embd; non_causal_ = non_causal; want_h_ = h_last != nullptr;
    std::string err = ubatch(nullptr, n, n_past_, true, logits);
    embd_in_ = nullptr; non_causal_ = false; want_h_ = false;
    if (!err.empty()) return err;
    if (h_last) *h_last = h_rows_;
    n_past_ += n;
    return "";
}

bool Engine::can_resume(int n) const {
    if (n <= 0) return true;
    std::vector<char> have(hp_.n_swa, 0);
    const int lo = std::max(0, n - hp_.n_swa);
    for (int32_t p : swa_.pos) if (p >= lo && p < n) have[p - lo] = 1;
    for (int p = lo; p < n; ++p) if (!have[p - lo]) return false;
    return true;
}

void Engine::rollback(int n_past) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    base_.seq_rm(n_past); swa_.seq_rm(n_past);
    n_past_ = n_past;
    if (n_past == 0) kv_shrink();   // a new conversation (verify windows never roll back to 0)
}

// ---------------------------------------------------------------- stage f: the MTP drafter (llama.cpp src/models/gemma4-assistant.cpp)
std::string Engine::load_mtp(const std::string & path) {
    Draft & D = mtp_;
    gguf_init_params gp = { true, &D.meta };
    D.gguf = gguf_init_from_file(path.c_str(), gp);
    if (!D.gguf) return "cannot read " + path;
    const std::string A = "gemma4-assistant.";
    D.n_layer = (int) kv_int(D.gguf, A + "block_count");
    D.n_embd = (int) kv_int(D.gguf, A + "embedding_length");
    D.n_head = (int) kv_int(D.gguf, A + "attention.head_count");
    D.head_dim = (int) kv_int(D.gguf, A + "attention.key_length");
    D.head_dim_swa = (int) kv_int(D.gguf, A + "attention.key_length_swa");
    D.n_rot = (int) kv_int(D.gguf, A + "rope.dimension_count");
    D.n_rot_swa = (int) kv_int(D.gguf, A + "rope.dimension_count_swa");
    D.n_ctx_train = (int) kv_int(D.gguf, A + "context_length");
    D.rope_base = kv_f32(D.gguf, A + "rope.freq_base");
    D.rope_base_swa = kv_f32(D.gguf, A + "rope.freq_base_swa");
    D.eps = kv_f32(D.gguf, A + "attention.layer_norm_rms_epsilon");
    for (auto v : kv_arr(D.gguf, A + "attention.sliding_window_pattern")) D.is_swa.push_back(v != 0);
    if (kv_int(D.gguf, A + "embedding_length_out") != hp_.n_embd || (int) D.is_swa.size() != D.n_layer)
        return "MTP drafter does not match the target";
    const int n_tensors = (int) gguf_get_n_tensors(D.gguf);
    ggml_init_params ip = { (size_t) (n_tensors + 4) * ggml_tensor_overhead(), nullptr, true };
    D.ctx = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, size_t>> src;
    const size_t data_off = gguf_get_data_offset(D.gguf);
    auto make = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * m = ggml_get_tensor(D.meta, name.c_str());
        if (!m) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(D.ctx, m);
        ggml_format_name(t, "mtp.%s", name.c_str());
        src.push_back({ t, data_off + gguf_get_tensor_offset(D.gguf, gguf_find_tensor(D.gguf, name.c_str())) });
        return t;
    };
    D.tok_embd = make("token_embd.weight"); D.output_norm = make("output_norm.weight"); D.rope_freqs = make("rope_freqs.weight");
    D.pre_proj = make("nextn.pre_projection.weight"); D.post_proj = make("nextn.post_projection.weight");
    D.L.resize(D.n_layer);
    for (int il = 0; il < D.n_layer; ++il) {
        auto b = [&](const char * s2) { return "blk." + std::to_string(il) + "." + s2; };
        DraftLayer & l = D.L[il];
        l.attn_norm = make(b("attn_norm.weight")); l.wq = make(b("attn_q.weight")); l.wo = make(b("attn_output.weight"));
        l.q_norm = make(b("attn_q_norm.weight")); l.attn_post_norm = make(b("post_attention_norm.weight"));
        l.out_scale = make(b("layer_output_scale.weight")); l.ffn_norm = make(b("ffn_norm.weight"));
        l.ffn_gate = make(b("ffn_gate.weight")); l.ffn_up = make(b("ffn_up.weight")); l.ffn_down = make(b("ffn_down.weight"));
        l.ffn_post_norm = make(b("post_ffw_norm.weight"));
        if (!l.attn_norm || !l.wq || !l.wo || !l.ffn_down) return "MTP drafter layer " + std::to_string(il) + ": missing tensors";
    }
    if (!D.tok_embd || !D.pre_proj || !D.post_proj || !D.output_norm) return "MTP drafter: missing tensors";
    D.buf = ggml_backend_alloc_ctx_tensors_from_buft(D.ctx, ggml_backend_get_default_buffer_type(gpu_));
    if (!D.buf) return "MTP drafter allocation failed";
    ggml_backend_buffer_set_usage(D.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::vector<uint8_t> tmp;
    for (auto & [t, off] : src) {
        tmp.resize(ggml_nbytes(t));
        fseek(f, (long) off, SEEK_SET);
        if (fread(tmp.data(), 1, tmp.size(), f) != tmp.size()) { fclose(f); return "short read in " + path; }
        ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
    }
    fclose(f);
    return "";
}

ggml_tensor * Engine::norm_w(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), w);
}

std::string Engine::draft(int32_t tok, const float * h, int pos0, int32_t * out, std::vector<float> * h_out) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    headroom_for(pos0 + 1);
    const auto td0 = std::chrono::steady_clock::now();
    struct T { Stats & s; std::chrono::steady_clock::time_point t; ~T() { s.t_drafts++; s.t_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); } } timer{ stats_, td0 };
    const Draft & D = mtp_;
    if (!D.buf) return "no MTP drafter loaded";
    ggml_init_params ip = { graph_buf_.size(), graph_buf_.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_backend_sched_reset(sched_);
    const int64_t nb = hp_.n_embd;
    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1); ggml_set_input(t_tok);
    ggml_tensor * t_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nb, 1); ggml_set_input(t_h);
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1); ggml_set_input(t_pos);
    // the shared caches: the drafter's window layers read the target's layer n-2, its global layer layer n-1
    const int L_swa = hp_.n_layer - 2, L_glb = hp_.n_layer - 1;
    const uint32_t nkv_b = base_.calc_n_kv(), nkv_s = swa_.calc_n_kv();
    ggml_tensor * m_b = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv_b, 1, 1, 1); ggml_set_input(m_b);
    ggml_tensor * m_s = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv_s, 1, 1, 1); ggml_set_input(m_s);

    const bool rk = ggml_is_quantized(o_.type_k), rv = ggml_is_quantized(o_.type_v);
    ggml_tensor * dk_rot = nullptr, * dv_rot = nullptr, * dk_rot_swa = nullptr, * dv_rot_swa = nullptr;
    if (rk) { int r = k_rot_size(hp_.head_dim); dk_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(dk_rot);
              r = k_rot_size(hp_.head_dim_swa); dk_rot_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(dk_rot_swa); }
    if (rv) { dv_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(dv_rot);
              dv_rot_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(dv_rot_swa); }
    ggml_tensor * x = ggml_get_rows(ctx, tok_embd_, t_tok);
    x = ggml_scale(ctx, x, sqrtf((float) nb));
    ggml_tensor * cur = ggml_mul_mat(ctx, D.pre_proj, ggml_concat(ctx, x, t_h, 0));
    ggml_tensor * inpL = cur;
    for (int il = 0; il < D.n_layer; ++il) {
        const DraftLayer & l = D.L[il];
        const bool swa = D.is_swa[il];
        const int tl = swa ? L_swa : L_glb;
        Cache & C = swa ? swa_ : base_;
        const int64_t hd = swa ? D.head_dim_swa : D.head_dim, nkv_h = hp_.n_head_kv[tl], n_kv = swa ? nkv_s : nkv_b;
        ggml_tensor * q = ggml_mul_mat(ctx, l.wq, norm_w(ctx, inpL, l.attn_norm, D.eps));
        q = ggml_reshape_3d(ctx, q, hd, D.n_head, 1);
        q = norm_w(ctx, q, l.q_norm, D.eps);
        q = ggml_rope_ext(ctx, q, t_pos, swa ? nullptr : D.rope_freqs, swa ? D.n_rot_swa : D.n_rot, kRopeType, D.n_ctx_train,
                          swa ? D.rope_base_swa : D.rope_base, 1.0f, kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
        ggml_tensor * kr = swa ? dk_rot_swa : dk_rot, * vr = swa ? dv_rot_swa : dv_rot;   // the shared caches' rotations
        if (kr) q = mul_mat_hadamard(ctx, q, kr);
        const int64_t n_gqa = hd * nkv_h;
        ggml_tensor * k = ggml_view_4d(ctx, C.k[tl], hd, nkv_h, n_kv, 1, ggml_row_size(C.k[tl]->type, hd),
                                       ggml_row_size(C.k[tl]->type, n_gqa), ggml_row_size(C.k[tl]->type, n_gqa * C.size), 0);
        ggml_tensor * v = ggml_view_4d(ctx, C.v[tl], hd, nkv_h, n_kv, 1, ggml_row_size(C.v[tl]->type, hd),
                                       ggml_row_size(C.v[tl]->type, n_gqa), ggml_row_size(C.v[tl]->type, n_gqa * C.size), 0);
        ggml_tensor * a = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
                                              ggml_permute(ctx, v, 0, 2, 1, 3), swa ? m_s : m_b, 1.0f, 0.0f, 0.0f);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        a = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
        if (vr) a = mul_mat_hadamard(ctx, a, vr);
        a = ggml_mul_mat(ctx, l.wo, a);
        ggml_tensor * attn_out = ggml_add(ctx, norm_w(ctx, a, l.attn_post_norm, D.eps), inpL);
        ggml_tensor * f = norm_w(ctx, attn_out, l.ffn_norm, D.eps);
        ggml_tensor * g = ggml_geglu_split(ctx, ggml_mul_mat(ctx, l.ffn_gate, f), ggml_mul_mat(ctx, l.ffn_up, f));
        f = ggml_mul_mat(ctx, l.ffn_down, g);
        cur = ggml_add(ctx, norm_w(ctx, f, l.ffn_post_norm, D.eps), attn_out);
        inpL = ggml_mul(ctx, cur, l.out_scale);
    }
    cur = norm_w(ctx, inpL, D.output_norm, D.eps);
    ggml_tensor * best = ggml_argmax(ctx, ggml_mul_mat(ctx, D.tok_embd, cur));   // the draft: top-1 of its logits
    ggml_tensor * h_next = ggml_mul_mat(ctx, D.post_proj, cur);
    ggml_set_output(best); ggml_set_output(h_next);
    ggml_build_forward_expand(gf, best);
    ggml_build_forward_expand(gf, h_next);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) { ggml_free(ctx); return "draft graph allocation failed"; }
    ggml_backend_tensor_set(t_tok, &tok, 0, sizeof(int32_t));
    ggml_backend_tensor_set(t_h, h, 0, nb * sizeof(float));
    ggml_backend_tensor_set(t_pos, &pos0, 0, sizeof(int32_t));
    auto mask = [&](const Cache & C, uint32_t n_kv, ggml_tensor * m) {
        std::vector<ggml_fp16_t> md(n_kv);
        for (uint32_t j = 0; j < n_kv; ++j) {
            const int32_t p = C.pos[j];
            md[j] = ggml_fp32_to_fp16((p < 0 || p > pos0 || (C.swa && pos0 - p >= hp_.n_swa)) ? -INFINITY : 0.0f);
        }
        ggml_backend_tensor_set(m, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
    };
    mask(base_, nkv_b, m_b); mask(swa_, nkv_s, m_s);
    for (ggml_tensor * r : { dk_rot, dv_rot, dk_rot_swa, dv_rot_swa }) {
        if (!r) continue;
        const std::vector<float> h = gen_hadamard((int) r->ne[0]);
        ggml_backend_tensor_set(r, h.data(), 0, h.size() * sizeof(float));
    }
    const ggml_status st = ggml_backend_sched_graph_compute(sched_, gf);
    if (st != GGML_STATUS_SUCCESS) { ggml_free(ctx); return "draft compute failed"; }
    ggml_backend_tensor_get(best, out, 0, sizeof(int32_t));
    h_out->resize(nb);
    ggml_backend_tensor_get(h_next, h_out->data(), 0, nb * sizeof(float));
    ggml_free(ctx);
    return "";
}

}  // namespace g4
