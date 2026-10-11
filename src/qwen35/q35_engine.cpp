// Strata-Lite qwen35moe engine, stage b. See q35_engine.hpp. Every graph-building step below names the llama.cpp
// function it mirrors (the experiment's copy, b6b003d: src/models/qwen35moe.cpp, src/models/delta-net-base.cpp,
// src/llama-graph.cpp, src/llama-kv-cache.cpp); the order of the ggml calls is kept, because the CUDA backend fuses
// adjacent nodes and the scheduler splits by node order. Node names follow llama.cpp's callbacks ("<name>-<layer>").
#include "q35_engine.hpp"

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace q35 {

static constexpr int   kMaxNodes   = 16384;
// rope parameters as llama.cpp's cparams resolve them for this model (no rope scaling: ext_factor 0)
static constexpr int   kRopeType   = GGML_ROPE_TYPE_IMROPE;
static constexpr float kExtFactor  = 0.0f;
static constexpr float kAttnFactor = 1.0f;
static constexpr float kBetaFast   = 32.0f;
static constexpr float kBetaSlow   = 1.0f;
// ggml-cuda offloads an op on CPU-resident weights to the GPU from this many tokens: below it llama.cpp computes the
// experts on the CPU, and so does the pool
static constexpr int   kOffloadMin = 32;
// the slot path's misses all point at one zero slot, so a token's ids repeat: ggml-cuda's mat-vec MoE kernel (<= 8
// tokens) takes that, its batched MMQ kernel does not; 9-31 token ubatches go through the pool for every expert
static constexpr int   kSlotMaxTokens = 8;
static constexpr int   kMinSlots = 8;          // below this the arena is not worth its VRAM
// Auto chunk: a read's compute buffers ~ fixed + per token + the KQ mask (chunk x context, f16); calibrated on the 5070
// with Qwen's graph (stage g)
static constexpr size_t kChunkFixed  = (size_t) 768 << 20;
static constexpr size_t kChunkPerTok = 200 * 1024;
static constexpr int    kChunkMin    = 4096;

namespace scpu = strata::kernels::cpu;

// q4 KV: llama.cpp rotates K (and Q) and V by orthonormal Walsh-Hadamard matrices before quantising (attn_rot_k/v,
// llama-kv-cache.cpp), and V's rotation back after attention. ggml_gen_hadamard:
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
// llama_kv_cache::build_input_k_rot: the largest power of two (from 64) dividing the head size; V always 64
static int k_rot_size(int head) { int r = 64; do { r *= 2; } while (head % r == 0); return r / 2; }
static std::mutex g_pool_mu;   // the ExpertPool is not reentrant

// ---------------------------------------------------------------- stage c: the experts through Strata's pool
struct Engine::PoolLayer {
    scpu::NativeFmt f;
    const uint8_t * gate = nullptr; size_t g_stride = 0;   // expert e's gate rows: gate + e*g_stride; its up rows f.up_off on
    const uint8_t * dn = nullptr;   size_t d_stride = 0;   // its down rows
    scpu::ExpertPool * pool = nullptr;
    const int * res = nullptr;                             // stage d: slot of each expert, -1 = not resident (nullptr: none)
    Stats * stats = nullptr;
    std::vector<uint8_t> act;
    std::vector<scpu::ExpertJobMulti> jobs;
    // overlap: the misses' rows, written by the pool on the driver thread, copied out by the join op
    std::vector<float> out;
    std::mutex mu; std::condition_variable cv; bool done = true;
    Engine::Driver * driver = nullptr;
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
            const auto t0 = std::chrono::steady_clock::now();
            { std::lock_guard<std::mutex> g(g_pool_mu); L->pool->run_split_multi_native(L->f, L->jobs.data(), (int) L->jobs.size()); }
            if (L->stats) L->stats->pool_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            { std::lock_guard<std::mutex> g(L->mu); L->done = true; }
            L->cv.notify_all();
        }
    }
};

// overlap, start: the routing and the activations are here (the scheduler synced the router for them); quantize, plan
// the misses into L->out, hand them to the driver thread and return at once. dst is a 1-element token for ordering.
static void moe_start_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
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
                jb.blob = L->gate + (size_t) e * L->g_stride;
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
    L->stats->join_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    memcpy(dst->data, L->out.data(), ggml_nbytes(dst));
}

// x [n_embd, n] f32, ids [k, n] i32 (argsort_top_k view) -> dst [n_embd, k, n]: exactly what build_moe_ffn's
// mul_mat_id (up) + mul_mat_id (gate) -> swiglu -> mul_mat_id (down) gives on ggml-cpu for a RAM-resident layer below
// the offload size (same quantizers, dots and SwiGLU). The router weights and the sum over k stay ggml ops after it.
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
                jb.blob = L->gate + (size_t) e * L->g_stride;
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
    if (!L->jobs.empty()) { std::lock_guard<std::mutex> g(g_pool_mu); L->pool->run_split_multi_native(L->f, L->jobs.data(), (int) L->jobs.size()); }
}

// the slot arena of one cached layer
struct Engine::CacheLayer {
    bool full = false;                  // every expert resident, identity order: exactly llama.cpp's GPU-layer graph
    int  S = 0;                         // resident slots (the tensors have S + 1: slot S is all zeros)
    std::vector<ggml_tensor *> src, slot, all;   // per expert tensor: the RAM source, its slot tensor, the dcheck twin
    ggml_tensor * res_f = nullptr;      // expert -> slot (S for a miss), f32 for the graph's get_rows
    std::vector<int> res;               // expert -> slot, -1 = in RAM only
    std::vector<int> owner;             // slot -> expert
    std::vector<double> score;          // decayed usage counts (stage e)
    std::vector<char> pending;          // expert with a copy in flight (async swaps)
    std::unique_ptr<PoolLayer> miss;    // the pool for the misses (res set)
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
    std::vector<int64_t> out;
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(g, id);
    const void * d = gguf_get_arr_data(g, id);
    switch (gguf_get_arr_type(g, id)) {
        case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; ++i) out.push_back(((const int32_t *) d)[i]); break;
        case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; ++i) out.push_back(((const uint32_t *) d)[i]); break;
        default: break;
    }
    return out;
}

// ---------------------------------------------------------------- KV cells (llama_kv_cache, one sequence, no SWA)
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
bool Engine::Cache::find_slot(uint32_t n_tokens) {
    idxs.clear();
    uint32_t head_cur = head;
    if (head_cur > used() + 2 * n_tokens) head_cur = 0;
    if (n_tokens > size) return false;
    uint32_t n_tested = 0;
    while (true) {
        if (head_cur + 1 > size) { n_tested += size - head_cur; head_cur = 0; continue; }
        const uint32_t idx = head_cur++;
        n_tested++;
        if (pos[idx] < 0) idxs.push_back(idx);
        if (idxs.size() == n_tokens) break;
        if (n_tested >= size) return false;
    }
    return true;
}
// llama_kv_cache::apply_ubatch + get_n_kv
void Engine::Cache::apply(const std::vector<int32_t> & positions) {
    for (size_t i = 0; i < idxs.size(); ++i) pos[idxs[i]] = positions[i];
    head = (uint32_t) idxs.back() + 1;
    const uint32_t pad = 256, um = used_max_p1();
    n_kv = std::min(size, std::max(pad, (um + pad - 1) / pad * pad));
}

// llama_kv_cache::seq_rm(seq, p0, -1): free the cells at p0 and later; the search restarts at the first freed one
void Engine::Cache::seq_rm(int32_t p0) {
    uint32_t new_head = size;
    for (uint32_t i = 0; i < size; ++i)
        if (pos[i] >= p0) { pos[i] = -1; if (new_head == size) new_head = i; }
    if (new_head != size && new_head < head) head = new_head;
}

// ---------------------------------------------------------------- load
Engine::Engine() = default;

Engine::~Engine() {
    if (mtp_.sched) ggml_backend_sched_free(mtp_.sched);
    if (mtp_.buf_kv) ggml_backend_buffer_free(mtp_.buf_kv);
    if (mtp_.buf_gpu) ggml_backend_buffer_free(mtp_.buf_gpu);
    if (mtp_.buf_map) ggml_backend_buffer_free(mtp_.buf_map);
    if (mtp_.ctx_kv) ggml_free(mtp_.ctx_kv);
    if (mtp_.ctx_gpu) ggml_free(mtp_.ctx_gpu);
    if (mtp_.ctx_w) ggml_free(mtp_.ctx_w);
    if (mtp_.meta) ggml_free(mtp_.meta);
    if (mtp_.gguf) gguf_free(mtp_.gguf);
    if (mtp_.map) munmap(mtp_.map, mtp_.map_size);
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); cudaStreamDestroy((cudaStream_t) copy_stream_); }
    for (auto & b : pending_) cudaEventDestroy((cudaEvent_t) b.ev);
    for (void * c : ckpt_) if (c) cudaFree(c);
    driver_.reset();
    if (sched_) ggml_backend_sched_free(sched_);
    if (buf_cache_) ggml_backend_buffer_free(buf_cache_);
    if (ctx_cache_) ggml_free(ctx_cache_);
    if (buf_kv_) ggml_backend_buffer_free(buf_kv_);
    if (buf_gpu_) ggml_backend_buffer_free(buf_gpu_);
    if (buf_map_) ggml_backend_buffer_free(buf_map_);
    if (buf_arena_) ggml_backend_buffer_free(buf_arena_);
    if (arena_) cudaFreeHost(arena_);
    if (ctx_kv_) ggml_free(ctx_kv_);
    if (ctx_gpu_) ggml_free(ctx_gpu_);
    if (ctx_w_) ggml_free(ctx_w_);
    if (meta_) ggml_free(meta_);
    if (gguf_) gguf_free(gguf_);
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
    if (map_) munmap(map_, map_size_);
}

std::string Engine::load(const Options & o) {
    o_ = o;
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta_ };
    gguf_ = gguf_init_from_file(o.gguf.c_str(), gp);
    if (!gguf_) return "cannot read " + o.gguf;
    {
        const int64_t id = gguf_find_key(gguf_, "general.architecture");
        const std::string arch = id < 0 ? "" : gguf_get_val_str(gguf_, id);
        if (arch != "qwen35moe") return "not a qwen35moe GGUF (" + arch + ")";
    }
    const std::string A = "qwen35moe.";
    HParams & H = hp_;
    H.n_layer       = (int) kv_int(gguf_, A + "block_count");
    // an MTP head (nextn_predict_layers) is an extra block the main pass never runs (qwen35moe.cpp: n_layer() excludes it)
    H.n_layer      -= (int) kv_int(gguf_, A + "nextn_predict_layers", 0);
    H.n_embd        = (int) kv_int(gguf_, A + "embedding_length");
    H.n_head        = (int) kv_int(gguf_, A + "attention.head_count");
    H.n_head_kv     = (int) kv_int(gguf_, A + "attention.head_count_kv");
    H.head_dim      = (int) kv_int(gguf_, A + "attention.key_length");
    H.n_rot         = (int) kv_int(gguf_, A + "rope.dimension_count");
    H.n_ctx_train   = (int) kv_int(gguf_, A + "context_length");
    H.n_expert      = (int) kv_int(gguf_, A + "expert_count");
    H.n_expert_used = (int) kv_int(gguf_, A + "expert_used_count");
    H.n_ff_exp      = (int) kv_int(gguf_, A + "expert_feed_forward_length");
    H.n_ff_shexp    = (int) kv_int(gguf_, A + "expert_shared_feed_forward_length");
    H.ssm_d_conv    = (int) kv_int(gguf_, A + "ssm.conv_kernel");
    H.ssm_d_inner   = (int) kv_int(gguf_, A + "ssm.inner_size");
    H.ssm_d_state   = (int) kv_int(gguf_, A + "ssm.state_size");
    H.ssm_n_group   = (int) kv_int(gguf_, A + "ssm.group_count");
    H.ssm_dt_rank   = (int) kv_int(gguf_, A + "ssm.time_step_rank");
    H.full_attn_interval = (int) kv_int(gguf_, A + "full_attention_interval", 4);
    H.rope_base     = kv_f32(gguf_, A + "rope.freq_base");
    H.eps           = kv_f32(gguf_, A + "attention.layer_norm_rms_epsilon");
    {
        const auto s = kv_arr(gguf_, A + "rope.dimension_sections");
        for (size_t i = 0; i < 4 && i < s.size(); ++i) H.rope_sections[i] = (int) s[i];
    }
    {
        const int64_t id = gguf_find_key(gguf_, "tokenizer.ggml.tokens");
        H.n_vocab = id < 0 ? 0 : (int) gguf_get_arr_n(gguf_, id);
    }
    // qwen35moe load_arch_hparams: recurrent unless every full_attention_interval-th layer
    H.is_recr.resize(H.n_layer);
    for (int il = 0; il < H.n_layer; ++il) H.is_recr[il] = (il + 1) % H.full_attn_interval != 0;
    if (H.n_layer <= 0 || H.n_embd <= 0 || H.head_dim <= 0 || H.ssm_d_state <= 0 || H.n_expert <= 0) return "unexpected qwen35moe hparams";

    // backends: [CUDA0, CPU], the CPU's compute buffers in the GPU's pinned host memory (llama_context)
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) return "no GPU device";
    gpu_ = ggml_backend_dev_init(gpu_dev, nullptr);
    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu_ || !cpu_) return "backend init failed";
    ggml_backend_cpu_set_n_threads(cpu_, o.n_threads);

    // the file, mapped: the CPU-resident weights are used in place (llama.cpp with mmap)
    const int fd = open(o.gguf.c_str(), O_RDONLY);
    if (fd < 0) return "open failed";
    struct stat sb; fstat(fd, &sb); map_size_ = sb.st_size;
    map_ = mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (map_ == MAP_FAILED) { map_ = nullptr; return "mmap failed"; }
    const size_t data_off = gguf_get_data_offset(gguf_);

    // `-ngl 99 --n-cpu-moe N`: token_embd (the input layer) and the first N layers' routed experts in RAM, the rest on
    // the GPU. The output head is its own tensor here (not tied).
    const int n_tensors = (int) gguf_get_n_tensors(gguf_);
    ggml_init_params ip = { (size_t) (n_tensors + 8) * ggml_tensor_overhead(), nullptr, true };
    ctx_w_ = ggml_init(ip);
    ctx_gpu_ = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, size_t>> on_cpu, on_gpu;
    auto make = [&](const std::string & name, bool cpu) -> ggml_tensor * {
        ggml_tensor * m = ggml_get_tensor(meta_, name.c_str());
        if (!m) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(cpu ? ctx_w_ : ctx_gpu_, m);
        ggml_set_name(t, name.c_str());
        const size_t off = data_off + gguf_get_tensor_offset(gguf_, gguf_find_tensor(gguf_, name.c_str()));
        (cpu ? on_cpu : on_gpu).push_back({ t, off });
        return t;
    };
    tok_embd_    = make("token_embd.weight", true);
    output_      = make("output.weight", false);
    if (!output_) output_ = make("token_embd.weight", false);   // tied (llama.cpp: TENSOR_DUPLICATED)
    output_norm_ = make("output_norm.weight", false);
    if (!tok_embd_ || !output_ || !output_norm_) return "missing token_embd / output / output_norm";
    L_.assign(H.n_layer, Layer{});
    for (int il = 0; il < H.n_layer; ++il) {
        auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
        Layer & l = L_[il];
        const bool exps_cpu = il < o.n_cpu_moe;
        l.attn_norm      = make(b("attn_norm.weight"), false);
        l.attn_post_norm = make(b("post_attention_norm.weight"), false);
        if (!H.is_recr[il]) {
            l.wq     = make(b("attn_q.weight"), false);
            l.wk     = make(b("attn_k.weight"), false);
            l.wv     = make(b("attn_v.weight"), false);
            l.wo     = make(b("attn_output.weight"), false);
            l.q_norm = make(b("attn_q_norm.weight"), false);
            l.k_norm = make(b("attn_k_norm.weight"), false);
            if (!l.wq || !l.wk || !l.wv || !l.wo || !l.q_norm || !l.k_norm) return "layer " + std::to_string(il) + ": missing attention tensors";
        } else {
            l.wqkv       = make(b("attn_qkv.weight"), false);
            l.wqkv_gate  = make(b("attn_gate.weight"), false);
            l.ssm_conv1d = make(b("ssm_conv1d.weight"), false);
            l.ssm_dt     = make(b("ssm_dt.bias"), false);
            l.ssm_a      = make(b("ssm_a"), false);
            l.ssm_beta   = make(b("ssm_beta.weight"), false);
            l.ssm_alpha  = make(b("ssm_alpha.weight"), false);
            l.ssm_norm   = make(b("ssm_norm.weight"), false);
            l.ssm_out    = make(b("ssm_out.weight"), false);
            if (!l.wqkv || !l.wqkv_gate || !l.ssm_conv1d || !l.ssm_dt || !l.ssm_a || !l.ssm_beta || !l.ssm_alpha || !l.ssm_norm || !l.ssm_out)
                return "layer " + std::to_string(il) + ": missing delta-net tensors";
        }
        l.gate_inp       = make(b("ffn_gate_inp.weight"), false);
        l.gate_exps      = make(b("ffn_gate_exps.weight"), exps_cpu);
        l.up_exps        = make(b("ffn_up_exps.weight"), exps_cpu);
        l.down_exps      = make(b("ffn_down_exps.weight"), exps_cpu);
        l.gate_inp_shexp = make(b("ffn_gate_inp_shexp.weight"), false);
        l.gate_shexp     = make(b("ffn_gate_shexp.weight"), false);
        l.up_shexp       = make(b("ffn_up_shexp.weight"), false);
        l.down_shexp     = make(b("ffn_down_shexp.weight"), false);
        if (!l.attn_norm || !l.attn_post_norm || !l.gate_inp || !l.gate_exps || !l.up_exps || !l.down_exps || !l.up_shexp)
            return "layer " + std::to_string(il) + ": missing tensors";
    }
    buf_gpu_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gpu_, ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_gpu_) return "GPU weight allocation failed";
    ggml_backend_buffer_set_usage(buf_gpu_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // the scheduler places ops by their weights
    for (auto & [t, off] : on_gpu) ggml_backend_tensor_set(t, (const char *) map_ + off, 0, ggml_nbytes(t));
    buf_map_ = ggml_backend_cpu_buffer_from_ptr(map_, map_size_);
    ggml_backend_buffer_set_usage(buf_map_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // with async swaps the RAM-resident experts live in a page-locked arena (Strata's expert arena): the copy stream
    // needs pinned sources, and a read-only file mapping cannot be registered. Same bytes, so the arithmetic is unchanged
    // (the pool and ggml's prompt path read the arena too)
    auto in_arena = [&](ggml_tensor * t) { return o.async_swaps && strstr(ggml_get_name(t), "_exps.") != nullptr; };
    if (o.async_swaps) {
        for (auto & [t, off] : on_cpu) if (in_arena(t)) arena_size_ += GGML_PAD(ggml_nbytes(t), 4096);
        if (cudaHostAlloc(&arena_, arena_size_, cudaHostAllocPortable) != cudaSuccess) { cudaGetLastError(); return "cannot allocate the pinned expert arena"; }
        buf_arena_ = ggml_backend_cpu_buffer_from_ptr(arena_, arena_size_);
        ggml_backend_buffer_set_usage(buf_arena_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
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

    // the hybrid memory (llama_memory_hybrid): a KV cache for the attention layers (n_ctx cells, K and V [n_embd_gqa,
    // cells], V not transposed: flash attention), and one recurrent cell per delta-net layer (conv state [n_embd_r],
    // ssm state [n_embd_s], f32; with n_rs_seq, 1 + n_rs_seq rows each: row s = the state s tokens back)
    kv_.size = (uint32_t) ((o.n_ctx + 255) / 256 * 256);
    kv_.pos.assign(kv_.size, -1);
    kv_.k.assign(H.n_layer, nullptr); kv_.v.assign(H.n_layer, nullptr);
    r_l_.assign(H.n_layer, nullptr); s_l_.assign(H.n_layer, nullptr);
    ggml_init_params kp = { (size_t) (4 * H.n_layer + 8) * ggml_tensor_overhead(), nullptr, true };
    ctx_kv_ = ggml_init(kp);
    const int64_t n_gqa = (int64_t) H.head_dim * H.n_head_kv;
    for (int il = 0; il < H.n_layer; ++il) {
        if (H.is_recr[il]) {
            r_l_[il] = ggml_new_tensor_2d(ctx_kv_, GGML_TYPE_F32, H.n_embd_r(), 1 + o.n_rs_seq);
            s_l_[il] = ggml_new_tensor_2d(ctx_kv_, GGML_TYPE_F32, H.n_embd_s(), 1 + o.n_rs_seq);
            ggml_format_name(r_l_[il], "cache_r_l%d", il);
            ggml_format_name(s_l_[il], "cache_s_l%d", il);
        } else {
            kv_.k[il] = ggml_new_tensor_3d(ctx_kv_, o.type_k, n_gqa, kv_.size, 1);
            kv_.v[il] = ggml_new_tensor_3d(ctx_kv_, o.type_v, n_gqa, kv_.size, 1);
            ggml_format_name(kv_.k[il], "cache_k_l%d", il);
            ggml_format_name(kv_.v[il], "cache_v_l%d", il);
        }
    }
    buf_kv_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_kv_, ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_kv_) return "KV / recurrent state allocation failed";
    ggml_backend_buffer_clear(buf_kv_, 0);

    if (o.pool) {   // stage c: Strata's CPU expert pool for the RAM-resident layers
        pool_ = std::make_unique<scpu::ExpertPool>(o.pool_threads, /*pin*/ false);   // never pin: machine A runs production
        pool_layers_.resize(H.n_layer);
        for (int il = 0; il < H.n_layer && il < o.n_cpu_moe; ++il) {
            const Layer & l = L_[il];
            // the kernel reads up rows at a fixed distance from the gate rows: separate gate/up tensors qualify when they
            // share type and stride (else the layer keeps ggml's CPU path)
            if (l.gate_exps->type != l.up_exps->type || l.gate_exps->nb[2] != l.up_exps->nb[2]) continue;
            auto pl = std::make_unique<PoolLayer>();
            std::string e;
            if (!scpu::native_fmt(l.gate_exps->type, l.down_exps->type, H.n_embd, H.n_ff_exp, pl->f, e,
                                  scpu::NativeAct::SwiGLU_ggml, /*separate_down*/ true))
                return "layer " + std::to_string(il) + ": " + e;
            pl->gate = (const uint8_t *) l.gate_exps->data; pl->g_stride = l.gate_exps->nb[2];
            pl->f.up_off = (size_t) ((const uint8_t *) l.up_exps->data - pl->gate);   // may wrap: the sum is the up rows
            pl->dn = (const uint8_t *) l.down_exps->data;   pl->d_stride = l.down_exps->nb[2];
            pl->pool = pool_.get();
            if (pl->g_stride != (size_t) H.n_ff_exp * pl->f.gu_row || pl->d_stride != pl->f.d_row * (size_t) H.n_embd)
                return "layer " + std::to_string(il) + ": expert strides do not match the pool's format";
            pool_layers_[il] = std::move(pl);
        }
    }

    if (o.cache_slots != 0) {   // stage d
        if (!o.pool && o.cache_slots < H.n_expert) return "--cache-slots needs --pool (the misses go to the pool)";
        rank_.assign(H.n_layer, {});
        if (o.profile.empty()) {
            for (auto & r : rank_) for (int e = 0; e < H.n_expert; ++e) r.push_back(e);
        } else {
            FILE * pf = fopen(o.profile.c_str(), "r");
            if (!pf) return "cannot read the profile " + o.profile;
            char line[16384];
            while (fgets(line, sizeof line, pf)) {
                char * q = line; const int il = (int) strtol(q, &q, 10);
                if (*q == ':') ++q;
                if (il < 0 || il >= H.n_layer) continue;
                for (;;) { char * e2; const long v = strtol(q, &e2, 10); if (e2 == q) break; rank_[il].push_back((int) v); q = e2; }
            }
            fclose(pf);
        }
        cache_.resize(H.n_layer);
        for (int il = std::max(0, o.cache_l0); il < std::min(o.n_cpu_moe, H.n_layer); ++il) {
            const Layer & l = L_[il];
            auto C = std::make_unique<CacheLayer>();
            C->full = o.cache_slots >= H.n_expert;
            C->src = { l.gate_exps, l.up_exps, l.down_exps };
            C->res.assign(H.n_expert, -1);
            C->score.assign(H.n_expert, 0.0);
            C->pending.assign(H.n_expert, 0);
            if (!C->full) {
                if (!pool_layers_[il]) return "layer " + std::to_string(il) + ": no pool layer for the misses";
                if ((int) rank_[il].size() < std::min(o.cache_slots < 0 ? kMinSlots : o.cache_slots, H.n_expert))
                    return "profile: layer " + std::to_string(il) + " ranks too few experts";
            }
            cache_[il] = std::move(C);
        }
        // auto: a small arena first; sized from the VRAM left once everything else has loaded (end of load)
        const std::string ae = arena_build(o.cache_slots < 0 ? kMinSlots : o.cache_slots, true);
        if (!ae.empty()) return ae;
        for (int il = 0; il < H.n_layer; ++il) {
            CacheLayer * C = cache_[il].get();
            if (!C || C->full) continue;
            const PoolLayer & P = *pool_layers_[il];
            C->miss = std::make_unique<PoolLayer>();
            C->miss->f = P.f; C->miss->gate = P.gate; C->miss->g_stride = P.g_stride; C->miss->dn = P.dn; C->miss->d_stride = P.d_stride;
            C->miss->pool = P.pool; C->miss->res = C->res.data(); C->miss->stats = &stats_;
        }
    }
    if (o.overlap) {
        if (cache_.empty()) return "--overlap needs --cache-slots";
        driver_ = std::make_unique<Driver>();
        for (auto & C : cache_) if (C && C->miss) C->miss->driver = driver_.get();
    }
    if (o.async_swaps) {   // the copy stream: expert bytes from the pinned arena, beside the compute stream
        if (!o.adapt) return "--async-swaps needs --adapt";
        cudaStream_t cs;
        if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) return "cannot create the copy stream";
        copy_stream_ = cs;
    }

    new_sched();
    graph_buf_.resize(ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false));
    {
        const std::string e = reserve_graphs();
        if (!e.empty()) return e;
    }
    mtp_h_prev_.assign(H.n_embd, 0.0f);   // draft-mtp's pending_h starts at zeros
    if (!o.mtp.empty()) {
        const std::string e = load_mtp();
        if (!e.empty()) return "mtp: " + e;
    }
    for (int il = 0; il < H.n_layer; ++il) if (H.is_recr[il]) ckpt_bytes_ += r_l_[il]->nb[1] + s_l_[il]->nb[1];
    for (void *& c : ckpt_)
        if (cudaMalloc(&c, ckpt_bytes_) != cudaSuccess) { cudaGetLastError(); c = nullptr; return "cannot allocate the recurrent checkpoints"; }
    if (o.cache_slots < 0 && !cache_.empty()) {   // auto slots: what the free VRAM holds after everything above, less the reserve
        size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
        const size_t per = slot_bytes(), have = (size_t) (slots_now_ + 1) * per, reserve = (size_t) o.vram_reserve_mb << 20;
        int S = fr + have > reserve + 2 * per ? (int) ((fr + have - reserve) / per) - 1 : kMinSlots;
        S = std::max(kMinSlots, std::min(S, H.n_expert - 1));
        std::string err;
        while (!(err = arena_build(S, true)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 2);
        if (!err.empty()) return "auto slots: " + err;
    }
    return "";
}

size_t Engine::slot_bytes() const {
    size_t b = 0;
    for (const auto & C : cache_) if (C && !C->full) for (ggml_tensor * t : C->src) b += t->nb[2];
    return b;
}

// VRAM for something else: fewer slots (refilled from the usage counts unless keep_profile)
int Engine::ensure_free(size_t bytes, bool keep_profile) {
    if (!buf_cache_ || slot_bytes() == 0) return slots_now_;
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    if (fr >= bytes) return slots_now_;
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
    const size_t per = slot_bytes();
    int S = std::max(kMinSlots, slots_now_ - (int) ((bytes - fr + per - 1) / per) - 1);
    const int before = slots_now_;
    std::string err;
    while (!(err = arena_build(S, keep_profile)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 2);
    if (!err.empty()) { fatal_ = "making room: " + err; return 0; }
    if (S < before) { stats_.arena_shrinks++; fprintf(stderr, "strata: expert arena %d -> %d slots per layer (VRAM)\n", before, S); }
    if (keep_profile) return slots_now_;
    refill_pending_ = true;   // the next small ubatch refills every layer to its top-S by usage
    read_end_ = std::chrono::steady_clock::now(); refill_watch_ = true;
    return slots_now_;
}

int Engine::grow_to(int S, size_t headroom) {
    if (!buf_cache_ || slot_bytes() == 0 || S <= slots_now_) return slots_now_;
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    const size_t per = slot_bytes(), now = (size_t) (slots_now_ + 1) * per;
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

// ---------------------------------------------------------------- the prompt cache
int Engine::resume_point(int pos) const {
    if (pos >= n_past_) return n_past_;
    const int r = n_past_ - pos;
    if (rs_src_ == 0 && r <= o_.n_rs_seq && r < last_ub_n_) return pos;   // inside the last ubatch's snapshots
    int best = 0;
    for (int c = 0; c < kCkpts; ++c) if (ckpt_pos_[c] > best && ckpt_pos_[c] <= pos) best = ckpt_pos_[c];
    return best;
}

std::string Engine::rollback_to(int pos) {
    if (pos == n_past_) return "";
    if (pos == 0) { reset(); return ""; }
    const int r = n_past_ - pos;
    if (r > 0 && rs_src_ == 0 && r <= o_.n_rs_seq && r < last_ub_n_) return rollback(pos);
    int c = 0;
    while (c < kCkpts && !(ckpt_[c] && ckpt_pos_[c] == pos)) ++c;
    if (c == kCkpts) return "cannot resume at position " + std::to_string(pos);
    size_t off = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_recr[il]) continue;
        cudaMemcpy(r_l_[il]->data, (char *) ckpt_[c] + off, r_l_[il]->nb[1], cudaMemcpyDeviceToDevice); off += r_l_[il]->nb[1];
        cudaMemcpy(s_l_[il]->data, (char *) ckpt_[c] + off, s_l_[il]->nb[1], cudaMemcpyDeviceToDevice); off += s_l_[il]->nb[1];
    }
    rs_src_ = 0; last_ub_n_ = 0;
    kv_.seq_rm(pos);
    for (int k = 0; k < kCkpts; ++k) if (ckpt_pos_[k] > pos) ckpt_pos_[k] = -1;   // their attention cells are gone
    if (mtp_.loaded) { mtp_.kv.seq_rm(pos); mtp_h_prev_ = ckpt_h_[c]; }
    n_past_ = pos;
    return "";
}

// the recurrent state now (its newest snapshot, or the one a pending rollback points at), the drafter's pending row
void Engine::checkpoint() {
    int c = 0;   // the same position again, else the oldest
    for (int k = 0; k < kCkpts; ++k) if (ckpt_pos_[k] == n_past_) { c = k; goto found; }
    for (int k = 1; k < kCkpts; ++k) if (ckpt_age_[k] < ckpt_age_[c]) c = k;
found:
    if (!ckpt_[c]) return;
    size_t off = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_recr[il]) continue;
        cudaMemcpy((char *) ckpt_[c] + off, (char *) r_l_[il]->data + (size_t) rs_src_ * r_l_[il]->nb[1], r_l_[il]->nb[1], cudaMemcpyDeviceToDevice); off += r_l_[il]->nb[1];
        cudaMemcpy((char *) ckpt_[c] + off, (char *) s_l_[il]->data + (size_t) rs_src_ * s_l_[il]->nb[1], s_l_[il]->nb[1], cudaMemcpyDeviceToDevice); off += s_l_[il]->nb[1];
    }
    ckpt_pos_[c] = n_past_;
    ckpt_age_[c] = ++ckpt_clock_;
    ckpt_h_[c] = mtp_h_prev_;
}

// an image chunk: one ubatch (a big one streams the experts through the card like a text read, so it borrows the
// arena's VRAM the same way)
std::string Engine::decode_embd(const float * embd, int n, const int32_t * pos4, int n_pos, std::vector<float> * logits) {
    const bool big = n > o_.n_ubatch || (o_.lend && n >= kOffloadMin);
    if (big) begin_read(std::max(n, o_.n_ubatch));
    embd_in_ = embd; pos4_in_ = pos4;
    const std::string err = ubatch(nullptr, n, n_past_, logits ? 1 : 0, logits);
    embd_in_ = nullptr; pos4_in_ = nullptr;
    if (!err.empty()) return err;
    n_past_ += n_pos;
    last_ub_n_ = 0;   // no snapshot rollback into an image
    return "";
}

// [CUDA0, CPU], the CPU's compute buffers in the GPU's pinned host memory (llama_context)
void Engine::new_sched() {
    if (sched_) ggml_backend_sched_free(sched_);
    ggml_backend_t backs[2] = { gpu_, cpu_ };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(gpu_), ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu_)) };
    if (!bufts[1]) bufts[1] = ggml_backend_get_default_buffer_type(cpu_);
    sched_ = ggml_backend_sched_new(backs, bufts, 2, kMaxNodes, /*parallel*/ false, /*op_offload*/ true);
}

// llama_context::sched_reserve: pp (n_ubatch tokens, n_outputs = min(n_ubatch, n_batch)), tg (1, 1), pp again. Run at load
// and again after a chunked read, so small steps get the same allocation plan (and ggml-cuda fusions) as llama.cpp's
std::string Engine::reserve_graphs() {
    const int n_pp = std::min(o_.n_ctx, o_.n_ubatch);
    std::vector<int32_t> zeros(n_pp, 0);
    for (auto [nt, no] : { std::pair<int, int>{ n_pp, n_pp }, { 1, 1 }, { n_pp, n_pp } }) {
        const std::string e = ubatch(zeros.data(), nt, 0, no, nullptr, no);
        if (!e.empty()) return "reserve: " + e;
    }
    return "";
}

// ---------------------------------------------------------------- stage g: chunked reads with VRAM lent from the arena
void Engine::begin_read(int ub) {
    (void) ub;
    const auto t0 = std::chrono::steady_clock::now();
    reading_ = true;
    if (o_.lend && buf_cache_) {   // give the arena's VRAM to the read
        if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
        stats_.lent_bytes = ggml_backend_buffer_get_size(buf_cache_);
        ggml_backend_buffer_free(buf_cache_); buf_cache_ = nullptr;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_cache_); t; t = ggml_get_next_tensor(ctx_cache_, t)) { t->data = nullptr; t->buffer = nullptr; }
        for (auto & C : cache_) if (C) {
            std::fill(C->res.begin(), C->res.end(), -1); std::fill(C->owner.begin(), C->owner.end(), -1);
            std::fill(C->pending.begin(), C->pending.end(), 0);
        }
        new_sched();   // its compute buffers grow for the big ubatches
    }
    stats_.lend_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void Engine::end_read() {
    const auto t0 = std::chrono::steady_clock::now();
    reading_ = false;
    new_sched();   // drop the big ubatches' compute buffers
    if (o_.lend && !buf_cache_ && ctx_cache_) {   // the arena comes back empty; the refill copies in the background
        // the read may leave less VRAM than it lent (ggml-cuda's pool keeps what the big ubatches grew): fewer slots then
        int S = slots_now_;
        const int before = S;
        std::string err;
        while (!(err = arena_build(S, false)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 4);
        if (!err.empty()) { fatal_ = "after the read: " + err; return; }
        if (S < before) { stats_.arena_shrinks++; fprintf(stderr, "strata: expert arena %d -> %d slots per layer (VRAM)\n", before, S); }
    }
    const std::string e = reserve_graphs();
    if (!e.empty()) { fatal_ = "after the read: " + e; return; }
    read_end_ = std::chrono::steady_clock::now();
    refill_watch_ = true;
    stats_.restore_ms += std::chrono::duration<double, std::milli>(read_end_ - t0).count();
}

int Engine::pick_chunk(int n) {
    // what the read can use: free VRAM, the arena it is about to borrow, and (a read already under way, e.g. a text run
    // after an image) the compute buffers the read holds
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    const bool lending = o_.lend && (buf_cache_ || reading_);
    const size_t avail = fr + ((o_.lend && buf_cache_) ? ggml_backend_buffer_get_size(buf_cache_) : 0)
                            + (reading_ ? ggml_backend_sched_get_buffer_size(sched_, gpu_) : 0);
    const size_t n_kv_end = (size_t) n_past_ + n, margin = (size_t) 256 << 20;
    for (int P : { 32768, 16384, 8192, kChunkMin }) {
        const size_t m = (size_t) std::min(P, n);
        if (kChunkFixed + m * kChunkPerTok + m * n_kv_end * 2 + margin <= avail) return P;
    }
    return lending ? kChunkMin : o_.n_ubatch;   // the maintainer, 2026-10-10: minimum chunk 4k
}

// every cached layer gets S slots (+ the zero slot) per expert tensor; with fill_profile the top S of the profile are
// copied in (a full layer: every expert, identity order)
std::string Engine::arena_build(int S, bool fill_profile) {
    if (buf_cache_) { ggml_backend_buffer_free(buf_cache_); buf_cache_ = nullptr; }
    if (ctx_cache_) { ggml_free(ctx_cache_); ctx_cache_ = nullptr; }
    ggml_init_params cp = { (size_t) (12 * hp_.n_layer + 8) * ggml_tensor_overhead(), nullptr, true };
    ctx_cache_ = ggml_init(cp);
    const int NE = hp_.n_expert;
    for (int il = 0; il < (int) cache_.size(); ++il) {
        CacheLayer * C = cache_[il].get();
        if (!C) continue;
        C->S = C->full ? NE : S;
        const int alloc = C->full ? NE : C->S + 1;
        C->slot.clear(); C->all.clear();
        static const char * nm[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
        for (size_t k = 0; k < C->src.size(); ++k) {
            ggml_tensor * t = ggml_new_tensor_3d(ctx_cache_, C->src[k]->type, C->src[k]->ne[0], C->src[k]->ne[1], alloc);
            ggml_format_name(t, "blk.%d.%s.slots", il, nm[k]);
            C->slot.push_back(t);
            if (!C->full && il == o_.dcheck_layer) C->all.push_back(ggml_dup_tensor(ctx_cache_, C->src[k]));
        }
        std::fill(C->res.begin(), C->res.end(), -1);
        C->res_f = nullptr;
        if (C->full) { for (int e = 0; e < NE; ++e) C->res[e] = e; continue; }
        C->owner.assign(C->S, -1);
        C->res_f = ggml_new_tensor_2d(ctx_cache_, GGML_TYPE_F32, 1, NE);
        ggml_format_name(C->res_f, "blk.%d.ffn_exps.slot_of", il);
    }
    buf_cache_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cache_, ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_cache_) return "expert cache allocation failed (" + std::to_string(S) + " slots do not fit on this card)";
    ggml_backend_buffer_clear(buf_cache_, 0);                     // slot S (the misses' target) stays all zeros
    ggml_backend_buffer_set_usage(buf_cache_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int il = 0; il < (int) cache_.size(); ++il) {
        CacheLayer * C = cache_[il].get();
        if (!C) continue;
        for (size_t k = 0; k < C->src.size(); ++k) {
            ggml_tensor * src = C->src[k];
            const size_t es = src->nb[2];
            if (C->full) ggml_backend_tensor_set(C->slot[k], src->data, 0, ggml_nbytes(src));
            else if (fill_profile)
                for (int j = 0; j < std::min(C->S, (int) rank_[il].size()); ++j)
                    ggml_backend_tensor_set(C->slot[k], (const char *) src->data + (size_t) rank_[il][j] * es, (size_t) j * es, es);
            if (!C->all.empty()) ggml_backend_tensor_set(C->all[k], src->data, 0, ggml_nbytes(src));
        }
        if (C->full) continue;
        if (fill_profile) for (int j = 0; j < std::min(C->S, (int) rank_[il].size()); ++j) { C->res[rank_[il][j]] = j; C->owner[j] = rank_[il][j]; }
        std::vector<float> rf(NE);
        for (int e = 0; e < NE; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
        ggml_backend_tensor_set(C->res_f, rf.data(), 0, rf.size() * sizeof(float));
    }
    slots_now_ = S;
    return "";
}

// ---------------------------------------------------------------- stage e: adaptation
// After a ubatch every cached layer's scores decay per token and count its routed experts (Gate 0's lfu_decay rule). A
// prompt read (a big ubatch) only marks a refill: the next small ubatch first re-sets every layer to its top-S, then
// each small ubatch swaps at most adapt_budget experts per layer. The copies are issued after the step's compute and its
// twin check: synchronous, or on the copy stream with async_swaps (committed at the start of a later ubatch).
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
            for (int j = 0; j < k; ++j) {
                const int32_t e = ids[t * ne + j];
                if (e < 0 || e >= hp_.n_expert) {
                    fprintf(stderr, "adapt: layer %d token %d slot %d id %d (argsort %s op %s ne %lld,%lld flags %d)\n", il, t, j, e,
                            argsorts[il]->name, ggml_op_desc(argsorts[il]), (long long) argsorts[il]->ne[0], (long long) argsorts[il]->ne[1],
                            argsorts[il]->flags);
                    abort();
                }
                C->score[e] += 1.0;
            }
        }
    }
    if (n >= kOffloadMin) { refill_pending_ = o_.adapt_refill; return; }
    const bool refill = refill_pending_;
    refill_pending_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t before = stats_.swaps;
    for (int il = 0; il < hp_.n_layer; ++il)
        if (argsorts[il]) swap_layer(il, refill ? INT32_MAX : o_.adapt_budget);
    if (copy_stream_ && !pending_.empty() && !pending_.back().ev) {   // one event closes this step's batch of copies
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
    const int NE = hp_.n_expert;
    std::vector<int> want, worst;
    for (int e = 0; e < NE; ++e) (C->res[e] < 0 ? want : worst).push_back(e);
    std::stable_sort(want.begin(), want.end(), [&](int a, int b) { return C->score[a] > C->score[b]; });
    std::stable_sort(worst.begin(), worst.end(), [&](int a, int b) { return C->score[a] < C->score[b]; });
    if (copy_stream_) {   // async: the victim goes to the CPU now, the newcomer when its copy has landed
        want.erase(std::remove_if(want.begin(), want.end(), [&](int e) { return C->pending[e] != 0; }), want.end());
        if (pending_.empty() || pending_.back().ev) pending_.push_back({ nullptr, {} });
    }
    std::vector<int> free_slots;
    for (int j = 0; j < C->S; ++j) if (C->owner[j] == -1) free_slots.push_back(j);
    size_t wi = 0;
    int done = 0;
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
        for (size_t k = 0; k < C->src.size(); ++k) {
            const size_t es = C->src[k]->nb[2];
            if (copy_stream_)
                cudaMemcpyAsync((char *) C->slot[k]->data + (size_t) slot * es, (const char *) C->src[k]->data + (size_t) a * es, es,
                                cudaMemcpyHostToDevice, (cudaStream_t) copy_stream_);
            else ggml_backend_tensor_set(C->slot[k], (const char *) C->src[k]->data + (size_t) a * es, (size_t) slot * es, es);
            stats_.swap_bytes += es;
        }
        if (w >= 0) C->res[w] = -1;
        if (copy_stream_) {   // a slot being written is referenced by no expert until it commits
            C->owner[slot] = -2; C->pending[a] = 1;
            pending_.back().sw.push_back({ il, a, slot });
        } else { C->res[a] = slot; C->owner[slot] = a; }
        stats_.swaps++;
        done++;
    }
    if (done) {
        std::vector<float> rf(NE);
        for (int e = 0; e < NE; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
        ggml_backend_tensor_set(C->res_f, rf.data(), 0, NE * sizeof(float));
    }
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

void Engine::reset() {
    kv_.pos.assign(kv_.size, -1); kv_.head = 0; kv_.n_kv = 0;
    ggml_backend_buffer_clear(buf_kv_, 0);   // the recurrent cells start at zero (llama.cpp clears a new cell: rs_z)
    rs_fresh_ = true;
    rs_src_ = 0;
    n_past_ = 0;
    last_ub_n_ = 0;
    for (int & c : ckpt_pos_) c = -1;   // their attention cells are gone
    mtp_h_prev_.assign(hp_.n_embd, 0.0f);
    if (mtp_.loaded) { mtp_.kv.pos.assign(mtp_.kv.size, -1); mtp_.kv.head = 0; mtp_.kv.n_kv = 0; }
}

// ---------------------------------------------------------------- the graph
// llm_graph_context::build_norm (LLM_NORM_RMS, weight) + llama_context::graph_get_cb's pin of "norm" nodes to the layer's
// device (full offload)
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

// TDUMP=<file|-> TDUMP_N=<n|-2>: the same named-node hashing (and so the same scheduler split points) as route-oracle's
static bool tdump_want(const ggml_tensor * t) {
    static const char * pfx[] = { "attn_norm-", "Qcur_pos-", "Kcur_pos-", "kqv_out-", "attn_out-", "ffn_mlp-", "ffn_moe_logits-",
                                  "ffn_moe_weights_norm-", "ffn_moe_out-", "l_out-", "result_norm", "result_output", nullptr };
    static const int n = getenv("TDUMP_N") ? atoi(getenv("TDUMP_N")) : -1;
    if (!getenv("TDUMP")) return false;
    bool ok = false;
    for (int i = 0; pfx[i]; ++i) if (strncmp(t->name, pfx[i], strlen(pfx[i])) == 0) ok = true;
    if (!ok) return false;
    const int64_t nt = strncmp(t->name, "Qcur_pos-", 9) == 0 || strncmp(t->name, "Kcur_pos-", 9) == 0 ? t->ne[2] : t->ne[1];
    return (n == -2 || nt == n) && ggml_is_contiguous(t);
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
// Q35_GRAPHDUMP=<file>: every node of the 3rd single-token ubatch, as route-oracle's ROUTE_GRAPHDUMP writes them
static FILE * g_gdump = nullptr;
static bool gdump_cb(ggml_tensor * t, bool ask, void *) {
    if (!ask) return true;
    fprintf(g_gdump, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(g_gdump, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(g_gdump, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(g_gdump, "\n");
    return false;
}
// GDUMP=<file> GDUMP_N=<n>: every node of the ubatch with n tokens (route-oracle's GDUMP format)
static void gdump_multi(const ggml_tensor * t) {
    static const char * path = getenv("GDUMP");
    static const int n = getenv("GDUMP_N") ? atoi(getenv("GDUMP_N")) : -1;
    static bool on = false;
    static FILE * f = nullptr;
    if (!path) return;
    if (!on && strcmp(t->name, "model.input_embed") == 0 && t->ne[1] == n) { on = true; f = fopen(path, "w"); }
    if (!on) return;
    fprintf(f, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(f, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(f, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(f, "\t%d\n", (int) t->flags);
    if (strcmp(t->name, "result_output") == 0) { fclose(f); f = nullptr; on = false; path = nullptr; }
}
static bool tdump_cb(ggml_tensor * t, bool ask, void *) {
    if (ask) gdump_multi(t);
    if (!tdump_want(t)) return ask ? false : true;
    if (ask) return true;
    tdump_put(t);
    return true;
}

// reserve = llama_context::graph_reserve: the graph for n tokens and n_outputs outputs over the full KV cache (init_full),
// handed to ggml_backend_sched_reserve instead of computed. The allocator keeps the plan of the last reserve and reuses
// it for every real graph that fits, so the reserve sequence decides where tensors land (and with it which ggml-cuda
// fusions pass their memory-overlap check).
std::string Engine::ubatch(const int32_t * toks, int n, int pos0, int n_out, std::vector<float> * logits, int reserve_outputs) {
    const HParams & H = hp_;
    const bool reserve = reserve_outputs > 0;
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = pos4_in_ ? pos4_in_[i] : pos0 + i;   // M-RoPE: the KV cells keep section t
    if (!reserve) {
        if (!fatal_.empty()) return fatal_;
        if (reading_ && n < kOffloadMin) { end_read(); if (!fatal_.empty()) return fatal_; }
        if (copy_stream_) commit_landed();
        if (!kv_.find_slot(n)) return "KV cache full";
        kv_.apply(positions);
    }
    const uint32_t n_kv = reserve ? kv_.size : kv_.n_kv;

    ggml_init_params ip = { graph_buf_.size(), graph_buf_.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_backend_sched_reset(sched_);
    const int n_outputs = reserve ? reserve_outputs : n_out;
    const int64_t K = (int64_t) o_.n_rs_seq + 1;   // recurrent snapshot slots
    const int64_t n_embd = H.n_embd;
    auto nm = [&](ggml_tensor * t, const char * s, int il) { if (il >= 0) ggml_format_name(t, "%s-%d", s, il); else ggml_set_name(t, s); };

    // build_inp_embd (token ids; the raw-embedding branch of ggml_build_forward_select is kept for the graph's shape)
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(inp_tokens, "inp_tokens"); ggml_set_input(inp_tokens);
    ggml_tensor * inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n);
    ggml_set_name(inp_embd, "inp_embd"); ggml_set_input(inp_embd);
    ggml_tensor * inps[2] = { ggml_get_rows(ctx, tok_embd_, inp_tokens), inp_embd };
    ggml_tensor * inpL = ggml_build_forward_select(gf, inps, 2, embd_in_ ? 1 : 0);
    ggml_set_name(inpL, "model.input_embed");
    ggml_build_forward_expand(gf, inpL);

    // build_inp_mem_hybrid: attention inputs (k/v idxs, the causal mask) + recurrent inputs (s_copy: cell 0)
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n, 1, 1);
    ggml_set_input(kq_mask); ggml_set_name(kq_mask, "attn_inp_kq_mask");
    // build_attn_inp_kv_impl: the rotations of a quantized cache (head size a multiple of 64)
    ggml_tensor * k_rot = nullptr, * v_rot = nullptr;
    if (ggml_is_quantized(o_.type_k) && H.head_dim % 64 == 0) {
        const int r = k_rot_size(H.head_dim);
        k_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(k_rot); ggml_set_name(k_rot, "attn_inp_k_rot");
    }
    if (ggml_is_quantized(o_.type_v) && H.head_dim % 64 == 0) {
        v_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(v_rot); ggml_set_name(v_rot, "attn_inp_v_rot");
    }
    // build_rs_inp_impl: s_copy [n_rs], its main view [n_seqs] and extra view [n_rs - n_seqs] (empty for one sequence)
    ggml_tensor * s_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(s_copy);
    ggml_tensor * s_copy_main  = ggml_view_1d(ctx, s_copy, 1, 0);
    ggml_tensor * s_copy_extra = ggml_view_1d(ctx, s_copy, 0, 1 * s_copy->nb[0]);
    const int32_t rs_zero = (rs_fresh_ && !reserve) ? 0 : -1;   // a new cell is cleared before its first use (llama_memory_recurrent: rs_z)
    // build_inp_pos: IMROPE takes 4 positions per token (text: the same position three times, then 0)
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n * 4);
    ggml_set_input(inp_pos);
    ggml_tensor * inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    ggml_set_input(inp_out_ids);

    const int64_t head_k_dim = H.ssm_d_state, num_k_heads = H.ssm_n_group, num_v_heads = H.ssm_dt_rank;
    const int64_t head_v_dim = H.ssm_d_inner / num_v_heads;
    const int64_t conv_channels = H.ssm_d_inner + 2 * H.ssm_n_group * H.ssm_d_state;
    const int64_t n_embd_head = H.head_dim, n_gqa = (int64_t) H.head_dim * H.n_head_kv;
    int sections[4]; std::copy(H.rope_sections, H.rope_sections + 4, sections);

    // llm_graph_context::build_rs, node for node (the clear and the extra copy are empty views for one live cell, but
    // they sit between the nodes ggml-cuda would otherwise fuse)
    auto build_rs = [&](ggml_tensor * s, int64_t state_size) {
        const int64_t n_rs = 1, n_seqs = 1, rs_head = 0;
        ggml_tensor * states = ggml_reshape_2d(ctx, s, state_size, s->ne[1]);
        ggml_tensor * state_zero = ggml_view_1d(ctx, states, state_size * (rs_zero >= 0), rs_zero * states->nb[1] * (rs_zero >= 0));
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx, state_zero, 0));
        ggml_tensor * out = ggml_get_rows(ctx, states, s_copy_main);
        ggml_build_forward_expand(gf, out);
        ggml_tensor * states_extra = ggml_get_rows(ctx, states, s_copy_extra);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, states_extra,
            ggml_view_2d(ctx, s, state_size, n_rs - n_seqs, s->nb[1], (rs_head + n_seqs) * s->nb[1])));
        return out;
    };

    std::vector<ggml_tensor *> adapt_ids(o_.adapt ? H.n_layer : 0, nullptr);   // stage e: each cached layer's routing
    for (int il = 0; il < H.n_layer; ++il) {
        const Layer & l = L_[il];
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = norm(ctx, inpL, l.attn_norm, il);
        nm(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (H.is_recr[il]) {
            // ---- build_layer_attn_linear (gated delta net)
            ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, l.wqkv, cur);
            qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], n, 1);
            nm(qkv_mixed, "linear_attn_qkv_mixed", il);
            ggml_tensor * z = ggml_mul_mat(ctx, l.wqkv_gate, cur);
            nm(z, "z", il);
            ggml_tensor * beta = ggml_mul_mat(ctx, l.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n, 1);
            nm(beta, "beta", il);
            beta = ggml_sigmoid(ctx, beta);
            nm(beta, "beta_sigmoid", il);
            ggml_tensor * alpha = ggml_mul_mat(ctx, l.ssm_alpha, cur);
            alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n, 1);
            nm(alpha, "alpha", il);
            ggml_tensor * alpha_biased = ggml_add(ctx, alpha, l.ssm_dt);
            ggml_tensor * alpha_softplus = ggml_softplus(ctx, alpha_biased);
            nm(alpha_softplus, "a_softplus", il);
            ggml_tensor * gate = ggml_mul(ctx, alpha_softplus, l.ssm_a);   // -A_log.exp() * softplus
            nm(gate, "gate", il);
            gate = ggml_reshape_4d(ctx, gate, 1, num_v_heads, n, 1);

            // build_conv_state
            ggml_tensor * conv_states = build_rs(r_l_[il], H.n_embd_r());
            nm(conv_states, "conv_states", il);
            conv_states = ggml_reshape_3d(ctx, conv_states, H.ssm_d_conv - 1, conv_channels, 1);
            nm(conv_states, "conv_states_reshaped", il);
            ggml_tensor * qkv_t = ggml_transpose(ctx, qkv_mixed);
            nm(qkv_t, "qkv_mixed_transposed", il);
            ggml_tensor * conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);
            nm(conv_input, "conv_input", il);
            if (o_.n_rs_seq > 0) {   // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K conv windows, slot K - t
                const int64_t row_count = (int64_t) (H.ssm_d_conv - 1) * conv_channels;
                const size_t row_size = ggml_row_size(r_l_[il]->type, row_count);
                for (int64_t t = 1; t <= K; ++t) {
                    const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - conv_states->ne[0] - K + t);
                    const int64_t s_slot = K - t;
                    ggml_tensor * last = ggml_view_3d(ctx, conv_input, H.ssm_d_conv - 1, conv_channels, 1,
                                                      conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, s_idx));
                    ggml_tensor * upd = ggml_view_2d(ctx, r_l_[il], row_count, 1, r_l_[il]->nb[1], (size_t) s_slot * row_size);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
                }
            } else {
                const int64_t row_count = (int64_t) (H.ssm_d_conv - 1) * conv_channels;
                const size_t row_size = ggml_row_size(r_l_[il]->type, row_count);
                const int64_t s_idx = conv_input->ne[0] - conv_states->ne[0];
                ggml_tensor * last = ggml_view_3d(ctx, conv_input, H.ssm_d_conv - 1, conv_channels, 1,
                                                  conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, s_idx));
                nm(last, "conv_state_last", il);
                ggml_tensor * upd = ggml_view_2d(ctx, r_l_[il], row_count, 1, r_l_[il]->nb[1], 0 * row_size);
                nm(upd, "conv_state_update", il);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
            }

            ggml_tensor * state = build_rs(s_l_[il], H.n_embd_s());
            state = ggml_reshape_4d(ctx, state, head_v_dim, head_v_dim, num_v_heads, 1);
            nm(state, "state_predelta", il);

            ggml_tensor * conv_out = ggml_ssm_conv(ctx, conv_input, l.ssm_conv1d);
            nm(conv_out, "conv_output_raw", il);
            ggml_tensor * conv_silu = ggml_silu(ctx, conv_out);
            nm(conv_silu, "conv_output_silu", il);
            const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
            const size_t nb1_qkv = ggml_row_size(conv_silu->type, qkv_dim);
            ggml_tensor * q_conv = ggml_view_4d(ctx, conv_silu, head_k_dim, num_k_heads, n, 1,
                                                ggml_row_size(conv_silu->type, head_k_dim), nb1_qkv, nb1_qkv * n, 0);
            ggml_tensor * k_conv = ggml_view_4d(ctx, conv_silu, head_k_dim, num_k_heads, n, 1,
                                                ggml_row_size(conv_silu->type, head_k_dim), nb1_qkv, nb1_qkv * n,
                                                head_k_dim * num_k_heads * ggml_element_size(conv_silu));
            ggml_tensor * v_conv = ggml_view_4d(ctx, conv_silu, head_v_dim, num_v_heads, n, 1,
                                                ggml_row_size(conv_silu->type, head_v_dim), nb1_qkv, nb1_qkv * n,
                                                ggml_row_size(conv_silu->type, 2 * head_k_dim * num_k_heads));
            nm(q_conv, "q_conv", il); nm(k_conv, "k_conv", il); nm(v_conv, "v_conv", il);
            // build_gdn_l2_norm
            auto l2 = [&](ggml_tensor * x) {
                const float nx = (float) x->ne[0];
                return ggml_scale(ctx, ggml_rms_norm(ctx, x, H.eps / nx), 1.0f / sqrtf(nx));
            };
            q_conv = l2(q_conv);
            k_conv = l2(k_conv);
            nm(q_conv, "q_conv_predelta", il); nm(k_conv, "k_conv_predelta", il); nm(v_conv, "v_conv_predelta", il);

            // build_recurrent_attn -> build_delta_net_fused (cparams.fused_gdn_ar / _ch); with n_rs_seq the op writes K
            // snapshots (slot 0 = the newest state, slot s = s tokens back; the last min(n, K) are written)
            ggml_tensor * gdn = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv, gate, beta, state, K);
            ggml_tensor * out = ggml_view_4d(ctx, gdn, head_v_dim, num_v_heads, n, 1,
                                             ggml_row_size(gdn->type, head_v_dim), ggml_row_size(gdn->type, head_v_dim * num_v_heads),
                                             ggml_row_size(gdn->type, head_v_dim * num_v_heads * n), 0);
            if (o_.n_rs_seq > 0) {
                nm(out, "attn_output", il);
                const int64_t D = head_v_dim * head_v_dim * num_v_heads;
                const int64_t n_written = std::min<int64_t>(n, K);
                const size_t row_size = H.n_embd_s() * ggml_element_size(s_l_[il]);
                ggml_tensor * src = ggml_view_3d(ctx, gdn, D, 1, n_written, ggml_row_size(gdn->type, D), ggml_row_size(gdn->type, D),
                                                 ggml_row_size(gdn->type, head_v_dim * num_v_heads * n));
                ggml_tensor * dst = ggml_view_3d(ctx, s_l_[il], D, 1, n_written, s_l_[il]->nb[1], row_size, 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
            } else {
            ggml_tensor * new_state = ggml_view_4d(ctx, gdn, head_v_dim, head_v_dim, num_v_heads, 1,
                                                   ggml_row_size(gdn->type, head_v_dim), ggml_row_size(gdn->type, head_v_dim * head_v_dim),
                                                   ggml_row_size(gdn->type, head_v_dim * head_v_dim * num_v_heads),
                                                   ggml_row_size(gdn->type, head_v_dim * num_v_heads * n));
            nm(out, "attn_output", il); nm(new_state, "new_state", il);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state,
                ggml_view_2d(ctx, s_l_[il], H.n_embd_s(), 1, s_l_[il]->nb[1], 0)));
            }

            ggml_tensor * z_2d = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n, 1);
            // build_norm_gated
            ggml_tensor * normalized = norm(ctx, out, l.ssm_norm, il);
            ggml_tensor * gated_silu = ggml_silu(ctx, z_2d);
            ggml_tensor * attn_out_norm = ggml_mul(ctx, normalized, gated_silu);
            ggml_tensor * final_output = ggml_reshape_3d(ctx, attn_out_norm, head_v_dim * num_v_heads, n, 1);
            nm(final_output, "final_output", il);
            cur = ggml_mul_mat(ctx, l.ssm_out, final_output);
            nm(cur, "linear_attn_out", il);
            cur = ggml_reshape_2d(ctx, cur, n_embd, n);
        } else {
            // ---- build_layer_attn (gated attention)
            ggml_tensor * Qcur_full = ggml_mul_mat(ctx, l.wq, cur);   // build_qkv(..., reshape = false): separate Q/K/V
            ggml_tensor * Kcur = ggml_mul_mat(ctx, l.wk, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx, l.wv, cur);
            nm(Qcur_full, "Qcur_full", il); nm(Kcur, "Kcur", il); nm(Vcur, "Vcur", il);
            ggml_tensor * Qcur = ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n,
                                              ggml_element_size(Qcur_full) * n_embd_head * 2,
                                              ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head, 0);
            nm(Qcur, "Qcur_reshaped", il);
            Qcur = norm(ctx, Qcur, l.q_norm, il);
            nm(Qcur, "Qcur_normed", il);
            Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, H.n_head_kv, n);
            Kcur = norm(ctx, Kcur, l.k_norm, il);
            nm(Kcur, "Kcur_normed", il);
            ggml_tensor * gate = ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n,
                                              ggml_element_size(Qcur_full) * n_embd_head * 2,
                                              ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head,
                                              ggml_element_size(Qcur_full) * n_embd_head);
            gate = ggml_cont_2d(ctx, gate, n_embd_head * H.n_head, n);
            nm(gate, "gate_reshaped", il);
            Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, H.n_head_kv, n);
            Qcur = ggml_rope_multi(ctx, Qcur, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                                   kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
            Kcur = ggml_rope_multi(ctx, Kcur, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                                   kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
            nm(Qcur, "Qcur", il); nm(Kcur, "Kcur", il); nm(Vcur, "Vcur", il);
            const float kq_scale = 1.0f / sqrtf((float) n_embd_head);

            // build_attn (kv): the rotations, cpy_k / cpy_v, then build_attn_mha with flash attention
            if (k_rot) { Qcur = mul_mat_hadamard(ctx, Qcur, k_rot); Kcur = mul_mat_hadamard(ctx, Kcur, k_rot); }
            if (v_rot) Vcur = mul_mat_hadamard(ctx, Vcur, v_rot);
            ggml_build_forward_expand(gf, Qcur);   // llm_graph_context::build_attn (kv): q, then v, then k
            ggml_build_forward_expand(gf, Vcur);
            ggml_build_forward_expand(gf, Kcur);
            {
                ggml_tensor * kc = ggml_view_2d(ctx, Kcur, n_gqa, n, Kcur->nb[2], 0);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, kv_.k[il], kc, k_idxs));
                ggml_tensor * vc = ggml_view_2d(ctx, Vcur, n_gqa, n, Vcur->nb[2], 0);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, kv_.v[il], vc, v_idxs));
            }
            ggml_tensor * k = ggml_view_4d(ctx, kv_.k[il], n_embd_head, H.n_head_kv, n_kv, 1,
                                           ggml_row_size(kv_.k[il]->type, n_embd_head), ggml_row_size(kv_.k[il]->type, n_gqa),
                                           ggml_row_size(kv_.k[il]->type, n_gqa * kv_.size), 0);
            ggml_tensor * v = ggml_view_4d(ctx, kv_.v[il], n_embd_head, H.n_head_kv, n_kv, 1,
                                           ggml_row_size(kv_.v[il]->type, n_embd_head), ggml_row_size(kv_.v[il]->type, n_gqa),
                                           ggml_row_size(kv_.v[il]->type, n_gqa * kv_.size), 0);
            ggml_tensor * q = ggml_view_4d(ctx, Qcur, Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], 1, Qcur->nb[1], Qcur->nb[2], Qcur->nb[3], 0);
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
            cur = ggml_flash_attn_ext(ctx, q, kp, vp, kq_mask, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(cur, nullptr);
            ggml_flash_attn_ext_set_n_kv_max(cur, 0);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
            nm(cur, "kqv_out", il);
            if (v_rot) cur = mul_mat_hadamard(ctx, cur, v_rot);
            nm(cur, "attn_pregate", il);
            ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx, gate);
            nm(gate_sigmoid, "gate_sigmoid", il);
            cur = ggml_mul(ctx, cur, gate_sigmoid);
            nm(cur, "attn_gated", il);
            cur = ggml_mul_mat(ctx, l.wo, cur);
            nm(cur, "attn_output", il);
        }

        cur = ggml_add(ctx, cur, inpSA);
        nm(cur, "attn_residual", il);
        ggml_tensor * ffn_residual = cur;
        ggml_tensor * attn_post_norm = norm(ctx, cur, l.attn_post_norm, il);
        nm(attn_post_norm, "attn_post_norm", il);

        // ---- build_layer_ffn: build_moe_ffn(SILU, norm_w, SOFTMAX, separate gate/up) + the gated shared expert
        ggml_tensor * x = attn_post_norm;
        ggml_tensor * moe_out;
        {
            const int64_t n_expert = H.n_expert, n_used = H.n_expert_used;
            ggml_tensor * logits_r = ggml_mul_mat(ctx, l.gate_inp, x);
            nm(logits_r, "ffn_moe_logits", il);
            ggml_tensor * probs = ggml_soft_max(ctx, logits_r);
            nm(probs, "ffn_moe_probs", il);
            ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, (int) n_used);
            nm(selected->src[0], "ffn_moe_argsort", il);
            nm(selected, "ffn_moe_topk", il);
            probs = ggml_reshape_3d(ctx, probs, 1, n_expert, n);
            ggml_tensor * weights = ggml_get_rows(ctx, probs, selected);
            nm(weights, "ffn_moe_weights", il);
            weights = ggml_reshape_2d(ctx, weights, n_used, n);
            ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
            nm(wsum, "ffn_moe_weights_sum", il);
            wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
            nm(wsum, "ffn_moe_weights_sum_clamped", il);
            weights = ggml_div(ctx, weights, wsum);
            nm(weights, "ffn_moe_weights_norm", il);
            weights = ggml_reshape_3d(ctx, weights, 1, n_used, n);
            ggml_build_forward_expand(gf, weights);
            // up -> gate -> swiglu -> down over expert tensors (gate, up, down) addressed by `ids` (build_moe_ffn's middle)
            auto mm_path = [&](ggml_tensor * gw, ggml_tensor * uw, ggml_tensor * dw, ggml_tensor * ids) {
                ggml_tensor * xr = ggml_reshape_3d(ctx, x, n_embd, 1, n);
                ggml_tensor * up = ggml_mul_mat_id(ctx, uw, xr, ids);
                nm(up, "ffn_moe_up", il);
                ggml_tensor * g = ggml_mul_mat_id(ctx, gw, xr, ids);
                nm(g, "ffn_moe_gate", il);
                g = ggml_swiglu_split(ctx, g, up);
                nm(g, "ffn_moe_swiglu", il);
                return ggml_mul_mat_id(ctx, dw, g, ids);
            };
            ggml_tensor * experts;
            PoolLayer * PL = pool_layers_.empty() ? nullptr : pool_layers_[il].get();
            CacheLayer * C = cache_.empty() ? nullptr : cache_[il].get();
            const bool small = n < kOffloadMin;
            if (o_.adapt && C && !C->full) {   // flagged in the reserve graphs too: the allocator keeps the reserved plan and
                ggml_set_output(selected->src[0]);   // ignores a flag added later, so the ids would be overwritten
                if (!reserve) adapt_ids[il] = selected->src[0];
            }
            if (C && C->full) {                                     // stage d, whole layer: llama.cpp's GPU-layer path
                experts = mm_path(C->slot[0], C->slot[1], C->slot[2], selected);
                if (small) { stats_.accesses += (uint64_t) n * n_used; stats_.hits += (uint64_t) n * n_used; }
            } else if (C && small && n <= kSlotMaxTokens) {         // stage d: hits on the GPU slots, misses in the pool
                ggml_tensor * slot = ggml_reshape_3d(ctx, C->res_f, 1, n_expert, 1);   // per token, as llama gathers weights
                slot = ggml_repeat_4d(ctx, slot, 1, n_expert, n, 1);
                slot = ggml_get_rows(ctx, slot, selected);                          // [1, k, n]: slot, or S (zeros)
                slot = ggml_reshape_2d(ctx, slot, n_used, n);
                ggml_tensor * slot_ids = ggml_cast(ctx, slot, GGML_TYPE_I32);
                nm(slot_ids, "ffn_moe_slots", il);
                ggml_tensor * args[2] = { x, selected };
                ggml_tensor * eg, * ec;
                if (o_.overlap) {   // start (CPU) -> hits (GPU, launched async) -> join (CPU, waits on the pool) -> add
                    ggml_tensor * st = ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, moe_start_op, 1, C->miss.get());
                    nm(st, "ffn_moe_start", il);
                    ggml_build_forward_expand(gf, st);
                    eg = mm_path(C->slot[0], C->slot[1], C->slot[2], slot_ids);
                    ggml_build_forward_expand(gf, eg);
                    ggml_tensor * jargs[1] = { st };
                    ec = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, jargs, 1, moe_join_op, 1, C->miss.get());
                } else {
                    eg = mm_path(C->slot[0], C->slot[1], C->slot[2], slot_ids);
                    ec = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1, C->miss.get());
                }
                nm(ec, "ffn_moe_misses", il);
                experts = ggml_add(ctx, eg, ec);                    // each row: its GPU value + 0, or 0 + its CPU value
                if (!C->all.empty()) {                              // dcheck twins: the whole layer both ways
                    ggml_tensor * tg = mm_path(C->all[0], C->all[1], C->all[2], selected);
                    ggml_tensor * tc = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1, PL);
                    ggml_set_output(tg); ggml_set_output(tc); ggml_set_output(experts); ggml_set_output(selected->src[0]);
                    ggml_build_forward_expand(gf, tg); ggml_build_forward_expand(gf, tc);
                    dcheck_ = { experts, tg, tc, selected->src[0], C };
                }
            } else if (PL && small) {                               // stage c: all of the layer's experts in the pool
                ggml_tensor * args[2] = { x, selected };
                experts = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1, PL);
                if (C) stats_.accesses += (uint64_t) n * n_used;
            } else {
                experts = mm_path(l.gate_exps, l.up_exps, l.down_exps, selected);
            }
            nm(experts, "ffn_moe_down", il);
            experts = ggml_mul(ctx, experts, weights);
            nm(experts, "ffn_moe_weighted", il);
            ggml_build_forward_expand(gf, experts);
            ggml_tensor * parts[64];
            for (int i = 0; i < n_used; ++i) {
                parts[i] = ggml_view_2d(ctx, experts, n_embd, n, experts->nb[2], i * experts->nb[1]);
                ggml_build_forward_expand(gf, parts[i]);
            }
            moe_out = parts[0];
            for (int i = 1; i < n_used; ++i) {
                moe_out = ggml_add(ctx, moe_out, parts[i]);
                ggml_build_forward_expand(gf, moe_out);
            }
            nm(moe_out, "ffn_moe_out", il);
        }
        {   // build_ffn(SILU, PAR): up, gate, swiglu, down
            ggml_tensor * tmp = ggml_mul_mat(ctx, l.up_shexp, x);
            nm(tmp, "ffn_up", il);
            ggml_tensor * g = ggml_mul_mat(ctx, l.gate_shexp, x);
            nm(g, "ffn_gate", il);
            g = ggml_swiglu_split(ctx, g, tmp);
            nm(g, "ffn_swiglu", il);
            ggml_tensor * shexp = ggml_mul_mat(ctx, l.down_shexp, g);
            nm(shexp, "ffn_down", il);
            nm(shexp, "ffn_shexp", il);
            ggml_tensor * sg = ggml_mul_mat(ctx, l.gate_inp_shexp, x);
            nm(sg, "shared_expert_gate", il);
            sg = ggml_sigmoid(ctx, sg);
            nm(sg, "shared_expert_gate_sigmoid", il);
            shexp = ggml_mul(ctx, shexp, sg);
            nm(shexp, "ffn_shexp_gated", il);
            cur = ggml_add(ctx, moe_out, shexp);
            nm(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx, cur, ffn_residual);
        nm(cur, "post_moe", il);
        nm(cur, "l_out", il);
        inpL = cur;
    }
    // llama_context marks the graph's results as outputs (t_h_nextn, t_embd, t_logits); the flag also keeps ggml-cuda
    // from fusing across them, so it matters for bit-exactness
    ggml_tensor * cur = norm(ctx, inpL, output_norm_, -1);
    ggml_set_name(cur, "h_nextn");
    ggml_set_output(cur);
    ggml_tensor * t_h = want_h_ ? cur : nullptr;   // every row (draft-mtp: llama_set_embeddings_nextn(ctx_tgt, true, unmasked))
    cur = ggml_get_rows(ctx, cur, inp_out_ids);
    ggml_set_name(cur, "result_norm");
    ggml_set_output(cur);
    cur = ggml_mul_mat(ctx, output_, cur);
    ggml_set_name(cur, "result_output");
    ggml_set_output(cur);
    ggml_tensor * t_logits = cur;
    ggml_build_forward_expand(gf, cur);

    if (reserve) {
        dcheck_ = {};   // a reserve graph is never computed
        const bool ok = ggml_backend_sched_reserve(sched_, gf);
        ggml_free(ctx);
        return ok ? "" : "graph reserve failed";
    }
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) { ggml_free(ctx); return "graph allocation failed"; }

    // inputs
    if (embd_in_) {
        std::vector<int32_t> zero(n, 0);
        ggml_backend_tensor_set(inp_tokens, zero.data(), 0, n * sizeof(int32_t));
        ggml_backend_tensor_set(inp_embd, embd_in_, 0, (size_t) n * n_embd * sizeof(float));
    } else ggml_backend_tensor_set(inp_tokens, toks, 0, n * sizeof(int32_t));
    if (pos4_in_) ggml_backend_tensor_set(inp_pos, pos4_in_, 0, (size_t) n * 4 * sizeof(int32_t));
    else {
        std::vector<int32_t> p4((size_t) n * 4);
        for (int i = 0; i < n; ++i) { p4[i] = positions[i]; p4[n + i] = positions[i]; p4[2 * n + i] = positions[i]; p4[3 * n + i] = 0; }
        ggml_backend_tensor_set(inp_pos, p4.data(), 0, p4.size() * sizeof(int32_t));
    }
    ggml_backend_tensor_set(k_idxs, kv_.idxs.data(), 0, n * sizeof(int64_t));
    ggml_backend_tensor_set(v_idxs, kv_.idxs.data(), 0, n * sizeof(int64_t));
    {
        std::vector<ggml_fp16_t> md((size_t) kv_.n_kv * n);
        const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n; ++i)
            for (uint32_t j = 0; j < kv_.n_kv; ++j) {
                const int32_t p0 = kv_.pos[j];
                md[(size_t) i * kv_.n_kv + j] = (p0 < 0 || p0 > positions[i]) ? drop : keep;
            }
        ggml_backend_tensor_set(kq_mask, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
    }
    { const int32_t c0 = rs_src_; ggml_backend_tensor_set(s_copy, &c0, 0, sizeof(int32_t)); }
    for (ggml_tensor * r : { k_rot, v_rot }) {
        if (!r) continue;
        const std::vector<float> hd = gen_hadamard((int) r->ne[0]);
        ggml_backend_tensor_set(r, hd.data(), 0, hd.size() * sizeof(float));
    }   // s_copy: idx * size + src0
    if (n_outputs) {
        std::vector<int32_t> oid(n_outputs);
        for (int i = 0; i < n_outputs; ++i) oid[i] = n - n_outputs + i;
        ggml_backend_tensor_set(inp_out_ids, oid.data(), 0, oid.size() * sizeof(int32_t));
    }

    static int n_single = 0;
    if (n == 1) ++n_single;
    if (getenv("Q35_GRAPHDUMP") && n == 1 && n_single == 3) {
        g_gdump = fopen(getenv("Q35_GRAPHDUMP"), "w");
        ggml_backend_sched_set_eval_callback(sched_, gdump_cb, nullptr);
    } else if (getenv("TDUMP") || getenv("GDUMP")) ggml_backend_sched_set_eval_callback(sched_, tdump_cb, nullptr);
    const uint64_t acc0 = stats_.accesses, hit0 = stats_.hits;
    const ggml_status st = ggml_backend_sched_graph_compute(sched_, gf);
    if (n < kOffloadMin && !cache_.empty()) stats_.step_hits.push_back({ (uint32_t) (stats_.accesses - acc0), (uint32_t) (stats_.hits - hit0) });
    if (reading_) stats_.read_buf_bytes = std::max<uint64_t>(stats_.read_buf_bytes, ggml_backend_sched_get_buffer_size(sched_, gpu_));
    if (g_gdump) { fclose(g_gdump); g_gdump = nullptr; ggml_backend_sched_set_eval_callback(sched_, nullptr, nullptr); }
    if (st != GGML_STATUS_SUCCESS) { ggml_free(ctx); return "compute failed"; }
    rs_fresh_ = false;
    rs_src_ = 0;                                   // a rollback index is single-use (llama_memory_recurrent_context::s_copy)
    last_ub_n_ = embd_in_ ? 0 : n;
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
    if (o_.adapt) adapt_after(n, adapt_ids);
    if (n_outputs && logits) {
        logits->resize((size_t) n_outputs * H.n_vocab);
        ggml_backend_tensor_get(t_logits, logits->data(), 0, logits->size() * sizeof(float));
    }
    if (t_h) {
        h_rows_.resize((size_t) n * n_embd);
        ggml_backend_tensor_get(t_h, h_rows_.data(), 0, h_rows_.size() * sizeof(float));
    }
    ggml_free(ctx);
    return "";
}

// llama_decode: one batch, split into n_ubatch pieces; only the batch's last token outputs (llama_batch_get_one). With
// n_rs_seq the split keeps the batch's last n_rs_seq + 1 tokens in one ubatch (split_equal's n_keep_tail). With MTP
// the drafter re-reads each ubatch after it (common_speculative_impl_draft_mtp::process does it per batch; per ubatch
// keeps a long read's hidden rows bounded, and the drafter splits in n_ubatch pieces either way).
std::string Engine::decode(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h_last) {
    const int n = (int) toks.size(), tail = o_.n_rs_seq > 0 ? o_.n_rs_seq + 1 : 0;
    const int64_t ne = hp_.n_embd;
    // stage g: the read's ubatch size
    const int pu = (o_.prefill_ubatch < 0 && n >= kOffloadMin) ? pick_chunk(n) : o_.prefill_ubatch;
    const int ub = (pu > 0 && n >= kOffloadMin) ? pu : o_.n_ubatch;
    if (n >= kOffloadMin) stats_.last_chunk = std::min(ub, n);
    if (o_.prefill_ubatch < 0 && n >= kOffloadMin)
        fprintf(stderr, "strata: reading %d tokens at position %d in %d-token chunks (cap %d)\n", n, n_past_, std::min(ub, n), ub);
    const auto tr0 = std::chrono::steady_clock::now();
    if (ub != o_.n_ubatch || (o_.lend && n >= kOffloadMin)) begin_read(ub);
    for (int i = 0; i < n;) {
        int m = std::min(ub, n - i);
        const int rem = n - i - m;
        if (tail && rem > 0 && rem < tail) m -= tail - rem;
        const bool out = (i + m == n);
        want_h_ = (out && h_last) || mtp_.loaded;
        std::string err = ubatch(toks.data() + i, m, n_past_ + i, out ? 1 : 0, out ? logits : nullptr);
        want_h_ = false;
        if (!err.empty()) return err;
        if (out && h_last) h_last->assign(h_rows_.end() - ne, h_rows_.end());
        if (mtp_.loaded) {
            err = mtp_catchup(toks.data() + i, m, n_past_ + i, h_rows_.data());
            if (!err.empty()) return err;
        }
        i += m;
    }
    n_past_ += n;
    if (n >= kOffloadMin) stats_.prefill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
    return "";
}

// stage f: a verify window [id_last, d1..dk] in one ubatch, every row output (the server's spec batch: logits on all)
std::string Engine::decode_all(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h) {
    const int n = (int) toks.size();
    if (n > o_.n_ubatch) return "verify window larger than a ubatch";
    want_h_ = true;
    std::string err = ubatch(toks.data(), n, n_past_, n, logits);
    want_h_ = false;
    if (!err.empty()) return err;
    win_h_ = h_rows_;
    win_pos0_ = n_past_;
    if (h) *h = h_rows_;
    if (mtp_.loaded) {
        err = mtp_catchup(toks.data(), n, n_past_, win_h_.data());
        if (!err.empty()) return err;
    }
    n_past_ += n;
    return "";
}

// stage f: keep positions < n_keep. The attention layers free their cells (seq_rm); the delta-net layers read the
// snapshot r = n_past - n_keep tokens back in the next ubatch (llama_memory_recurrent::seq_rm's partial rollback, bounded
// by n_rs_seq); the drafter frees its cells and takes the kept position's hidden row (draft-mtp's accept)
std::string Engine::rollback(int n_keep) {
    const int r = n_past_ - n_keep;
    if (r <= 0) return "";
    if (r > o_.n_rs_seq || rs_src_ != 0 || r >= last_ub_n_) return "rollback of " + std::to_string(r) + " tokens is past the recurrent snapshots";
    kv_.seq_rm(n_keep);
    for (int & c : ckpt_pos_) if (n_keep < c) c = -1;
    last_ub_n_ -= r;
    rs_src_ = r;
    n_past_ = n_keep;
    if (mtp_.loaded) {
        mtp_.kv.seq_rm(n_keep);
        const int64_t row = n_keep - 1 - win_pos0_, ne = hp_.n_embd;
        if (row >= 0 && (size_t) (row + 1) * ne <= win_h_.size())
            mtp_h_prev_.assign(win_h_.begin() + row * ne, win_h_.begin() + (row + 1) * ne);
    }
    return "";
}

void Engine::mtp_trim(int pos) { if (mtp_.loaded) mtp_.kv.seq_rm(pos); }

// ---------------------------------------------------------------- stage f: the MTP drafter
// load: the MTP-only GGUF (its nextn block, plus token_embd / output unless the target's are shared), every drafter
// weight on the GPU but token_embd (the input layer stays in RAM, as llama.cpp keeps it); its own KV cells (f16, the
// target's size) and its own scheduler, so the target's allocation plan (and with it ggml-cuda's fusions) is untouched
std::string Engine::load_mtp() {
    Mtp & M = mtp_;
    const HParams & H = hp_;
    gguf_init_params gp = { true, &M.meta };
    M.gguf = gguf_init_from_file(o_.mtp.c_str(), gp);
    if (!M.gguf) return "cannot read " + o_.mtp;
    {
        const int64_t id = gguf_find_key(M.gguf, "general.architecture");
        const std::string arch = id < 0 ? "" : gguf_get_val_str(M.gguf, id);
        if (arch != "qwen35moe") return "not a qwen35moe GGUF (" + arch + ")";
        if (kv_int(M.gguf, "qwen35moe.nextn_predict_layers", 0) != 1) return "expects one MTP block (nextn_predict_layers = 1)";
        if (kv_int(M.gguf, "qwen35moe.embedding_length") != H.n_embd || kv_int(M.gguf, "qwen35moe.expert_count") != H.n_expert)
            return "the drafter's shape does not match the target";
    }
    const int il = (int) kv_int(M.gguf, "qwen35moe.block_count") - 1;
    const int fd = open(o_.mtp.c_str(), O_RDONLY);
    if (fd < 0) return "open failed";
    struct stat sb; fstat(fd, &sb); M.map_size = sb.st_size;
    M.map = mmap(nullptr, M.map_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (M.map == MAP_FAILED) { M.map = nullptr; return "mmap failed"; }
    const size_t data_off = gguf_get_data_offset(M.gguf);
    const int n_tensors = (int) gguf_get_n_tensors(M.gguf);
    ggml_init_params ip = { (size_t) (n_tensors + 8) * ggml_tensor_overhead(), nullptr, true };
    M.ctx_w = ggml_init(ip);
    M.ctx_gpu = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, size_t>> on_cpu, on_gpu;
    auto make = [&](const std::string & name, bool cpu) -> ggml_tensor * {
        ggml_tensor * m = ggml_get_tensor(M.meta, name.c_str());
        if (!m) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(cpu ? M.ctx_w : M.ctx_gpu, m);
        ggml_set_name(t, name.c_str());
        (cpu ? on_cpu : on_gpu).push_back({ t, data_off + gguf_get_tensor_offset(M.gguf, gguf_find_tensor(M.gguf, name.c_str())) });
        return t;
    };
    auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
    M.tok_embd = o_.mtp_share_head ? nullptr : make("token_embd.weight", true);
    M.output   = o_.mtp_share_head ? nullptr : make("output.weight", false);
    if (!M.tok_embd) M.tok_embd = tok_embd_;   // nextn.embed_tokens ? : model.tok_embd
    if (!M.output) M.output = output_;
    Layer & l = M.l;
    l.attn_norm      = make(b("attn_norm.weight"), false);
    l.attn_post_norm = make(b("post_attention_norm.weight"), false);
    l.wq = make(b("attn_q.weight"), false); l.wk = make(b("attn_k.weight"), false); l.wv = make(b("attn_v.weight"), false);
    l.wo = make(b("attn_output.weight"), false);
    l.q_norm = make(b("attn_q_norm.weight"), false); l.k_norm = make(b("attn_k_norm.weight"), false);
    l.gate_inp  = make(b("ffn_gate_inp.weight"), false);
    l.gate_exps = make(b("ffn_gate_exps.weight"), false); l.up_exps = make(b("ffn_up_exps.weight"), false);
    l.down_exps = make(b("ffn_down_exps.weight"), false);
    l.gate_inp_shexp = make(b("ffn_gate_inp_shexp.weight"), false);
    l.gate_shexp = make(b("ffn_gate_shexp.weight"), false); l.up_shexp = make(b("ffn_up_shexp.weight"), false);
    l.down_shexp = make(b("ffn_down_shexp.weight"), false);
    M.eh_proj   = make(b("nextn.eh_proj.weight"), false);
    M.enorm     = make(b("nextn.enorm.weight"), false);
    M.hnorm     = make(b("nextn.hnorm.weight"), false);
    M.head_norm = make(b("nextn.shared_head_norm.weight"), false);
    if (!M.head_norm) M.head_norm = output_norm_;
    if (!l.attn_norm || !l.attn_post_norm || !l.wq || !l.wk || !l.wv || !l.wo || !l.q_norm || !l.k_norm || !l.gate_inp ||
        !l.gate_exps || !l.up_exps || !l.down_exps || !M.eh_proj || !M.enorm || !M.hnorm)
        return "missing MTP block tensors";
    M.buf_gpu = ggml_backend_alloc_ctx_tensors_from_buft(M.ctx_gpu, ggml_backend_get_default_buffer_type(gpu_));
    if (!M.buf_gpu) return "drafter weight allocation failed";
    ggml_backend_buffer_set_usage(M.buf_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto & [t, off] : on_gpu) ggml_backend_tensor_set(t, (const char *) M.map + off, 0, ggml_nbytes(t));
    if (!on_cpu.empty()) {
        M.buf_map = ggml_backend_cpu_buffer_from_ptr(M.map, M.map_size);
        ggml_backend_buffer_set_usage(M.buf_map, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & [t, off] : on_cpu)
            if (ggml_backend_tensor_alloc(M.buf_map, t, (char *) M.map + off) != GGML_STATUS_SUCCESS) return "map alloc failed";
    }
    M.kv.size = kv_.size;
    M.kv.pos.assign(M.kv.size, -1);
    M.kv.k.assign(1, nullptr); M.kv.v.assign(1, nullptr);
    ggml_init_params kp = { 4 * ggml_tensor_overhead(), nullptr, true };
    M.ctx_kv = ggml_init(kp);
    const int64_t n_gqa = (int64_t) H.head_dim * H.n_head_kv;
    M.kv.k[0] = ggml_new_tensor_3d(M.ctx_kv, GGML_TYPE_F16, n_gqa, M.kv.size, 1);
    M.kv.v[0] = ggml_new_tensor_3d(M.ctx_kv, GGML_TYPE_F16, n_gqa, M.kv.size, 1);
    ggml_format_name(M.kv.k[0], "cache_k_l%d", il); ggml_format_name(M.kv.v[0], "cache_v_l%d", il);
    M.buf_kv = ggml_backend_alloc_ctx_tensors_from_buft(M.ctx_kv, ggml_backend_get_default_buffer_type(gpu_));
    if (!M.buf_kv) return "drafter KV allocation failed";
    ggml_backend_buffer_clear(M.buf_kv, 0);
    ggml_backend_t backs[2] = { gpu_, cpu_ };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(gpu_), ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu_)) };
    if (!bufts[1]) bufts[1] = ggml_backend_get_default_buffer_type(cpu_);
    M.sched = ggml_backend_sched_new(backs, bufts, 2, kMaxNodes, false, true);
    M.loaded = true;
    return "";
}

// draft-mtp's process(): the drafter re-reads a target batch at the same positions, each token paired with the target's
// hidden row of the position before (row 0: the pending row from the previous batch), in n_ubatch pieces (its context
// splits plainly). Only its KV cells matter here, so by default the graph stops at the K/V writes; Q35_MTP_FULL=1 runs
// the whole block as llama.cpp does (the cells come out the same)
std::string Engine::mtp_catchup(const int32_t * toks, int n, int pos0, const float * h_rows) {
    const int64_t ne = hp_.n_embd;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> hs((size_t) n * ne);
    std::copy(mtp_h_prev_.begin(), mtp_h_prev_.end(), hs.begin());
    if (n > 1) std::copy(h_rows, h_rows + (size_t) (n - 1) * ne, hs.begin() + ne);
    mtp_h_prev_.assign(h_rows + (size_t) (n - 1) * ne, h_rows + (size_t) n * ne);
    static const bool full = getenv("Q35_MTP_FULL") != nullptr;
    for (int i = 0; i < n; i += o_.n_ubatch) {
        const int m = std::min(o_.n_ubatch, n - i);
        int32_t dummy;
        const std::string err = mtp_ubatch(toks + i, hs.data() + (size_t) i * ne, m, pos0 + i, full, &dummy, nullptr);
        if (!err.empty()) return "drafter: " + err;
    }
    stats_.catchup_tokens += n;
    stats_.catchup_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return "";
}

std::string Engine::draft(int32_t tok, const float * h, int pos, int32_t * out, std::vector<float> * h_out) {
    if (!mtp_.loaded) return "no MTP drafter loaded";
    const auto t0 = std::chrono::steady_clock::now();
    const std::string err = mtp_ubatch(&tok, h, 1, pos, true, out, h_out);
    stats_.drafts++;
    stats_.draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return err;
}

// llama_model_qwen35moe::graph_mtp, node for node; out = the last row's logits (argmax) and post-norm hidden row
std::string Engine::mtp_ubatch(const int32_t * toks, const float * h, int n, int pos0, bool out, int32_t * amax, std::vector<float> * h_out) {
    const HParams & H = hp_;
    Mtp & M = mtp_;
    const Layer & l = M.l;
    const int il = H.n_layer;   // the nextn block's index
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = pos0 + i;
    if (!M.kv.find_slot(n)) return "KV cache full";
    M.kv.apply(positions);
    const uint32_t n_kv = M.kv.n_kv;
    ggml_init_params ip = { graph_buf_.size(), graph_buf_.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_backend_sched_reset(M.sched);
    const int64_t n_embd = H.n_embd, n_embd_head = H.head_dim, n_gqa = (int64_t) H.head_dim * H.n_head_kv;
    const int n_outputs = out ? 1 : 0;
    auto nm = [&](ggml_tensor * t, const char * s) { ggml_format_name(t, "%s-%d", s, il); };
    auto mnorm = [&](ggml_tensor * cur, ggml_tensor * w, int lay) {   // build_norm, pinned to the layer's device
        cur = ggml_rms_norm(ctx, cur, H.eps);
        if (lay >= 0) {
            ggml_format_name(cur, "norm-%d", lay);
            if (ggml_backend_supports_op(gpu_, cur)) ggml_backend_sched_set_tensor_backend(M.sched, cur, gpu_);
        } else ggml_set_name(cur, "norm");
        return ggml_mul(ctx, cur, w);
    };
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(inp_tokens);
    ggml_tensor * tok_embd = ggml_get_rows(ctx, M.tok_embd, inp_tokens);
    nm(tok_embd, "mtp_tok_embd");
    ggml_tensor * inp_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n);
    ggml_set_input(inp_h); ggml_set_name(inp_h, "mtp_h_input");
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n * 4);
    ggml_set_input(inp_pos);
    ggml_tensor * inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    ggml_set_input(inp_out_ids);
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n, 1, 1);
    ggml_set_input(kq_mask);

    ggml_tensor * h_norm = mnorm(inp_h, M.hnorm, il);
    nm(h_norm, "mtp_hnorm");
    ggml_tensor * e_norm = mnorm(tok_embd, M.enorm, il);
    nm(e_norm, "mtp_enorm");
    ggml_tensor * concat = ggml_concat(ctx, e_norm, h_norm, 0);
    nm(concat, "mtp_concat");
    ggml_tensor * cur = ggml_mul_mat(ctx, M.eh_proj, concat);
    nm(cur, "mtp_eh_proj");
    ggml_tensor * inpSA = cur;
    cur = mnorm(cur, l.attn_norm, il);
    nm(cur, "mtp_attn_norm");
    ggml_tensor * Qcur_full = ggml_mul_mat(ctx, l.wq, cur);   // build_qkv(..., reshape = false)
    ggml_tensor * Kcur = ggml_mul_mat(ctx, l.wk, cur);
    ggml_tensor * Vcur = ggml_mul_mat(ctx, l.wv, cur);
    nm(Qcur_full, "mtp_Qcur_full");
    int sections[4]; std::copy(H.rope_sections, H.rope_sections + 4, sections);
    ggml_tensor * Qcur = nullptr, * gate = nullptr;
    if (out) {
        Qcur = ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n, ggml_element_size(Qcur_full) * n_embd_head * 2,
                            ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head, 0);
        Qcur = mnorm(Qcur, l.q_norm, il);
        nm(Qcur, "mtp_Qcur_normed");
        gate = ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n, ggml_element_size(Qcur_full) * n_embd_head * 2,
                            ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head, ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx, gate, n_embd_head * H.n_head, n);
        nm(gate, "mtp_gate");
    }
    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, H.n_head_kv, n);
    Kcur = mnorm(Kcur, l.k_norm, il);
    nm(Kcur, "mtp_Kcur_normed");
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, H.n_head_kv, n);
    nm(Vcur, "mtp_Vcur");
    if (out) Qcur = ggml_rope_multi(ctx, Qcur, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                                    kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
    Kcur = ggml_rope_multi(ctx, Kcur, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                           kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
    // build_attn (kv): q, v, k, then the cache writes
    if (out) ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Vcur);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, M.kv.k[0], ggml_view_2d(ctx, Kcur, n_gqa, n, Kcur->nb[2], 0), k_idxs));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, M.kv.v[0], ggml_view_2d(ctx, Vcur, n_gqa, n, Vcur->nb[2], 0), v_idxs));
    ggml_tensor * t_h = nullptr, * t_logits = nullptr;
    if (out) {
        ggml_tensor * k = ggml_view_4d(ctx, M.kv.k[0], n_embd_head, H.n_head_kv, n_kv, 1, ggml_row_size(M.kv.k[0]->type, n_embd_head),
                                       ggml_row_size(M.kv.k[0]->type, n_gqa), ggml_row_size(M.kv.k[0]->type, n_gqa * M.kv.size), 0);
        ggml_tensor * v = ggml_view_4d(ctx, M.kv.v[0], n_embd_head, H.n_head_kv, n_kv, 1, ggml_row_size(M.kv.v[0]->type, n_embd_head),
                                       ggml_row_size(M.kv.v[0]->type, n_gqa), ggml_row_size(M.kv.v[0]->type, n_gqa * M.kv.size), 0);
        ggml_tensor * q = ggml_view_4d(ctx, Qcur, Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], 1, Qcur->nb[1], Qcur->nb[2], Qcur->nb[3], 0);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        cur = ggml_flash_attn_ext(ctx, q, ggml_permute(ctx, k, 0, 2, 1, 3), ggml_permute(ctx, v, 0, 2, 1, 3), kq_mask,
                                  1.0f / sqrtf((float) n_embd_head), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(cur, nullptr);
        ggml_flash_attn_ext_set_n_kv_max(cur, 0);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        nm(cur, "mtp_attn_pregate");
        cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));
        cur = ggml_mul_mat(ctx, l.wo, cur);
        nm(cur, "mtp_attn_out");
        cur = ggml_add(ctx, cur, inpSA);
        nm(cur, "mtp_attn_residual");
        ggml_tensor * ffn_residual = cur;
        cur = mnorm(cur, l.attn_post_norm, il);
        nm(cur, "mtp_attn_post_norm");
        ggml_tensor * x = cur;
        // build_moe_ffn (SILU, norm_w, SOFTMAX, separate gate/up), every expert on the GPU
        const int64_t n_expert = H.n_expert, n_used = H.n_expert_used;
        ggml_tensor * logits_r = ggml_mul_mat(ctx, l.gate_inp, x);
        nm(logits_r, "ffn_moe_logits");
        ggml_tensor * probs = ggml_soft_max(ctx, logits_r);
        nm(probs, "ffn_moe_probs");
        ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, (int) n_used);
        nm(selected->src[0], "ffn_moe_argsort"); nm(selected, "ffn_moe_topk");
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
        ggml_tensor * xr = ggml_reshape_3d(ctx, x, n_embd, 1, n);
        ggml_tensor * up = ggml_mul_mat_id(ctx, l.up_exps, xr, selected);
        nm(up, "ffn_moe_up");
        ggml_tensor * g = ggml_mul_mat_id(ctx, l.gate_exps, xr, selected);
        nm(g, "ffn_moe_gate");
        g = ggml_swiglu_split(ctx, g, up);
        nm(g, "ffn_moe_swiglu");
        ggml_tensor * experts = ggml_mul_mat_id(ctx, l.down_exps, g, selected);
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
        for (int i = 1; i < n_used; ++i) { moe_out = ggml_add(ctx, moe_out, parts[i]); ggml_build_forward_expand(gf, moe_out); }
        nm(moe_out, "mtp_ffn_moe_out");
        if (l.up_shexp) {   // build_ffn(SILU, PAR) + the sigmoid gate
            ggml_tensor * tmp = ggml_mul_mat(ctx, l.up_shexp, x);
            nm(tmp, "ffn_up");
            ggml_tensor * gs = ggml_mul_mat(ctx, l.gate_shexp, x);
            nm(gs, "ffn_gate");
            gs = ggml_swiglu_split(ctx, gs, tmp);
            nm(gs, "ffn_swiglu");
            ggml_tensor * shexp = ggml_mul_mat(ctx, l.down_shexp, gs);
            nm(shexp, "mtp_ffn_shexp");
            ggml_tensor * sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, l.gate_inp_shexp, x));
            nm(sg, "mtp_shared_expert_gate_sigmoid");
            shexp = ggml_mul(ctx, shexp, sg);
            nm(shexp, "mtp_ffn_shexp_gated");
            cur = ggml_add(ctx, moe_out, shexp);
        } else cur = moe_out;
        nm(cur, "mtp_ffn_out");
        cur = ggml_add(ctx, cur, ffn_residual);
        nm(cur, "mtp_post_ffn");
        cur = mnorm(cur, M.head_norm, -1);
        ggml_set_name(cur, "h_nextn"); ggml_set_output(cur);
        t_h = cur;
        cur = ggml_get_rows(ctx, cur, inp_out_ids);
        ggml_set_name(cur, "mtp_shared_head_norm");
        cur = ggml_mul_mat(ctx, M.output, cur);
        ggml_set_name(cur, "result_output"); ggml_set_output(cur);
        t_logits = cur;
        ggml_build_forward_expand(gf, cur);
    }
    if (!ggml_backend_sched_alloc_graph(M.sched, gf)) { ggml_free(ctx); return "graph allocation failed"; }
    ggml_backend_tensor_set(inp_tokens, toks, 0, n * sizeof(int32_t));
    ggml_backend_tensor_set(inp_h, h, 0, (size_t) n * n_embd * sizeof(float));
    {
        std::vector<int32_t> p4((size_t) n * 4);
        for (int i = 0; i < n; ++i) { p4[i] = positions[i]; p4[n + i] = positions[i]; p4[2 * n + i] = positions[i]; p4[3 * n + i] = 0; }
        ggml_backend_tensor_set(inp_pos, p4.data(), 0, p4.size() * sizeof(int32_t));
    }
    ggml_backend_tensor_set(k_idxs, M.kv.idxs.data(), 0, n * sizeof(int64_t));
    ggml_backend_tensor_set(v_idxs, M.kv.idxs.data(), 0, n * sizeof(int64_t));
    if (out) {   // the K/V-only graph has no attention
        std::vector<ggml_fp16_t> md((size_t) n_kv * n);
        const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n_kv; ++j) {
                const int32_t p0 = M.kv.pos[j];
                md[(size_t) i * n_kv + j] = (p0 < 0 || p0 > positions[i]) ? drop : keep;
            }
        ggml_backend_tensor_set(kq_mask, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
    }
    if (n_outputs) { const int32_t oid = n - 1; ggml_backend_tensor_set(inp_out_ids, &oid, 0, sizeof(int32_t)); }
    const ggml_status st = ggml_backend_sched_graph_compute(M.sched, gf);
    if (st != GGML_STATUS_SUCCESS) { ggml_free(ctx); return "compute failed"; }
    if (out) {
        std::vector<float> lg((size_t) H.n_vocab);
        ggml_backend_tensor_get(t_logits, lg.data(), 0, lg.size() * sizeof(float));
        *amax = (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        if (h_out) {
            h_out->resize((size_t) n_embd);
            ggml_backend_tensor_get(t_h, h_out->data(), (size_t) (n - 1) * n_embd * sizeof(float), (size_t) n_embd * sizeof(float));
        }
    }
    ggml_free(ctx);
    return "";
}

} // namespace q35
