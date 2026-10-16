// Strata-Lite hybrid MoE engine: Qwen3.5 / 3.6 (DeltaNet, gated attention, shared expert) and
// LFM2.5 (short convolution, attention, leading dense FFNs, biased sigmoid router). llama.cpp's graph (src/models/qwen35moe.cpp, delta-net-base.cpp, llama-graph.cpp) rebuilt op for op with ggml,
// so every stage compares bit for bit with llama.cpp where the arithmetic is the same; in front of the RAM-resident
// experts, Strata's GPU expert cache, CPU pool, adaptation, MTP drafting and chunked reads (docs/QWEN36.md).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "../lite/lfm_lora.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels::cpu { class ExpertPool; }

namespace q35 {

struct Options {
    std::string gguf;
    std::string lora;             // opt-in rank-one LFM mixer-output/expert-down adapter
    float lora_scale = 1.0f;      // fixed for this engine instance; prompt cache cannot mix scales
    int  n_ctx     = 32768;
    int  n_ubatch  = 512;
    int  n_cpu_moe = 40;        // the first N layers' experts stay in RAM (llama.cpp --n-cpu-moe)
    int  n_threads = 16;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    bool pool        = false;   // stage c: RAM-resident experts of small ubatches through Strata's expert pool
    int  pool_threads = 16;
    // stage d: a GPU expert cache in front of the pool for the RAM-resident layers [cache_l0, n_cpu_moe)
    int  cache_slots  = 0;      // resident experts per layer (0 = off; >= n_expert = the whole layer, llama.cpp's GPU path;
                                // -1 = auto: as many as the free VRAM holds after everything else, less vram_reserve_mb)
    int  vram_reserve_mb = 512;
    int  cache_l0     = 0;
    std::string profile;        // per layer, experts ranked by use (tools/g4_profile.py); empty = plain order
    int  dcheck_layer = -1;     // a cached layer whose merged rows are checked against all-GPU and all-CPU twins
    // stage e: adaptation. Decayed usage counts per layer; after a prompt read every layer is re-set to its top-S, after
    // each small ubatch at most adapt_budget swaps per layer (a newcomer must score above its victim)
    bool  adapt           = false;
    float adapt_half_life = 64.0f;   // tokens (Gate 0, Qwen: 64 keeps the hits of 32 with half the swaps)
    int   adapt_budget    = 2;
    bool  adapt_refill    = true;
    // speed (Gemma's speed work, ported): swaps on their own CUDA stream from a page-locked expert arena, the newcomer
    // committed when its copy has landed (the victim goes to the CPU at once); the misses run in the pool while the GPU
    // computes the hits (start / join ops around the slot path)
    bool  async_swaps     = false;
    bool  overlap         = false;
    // stage g: prompt reads in ubatches of prefill_ubatch tokens (0 = n_ubatch, -1 = auto: the largest of 32k / 16k / 8k
    // / 4k whose compute buffers fit the VRAM the read can borrow); with lend the slot arena's VRAM goes to the read and
    // the arena comes back empty after it, refilled in the background from the read's routing counts
    int   prefill_ubatch  = 0;
    bool  lend            = false;
    // stage f: MTP (llama.cpp draft-mtp). The model's next-token head, an MTP-only GGUF (tools/qwen35_mtp_extract.py),
    // drafts; it keeps its own KV cache, filled after every target batch from the target's post-norm hidden rows. The
    // delta-net layers keep n_rs_seq + 1 state snapshots so a verify window can roll back (llama.cpp: n_rs_seq = draft max)
    std::string mtp;
    bool mtp_share_head = false;     // the target's token_embd / output instead of the drafter file's own
    int  n_rs_seq       = 0;
    // the KV pager: the attention layers' K/V start with kv_initial cells on the card (0 = the whole context at load, as
    // llama.cpp) and grow in kv_step steps as the conversation needs them; the slot arena gives way (ensure_free)
    int  kv_initial     = 0;
    int  kv_step        = 16384;
};

struct Stats {
    uint64_t accesses = 0, hits = 0;                       // routed (token, expert) pairs of the cached layers, small ubatches
    uint64_t dcheck_rows = 0, dcheck_gpu_rows = 0, dcheck_bad = 0;
    uint64_t swaps = 0, refill_swaps = 0, swap_bytes = 0;
    double   swap_ms = 0, refill_ms = 0;
    uint64_t drafts = 0, catchup_tokens = 0;              // stage f: drafter steps, tokens the drafter re-read
    double   draft_ms = 0, catchup_ms = 0;
    uint64_t commits = 0, commit_batches = 0, pending_peak = 0;   // async swaps
    uint64_t ov_layers = 0, ov_missfree = 0;                      // overlap: layer-steps, and those with no miss
    double   join_wait_ms = 0, pool_ms = 0;
    double   prefill_ms = 0, lend_ms = 0, restore_ms = 0, refill_land_ms = -1;   // stage g
    uint64_t lent_bytes = 0, read_buf_bytes = 0;     // the arena lent; the largest compute buffer a read ubatch took
    int      last_chunk = 0, arena_shrinks = 0;
    std::vector<std::pair<uint32_t, uint32_t>> step_hits;   // per small ubatch after load: routed pairs, hits (cached layers)
};

struct HParams {
    bool lfm2 = false;
    int n_dense = 0, n_ff = 0;
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0, n_vocab = 0, n_ctx_train = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    int ssm_d_conv = 0, ssm_d_inner = 0, ssm_d_state = 0, ssm_n_group = 0, ssm_dt_rank = 0, full_attn_interval = 4;
    int rope_sections[4] = { 0, 0, 0, 0 };
    float rope_base = 0, eps = 0;
    std::vector<bool> is_recr;          // DeltaNet or LFM shortconv layer (else full attention)
    int64_t n_embd_r() const { return (int64_t) (ssm_d_conv - 1) * (lfm2 ? n_embd : ssm_d_inner + 2 * ssm_n_group * ssm_d_state); }
    int64_t n_embd_s() const { return (int64_t) ssm_d_state * ssm_d_inner; }
};

class Engine {
public:
    Engine();
    ~Engine();
    std::string load(const Options & o);
    // one llama_decode call: tokens at the next positions; the last token's logits (and h_last: its post-norm hidden row)
    std::string decode(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h_last = nullptr);
    void reset();               // forget everything (positions, KV cells, recurrent states)
    // stage f. A verify window in one ubatch: logits [n][n_vocab] and post-norm hidden rows [n][n_embd] of every row
    std::string decode_all(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h);
    // one draft step: token at pos with hidden row h (the target's, then the drafter's own) -> argmax + the next h
    std::string draft(int32_t tok, const float * h, int pos, int32_t * out, std::vector<float> * h_out);
    void mtp_trim(int pos);                       // drop the drafter's cells from pos on (its draft steps)
    std::string rollback(int n_keep);             // keep positions < n_keep: KV cells, recurrent snapshot, drafter
    const std::vector<float> & mtp_h() const { return mtp_h_prev_; }   // h of the last kept position (the next draft's)
    bool has_mtp() const { return mtp_.loaded; }
    const HParams & hp() const { return hp_; }
    const Stats & stats() const { return stats_; }
    int slots() const { return slots_now_; }
    int n_past() const { return n_past_; }
    int n_ctx() const { return (int) kv_.size; }
    int kv_cap() const { return (int) kv_cap_; }   // KV cells on the card now (<= n_ctx)
    // an image chunk (mtmd's embeddings, n rows): pos4 = 4 x n M-RoPE positions (t, y, x, z sections, mtmd's
    // decoder positions), n_pos = how far the position advances (mtmd_input_chunk_get_n_pos). The drafter skips it, as
    // draft-mtp's process() skips embedding batches.
    std::string decode_embd(const float * embd, int n, const int32_t * pos4, int n_pos, std::vector<float> * logits);
    // the prompt cache. A recurrent state cannot rewind to any position: the engine can return to pos when pos is the
    // current position, within the last ubatch's snapshots, a checkpoint (the server takes up to four per prompt), or 0.
    // resume_point(pos) = the largest of those <= pos; rollback_to(p) goes there.
    int resume_point(int pos) const;
    std::string rollback_to(int pos);
    void checkpoint();
    // VRAM for something else (the projector, its encode): fewer slots; and back up to S slots when it is free again
    int ensure_free(size_t bytes, bool keep_profile = false);
    int grow_to(int S, size_t headroom);

private:
    struct Layer {
        ggml_tensor *attn_norm, *attn_post_norm;
        ggml_tensor *wq, *wk, *wv, *wo, *q_norm, *k_norm;                                            // attention layers
        ggml_tensor *wqkv, *wqkv_gate, *ssm_conv1d, *ssm_dt, *ssm_a, *ssm_beta, *ssm_alpha, *ssm_norm, *ssm_out;   // delta-net
        ggml_tensor *conv_in, *conv_out, *conv_kernel; // LFM2 short convolution
        ggml_tensor *dense_gate, *dense_up, *dense_down, *expert_bias;
        ggml_tensor *gate_inp, *gate_exps, *up_exps, *down_exps;                                    // routed experts
        ggml_tensor *gate_inp_shexp, *gate_shexp, *up_shexp, *down_shexp;                          // shared expert
    };
public:
    struct PoolLayer;                    // one layer's experts as the pool reads them (q35_engine.cpp)
    struct CacheLayer;                   // one layer's GPU expert slots (stage d)
    struct Driver;                       // overlap: the thread that runs the pool for a start op
private:
    struct Cache {                       // llama_kv_cache, one sequence, no sliding window (the attention layers)
        uint32_t size = 0, head = 0, n_kv = 0;
        std::vector<int32_t> pos;
        std::vector<int64_t> idxs;
        std::vector<ggml_tensor *> k, v;
        uint32_t used() const;
        uint32_t used_max_p1() const;
        int32_t  pos_min() const;
        int32_t  pos_max() const;
        bool find_slot(uint32_t n_tokens);
        void apply(const std::vector<int32_t> & positions);
        void seq_rm(int32_t p0);         // llama_kv_cache::seq_rm(seq, p0, -1)
    };

    // n_out: how many trailing rows output logits (0, 1 or n)
    std::string ubatch(const int32_t * toks, int n, int pos0, int n_out, std::vector<float> * logits, int reserve_outputs = 0);
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, int il);

    Options o_;
    std::unique_ptr<lite::LfmLora> lora_;
    HParams hp_;
    gguf_context * gguf_ = nullptr;
    ggml_context * meta_ = nullptr, * ctx_w_ = nullptr, * ctx_gpu_ = nullptr, * ctx_kv_ = nullptr;
    void * map_ = nullptr;
    size_t map_size_ = 0;
    ggml_backend_t gpu_ = nullptr, cpu_ = nullptr;
    ggml_backend_buffer_t buf_gpu_ = nullptr, buf_map_ = nullptr, buf_kv_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    ggml_tensor *tok_embd_ = nullptr, *output_ = nullptr, *output_norm_ = nullptr;
    std::vector<Layer> L_;
    Cache kv_;
    std::vector<ggml_tensor *> r_l_, s_l_;   // recurrent states per delta-net layer: conv [n_embd_r, 1], ssm [n_embd_s, 1]
    int n_past_ = 0;
    std::vector<uint8_t> graph_buf_;
    std::vector<float> h_rows_;
    bool want_h_ = false;
    bool rs_fresh_ = true;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool_;
    std::vector<std::unique_ptr<PoolLayer>> pool_layers_;
    std::unique_ptr<Driver> driver_;
    // async swaps: the RAM-resident experts in a pinned arena, the copy stream, and the batches of copies in flight
    void * arena_ = nullptr;
    size_t arena_size_ = 0;
    ggml_backend_buffer_t buf_arena_ = nullptr;
    void * copy_stream_ = nullptr;
    struct PendingSwap { int il, e, slot; };
    struct PendingBatch { void * ev; std::vector<PendingSwap> sw; };
    std::vector<PendingBatch> pending_;
    void commit_landed();
    // stage g
    bool reading_ = false, refill_watch_ = false;
    std::chrono::steady_clock::time_point read_end_;
    std::string fatal_;
    void begin_read(int ubatch);
    void end_read();
    int pick_chunk(int n);
    std::string reserve_graphs();
    void new_sched();
    // stage d: the slot arena. Model-neutral: an expert is a list of tensors (here gate, up, down), each layer gets one
    // slot tensor per expert tensor with S + 1 slots (slot S all zeros: where the misses point)
    std::vector<std::unique_ptr<CacheLayer>> cache_;
    ggml_context * ctx_cache_ = nullptr;
    ggml_backend_buffer_t buf_cache_ = nullptr;
    std::vector<std::vector<int>> rank_;
    int slots_now_ = 0;
    int slots_load_ = 0;
    size_t compute_bytes_ = 0;   // the reserved graphs' GPU buffers at load   // the slot count at load: a new conversation grows the arena back to it (the KV pager)
    std::string arena_build(int S, bool fill_profile);
    void adapt_after(int n, const std::vector<ggml_tensor *> & argsorts);
    void swap_layer(int il, int budget);
    bool refill_pending_ = false;
    struct DCheck { ggml_tensor * merged = nullptr, * twin_gpu = nullptr, * twin_cpu = nullptr, * argsort = nullptr; CacheLayer * C = nullptr; };
    DCheck dcheck_;
    Stats stats_;
    // stage f: the drafter (graph_mtp in llama.cpp's qwen35moe.cpp), its own weights, KV cells and scheduler
    struct Mtp {
        bool loaded = false;
        gguf_context * gguf = nullptr;
        ggml_context * meta = nullptr, * ctx_w = nullptr, * ctx_gpu = nullptr, * ctx_kv = nullptr;
        void * map = nullptr;
        size_t map_size = 0;
        ggml_backend_buffer_t buf_gpu = nullptr, buf_map = nullptr, buf_kv = nullptr;
        ggml_backend_sched_t sched = nullptr;
        ggml_tensor *tok_embd = nullptr, *output = nullptr, *eh_proj = nullptr, *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
        Layer l{};
        Cache kv;
    } mtp_;
    std::vector<float> mtp_h_prev_, win_h_;
    int win_pos0_ = 0;
    int rs_src_ = 0;                 // the snapshot the next ubatch reads its recurrent state from (0 = newest)
    int last_ub_n_ = 0;              // text tokens of the last target ubatch (its snapshots are the only valid ones)
    const float * embd_in_ = nullptr;
    const int32_t * pos4_in_ = nullptr;
    static constexpr int kCkpts = 4;  // checkpoints: every delta-net layer's conv + ssm state, on the GPU (the oldest goes)
    void * ckpt_[kCkpts] = {};
    size_t ckpt_bytes_ = 0;
    int ckpt_pos_[kCkpts] = { -1, -1, -1, -1 };
    uint64_t ckpt_age_[kCkpts] = {}, ckpt_clock_ = 0;
    std::vector<float> ckpt_h_[kCkpts];
    size_t slot_bytes() const;
    // the KV pager: each attention layer's K/V in its own buffer of kv_cap_ cells (a multiple of 256, <= kv_.size). The
    // reserve graphs see full-size tensors over the same memory (kfull_ / vfull_, metadata only, never computed), so the
    // allocator's plan is the one a whole-context KV gives
    struct KvLayer { ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr; };
    std::vector<KvLayer> kvl_;
    ggml_context * ctx_kvfull_ = nullptr;
    std::vector<ggml_tensor *> kfull_, vfull_;
    uint32_t kv_cap_ = 0;
    size_t kv_bytes(uint32_t cells) const;
    std::string kv_alloc_layer(int il, uint32_t cap);
    std::string kv_grow(uint32_t cells);
    std::string load_mtp();
    std::string mtp_ubatch(const int32_t * toks, const float * h, int n, int pos0, bool out, int32_t * amax, std::vector<float> * h_out);
    std::string mtp_catchup(const int32_t * toks, int n, int pos0, const float * h_rows);
};

} // namespace q35
