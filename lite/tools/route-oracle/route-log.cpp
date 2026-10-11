// Strata-Lite Gate 0: log the routed expert ids per layer per token (prefill and decode) from llama.cpp's MoE path.
// No change to llama.cpp itself: the scheduler's eval callback reads the "ffn_moe_topk-<il>" tensors
// (argsort_top_k view, [n_expert_used, n_tokens]) as they are computed.
//
// Usage (common llama.cpp args, plus env):
//   ROUTE_REQ=<OpenAI chat body .json>  ROUTE_OUT=<prefix>  [ROUTE_NGEN=512] [ROUTE_MAX_PROMPT=200000]
//   oracle mode for Gate 1 (Strata-Lite): ROUTE_LOGITS=1 also writes <prefix>.tokens (u32 n_prompt, u32 n_gen, then the
//   ids) and <prefix>.logits (f32 n_vocab per step: after the prompt, then after each generated token);
//   ROUTE_FORCE=<.tokens file> feeds that file's prompt and continuation instead of the request and the sampler
//   (teacher forcing: the same tokens, this run's logits)
//   route-log -m model.gguf -ngl 99 --n-cpu-moe N -fa on -c CTX -ctk q4_0 -ctv q4_0 --temp 0.6 ... --seed 42
// Writes <prefix>.routes (records: u8 phase(0 prefill,1 decode), u8 layer, u16 pad, u32 n_tokens, u32 pos0, then n_tokens*k u8 ids)
// and <prefix>.json (token counts, timings, generated text).
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct route_state {
    FILE *   f        = nullptr;
    FILE *   gdump    = nullptr;   // ROUTE_GRAPHDUMP: every node the scheduler runs (name op type ne op_params), one decode
    bool     gdump_on = false;
    bool     capture  = true;      // off in oracle mode: asking for a node splits the graph there (and can undo CUDA fusions)
    uint8_t  phase    = 0;
    uint32_t pos0     = 0;
    uint64_t records  = 0;
    uint64_t bad      = 0;
    std::vector<uint8_t> buf;
};

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
    if (!on && (strcmp(t->name, "inp_scaled") == 0 || strcmp(t->name, "model.input_embed") == 0) && t->ne[1] == n) { on = true; f = fopen(path, "w"); }
    if (!on) return;
    fprintf(f, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(f, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(f, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(f, "\t%d\n", (int) t->flags);
    if (strcmp(t->name, "result_output") == 0) { fclose(f); f = nullptr; on = false; path = nullptr; }
}

static bool route_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (route_state *) ud;
    if (ask && st->gdump_on) {
        fprintf(st->gdump, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        for (int i = 0; i < 16; ++i) fprintf(st->gdump, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
        for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(st->gdump, "\t%s", t->src[i] ? t->src[i]->name : "-");
        fprintf(st->gdump, "\n");
    }
    if (ask) gdump_node(t);
    if (tdump_want(t)) {
        if (ask) return true;
        tdump_put(t);
        return true;
    }
    if (!st->capture || strncmp(t->name, "ffn_moe_topk-", 13) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    const int il = atoi(t->name + 13);
    const int64_t k = t->ne[0], n = t->ne[1];
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    st->buf.resize(k * n);
    for (int64_t r = 0; r < n; ++r) {
        const int32_t * row = (const int32_t *) (raw.data() + r * t->nb[1]);
        for (int64_t j = 0; j < k; ++j) {
            int32_t id = row[j];
            if (id < 0 || id > 255) { st->bad++; id = 255; }
            st->buf[r * k + j] = (uint8_t) id;
        }
    }
    uint8_t  hdr0[4] = { st->phase, (uint8_t) il, (uint8_t) k, 0 };
    uint32_t hdr1[2] = { (uint32_t) n, st->pos0 };
    fwrite(hdr0, 1, 4, st->f);
    fwrite(hdr1, 4, 2, st->f);
    fwrite(st->buf.data(), 1, st->buf.size(), st->f);
    st->records++;
    return true;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char) c;
        }
    }
    return o;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const char * req_path = getenv("ROUTE_REQ");
    const char * out_pfx  = getenv("ROUTE_OUT");
    const int    n_gen    = getenv("ROUTE_NGEN") ? atoi(getenv("ROUTE_NGEN")) : 512;
    const int    max_prompt = getenv("ROUTE_MAX_PROMPT") ? atoi(getenv("ROUTE_MAX_PROMPT")) : 0;
    const bool dump_logits = getenv("ROUTE_LOGITS") != nullptr;
    const char * force_path = getenv("ROUTE_FORCE");
    if (!out_pfx || (!req_path && !force_path)) {
        fprintf(stderr, "set ROUTE_REQ and ROUTE_OUT\n");
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    route_state st;
    st.f = fopen((std::string(out_pfx) + ".routes").c_str(), "wb");
    if (!st.f) { fprintf(stderr, "cannot open output\n"); return 1; }
    st.capture = !dump_logits;
    params.cb_eval           = route_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_model *   model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks, forced;
    common_chat_templates_inputs ti;
    ti.enable_thinking = false;
    if (force_path) {
        FILE * ff = fopen(force_path, "rb");
        uint32_t np = 0, ng = 0;
        if (!ff || fread(&np, 4, 1, ff) != 1 || fread(&ng, 4, 1, ff) != 1) { fprintf(stderr, "bad %s\n", force_path); return 1; }
        std::vector<uint32_t> ids(np + ng);
        if (fread(ids.data(), 4, np + ng, ff) != np + ng) { fprintf(stderr, "short %s\n", force_path); return 1; }
        fclose(ff);
        toks.assign(ids.begin(), ids.begin() + np);
        forced.assign(ids.begin() + np, ids.end());
        req_path = force_path;
    } else {
    // the request, formatted with the model's own jinja template (as llama-server does)
    std::ifstream in(req_path);
    std::stringstream ss; ss << in.rdbuf();
    common_json body = common_json::parse(ss.str());
    auto tmpls = common_chat_templates_init(model, "");
    ti.messages = common_chat_msgs_parse_oaicompat(body.at("messages"));
    if (body.contains("tools")) {
        ti.tools = common_chat_tools_parse_oaicompat(body.at("tools"));
    }
    ti.use_jinja = true;
    ti.add_generation_prompt = true;
    if (body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").contains("enable_thinking")) {
        ti.enable_thinking = body.at("chat_template_kwargs").at("enable_thinking").get<bool>();
    }
    const auto chat = common_chat_templates_apply(tmpls.get(), ti);

    toks = common_tokenize(vocab, chat.prompt, true, true);
    }
    if (getenv("ROUTE_TOKENIZE_ONLY")) {   // Strata-Lite stage g: the prompt's token ids only (np, ng = 0)
        FILE * ft = fopen((std::string(out_pfx) + ".tokens").c_str(), "wb");
        uint32_t np = toks.size(), ng = 0;
        fwrite(&np, 4, 1, ft); fwrite(&ng, 4, 1, ft);
        for (auto t : toks) { uint32_t u = t; fwrite(&u, 4, 1, ft); }
        fclose(ft);
        fprintf(stderr, "route-log: tokenized %zu tokens\n", toks.size());
        return 0;
    }
    size_t n_dropped = 0;
    if (max_prompt > 0 && (int) toks.size() > max_prompt) {
        // keep the head (system) and the tail (the turn); drop the middle
        const size_t head = max_prompt / 4, tail = max_prompt - head;
        n_dropped = toks.size() - max_prompt;
        std::vector<llama_token> t2(toks.begin(), toks.begin() + head);
        t2.insert(t2.end(), toks.end() - tail, toks.end());
        toks.swap(t2);
    }
    const int n_ctx = llama_n_ctx(ctx);
    if ((int) toks.size() + n_gen + 8 > n_ctx) {
        fprintf(stderr, "prompt %zu + gen %d > ctx %d\n", toks.size(), n_gen, n_ctx);
        return 1;
    }
    fprintf(stderr, "route-log: prompt %zu tokens (dropped %zu), thinking %d, gen %d\n", toks.size(), n_dropped, (int) ti.enable_thinking, n_gen);

    // prefill in n_batch chunks
    const int n_batch = llama_n_batch(ctx);
    auto t0 = std::chrono::steady_clock::now();
    st.phase = 0;
    for (size_t i = 0; i < toks.size(); i += n_batch) {
        const int n = std::min((size_t) n_batch, toks.size() - i);
        st.pos0 = (uint32_t) i;
        llama_batch b = llama_batch_get_one(toks.data() + i, n);
        if (llama_decode(ctx, b)) { fprintf(stderr, "prefill decode failed at %zu\n", i); return 1; }
    }
    auto t1 = std::chrono::steady_clock::now();

    const int n_vocab = llama_vocab_n_tokens(vocab);
    FILE * flog = dump_logits ? fopen((std::string(out_pfx) + ".logits").c_str(), "wb") : nullptr;
    auto put_logits = [&]() { if (flog) fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), n_vocab, flog); };
    put_logits();
    std::vector<llama_token> gen;
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    for (auto t : toks) common_sampler_accept(smpl, t, false);
    std::string text;
    int n_out = 0;
    st.phase = 1;
    const bool forcing = force_path && !forced.empty();   // a .tokens file with no continuation: its prompt, then generate
    const int n_steps = forcing ? (int) forced.size() : n_gen;
    llama_token cur = forcing ? forced[0] : common_sampler_sample(smpl, ctx, -1);
    for (; n_out < n_steps; ++n_out) {
        common_sampler_accept(smpl, cur, true);
        gen.push_back(cur);
        if (!forcing && llama_vocab_is_eog(vocab, cur)) break;
        text += common_token_to_piece(ctx, cur);
        if (forcing && n_out + 1 == n_steps) break;     // the last forced token needs no forward
        st.pos0 = (uint32_t) (toks.size() + n_out);
        if (n_out == 2 && getenv("ROUTE_GRAPHDUMP")) { st.gdump = fopen(getenv("ROUTE_GRAPHDUMP"), "w"); st.gdump_on = true; }
        llama_batch b = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        if (st.gdump_on) { fclose(st.gdump); st.gdump_on = false; }
        put_logits();
        cur = forcing ? forced[n_out + 1] : common_sampler_sample(smpl, ctx, -1);
    }
    if (flog) fclose(flog);
    if (dump_logits) {
        FILE * ft = fopen((std::string(out_pfx) + ".tokens").c_str(), "wb");
        uint32_t np = toks.size(), ng = gen.size();
        fwrite(&np, 4, 1, ft); fwrite(&ng, 4, 1, ft);
        for (auto t : toks) { uint32_t u = t; fwrite(&u, 4, 1, ft); }
        for (auto t : gen)  { uint32_t u = t; fwrite(&u, 4, 1, ft); }
        fclose(ft);
    }
    auto t2 = std::chrono::steady_clock::now();
    fclose(st.f);

    const double pre_s = std::chrono::duration<double>(t1 - t0).count();
    const double dec_s = std::chrono::duration<double>(t2 - t1).count();
    FILE * j = fopen((std::string(out_pfx) + ".json").c_str(), "w");
    fprintf(j, "{\"req\":\"%s\",\"prompt_tokens\":%zu,\"dropped\":%zu,\"thinking\":%d,\"gen_tokens\":%d,"
               "\"prefill_s\":%.2f,\"decode_s\":%.2f,\"records\":%llu,\"bad_ids\":%llu,\"text\":\"%s\"}\n",
            json_escape(req_path).c_str(), toks.size(), n_dropped, (int) ti.enable_thinking, n_out, pre_s, dec_s,
            (unsigned long long) st.records, (unsigned long long) st.bad, json_escape(text).c_str());
    fclose(j);
    fprintf(stderr, "route-log: done prompt %zu (%.1f s, %.0f tok/s) gen %d (%.1f s, %.1f tok/s) records %llu bad %llu\n",
            toks.size(), pre_s, toks.size() / pre_s, n_out, dec_s, n_out / dec_s, (unsigned long long) st.records,
            (unsigned long long) st.bad);
    common_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
