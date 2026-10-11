// strata-gemma4: run a token file through the gemma4 engine and compare with llama.cpp's logits (Gate 1 checks).
//   strata-gemma4 -m MODEL.gguf --tokens X.tokens [--oracle X.logits] [--mode force|greedy] [--n-cpu-moe N] [-c CTX]
//                 [-t THREADS] [--graph-dump FILE] [--out PREFIX]
// X.tokens / X.logits come from route-log's oracle mode (ROUTE_LOGITS=1): u32 n_prompt, u32 n_gen, ids; f32 n_vocab per
// step (after the prompt, then after each generated token). force = feed the oracle's continuation and compare every
// step's logits; greedy = take the argmax and compare tokens.
#include "g4_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    g4::Options o;
    std::string tokens_path, oracle_path, mode = "force", out, mtp_path, dump_tokens;
    int n_draft = 3, n_gen = 0;   // n_gen: greedy length when the token file has no (or a shorter) continuation
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "-m") o.gguf = next();
        else if (a == "--tokens") tokens_path = next();
        else if (a == "--oracle") oracle_path = next();
        else if (a == "--mode") mode = next();
        else if (a == "--n-cpu-moe") o.n_cpu_moe = std::stoi(next());
        else if (a == "-c") o.n_ctx = std::stoi(next());
        else if (a == "-t") o.n_threads = std::stoi(next());
        else if (a == "--graph-dump") o.graph_dump = next();
        else if (a == "--out") out = next();
        else if (a == "--pool") o.pool = true;
        else if (a == "--mtp") mtp_path = next();
        else if (a == "--n-draft") n_draft = std::stoi(next());
        else if (a == "--n-gen") n_gen = std::stoi(next());
        else if (a == "--dump-tokens") dump_tokens = next();
        else if (a == "--pool-threads") o.pool_threads = std::stoi(next());
        else if (a == "--cache-slots") { std::string v = next(); o.cache_slots = v == "auto" ? -1 : std::stoi(v); }
        else if (a == "--cache-l0") o.cache_l0 = std::stoi(next());
        else if (a == "--vram-reserve-mb") o.vram_reserve_mb = std::stoi(next());
        else if (a == "--profile") o.profile = next();
        else if (a == "--dcheck-layer") o.dcheck_layer = std::stoi(next());
        else if (a == "--adapt") o.adapt = true;
        else if (a == "--adapt-half-life") o.adapt_half_life = std::stof(next());
        else if (a == "--adapt-budget") o.adapt_budget = std::stoi(next());
        else if (a == "--adapt-refill") o.adapt_refill = std::stoi(next()) != 0;
        else if (a == "--async-swaps") o.async_swaps = true;
        else if (a == "--overlap") o.overlap = true;
        else if (a == "--prefill-ubatch") { std::string v = next(); o.prefill_ubatch = v == "auto" ? -1 : std::stoi(v); }
        else if (a == "--kv-initial") o.kv_initial = std::stoi(next());   // KV pager: cells on the card at load (0 = all)
        else if (a == "--kv-step") o.kv_step = std::stoi(next());
        else if (a == "--lend") o.lend = true;
        else if (a == "-ctk" || a == "-ctv") {
            const std::string t = next();
            const ggml_type ty = t == "q4_0" ? GGML_TYPE_Q4_0 : t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
            (a == "-ctk" ? o.type_k : o.type_v) = ty;
        }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    FILE * f = fopen(tokens_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "no tokens file\n"); return 2; }
    uint32_t np = 0, ng = 0;
    if (fread(&np, 4, 1, f) != 1 || fread(&ng, 4, 1, f) != 1) return 2;
    std::vector<int32_t> ids(np + ng);
    if (fread(ids.data(), 4, np + ng, f) != np + ng) return 2;
    fclose(f);
    std::vector<int32_t> prompt(ids.begin(), ids.begin() + np), cont(ids.begin() + np, ids.end());

    if (o.prefill_ubatch > o.n_batch) o.n_batch = o.prefill_ubatch;   // a chunk is one llama_decode call
    if (o.prefill_ubatch < 0 && o.n_batch < 32768) o.n_batch = 32768;
    g4::Engine e;
    auto t0 = std::chrono::steady_clock::now();
    std::string err = e.load(o);
    if (!err.empty()) { fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    const int n_vocab = e.hp().n_vocab;
    const bool mtp = mode.rfind("mtp-", 0) == 0;
    if (mtp) {
        if (mtp_path.empty()) { fprintf(stderr, "--mtp PATH needed for %s\n", mode.c_str()); return 2; }
        err = e.load_mtp(mtp_path);
        if (!err.empty()) { fprintf(stderr, "mtp: %s\n", err.c_str()); return 1; }
    }
    FILE * fo = oracle_path.empty() ? nullptr : fopen(oracle_path.c_str(), "rb");
    std::vector<float> ref(n_vocab), lg, h_last;
    auto t1 = std::chrono::steady_clock::now();

    // the prompt in n_batch pieces, as route-log feeds llama_decode
    for (size_t i = 0; i < prompt.size(); i += o.n_batch) {
        std::vector<int32_t> piece(prompt.begin() + i, prompt.begin() + std::min(prompt.size(), i + o.n_batch));
        err = e.decode(piece, &lg, mtp ? &h_last : nullptr);
        if (!err.empty()) { fprintf(stderr, "prefill: %s\n", err.c_str()); return 1; }
    }
    auto t2 = std::chrono::steady_clock::now();

    int steps = 0, identical = 0, argmax_same = 0, first_diff = -1, first_tok_diff = -1, greedy_same = 0;
    double max_abs = 0, kl_sum = 0, kl_max = 0;
    FILE * fsteps = getenv("G4_KL_STEPS") ? fopen(getenv("G4_KL_STEPS"), "w") : nullptr;   // per-step "kl top1_same"
    std::vector<int32_t> gen;
    auto argmax = [&](const std::vector<float> & v) { return (int32_t) (std::max_element(v.begin(), v.end()) - v.begin()); };
    // compare one step's logits (predicting token `s` of the continuation) with the oracle's step s
    // G4_LOGIT_HASH=FILE: one FNV-1a hash of every scored step's logits (bit-for-bit comparison of two builds, no oracle)
    FILE * fhash = getenv("G4_LOGIT_HASH") ? fopen(getenv("G4_LOGIT_HASH"), "w") : nullptr;
    auto compare = [&](const float * L) {
        if (fhash) {
            uint64_t h = 1469598103934665603ull;
            const unsigned char * b = (const unsigned char *) L;
            for (size_t j = 0; j < (size_t) n_vocab * sizeof(float); ++j) { h ^= b[j]; h *= 1099511628211ull; }
            fprintf(fhash, "%016llx\n", (unsigned long long) h);
        }
        if (!(fo && fread(ref.data(), sizeof(float), n_vocab, fo) == (size_t) n_vocab)) return;
        steps++;
        if (memcmp(ref.data(), L, n_vocab * sizeof(float)) == 0) identical++;
        else if (first_diff < 0) first_diff = steps - 1;
        int ia = 0, ib = 0;
        for (int j = 0; j < n_vocab; ++j) {
            max_abs = std::max(max_abs, (double) fabsf(ref[j] - L[j]));
            if (ref[j] > ref[ia]) ia = j;
            if (L[j] > L[ib]) ib = j;
        }
        if (ia == ib) argmax_same++;
        const double ma = ref[ia], mb = L[ib];
        double za = 0, zb = 0;
        for (int j = 0; j < n_vocab; ++j) { za += exp(ref[j] - ma); zb += exp(L[j] - mb); }
        const double la = log(za), lb = log(zb);
        double kl = 0;
        for (int j = 0; j < n_vocab; ++j) { const double pa = ref[j] - ma - la; kl += exp(pa) * (pa - (L[j] - mb - lb)); }
        kl = std::max(kl, 0.0); kl_sum += kl; kl_max = std::max(kl_max, kl);
        if (fsteps) fprintf(fsteps, "%.9g %d\n", kl, (int) (ia == ib));
    };
    auto argmax_p = [&](const float * L) { return (int32_t) (std::max_element(L, L + n_vocab) - L); };
    long long drafted = 0, accepted = 0, windows = 0;
    if (!mtp) {
    for (size_t s = 0; ; ++s) {
        compare(lg.data());
        const int32_t pick = argmax(lg);
        gen.push_back(pick);
        if (s < cont.size()) {
            if (pick == cont[s] && first_tok_diff < 0) greedy_same++;
            else if (first_tok_diff < 0) first_tok_diff = (int) s;
        }
        if (s + 1 >= (mode == "force" ? cont.size() : std::max(cont.size(), (size_t) n_gen))) break;
        const int32_t next = mode == "force" ? cont[s] : pick;
        err = e.decode({ next }, &lg);
        if (!err.empty()) { fprintf(stderr, "decode: %s\n", err.c_str()); return 1; }
    }
    } else {
        // stage f: draft-mtp (llama.cpp common/speculative.cpp: drafts at one shared position, verify window, accept the
        // agreeing prefix, roll back the rest, carry the hidden row of the last accepted position)
        const bool force = mode == "mtp-force";
        const size_t total = force ? cont.size() : std::max(cont.size(), (size_t) n_gen);
        auto tok_at = [&](size_t i) { return force ? cont[i] : gen[i]; };
        compare(lg.data());
        if (total == 0) { fprintf(stderr, "nothing to generate: give --n-gen\n"); return 2; }
        gen.push_back(force ? cont[0] : argmax_p(lg.data()));
        std::vector<float> pending_h = h_last, L, Hr, hh;
        while (gen.size() < total) {
            const size_t m = gen.size();
            const int pos0 = e.n_past();
            const int32_t id_last = gen.back();
            std::vector<int32_t> win = { id_last };
            std::vector<float> hcur = pending_h;
            const int k = (int) std::min<size_t>(n_draft, total - m);
            for (int i = 0; i < k; ++i) {
                int32_t d = 0;
                err = e.draft(win.back(), hcur.data(), pos0, &d, &hh);
                if (!err.empty()) { fprintf(stderr, "draft: %s\n", err.c_str()); return 1; }
                win.push_back(d); hcur = hh;
            }
            err = e.decode_all(win, &L, &Hr);
            if (!err.empty()) { fprintf(stderr, "verify: %s\n", err.c_str()); return 1; }
            windows++; drafted += k;
            int n_acc = 0;
            for (; n_acc < k; ++n_acc) {
                const int32_t want = force ? cont[m + n_acc] : argmax_p(L.data() + (size_t) n_acc * n_vocab);
                if (win[1 + n_acc] != want) break;
                if (!force) gen.push_back(want);
                else gen.push_back(cont[m + n_acc]);
            }
            accepted += n_acc;
            for (int r = 0; r <= n_acc; ++r) compare(L.data() + (size_t) r * n_vocab);
            if (gen.size() < total) gen.push_back(force ? cont[m + n_acc] : argmax_p(L.data() + (size_t) n_acc * n_vocab));
            e.rollback(pos0 + n_acc + 1);
            pending_h.assign(Hr.begin() + (size_t) n_acc * e.hp().n_embd, Hr.begin() + (size_t) (n_acc + 1) * e.hp().n_embd);
        }
        for (size_t s = 0; s < gen.size() && s < cont.size(); ++s) {
            if (gen[s] == cont[s] && first_tok_diff < 0) greedy_same++;
            else if (first_tok_diff < 0) first_tok_diff = (int) s;
        }
        (void) tok_at;
    }
    if (!dump_tokens.empty()) { FILE * fd = fopen(dump_tokens.c_str(), "wb"); fwrite(gen.data(), 4, gen.size(), fd); fclose(fd); }
    auto t3 = std::chrono::steady_clock::now();
    if (fo) fclose(fo);
    if (fsteps) fclose(fsteps);
    if (fhash) fclose(fhash);
    const double ls = std::chrono::duration<double>(t1 - t0).count(), ps = std::chrono::duration<double>(t2 - t1).count(),
                 ds = std::chrono::duration<double>(t3 - t2).count();
    char buf[4096];
    snprintf(buf, sizeof buf,
             "{\"tokens\":\"%s\",\"mode\":\"%s\",\"pool\":%d,\"n_cpu_moe\":%d,\"prompt\":%u,\"cont\":%u,\"steps_compared\":%d,"
             "\"logits_bit_identical\":%d,\"first_nonidentical_step\":%d,\"max_abs_diff\":%.6g,\"argmax_same\":%d,"
             "\"greedy_tokens_matching_before_first_diff\":%d,\"first_token_diff\":%d,\"load_s\":%.1f,\"prefill_s\":%.2f,"
             "\"decode_s\":%.2f,\"cache_slots\":%d,\"cache_l0\":%d,\"accesses\":%llu,\"hits\":%llu,\"hit_rate\":%.4f,\"dcheck_rows\":%llu,\"dcheck_gpu_rows\":%llu,\"dcheck_bad\":%llu,\"decode_tok_s\":%.2f,\"kl_mean\":%.6g,\"kl_max\":%.6g,\"adapt\":%d,\"swaps\":%llu,\"swap_ms\":%.1f,\"refill_swaps\":%llu,\"refill_ms\":%.1f,\"swap_mb\":%.1f,\"mtp_windows\":%lld,\"drafted\":%lld,\"accepted\":%lld,\"tokens\":%zu,\"timeline\":{\"ubatches\":%llu,\"splits\":%llu,\"build_ms\":%.1f,\"compute_ms\":%.1f,\"pool_ms\":%.1f,\"readout_ms\":%.1f,\"adapt_ms\":%.1f,\"drafts\":%llu,\"draft_ms\":%.1f},\"commits\":%llu,\"pending_peak\":%llu,\"ov_layers\":%llu,\"ov_missfree\":%llu,\"join_wait_ms\":%.1f,\"prefill_ubatch\":%d,\"lend\":%d,\"read_ms\":%.1f,\"lend_ms\":%.1f,\"restore_ms\":%.1f,\"lent_gb\":%.2f,\"swa_big_cells\":%llu,\"refill_land_ms\":%.1f,\"arena_shrinks\":%d}",
             tokens_path.c_str(), mode.c_str(), (int) o.pool, o.n_cpu_moe, np, ng, steps, identical, first_diff, max_abs, argmax_same,
             greedy_same, first_tok_diff, ls, ps, ds, e.slots(), o.cache_l0, (unsigned long long) e.stats().accesses,
             (unsigned long long) e.stats().hits, e.stats().accesses ? (double) e.stats().hits / e.stats().accesses : 0.0,
             (unsigned long long) e.stats().dcheck_rows, (unsigned long long) e.stats().dcheck_gpu_rows,
             (unsigned long long) e.stats().dcheck_bad, ds > 0 ? (gen.size() - 1) / ds : 0.0, steps ? kl_sum / steps : 0.0, kl_max, (int) o.adapt, (unsigned long long) e.stats().swaps, e.stats().swap_ms,
             (unsigned long long) e.stats().refill_swaps, e.stats().refill_ms, e.stats().swap_bytes / 1e6, windows, drafted, accepted, gen.size(), (unsigned long long) e.stats().t_ubatches, (unsigned long long) e.stats().t_splits,
             e.stats().t_build, e.stats().t_compute, e.stats().t_pool, e.stats().t_readout, e.stats().t_scores,
             (unsigned long long) e.stats().t_drafts, e.stats().t_draft, (unsigned long long) e.stats().commits, (unsigned long long) e.stats().pending_peak, (unsigned long long) e.stats().ov_layers, (unsigned long long) e.stats().ov_missfree, e.stats().t_join_wait, o.prefill_ubatch, (int) o.lend, e.stats().prefill_ms,
             e.stats().lend_ms, e.stats().restore_ms, e.stats().lent_bytes / 1e9, (unsigned long long) e.stats().swa_big_cells, e.stats().refill_land_ms, e.stats().arena_shrinks);
    if (getenv("G4_STEP_HITS")) {   // per small ubatch after the read: routed pairs, hits
        FILE * fh = fopen(getenv("G4_STEP_HITS"), "w");
        for (size_t i = 0; i < e.stats().step_acc.size(); ++i) fprintf(fh, "%u %u\n", e.stats().step_acc[i], e.stats().step_hit[i]);
        fclose(fh);
    }
    printf("%s\n", buf);
    if (!out.empty()) { FILE * j = fopen((out + ".json").c_str(), "w"); fprintf(j, "%s\n", buf); fclose(j); }
    return 0;
}
