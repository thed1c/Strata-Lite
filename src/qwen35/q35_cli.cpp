// strata-qwen35: run a token file through the qwen35moe engine and compare with llama.cpp's logits (stage checks).
//   strata-qwen35 -m MODEL.gguf --tokens X.tokens [--oracle X.logits] [--mode force|greedy] [--n-cpu-moe N] [-c CTX]
//                 [-t THREADS] [-b N_BATCH] [--out PREFIX] [--mtp DRAFTER.gguf --n-draft N [--mtp-share-head]]
// X.tokens / X.logits come from route-oracle (ROUTE_LOGITS=1): u32 n_prompt, u32 n_gen, ids; f32 n_vocab per step
// (after the prompt, then after each generated token). force = feed the oracle's continuation and compare every step's
// logits; greedy = take the argmax and compare tokens. Stage f: mtp-force / mtp-greedy draft with the MTP head
// (llama.cpp draft-mtp, n_draft per window), verify each window in one ubatch, accept the agreeing prefix and roll the
// rest back; mtp-force accepts by the reference tokens, so every scored row comes from a verify window. Q35_KL_STEPS=FILE
// writes "kl top1_same" per scored step. Prints one JSON line (numbers only).
#include "q35_engine.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    q35::Options o;
    std::string tokens_path, oracle_path, mode = "force", out, dump_tokens;
    int n_batch = 2048, n_draft = 3, n_gen = 0;
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
        else if (a == "-b") n_batch = std::stoi(next());
        else if (a == "-ub") o.n_ubatch = std::stoi(next());
        else if (a == "--out") out = next();
        else if (a == "--pool") o.pool = true;
        else if (a == "--cache-slots") o.cache_slots = std::stoi(next());
        else if (a == "--cache-l0") o.cache_l0 = std::stoi(next());
        else if (a == "--profile") o.profile = next();
        else if (a == "--dcheck-layer") o.dcheck_layer = std::stoi(next());
        else if (a == "--adapt") o.adapt = true;
        else if (a == "--adapt-half-life") o.adapt_half_life = std::stof(next());
        else if (a == "--adapt-budget") o.adapt_budget = std::stoi(next());
        else if (a == "--pool-threads") o.pool_threads = std::stoi(next());
        else if (a == "--mtp") o.mtp = next();
        else if (a == "--n-draft") n_draft = std::stoi(next());
        else if (a == "--mtp-share-head") o.mtp_share_head = true;
        else if (a == "--async-swaps") o.async_swaps = true;
        else if (a == "--overlap") o.overlap = true;
        else if (a == "--prefill-ubatch") { std::string v = next(); o.prefill_ubatch = v == "auto" ? -1 : std::stoi(v); }
        else if (a == "--lend") o.lend = true;
        else if (a == "--n-gen") n_gen = std::stoi(next());
        else if (a == "--dump-tokens") dump_tokens = next();
        else if (a == "-ctk") { std::string t = next(); o.type_k = t == "q4_0" ? GGML_TYPE_Q4_0 : t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; }
        else if (a == "-ctv") { std::string t = next(); o.type_v = t == "q4_0" ? GGML_TYPE_Q4_0 : t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; }
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

    if (o.prefill_ubatch > n_batch) n_batch = o.prefill_ubatch;   // a chunk is one llama_decode call
    if (o.prefill_ubatch < 0) n_batch = INT32_MAX;   // auto: the whole prompt in one call, as the server reads a text run
    const bool mtp = mode.rfind("mtp-", 0) == 0;
    if (mtp && o.mtp.empty()) { fprintf(stderr, "--mtp PATH needed for %s\n", mode.c_str()); return 2; }
    if (!mtp) o.mtp.clear();
    if (mtp) o.n_rs_seq = n_draft;   // common_params_speculative::need_n_rs_seq: the draft max
    q35::Engine e;
    const auto t0 = std::chrono::steady_clock::now();
    std::string err = e.load(o);
    if (!err.empty()) { fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    const int n_vocab = e.hp().n_vocab;
    FILE * fo = oracle_path.empty() ? nullptr : fopen(oracle_path.c_str(), "rb");
    std::vector<float> ref(n_vocab), lg;
    const auto t1 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < prompt.size(); i += n_batch) {   // the prompt in n_batch pieces, as route-oracle feeds llama_decode
        std::vector<int32_t> piece(prompt.begin() + i, prompt.begin() + std::min(prompt.size(), i + n_batch));
        err = e.decode(piece, &lg);
        if (!err.empty()) { fprintf(stderr, "prefill: %s\n", err.c_str()); return 1; }
    }
    const auto t2 = std::chrono::steady_clock::now();

    FILE * fsteps = getenv("Q35_KL_STEPS") ? fopen(getenv("Q35_KL_STEPS"), "w") : nullptr;
    int steps = 0, identical = 0, argmax_same = 0, first_diff = -1, first_tok_diff = -1, greedy_same = 0;
    double max_abs = 0, kl_sum = 0, kl_max = 0;
    // Q35_LOGIT_HASH=FILE: one FNV-1a hash of every scored step's logits (bit-for-bit comparison of two builds, no oracle)
    FILE * fhash = getenv("Q35_LOGIT_HASH") ? fopen(getenv("Q35_LOGIT_HASH"), "w") : nullptr;
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
    std::vector<int32_t> gen;
    long long drafted = 0, accepted = 0, windows = 0;
    if (!mtp) {
        const size_t total = mode == "force" ? cont.size() : std::max(cont.size(), (size_t) n_gen);
        for (size_t s = 0; ; ++s) {
            compare(lg.data());
            const int32_t pick = argmax_p(lg.data());
            gen.push_back(pick);
            if (s < cont.size()) {
                if (pick == cont[s] && first_tok_diff < 0) greedy_same++;
                else if (first_tok_diff < 0) first_tok_diff = (int) s;
            }
            if (s + 1 >= total) break;
            err = e.decode({ mode == "force" ? cont[s] : pick }, &lg);
            if (!err.empty()) { fprintf(stderr, "decode: %s\n", err.c_str()); return 1; }
        }
    } else {
        // draft-mtp (common/speculative.cpp + the server's verify): drafts at pos0, pos0 + 1, ... from the pending hidden
        // row; the drafter's draft cells dropped; [last, d1..dk] verified in one ubatch (the drafter re-reads it); the
        // agreeing prefix accepted, the rest rolled back (recurrent snapshot), the kept row's hidden state carried on
        const bool force = mode == "mtp-force";
        const size_t total = force ? cont.size() : std::max(cont.size(), (size_t) n_gen);
        compare(lg.data());
        if (total == 0) { fprintf(stderr, "nothing to generate: give --n-gen\n"); return 2; }
        gen.push_back(force ? cont[0] : argmax_p(lg.data()));
        std::vector<float> L, Hr, hcur, hh;
        while (gen.size() < total) {
            const size_t m = gen.size();
            const int pos0 = e.n_past();
            std::vector<int32_t> win = { gen.back() };
            hcur = e.mtp_h();
            const int k = (int) std::min<size_t>(n_draft, total - m);
            for (int i = 0; i < k; ++i) {
                int32_t d = 0;
                err = e.draft(win.back(), hcur.data(), pos0 + i, &d, &hh);
                if (!err.empty()) { fprintf(stderr, "draft: %s\n", err.c_str()); return 1; }
                win.push_back(d); hcur = hh;
            }
            e.mtp_trim(pos0);
            err = e.decode_all(win, &L, &Hr);
            if (!err.empty()) { fprintf(stderr, "verify: %s\n", err.c_str()); return 1; }
            windows++; drafted += k;
            int n_acc = 0;
            for (; n_acc < k; ++n_acc) {
                const int32_t want = force ? cont[m + n_acc] : argmax_p(L.data() + (size_t) n_acc * n_vocab);
                if (win[1 + n_acc] != want) break;
                gen.push_back(want);
            }
            accepted += n_acc;
            for (int r = 0; r <= n_acc; ++r) compare(L.data() + (size_t) r * n_vocab);
            if (gen.size() < total) gen.push_back(force ? cont[m + n_acc] : argmax_p(L.data() + (size_t) n_acc * n_vocab));
            err = e.rollback(pos0 + n_acc + 1);
            if (!err.empty()) { fprintf(stderr, "rollback: %s\n", err.c_str()); return 1; }
        }
        for (size_t s = 0; s < gen.size() && s < cont.size(); ++s) {
            if (gen[s] == cont[s] && first_tok_diff < 0) greedy_same++;
            else if (first_tok_diff < 0) first_tok_diff = (int) s;
        }
    }
    if (!dump_tokens.empty()) { FILE * fd = fopen(dump_tokens.c_str(), "wb"); if (fd) { fwrite(gen.data(), 4, gen.size(), fd); fclose(fd); } }
    if (fsteps) fclose(fsteps);
    if (fhash) fclose(fhash);
    const auto t3 = std::chrono::steady_clock::now();
    auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    char buf[4096];
    snprintf(buf, sizeof buf,
             "{\"tokens\":\"%s\",\"mode\":\"%s\",\"n_cpu_moe\":%d,\"prompt\":%u,\"cont\":%u,\"steps_compared\":%d,"
             "\"logits_bit_identical\":%d,\"first_nonidentical_step\":%d,\"max_abs_diff\":%.6g,\"argmax_same\":%d,"
             "\"greedy_tokens_matching_before_first_diff\":%d,\"first_token_diff\":%d,\"kl_mean\":%.6g,\"kl_max\":%.6g,"
             "\"load_s\":%.2f,\"prefill_s\":%.2f,\"decode_s\":%.2f,\"cache_slots\":%d,\"cache_l0\":%d,\"accesses\":%llu,"
             "\"hits\":%llu,\"hit_rate\":%.4f,\"dcheck_rows\":%llu,\"dcheck_gpu_rows\":%llu,\"dcheck_bad\":%llu,\"swaps\":%llu,\"refill_swaps\":%llu,\"swap_mb\":%.1f,\"swap_ms\":%.1f,"
             "\"decode_tok_s\":%.2f,\"tokens\":%zu,\"n_draft\":%d,\"mtp_windows\":%lld,\"drafted\":%lld,\"accepted\":%lld,\"drafts\":%llu,"
             "\"draft_ms\":%.1f,\"catchup_tokens\":%llu,\"catchup_ms\":%.1f,\"share_head\":%d,\"async_swaps\":%d,\"overlap\":%d,"
             "\"commits\":%llu,\"pending_peak\":%llu,\"ov_layers\":%llu,\"ov_missfree\":%llu,\"join_wait_ms\":%.1f,\"pool_ms\":%.1f,"
             "\"prefill_ubatch\":%d,\"lend\":%d,\"last_chunk\":%d,\"read_ms\":%.1f,\"lend_ms\":%.1f,\"restore_ms\":%.1f,\"lent_gb\":%.2f,"
             "\"read_buf_mb\":%.0f,\"refill_land_ms\":%.1f,\"slots_end\":%d,\"arena_shrinks\":%d}",
             tokens_path.c_str(), mode.c_str(), o.n_cpu_moe, np, ng, steps, identical, first_diff, max_abs, argmax_same, greedy_same,
             first_tok_diff, steps ? kl_sum / steps : 0.0, kl_max, sec(t0, t1), sec(t1, t2), sec(t2, t3), o.cache_slots, o.cache_l0,
             (unsigned long long) e.stats().accesses, (unsigned long long) e.stats().hits,
             e.stats().accesses ? (double) e.stats().hits / e.stats().accesses : 0.0, (unsigned long long) e.stats().dcheck_rows,
             (unsigned long long) e.stats().dcheck_gpu_rows, (unsigned long long) e.stats().dcheck_bad,
             (unsigned long long) e.stats().swaps, (unsigned long long) e.stats().refill_swaps, e.stats().swap_bytes / 1e6, e.stats().swap_ms,
             sec(t2, t3) > 0 && !gen.empty() ? (gen.size() - 1) / sec(t2, t3) : 0.0, gen.size(), mtp ? n_draft : 0, windows, drafted, accepted,
             (unsigned long long) e.stats().drafts, e.stats().draft_ms, (unsigned long long) e.stats().catchup_tokens, e.stats().catchup_ms,
             (int) o.mtp_share_head, (int) o.async_swaps, (int) o.overlap, (unsigned long long) e.stats().commits,
             (unsigned long long) e.stats().pending_peak, (unsigned long long) e.stats().ov_layers, (unsigned long long) e.stats().ov_missfree,
             e.stats().join_wait_ms, e.stats().pool_ms, o.prefill_ubatch, (int) o.lend, e.stats().last_chunk, e.stats().prefill_ms,
             e.stats().lend_ms, e.stats().restore_ms, e.stats().lent_bytes / 1e9, e.stats().read_buf_bytes / 1048576.0, e.stats().refill_land_ms,
             e.slots(), e.stats().arena_shrinks);
    printf("%s\n", buf);
    if (getenv("Q35_STEP_HITS")) {   // per small ubatch after load: routed pairs of the cached layers, hits
        FILE * fh = fopen(getenv("Q35_STEP_HITS"), "w");
        if (fh) { for (auto & [a, h] : e.stats().step_hits) fprintf(fh, "%u %u\n", a, h); fclose(fh); }
    }
    if (!out.empty()) { FILE * fj = fopen((out + ".json").c_str(), "w"); if (fj) { fprintf(fj, "%s\n", buf); fclose(fj); } }
    return 0;
}
