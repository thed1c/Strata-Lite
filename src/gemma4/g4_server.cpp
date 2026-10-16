#include "../lite/platform.hpp"
#include "../lite/chat_page.hpp"
#include "../lite/utf8.hpp"
#include <filesystem>
#include <thread>
extern "C" int stbi_info_from_memory(const unsigned char * buffer, int len, int * x, int * y, int * comp);   // in libmtmd (stb_image)
#include <cstring>
#include "gguf.h"
// strata-lite-server: an OpenAI-compatible server for the gemma4 engine, API-identical to llama-server for what
// the reference server's clients use (docs/API.md): POST /v1/chat/completions (and /chat/completions), GET /health,
// /v1/models (/models), /props, /slots.
//
// llama-server's own code does the request parsing (oaicompat_chat_params_parse: chat template, tools, response_format
// -> grammar, chat_template_kwargs, image parts -> media markers), the image tokenization (process_mtmd_prompt), the
// sampling parameters (eval_llama_cmpl_schema), the sampler chain with its grammar and reasoning budget (common_sampler,
// through common_sampler_sample_logits) and the response JSON (server_task_result_cmpl_final). The engine does the
// compute: text and image chunks, MTP verify windows (llama.cpp's sample-and-accept), the prompt cache.
// One request at a time (the reference server runs --parallel 1).
#include "g4_engine.hpp"

#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace g4srv {

struct Args {
    std::string model, mmproj, mtp, alias = "strata-lite", host = "127.0.0.1", profile;
    int port = 8080, n_draft = 3, image_max_tokens = 384, threads = 0;   // 384: the maintainer, 2026-10-10; threads 0 = physical cores
    bool reserve_set = false;
    std::string mmproj_place;   // second: on a second visible GPU (default when one has room); gpu: always on the engine's card;
                                // cpu: in RAM; auto: in RAM, on the card while images need it;
                                // lazy: not loaded until an image comes (RAM or card by the rules), unloaded when idle. Default: lazy
                                // under 32 GB of RAM, auto above
    int mmproj_idle = 15, mmproj_min_size = 128, mmproj_gpu_max_ctx = 65536;   // 15 s, 64k: the maintainer, 2026-10-10
    g4::Options eo;
    common_params params;
};

static void usage() {
    fprintf(stderr, "strata-lite-server -m MODEL [--mmproj F] [--mtp F] [--alias A] [--host H] [--port P] [-c CTX] [-ub N]\n"
                    "  [--mmproj-place second|auto|lazy|gpu|cpu --mmproj-min-size PX --mmproj-gpu-max-ctx N --mmproj-idle S]\n"
                    "  [--adapt-half-life T --adapt-budget N --no-adapt --no-adapt-refill --no-async-swaps --no-overlap --no-lend]\n"
                    "  [--temp T --top-k K --top-p P --min-p M --repeat-penalty R] [--image-max-tokens N] [-t N]\n"
                    "  [--n-cpu-moe N --cache-slots S --profile F --pool-threads N --prefill-ubatch N --lend -ctk T -ctv T --n-draft N]\n");
}

static ggml_type kv_type(const std::string & t) { return t == "q4_0" ? GGML_TYPE_Q4_0 : t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; }

// strata-lite-server is the one entry point: a GGUF of another architecture goes to that engine's server with the same
// arguments. The release build links both servers into one binary (STRATA_LITE_COMBINED); a per-engine build starts
// strata-lite-server-qwen35 from $STRATA_LITE_QWEN35_SERVER or next to this binary.
static std::string model_architecture(int argc, char ** argv) {
    std::string model;
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) model = argv[i + 1];
    if (model.empty()) return "";
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(model.c_str(), gp);
    if (!g) return "";
    const int64_t id = gguf_find_key(g, "general.architecture");
    const std::string arch = id < 0 ? "" : gguf_get_val_str(g, id);
    gguf_free(g);
    return arch;
}

#if !defined(STRATA_LITE_COMBINED) && !defined(_WIN32)
static void exec_qwen_server(int argc, char ** argv) {
    std::string exe;
    if (const char * e = getenv("STRATA_LITE_QWEN35_SERVER")) exe = e;
    else exe = lite::self_dir() + "strata-lite-server-qwen35";
    std::vector<char *> av = { (char *) exe.c_str() };
    for (int i = 1; i < argc; ++i) av.push_back(argv[i]);
    av.push_back(nullptr);
    fprintf(stderr, "strata-lite-server: a hybrid MoE model: starting %s\n", exe.c_str());
    execv(exe.c_str(), av.data());
    fprintf(stderr, "strata-lite-server: cannot start %s (build src/qwen35, or set STRATA_LITE_QWEN35_SERVER)\n", exe.c_str());
    exit(1);
}
#endif

int run(int argc, char ** argv) {
    Args A;
    common_params & P = A.params;
    // neutral public defaults (the maintainer, 2026-10-10: "we will make them neutral public"): llama-server's sampling, a
    // neutral alias and port, threads from the machine, no profile needed
    P.use_jinja = true;
    P.n_parallel = 1;
    A.eo.n_ctx = 262144; A.eo.n_cpu_moe = 30; A.eo.pool = true; A.eo.pool_threads = 0; A.eo.cache_slots = -1;
    A.eo.adapt = true; A.eo.async_swaps = true; A.eo.overlap = true; A.eo.prefill_ubatch = -1; A.eo.lend = true;
    A.eo.kv_initial = 32768;   // the KV pager: 32k global-layer cells on the card at load, grown as needed (0 = all at load)
    // q8_0 K/V (the maintainer, 2026-10-11): at 64k q4_0 costs Gemma 4 top-1 80.5 % vs a noise floor of 87.1 %, q8_0 is back at
    // 87.1 % (Qwen3.6 keeps q4_0: its q4 cost is inside the floor); the KV pager keeps the extra VRAM to what the chat uses
    A.eo.type_k = GGML_TYPE_Q8_0; A.eo.type_v = GGML_TYPE_Q8_0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { usage(); exit(2); } return std::string(argv[++i]); };
        if (a == "-m" || a == "--model") A.model = nx();
        else if (a == "--mmproj") A.mmproj = nx();
        else if (a == "--mtp") A.mtp = nx();
        else if (a == "--alias") A.alias = nx();
        else if (a == "--host") A.host = nx();
        else if (a == "--port") A.port = std::stoi(nx());
        else if (a == "-c" || a == "--ctx-size") A.eo.n_ctx = std::stoi(nx());
        else if (a == "-ub" || a == "--ubatch-size") A.eo.n_ubatch = std::stoi(nx());
        else if (a == "--temp") P.sampling.temp = std::stof(nx());
        else if (a == "--top-k") P.sampling.top_k = std::stoi(nx());
        else if (a == "--top-p") P.sampling.top_p = std::stof(nx());
        else if (a == "--min-p") P.sampling.min_p = std::stof(nx());
        else if (a == "--repeat-penalty") P.sampling.penalty_repeat = std::stof(nx());
        else if (a == "--image-max-tokens") A.image_max_tokens = std::stoi(nx());
        else if (a == "--mmproj-place") A.mmproj_place = nx();
        else if (a == "--mmproj-idle") A.mmproj_idle = std::stoi(nx());
        else if (a == "--mmproj-min-size") A.mmproj_min_size = std::stoi(nx());   // px: smaller images encode on the CPU (auto)
        else if (a == "--mmproj-gpu-max-ctx") A.mmproj_gpu_max_ctx = std::stoi(nx());
        else if (a == "-t" || a == "--threads") A.threads = std::stoi(nx());
        else if (a == "--n-cpu-moe") A.eo.n_cpu_moe = std::stoi(nx());
        else if (a == "--cache-slots") { std::string v = nx(); A.eo.cache_slots = v == "auto" ? -1 : std::stoi(v); }
        else if (a == "--vram-reserve-mb") { A.eo.vram_reserve_mb = std::stoi(nx()); A.reserve_set = true; }
        else if (a == "--profile") A.eo.profile = nx();
        else if (a == "--pool-threads") A.eo.pool_threads = std::stoi(nx());
        else if (a == "--prefill-ubatch") { std::string v = nx(); A.eo.prefill_ubatch = v == "auto" ? -1 : std::stoi(v); }
        else if (a == "--kv-initial") A.eo.kv_initial = std::stoi(nx());   // KV pager: cells on the card at load (0 = all)
        else if (a == "--kv-step") A.eo.kv_step = std::stoi(nx());
        else if (a == "--no-lend") A.eo.lend = false;
        // the expert cache's behaviour (defaults: Gate 0's adaptive policy, the copy stream and the CPU/GPU overlap on)
        else if (a == "--adapt-half-life") A.eo.adapt_half_life = std::stof(nx());     // tokens; usage counts halve over this many
        else if (a == "--adapt-budget") A.eo.adapt_budget = std::stoi(nx());          // swaps per layer per decode step
        else if (a == "--no-adapt") A.eo.adapt = false;                               // a fixed cache (the profile's, or plain order)
        else if (a == "--no-adapt-refill") A.eo.adapt_refill = false;                 // keep the cache as it was after a prompt read
        else if (a == "--no-async-swaps") A.eo.async_swaps = false;                   // swaps copied in the step, not in the background
        else if (a == "--no-overlap") A.eo.overlap = false;                           // the CPU pool waits for the GPU, no overlap
        else if (a == "-ctk") A.eo.type_k = kv_type(nx());
        else if (a == "-ctv") A.eo.type_v = kv_type(nx());
        else if (a == "--n-draft") A.n_draft = std::stoi(nx());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 2; }
    }
    if (A.model.empty()) { usage(); return 2; }
    if (!A.eo.adapt) {   // a fixed cache: no swaps to copy, and a lent arena would come back empty with nothing to refill it
        A.eo.async_swaps = false;
        if (A.eo.lend) { A.eo.lend = false; fprintf(stderr, "strata-lite: --no-adapt reads without lending the cache's VRAM\n"); }
    }
    if (A.threads <= 0 || A.eo.pool_threads <= 0) {   // physical cores among the CPUs this process may run on (numactl respected)
        const int cores = lite::physical_cores_or_half();
        if (A.threads <= 0) A.threads = cores;
        if (A.eo.pool_threads <= 0) A.eo.pool_threads = cores;
    }
    A.eo.gguf = A.model; A.eo.n_threads = A.threads;
    if (A.eo.prefill_ubatch > A.eo.n_batch) A.eo.n_batch = A.eo.prefill_ubatch;
    if (A.eo.prefill_ubatch < 0 && A.eo.n_batch < 32768) A.eo.n_batch = 32768;
    P.model.path = A.model;
    P.n_ctx = A.eo.n_ctx;

    llama_backend_init();
    // the vocabulary, the chat template and the tokenizer come from llama.cpp (vocab-only load: no weights)
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(A.model.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load the vocabulary\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n_vocab; ++t) if (llama_vocab_is_eog(vocab, t)) P.sampling.logit_bias_eog.push_back({ t, -INFINITY });

    // auto slots: what loads after the arena (the drafter, the projector and its largest image encode) plus step headroom
    auto fsize = [](const std::string & p) -> size_t { std::error_code ec; const auto n = p.empty() ? 0 : std::filesystem::file_size(p, ec); return ec ? 0 : (size_t) n; };
    // a second visible GPU takes the projector whole (the maintainer, 2026-10-10: "if there is a second card detected it
    // automatically puts mmproj there"): no expert slots given up, no moves, no CPU encodes
    ggml_backend_dev_t second_dev = nullptr;
    size_t second_free = 0;
    {
        int k = 0;
        for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(d);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if (k++ == 1) { size_t tot = 0; ggml_backend_dev_memory(dev, &second_free, &tot); second_dev = dev; break; }
        }
    }
    const bool second_fits = second_dev && second_free >= (((fsize(A.mmproj) >> 20) + 768) << 20);
    if (A.mmproj_place.empty()) {
        const double gb = lite::total_ram_gb();
        A.mmproj_place = second_fits ? "second" : gb > 0 && gb < 32 ? "lazy" : "auto";   // the maintainer, 2026-10-10: "32Gb for mmproj"
    }
    if (A.mmproj_place != "second" && A.mmproj_place != "gpu" && A.mmproj_place != "cpu" && A.mmproj_place != "auto" && A.mmproj_place != "lazy") {
        fprintf(stderr, "--mmproj-place second|gpu|cpu|auto|lazy\n"); return 1;
    }
    if (A.mmproj_place == "second" && !A.mmproj.empty() && !second_fits) {
        fprintf(stderr, "--mmproj-place second: %s\n", second_dev ? "the second GPU has too little free memory" : "no second GPU is visible"); return 1;
    }
    const bool mm_gpu_always = !A.mmproj.empty() && A.mmproj_place == "gpu", mm_auto = !A.mmproj.empty() && A.mmproj_place == "auto";
    const bool mm_lazy = !A.mmproj.empty() && A.mmproj_place == "lazy";
    const bool mm_second = !A.mmproj.empty() && A.mmproj_place == "second";
    const size_t mmproj_card = (fsize(A.mmproj) >> 20) + 768;   // the projector on the card + its 1024 px encode, MiB
    const size_t after_mb = 256 + (A.mtp.empty() ? 0 : (fsize(A.mtp) >> 20) + 128) + (mm_gpu_always ? mmproj_card - 256 : 0);
    if (A.eo.cache_slots < 0 && !A.reserve_set) A.eo.vram_reserve_mb = (int) after_mb;
    g4::Engine eng;
    std::string err = eng.load(A.eo);
    if (!err.empty()) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
    const bool mtp = !A.mtp.empty();
    if (A.eo.cache_slots >= 0) eng.ensure_free(after_mb << 20, true);   // fixed slots that leave no room: fewer
    if (mtp) { err = eng.load_mtp(A.mtp); if (!err.empty()) { fprintf(stderr, "mtp: %s\n", err.c_str()); return 1; } }

    // the projector: on the card (gpu), in RAM only (cpu), or in RAM with a card copy while images need it (auto: small
    // images such as a watchdog probe encode on the CPU; bigger ones move the projector to the card, slots give way, and
    // after mmproj_idle seconds without one it leaves the card and the slots come back)
    auto mm_init = [&](bool gpu, ggml_backend_dev_t dev = nullptr) -> mtmd_context * {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = gpu; mparams.device = gpu ? (dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) : nullptr;
        mparams.print_timings = false; mparams.n_threads = A.threads;
        mparams.image_max_tokens = A.image_max_tokens; mparams.media_marker = get_media_marker();
        return mtmd_init_from_file(A.mmproj.c_str(), model, mparams);
    };
    auto mm_warm = [&](mtmd_context * c) {   // the encoder's compute buffer at Studio's largest image (1024 px), kept by mtmd
        std::vector<unsigned char> px((size_t) 1024 * 1024 * 3, 128);
        mtmd_bitmap * bm = mtmd_bitmap_init(1024, 1024, px.data());
        mtmd_input_chunks * chs = mtmd_input_chunks_init();
        const std::string marker = get_media_marker();
        mtmd_input_text txt; txt.text = marker.c_str(); txt.text_len = marker.size(); txt.add_special = false; txt.parse_special = true;
        const mtmd_bitmap * bms[1] = { bm };
        if (mtmd_tokenize(c, chs, &txt, bms, 1) == 0)
            for (size_t i = 0; i < mtmd_input_chunks_size(chs); ++i) {
                const mtmd_input_chunk * k = mtmd_input_chunks_get(chs, i);
                if (mtmd_input_chunk_get_type(k) == MTMD_INPUT_CHUNK_TYPE_IMAGE) mtmd_encode_chunk(c, k);
            }
        mtmd_input_chunks_free(chs); mtmd_bitmap_free(bm);
    };
    // mctx: the projector in RAM (cpu/auto: resident; lazy: while images come); gpu: the card copy (mctx too when gpu-always)
    std::atomic<mtmd_context *> mctx{ nullptr }, mctx_gpu{ nullptr };
    if (!A.mmproj.empty() && !mm_lazy) {
        mctx = mm_init(mm_gpu_always || mm_second, mm_second ? second_dev : nullptr);
        if (!mctx) { fprintf(stderr, "cannot load the mmproj\n"); return 1; }
        if (mm_second) { mm_warm(mctx); fprintf(stderr, "strata-lite: projector on the second GPU (%s, %zu MiB free before)\n", ggml_backend_dev_description(second_dev), second_free >> 20); }
        if (mm_gpu_always) { mctx_gpu = mctx.load(); mm_warm(mctx_gpu); }
    }
    std::atomic<int64_t> mm_last_gpu{ 0 }, mm_last_use{ 0 };
    auto now_s = []() -> int64_t { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    std::atomic<int> mm_moves{ 0 };
    eng.ensure_free((size_t) 256 << 20, true);   // step headroom
    const int slots_full = eng.slots();
    auto need_gpu = [&]() -> mtmd_context * {   // the projector on the card: slots give way first
        if (!mctx_gpu) {
            const auto tm0 = std::chrono::steady_clock::now();
            eng.ensure_free(mmproj_card << 20);
            mctx_gpu = mm_init(true);
            if (!mctx_gpu) return nullptr;
            mm_moves++;
            fprintf(stderr, "strata-lite: projector to the card in %.2f s (%d slots per layer)\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tm0).count(), eng.slots());
        }
        return mctx_gpu;
    };
    auto need_cpu = [&]() -> mtmd_context * {   // lazy: the projector into RAM
        if (!mctx) {
            const auto tm0 = std::chrono::steady_clock::now();
            mctx = mm_init(false);
            if (mctx) fprintf(stderr, "strata-lite: projector loaded into RAM in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tm0).count());
        }
        return mctx;
    };
    fprintf(stderr, "strata-lite: %d expert slots per layer (projector: %s)\n", eng.slots(), A.mmproj.empty() ? "none" : A.mmproj_place.c_str());

    server_chat_params chat;
    {
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, P.chat_template);
        const bool thinking = P.enable_reasoning != 0 && common_chat_templates_support_enable_thinking(tmpls.get());
        chat = { P.use_jinja, P.prefill_assistant, P.reasoning_format, P.default_template_kwargs, std::move(tmpls),
                 !A.mmproj.empty(), false, false, thinking,
                 P.sampling.reasoning_budget_tokens, P.sampling.reasoning_budget_message, P.media_path, P.force_pure_content_parser };
    }

    // ---------------------------------------------------------------- generation
    std::mutex busy;
    std::atomic<bool> processing{ false };
    server_tokens cached;                 // the prompt cache: what the engine's KV holds (prompt + generated)
    std::vector<float> h_pending;         // MTP: the hidden row that pairs with the next token

    auto generate = [&](const json & body_in, int & http_code, bool chat_api) -> json {
        json body = body_in;
        std::vector<raw_buffer> files;
        std::vector<int> img_side;
        size_t img_k = 0;
        json data;
        if (chat_api) {
            try { data = oaicompat_chat_params_parse(body, chat, files); }
            catch (const std::exception & e) { http_code = 400; return format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST); }
        } else {   // /completion: llama-server's native endpoint; "prompt" may be text or token ids
            data = body;
            if (!data.contains("prompt")) { http_code = 400; return format_error_response("\"prompt\" is required", ERROR_TYPE_INVALID_REQUEST); }
        }
        server_tokens prompt;
        mtmd_context * tok_ctx = nullptr;   // the projector copy that tokenized this request (lazy: it also encodes)
        if (!files.empty()) {
            if (A.mmproj.empty()) { http_code = 400; return format_error_response("image input is not supported (no --mmproj)", ERROR_TYPE_NOT_SUPPORTED); }
            bool any_big = false;
            for (const auto & fb : files) {   // each image's longer side in pixels from its header, in prompt order (--mmproj-min-size)
                int x = 0, y = 0, c = 0;
                const int side = stbi_info_from_memory(fb.data(), (int) fb.size(), &x, &y, &c) ? std::max(x, y) : 0;
                img_side.push_back(side); any_big = any_big || side >= A.mmproj_min_size;
            }
            if (mm_lazy) {   // load only the copy this request needs: the card for big images under the context limit, else RAM
                const std::string ptxt = data.at("prompt").get<std::string>();
                const size_t est = common_tokenize(vocab, ptxt, true, true).size() + files.size() * (size_t) A.image_max_tokens;
                tok_ctx = (any_big && est <= (size_t) A.mmproj_gpu_max_ctx) ? need_gpu() : need_cpu();
            } else tok_ctx = mctx;
            if (!tok_ctx) { http_code = 500; return format_error_response("cannot load the mmproj", ERROR_TYPE_SERVER); }
            mm_last_use = now_s();
            prompt = process_mtmd_prompt(tok_ctx, data.at("prompt").get<std::string>(), files, mtmd_helper_init_opt_default());
        } else {
            auto v = tokenize_input_prompts(vocab, nullptr, data.at("prompt"), true, true, mtmd_helper_init_opt_default());
            prompt = std::move(v.at(0));
        }
        task_params tp;
        try { tp = server_schema::eval_llama_cmpl_schema(vocab, P, P.sampling.logit_bias_eog, data); }
        catch (const std::exception & e) { http_code = 400; return format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST); }
        tp.res_type = chat_api ? TASK_RESPONSE_TYPE_OAI_CHAT : TASK_RESPONSE_TYPE_NONE;
        tp.oaicompat_model = A.alias;
        tp.oaicompat_cmpl_id = gen_chatcmplid();
        const int n_ctx = eng.n_ctx(), n_prompt = (int) prompt.size();
        if (n_prompt >= n_ctx) {   // --no-context-shift: a prompt that does not fit is the client's error (4xx)
            http_code = 400;
            json e = format_error_response("the request exceeds the available context size, try increasing it", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            e["n_prompt_tokens"] = n_prompt; e["n_ctx"] = n_ctx;
            return e;
        }

        // auto: a request past mmproj_gpu_max_ctx keeps its images on the CPU, and a projector on the card leaves it first,
        // so the long read and its steps get those slots back (the maintainer, 2026-10-10)
        const bool long_ctx = (mm_auto || mm_lazy) && n_prompt > A.mmproj_gpu_max_ctx;
        if (long_ctx && mctx_gpu && tok_ctx != mctx_gpu.load()) {
            mtmd_free(mctx_gpu); mctx_gpu = nullptr;
            const int s0 = eng.slots();
            eng.grow_to(slots_full, (size_t) 256 << 20);
            fprintf(stderr, "strata-lite: projector off the card for a %d-token prompt (%d -> %d slots per layer)\n", n_prompt, s0, eng.slots());
        }
        server_slot_stats stats;
        stats.update_prompt_start();
        // prompt cache: keep the common prefix if the window cache still holds what follows from it
        size_t keep = 0;
        if (tp.cache_prompt) {
            keep = cached.get_common_prefix(prompt);
            if (keep >= prompt.size()) keep = prompt.size() - 1;               // re-read the last token for its logits
            while (keep > 0 && prompt[keep - 1] == LLAMA_TOKEN_NULL) --keep;   // never resume inside an image
            if (!eng.can_resume((int) keep)) keep = 0;                         // the window cache moved on: read again
        }
        eng.rollback((int) keep);
        cached.keep_first(keep);
        stats.n_prompt_cached = keep;
        // read the rest: text runs through decode(), images through mtmd + decode_embd()
        std::vector<float> lg, h;
        size_t i = keep;
        for (size_t p = 0; p < keep; ++p)   // images already in the cache: the size list continues after them
            if (prompt[p] == LLAMA_TOKEN_NULL && (p == 0 || prompt[p - 1] != LLAMA_TOKEN_NULL)) img_k++;
        while (i < prompt.size()) {
            // find_next_media_chunk(i) is the first image starting strictly after i: an image starting AT i (a cache resume
            // stepped back to an image's first position) is looked up by position
            std::pair<const mtmd::input_chunk_ptr *, size_t> next = { nullptr, 0 };
            if (prompt[i] == LLAMA_TOKEN_NULL) next = { &prompt.find_chunk(i), i };
            else next = prompt.find_next_media_chunk(i);
            const size_t text_end = next.first ? next.second : prompt.size();
            if (text_end > i) {
                std::vector<int32_t> t;   // get_tokens() asserts on a prompt with media; index instead
                for (size_t j = i; j < text_end; ++j) t.push_back(prompt[j]);
                err = eng.decode(t, &lg, mtp ? &h : nullptr);
                if (!err.empty()) { http_code = 500; return format_error_response("decode: " + err, ERROR_TYPE_SERVER); }
                i = text_end;
            }
            if (next.first) {
                const mtmd_input_chunk * ch = next.first->get();
                const int nt = (int) mtmd_input_chunk_get_n_tokens(ch);
                mtmd_context * enc = tok_ctx;
                const int side = img_k < img_side.size() ? img_side[img_k] : 0; img_k++;
                if (mm_gpu_always || (mm_auto && side >= A.mmproj_min_size && !long_ctx)) enc = need_gpu();
                if (!enc) { http_code = 500; return format_error_response("cannot load the mmproj on the card", ERROR_TYPE_SERVER); }
                if (enc == mctx_gpu.load()) {
                    mm_last_gpu = now_s();
                    eng.ensure_free((size_t) 256 << 20);   // a margin before the encode; slots give way, never the step
                }
                mm_last_use = now_s();
                if (mtmd_encode_chunk(enc, ch) != 0) { http_code = 500; return format_error_response("image encode failed", ERROR_TYPE_SERVER); }
                err = eng.decode_embd(mtmd_get_output_embd(enc), nt, mtmd_decode_use_non_causal(enc, ch), &lg, mtp ? &h : nullptr);
                if (!err.empty()) { http_code = 500; return format_error_response("image decode: " + err, ERROR_TYPE_SERVER); }
                i = next.second + nt;
            }
        }
        cached = std::move(prompt);
        stats.n_prompt_processed = n_prompt - keep;
        stats.update_prompt_last();

        // sample (llama-server: the prompt's text tokens enter the sampler's history first)
        common_sampler * smpl = common_sampler_init(model, tp.sampling);
        if (!smpl) { http_code = 400; return format_error_response("bad sampling parameters", ERROR_TYPE_INVALID_REQUEST); }
        for (size_t j = 0; j < cached.size(); ++j) if (cached[j] != LLAMA_TOKEN_NULL) common_sampler_accept(smpl, cached[j], false);
        std::string text;
        llama_tokens out;
        stop_type stop = STOP_TYPE_NONE;
        std::string stopping_word;
        const int n_predict = tp.n_predict < 0 ? INT32_MAX : tp.n_predict;
        auto emit = [&](llama_token t) -> bool {   // false = stop here
            out.push_back(t); stats.n_gen++;
            if (llama_vocab_is_eog(vocab, t)) { stop = STOP_TYPE_EOS; return false; }
            // Match llama-server: parser-required markers survive even when ordinary special tokens are hidden.
            const bool special = P.special || tp.sampling.preserved_tokens.count(t) != 0;
            text += common_token_to_piece(vocab, t, special);
            for (const auto & w : tp.antiprompt) {
                const size_t p = text.find(w, text.size() > w.size() + 16 ? text.size() - w.size() - 16 : 0);
                if (!w.empty() && p != std::string::npos) { text.resize(p); stop = STOP_TYPE_WORD; stopping_word = w; return false; }
            }
            if ((int) out.size() >= n_predict || eng.n_past() + 2 >= n_ctx) { stop = STOP_TYPE_LIMIT; return false; }
            return true;
        };
        llama_token tok = common_sampler_sample_logits(smpl, lg.data(), n_vocab);
        common_sampler_accept(smpl, tok, true);
        bool go = emit(tok);
        h_pending = h;
        std::vector<float> L, Hr, hh;
        while (go) {
            if (!mtp) {
                err = eng.decode({ tok }, &lg);
                if (!err.empty()) break;
                cached.push_back(tok);
                tok = common_sampler_sample_logits(smpl, lg.data(), n_vocab);
                common_sampler_accept(smpl, tok, true);
                go = emit(tok);
                continue;
            }
            // MTP: drafts at one position, one verify window, llama.cpp's sample-and-accept
            const int pos0 = eng.n_past();
            const int room = std::min(A.n_draft, std::min(n_predict - (int) out.size(), n_ctx - pos0 - 2));
            std::vector<int32_t> win = { tok };
            std::vector<float> hc = h_pending;
            for (int k = 0; k < room; ++k) {
                int32_t d = 0;
                if (!eng.draft(win.back(), hc.data(), pos0, &d, &hh).empty()) break;
                win.push_back(d); hc = hh;
            }
            err = eng.decode_all(win, &L, &Hr);
            if (!err.empty()) break;
            stats.n_draft_tokens += win.size() - 1; stats.n_draft_verif_steps++;
            int r = 0;
            for (;; ++r) {
                const llama_token t = common_sampler_sample_logits(smpl, L.data() + (size_t) r * n_vocab, n_vocab);
                common_sampler_accept(smpl, t, true);
                cached.push_back(win[r]);
                tok = t;
                go = emit(t);
                if (!go || r + 1 >= (int) win.size() || t != win[r + 1]) break;
                stats.n_draft_accepted++;
            }
            eng.rollback(pos0 + r + 1);
            h_pending.assign(Hr.begin() + (size_t) r * eng.hp().n_embd, Hr.begin() + (size_t) (r + 1) * eng.hp().n_embd);
        }
        common_sampler_free(smpl);
        if (!err.empty()) { http_code = 500; return format_error_response("generation: " + err, ERROR_TYPE_SERVER); }
        stats.update_gen_last();

        text = lite::utf8_clean(text);   // valid UTF-8 for the chat parser below (src/lite/utf8.hpp: it was a 500)
        server_task_result_cmpl_final res;
        res.content = text; res.tokens = out; res.stream = tp.stream; res.include_usage = tp.include_usage; res.stats = stats;
        res.truncated = false; res.n_decoded = (int32_t) out.size(); res.n_prompt_tokens = n_prompt;
        res.n_prompt_tokens_cache = (int32_t) keep; res.n_tokens_cached = eng.n_past(); res.has_new_line = false;
        res.stopping_word = stopping_word; res.stop = stop; res.post_sampling_probs = false;
        res.generation_params = tp; res.verbose = false; res.res_type = tp.res_type;
        res.oaicompat_model = tp.oaicompat_model; res.oaicompat_cmpl_id = tp.oaicompat_cmpl_id;
        task_result_state state(tp.chat_parser_params);
        res.update(state);
        http_code = 200;
        return res.to_json();
    };

    // ---------------------------------------------------------------- HTTP
    httplib::Server srv;
    const auto t_created = std::time(nullptr);
    auto send_json = [](httplib::Response & r, const json & j, int code) { r.status = code; r.set_content(safe_json_to_str(j), "application/json; charset=utf-8"); };
    srv.Get("/health", [&](const httplib::Request &, httplib::Response & r) { send_json(r, json{ { "status", "ok" } }, 200); });
    srv.Get("/", [&](const httplib::Request &, httplib::Response & r) { r.set_content(lite::chat_page(), "text/html; charset=utf-8"); });
    auto models = [&](const httplib::Request &, httplib::Response & r) {
        json meta = { { "vocab_type", llama_vocab_type(vocab) }, { "n_vocab", n_vocab }, { "n_ctx_train", eng.hp().n_ctx_train },
                      { "n_embd", eng.hp().n_embd }, { "n_ctx", eng.n_ctx() } };
        send_json(r, json{ { "object", "list" },
                           { "data", json::array({ json{ { "id", A.alias }, { "aliases", json::array({ A.alias }) }, { "object", "model" },
                                                         { "created", t_created }, { "owned_by", "llamacpp" }, { "meta", meta } } }) } }, 200);
    };
    srv.Get("/v1/models", models);
    srv.Get("/models", models);
    srv.Get("/props", [&](const httplib::Request &, httplib::Response & r) {
        task_params tparams; tparams.sampling = P.sampling;
        json props = {
            { "default_generation_settings", json{ { "params", tparams.to_json(true) }, { "n_ctx", eng.n_ctx() } } },
            { "total_slots", 1 }, { "model_alias", A.alias }, { "model_path", A.model },
            { "modalities", json{ { "vision", chat.allow_image }, { "video", false }, { "audio", false } } },
            { "media_marker", get_media_marker() }, { "endpoint_slots", true }, { "endpoint_props", false }, { "endpoint_metrics", true },
            { "chat_template", common_chat_templates_source(chat.tmpls.get(), "") },
            { "chat_template_caps", common_chat_templates_get_caps(chat.tmpls.get()) },
            { "build_info", std::string("strata-lite (gemma4 engine) / ") + llama_build_info() }, { "is_sleeping", false },
            { "strata", json{ { "slots", eng.slots() }, { "arena_shrinks", eng.stats().arena_shrinks }, { "n_cpu_moe", A.eo.n_cpu_moe },
                              { "prefill_ubatch", A.eo.prefill_ubatch < 0 ? json("auto") : json(A.eo.prefill_ubatch) }, { "last_chunk", eng.stats().last_chunk }, { "lend", A.eo.lend }, { "mtp", mtp }, { "vision", !A.mmproj.empty() }, { "projector_in_ram", mctx.load() != nullptr && !mm_gpu_always },
                              { "projector", A.mmproj.empty() ? "none" : A.mmproj_place }, { "projector_on_card", mctx_gpu != nullptr },
                              { "projector_moves", mm_moves.load() }, { "slots_full", slots_full },
                              { "projector_min_size", A.mmproj_min_size }, { "projector_gpu_max_ctx", A.mmproj_gpu_max_ctx },
                              { "projector_idle_s", A.mmproj_idle },
                              { "adapt", A.eo.adapt }, { "adapt_half_life", A.eo.adapt_half_life }, { "adapt_budget", A.eo.adapt_budget },
                              { "adapt_refill", A.eo.adapt_refill }, { "async_swaps", A.eo.async_swaps }, { "overlap", A.eo.overlap },
                              { "threads", A.threads }, { "pool_threads", A.eo.pool_threads }, { "n_draft", A.n_draft } } },
        };
        send_json(r, props, 200);
    });
    srv.Get("/slots", [&](const httplib::Request &, httplib::Response & r) {
        send_json(r, json::array({ json{ { "id", 0 }, { "n_ctx", eng.n_ctx() }, { "speculative", mtp }, { "is_processing", processing.load() } } }), 200);
    });
    auto completions_any = [&](const httplib::Request & req, httplib::Response & r, bool chat_api) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { send_json(r, json{ { "error", format_error_response("invalid JSON body", ERROR_TYPE_INVALID_REQUEST) } }, 400); return; }
        std::lock_guard<std::mutex> g(busy);
        processing = true;
        int code = 500;
        json out;
        try { out = generate(body, code, chat_api); }
        catch (const std::exception & e) { code = 500; out = format_error_response(e.what(), ERROR_TYPE_SERVER); }
        processing = false;
        if (code == 200 && out.is_array()) {   // stream: the whole answer as one SSE burst (no the reference server client streams)
            std::string sse;
            for (const auto & ev : out) sse += format_oai_sse(ev);
            sse += "data: [DONE]\n\n";
            r.status = 200; r.set_content(sse, "text/event-stream");
            return;
        }
        send_json(r, code == 200 ? out : json{ { "error", out } }, code);   // llama-server wraps errors as {"error": {...}}
    };
    auto completions = [&](const httplib::Request & req, httplib::Response & r) { completions_any(req, r, true); };
    auto completion = [&](const httplib::Request & req, httplib::Response & r) { completions_any(req, r, false); };
    srv.Post("/v1/chat/completions", completions);
    srv.Post("/chat/completions", completions);
    srv.Post("/completion", completion);
    srv.Post("/completions", completion);
    srv.set_read_timeout(7200); srv.set_write_timeout(7200);
    fprintf(stderr, "strata-lite-server: listening on http://%s:%d (alias %s, ctx %d, vision %d, mtp %d)\n",
            A.host.c_str(), A.port, A.alias.c_str(), eng.n_ctx(), (int) chat.allow_image, (int) mtp);
    if (mm_auto || mm_lazy) std::thread([&]() {   // after mmproj_idle seconds: off the card (auto, lazy), out of RAM (lazy)
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const int64_t now = now_s();
            const bool gpu_idle = mctx_gpu && now - (mm_lazy ? mm_last_use.load() : mm_last_gpu.load()) >= A.mmproj_idle;
            const bool ram_idle = mm_lazy && mctx && now - mm_last_use.load() >= A.mmproj_idle;
            if (!gpu_idle && !ram_idle) continue;
            std::lock_guard<std::mutex> g(busy);
            if (gpu_idle && mctx_gpu) {
                mtmd_free(mctx_gpu); mctx_gpu = nullptr;
                const int s0 = eng.slots();
                eng.grow_to(slots_full, (size_t) 256 << 20);
                fprintf(stderr, "strata-lite: projector off the card after %d s idle (%d -> %d slots per layer)\n", A.mmproj_idle, s0, eng.slots());
            }
            if (ram_idle && mctx) { mtmd_free(mctx); mctx = nullptr; fprintf(stderr, "strata-lite: projector out of RAM after %d s idle\n", A.mmproj_idle); }
        }
    }).detach();
    if (!srv.listen(A.host, A.port)) { fprintf(stderr, "strata-lite-server: cannot listen on %s:%d\n", A.host.c_str(), A.port); return 1; }
    if (mctx_gpu && mctx_gpu.load() != mctx.load()) mtmd_free(mctx_gpu);
    if (mctx) mtmd_free(mctx);
    llama_model_free(model);
    return 0;
}

}  // namespace g4srv

#ifdef STRATA_LITE_COMBINED
namespace q35srv { int run(int argc, char ** argv); }
int main(int argc, char ** argv) {
    if (const auto arch = g4srv::model_architecture(argc, argv); arch == "qwen35moe" || arch == "lfm2moe") return q35srv::run(argc, argv);
    return g4srv::run(argc, argv);
}
#else
int main(int argc, char ** argv) {
#ifndef _WIN32
    if (const auto arch = g4srv::model_architecture(argc, argv); arch == "qwen35moe" || arch == "lfm2moe") g4srv::exec_qwen_server(argc, argv);
#endif
    return g4srv::run(argc, argv);
}
#endif
