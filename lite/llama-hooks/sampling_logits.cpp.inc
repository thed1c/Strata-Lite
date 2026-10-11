
// ---- Strata-Lite (lite/llama-hooks): appended at configure time by lite/CMakeLists.txt ----
llama_token common_sampler_sample_logits(struct common_sampler * gsmpl, const float * logits, int n_vocab, bool grammar_first) {
    const auto tm = gsmpl->tm();
    auto & grmr = gsmpl->grmr; auto & rbudget = gsmpl->rbudget; auto & chain = gsmpl->chain; auto & cur_p = gsmpl->cur_p;
    auto fill = [&]() {
        gsmpl->cur.resize(n_vocab);
        for (llama_token t = 0; t < n_vocab; t++) gsmpl->cur[t] = llama_token_data{ t, logits[t], 0.0f };
        cur_p = { gsmpl->cur.data(), gsmpl->cur.size(), -1, false };
    };
    fill();
    llama_sampler_apply(rbudget, &cur_p);
    if (grammar_first && grammar_should_apply(gsmpl)) llama_sampler_apply(grmr, &cur_p);
    llama_sampler_apply(chain, &cur_p);
    llama_token id = cur_p.data[cur_p.selected].id;
    if (grammar_first || !grammar_should_apply(gsmpl)) return id;
    {
        llama_token_data single = { id, 1.0f, 0.0f };
        llama_token_data_array arr = { &single, 1, -1, false };
        llama_sampler_apply(grmr, &arr);
        if (arr.data[0].logit != -INFINITY) return id;
    }
    fill();
    llama_sampler_apply(rbudget, &cur_p);
    if (grammar_should_apply(gsmpl)) llama_sampler_apply(grmr, &cur_p);
    llama_sampler_apply(chain, &cur_p);
    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");
    return cur_p.data[cur_p.selected].id;
}
