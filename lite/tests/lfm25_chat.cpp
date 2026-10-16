// Exercise LFM's public template/parser/grammar contract, without loading model weights.
#include "chat.h"
#include "chat-peg-parser.h"
#include "llama-grammar.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include "json.h"

static bool accepts(const std::string & grammar, const std::string & text) {
    llama_grammar * g = llama_grammar_init_impl(nullptr, grammar.c_str(), "root", false, nullptr, 0, nullptr, 0);
    if (!g) throw std::runtime_error("invalid grammar");
    bool ok = true;
    for (char c : text) {
        try { llama_grammar_accept_token(*g, 0, std::string(1, c)); }
        catch (...) { ok = false; break; }
        if (llama_grammar_get_stacks(g).empty()) { ok = false; break; }
    }
    bool complete = false;
    if (ok) for (const auto & stack : llama_grammar_get_stacks(g)) complete |= stack.empty();
    llama_grammar_free_impl(g);
    return complete;
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    std::ifstream in(argv[1]);
    if (!in) return 2;
    const std::string src{std::istreambuf_iterator<char>(in), {}};
    auto tmpls = common_chat_templates_init(nullptr, src);
    common_chat_templates_inputs inputs;
    inputs.messages = {{"user", "Return the recorded gate and key as JSON."}};
    inputs.tools = {{"read_scene_state", "Read scene state", R"({"type":"object","properties":{"scene_id":{"type":"string"}},"required":["scene_id"],"additionalProperties":false})"}};
    inputs.json_schema = R"({"type":"object","properties":{"gate":{"type":"string"},"key":{"type":"string"}},"required":["gate","key"],"additionalProperties":false})";
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const std::string answer = R"({"gate":"locked","key":"absent"})";
    const std::string call = "<|tool_call_start|>[read_scene_state(scene_id=\"S12\")]<|tool_call_end|>";
    int failed = 0;
    for (auto choice : {COMMON_CHAT_TOOL_CHOICE_AUTO, COMMON_CHAT_TOOL_CHOICE_REQUIRED, COMMON_CHAT_TOOL_CHOICE_NONE}) {
        inputs.tool_choice = choice;
        auto p = common_chat_templates_apply(tmpls.get(), inputs);
        std::string prefix = p.generation_prompt;
        if (prefix.size() >= 7 && prefix.substr(prefix.size()-7) == "<think>") prefix += "</think>";
        const bool json_ok = accepts(p.grammar, prefix + answer);
        const bool call_ok = accepts(p.grammar, prefix + call);
        const bool bad_json_ok = accepts(p.grammar, prefix + "{\"unexpected\":5}");
        const bool expected_json = choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        const bool expected_call = choice != COMMON_CHAT_TOOL_CHOICE_NONE;
        common_peg_arena arena;
        arena.load(p.parser);
        common_chat_parser_params parser_params(p);
        // Parser extraction and grammar acceptance must agree about the same reply.
        const std::string close_think = p.generation_prompt.size() >= 7 &&
            p.generation_prompt.substr(p.generation_prompt.size()-7) == "<think>" ? "</think>" : "";
        bool extraction_ok = true;
        if (expected_json && json_ok) {
            const auto msg = common_chat_peg_parse(arena, close_think + answer, false, parser_params);
            extraction_ok &= msg.content == answer && msg.tool_calls.empty();
        }
        if (expected_call && call_ok) {
            const auto msg = common_chat_peg_parse(arena, close_think + call, false, parser_params);
            extraction_ok &= msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "read_scene_state" &&
                common_json::parse(msg.tool_calls[0].arguments) == common_json::parse(R"({"scene_id":"S12"})");
        }
        const bool ok = json_ok == expected_json && call_ok == expected_call && !bad_json_ok && extraction_ok;
        failed += !ok;
        std::cout << "choice=" << choice << " json=" << json_ok << " call=" << call_ok << " invalid_json=" << bad_json_ok << " pass=" << ok << '\n';
    }
    return failed ? 1 : 0;
}
