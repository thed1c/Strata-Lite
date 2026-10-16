# Pinned b6b003d LFM parser: auto tools must also allow a schema-constrained final answer.
# Generate a private source copy; never mutate the shared llama.cpp checkout.
set(_lfm_upstream "${LLAMA_DIR}/common/parsers/lfm2.cpp")
file(READ "${_lfm_upstream}" _lfm_source)
set(_lfm_anchor "        auto content = p.content(p.until(TOOL_CALL_START));")
string(FIND "${_lfm_source}" "${_lfm_anchor}" _lfm_at)
if(_lfm_at EQUAL -1)
  message(FATAL_ERROR "LFM JSON/tools hook does not match the pinned llama.cpp source")
endif()
set(_lfm_replacement [=[        // Auto tool choice permits a tool call OR the requested final response format.
        // A schema must not turn the lazy tool-only branch into a mandatory call.
        if (has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO) {
            auto response_format = p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema));
            return generation_prompt + reasoning + (tool_calls | response_format) + end;
        }
        auto content = p.content(p.until(TOOL_CALL_START));]=])
string(REPLACE "${_lfm_anchor}" "${_lfm_replacement}" _lfm_source "${_lfm_source}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/llama-hooks")
set(_lfm_generated "${CMAKE_CURRENT_BINARY_DIR}/llama-hooks/lfm2.cpp")
file(WRITE "${_lfm_generated}" "${_lfm_source}")
get_target_property(_lfm_sources llama-common SOURCES)
list(REMOVE_ITEM _lfm_sources "${_lfm_upstream}")
list(APPEND _lfm_sources "${_lfm_generated}")
set_property(TARGET llama-common PROPERTY SOURCES "${_lfm_sources}")
target_include_directories(llama-common PRIVATE "${LLAMA_DIR}/common/parsers")
