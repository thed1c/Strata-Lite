# Strata-Lite: one build for the Gemma 4 and Qwen3.6 engines and their server, on Linux and Windows.
#
#   cmake -S lite -B build-lite -DCMAKE_BUILD_TYPE=Release          (Windows: from a "x64 Native Tools" prompt, -G Ninja)
#   cmake --build build-lite --config Release
#
# It fetches llama.cpp at the commit every number in lite/ was checked against (b6b003d, as a zip: no git needed),
# appends Strata-Lite's one addition to it (lite/llama-hooks: llama.cpp's sampler run on a logits vector, which the
# server needs because the forward pass is the engine's, not a llama_context), and builds the parts of llama.cpp the
# server links - ggml with CUDA, llama, common, the server's request handling, mtmd for images - statically into one
# binary, strata-lite-server, which serves either model. ggml is linked statically on purpose: the CPU expert pool
# calls ggml-cpu's own dot products and SwiGLU so its rows round exactly as llama.cpp's CPU path does, and those are
# internal symbols a shared ggml does not export on Windows.
#
# Options:
#   -DSTRATA_LITE_LLAMA_DIR=<path>   use a llama.cpp b6b003d source tree already on disk (the setup's download)
#   -DSTRATA_LITE_PORTABLE=ON        a binary for other PCs: ggml-cpu at an AVX2 baseline instead of this CPU's features
#   -DCMAKE_CUDA_ARCHITECTURES=...   default 86-real;89-real;120a-real (RTX 30 / 40 / 50; 120a needs CUDA 12.8+)
#   -DSTRATA_LITE_TOOLS=ON           also the test drivers (strata-gemma4, strata-qwen35, logit-cmp) and route-oracle
cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)   # FetchContent_Populate of a declared zip (deprecated from CMake 3.30)
endif()
project(strata_lite C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(STRATA_LITE_LLAMA_COMMIT b6b003d2cb29647d968302eb2db8da6f66303b3e)
set(STRATA_LITE_LLAMA_DIR "" CACHE PATH "llama.cpp b6b003d source tree (empty: download it)")
option(STRATA_LITE_PORTABLE "ggml-cpu at an AVX2 baseline (binaries for other PCs)" OFF)
option(STRATA_LITE_TOOLS "also build the test drivers and route-oracle" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86-real;89-real;120a-real")
endif()
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(cfg Release RelWithDebInfo Debug MinSizeRel)
  string(TOUPPER ${cfg} CFG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

# ---- llama.cpp b6b003d
if(STRATA_LITE_LLAMA_DIR)
  set(LLAMA_DIR ${STRATA_LITE_LLAMA_DIR})
else()
  include(FetchContent)
  FetchContent_Declare(strata_lite_llamacpp
    URL https://github.com/ggml-org/llama.cpp/archive/${STRATA_LITE_LLAMA_COMMIT}.zip)
  FetchContent_GetProperties(strata_lite_llamacpp)
  if(NOT strata_lite_llamacpp_POPULATED)
    FetchContent_Populate(strata_lite_llamacpp)
  endif()
  set(LLAMA_DIR ${strata_lite_llamacpp_SOURCE_DIR})
endif()
if(NOT EXISTS ${LLAMA_DIR}/common/sampling.cpp)
  message(FATAL_ERROR "no llama.cpp source at ${LLAMA_DIR}")
endif()
# the hook, appended once (a tree that already has it - the patched checkout of lite/patches - is left as it is)
foreach(pair "sampling.cpp|sampling_logits.cpp.inc" "sampling.h|sampling_logits.h.inc")
  string(REPLACE "|" ";" pair "${pair}")
  list(GET pair 0 target)
  list(GET pair 1 hook)
  file(READ ${LLAMA_DIR}/common/${target} text)
  string(FIND "${text}" "common_sampler_sample_logits" found)
  if(found EQUAL -1)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/llama-hooks/${hook} add)
    file(APPEND ${LLAMA_DIR}/common/${target} "${add}")
    message(STATUS "strata-lite: added common_sampler_sample_logits to common/${target}")
  endif()
endforeach()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_FA_QUANTS "q8_0-q8_0;q4_0-q4_0;f16-f16" CACHE STRING "" FORCE)   # what q4 KV needs, and no more
if(STRATA_LITE_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  foreach(f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${f} ON CACHE BOOL "" FORCE)
  endforeach()
endif()
foreach(o LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_APP LLAMA_BUILD_UI LLAMA_USE_PREBUILT_UI LLAMA_OPENSSL LLAMA_TOOLS_INSTALL)
  set(${o} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(o LLAMA_BUILD_COMMON LLAMA_BUILD_TOOLS LLAMA_BUILD_SERVER)
  set(${o} ON CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
# the server code finds its precompiled header and includes from CMAKE_SOURCE_DIR (llama.cpp's root when it is built alone,
# this folder here): no precompiled header, and llama.cpp's root as an include folder
set_property(TARGET server-context PROPERTY PRECOMPILE_HEADERS "")
target_include_directories(server-context PRIVATE ${LLAMA_DIR})
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ---- Strata's CPU expert pool (src/kernels/cpu), with the same per-file ISA flags as Strata's own build
set(SK ${ROOT}/src/kernels/cpu)
add_library(strata_lite_pool STATIC ${SK}/pool.cpp ${SK}/expert.cpp ${SK}/expert_layout.cpp ${SK}/q2_avx2.cpp ${SK}/native_expert.cpp
            ${SK}/iq_avx512.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp ${SK}/kq_avx1.cpp ${SK}/q2_avx1.cpp ${SK}/iq_avx1.cpp)
target_include_directories(strata_lite_pool PUBLIC ${ROOT}/include)
target_include_directories(strata_lite_pool PRIVATE ${ROOT}/third_party/ggml)
target_compile_definitions(strata_lite_pool PUBLIC STRATA_NATIVE_EXPERTS=1)
if(MSVC)
  set_source_files_properties(${SK}/expert.cpp ${SK}/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(${SK}/kq_avx1.cpp ${SK}/q2_avx1.cpp ${SK}/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
else()
  set_source_files_properties(${SK}/expert.cpp ${SK}/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${SK}/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
  set_source_files_properties(${SK}/q2_avx1.cpp ${SK}/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
endif()
# the AVX-VNNI copies stay off: the engines' checks were made with the plain AVX2 rows
set_property(SOURCE ${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_AVXVNNI=0)
target_link_libraries(strata_lite_pool PUBLIC ggml-cpu ggml-base Threads::Threads)

# ---- the engines
if(MSVC)
  set(LITE_FLAGS /bigobj /utf-8)
else()
  set(LITE_FLAGS -ffp-contract=off)   # the pool op's combine must round like ggml's mul/add
endif()
foreach(e g4 q35)
  if(e STREQUAL g4)
    set(src ${ROOT}/src/gemma4/g4_engine.cpp)
  else()
    set(src ${ROOT}/src/qwen35/q35_engine.cpp)
  endif()
  add_library(${e}_engine STATIC ${src})
  target_compile_options(${e}_engine PRIVATE ${LITE_FLAGS})
  target_include_directories(${e}_engine PUBLIC ${LLAMA_DIR}/ggml/include ${ROOT}/include)
  target_link_libraries(${e}_engine PUBLIC strata_lite_pool ggml ggml-base ggml-cpu ggml-cuda CUDA::cudart Threads::Threads)
endforeach()

# ---- the server: both engines in one binary; the GGUF's architecture picks one
set(SERVER_INC ${LLAMA_DIR}/tools/server ${LLAMA_DIR}/tools/mtmd ${LLAMA_DIR}/common ${LLAMA_DIR}/include ${LLAMA_DIR}/vendor
               ${LLAMA_DIR}/vendor/cpp-httplib ${LLAMA_DIR})
add_executable(strata-lite-server ${ROOT}/src/gemma4/g4_server.cpp ${ROOT}/src/qwen35/q35_server.cpp)
target_compile_definitions(strata-lite-server PRIVATE STRATA_LITE_COMBINED=1)
target_compile_options(strata-lite-server PRIVATE ${LITE_FLAGS})
target_include_directories(strata-lite-server PRIVATE ${SERVER_INC} ${ROOT}/src/gemma4 ${ROOT}/src/qwen35)
target_link_libraries(strata-lite-server PRIVATE g4_engine q35_engine server-context mtmd llama-common llama cpp-httplib Threads::Threads)

if(STRATA_LITE_TOOLS)
  add_executable(strata-gemma4 ${ROOT}/src/gemma4/g4_cli.cpp)
  target_link_libraries(strata-gemma4 PRIVATE g4_engine)
  add_executable(strata-qwen35 ${ROOT}/src/qwen35/q35_cli.cpp)
  target_link_libraries(strata-qwen35 PRIVATE q35_engine)
  add_executable(logit-cmp ${ROOT}/src/gemma4/logit_cmp.cpp)
  foreach(t route-oracle route-log)
    add_executable(${t} ${CMAKE_CURRENT_SOURCE_DIR}/tools/${t}/route-log.cpp)
    target_include_directories(${t} PRIVATE ${LLAMA_DIR}/common ${LLAMA_DIR}/include ${LLAMA_DIR}/vendor)
    target_link_libraries(${t} PRIVATE llama-common llama Threads::Threads)
  endforeach()
endif()
