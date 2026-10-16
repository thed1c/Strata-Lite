# Strata-Lite qwen35moe engine. Built against the experiment's llama.cpp copy, whose ggml (CUDA + CPU) is the oracle's:
# cmake -S src/qwen35 -B build-q35 -DLLAMA_DIR=<llama.cpp source> -DLLAMA_BUILD=<its build dir>
# Strata's CPU expert pool (src/kernels/cpu) is compiled in from this tree, against the same ggml-cpu.
cmake_minimum_required(VERSION 3.18)
project(strata_qwen35 C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(LLAMA_DIR "" CACHE PATH "llama.cpp source")
set(LLAMA_BUILD "" CACHE PATH "llama.cpp build dir")
set(STRATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Strata's CPU expert kernels and pool (the same files and ISA flags as the gemma4 build)
set(SK ${STRATA_DIR}/src/kernels/cpu)
add_library(strata_cpu_pool_q35 STATIC ${SK}/pool.cpp ${SK}/expert.cpp ${SK}/expert_layout.cpp ${SK}/q2_avx2.cpp
            ${SK}/native_expert.cpp ${SK}/iq_avx512.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp ${SK}/kq_avx1.cpp ${SK}/q2_avx1.cpp ${SK}/iq_avx1.cpp)
target_include_directories(strata_cpu_pool_q35 PUBLIC ${STRATA_DIR}/include ${LLAMA_DIR}/ggml/include)
target_include_directories(strata_cpu_pool_q35 PRIVATE ${STRATA_DIR}/third_party/ggml)
target_compile_definitions(strata_cpu_pool_q35 PUBLIC STRATA_NATIVE_EXPERTS=1)
set(AVX512 "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(${SK}/expert.cpp ${SK}/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${AVX512}")
set_source_files_properties(${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(${SK}/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
set_source_files_properties(${SK}/q2_avx1.cpp ${SK}/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
set_property(SOURCE ${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_AVXVNNI=0)
target_link_directories(strata_cpu_pool_q35 PUBLIC ${LLAMA_BUILD}/bin)
target_link_libraries(strata_cpu_pool_q35 PUBLIC ggml-cpu ggml-base Threads::Threads)

add_library(q35_engine STATIC q35_engine.cpp ../lite/lfm_lora.cpp)
target_compile_options(q35_engine PRIVATE -ffp-contract=off)
target_include_directories(q35_engine PUBLIC ${LLAMA_DIR}/ggml/include ${STRATA_DIR}/include)
target_link_directories(q35_engine PUBLIC ${LLAMA_BUILD}/bin)
target_link_libraries(q35_engine PUBLIC strata_cpu_pool_q35 ggml ggml-base ggml-cpu CUDA::cudart Threads::Threads)
add_executable(strata-qwen35 q35_cli.cpp)
target_link_libraries(strata-qwen35 PRIVATE q35_engine)
set_target_properties(strata-qwen35 PROPERTIES BUILD_RPATH ${LLAMA_BUILD}/bin)

# strata-lite-server-qwen35: llama-server's request/response code (server-context, static) + the engine (g4_server.cpp's
# handling, q35_server.cpp). strata-lite-server (the gemma4 build) execs it for a qwen35moe GGUF.
set(LB ${LLAMA_BUILD})
add_executable(strata-lite-server-qwen35 q35_server.cpp)
target_include_directories(strata-lite-server-qwen35 PRIVATE ${LLAMA_DIR}/tools/server ${LLAMA_DIR}/tools/mtmd ${LLAMA_DIR}/common
                           ${LLAMA_DIR}/include ${LLAMA_DIR}/vendor ${LLAMA_DIR}/vendor/cpp-httplib ${LLAMA_DIR})
target_link_libraries(strata-lite-server-qwen35 PRIVATE q35_engine ${LB}/tools/server/libserver-context.a ${LB}/vendor/cpp-httplib/libcpp-httplib.a
                      ${LB}/bin/libmtmd.so ${LB}/bin/libllama-common.so ${LB}/bin/libllama.so Threads::Threads)
set_target_properties(strata-lite-server-qwen35 PROPERTIES BUILD_RPATH ${LB}/bin)
