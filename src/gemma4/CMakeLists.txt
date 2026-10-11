# Strata-Lite gemma4 engine. Built against the experiment's llama.cpp copy (b6b003d), whose ggml (CUDA + CPU) is the
# oracle's: cmake -S src/gemma4 -B build-g4 -DLLAMA_DIR=<llama.cpp source> -DLLAMA_BUILD=<its build dir>
# Strata's CPU expert pool (src/kernels/cpu) is compiled in from this tree, against the same ggml-cpu.
cmake_minimum_required(VERSION 3.18)
project(strata_gemma4 C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(LLAMA_DIR "" CACHE PATH "llama.cpp source")
set(LLAMA_BUILD "" CACHE PATH "llama.cpp build dir")
set(STRATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)   # the copy stream (async expert swaps)

# Strata's CPU expert kernels and pool (the same files and ISA flags as strata_kernels_cpu)
set(SK ${STRATA_DIR}/src/kernels/cpu)
add_library(strata_cpu_pool STATIC ${SK}/pool.cpp ${SK}/expert.cpp ${SK}/expert_layout.cpp ${SK}/q2_avx2.cpp
            ${SK}/native_expert.cpp ${SK}/iq_avx512.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp ${SK}/kq_avx1.cpp)
target_include_directories(strata_cpu_pool PUBLIC ${STRATA_DIR}/include ${LLAMA_DIR}/ggml/include)
target_include_directories(strata_cpu_pool PRIVATE ${STRATA_DIR}/third_party/ggml)
target_compile_definitions(strata_cpu_pool PUBLIC STRATA_NATIVE_EXPERTS=1)
set(AVX512 "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(${SK}/expert.cpp ${SK}/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${AVX512}")
set_source_files_properties(${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(${SK}/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
set_property(SOURCE ${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_AVXVNNI=0)
target_link_directories(strata_cpu_pool PUBLIC ${LLAMA_BUILD}/bin)
target_link_libraries(strata_cpu_pool PUBLIC ggml-cpu ggml-base Threads::Threads)

add_library(g4_engine STATIC g4_engine.cpp)
target_compile_options(g4_engine PRIVATE -ffp-contract=off)   # the pool op's combine must round like ggml's mul/add
target_include_directories(g4_engine PUBLIC ${LLAMA_DIR}/ggml/include)
target_link_directories(g4_engine PUBLIC ${LLAMA_BUILD}/bin)
target_link_libraries(g4_engine PUBLIC strata_cpu_pool ggml ggml-base ggml-cpu CUDA::cudart)
add_executable(strata-gemma4 g4_cli.cpp)
target_link_libraries(strata-gemma4 PRIVATE g4_engine)
set_target_properties(strata-gemma4 PROPERTIES BUILD_RPATH ${LLAMA_BUILD}/bin)
add_executable(logit-cmp logit_cmp.cpp)

# strata-lite-server: llama-server's request/response code (server-context, static) + the engine
set(LB ${LLAMA_BUILD})
add_executable(strata-lite-server g4_server.cpp)
target_include_directories(strata-lite-server PRIVATE ${LLAMA_DIR}/tools/server ${LLAMA_DIR}/tools/mtmd ${LLAMA_DIR}/common
                           ${LLAMA_DIR}/include ${LLAMA_DIR}/vendor ${LLAMA_DIR}/vendor/cpp-httplib ${LLAMA_DIR})
target_link_libraries(strata-lite-server PRIVATE g4_engine ${LB}/tools/server/libserver-context.a ${LB}/vendor/cpp-httplib/libcpp-httplib.a
                      ${LB}/bin/libmtmd.so ${LB}/bin/libllama-common.so ${LB}/bin/libllama.so Threads::Threads)
set_target_properties(strata-lite-server PROPERTIES BUILD_RPATH ${LB}/bin)
