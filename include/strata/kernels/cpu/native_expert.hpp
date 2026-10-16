// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized activation any native layer uses (2560 values as Q8_K: 10 x 292).
inline constexpr size_t kNativeActBytes = 4096;
/// Bytes for the widest down activation: LFM2.5 has 1536 values as Q8_0 (48 x 34 = 1632).
inline constexpr size_t kNativeHBytes = 2048;
/// Strata-Lite: the widest expert FFN a native layer may have (Qwen3.8 640, Gemma 4 704, LFM2.5 1536); the pool's buffers.
inline constexpr int kNativeMaxFF = 1536;

/// The expert's gated activation.
enum class NativeAct : int {
    SwiGLU = 0,   ///< silu(gate) * up (Qwen3.8)
    GeGLU  = 1,   ///< gelu(gate) * up exactly as ggml-cpu's ggml_vec_geglu_f32 (its fp16 GELU table; gemma4)
    SwiGLU_ggml = 2,   ///< silu(gate) * up exactly as ggml-cpu's ggml_vec_swiglu_f32, ggml's own dots (Strata-Lite qwen35moe)
};

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    NativeAct act = NativeAct::SwiGLU;
    bool separate_down = false;         ///< down rows live in their own tensor (gemma4's fused gate_up + down):
                                        ///< down_off = 0 and the jobs carry a separate down pointer
};

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err,
                NativeAct act = NativeAct::SwiGLU, bool separate_down = false);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens (GeGLU: gelu(...) * ...).
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).  `blob` is the expert blob, or its down rows when
/// f.separate_down (down_off is then 0).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace strata::kernels::cpu
