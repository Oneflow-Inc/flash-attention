// Copyright (c) 2022, Tri Dao.

#pragma once

#include "static_switch.h"
#include "fmha.h"
#include "fmha_dgrad_kernel_1xN_loop.h"


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Need_attn_mask, bool Need_attn_bias, int loop_steps=-1>
__global__ void fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel(FMHA_dgrad_params params) {
    fmha::compute_dq_dk_dv_1xN<Kernel_traits, Is_dropout, Is_causal, Need_attn_mask, Need_attn_bias, loop_steps>(params);
}

template<typename Kernel_traits>
void run_fmha_bwd_loop(const FMHA_dgrad_params &params, cudaStream_t stream) {
    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_dq = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;

    using Smem_tile_s = fmha::Smem_tile_mma_transposed<typename Kernel_traits::Cta_tile_p>;
    constexpr int smem_size_s = Smem_tile_s::BYTES_PER_TILE;
    static_assert(smem_size_s == 16 * Kernel_traits::Cta_tile_p::N * 2);
    static_assert(smem_size_dq == 16 * Kernel_traits::Cta_tile_p::K * 4 * Kernel_traits::Cta_tile_p::WARPS_N);

    constexpr int smem_size_dq_dk_dv = smem_size_q * 2 + smem_size_v * (Kernel_traits::V_IN_REGS ? 1 : 2) + smem_size_dq + smem_size_s * 2;
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    // printf("blocksize_c = %d, WARPS_N = %d, Smem size = %d\n", blocksize_c, Kernel_traits::Cta_tile_p::WARPS_N, smem_size_dq_dk_dv);

    bool is_dropout = params.p_dropout < 1.f;  // params.p_dropout is the probability of "keeping"

    bool has_attn_mask = !(params.attn_mask_ptr == nullptr);
    bool has_attn_bias = !(params.attn_bias_ptr == nullptr);

    if (has_attn_mask) {
        if (has_attn_bias) {
            BOOL_SWITCH(is_dropout, IsDropoutConst, [&] {
                auto kernel = params.is_causal
                    ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, true, true>
                    : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, true, true>;
                if (params.seqlen_k == blocksize_c) {
                    kernel = params.is_causal
                        ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, true, true, /*loop_steps=*/1>
                        : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, true, true, /*loop_steps=*/1>;
                } else if (params.seqlen_k == blocksize_c * 2) {
                    kernel = params.is_causal
                        ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, true, true, /*loop_steps=*/2>
                        : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, true, true, /*loop_steps=*/2>;
                }
                if( smem_size_dq_dk_dv >= 48 * 1024 ) {
                    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                dim3 grid(params.b, params.h);
                kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
                FMHA_CHECK_CUDA(cudaPeekAtLastError());
            });
        }else{
            BOOL_SWITCH(is_dropout, IsDropoutConst, [&] {
                auto kernel = params.is_causal
                    ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, true, false>
                    : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, true, false>;
                if (params.seqlen_k == blocksize_c) {
                    kernel = params.is_causal
                        ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, true, false, /*loop_steps=*/1>
                        : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, true, false, /*loop_steps=*/1>;
                } else if (params.seqlen_k == blocksize_c * 2) {
                    kernel = params.is_causal
                        ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, true, false, /*loop_steps=*/2>
                        : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, true, false, /*loop_steps=*/2>;
                }
                if( smem_size_dq_dk_dv >= 48 * 1024 ) {
                    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                dim3 grid(params.b, params.h);
                kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
                FMHA_CHECK_CUDA(cudaPeekAtLastError());
            });
        }
    }else{
        if (has_attn_bias) {
            BOOL_SWITCH(is_dropout, IsDropoutConst, [&] {
                auto kernel = params.is_causal
                    ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, false, true>
                    : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, false, true>;
                if (params.seqlen_k == blocksize_c) {
                    kernel = params.is_causal
                        ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, false, true, /*loop_steps=*/1>
                        : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, false, true, /*loop_steps=*/1>;
                } else if (params.seqlen_k == blocksize_c * 2) {
                    kernel = params.is_causal
                        ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, false, true, /*loop_steps=*/2>
                        : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, false, true, /*loop_steps=*/2>;
                }
                if( smem_size_dq_dk_dv >= 48 * 1024 ) {
                    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                dim3 grid(params.b, params.h);
                kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
                FMHA_CHECK_CUDA(cudaPeekAtLastError());
            });
        }else{
            BOOL_SWITCH(is_dropout, IsDropoutConst, [&] {
                auto kernel = params.is_causal
                    ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, false, false>
                    : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, false, false>;
                if (params.seqlen_k == blocksize_c) {
                    kernel = params.is_causal
                        ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, false, false, /*loop_steps=*/1>
                        : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, false, false, /*loop_steps=*/1>;
                } else if (params.seqlen_k == blocksize_c * 2) {
                    kernel = params.is_causal
                        ? &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, true, false, false, /*loop_steps=*/2>
                        : &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, false, false, false, /*loop_steps=*/2>;
                }
                if( smem_size_dq_dk_dv >= 48 * 1024 ) {
                    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                dim3 grid(params.b, params.h);
                kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
                FMHA_CHECK_CUDA(cudaPeekAtLastError());
            });
        }
    }
}
