// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "fmha_bwd_launch_template.cuh"

void run_fmha_bwd_hdim32_bias(FMHA_dgrad_params &params, cudaStream_t stream) {
    FP16_SWITCH(params.is_bf16, ([&] {
        if (params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits, false, true>(params, stream);
        } else if (params.seqlen_k >= 256) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits, false, true>(params, stream);
        }
    }));
}