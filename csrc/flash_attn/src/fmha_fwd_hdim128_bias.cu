// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "fmha_fwd_launch_template.cuh"

void run_fmha_fwd_hdim128_bias(Launch_params<FMHA_fprop_params> &launch_params, const bool configure) {
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
        using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, elem_type>;
        run_fmha_fwd_loop<Kernel_traits, false, true>(launch_params, configure);
    }));
}