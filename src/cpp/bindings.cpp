#include <torch/extension.h>
#include "../include/kernels.h"

// Wrappers
torch::Tensor gemm_forward(torch::Tensor a, torch::Tensor b) {
    return gemm_cuda_forward(a, b);
}

torch::Tensor ops_block_1(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Concept: Execute a block of ops without Python overhead
    // Not fully implemented, just a concept
    return input; 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward", &gemm_cuda_forward, "NeuroFlux GEMM Forward");
    m.def("fused_conv_bias_relu_forward", &fused_conv_bias_relu_cuda_forward, "NeuroFlux Fused Conv");
    m.def("wmma_gemm_forward", &wmma_gemm_forward, "Tensor Core GEMM (FP16)");
    m.def("flash_attn_forward", &flash_attn_forward, "FlashAttention V1 (Simplified)");
    m.def("layernorm_forward", &layernorm_forward, "Optimized LayerNorm");
}
