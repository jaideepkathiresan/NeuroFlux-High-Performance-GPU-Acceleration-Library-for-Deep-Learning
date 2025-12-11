#pragma once
#include <torch/extension.h>

// Existings
torch::Tensor gemm_cuda_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor fused_conv_bias_relu_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
);

// New Advanced Kernels
torch::Tensor wmma_gemm_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor flash_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);

// C++ Graph Executor Mockup
void graph_capture_start();
void graph_capture_end();
