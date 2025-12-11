#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Direct 2D Convolution Kernel + Bias + ReLU Fusion
// Assumption: Single batch, Single channel for simplicity of this demo, 3x3 kernel
// This demonstrates the core concept of avoiding memory round-trips for creating fusion
__global__ void fused_conv_bias_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int H, int W,
    int H_out, int W_out,
    int kernel_size,
    int stride, int padding) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < H_out && col < W_out) {
        float val = 0.0f;

        // Convolution window
        // Start position in input
        int h_in_start = row * stride - padding;
        int w_in_start = col * stride - padding;

        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int h_in = h_in_start + i;
                int w_in = w_in_start + j;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    val += input[h_in * W + w_in] * weight[i * kernel_size + j];
                }
            }
        }

        // Fused Bias Add
        val += bias[0]; // Simplified broadcast for single channel example

        // Fused ReLU
        val = fmaxf(0.0f, val);

        output[row * W_out + col] = val;
    }
}

torch::Tensor fused_conv_bias_relu_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

    // Strict simplified checks for this demo kernel
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)"); // We will treat N=1, C=1 for kernel simplicity
    
    int H = input.size(2);
    int W = input.size(3);
    int kernel_size = weight.size(2);

    int H_out = ((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    int W_out = ((W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::zeros({input.size(0), input.size(1), H_out, W_out}, input.options());

    dim3 threads(16, 16);
    dim3 blocks((W_out + threads.x - 1) / threads.x, (H_out + threads.y - 1) / threads.y);

    fused_conv_bias_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W,
        H_out, W_out,
        kernel_size,
        stride, padding
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in fused_conv_bias_relu_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}
