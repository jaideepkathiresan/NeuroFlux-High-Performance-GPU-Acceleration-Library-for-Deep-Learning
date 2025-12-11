#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized LayerNorm using Warp Shuffle Reductions
// Much faster than standard reduction by avoiding shared memory for intra-warp comms
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void layernorm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int D, float eps) {
    
    // Assuming D <= 1024, one block per row
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= N) return;
    
    const float* row_x = x + bid * D;
    float* row_out = out + bid * D;
    
    // 1. Mean
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        sum += row_x[i];
    }
    
    // Block-wide reduction
    // First warp reduction
    sum = warpReduceSum(sum);
    
    // Shared memory for partial sums from warps
    __shared__ float shared_sum[32]; // Max 32 warps
    int lane = tid % warpSize;
    int warp = tid / warpSize;
    
    if (lane == 0) shared_sum[warp] = sum;
    __syncthreads();
    
    // Reduce shared sums
    if (warp == 0) {
        float block_sum = (tid < (blockDim.x / warpSize)) ? shared_sum[lane] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        if (tid == 0) shared_sum[0] = block_sum;
    }
    __syncthreads();
    
    float mean = shared_sum[0] / D;
    
    // 2. Variance
    float sum_sq_diff = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = row_x[i] - mean;
        sum_sq_diff += diff * diff;
    }
    
    sum_sq_diff = warpReduceSum(sum_sq_diff);
    
    if (lane == 0) shared_sum[warp] = sum_sq_diff;
    __syncthreads();
    
    if (warp == 0) {
        float block_sum = (tid < (blockDim.x / warpSize)) ? shared_sum[lane] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        if (tid == 0) shared_sum[0] = block_sum;
    }
    __syncthreads();
    
    float var = shared_sum[0] / D;
    float inv_std = rsqrtf(var + eps);
    
    // 3. Normalize and Scale/Shift
    for (int i = tid; i < D; i += blockDim.x) {
        row_out[i] = (row_x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    int N = input.size(0);
    int D = input.size(1);
    
    auto out = torch::empty_like(input);
    
    // 1 block per row, max 1024 threads
    int threads = (D > 1024) ? 1024 : D;
    // ensure threads is multiple of 32
    if (threads % 32 != 0) threads = ((threads + 31) / 32) * 32;
    
    layernorm_kernel<<<N, threads>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, D, eps
    );
    
    return out;
}
