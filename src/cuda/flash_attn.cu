#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// FlashAttention V1 Simplified Concept
// Attention(Q, K, V) = Softmax(Q K^T / sqrt(d)) V
// Key innovation: Tiling Q, K, V in shared memory to avoid HBM access for the large N^2 matrix.

#define BLOCK_SIZE 128

__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, float softmax_scale) {
    
    // Pointers and strides
    // Simple batch=1, head=1 case.
    // Q, K, V: [N, d]
    
    // Shared Memory for KV blocks
    // In real implementation, we need larger shared memory and dynamic sizing
    __shared__ float K_tile[BLOCK_SIZE][64]; // Fixed d=64 for demo
    __shared__ float V_tile[BLOCK_SIZE][64];
    
    int tx = threadIdx.x;
    int bx = blockIdx.x; // Block index for Q
    
    // Accumulators for Output and Normalization (LogSumExp)
    float l_i = -1e20f; // Max score
    float m_i = 0.0f;   // Sum exp
    float acc[64] = {0.0f}; // Output row accumulator
    
    int q_idx = bx * BLOCK_SIZE + tx;
    
    // Load Q into registers (assuming d=64 fits)
    float q_reg[64];
    if (q_idx < N) {
        for (int i = 0; i < 64; ++i) q_reg[i] = Q[q_idx * d + i];
    }
    
    // Loop over K, V blocks
    for (int j = 0; j < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++j) {
        
        // Load K, V tile to shared
        int kv_idx = j * BLOCK_SIZE + tx;
        if (kv_idx < N) {
            for (int i = 0; i < 64; ++i) {
                K_tile[tx][i] = K[kv_idx * d + i];
                V_tile[tx][i] = V[kv_idx * d + i];
            }
        } else {
             for (int i = 0; i < 64; ++i) {
                K_tile[tx][i] = 0.0f;
                V_tile[tx][i] = 0.0f;
             }
        }
        __syncthreads();
        
        // Compute Attention Scores S_ij = Q_i . K_j
        // Each thread handles one Query row vs all Keys in this tile.
        
        if (q_idx < N) {
            for (int t = 0; t < BLOCK_SIZE; ++t) {
                if (j * BLOCK_SIZE + t >= N) break;
                
                float score = 0.0f;
                for (int x = 0; x < 64; ++x) {
                    score += q_reg[x] * K_tile[t][x];
                }
                score *= softmax_scale;
                
                // Online Softmax (Safe Softmax)
                // m_new = max(m_old, score)
                // l_new = l_old * exp(m_old - m_new) + exp(score - m_new)
                
                float m_prev = l_i;
                l_i = fmaxf(l_i, score);
                float exp_score = expf(score - l_i);
                float exp_prev = expf(m_prev - l_i);
                
                m_i = m_i * exp_prev + exp_score;
                
                // Update Accumulator
                // O_new = O_old * exp_prev + V_t * exp_score
                for (int x = 0; x < 64; ++x) {
                    acc[x] = acc[x] * exp_prev + V_tile[t][x] * exp_score;
                }
            }
        }
        __syncthreads();
    }
    
    // Write Output
    // O_final = O_acc / m_i
    if (q_idx < N) {
        for (int x = 0; x < 64; ++x) {
            O[q_idx * d + x] = acc[x] / m_i;
        }
    }
}

torch::Tensor flash_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(q.size(-1) == 64, "Hidden dimension must be 64 for this highly optimized demo");
    
    int N = q.size(0);
    int d = q.size(1);
    float scale = 1.0f / sqrtf((float)d);
    
    auto o = torch::zeros_like(q);
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // size_t shared_mem = 2 * BLOCK_SIZE * d * sizeof(float); // Dynamic shared mem size calculation
    
    flash_attn_fwd_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        o.data_ptr<float>(),
        N, d, scale
    );
    
    return o;
}
