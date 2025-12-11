#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

// CUDA Kernel: Tiled SGEMM
// C = A * B
// A: [M, K], B: [K, N], C: [M, N]
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    // Block row and column
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread row and column within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and Column of the global result matrix C to work on
    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Cvalue = 0.0f;

    // Loop over the tiles of A and B required to compute the block of C
    for (int p = 0; p < (K + TILE_SIZE - 1) / TILE_SIZE; ++p) {
        
        // Shared memory for tiles
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // Load A tile into shared memory
        if (Row < M && (p * TILE_SIZE + tx) < K) {
             As[ty][tx] = A[Row * K + p * TILE_SIZE + tx];
        } else {
             As[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        if ((p * TILE_SIZE + ty) < K && Col < N) {
             Bs[ty][tx] = B[(p * TILE_SIZE + ty) * N + Col];
        } else {
             Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the two tiles together and accumulate
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure calculations are done before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    if (Row < M && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

// C++ Wrapper Implementation
torch::Tensor gemm_cuda_forward(torch::Tensor a, torch::Tensor b) {
    // Check inputs
    TORCH_CHECK(a.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(a.dim() == 2, "A must be 2D");
    TORCH_CHECK(b.dim() == 2, "B must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "A and B dimensions must match for multiplication");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    auto c = torch::zeros({M, N}, a.options());

    // Grid and Block dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    gemm_tiled_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in gemm_tiled_kernel: %s\n", cudaGetErrorString(err));
    }

    return c;
}
