import torch
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroflux import NeuroFluxGEMM, NeuroFluxFusedConv

def benchmark_gemm(M=1024, N=1024, K=1024, iters=50):
    print(f"Benchmarking GEMM ({M}x{K} @ {K}x{N})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available. Skipping benchmark.")
        return

    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    
    # Warmup
    for _ in range(10):
        torch.matmul(a, b)

    # PyTorch Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iters * 1000 # ms

    # NeuroFlux Benchmark
    nf_gemm = NeuroFluxGEMM()
    
    # Check if compiled
    try:
        nf_gemm(a, b)
    except Exception as e:
        print(f"NeuroFlux GEMM failed (likely not compiled): {e}")
        return

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        nf_gemm(a, b)
    torch.cuda.synchronize()
    nf_time = (time.time() - start) / iters * 1000 # ms

    print(f"PyTorch: {torch_time:.4f} ms")
    print(f"NeuroFlux: {nf_time:.4f} ms")
    print(f"Difference: {torch_time/nf_time:.2f}x speedup/slowdown (Note: PyTorch uses highly tuned cuBLAS)")

def benchmark_fused_conv():
    print("\nBenchmarking Fused Conv+Bias+ReLU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        return

    # Dimensions
    N, C, H, W = 1, 64, 256, 256
    K = 64 # Out channels
    
    input = torch.randn(N, C, H, W, device=device)
    layer = NeuroFluxFusedConv(C, K, kernel_size=3, stride=1, padding=1).to(device)

    # Warmup
    layer(input)

    # Note: Comparing against Separated PyTorch Ops
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        # Native separated ops
        x = torch.nn.functional.conv2d(input, layer.weight, layer.bias, padding=1)
        x = torch.nn.functional.relu(x)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 50 * 1000

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        layer(input)
    torch.cuda.synchronize()
    nf_time = (time.time() - start) / 50 * 1000

    print(f"PyTorch (Separated): {torch_time:.4f} ms")
    print(f"NeuroFlux (Fused): {nf_time:.4f} ms")
    print(f"Speedup: {torch_time/nf_time:.2f}x")

if __name__ == "__main__":
    benchmark_gemm()
    benchmark_fused_conv()
