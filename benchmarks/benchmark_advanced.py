import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroflux.ops.advanced import FlashAttention, FusedLayerNorm
from neuroflux.engine.profiler import NeuroProfiler

def benchmark_advanced():
    print("\n[Benchmarks] Advanced Systems Kernels")
    profiler = NeuroProfiler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("Skipping - Needs CUDA")
        return

    # 1. FlashAttention Benchmark
    print("1. Benchmarking Attention Mechanism...")
    B, N, D = 16, 1024, 64 # Batch, SeqLen, HeadDim
    q = torch.randn(B*N, D, dtype=torch.float32, device=device) # Flattened for C++ demo
    k = torch.randn(B*N, D, dtype=torch.float32, device=device)
    v = torch.randn(B*N, D, dtype=torch.float32, device=device)
    
    model = FlashAttention(64, 1)
    
    # Warmup
    model(q, k, v)
    
    # Run
    with profiler.record("FlashAttention (Custom)"):
        for _ in range(50):
            model(q, k, v)
            
    # PyTorch Baseline (Naive)
    with profiler.record("PyTorch Attention (Naive)"):
        for _ in range(50):
            # Simple dot product
            attn = (q @ k.t()) * (D**-0.5)
            attn = attn.softmax(dim=-1)
            out = attn @ v

    # 2. LayerNorm Benchmark
    print("2. Benchmarking LayerNorm...")
    x = torch.randn(4096, 1024, device=device)
    ln_custom = FusedLayerNorm(1024).to(device)
    ln_torch = torch.nn.LayerNorm(1024).to(device)
    
    # Warmup
    ln_custom(x)
    
    with profiler.record("Shuffle LayerNorm (Custom)"):
        for _ in range(100):
            ln_custom(x)
            
    with profiler.record("PyTorch LayerNorm"):
        for _ in range(100):
            ln_torch(x)

    profiler.summary()

if __name__ == "__main__":
    benchmark_advanced()
