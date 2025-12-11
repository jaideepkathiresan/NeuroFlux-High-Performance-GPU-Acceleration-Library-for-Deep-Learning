import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroflux import NeuroFluxGEMM, NeuroFluxFusedConv

class TestNeuroFlux(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            print("CUDA not available. Skipping tests.")
            raise unittest.SkipTest("CUDA not available")

    def test_gemm_correctness(self):
        M, N, K = 128, 128, 128
        device = torch.device('cuda')
        
        a = torch.randn(M, K, device=device)
        b = torch.randn(K, N, device=device)
        
        nf_gemm = NeuroFluxGEMM()
        
        expected = torch.matmul(a, b)
        result = nf_gemm(a, b)
        
        # Check correctness with tolerance
        self.assertTrue(torch.allclose(expected, result, atol=1e-4), "GEMM Output mismatch")

    def test_fused_conv_correctness(self):
        # N=1, C=1, H=16, W=16 for simple correctness check
        # Note: The custom kernel setup for this demo assumes specific simplified layouts
        N, C, H, W = 1, 1, 16, 16 
        device = torch.device('cuda')
        
        input = torch.randn(N, C, H, W, device=device)
        # 1 In-channel, 1 Out-channel, 3x3 kernel
        layer = NeuroFluxFusedConv(1, 1, kernel_size=3, padding=1).to(device)
        
        # PyTorch Reference
        ref_conv = torch.nn.functional.conv2d(input, layer.weight, layer.bias, padding=1)
        ref_out = torch.nn.functional.relu(ref_conv)
        
        # NeuroFlux
        nf_out = layer(input)
        
        self.assertTrue(torch.allclose(ref_out, nf_out, atol=1e-4), "Fused Conv Output mismatch")

if __name__ == '__main__':
    unittest.main()
