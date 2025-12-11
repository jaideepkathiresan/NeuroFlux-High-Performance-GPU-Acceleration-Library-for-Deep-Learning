import torch
import torch.nn as nn
from torch.autograd import Function

# Try to import the compiled extension
try:
    import neuroflux_cpp
except ImportError:
    print("\033[93mWarning: neuroflux_cpp extension not found. Please run `python setup.py install` to build the CUDA kernels.\033[0m")
    neuroflux_cpp = None

class NeuroFluxGEMM(nn.Module):
    """
    Bio-inspired optimized Matrix Multiplication layer.
    Uses tiles shared memory efficient kernel.
    """
    def __init__(self):
        super(NeuroFluxGEMM, self).__init__()

    def forward(self, a, b):
        if neuroflux_cpp:
            return neuroflux_cpp.gemm_forward(a, b)
        else:
            # Fallback for when extension isn't built yet (allows inspection of code)
            return torch.matmul(a, b)

class NeuroFluxFusedConv(nn.Module):
    """
    Fused Convolution + Bias + ReLU layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NeuroFluxFusedConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if neuroflux_cpp:
            return neuroflux_cpp.fused_conv_bias_relu_forward(
                x, self.weight, self.bias, self.stride, self.padding, 1
            )
        else:
            x = torch.nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
            return torch.nn.functional.relu(x)
