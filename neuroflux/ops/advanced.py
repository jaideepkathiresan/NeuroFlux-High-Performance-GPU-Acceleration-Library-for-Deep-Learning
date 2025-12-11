import torch
import torch.nn as nn
try:
    import neuroflux_cpp
except ImportError:
    neuroflux_cpp = None

class FlashAttention(nn.Module):
    """
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim == 64, "NeuroFlux FlashAttn currently optimized only for head_dim=64"

    def forward(self, q, k, v):
        """
        Args:
            q, k, v: [Batch, SeqLen, EmbedDim] (Simplified for demo)
        """
        # Reshape to [Batch*Heads, SeqLen, HeadDim] not fully handled in C++ demo
        # The C++ kernel assumes [N, d] flat or [B, N, d] logic.
        # For this high-tech demo, we assume the user passes flattened QKV
        
        if neuroflux_cpp:
            return neuroflux_cpp.flash_attn_forward(q, k, v)
        else:
            # Fallback to standard attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            return attn @ v

class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if neuroflux_cpp:
            return neuroflux_cpp.layernorm_forward(x, self.gamma, self.beta, self.eps)
        else:
            return nn.functional.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
