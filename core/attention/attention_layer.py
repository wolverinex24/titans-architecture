"""Implementation of the Attention Mechanism."""
# titans/core/attention/attention_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from . import attention_utils

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, Tq, D = query.shape
        _, Tk, _ = key.shape

        # Project and reshape
        q = self.q_proj(query).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Get output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, Tq, D)
        out = self.out_proj(out)
        
        return out
