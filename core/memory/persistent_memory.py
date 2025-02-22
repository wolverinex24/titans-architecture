"""Implementation of the Persistent Memory Component."""
# titans/core/memory/persistent_memory.py
import torch
import torch.nn as nn
import math
from typing import Optional

class PersistentMemory(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        init_scale: float = 0.02,
    ):
        """Persistent Memory component following Titans paper design.
        
        Args:
            num_tokens: Number of persistent memory tokens
            dim: Dimension of tokens (same as model dimension)
            init_scale: Initialization scale for memory tokens
        """
        super().__init__()
        
        # Initialize learnable persistent memory tokens
        self.memory_tokens = nn.Parameter(
            torch.randn(num_tokens, dim) * init_scale
        )
        
        # Initialize with scaled normal distribution
        self._init_parameters()

    def _init_parameters(self):
        """Initialize memory tokens using scaled initialization."""
        std = math.sqrt(2.0 / (self.memory_tokens.size(0) + self.memory_tokens.size(1)))
        nn.init.normal_(self.memory_tokens, mean=0.0, std=std)

    def forward(
        self,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Get persistent memory tokens.
        
        Args:
            batch_size: Optional batch size for expanding tokens
            
        Returns:
            memory_tokens: [batch_size, num_tokens, dim] if batch_size provided
                         else [num_tokens, dim]
        """
        if batch_size is not None:
            # Expand tokens for batch processing
            return self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return self.memory_tokens