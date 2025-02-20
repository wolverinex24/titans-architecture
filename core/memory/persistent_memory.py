"""Implementation of the Persistent Memory Component."""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from ..attention import attention_utils

class PersistentMemory(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        input_dim: int,
        memory_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        init_scale: float = 0.02
    ):
        """Persistent Memory component that stores task-specific knowledge.
        
        Args:
            num_tokens: Number of persistent memory tokens
            input_dim: Dimension of input features
            memory_dim: Dimension of memory state
            num_heads: Number of attention heads for memory
            dropout: Dropout probability
            init_scale: Initialization scale for memory tokens
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        # Initialize persistent memory tokens
        self.memory_tokens = nn.Parameter(
            torch.randn(num_tokens, memory_dim) * init_scale
        )
        
        # Projections for persistent memory
        self.query_proj = nn.Linear(input_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        
        self.output_proj = nn.Linear(memory_dim, input_dim)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(memory_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize using custom method for better stability
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters using scaled initialization."""
        def _scaled_init(tensor, scale=1.0):
            nn.init.normal_(tensor, mean=0.0, std=scale * math.sqrt(2.0 / (tensor.shape[0] + tensor.shape[1])))
            
        _scaled_init(self.memory_tokens)
        _scaled_init(self.query_proj.weight)
        _scaled_init(self.key_proj.weight)
        _scaled_init(self.value_proj.weight)
        _scaled_init(self.output_proj.weight)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass accessing persistent memory.
        
        Args:
            inputs: Input tensor [B, T, D]
            mask: Optional attention mask [B, T]
            
        Returns:
            outputs: Output after memory access [B, T, D]
            memory_tokens: Current memory tokens [N, M]
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Project queries from input
        queries = self.query_proj(inputs)
        
        # Get keys and values from memory tokens
        keys = self.key_proj(self.memory_tokens)
        values = self.value_proj(self.memory_tokens)
        
        # Reshape for multi-head attention
        queries = self._split_heads(queries)  # [B, H, T, M/H]
        keys = self._split_heads(keys)        # [B, H, N, M/H] 
        values = self._split_heads(values)    # [B, H, N, M/H]

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.memory_dim // self.num_heads)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Get attention weights and weighted sum of values
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        retrieved = torch.matmul(weights, values)  # [B, H, T, M/H]
        retrieved = self._merge_heads(retrieved)   # [B, T, M]
        
        # Project back to input dimension
        outputs = self.output_proj(retrieved)
        
        return outputs, self.memory_tokens
        
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Split tensor into multiple heads."""
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        return tensor.permute(0, 2, 1, 3)
        
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Merge multiple heads back into original shape."""
        batch_size, num_heads, seq_len, dim = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor.reshape(batch_size, seq_len, num_heads * dim)

    def get_memory_tokens(self) -> torch.Tensor:
        """Get the current state of memory tokens.
        
        Returns:
            memory_tokens: Current memory tokens [N, M]
        """
        return self.memory_tokens
        
    def update_memory(self, new_tokens: torch.Tensor):
        """Update persistent memory tokens (used during training).
        
        Args:
            new_tokens: New memory token values [N, M]
        """
        with torch.no_grad():
            self.memory_tokens.copy_(new_tokens)