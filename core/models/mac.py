"""Implementation of the Memory as Context (MAC) Architecture."""
# titans/core/models/mac.py
import torch
import torch.nn as nn
from ..memory import NeuralMemoryModule, PersistentMemory
from ..attention import MultiHeadAttention, attention_utils
from ...utils.config import ModelConfig
from typing import Optional, Tuple, Dict

class TitansMAC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_memory_tokens: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Memory modules
        self.neural_memory = NeuralMemoryModule(
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.persistent_memory = PersistentMemory(
            num_tokens=num_memory_tokens,
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Attention module
        self.attention = MultiHeadAttention(
            dim=input_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output layers
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        inputs: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get persistent memory output
        persistent_out, _ = self.persistent_memory(inputs, attention_mask)
        
        # Update and retrieve from neural memory
        memory_state, momentum = self.neural_memory(inputs, memory_state)
        memory_out = self.neural_memory.retrieve(inputs, memory_state)
        
        # Combine inputs with memory outputs
        combined = torch.cat([persistent_out, memory_out, inputs], dim=1)
        
        # Apply attention
        if attention_mask is not None:
            # Extend mask for concatenated sequence
            B, T = attention_mask.shape
            extended_mask = torch.ones(B, combined.shape[1], device=attention_mask.device)
            extended_mask[:, -T:] = attention_mask
            attention_mask = extended_mask
            
        output = self.attention(
            query=combined,
            key=combined,
            value=combined,
            mask=attention_mask
        )
        
        # Final processing
        output = self.norm(output)
        output = self.dropout(output)
        
        return output, memory_state