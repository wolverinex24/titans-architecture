# titans/core/models/mac.py
import logging
import torch
import torch.nn as nn
from ..memory import NeuralMemoryModule, PersistentMemory
from ..attention import MultiHeadAttention, attention_utils
from utils.config import ModelConfig
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class TitansMAC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_memory_tokens: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        vocab_size: int = 50000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_memory_tokens = num_memory_tokens
        
        # Add embedding layer
        self.embedding = nn.Embedding(vocab_size, input_dim)
        
        # Memory modules
        self.neural_memory = NeuralMemoryModule(
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.persistent_memory = PersistentMemory(
            num_tokens=num_memory_tokens,
            dim=input_dim
        )
        
        # Attention module
        self.attention = MultiHeadAttention(
            dim=input_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, vocab_size)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def create_attention_mask(
        self,
        batch_size: int,
        seq_length: int,
        attention_mask: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Create attention mask for the combined sequence."""
        # Convert input attention mask to boolean
        attention_mask = attention_mask.bool()
        
        # Create mask for persistent tokens (always attended to)
        persistent_mask = torch.ones(
            (batch_size, self.num_memory_tokens),
            device=device,
            dtype=torch.bool
        )
        
        # Combine masks for the full sequence
        full_mask = torch.cat([
            persistent_mask,      # [B, Np]
            attention_mask,       # [B, T]
            attention_mask        # [B, T]
        ], dim=1)  # [B, Np + 2T]
        
        # Create causal mask
        total_length = self.num_memory_tokens + 2 * seq_length
        causal_mask = torch.triu(
            torch.ones(total_length, total_length, device=device),
            diagonal=1
        ).bool()
        
        # Create final attention mask
        mask = torch.zeros(
            (batch_size, total_length, total_length),
            device=device,
            dtype=torch.bool
        )
        
        # Allow full attention to persistent tokens
        mask[:, :, :self.num_memory_tokens] = True
        
        # Apply causal masking to the rest
        mask[:, :, self.num_memory_tokens:] = ~causal_mask[None, :, self.num_memory_tokens:]
        
        # Combine with attention mask
        mask = mask & full_mask.unsqueeze(1)
        
        # Convert to float for attention computation
        return mask.float()
        
    def forward(
        self,
        inputs: torch.Tensor,  # [B, T] Long tensor of token IDs
        memory_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Titans MAC model.
        
        Args:
            inputs: Input token IDs [B, T]
            memory_state: Optional previous memory state 
            attention_mask: Optional attention mask
            is_inference: Whether in inference mode
        """
        batch_size, seq_length = inputs.size()
        
        # Convert token IDs to embeddings
        inputs_emb = self.embedding(inputs)  # [B, T, D]
        
        # Get persistent memory tokens
        persistent_tokens = self.persistent_memory(batch_size)  # [B, Np, D]
        
        # Initialize or handle memory state
        if memory_state is None:
            memory_state = torch.zeros(
                batch_size, 
                self.neural_memory.memory_dim,
                device=inputs.device
            )
        elif isinstance(memory_state, tuple):
            memory_state = memory_state[0]
        
        # Update neural memory and retrieve memory-based representations
        with torch.set_grad_enabled(True):  # Enable gradients for test-time learning
            memory_output, new_memory_state = self.neural_memory(
                inputs_emb,
                memory_state=memory_state,
                is_inference=is_inference
            )
            
            # Retrieve memory-conditioned representations
            memory_out = self.neural_memory.retrieve(inputs_emb, new_memory_state)
        
        # Combine all information streams
        combined = torch.cat([
            persistent_tokens,  # Task knowledge [B, Np, D]
            memory_out,        # Memory context [B, T, D]
            inputs_emb         # Current input [B, T, D]
        ], dim=1)
        
        # Create attention mask if needed
        mask = None
        if attention_mask is not None:
            mask = self.create_attention_mask(
                batch_size=batch_size,
                seq_length=seq_length,
                attention_mask=attention_mask,
                device=inputs.device
            )
        
        # Process through attention
        output = self.attention(
            query=combined,
            key=combined,
            value=combined,
            mask=mask
        )
        
        # Apply normalization and regularization
        output = self.norm(output)
        output = self.dropout(output)
        
        # Extract relevant output (removing persistent tokens)
        output = output[:, self.num_memory_tokens:]  # [B, 2T, D]
        
        # Project to vocabulary
        logits = self.output_proj(output)  # [B, 2T, vocab_size]
        
        # Keep only the relevant predictions
        logits = logits[:, :seq_length, :]  # [B, T, vocab_size]
        
        return logits, new_memory_state