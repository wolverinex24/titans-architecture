"""Implementation of the Neural Memory Module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils.metrics import MemoryMetrics

class NeuralMemoryModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_layers = num_layers
        
        # Input projections
        self.key_proj = nn.Linear(input_dim, memory_dim)
        self.value_proj = nn.Linear(input_dim, memory_dim)
        self.query_proj = nn.Linear(input_dim, memory_dim)

        # Deep memory layers
        self.memory_layers = nn.ModuleList([
            nn.Linear(memory_dim, memory_dim) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(memory_dim) for _ in range(num_layers)
        ])
        
        # Gates
        gate_dim = input_dim + memory_dim
        self.forget_gate = nn.Linear(gate_dim, memory_dim)
        self.momentum_gate = nn.Linear(gate_dim, memory_dim)
        self.update_gate = nn.Linear(gate_dim, memory_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('momentum', torch.zeros(1, memory_dim))

    def _process_memory(self, memory_state: torch.Tensor) -> torch.Tensor:
        """Process memory state through memory layers."""
        hidden = memory_state
        for layer, norm in zip(self.memory_layers, self.layer_norms):
            hidden = self.dropout(F.silu(norm(layer(hidden))))
        return hidden

    def _compute_surprise(
        self,
        memory_state: torch.Tensor,  # [B, M]
        keys: torch.Tensor,          # [B, T, M]
        values: torch.Tensor         # [B, T, M]
    ) -> torch.Tensor:
        # Process memory state
        hidden = self._process_memory(memory_state)  # [B, M]
        
        # Expand hidden for batch matrix multiplication
        hidden_expanded = hidden.unsqueeze(1)  # [B, 1, M]
        
        # Compute predictions
        pred_values = torch.bmm(
            keys,                    # [B, T, M]
            hidden_expanded.transpose(-2, -1)  # [B, M, 1]
        )  # [B, T, 1]
        
        # Compute surprise
        surprise = values - pred_values.squeeze(-1).unsqueeze(-1)  # [B, T, M]
        return surprise

    def forward(
        self,
        inputs: torch.Tensor,        # [B, T, D] or [B, D]
        memory_state: Optional[torch.Tensor] = None,
        momentum_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle input dimensions
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, _ = inputs.shape
        
        # Initialize states
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.memory_dim, device=inputs.device)
        if momentum_state is None:
            momentum_state = self.momentum.expand(batch_size, -1)
            
        # Project inputs
        keys = self.key_proj(inputs)      # [B, T, M]
        values = self.value_proj(inputs)  # [B, T, M]
        
        # Compute gates with proper dimensions
        inputs_pooled = inputs.mean(dim=1)  # [B, D]
        gate_inputs = torch.cat([inputs_pooled, memory_state], dim=-1)  # [B, D+M]
        
        forget_gate = torch.sigmoid(self.forget_gate(gate_inputs))
        momentum_gate = torch.sigmoid(self.momentum_gate(gate_inputs))
        update_gate = torch.sigmoid(self.update_gate(gate_inputs))
        
        # Compute surprise and update momentum
        surprise = self._compute_surprise(memory_state, keys, values)  # [B, T, M]
        surprise_pooled = surprise.mean(dim=1)  # [B, M]
        
        new_momentum = momentum_gate * momentum_state + \
                      (1 - momentum_gate) * surprise_pooled
        
        # Update memory state
        memory_state = forget_gate * memory_state + \
                      update_gate * (memory_state + new_momentum)
        
        return memory_state, new_momentum

    def retrieve(
        self,
        queries: torch.Tensor,      # [B, T, D]
        memory_state: torch.Tensor  # [B, M]
    ) -> torch.Tensor:
        # Project queries
        queries = self.query_proj(queries)  # [B, T, M]
        
        # Process memory
        hidden = self._process_memory(memory_state)  # [B, M]
        
        # Compute attention and retrieve
        hidden_expanded = hidden.unsqueeze(1)  # [B, 1, M]
        attn = torch.bmm(queries, hidden_expanded.transpose(-2, -1))  # [B, T, 1]
        retrieved = torch.bmm(attn, hidden_expanded)  # [B, T, M]
        
        return retrieved