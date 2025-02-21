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
        """Neural Memory Module with momentum-based updates and forget mechanism.
        
        Args:
            input_dim: Dimension of input features
            memory_dim: Dimension of memory state
            num_layers: Number of layers in memory MLP
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_layers = num_layers
        
        # Input projections for keys and values
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
        
        # Parameters for memory update
        self.forget_gate = nn.Linear(input_dim + memory_dim, memory_dim)
        self.momentum_gate = nn.Linear(input_dim + memory_dim, memory_dim)
        self.update_gate = nn.Linear(input_dim + memory_dim, memory_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize momentum state
        self.register_buffer('momentum', torch.zeros(1, memory_dim))

    def _compute_surprise(
        self,
        memory_state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """Compute surprise score based on current memory and input.
        
        Args:
            memory_state: Current memory state [B, M]
            keys: Input keys [B, T, M] 
            values: Input values [B, T, M]
            
        Returns:
            surprise_score: Surprise scores [B, T, M]
        """
        # Forward pass through memory layers
        hidden = memory_state
        for layer, norm in zip(self.memory_layers, self.layer_norms):
            hidden = self.dropout(F.silu(norm(layer(hidden))))
        
        # Compute memory prediction
        pred_values = torch.bmm(keys, hidden.unsqueeze(-1)).squeeze(-1)
        
        # Compute surprise as prediction error
        surprise = values - pred_values
        return surprise

    def forward(
        self,
        inputs: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        momentum_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass updating memory state with new inputs.
        
        Args:
            inputs: Input tensor [B, T, D]
            memory_state: Previous memory state [B, M]
            momentum_state: Previous momentum state [B, M]
            
        Returns:
            new_memory: Updated memory state
            new_momentum: Updated momentum state
        """
        batch_size = inputs.shape[0]
        
        # Initialize states if None
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.memory_dim, device=inputs.device)
        if momentum_state is None:
            momentum_state = self.momentum.expand(batch_size, -1)
            
        # Project inputs to keys, values
        keys = self.key_proj(inputs)
        values = self.value_proj(inputs)
        
        # Compute gates
        gate_inputs = torch.cat([inputs.mean(dim=1), memory_state], dim=-1)
        forget_gate = torch.sigmoid(self.forget_gate(gate_inputs))
        momentum_gate = torch.sigmoid(self.momentum_gate(gate_inputs))
        update_gate = torch.sigmoid(self.update_gate(gate_inputs))
        
        # Compute surprise and update momentum
        surprise = self._compute_surprise(memory_state, keys, values)
        new_momentum = momentum_gate * momentum_state + (1 - momentum_gate) * surprise.mean(dim=1)
        
        # Update memory state
        memory_state = forget_gate * memory_state + update_gate * (
            memory_state + new_momentum
        )
        
        return memory_state, new_momentum

    def retrieve(
        self,
        queries: torch.Tensor,
        memory_state: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve information from memory given queries.
        
        Args:
            queries: Query tensor [B, T, D]
            memory_state: Current memory state [B, M]
            
        Returns:
            retrieved: Retrieved memory values [B, T, M]
        """
        # Project queries
        queries = self.query_proj(queries)
        
        # Forward pass through memory layers
        hidden = memory_state
        for layer, norm in zip(self.memory_layers, self.layer_norms):
            hidden = self.dropout(F.silu(norm(layer(hidden))))
            
        # Compute attention weights and retrieve
        attn = torch.bmm(queries, hidden.unsqueeze(-1)).squeeze(-1)
        retrieved = torch.bmm(attn.unsqueeze(1), hidden.unsqueeze(1))
        
        return retrieved