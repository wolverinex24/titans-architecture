"""Implementation of the Neural Memory Module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from utils.metrics import MemoryMetrics

class NeuralMemoryModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        test_lr: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_layers = num_layers
        self.test_lr = test_lr
        
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
        self.forget_gate = nn.Linear(input_dim + memory_dim, memory_dim)
        self.momentum_gate = nn.Linear(input_dim + memory_dim, memory_dim)
        self.update_gate = nn.Linear(input_dim + memory_dim, memory_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Momentum parameters
        self.eta = nn.Parameter(torch.ones(1) * 0.9)  # η (momentum decay)
        self.theta = nn.Parameter(torch.ones(1) * 0.1)  # θ (learning rate)
        
        # # Register momentum buffer - single unified momentum state
        # self.register_buffer('momentum_state', None)  # Will be initialized in forward
        self.register_buffer('momentum_state', torch.zeros(2, memory_dim))
    
    def get_memory_metrics(self, memory_state: Optional[torch.Tensor]) -> Dict[str, float]:
        """Get memory state metrics for analysis.
        
        Args:
            memory_state: Current memory state tensor
                
        Returns:
            Dictionary containing memory metrics
        """
        if memory_state is None:
            return {
                'mean_activation': 0.0,
                'max_activation': 0.0,
                'sparsity': 1.0,
                'memory_usage': 0.0,
                'entropy': 0.0,
                'memory_size': 0,
                'momentum_norm': 0.0,
                'momentum_mean': 0.0
            }
        
        try:
            with torch.no_grad():
                # Ensure we're working with a valid tensor
                if not torch.isfinite(memory_state).all():
                    print("Memory state contains non-finite values")
                    return {
                        'mean_activation': float('nan'),
                        'max_activation': float('nan'),
                        'sparsity': float('nan'),
                        'memory_usage': float('nan'),
                        'entropy': float('nan'),
                        'memory_size': memory_state.numel(),
                        'momentum_norm': float('nan'),
                        'momentum_mean': float('nan')
                    }
                
                # Calculate memory metrics
                abs_memory = torch.abs(memory_state)
                mean_activation = abs_memory.mean().item()
                max_activation = abs_memory.max().item()
                sparsity = (abs_memory < 1e-6).float().mean().item()
                memory_usage = torch.norm(memory_state).item()
                
                # Entropy calculation
                abs_memory_flat = abs_memory.flatten() + 1e-10
                normalized = abs_memory_flat / abs_memory_flat.sum()
                entropy = -torch.sum(normalized * torch.log2(normalized)).item()
                
                # Momentum metrics
                momentum_norm = torch.norm(self.momentum_state).item()
                momentum_mean = self.momentum_state.mean().item()
                
                return {
                    'mean_activation': mean_activation,
                    'max_activation': max_activation,
                    'sparsity': sparsity,
                    'memory_usage': memory_usage,
                    'entropy': entropy,
                    'memory_size': memory_state.numel(),
                    'momentum_norm': momentum_norm,
                    'momentum_mean': momentum_mean
                }
        except Exception as e:
            print(f"Error computing memory metrics: {str(e)}")
            return {
                'mean_activation': 0.0,
                'max_activation': 0.0,
                'sparsity': 1.0,
                'memory_usage': 0.0,
                'entropy': 0.0,
                'memory_size': 0,
                'momentum_norm': 0.0,
                'momentum_mean': 0.0
            }

    def _compute_associative_loss(
        self, 
        memory_state: torch.Tensor,  # [B, M]
        keys: torch.Tensor,          # [B, T, M]
        values: torch.Tensor         # [B, T, M]
    ) -> torch.Tensor:
        """Compute associative memory loss."""
        memory_output = self._process_memory(memory_state)  # [B, M]
        memory_output = memory_output.unsqueeze(1)  # [B, 1, M]
        
        # Compute predictions with proper broadcasting
        pred_values = torch.bmm(
            keys,                            # [B, T, M]
            memory_output.transpose(-2, -1)  # [B, M, 1]
        )  # [B, T, 1]
        
        # Reshape predictions to match values
        pred_values = pred_values.squeeze(-1).unsqueeze(-1).expand_as(values)
        
        return 0.5 * torch.mean((pred_values - values) ** 2)

    def _compute_surprise(
        self,
        memory_state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute surprise and its gradient."""
        loss = self._compute_associative_loss(memory_state, keys, values)
        grad = torch.autograd.grad(loss, memory_state, create_graph=True)[0]
        return loss, grad

    def _process_memory(self, memory_state: torch.Tensor) -> torch.Tensor:
        """Process memory state through deep layers."""
        hidden = memory_state
        for layer, norm in zip(self.memory_layers, self.layer_norms):
            hidden = self.dropout(F.silu(norm(layer(hidden))))
        return hidden

    def _update_memory_state(
        self,
        current_memory: torch.Tensor,
        inputs: torch.Tensor,
        surprise_grad: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update memory state using momentum-based mechanism."""
        # Handle momentum state for different batch sizes
        if batch_size > self.momentum_state.size(0):
            # Expand momentum state for larger batches
            expanded_momentum = self.momentum_state.repeat((batch_size + 1) // 2, 1)[:batch_size]
            self.momentum_state = expanded_momentum
        elif batch_size < self.momentum_state.size(0):
            # Use subset of momentum state for smaller batches
            self.momentum_state = self.momentum_state[:batch_size]

        # Rest of your existing code remains the same
        inputs_pooled = inputs.mean(dim=1)
        gate_inputs = torch.cat([inputs_pooled, current_memory], dim=-1)
        
        momentum_gate = torch.sigmoid(self.momentum_gate(gate_inputs))
        forget_gate = torch.sigmoid(self.forget_gate(gate_inputs))
        update_gate = torch.sigmoid(self.update_gate(gate_inputs))

        new_momentum = (
            self.eta * self.momentum_state + 
            self.theta * surprise_grad
        )

        new_memory = (
            (1 - forget_gate) * current_memory + 
            update_gate * new_momentum
        )

        # Store just first 2 states for checkpoint compatibility
        self.momentum_state = new_momentum[:2].detach()

        return new_memory, new_momentum
    def forward(
        self,
        inputs: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        is_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle input dimensions
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        
        batch_size, seq_len, _ = inputs.shape
        
        # Initialize memory state if needed
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.memory_dim, device=inputs.device)
        
        # Project inputs
        keys = self.key_proj(inputs)      # [B, T, M]
        values = self.value_proj(inputs)  # [B, T, M]
        
        with torch.set_grad_enabled(True):
            memory_state.requires_grad_(True)
            
            # Compute surprise and gradients
            loss, surprise_grad = self._compute_surprise(memory_state, keys, values)
            
            # Update memory state with momentum mechanism
            new_memory, _ = self._update_memory_state(
                memory_state, 
                inputs, 
                surprise_grad,
                batch_size
            )
            
            # Process final memory state
            processed_memory = self._process_memory(new_memory)
            
            return processed_memory, new_memory.detach()

    def retrieve(
        self,
        queries: torch.Tensor,      # [B, T, D]
        memory_state: torch.Tensor  # [B, M]
    ) -> torch.Tensor:
        """Retrieve from memory using queries.
        
        Args:
            queries: Input queries of shape [batch_size, seq_len, input_dim]
            memory_state: Memory state of shape [batch_size, memory_dim]
        
        Returns:
            Retrieved memory of shape [batch_size, seq_len, memory_dim]
        """
        # Project queries
        queries = self.query_proj(queries)  # [B, T, M]
        
        # Process memory
        processed_memory = self._process_memory(memory_state)  # [B, M]
        
        # Attention computation with proper dimensions
        attn_scores = torch.matmul(
            queries,                         # [B, T, M]
            processed_memory.unsqueeze(2)    # [B, M, 1]
        )  # [B, T, 1]
        
        # Apply attention to memory
        retrieved = attn_scores * processed_memory.unsqueeze(1)  # [B, T, M]
        
        return retrieved