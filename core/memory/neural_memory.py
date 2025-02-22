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
        
        # Test-time learning states
        self.register_buffer('momentum', torch.zeros(1, memory_dim))
        self.register_buffer('prev_surprise', torch.zeros(1, memory_dim))
        
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
                'memory_size': 0
            }
        
        try:
            with torch.no_grad():
                # Handle if memory state is a tuple
                if isinstance(memory_state, tuple):
                    memory_state = memory_state[0]
                
                # Ensure we're working with a valid tensor
                if not torch.isfinite(memory_state).all():
                    print("Memory state contains non-finite values")
                    return {
                        'mean_activation': float('nan'),
                        'max_activation': float('nan'),
                        'sparsity': float('nan'),
                        'memory_usage': float('nan'),
                        'entropy': float('nan'),
                        'memory_size': memory_state.numel()
                    }
                
                # Calculate metrics on device
                abs_memory = torch.abs(memory_state)
                mean_activation = abs_memory.mean().item()
                max_activation = abs_memory.max().item()
                sparsity = (abs_memory < 1e-6).float().mean().item()
                memory_usage = torch.norm(memory_state).item()
                
                # Entropy calculation
                abs_memory = abs_memory.flatten() + 1e-10
                normalized = abs_memory / abs_memory.sum()
                entropy = -torch.sum(normalized * torch.log2(normalized)).item()
                
                return {
                    'mean_activation': mean_activation,
                    'max_activation': max_activation,
                    'sparsity': sparsity,
                    'memory_usage': memory_usage,
                    'entropy': entropy,
                    'memory_size': memory_state.numel()
                }
        except Exception as e:
            print(f"Error computing memory metrics: {str(e)}")
            return {
                'mean_activation': 0.0,
                'max_activation': 0.0,
                'sparsity': 1.0,
                'memory_usage': 0.0,
                'entropy': 0.0,
                'memory_size': 0
            }
        
    def _process_memory(self, memory_state: torch.Tensor) -> torch.Tensor:
        """Process memory state through memory layers."""
        hidden = memory_state
        for layer, norm in zip(self.memory_layers, self.layer_norms):
            hidden = self.dropout(F.silu(norm(layer(hidden))))
        return hidden
    
    def _compute_associative_loss(
        self, 
        memory_state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """Compute associative memory loss for test-time learning."""
        # Process memory state
        memory_output = self._process_memory(memory_state)
        
        # Compute predictions
        pred_values = torch.matmul(keys, memory_output.transpose(-2, -1))
        
        # Compute loss
        loss = 0.5 * torch.mean((pred_values - values) ** 2)
        return loss
        
    def _compute_surprise(
        self,
        memory_state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        compute_grad: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute surprise metric and optional gradients."""
        if compute_grad:
            # Compute loss and its gradients for surprise
            loss = self._compute_associative_loss(memory_state, keys, values)
            grad = torch.autograd.grad(loss, memory_state)[0]
            return loss, grad
        else:
            with torch.no_grad():
                loss = self._compute_associative_loss(memory_state, keys, values)
            return loss, None

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
        
        # Initialize states if needed
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.memory_dim, device=inputs.device)
            
        # Project inputs
        keys = self.key_proj(inputs)
        values = self.value_proj(inputs)
        
        if is_inference:
            # Enable gradients for test-time learning
            memory_state.requires_grad_(True)
            
            # Compute surprise with gradients
            loss, grad = self._compute_surprise(memory_state, keys, values, compute_grad=True)
            
            # Update momentum
            inputs_pooled = inputs.mean(dim=1)
            gate_inputs = torch.cat([inputs_pooled, memory_state], dim=-1)
            momentum_gate = torch.sigmoid(self.momentum_gate(gate_inputs))
            
            # Compute new surprise using both past and current
            new_surprise = momentum_gate * self.prev_surprise + (1 - momentum_gate) * grad
            
            # Update memory with surprise
            memory_state = memory_state - self.test_lr * new_surprise
            
            # Update forget gate
            forget_gate = torch.sigmoid(self.forget_gate(gate_inputs))
            memory_state = (1 - forget_gate) * memory_state
            
            # Store surprise for next iteration
            self.prev_surprise = new_surprise.detach()
            
        # Process final memory state
        processed_memory = self._process_memory(memory_state)
        
        return processed_memory, memory_state.detach()

    def retrieve(
        self,
        queries: torch.Tensor,
        memory_state: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve from memory using queries."""
        # Project queries
        queries = self.query_proj(queries)
        
        # Process memory
        processed_memory = self._process_memory(memory_state)
        
        # Compute attention and retrieve
        attn = torch.matmul(queries, processed_memory.transpose(-2, -1))
        retrieved = torch.matmul(attn, processed_memory)
        
        return retrieved