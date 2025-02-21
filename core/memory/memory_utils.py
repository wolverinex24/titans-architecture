"""Helper functions for memory operations."""
# titans/core/memory/memory_utils.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List

def parallel_chunk_processing(sequence: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Process sequence in parallel chunks.
    
    Args:
        sequence: Input tensor of shape [batch_size, seq_len, dim]
        chunk_size: Size of each chunk
        
    Returns:
        Chunked tensor of shape [batch_size, num_chunks, chunk_size, dim]
    """
    B, T, D = sequence.shape
    num_chunks = T // chunk_size + (1 if T % chunk_size > 0 else 0)
    # Pad sequence if needed
    if T % chunk_size > 0:
        pad_size = chunk_size - (T % chunk_size)
        sequence = F.pad(sequence, (0, 0, 0, pad_size))
    chunks = sequence.view(B, num_chunks, chunk_size, D)
    return chunks

def update_momentum(
    momentum: torch.Tensor,
    surprise: torch.Tensor,
    eta: float
) -> torch.Tensor:
    """
    Update momentum based on surprise.
    
    Args:
        momentum: Current momentum tensor
        surprise: Current surprise tensor
        eta: Momentum coefficient (between 0 and 1)
    
    Returns:
        Updated momentum tensor
    """
    return eta * momentum + (1 - eta) * surprise

def compute_surprise_metric(
    memory_output: torch.Tensor,
    target: torch.Tensor,
    prev_surprise: Optional[torch.Tensor] = None,
    eta: float = 0.9,
    theta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute surprise metric using memory output and target.
    
    Args:
        memory_output: Output from memory module
        target: Target tensor
        prev_surprise: Previous surprise value
        eta: Momentum coefficient
        theta: Learning rate for surprise update
        
    Returns:
        Tuple of (current_surprise, new_surprise_momentum)
    """
    # Compute current surprise using MSE
    current_surprise = 0.5 * ((memory_output - target) ** 2).mean()
    
    # Update surprise momentum
    if prev_surprise is None:
        new_surprise = current_surprise
    else:
        new_surprise = update_momentum(prev_surprise, current_surprise, eta)
        
    return current_surprise, new_surprise

def apply_forget_gate(
    state: torch.Tensor,
    forget_gate: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply forget gate to memory state.
    
    Args:
        state: Current memory state
        forget_gate: Forget gate values (pre-activation)
        eps: Small value for numerical stability
        
    Returns:
        Updated state after applying forget gate
    """
    gate = torch.sigmoid(forget_gate.clamp(-10, 10))
    return (1 - gate + eps) * state

def tensorize_gradient_descent(
    loss_values: List[torch.Tensor],
    learning_rates: List[float],
    chunk_size: int
) -> torch.Tensor:
    """
    Tensorize gradient descent for parallel processing within chunks.
    
    Args:
        loss_values: List of loss tensors for each step
        learning_rates: List of learning rates for each step
        chunk_size: Size of processing chunks
        
    Returns:
        Tensorized gradient updates
    """
    # Stack losses and learning rates
    losses = torch.stack(loss_values)
    lrs = torch.tensor(learning_rates, device=losses.device)
    
    # Reshape for parallel processing
    B = losses.size(1)  # batch size
    losses = losses.view(-1, chunk_size, B)
    lrs = lrs.view(-1, 1)
    
    # Compute gradients in parallel
    grads = torch.autograd.grad(
        losses.sum(), 
        [param for param in losses.requires_grad_()],
        create_graph=True
    )[0]
    
    return (lrs.unsqueeze(-1) * grads).sum(0)

def create_causal_mask(
    seq_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create causal mask for memory attention.
    
    Args:
        seq_length: Length of sequence
        device: Device to create mask on
        
    Returns:
        Causal attention mask
    """
    return torch.triu(
        torch.ones(seq_length, seq_length, device=device), 
        diagonal=1
    ).bool()