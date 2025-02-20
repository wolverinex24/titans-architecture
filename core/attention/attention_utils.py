"""Utilities for attention computations."""
# titans/core/attention/attention_utils.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

def create_causal_mask(
    seq_length: int,
    device: torch.device = None
) -> torch.Tensor:
    """Create causal attention mask.
    
    Args:
        seq_length: Length of sequence
        device: Device to create mask on
        
    Returns:
        Causal attention mask [seq_length, seq_length]
    """
    mask = torch.triu(
        torch.ones(seq_length, seq_length, device=device),
        diagonal=1
    )
    return mask == 0

def create_memory_attention_mask(
    seq_length: int,
    memory_length: int,
    persistent_length: int,
    device: torch.device = None
) -> torch.Tensor:
    """Create attention mask for sequence with memory tokens.
    
    Args:
        seq_length: Length of main sequence
        memory_length: Length of memory tokens
        persistent_length: Length of persistent memory tokens
        device: Device to create mask on
        
    Returns:
        Attention mask [(memory_length + persistent_length + seq_length), 
                       (memory_length + persistent_length + seq_length)]
    """
    total_length = persistent_length + memory_length + seq_length
    
    # Initialize mask allowing attention to persistent memory
    mask = torch.zeros(total_length, total_length, device=device)
    
    # Allow attention to persistent memory from everywhere
    mask[:, :persistent_length] = 1
    
    # Allow attention to memory tokens from sequence
    mask[persistent_length + memory_length:, 
         persistent_length:persistent_length + memory_length] = 1
    
    # Add causal mask for sequence portion
    seq_start = persistent_length + memory_length
    causal_mask = create_causal_mask(seq_length, device)
    mask[seq_start:, seq_start:] = causal_mask
    
    return mask

def create_chunked_attention_mask(
    seq_length: int,
    chunk_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """Create attention mask for chunked sequence processing.
    
    Args:
        seq_length: Total sequence length
        chunk_size: Size of each chunk
        device: Device to create mask on
        
    Returns:
        Chunked attention mask [seq_length, seq_length]
    """
    num_chunks = math.ceil(seq_length / chunk_size)
    mask = torch.zeros(seq_length, seq_length, device=device)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, seq_length)
        
        # Allow attention within chunk
        chunk_mask = create_causal_mask(end_idx - start_idx, device)
        mask[start_idx:end_idx, start_idx:end_idx] = chunk_mask
        
    return mask

def sliding_window_attention_mask(
    seq_length: int,
    window_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """Create attention mask for sliding window attention.
    
    Args:
        seq_length: Length of sequence
        window_size: Size of attention window
        device: Device to create mask on
        
    Returns:
        Sliding window attention mask [seq_length, seq_length]
    """
    mask = torch.zeros(seq_length, seq_length, device=device)
    
    for i in range(seq_length):
        start_idx = max(0, i - window_size + 1)
        mask[i, start_idx:i+1] = 1
        
    return mask

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    causal: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot product attention.
    
    Args:
        query: Query tensor [B, H, T, D]
        key: Key tensor [B, H, S, D]
        value: Value tensor [B, H, S, D]
        mask: Optional attention mask [S, S]
        dropout: Dropout probability
        scale: Optional scaling factor (if None, uses 1/sqrt(d_k))
        causal: Whether to apply causal masking
        
    Returns:
        output: Attention output [B, H, T, D]
        attention: Attention weights [B, H, T, S]
    """
    d_k = query.size(-1)
    scale = scale if scale is not None else 1.0 / math.sqrt(d_k)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    elif causal:
        seq_len = query.size(-2)
        causal_mask = create_causal_mask(seq_len, device=query.device)
        scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    # Compute attention weights
    attention = F.softmax(scores, dim=-1)
    if dropout > 0:
        attention = F.dropout(attention, p=dropout)
    
    # Apply attention to values
    output = torch.matmul(attention, value)
    
    return output, attention

def relative_position_encoding(
    seq_length: int,
    d_model: int,
    max_distance: int = 128,
    device: torch.device = None
) -> torch.Tensor:
    """Compute relative position encoding.
    
    Args:
        seq_length: Length of sequence
        d_model: Model dimension
        max_distance: Maximum relative distance to consider
        device: Device to create tensors on
        
    Returns:
        Relative position encodings [seq_length, seq_length, d_model]
    """
    positions = torch.arange(seq_length, device=device)
    rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
    
    # Clip relative positions
    rel_positions = torch.clamp(rel_positions, -max_distance, max_distance)
    
    # Compute position encodings
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * 
        (-math.log(10000.0) / d_model)
    )
    
    pe = torch.zeros(seq_length, seq_length, d_model, device=device)
    pe[:, :, 0::2] = torch.sin(rel_positions.unsqueeze(-1) * div_term)
    pe[:, :, 1::2] = torch.cos(rel_positions.unsqueeze(-1) * div_term)
    
    return pe

class AttentionAnalyzer:
    """Utility class for analyzing attention patterns."""
    
    def __init__(self):
        self.attention_maps = []
        
    def store_attention_map(
        self,
        attention: torch.Tensor,
        layer_idx: int
    ):
        """Store attention map for analysis."""
        self.attention_maps.append({
            'layer': layer_idx,
            'map': attention.detach().cpu()
        })
        
    def get_average_attention_distance(self) -> float:
        """Compute average attention distance across stored maps."""
        total_distance = 0
        count = 0
        
        for attn_data in self.attention_maps:
            attn_map = attn_data['map']
            seq_len = attn_map.size(-1)
            
            # Create position indices
            pos_i = torch.arange(seq_len).unsqueeze(-1)
            pos_j = torch.arange(seq_len).unsqueeze(0)
            
            # Compute distances weighted by attention
            distances = torch.abs(pos_i - pos_j).float()
            weighted_distances = (attn_map * distances).sum(dim=(-2, -1))
            
            total_distance += weighted_distances.mean().item()
            count += 1
            
        return total_distance / count if count > 0 else 0.0
        
    def get_attention_sparsity(self) -> float:
        """Compute attention sparsity (fraction of near-zero attention weights)."""
        total_sparsity = 0
        count = 0
        
        for attn_data in self.attention_maps:
            attn_map = attn_data['map']
            sparsity = (attn_map < 0.01).float().mean().item()
            
            total_sparsity += sparsity
            count += 1
            
        return total_sparsity / count if count > 0 else 0.0
        
    def reset(self):
        """Clear stored attention maps."""
        self.attention_maps = []

def get_memory_attention_stats(
    neural_memory_attn: torch.Tensor,
    persistent_memory_attn: torch.Tensor
) -> dict:
    """Compute attention statistics for memory components.
    
    Args:
        neural_memory_attn: Attention weights for neural memory
        persistent_memory_attn: Attention weights for persistent memory
        
    Returns:
        Dictionary of attention statistics
    """
    stats = {
        'neural_memory': {
            'mean': neural_memory_attn.mean().item(),
            'max': neural_memory_attn.max().item(),
            'sparsity': (neural_memory_attn < 0.01).float().mean().item()
        },
        'persistent_memory': {
            'mean': persistent_memory_attn.mean().item(),
            'max': persistent_memory_attn.max().item(),
            'sparsity': (persistent_memory_attn < 0.01).float().mean().item()
        }
    }
    return stats