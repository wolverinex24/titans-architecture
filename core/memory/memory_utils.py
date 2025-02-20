"""Helper functions for memory operations."""
# titans/core/memory/memory_utils.py
import torch
from typing import Optional, Tuple

def parallel_chunk_processing(sequence: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Process sequence in parallel chunks."""
    B, T, D = sequence.shape
    num_chunks = T // chunk_size + (1 if T % chunk_size > 0 else 0)
    chunks = sequence.view(B, num_chunks, -1, D)
    return chunks

def update_momentum(
    momentum: torch.Tensor,
    surprise: torch.Tensor,
    eta: float
) -> torch.Tensor:
    """Update momentum based on surprise."""
    return eta * momentum + (1 - eta) * surprise