"""
Automatically generated __init__.py for memory

This module exports the following:
- Classes: NeuralMemoryModule, PersistentMemory
- Functions: apply_forget_gate, compute_surprise_metric, create_causal_mask, forward, forward, get_memory_tokens, parallel_chunk_processing, retrieve, tensorize_gradient_descent, update_memory, update_momentum
"""

from typing import List, Optional, Tuple
from .persistent_memory import PersistentMemory, forward, get_memory_tokens, update_memory
from .memory_utils import apply_forget_gate, compute_surprise_metric, create_causal_mask, parallel_chunk_processing, tensorize_gradient_descent, update_momentum
from .neural_memory import NeuralMemoryModule, forward, retrieve

__all__ = [
    'NeuralMemoryModule',
    'PersistentMemory',
    'apply_forget_gate',
    'compute_surprise_metric',
    'create_causal_mask',
    'forward',
    'forward',
    'get_memory_tokens',
    'parallel_chunk_processing',
    'retrieve',
    'tensorize_gradient_descent',
    'update_memory',
    'update_momentum'
]
