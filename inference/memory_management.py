"""Test-time memory management."""
# titans/inference/memory_management.py
import torch
from typing import Optional, Dict
import numpy as np
from ..utils.metrics import MemoryMetrics
from ..core.memory import NeuralMemoryModule

class MemoryManager:
    def __init__(
        self,
        memory_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_memories: int = 10
    ):
        """Manage and cache memory states during inference.
        
        Args:
            memory_dim: Dimension of memory state
            device: Device to store memories on
            max_memories: Maximum number of memories to cache
        """
        self.memory_dim = memory_dim
        self.device = device
        self.max_memories = max_memories
        self.cached_memories = {}
        self.memory_usage = []
        
    def get_memory(
        self,
        key: str,
        default_init: bool = True
    ) -> Optional[torch.Tensor]:
        """Retrieve memory state for given key."""
        if key in self.cached_memories:
            return self.cached_memories[key]
            
        if default_init:
            return torch.zeros(1, self.memory_dim, device=self.device)
            
        return None
        
    def update_memory(
        self,
        key: str,
        memory_state: torch.Tensor,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Update cached memory state."""
        self.cached_memories[key] = memory_state.detach()
        
        # Track memory usage if metrics provided
        if metrics is not None:
            self.memory_usage.append(metrics)
        
        # Remove oldest memory if cache is full
        if len(self.cached_memories) > self.max_memories:
            oldest_key = next(iter(self.cached_memories))
            del self.cached_memories[oldest_key]
            
    def clear_memory(self, key: Optional[str] = None):
        """Clear specific or all cached memories."""
        if key is not None:
            self.cached_memories.pop(key, None)
        else:
            self.cached_memories.clear()
            self.memory_usage.clear()
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about memory usage."""
        if not self.memory_usage:
            return {}
            
        usage_array = np.array([list(m.values()) for m in self.memory_usage])
        return {
            'mean_activity': float(np.mean(usage_array[:, 0])),
            'mean_entropy': float(np.mean(usage_array[:, 1])),
            'max_activity': float(np.max(usage_array[:, 0])),
            'min_activity': float(np.min(usage_array[:, 0]))
        }
