"""Test-time memory management."""
# titans/inference/memory_management.py
from collections import defaultdict
import torch
from typing import Any, Optional, Dict, Tuple
import numpy as np
from utils.metrics import MemoryMetrics
from core.memory import NeuralMemoryModule

class MemoryManager:
    def __init__(
        self,
        memory_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_memories: int = 10,
        momentum_beta: float = 0.9,
        forget_threshold: float = 0.1
    ):
        """Enhanced Memory Manager for test-time learning.
        
        Args:
            memory_dim: Dimension of memory state
            device: Device to store memories on
            max_memories: Maximum number of memories to cache
            momentum_beta: Momentum coefficient for surprise tracking
            forget_threshold: Threshold for memory forgetting
        """
        self.memory_dim = memory_dim
        self.device = device
        self.max_memories = max_memories
        self.momentum_beta = momentum_beta
        self.forget_threshold = forget_threshold
        
        # Memory storage
        self.cached_memories = {}
        self.surprise_history = {}
        self.momentum_states = {}
        self.memory_usage = []
        
        # Analysis trackers
        self.memory_metrics = defaultdict(list)
        
    def _compute_memory_importance(
        self,
        memory_state: torch.Tensor,
        surprise: torch.Tensor
    ) -> float:
        """Compute importance score for memory state."""
        # Consider both memory activation and surprise
        activation = torch.norm(memory_state, p=2)
        surprise_magnitude = torch.norm(surprise, p=2)
        return float(activation * surprise_magnitude)
    
    def get_memory(
        self,
        key: str,
        default_init: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get memory state and its momentum."""
        if key in self.cached_memories:
            memory_state = self.cached_memories[key]
            momentum = self.momentum_states.get(key, 
                torch.zeros_like(memory_state))
            return memory_state, momentum
            
        if default_init:
            memory_state = torch.zeros(1, self.memory_dim, device=self.device)
            momentum = torch.zeros_like(memory_state)
            return memory_state, momentum
            
        return None, None
        
    def update_memory(
        self,
        key: str,
        memory_state: torch.Tensor,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Update memory state with test-time learning stats."""
        # Store the memory state
        self.cached_memories[key] = memory_state.detach()
        
        # Update metrics if provided
        if metrics is not None:
            self.memory_metrics[key].append(metrics)
            self.memory_usage.append(metrics)
        
        # Memory management
        self._manage_memory_capacity()
        
    def _manage_memory_capacity(self):
        """Manage memory capacity with intelligent pruning."""
        if len(self.cached_memories) <= self.max_memories:
            return
            
        # Compute importance scores for all memories
        importance_scores = {}
        for key, memory in self.cached_memories.items():
            surprise = self.surprise_history.get(key, 
                torch.zeros_like(memory))
            importance_scores[key] = self._compute_memory_importance(
                memory, surprise)
        
        # Remove least important memories
        while len(self.cached_memories) > self.max_memories:
            least_important = min(importance_scores.items(), 
                key=lambda x: x[1])[0]
            self._remove_memory(least_important)
            
    def _remove_memory(self, key: str):
        """Remove memory and associated states."""
        self.cached_memories.pop(key, None)
        self.momentum_states.pop(key, None)
        self.surprise_history.pop(key, None)
        self.memory_metrics.pop(key, None)
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_usage:
            return {}
            
        stats = {
            'memory_count': len(self.cached_memories),
            'average_metrics': {},
            'memory_evolution': {},
            'surprise_stats': {}
        }
        
        # Process metrics for each memory key
        for key, metrics_list in self.memory_metrics.items():
            if not metrics_list:
                continue
                
            # Calculate average metrics
            avg_metrics = {
                k: sum(m[k] for m in metrics_list) / len(metrics_list)
                for k in metrics_list[0].keys()
            }
            stats['average_metrics'][key] = avg_metrics
            
            # Track metric evolution
            stats['memory_evolution'][key] = {
                k: [m[k] for m in metrics_list]
                for k in metrics_list[0].keys()
            }
            
            # Surprise statistics
            if key in self.surprise_history:
                surprise = self.surprise_history[key]
                stats['surprise_stats'][key] = {
                    'magnitude': float(torch.norm(surprise)),
                    'mean': float(surprise.mean()),
                    'std': float(surprise.std())
                }
        print(f" get memory status {stats}")
        return stats
        
    def export_memory_state(
        self,
        key: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Export full memory state for saving."""
        if key not in self.cached_memories:
            return None
            
        return {
            'memory_state': self.cached_memories[key],
            'momentum': self.momentum_states.get(key),
            'surprise': self.surprise_history.get(key),
            'metrics': self.memory_metrics.get(key, [])
        }
        
    def import_memory_state(
        self,
        key: str,
        state_dict: Dict[str, torch.Tensor]
    ):
        """Import full memory state."""
        self.cached_memories[key] = state_dict['memory_state']
        if 'momentum' in state_dict:
            self.momentum_states[key] = state_dict['momentum']
        if 'surprise' in state_dict:
            self.surprise_history[key] = state_dict['surprise']
        if 'metrics' in state_dict:
            self.memory_metrics[key] = state_dict['metrics']
