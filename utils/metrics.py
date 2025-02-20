"""Evaluation metrics."""
# titans/utils/metrics.py
import torch
import numpy as np
from typing import Dict, Any, Optional
from collections import defaultdict

class MetricsTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset metric states."""
        self.metrics = defaultdict(list)
        
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
            
    def compute_average(self) -> Dict[str, float]:
        """Compute average of tracked metrics."""
        return {
            key: np.mean(values) for key, values in self.metrics.items()
        }

def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from loss."""
    return torch.exp(loss)

def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """Compute accuracy ignoring padding tokens."""
    preds = logits.argmax(dim=-1)
    mask = labels != ignore_index
    correct = (preds == labels) & mask
    return correct.float().sum() / mask.float().sum()

class MemoryMetrics:
    @staticmethod
    def compute_memory_usage(
        memory_state: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """Compute memory usage statistics."""
        # Calculate activation sparsity
        active_cells = (torch.abs(memory_state) > threshold).float().mean().item()
        
        # Calculate memory entropy
        probs = torch.softmax(memory_state, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        
        return {
            "memory_activity": active_cells,
            "memory_entropy": entropy
        }