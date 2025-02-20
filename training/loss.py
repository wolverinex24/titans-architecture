"""Loss functions and metrics."""
# titans/training/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MemoryAwareLoss(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        
    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        prev_memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Main task loss
        task_loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Memory consistency loss if states are provided
        memory_loss = 0.0
        if memory_state is not None and prev_memory is not None:
            memory_loss = F.mse_loss(memory_state, prev_memory)
            
        return task_loss + self.alpha * memory_loss