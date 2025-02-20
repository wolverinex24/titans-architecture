# titans/utils/optimization.py
import math
import torch
from typing import List, Optional

def get_optimizer_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay: Optional[List[str]] = None
) -> List[dict]:
    """Get parameter groups for optimizer with selective weight decay."""
    if no_decay is None:
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        }
    ]
    
    return optimizer_grouped_parameters

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a schedule with a learning rate that decreases following the cosine function."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)