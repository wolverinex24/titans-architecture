"""Custom optimizers and schedulers."""
# titans/training/optimizer.py
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any
from utils.optimization import get_optimizer_groups

def get_optimizer(
    model_params,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999
) -> AdamW:
    """Get AdamW optimizer with weight decay."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params": [p for n, p in model_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_params, lr=lr, betas=(beta1, beta2))

def get_scheduler(
    optimizer: AdamW,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None
) -> CosineAnnealingLR:
    """Get learning rate scheduler."""
    if num_warmup_steps is None:
        num_warmup_steps = num_training_steps // 10
        
    return CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=0
    )

