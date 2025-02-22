"""Training loop implementation."""
# titans/training/trainer.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
from core.models import TitansMAC
from utils.metrics import MetricsTracker
from utils.checkpoint import CheckpointManager
from utils.logging import setup_logger
from data import create_dataloader
from tqdm import tqdm

class TitansTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        max_grad_norm: float = 1.0
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs, memory_state = self.model(
                inputs=batch["input_ids"],
                attention_mask=batch.get("attention_mask")
            )
            
            loss = self.compute_loss(outputs, batch["labels"])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            total_loss += loss.item()
            
        return {"loss": total_loss / len(dataloader)}
        
    @staticmethod
    def compute_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        return nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

