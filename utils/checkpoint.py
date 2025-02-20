# titans/utils/checkpoint.py
import torch
from pathlib import Path
from typing import Optional, Dict, Any

class CheckpointManager:
    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 5
    ):
        """Manage model checkpoints.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        
    def save(
        self,
        state_dict: Dict[str, Any],
        metrics: Dict[str, float],
        step: int
    ):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_{step}.pt"
        
        # Save checkpoint
        torch.save({
            'step': step,
            'state_dict': state_dict,
            'metrics': metrics
        }, checkpoint_path)
        
        # Update checkpoint list
        self.checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints if exceeding max_checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_checkpoint.unlink()
            
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint."""
        if not self.checkpoints:
            return None
            
        latest_checkpoint = self.checkpoints[-1]
        return torch.load(latest_checkpoint)
        
    def load_best(self, metric: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
        """Load best checkpoint according to metric."""
        if not self.checkpoints:
            return None
            
        best_checkpoint = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for checkpoint_path in self.checkpoints:
            checkpoint = torch.load(checkpoint_path)
            value = checkpoint['metrics'].get(metric)
            
            if value is None:
                continue
                
            if (mode == 'max' and value > best_value) or \
               (mode == 'min' and value < best_value):
                best_value = value
                best_checkpoint = checkpoint
                
        return best_checkpoint