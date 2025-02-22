# titans/utils/checkpoint.py
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import json
import os

logger = logging.getLogger(__name__)

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
        
        # Path to store checkpoint metadata
        self.meta_path = self.save_dir / "checkpoint_meta.json"
        
        # Load or initialize checkpoint list
        self.checkpoints = self._load_checkpoint_list()
        logger.info(f"Initialized checkpoint manager in {save_dir}")
        logger.info(f"Found {len(self.checkpoints)} existing checkpoints")
        
    def _load_checkpoint_list(self) -> List[Path]:
        """Load list of checkpoints from metadata file or scan directory."""
        checkpoints = []
        
        # Try to load from metadata file
        if self.meta_path.exists():
            try:
                with open(self.meta_path, 'r') as f:
                    checkpoint_paths = json.load(f)
                checkpoints = [Path(p) for p in checkpoint_paths if Path(p).exists()]
                logger.info(f"Loaded {len(checkpoints)} checkpoint paths from metadata")
            except Exception as e:
                logger.warning(f"Error loading checkpoint metadata: {e}")
        
        # If no metadata or loading failed, scan directory
        if not checkpoints:
            checkpoints = sorted(
                [p for p in self.save_dir.glob("checkpoint_*.pt") if p.is_file()],
                key=lambda p: int(p.stem.split('_')[1])
            )
            logger.info(f"Scanned directory and found {len(checkpoints)} checkpoints")
        
        # Update metadata file
        self._save_checkpoint_list(checkpoints)
        return checkpoints
    
    def _save_checkpoint_list(self, checkpoints: List[Path]):
        """Save checkpoint list to metadata file."""
        try:
            with open(self.meta_path, 'w') as f:
                json.dump([str(p) for p in checkpoints], f)
        except Exception as e:
            logger.warning(f"Error saving checkpoint metadata: {e}")
    
    def save(
        self,
        state_dict: Dict[str, Any],
        metrics: Dict[str, float],
        step: int
    ):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_{step}.pt"
        
        try:
            # Save checkpoint
            torch.save({
                'step': step,
                'state_dict': state_dict,
                'metrics': metrics
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Update checkpoint list
            self.checkpoints.append(checkpoint_path)
            
            # Remove old checkpoints if exceeding max_checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                try:
                    if old_checkpoint.exists():
                        old_checkpoint.unlink()
                        logger.info(f"Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"Error removing old checkpoint {old_checkpoint}: {e}")
            
            # Update metadata
            self._save_checkpoint_list(self.checkpoints)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
            
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint."""
        if not self.checkpoints:
            logger.info("No checkpoints found")
            return None
        
        latest_checkpoint = self.checkpoints[-1]
        try:
            checkpoint = torch.load(latest_checkpoint)
            logger.info(f"Loaded latest checkpoint: {latest_checkpoint}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint {latest_checkpoint}: {e}")
            return None
        
    def load_best(self, metric: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
        """Load best checkpoint according to metric."""
        if not self.checkpoints:
            return None
            
        best_checkpoint = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for checkpoint_path in self.checkpoints:
            try:
                checkpoint = torch.load(checkpoint_path)
                value = checkpoint['metrics'].get(metric)
                
                if value is None:
                    continue
                    
                if (mode == 'max' and value > best_value) or \
                   (mode == 'min' and value < best_value):
                    best_value = value
                    best_checkpoint = checkpoint
                    best_path = checkpoint_path
            except Exception as e:
                logger.warning(f"Error loading checkpoint {checkpoint_path}: {e}")
                continue
        
        if best_checkpoint:
            logger.info(f"Loaded best checkpoint ({metric}={best_value}): {best_path}")
        else:
            logger.info(f"No valid checkpoints found for metric {metric}")
            
        return best_checkpoint