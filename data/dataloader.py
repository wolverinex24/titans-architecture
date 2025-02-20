"""Data loading utilities."""
# titans/data/dataloader.py
import torch
from torch.utils.data import DataLoader
from .dataset import TitansDataset
from typing import Dict,List

def create_dataloader(
    dataset: TitansDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader with appropriate settings for Titans."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch
    )

def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    return {
        key: torch.stack([sample[key] for sample in batch])
        for key in batch[0].keys()
    }