"""Dataset implementations."""
# titans/data/dataset.py
import torch
from torch.utils.data import Dataset
from .preprocessing import SequenceProcessor
from typing import Dict, List, Optional, Union
import numpy as np

class TitansDataset(Dataset):
    def __init__(
        self,
        sequences: List[torch.Tensor],
        sequence_length: int = 8192,  # 8K sequence length
        stride: Optional[int] = None,
        pad_token_id: int = 0
    ):
        """Dataset for Titans model training.
        
        Args:
            sequences: List of input sequences
            sequence_length: Maximum sequence length
            stride: Stride for sliding window (if None, uses sequence_length)
            pad_token_id: Token ID for padding
        """
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.pad_token_id = pad_token_id
        
        # Pre-compute sequence chunks
        self.chunks = self._prepare_chunks()
        
    def _prepare_chunks(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare sequence chunks for training."""
        chunks = []
        
        for sequence in self.sequences:
            # Skip sequences shorter than 2 tokens
            if len(sequence) < 2:
                continue
                
            # Calculate number of chunks
            seq_len = len(sequence)
            num_chunks = max(1, (seq_len - self.sequence_length) // self.stride + 1)
            
            for i in range(num_chunks):
                start_idx = i * self.stride
                end_idx = start_idx + self.sequence_length
                
                # Get chunk and ensure it's the right length
                chunk = sequence[start_idx:end_idx]
                if len(chunk) < self.sequence_length:
                    padding = torch.full(
                        (self.sequence_length - len(chunk),),
                        self.pad_token_id,
                        dtype=torch.long
                    )
                    chunk = torch.cat([chunk, padding])
                
                # Create inputs and labels (shifted by 1)
                chunks.append({
                    "input_ids": chunk[:-1].clone(),
                    "labels": chunk[1:].clone(),
                    "attention_mask": (chunk != self.pad_token_id).float()[:-1]
                })
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.chunks[idx]