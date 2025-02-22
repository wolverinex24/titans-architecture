"""Dataset implementations."""
# titans/data/dataset.py
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TitansDataset(Dataset):
    def __init__(
        self,
        sequences: List[torch.Tensor],
        sequence_length: int = 8192,
        stride: Optional[int] = None,
        pad_token_id: int = 0
    ):
        """Dataset for Titans model training.
        
        Args:
            sequences: List of token ID sequences
            sequence_length: Maximum sequence length
            stride: Stride for sliding window
            pad_token_id: Token ID for padding
        """
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.pad_token_id = pad_token_id
        
        # Validate input sequences
        logger.info(f"Initializing dataset with {len(sequences)} sequences")
        for i, seq in enumerate(sequences):
            if not isinstance(seq, torch.Tensor):
                sequences[i] = torch.tensor(seq, dtype=torch.long)
            elif seq.dtype != torch.long:
                sequences[i] = seq.long()
        
        self.sequences = sequences
        self.chunks = self._prepare_chunks()
        logger.info(f"Created {len(self.chunks)} chunks")
        
    def _prepare_chunks(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare sequence chunks for training."""
        chunks = []
        
        for sequence in self.sequences:
            # Skip sequences shorter than 2 tokens
            if len(sequence) < 2:
                continue
                
            # Flatten if needed
            if sequence.dim() > 1:
                sequence = sequence.view(-1)
            
            # Calculate number of chunks
            seq_len = len(sequence)
            num_chunks = max(1, (seq_len - self.sequence_length) // self.stride + 1)
            
            for i in range(num_chunks):
                start_idx = i * self.stride
                end_idx = start_idx + self.sequence_length
                
                # Get chunk and ensure it's the right length
                chunk = sequence[start_idx:min(end_idx, seq_len)]
                if len(chunk) < self.sequence_length:
                    # Pad if needed
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
                    "attention_mask": (chunk[:-1] != self.pad_token_id).float()
                })
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.chunks[idx]