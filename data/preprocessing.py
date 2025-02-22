# titans/data/preprocessing.py
from typing import List, Optional, Union
import torch
from transformers import GPT2Tokenizer

class SequenceProcessor:
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        max_length: int = 8192,
        stride: Optional[int] = None
    ):
        """Process text sequences for Titans model.
        
        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length // 2
        
    def process_text(
        self,
        texts: Union[str, List[str]],
        add_special_tokens: bool = True
    ) -> List[torch.Tensor]:
        """Process text or list of texts into model inputs."""
        if isinstance(texts, str):
            texts = [texts]
            
        sequences = []
        for text in texts:
            # Tokenize
            tokens = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=False
            )["input_ids"].squeeze(0)
            
            sequences.append(tokens)
            
        return sequences
        
    def process_files(
        self,
        file_paths: List[str],
        chunk_size: int = 1000000  # Process 1M characters at a time
    ) -> List[torch.Tensor]:
        """Process text files in chunks to handle large files."""
        sequences = []
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                        
                    # Process chunk
                    chunk_sequences = self.process_text(chunk)
                    sequences.extend(chunk_sequences)
                    
        return sequences