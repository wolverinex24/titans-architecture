# titans/inference/batch_inference.py
import torch
from .predictor import TitansPredictor
from .memory_management import MemoryManager
from ..utils.logging import setup_logger
from typing import Dict, List, Optional
from tqdm import tqdm

class BatchPredictor:
    def __init__(
        self,
        predictor: TitansPredictor,
        batch_size: int = 32,
        max_length: int = 8192
    ):
        """Batch inference handler for Titans model.
        
        Args:
            predictor: TitansPredictor instance
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_length = max_length
        self.memory_manager = MemoryManager(
            predictor.model.memory_dim,
            predictor.device
        )
        
    def predict_batch(
        self,
        texts: List[str],
        max_new_tokens: int = 100,
        use_tqdm: bool = True
    ) -> List[str]:
        """Generate continuations for a batch of texts."""
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), disable=not use_tqdm):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = []
            
            # Process each text in batch
            for text in batch_texts:
                # Get cached memory if available
                memory_key = f"batch_{i}_{len(batch_results)}"
                memory_state = self.memory_manager.get_memory(memory_key)
                
                # Generate continuation
                generated_text, final_memory = self.predictor.predict(
                    text,
                    max_new_tokens=max_new_tokens,
                    memory_state=memory_state
                )
                
                # Update memory cache
                self.memory_manager.update_memory(
                    memory_key,
                    final_memory,
                    self.predictor.model.neural_memory.get_memory_metrics(final_memory)
                )
                
                batch_results.append(generated_text)
                
            results.extend(batch_results)
            
        return results