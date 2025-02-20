"""Inference pipeline."""
# titans/inference/predictor.py
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple
from ..core.models import TitansMAC
from ..data.preprocessing import SequenceProcessor
from .memory_management import MemoryManager
from ..utils.logging import setup_logger

class TitansPredictor:
    def __init__(
        self,
        model: TitansMAC,
        tokenizer: SequenceProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 8192,
    ):
        """Predictor class for Titans model inference.
        
        Args:
            model: Trained Titans model
            tokenizer: Sequence processor for tokenization
            device: Device to run inference on
            max_length: Maximum sequence length
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
    @torch.no_grad()
    def predict(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[str, torch.Tensor]:
        """Generate text continuation.
        
        Args:
            input_text: Input text to continue from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            memory_state: Optional previous memory state
            
        Returns:
            generated_text: Generated text continuation
            final_memory: Final memory state
        """
        # Process input text
        input_ids = self.tokenizer.process_text(input_text)[0].to(self.device)
        
        # Initialize generation
        generated_ids = [input_ids]
        current_memory = memory_state
        
        for _ in range(max_new_tokens):
            # Get input context (limited by max_length)
            context = torch.cat(generated_ids)[-self.max_length:]
            
            # Forward pass
            outputs, current_memory = self.model(
                context.unsqueeze(0),
                memory_state=current_memory
            )
            
            # Get next token logits
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids.append(next_token.unsqueeze(0))
            
        # Decode generated tokens
        generated_text = self.tokenizer.tokenizer.decode(
            torch.cat(generated_ids)[len(input_ids):].tolist()
        )
        
        return generated_text, current_memory