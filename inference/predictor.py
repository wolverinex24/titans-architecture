"""Inference pipeline."""
# titans/inference/predictor.py
import torch
import torch.nn as nn
from typing import Any, Optional, Dict, List, Union, Tuple
from core.models import TitansMAC
from data.preprocessing import SequenceProcessor
from .memory_management import MemoryManager
from utils.logging import setup_logger

class TitansPredictor:
    def __init__(
        self,
        model: TitansMAC,
        tokenizer: SequenceProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 8192,
        memory_manager: Optional[MemoryManager] = None
    ):
        """Predictor class for Titans model inference with test-time learning.
        
        Args:
            model: Trained Titans model
            tokenizer: Sequence processor for tokenization
            device: Device to run inference on
            max_length: Maximum sequence length
            memory_manager: Optional memory manager for caching
        """
        self.model = model.to(device)
        self.model.eval()  # Still use eval mode for dropout etc
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.memory_manager = memory_manager or MemoryManager(
            model.neural_memory.memory_dim,
            device=device
        )
        
    def _process_sequence(self, input_ids: torch.Tensor, memory_state: Optional[torch.Tensor] = None):
        pad_token_id = self.tokenizer.tokenizer.pad_token_id or 0
        attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        if pad_token_id is not None:
            attention_mask = (input_ids != pad_token_id).to(torch.float)
        
        # Ensure memory_state is a tensor
        if isinstance(memory_state, tuple):
            memory_state = memory_state[0]
            
        outputs, updated_memory = self.model(
            inputs=input_ids,
            memory_state=memory_state,
            attention_mask=attention_mask,
            is_inference=True
        )
        
        return outputs, updated_memory

    def predict(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        memory_state: Optional[torch.Tensor] = None,
        use_memory_cache: bool = True
    ) -> Tuple[str, torch.Tensor]:
        # Process input text
        input_ids = self.tokenizer.process_text(input_text)[0].to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
                
        # Initialize generation
        generated_ids = [input_ids]
        current_memory = memory_state
        
        try:
            for _ in range(max_new_tokens):
                # Prepare context window
                # Ensure all tensors have same shape before concatenation
                tensors_to_cat = []
                for tensor in generated_ids:
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    tensors_to_cat.append(tensor)
                
                context = torch.cat(tensors_to_cat, dim=1)
                if context.size(1) > self.max_length:
                    context = context[:, -self.max_length:]
                
                # Process sequence
                try:
                    outputs, current_memory = self._process_sequence(
                        context,
                        memory_state=current_memory
                    )
                    
                    # Get next token logits (with shape checking)
                    if outputs.size(0) > 0 and outputs.size(1) > 0:
                        next_token_logits = outputs[:, -1, :] / temperature
                    else:
                        print(f"Invalid outputs shape: {outputs.shape}")
                        break
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_values, _ = torch.topk(next_token_logits, top_k)
                        indices_to_remove = next_token_logits < top_k_values[..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted indices
                        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Add to generated sequence
                    generated_ids.append(next_token)
                    
                except RuntimeError as e:
                    print(f"Error during sequence processing: {str(e)}")
                    break
                
            # Concatenate all tensors with shape checking
            tensors_to_cat = []
            for tensor in generated_ids:
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                tensors_to_cat.append(tensor)
                
            generated_tokens = torch.cat(tensors_to_cat, dim=1)
            
            # Decode tokens
            generated_text = self.tokenizer.tokenizer.decode(
                generated_tokens[0, input_ids.size(1):].tolist()
            )
            
            # Update memory cache
            if use_memory_cache and current_memory is not None:
                try:
                    metrics = self.model.neural_memory.get_memory_metrics(current_memory)
                    print(f"matric for memory :{metrics}")
                    self.memory_manager.update_memory(str(hash(input_text[:100])), current_memory, metrics)
                except Exception as e:
                    print(f"Failed to update memory cache: {str(e)}")
            
            return generated_text, current_memory
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return "", None