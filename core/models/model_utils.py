"""Model-related utilities."""
# titans/core/models/model_utils.py
import torch
from typing import Dict, Any

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0

def get_model_config(model_size: str = "base") -> Dict[str, Any]:
    """Get model configuration based on size."""
    configs = {
        "small": {
            "input_dim": 512,
            "memory_dim": 512,
            "num_memory_tokens": 32,
            "num_heads": 8,
            "num_layers": 2,
        },
        "base": {
            "input_dim": 768,
            "memory_dim": 768,
            "num_memory_tokens": 64,
            "num_heads": 12,
            "num_layers": 2,
        },
        "large": {
            "input_dim": 1024,
            "memory_dim": 1024,
            "num_memory_tokens": 128,
            "num_heads": 16,
            "num_layers": 3,
        }
    }
    return configs.get(model_size, configs["base"])