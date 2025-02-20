"""Configuration management."""
# titans/utils/config.py
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class ModelConfig:
    input_dim: int = 768
    memory_dim: int = 768
    num_memory_tokens: int = 64
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    max_sequence_length: int = 8192

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 32
    gradient_clip: float = 1.0
    
@dataclass
class TitansConfig:
    model: ModelConfig
    training: TrainingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TitansConfig':
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        return cls(model=model_config, training=training_config)
    
    @classmethod
    def load_yaml(cls, config_path: str) -> 'TitansConfig':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, config_path: str):
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)