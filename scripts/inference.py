"""Inference script."""
# titans/scripts/inference.py
import argparse
import torch
from typing import Any
from transformers import GPT2Tokenizer  # Use this instead of PreTrainedTokenizer

from core.models import TitansMAC
from inference import TitansPredictor, MemoryManager
from data import SequenceProcessor
from utils import setup_logger
from utils.config_loader import ConfigLoader

def convert_type(value: Any, expected_type: type) -> Any:
    """Convert value to expected type."""
    if value is None:
        return None
    try:
        if expected_type == bool:
            return str(value).lower() in ('true', '1', 'yes')
        return expected_type(value)
    except (ValueError, TypeError):
        return value

def get_config_value(
    config: dict,
    path: str,
    default: Any = None,
    value_type: type = None
) -> Any:
    """Safely get nested config value with type conversion.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path to value
        default: Default value if path not found
        value_type: Expected type of the value
    """
    keys = path.split('.')
    current = config
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return convert_type(default, value_type) if value_type else default
        current = current[key]
    
    return convert_type(current, value_type) if value_type else current

def main():
    parser = argparse.ArgumentParser(description='Run inference with Titans MAC model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='base',
                       help='Model configuration name')
    parser.add_argument('--input_text', type=str, required=True,
                       help='Input text for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger("titans_inference")
    config_loader = ConfigLoader("configs")
    config = config_loader.load_model_config(args.config)
    
    # Initialize model and processor
    model = TitansMAC(
        input_dim=get_config_value(config, 'model.input_dim', 512, int),
        memory_dim=get_config_value(config, 'model.memory_dim', 512, int),
        num_memory_tokens=get_config_value(config, 'model.num_memory_tokens', 32, int),
        num_heads=get_config_value(config, 'model.num_heads', 8, int),
        num_layers=get_config_value(config, 'model.num_layers', 2, int),
        dropout=0.0,
        vocab_size=get_config_value(config, 'model.vocab_size', 50000, int)
    ).to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    print(checkpoint)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Use GPT2Tokenizer directly
    processor = SequenceProcessor(
        tokenizer,
        max_length=get_config_value(config, 'model.sequence.max_length', 1024, int)
    )
    
    # Setup predictor
    predictor = TitansPredictor(
        model=model,
        tokenizer=processor,
        device=args.device,
        max_length=get_config_value(config, 'model.sequence.max_length', 1024, int)
    )
    
    # Generate text
    logger.info("Generating text...")
    generated_text, memory_state = predictor.predict(
        args.input_text,
        max_new_tokens=args.max_new_tokens,
        temperature=get_config_value(config, 'inference.generation.temperature', 0.8, float),
        top_k=get_config_value(config, 'inference.generation.top_k', 50, int),
        top_p=get_config_value(config, 'inference.generation.top_p', 0.9, float)
    )
    
    logger.info("\nGenerated Text:")
    logger.info(f"{args.input_text}{generated_text}")

if __name__ == '__main__':
    main()