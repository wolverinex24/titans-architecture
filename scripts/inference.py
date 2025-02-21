"""Inference script."""
# titans/scripts/inference.py
import argparse
import torch
from transformers import PreTrainedTokenizer

from core.models import TitansMAC
from inference import TitansPredictor, MemoryManager
from data import SequenceProcessor
from utils import ConfigLoader, setup_logger

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
        input_dim=config['model']['input_dim'],
        memory_dim=config['model']['memory_dim'],
        num_memory_tokens=config['model']['num_memory_tokens'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=0.0
    ).to(args.device)
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    tokenizer = PreTrainedTokenizer.from_pretrained('gpt2')
    processor = SequenceProcessor(
        tokenizer,
        max_length=config['model']['sequence']['max_length']
    )
    
    # Setup predictor
    predictor = TitansPredictor(
        model=model,
        tokenizer=processor,
        device=args.device,
        max_length=config['model']['sequence']['max_length']
    )
    
    # Generate text
    logger.info("Generating text...")
    generated_text, memory_state = predictor.predict(
        args.input_text,
        max_new_tokens=args.max_new_tokens,
        temperature=config['inference']['generation']['temperature'],
        top_k=config['inference']['generation']['top_k'],
        top_p=config['inference']['generation']['top_p']
    )
    
    logger.info("\nGenerated Text:")
    logger.info(f"{args.input_text}{generated_text}")

if __name__ == '__main__':
    main()