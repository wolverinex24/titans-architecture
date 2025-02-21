"""Evaluation script."""
# titans/scripts/evaluate.py
import argparse
import torch
from pathlib import Path
from transformers import PreTrainedTokenizer

from core.models.mac import TitansMAC
from data.dataset import TitansDataset, create_dataloader, SequenceProcessor
from utils.logging import ConfigLoader, setup_logger, MetricsTracker

def evaluate(model, dataloader, device):
    model.eval()
    metrics = MetricsTracker()
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs, _ = model(
                inputs=batch['input_ids'],
                attention_mask=batch.get('attention_mask')
            )
            
            # Compute metrics
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch['labels'].view(-1),
                ignore_index=-100
            )
            
            metrics.update({
                'loss': loss.item(),
                'perplexity': torch.exp(loss).item()
            })
    
    return metrics.compute_average()

def main():
    parser = argparse.ArgumentParser(description='Evaluate Titans MAC model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='base',
                       help='Model configuration name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to evaluation data')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to evaluate on')
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger("titans_eval")
    config_loader = ConfigLoader("configs")
    config = config_loader.load_model_config(args.config)
    
    # Load model and checkpoint
    model = TitansMAC(
        input_dim=config['model']['input_dim'],
        memory_dim=config['model']['memory_dim'],
        num_memory_tokens=config['model']['num_memory_tokens'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=0.0  # No dropout during evaluation
    ).to(args.device)
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Setup data
    tokenizer = PreTrainedTokenizer.from_pretrained('gpt2')
    processor = SequenceProcessor(
        tokenizer,
        max_length=config['model']['sequence']['max_length']
    )
    
    sequences = processor.process_files([args.data_path])
    dataset = TitansDataset(
        sequences,
        sequence_length=config['model']['sequence']['max_length']
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=config['validation']['eval_batch_size'],
        shuffle=False
    )
    
    # Run evaluation
    metrics = evaluate(model, dataloader, args.device)
    
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")