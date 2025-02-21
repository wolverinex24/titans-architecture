"""Training script."""
# titans/scripts/train.py
import argparse
import torch
from pathlib import Path
from transformers import PreTrainedTokenizer

from core.models.mac import TitansMAC
from data.dataloader import create_dataloader
from data.dataset import TitansDataset
from data.preprocessing import SequenceProcessor
from training.trainer import TitansTrainer
from utils.checkpoint import CheckpointManager
from utils.logging import setup_logger, ConfigLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train Titans MAC model')
    parser.add_argument('--config', type=str, default='base',
                       help='Model configuration name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging and configuration
    logger = setup_logger("titans_training", f"{args.output_dir}/train.log")
    config_loader = ConfigLoader("configs")
    config = config_loader.load_model_config(args.config)
    
    logger.info(f"Starting training with configuration: {args.config}")
    
    # Initialize tokenizer and processor
    tokenizer = PreTrainedTokenizer.from_pretrained('gpt2')
    processor = SequenceProcessor(
        tokenizer,
        max_length=config['model']['sequence']['max_length']
    )
    
    # Load and preprocess data
    logger.info("Loading and processing data...")
    data_path = Path(args.data_path)
    sequences = processor.process_files([str(data_path)])
    
    # Create dataset and dataloader
    dataset = TitansDataset(
        sequences,
        sequence_length=config['model']['sequence']['max_length'],
        stride=config['model']['sequence']['stride']
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model
    model = TitansMAC(
        input_dim=config['model']['input_dim'],
        memory_dim=config['model']['memory_dim'],
        num_memory_tokens=config['model']['num_memory_tokens'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    trainer = TitansTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device
    )
    
    checkpoint_manager = CheckpointManager(
        save_dir=f"{args.output_dir}/checkpoints",
        max_checkpoints=config['validation']['max_checkpoints']
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config['training']['max_epochs']):
        metrics = trainer.train_epoch(
            dataloader,
            max_grad_norm=config['training']['optimizer']['gradient_clip']
        )
        
        logger.info(f"Epoch {epoch}: {metrics}")
        
        # Save checkpoint
        if (epoch + 1) % config['validation']['save_steps'] == 0:
            checkpoint_manager.save(
                model.state_dict(),
                metrics,
                epoch
            )
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()