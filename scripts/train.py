# titans/scripts/train.py
import argparse
import os
from typing import Any
import torch
from pathlib import Path
from transformers import PreTrainedTokenizer

from core.models.mac import TitansMAC
from data.dataloader import create_dataloader
from data.dataset import TitansDataset
from data.preprocessing import SequenceProcessor
from training.trainer import TitansTrainer
from utils.checkpoint import CheckpointManager
from utils.logging import setup_logger
from utils.config_loader import ConfigLoader

def convert_type(value: Any, expected_type: type) -> Any:
    if value is None:
        return None
    try:
        if expected_type == bool:
            return str(value).lower() in ('true', '1', 'yes')
        return expected_type(value)
    except (ValueError, TypeError):
        return value

def get_config_value(config: dict, path: str, default: Any = None, value_type: type = None) -> Any:
    keys = path.split('.')
    current = config
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return convert_type(default, value_type) if value_type else default
        current = current[key]
    
    return convert_type(current, value_type) if value_type else current

def setup_training():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def main():
    parser = argparse.ArgumentParser(description='Train Titans MAC model')
    parser.add_argument('--config', type=str, default='small',
                       help='Model configuration name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed .pt data file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing to save memory')
    # Add save_every argument with a reasonable default
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    args = parser.parse_args()
    # Setup training optimizations
    setup_training()
    
    # Create output directory and setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("titans_training", output_dir / "train.log")
    
    # Load configuration
    config_loader = ConfigLoader("configs")
    config = config_loader.load_model_config(args.config)
    
    # Load data
    data_path = Path(args.data_path)
    sequences = torch.load(data_path)
    
    # Create dataset and dataloader
    max_length = get_config_value(config, 'model.sequence.max_length', 1024, int)
    stride = get_config_value(config, 'model.sequence.stride', 256, int)
    batch_size = get_config_value(config, 'training.batch_size', 4, int)
    
    dataset = TitansDataset(
        sequences=sequences,
        sequence_length=max_length,
        stride=stride
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = TitansMAC(
        input_dim=get_config_value(config, 'model.input_dim', 512, int),
        memory_dim=get_config_value(config, 'model.memory_dim', 512, int),
        num_memory_tokens=get_config_value(config, 'model.num_memory_tokens', 32, int),
        num_heads=get_config_value(config, 'model.num_heads', 8, int),
        num_layers=get_config_value(config, 'model.num_layers', 2, int),
        dropout=get_config_value(config, 'model.dropout', 0.1, float),
        vocab_size=get_config_value(config, 'model.vocab_size', 50000, int)
    ).to(args.device)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=get_config_value(config, 'training.learning_rate', 1e-4, float),
        weight_decay=get_config_value(config, 'training.weight_decay', 0.01, float)
    )
    
    # Setup trainer and checkpoint manager
    trainer = TitansTrainer(model=model, optimizer=optimizer, device=args.device)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_manager = CheckpointManager(
        save_dir=str(checkpoint_dir),
        max_checkpoints=get_config_value(config, 'validation.max_checkpoints', 5, int)
    )
    
     # Training loop
    max_epochs = get_config_value(config, 'training.max_epochs', 100, int)
    save_steps = args.save_every  # Using command line argument
    grad_clip = get_config_value(config, 'training.optimizer.gradient_clip', 1.0, float)
    
    for epoch in range(max_epochs):
        metrics = trainer.train_epoch(dataloader, max_grad_norm=grad_clip)
        logger.info(f"Epoch {epoch}: {metrics}")
        
        # Save checkpoint
        if (epoch + 1) % save_steps == 0:
            logger.info(f"Saving checkpoint at epoch {epoch + 1}")
            
            # Format metrics to float
            metrics = {k: float(v) for k, v in metrics.items()}
            
            # Save model state
            checkpoint_state = {
                'state_dict': model.state_dict(),  # Just the model state dict
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }
            
            # Save
            checkpoint_manager.save(
                state_dict=checkpoint_state['state_dict'],  # Only pass model state dict
                metrics=metrics,
                step=epoch
            )
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()