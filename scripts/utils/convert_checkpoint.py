# titans/scripts/utils/convert_checkpoint.py
import torch
from typing import Dict, Any, Optional
from pathlib import Path

def convert_checkpoint(
    checkpoint_path: str,
    output_format: str = 'huggingface',
    output_path: Optional[str] = None
) -> None:
    """Convert model checkpoint to different formats."""
    checkpoint = torch.load(checkpoint_path)
    
    if output_format == 'huggingface':
        # Convert to HuggingFace format
        hf_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            # Map Titans keys to HF keys
            if key.startswith('neural_memory'):
                hf_key = f'memory_module.{key[13:]}'
            elif key.startswith('persistent_memory'):
                hf_key = f'persistent_memory.{key[17:]}'
            else:
                hf_key = key
            hf_state_dict[hf_key] = value
            
        output_dict = {
            'state_dict': hf_state_dict,
            'config': checkpoint.get('config', {}),
            'format': 'huggingface'
        }
    else:
        raise ValueError(f"Unsupported format: {output_format}")
        
    # Save converted checkpoint
    if output_path is None:
        output_path = str(Path(checkpoint_path).with_suffix(f'.{output_format}.pt'))
    torch.save(output_dict, output_path)