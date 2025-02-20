# titans/scripts/utils/prepare_data.py
import torch
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from transformers import PreTrainedTokenizer

from data.preprocessing import SequenceProcessor

from ..utils import setup_logger

def prepare_dataset(
    input_files: List[str],
    output_dir: str,
    max_length: int = 8192,
    chunk_size: int = 1000000
):
    """Prepare and preprocess dataset files."""
    logger = setup_logger("data_preparation")
    processor = SequenceProcessor(
        PreTrainedTokenizer.from_pretrained('gpt2'),
        max_length=max_length
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for file_path in input_files:
        logger.info(f"Processing {file_path}")
        sequences = processor.process_files(
            [file_path],
            chunk_size=chunk_size
        )
        
        # Save processed sequences
        output_path = output_dir / f"{Path(file_path).stem}_processed.pt"
        torch.save(sequences, output_path)
        
        logger.info(f"Saved processed data to {output_path}")