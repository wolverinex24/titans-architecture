# titans/scripts/utils/prepare_data.py
import torch
from pathlib import Path
from typing import List, Dict
from transformers import GPT2Tokenizer
import argparse
from data.preprocessing import SequenceProcessor
from utils.logging import setup_logger


def prepare_dataset(
    input_files: List[str],
    output_dir: str,
    max_length: int = 8192,
    chunk_size: int = 1000000
):
    """Prepare and preprocess dataset files."""
    logger = setup_logger("data_preparation")
    processor = SequenceProcessor(
        GPT2Tokenizer.from_pretrained('gpt2'),
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


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument('--input_files', nargs='+', required=True, help='List of input text files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    args = parser.parse_args()

    prepare_dataset(
        input_files=args.input_files,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()