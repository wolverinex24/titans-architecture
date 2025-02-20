# Titans: Neural Memory Augmented Transformer

A PyTorch implementation of Titans, featuring Neural Memory Module (NMM) for efficient long-context processing.

## Features
- Memory as Context (MAC) architecture implementation
- Neural Memory Module with test-time learning
- Efficient parallel training with tensorized operations 
- Support for long context windows (>2M tokens)
- Modular design for easy experimentation

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 64GB+ RAM recommended

### CUDA Setup
1. Download and install the NVIDIA GPU driver for your graphics card from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

2. Install CUDA Toolkit 11.7 or later:
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
   - Follow the installation instructions for your operating system
   - Add CUDA to your system PATH

3. Verify CUDA installation:
```bash
nvidia-smi  # Should show GPU info
nvcc --version  # Should show CUDA compiler version
```

4. Verify PyTorch CUDA support after installation:
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should show number of GPUs
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

Note: If you encounter any CUDA/PyTorch compatibility issues, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific CUDA version.

### Quick Install
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Script (train.py):
- Command-line interface for training
- Configuration loading
- Data preprocessing
- Training loop with checkpointing

### Evaluation Script (evaluate.py):
- Model evaluation metrics
- Checkpoint loading
- Batch processing
- Metric tracking

### Inference Script (inference.py):
- Text generation
- Memory state management
- Configuration-based inference settings

### Usage Examples:
```bash
# Prepare Data
python -m titans.scripts.utils.prepare_data \
    --input_files data/*.txt \
    --output_dir processed_data
    
# Training
python -m titans.scripts.train \
    --config base \
    --data_path /path/to/data \
    --output_dir ./outputs

# Evaluation
python -m titans.scripts.evaluate \
    --checkpoint ./outputs/checkpoints/latest.pt \
    --data_path /path/to/eval_data

# Inference
python -m titans.scripts.inference \
    --checkpoint ./outputs/checkpoints/latest.pt \
    --input_text "Once upon a time" \
    --max_new_tokens 100
```

## Project Structure
```
titans/
├── core/          # Core model components
├── data/          # Data loading and processing
├── training/      # Training utilities
├── inference/     # Inference utilities
├── scripts/       # Command-line tools
└── utils/         # Helper functions
```

## Citation
If you use this code, please cite:
```bibtex
@article{titans2024,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}
```

## License
MIT License
