# tests/conftest.py
import pytest
import torch
import logging
import os

# Configure logging for tests
@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Set up logging configuration for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Suppress excessive logging from third-party libraries
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

# Set up deterministic behavior
@pytest.fixture(scope="session", autouse=True)
def setup_deterministic():
    """Set up deterministic behavior for reproducible testing."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic operations
    os.environ["PYTHONHASHSEED"] = "42"
    
    # PyTorch 2.0+ deterministic flag
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)

# Device fixture to handle testing on CPU or GPU
@pytest.fixture(scope="session")
def device():
    """Determine device to run tests on."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Small config for creating test models
@pytest.fixture
def test_config():
    """Small model configuration for testing."""
    return {
        "input_dim": 64,
        "memory_dim": 32,
        "num_memory_tokens": 8,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.0,
        "vocab_size": 1000,
        "sequence": {
            "max_length": 512,
            "chunk_size": 128,
            "stride": 64
        }
    }

# Input data generator
@pytest.fixture
def get_sample_batch():
    """Factory function to create sample batches of different sizes."""
    def _get_batch(batch_size=2, seq_len=10, vocab_size=1000):
        # Create random token IDs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Create attention mask (all 1s for simplicity)
        attention_mask = torch.ones(batch_size, seq_len)
        # Create target token IDs (shifted right)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    return _get_batch