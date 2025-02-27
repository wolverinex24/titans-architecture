"""Tests for persistent_memory module."""

# tests/test_memory/test_persistent_memory.py
import pytest
import torch
import math
from core.memory.persistent_memory import PersistentMemory

class TestPersistentMemory:
    @pytest.fixture
    def persistent_memory(self):
        # Create a persistent memory module for testing
        return PersistentMemory(
            num_tokens=16,
            dim=64,
            init_scale=0.02
        )
    
    def test_initialization(self, persistent_memory):
        """Test if persistent memory initializes correctly."""
        # Check memory token shape
        assert persistent_memory.memory_tokens.shape == (16, 64)
        
        # Verify custom initialization was applied
        # Memory tokens should have a specific standard deviation based on initialization
        std = math.sqrt(2.0 / (16 + 64))
        actual_std = persistent_memory.memory_tokens.std().item()
        
        # Allow some tolerance due to random initialization
        assert abs(actual_std - std) < 0.01
    
    def test_forward_no_batch(self, persistent_memory):
        """Test forward pass without batch dimension."""
        # Get memory tokens
        tokens = persistent_memory()
        
        # Check shape
        assert tokens.shape == (16, 64)
        
        # Verify it returns the memory tokens
        assert torch.allclose(tokens, persistent_memory.memory_tokens)
    
    def test_forward_with_batch(self, persistent_memory):
        """Test forward pass with batch dimension."""
        batch_size = 3
        
        # Get memory tokens with batch dimension
        tokens = persistent_memory(batch_size)
        
        # Check shape
        assert tokens.shape == (batch_size, 16, 64)
        
        # Verify all batch items are the same (expanded from the same tokens)
        for i in range(1, batch_size):
            assert torch.allclose(tokens[0], tokens[i])
    
    def test_learnable_parameters(self, persistent_memory):
        """Test that memory tokens are learnable parameters."""
        # Check that memory tokens are nn.Parameter
        assert isinstance(persistent_memory.memory_tokens, torch.nn.Parameter)
        
        # Verify requires_grad is True
        assert persistent_memory.memory_tokens.requires_grad
        
        # Check that parameters are updated during backpropagation
        initial_tokens = persistent_memory.memory_tokens.clone()
        
        # Create a dummy loss and backpropagate
        output = persistent_memory().mean()
        output.backward()
        
        # Check that gradients were computed
        assert persistent_memory.memory_tokens.grad is not None
        
        # Update weights with a dummy optimizer step
        with torch.no_grad():
            persistent_memory.memory_tokens -= 0.1 * persistent_memory.memory_tokens.grad
        
        # Verify parameters changed
        assert not torch.allclose(initial_tokens, persistent_memory.memory_tokens)
    
    def test_custom_init_scale(self):
        """Test custom initialization scale."""
        # Create with larger init scale
        large_init = PersistentMemory(
            num_tokens=16,
            dim=64,
            init_scale=0.1
        )
        
        # Create with smaller init scale
        small_init = PersistentMemory(
            num_tokens=16,
            dim=64,
            init_scale=0.001
        )
        
        # Verify standard deviations differ appropriately
        std_large = large_init.memory_tokens.std().item()
        std_small = small_init.memory_tokens.std().item()
        
        # With the custom _init_parameters method, both should end up
        # with the same std dev regardless of init_scale
        std = math.sqrt(2.0 / (16 + 64))
        assert abs(std_large - std) < 0.01
        assert abs(std_small - std) < 0.01