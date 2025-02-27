"""Tests for neural_memory module."""

# tests/test_memory/test_neural_memory.py
import pytest
import torch
import torch.nn as nn
from core.memory.neural_memory import NeuralMemoryModule

class TestNeuralMemoryModule:
    @pytest.fixture
    def memory_module(self):
        # Create a small memory module for testing
        return NeuralMemoryModule(
            input_dim=64,
            memory_dim=32,
            num_layers=2,
            dropout=0.0  # Disable dropout for deterministic testing
        )
    
    def test_initialization(self, memory_module):
        """Test if memory module initializes correctly."""
        # Check key attributes
        assert memory_module.input_dim == 64
        assert memory_module.memory_dim == 32
        assert memory_module.num_layers == 2
        
        # Check layer initialization
        assert len(memory_module.memory_layers) == 2
        assert len(memory_module.layer_norms) == 2
        
        # Check projection layers
        assert isinstance(memory_module.key_proj, nn.Linear)
        assert isinstance(memory_module.value_proj, nn.Linear)
        assert isinstance(memory_module.query_proj, nn.Linear)

    def test_forward_pass(self, memory_module):
        """Test the forward pass with batch of sequences."""
        batch_size = 2
        seq_len = 5
        
        # Create input tensor
        inputs = torch.randn(batch_size, seq_len, 64)
        
        # Run forward pass
        memory_state, momentum = memory_module(inputs)
        
        # Check output shapes
        assert memory_state.shape == (batch_size, 32)
        assert momentum.shape == (batch_size, 32)
        
        # Check that memory state isn't all zeros (has been updated)
        assert not torch.allclose(memory_state, torch.zeros_like(memory_state))
    
    def test_single_token_input(self, memory_module):
        """Test processing a single token (no sequence dimension)."""
        batch_size = 3
        
        # Create input tensor without sequence dimension
        inputs = torch.randn(batch_size, 64)
        
        # Run forward pass
        memory_state, momentum = memory_module(inputs)
        
        # Check output shapes
        assert memory_state.shape == (batch_size, 32)
        assert momentum.shape == (batch_size, 32)
    
    def test_memory_state_persistence(self, memory_module):
        """Test that memory state is updated properly."""
        batch_size = 2
        seq_len = 4
        
        # Create input tensor
        inputs = torch.randn(batch_size, seq_len, 64)
        
        # Initial forward pass
        memory_state1, momentum1 = memory_module(inputs)
        
        # Second forward pass with same input but previous memory
        memory_state2, momentum2 = memory_module(inputs, memory_state=memory_state1)
        
        # Memory states should be different due to updates
        assert not torch.allclose(memory_state1, memory_state2)
    
    def test_memory_retrieval(self, memory_module):
        """Test memory retrieval functionality."""
        batch_size = 3
        seq_len = 6
        
        # Create input and memory state
        inputs = torch.randn(batch_size, seq_len, 64)
        memory_state, _ = memory_module(inputs)
        
        # Test retrieval
        queries = torch.randn(batch_size, seq_len, 64)
        retrieved = memory_module.retrieve(queries, memory_state)
        
        # Check output shape
        assert retrieved.shape == (batch_size, seq_len, 32)
        
        # Test retrieval with different sequence length
        new_seq_len = 3
        queries2 = torch.randn(batch_size, new_seq_len, 64)
        retrieved2 = memory_module.retrieve(queries2, memory_state)
        
        # Check output shape matches new sequence length
        assert retrieved2.shape == (batch_size, new_seq_len, 32)
    
    def test_surprise_computation(self, memory_module):
        """Test the surprise calculation mechanism."""
        batch_size = 2
        seq_len = 5
        
        # Create inputs with requires_grad=True
        inputs = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        
        # Project inputs to get keys and values
        keys = memory_module.key_proj(inputs)
        values = memory_module.value_proj(inputs)
        
        # Initialize memory state with requires_grad=True
        memory_state = torch.zeros(batch_size, 32, requires_grad=True)
        
        # Compute surprise
        surprise_output = memory_module._compute_surprise(memory_state, keys, values)
        
        # The function returns a tuple of (scalar_surprise, tensor_surprise)
        # Verify it's a tuple with 2 tensor elements
        assert isinstance(surprise_output, tuple)
        assert len(surprise_output) == 2
        assert isinstance(surprise_output[0], torch.Tensor)  # First element is scalar surprise
        assert isinstance(surprise_output[1], torch.Tensor)  # Second element is tensor surprise
        
        # The second element should have the shape [batch_size, memory_dim]
        assert surprise_output[1].shape == (batch_size, 32)
    
    def test_gradient_flow(self, memory_module):
        """Test that gradients flow properly through the module."""
        batch_size = 2
        seq_len = 4
        
        # Create input that requires gradients
        inputs = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        
        # Forward pass
        memory_state, momentum = memory_module(inputs)
        
        # Create a dummy loss and backpropagate
        loss = memory_state.sum() + momentum.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert inputs.grad is not None
        assert not torch.allclose(inputs.grad, torch.zeros_like(inputs.grad))