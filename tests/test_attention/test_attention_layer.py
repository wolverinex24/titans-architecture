"""Tests for attention_layer module."""

# tests/test_attention/test_attention_layer.py
import pytest
import torch
import torch.nn as nn
from core.attention.attention_layer import MultiHeadAttention

class TestMultiHeadAttention:
    @pytest.fixture
    def mha(self):  # Changed from model to mha to match usage in tests
        """Create a MultiHeadAttention instance for testing."""
        return MultiHeadAttention(
            dim=64,
            num_heads=4,
            dropout=0.0  # Set to 0 for deterministic testing
        )
    
    def test_initialization(self, mha):
        """Test if the MultiHeadAttention module initializes correctly."""
        # Check key attributes
        assert mha.dim == 64
        assert mha.num_heads == 4
        assert mha.head_dim == 16  # 64 / 4 = 16
        assert abs(mha.scale - 0.25) < 1e-6  # 1 / sqrt(16) = 0.25
        
        # Check projection layers
        assert isinstance(mha.q_proj, nn.Linear)
        assert isinstance(mha.k_proj, nn.Linear)
        assert isinstance(mha.v_proj, nn.Linear)
        assert isinstance(mha.out_proj, nn.Linear)
        
        # Check projection dimensions
        assert mha.q_proj.in_features == 64
        assert mha.q_proj.out_features == 64
    
    def test_forward_pass(self, mha):
        """Test forward pass with simple inputs."""
        batch_size = 2
        seq_len = 5
        
        # Create inputs - changed from 384 to 64 to match the fixture
        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)
        
        # Run forward pass
        output = mha(query, key, value)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 64)
    
    def test_different_sequence_lengths(self, mha):
        """Test attention with different sequence lengths for query and key/value."""
        batch_size = 2
        query_len = 5
        kv_len = 8
        
        # Create inputs
        query = torch.randn(batch_size, query_len, 64)
        key = torch.randn(batch_size, kv_len, 64)
        value = torch.randn(batch_size, kv_len, 64)
        
        # Run forward pass
        output = mha(query, key, value)
        
        # Check output shape (should match query sequence length)
        assert output.shape == (batch_size, query_len, 64)
    
    def test_mask_application(self, mha):
        """Test attention mask is correctly applied."""
        batch_size = 2
        seq_len = 6
        
        # Create inputs
        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)
        
        # Create attention mask - shape should be [batch_size, seq_len, seq_len]
        mask = torch.ones(batch_size, seq_len, seq_len)
        # Mask out the third position in each sequence for all queries
        mask[:, :, 2] = 0
        
        # Run forward pass with mask
        output_masked = mha(query, key, value, mask=mask)
        
        # Run forward pass without mask for comparison
        output_unmasked = mha(query, key, value)
        
        # Outputs should be different due to masking
        assert not torch.allclose(output_masked, output_unmasked, rtol=1e-3, atol=1e-3)
    
    def test_causal_mask(self):
        """Test attention with causal masking."""
        # Create a new attention module
        mha = MultiHeadAttention(
            dim=64,
            num_heads=4,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 5
        
        # Create inputs
        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)
        
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(batch_size, seq_len, seq_len))
        
        # Run forward pass with causal mask
        output = mha(query, key, value, mask=mask)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 64)
    
    def test_gradient_flow(self, mha):
        """Test that gradients flow properly through the attention."""
        batch_size = 2
        seq_len = 4
        
        # Create inputs that require gradients
        query = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        key = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        value = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        
        # Forward pass
        output = mha(query, key, value)
        
        # Create a dummy loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        
        # Check gradients are not zero
        assert not torch.allclose(query.grad, torch.zeros_like(query.grad))
        assert not torch.allclose(key.grad, torch.zeros_like(key.grad))
        assert not torch.allclose(value.grad, torch.zeros_like(value.grad))
    
    def test_multiple_forward_passes(self, mha):
        """Test consistency with multiple forward passes in eval mode."""
        mha.eval()  # Set to evaluation mode
        
        batch_size = 2
        seq_len = 4
        
        # Create inputs
        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)
        
        # Run forward pass twice
        with torch.no_grad():
            output1 = mha(query, key, value)
            output2 = mha(query, key, value)
        
        # Outputs should be identical in eval mode with no dropout
        assert torch.allclose(output1, output2)
