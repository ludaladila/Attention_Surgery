"""
Unit tests for ablation methods.
"""
import pytest
import torch
import torch.nn.functional as F
from attention_surgery.core.ablation import (
    ablate_head_zero,
    ablate_head_mean,
    ablate_head_random,
    ablate_head_previous,
    apply_ablation,
)


class TestAblationMethods:
    """Test suite for attention ablation functions."""

    @pytest.fixture
    def sample_pattern(self):
        """Create a sample attention pattern."""
        # [batch=1, heads=12, seq_q=5, seq_k=5]
        pattern = torch.rand(1, 12, 5, 5)
        # Normalize to make it a valid attention distribution
        pattern = F.softmax(pattern, dim=-1)
        return pattern

    def test_ablate_head_zero_with_renormalization(self, sample_pattern):
        """Test zero ablation with renormalization (default)."""
        result = ablate_head_zero(sample_pattern, head_idx=0, renormalize=True)

        # Check shape unchanged
        assert result.shape == sample_pattern.shape

        # Check that ablated head has uniform distribution
        ablated_head = result[0, 0, :, :]
        seq_k = ablated_head.shape[-1]
        expected_value = 1.0 / seq_k

        assert torch.allclose(ablated_head, torch.full_like(ablated_head, expected_value), atol=1e-6)

        # Check that each row sums to 1
        row_sums = ablated_head.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

        # Check other heads unchanged
        assert torch.allclose(result[0, 1, :, :], sample_pattern[0, 1, :, :])

    def test_ablate_head_mean(self, sample_pattern):
        """Test mean ablation."""
        result = ablate_head_mean(sample_pattern, head_idx=0)

        assert result.shape == sample_pattern.shape

        # Check that ablated head has uniform distribution (normalized)
        ablated_head = result[0, 0, :, :]
        row_sums = ablated_head.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_ablate_head_random_reproducibility(self, sample_pattern):
        """Test that random ablation is reproducible with seed."""
        result1 = ablate_head_random(sample_pattern, head_idx=0, seed=42)
        result2 = ablate_head_random(sample_pattern, head_idx=0, seed=42)

        # Results should be identical with same seed
        assert torch.allclose(result1, result2)

        # Check valid distribution
        ablated_head = result1[0, 0, :, :]
        row_sums = ablated_head.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_ablate_head_random_different_seeds(self, sample_pattern):
        """Test that different seeds produce different results."""
        result1 = ablate_head_random(sample_pattern, head_idx=0, seed=42)
        result2 = ablate_head_random(sample_pattern, head_idx=0, seed=99)

        # Results should be different with different seeds
        assert not torch.allclose(result1[0, 0, :, :], result2[0, 0, :, :])

    def test_ablate_head_previous_with_cache(self, sample_pattern):
        """Test previous layer ablation with cached patterns."""
        # Create previous layer patterns
        previous_patterns = {
            0: torch.rand(1, 12, 5, 5),
        }
        previous_patterns[0] = F.softmax(previous_patterns[0], dim=-1)

        result = ablate_head_previous(
            sample_pattern,
            head_idx=0,
            layer_idx=1,
            previous_layer_patterns=previous_patterns
        )

        # Should use previous layer's pattern for this head
        assert torch.allclose(result[0, 0, :, :], previous_patterns[0][0, 0, :, :])

    def test_ablate_head_previous_first_layer(self, sample_pattern):
        """Test previous ablation on first layer falls back to mean."""
        result = ablate_head_previous(
            sample_pattern,
            head_idx=0,
            layer_idx=0,
            previous_layer_patterns={}
        )

        # Should fall back to mean ablation
        ablated_head = result[0, 0, :, :]
        row_sums = ablated_head.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_apply_ablation_multiple_heads(self, sample_pattern):
        """Test applying ablation to multiple heads."""
        head_indices = [0, 3, 5]
        result = apply_ablation(
            sample_pattern,
            head_indices=head_indices,
            method="zero",
            layer_idx=0
        )

        # All specified heads should be ablated
        for head_idx in head_indices:
            ablated_head = result[0, head_idx, :, :]
            seq_k = ablated_head.shape[-1]
            expected_value = 1.0 / seq_k
            assert torch.allclose(ablated_head, torch.full_like(ablated_head, expected_value), atol=1e-6)

        # Other heads should be unchanged
        for head_idx in [1, 2, 4, 6, 7, 8, 9, 10, 11]:
            assert torch.allclose(result[0, head_idx, :, :], sample_pattern[0, head_idx, :, :])

    def test_apply_ablation_invalid_method(self, sample_pattern):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ablation method"):
            apply_ablation(
                sample_pattern,
                head_indices=[0],
                method="invalid_method",
                layer_idx=0
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
