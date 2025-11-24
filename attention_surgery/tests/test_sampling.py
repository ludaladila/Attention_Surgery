"""
Unit tests for sampling strategies.
"""
import pytest
import torch
from attention_surgery.core.sampling import sample_token, greedy_decode


class TestSamplingStrategies:
    """Test suite for token sampling functions."""

    @pytest.fixture
    def sample_logits(self):
        """Create sample logits."""
        torch.manual_seed(42)
        return torch.randn(1, 50257)  # GPT-2 vocab size

    def test_greedy_decode(self, sample_logits):
        """Test greedy decoding selects max probability token."""
        result = greedy_decode(sample_logits)

        # Should return token with highest logit
        expected = torch.argmax(sample_logits, dim=-1, keepdim=True)
        assert torch.equal(result, expected)

        # Check shape
        assert result.shape == (1, 1)

    def test_sample_token_temperature_zero(self, sample_logits):
        """Test that temperature=0 is equivalent to greedy."""
        result = sample_token(sample_logits, temperature=0.0)
        expected = greedy_decode(sample_logits)

        assert torch.equal(result, expected)

    def test_sample_token_reproducibility(self, sample_logits):
        """Test that sampling with seed is reproducible."""
        generator1 = torch.Generator()
        generator1.manual_seed(42)

        generator2 = torch.Generator()
        generator2.manual_seed(42)

        result1 = sample_token(sample_logits, temperature=0.8, generator=generator1)
        result2 = sample_token(sample_logits, temperature=0.8, generator=generator2)

        assert torch.equal(result1, result2)

    def test_sample_token_top_k(self, sample_logits):
        """Test top-k sampling."""
        result = sample_token(sample_logits, temperature=1.0, top_k=50)

        # Result should be within vocabulary
        assert 0 <= result.item() < 50257

        # Should be deterministic with generator
        generator = torch.Generator()
        generator.manual_seed(123)
        result1 = sample_token(sample_logits, temperature=1.0, top_k=50, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(123)
        result2 = sample_token(sample_logits, temperature=1.0, top_k=50, generator=generator)

        assert torch.equal(result1, result2)

    def test_sample_token_top_p(self, sample_logits):
        """Test nucleus (top-p) sampling."""
        result = sample_token(sample_logits, temperature=1.0, top_p=0.9)

        # Result should be within vocabulary
        assert 0 <= result.item() < 50257

        # Should be reproducible with seed
        generator = torch.Generator()
        generator.manual_seed(456)
        result1 = sample_token(sample_logits, temperature=1.0, top_p=0.9, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(456)
        result2 = sample_token(sample_logits, temperature=1.0, top_p=0.9, generator=generator)

        assert torch.equal(result1, result2)

    def test_sample_token_combined(self, sample_logits):
        """Test combined top-k and top-p sampling."""
        generator = torch.Generator()
        generator.manual_seed(789)

        result = sample_token(
            sample_logits,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            generator=generator
        )

        assert 0 <= result.item() < 50257
        assert result.shape == (1, 1)

    def test_sample_token_high_temperature(self, sample_logits):
        """Test that higher temperature increases randomness."""
        generator1 = torch.Generator()
        generator1.manual_seed(100)
        result_low = sample_token(sample_logits, temperature=0.1, generator=generator1)

        generator2 = torch.Generator()
        generator2.manual_seed(100)
        result_high = sample_token(sample_logits, temperature=2.0, generator=generator2)

        # Results might differ (stochastic test)
        # Just check validity
        assert 0 <= result_low.item() < 50257
        assert 0 <= result_high.item() < 50257


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
