"""
Unit tests for metrics computation.
"""
import pytest
import numpy as np
from attention_surgery.core.metrics import (
    _kl_divergence,
    _top1_changed_ratio,
    _perplexity,
    compute_metrics,
)


class TestMetrics:
    """Test suite for metrics functions."""

    @pytest.fixture
    def sample_logits(self):
        """Create sample logits."""
        np.random.seed(42)
        return np.random.randn(10, 50257).astype(np.float32)

    def test_kl_divergence_identical(self, sample_logits):
        """Test KL divergence is 0 for identical distributions."""
        kl = _kl_divergence(sample_logits, sample_logits)
        assert kl < 1e-5  # Should be very close to 0

    def test_kl_divergence_positive(self, sample_logits):
        """Test KL divergence is positive for different distributions."""
        logits_b = sample_logits + np.random.randn(*sample_logits.shape) * 0.5
        kl = _kl_divergence(sample_logits, logits_b)
        assert kl > 0  # KL divergence is always non-negative

    def test_top1_changed_ratio_identical(self, sample_logits):
        """Test top-1 change ratio is 0 for identical logits."""
        ratio = _top1_changed_ratio(sample_logits, sample_logits)
        assert ratio == 0.0

    def test_top1_changed_ratio_range(self, sample_logits):
        """Test top-1 change ratio is in [0, 1]."""
        logits_b = sample_logits + np.random.randn(*sample_logits.shape)
        ratio = _top1_changed_ratio(sample_logits, logits_b)
        assert 0.0 <= ratio <= 1.0

    def test_perplexity_computation(self, sample_logits):
        """Test perplexity computation."""
        token_ids = np.random.randint(0, 50257, size=11).tolist()
        ppl = _perplexity(sample_logits, token_ids)

        # Perplexity should be positive
        assert ppl > 0
        # Should be a finite number
        assert not np.isnan(ppl)
        assert not np.isinf(ppl)

    def test_compute_metrics_complete(self):
        """Test that compute_metrics returns all expected metrics."""
        np.random.seed(42)
        logits_orig = np.random.randn(10, 50257).astype(np.float32)
        logits_abl = logits_orig + np.random.randn(10, 50257).astype(np.float32) * 0.1

        input_ids = list(range(5))
        output_ids_orig = list(range(5, 10))
        output_ids_abl = list(range(5, 10))

        metrics = compute_metrics(
            logits_original=logits_orig,
            logits_ablated=logits_abl,
            input_token_ids=input_ids,
            output_token_ids_original=output_ids_orig,
            output_token_ids_ablated=output_ids_abl,
            attention_original={},
            attention_ablated={},
            activations_original={},
            activations_ablated={},
        )

        # Check all expected metrics are present
        expected_keys = [
            "kl_div",
            "top1_changed_ratio",
            "attention_diff",
            "l2_activation",
            "perplexity_original",
            "perplexity_ablated",
            "perplexity_delta",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

        # Check metrics are valid numbers
        for key, value in metrics.items():
            if not np.isnan(value):
                assert isinstance(value, (int, float)), f"{key} is not a number"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
