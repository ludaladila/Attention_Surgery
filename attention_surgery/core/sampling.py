"""
Sampling strategies for text generation.
"""
from typing import Optional
import torch
import torch.nn.functional as F


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample a token from logits using various strategies.

    Args:
        logits: Logits tensor of shape [batch, vocab_size] or [vocab_size]
        temperature: Temperature for sampling. Higher = more random.
                     temperature=0 is equivalent to greedy decoding.
        top_k: If provided, only sample from top-k tokens.
        top_p: If provided, use nucleus sampling (sample from smallest set
               of tokens whose cumulative probability >= top_p).
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Sampled token indices of shape [batch, 1] or [1]

    Examples:
        >>> logits = torch.randn(50257)  # GPT-2 vocab size
        >>> token = sample_token(logits, temperature=0.8, top_k=50)
        >>> token = sample_token(logits, temperature=0.9, top_p=0.95)
        >>> token = sample_token(logits, temperature=0.0)  # Greedy
    """
    # Handle shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # [1, vocab_size]

    # Greedy decoding (temperature=0)
    if temperature == 0.0 or temperature < 1e-8:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k is not None and top_k > 0:
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
        # Create mask for top-k
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(-1, top_k_indices, top_k_values)
        logits = logits_filtered

    # Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        # Keep at least one token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False  # Keep at least the top token

        # Scatter to original positions
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1, generator=generator)

    return sampled_token


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding: select token with highest probability.

    Args:
        logits: Logits tensor of shape [batch, vocab_size]

    Returns:
        Token indices of shape [batch, 1]
    """
    return torch.argmax(logits, dim=-1, keepdim=True)
