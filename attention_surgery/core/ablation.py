"""
Attention ablation methods.
"""
from typing import Dict, Optional
import torch
import torch.nn.functional as F


def ablate_head_zero(
    pattern: torch.Tensor,
    head_idx: int,
    renormalize: bool = False, # Changed default to False for Hard Zero
    uniform_value: float = 0.0 # Changed to 0.0
) -> torch.Tensor:
    """
    Zero ablation: Set attention pattern for specified head to zero.
    
    Args:
        pattern: Attention pattern tensor [batch, heads, seq_q, seq_k]
        head_idx: Index of head to ablate
        renormalize: If True, replace with uniform distribution. 
                     If False (default), set to pure zero (Hard Ablation).
    """
    pattern = pattern.clone()
    
    if renormalize:
        # Uniform distribution (Soft Ablation)
        batch, num_heads, seq_q, seq_k = pattern.shape
        uniform_pattern = torch.ones(
            batch, seq_q, seq_k,
            device=pattern.device,
            dtype=pattern.dtype
        ) / seq_k
        pattern[:, head_idx, :, :] = uniform_pattern
    else:
        # Pure Zero (Hard Ablation)
        # This effectively turns off the head's contribution
        pattern[:, head_idx, :, :] = 0.0
        
    return pattern


def ablate_head_mean(pattern: torch.Tensor, head_idx: int) -> torch.Tensor:
    """
    Mean ablation: Replace head's pattern with mean pattern.

    Args:
        pattern: Attention pattern tensor [batch, heads, seq_q, seq_k]
        head_idx: Index of head to ablate

    Returns:
        Modified attention pattern
    """
    pattern = pattern.clone()

    # Compute mean across sequence dimensions
    mean_pattern = pattern[:, head_idx, :, :].mean()

    # Create uniform pattern with this mean value
    batch, num_heads, seq_q, seq_k = pattern.shape
    uniform_pattern = torch.full(
        (batch, seq_q, seq_k),
        mean_pattern.item(),
        device=pattern.device,
        dtype=pattern.dtype
    )

    # Replace the head's pattern
    pattern[:, head_idx, :, :] = uniform_pattern

    # Renormalize to ensure it's a valid attention distribution
    pattern[:, head_idx, :, :] = F.softmax(
        torch.ones_like(pattern[:, head_idx, :, :]) * mean_pattern,
        dim=-1
    )

    return pattern


def ablate_head_random(
    pattern: torch.Tensor,
    head_idx: int,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Random ablation: Replace head's pattern with random attention pattern.

    Args:
        pattern: Attention pattern tensor [batch, heads, seq_q, seq_k]
        head_idx: Index of head to ablate
        seed: Random seed for reproducibility

    Returns:
        Modified attention pattern
    """
    pattern = pattern.clone()
    batch, num_heads, seq_q, seq_k = pattern.shape

    # Set seed if provided
    if seed is not None:
        generator = torch.Generator(device=pattern.device)
        generator.manual_seed(seed)
    else:
        generator = None

    # Generate random logits and apply softmax to get valid attention distribution
    random_logits = torch.randn(
        batch, seq_q, seq_k,
        device=pattern.device,
        dtype=pattern.dtype,
        generator=generator
    )
    random_pattern = F.softmax(random_logits, dim=-1)

    # Replace the head's pattern
    pattern[:, head_idx, :, :] = random_pattern

    return pattern


def ablate_head_previous(
    pattern: torch.Tensor,
    head_idx: int,
    layer_idx: int,
    previous_layer_patterns: Dict[int, torch.Tensor]
) -> torch.Tensor:
    """
    Previous layer ablation: Replace with same head from previous layer.

    Args:
        pattern: Attention pattern tensor [batch, heads, seq_q, seq_k]
        head_idx: Index of head to ablate
        layer_idx: Current layer index
        previous_layer_patterns: Cache of previous layer patterns

    Returns:
        Modified attention pattern
    """
    pattern = pattern.clone()

    # If this is layer 0 or no previous layer available, use mean ablation
    if layer_idx == 0 or (layer_idx - 1) not in previous_layer_patterns:
        return ablate_head_mean(pattern, head_idx)

    # Get previous layer's pattern
    prev_pattern = previous_layer_patterns[layer_idx - 1]

    # Handle potential shape mismatch (sequence length might differ)
    if prev_pattern.shape == pattern.shape:
        # Direct copy if shapes match
        pattern[:, head_idx, :, :] = prev_pattern[:, head_idx, :, :].clone()
    else:
        # If shapes don't match, interpolate or use mean ablation
        # For simplicity, fall back to mean ablation
        pattern = ablate_head_mean(pattern, head_idx)

    return pattern


def apply_ablation(
    pattern: torch.Tensor,
    head_indices: list,
    method: str,
    layer_idx: int,
    previous_layer_patterns: Optional[Dict[int, torch.Tensor]] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Apply ablation to multiple heads using specified method.

    Args:
        pattern: Attention pattern tensor [batch, heads, seq_q, seq_k]
        head_indices: List of head indices to ablate
        method: Ablation method ("zero", "mean", "random", "previous")
        layer_idx: Current layer index
        previous_layer_patterns: Cache for previous layer method
        seed: Random seed for random method

    Returns:
        Modified attention pattern
    """
    for head_idx in head_indices:
        if method == "zero":
            pattern = ablate_head_zero(pattern, head_idx)
        elif method == "mean":
            pattern = ablate_head_mean(pattern, head_idx)
        elif method == "random":
            pattern = ablate_head_random(pattern, head_idx, seed)
        elif method == "previous":
            pattern = ablate_head_previous(
                pattern, head_idx, layer_idx, previous_layer_patterns or {}
            )
        else:
            raise ValueError(f"Unknown ablation method: {method}")

    return pattern
