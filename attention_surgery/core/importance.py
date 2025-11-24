"""
Importance scoring utilities for attention heads.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch

from .. import config
from .cache import AblationConfig, HeadId, RunResult

if TYPE_CHECKING:
    from .model_wrapper import ModelWrapper


HeadScores = Dict[HeadId, float]


def compute_importance_scores(
    model_wrapper: "ModelWrapper",
    run_result: RunResult,
    max_new_tokens: int,
    methods: Optional[Sequence[str]] = None,
    ablation_sample_ratio: float = 1.0,
    ablation_priority_layers: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, HeadScores]:
    """
    Compute multiple importance rankings for attention heads.

    Args:
        model_wrapper: Model wrapper instance
        run_result: Result from previous generation
        max_new_tokens: Maximum tokens to generate
        methods: List of methods to use ("gradient", "rollback", "ablation")
        ablation_sample_ratio: For ablation method, fraction of heads to evaluate (0.0-1.0)
        ablation_priority_layers: For ablation method, layers to evaluate first
        progress_callback: Optional callback(method_name, current, total) for progress

    Returns:
        Dictionary mapping method name to head scores
    """
    methods = methods or config.IMPORTANCE_METHODS
    scores: Dict[str, HeadScores] = {}

    token_ids = run_result.input_token_ids + run_result.output_token_ids_original

    head_outputs = None
    if any(method in methods for method in ("gradient", "rollback")):
        grad_scores, head_outputs = _gradient_scores(model_wrapper, token_ids)
        if "gradient" in methods:
            scores["gradient"] = grad_scores
        if "rollback" in methods and head_outputs:
            scores["rollback"] = _rollback_scores(head_outputs)

    if "ablation" in methods:
        # Create progress callback wrapper if provided
        ablation_progress = None
        if progress_callback:
            ablation_progress = lambda curr, total: progress_callback("ablation", curr, total)

        scores["ablation"] = _ablation_scores(
            model_wrapper=model_wrapper,
            prompt=run_result.prompt,
            baseline_logits=run_result.logits_original,
            max_new_tokens=max_new_tokens,
            sample_ratio=ablation_sample_ratio,
            priority_layers=ablation_priority_layers,
            progress_callback=ablation_progress,
        )

    return scores


def _gradient_scores(
    model_wrapper: "ModelWrapper", token_ids: List[int]
) -> Tuple[HeadScores, Dict[int, torch.Tensor]]:
    device = model_wrapper.device
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    pattern_store: Dict[int, torch.Tensor] = {}
    head_outputs: Dict[int, torch.Tensor] = {}

    hooks = []
    for layer in range(model_wrapper.model.cfg.n_layers):
        hook_name = f"blocks.{layer}.attn.hook_pattern"

        def pattern_hook(tensor, hook, *, layer_idx=layer):
            tensor.retain_grad()
            pattern_store[layer_idx] = tensor
            return tensor

        hooks.append((hook_name, pattern_hook))

        z_hook_name = f"blocks.{layer}.attn.hook_z"

        def z_hook(tensor, hook, *, layer_idx=layer):
            head_outputs[layer_idx] = tensor.detach()
            return tensor

        hooks.append((z_hook_name, z_hook))

    model_wrapper.model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        with model_wrapper.model.hooks(fwd_hooks=hooks):
            logits = model_wrapper.model(tokens)
            target_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            loss = -torch.mean(torch.max(target_log_probs, dim=-1).values)
        loss.backward()

    head_scores: HeadScores = {}
    for layer in range(model_wrapper.model.cfg.n_layers):
        grad_tensor = pattern_store.get(layer)
        if grad_tensor is None or grad_tensor.grad is None:
            continue
        grad = grad_tensor.grad
        if grad is None:
            continue
        grad_head = grad.detach().abs().mean(dim=(0, 2, 3))
        for head_idx in range(grad_head.shape[0]):
            head_scores[(layer, head_idx)] = float(grad_head[head_idx].item())

    model_wrapper.model.zero_grad(set_to_none=True)
    return head_scores, head_outputs


def _rollback_scores(head_outputs: Dict[int, torch.Tensor]) -> HeadScores:
    scores: HeadScores = {}
    for layer, tensor in head_outputs.items():
        if tensor is None or tensor.ndim < 4:
            continue
        z = tensor[0]  # [seq, heads, d_head]
        if z.shape[0] == 0:
            continue
        final_vec = z[-1]  # [heads, d_head]
        norms = torch.linalg.norm(final_vec, dim=-1)
        for head_idx in range(norms.shape[0]):
            scores[(layer, head_idx)] = float(norms[head_idx].item())
    return scores


def _ablation_scores(
    model_wrapper: "ModelWrapper",
    prompt: str,
    baseline_logits: np.ndarray,
    max_new_tokens: int,
    sample_ratio: float = 1.0,
    priority_layers: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> HeadScores:
    """
    Compute ablation-based importance by measuring KL divergence.

    Args:
        model_wrapper: Model wrapper instance
        prompt: Input prompt
        baseline_logits: Original logits for comparison
        max_new_tokens: Maximum new tokens to generate
        sample_ratio: Fraction of heads to evaluate (0.0-1.0). 1.0 = all heads.
                      Useful for faster approximation.
        priority_layers: List of layers to evaluate first (e.g., [10, 11] for last 2 layers)
        progress_callback: Optional callback function(current, total) for progress tracking

    Returns:
        Dictionary mapping (layer, head) to KL divergence scores

    Performance notes:
        - Full evaluation: 144 forward passes (~2-5 min on GPU)
        - sample_ratio=0.5: 72 forward passes (~1-2 min on GPU)
        - priority_layers=[10,11]: 24 forward passes (~20-30 sec on GPU)
    """
    baseline = _ensure_seq(baseline_logits)
    scores: HeadScores = {}

    # Build list of heads to evaluate
    all_heads: List[HeadId] = []

    if priority_layers:
        # Add priority layers first
        for layer in priority_layers:
            if 0 <= layer < config.NUM_LAYERS:
                for head_idx in range(config.NUM_HEADS):
                    all_heads.append((layer, head_idx))
        # Add remaining layers
        for layer in range(config.NUM_LAYERS):
            if layer not in priority_layers:
                for head_idx in range(config.NUM_HEADS):
                    all_heads.append((layer, head_idx))
    else:
        # Default order: layer by layer
        for layer in range(config.NUM_LAYERS):
            for head_idx in range(config.NUM_HEADS):
                all_heads.append((layer, head_idx))

    # Sample heads if requested
    if 0.0 < sample_ratio < 1.0:
        import random
        num_to_sample = max(1, int(len(all_heads) * sample_ratio))
        all_heads = random.sample(all_heads, num_to_sample)
    elif sample_ratio <= 0.0:
        return scores  # Return empty if sample_ratio is 0

    total_heads = len(all_heads)

    # Evaluate each head
    for idx, (layer, head_idx) in enumerate(all_heads):
        config_obj = AblationConfig(
            method="zero", heads=[(layer, head_idx)]
        )
        _, logits, _ = model_wrapper.run_forward(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            ablation_config=config_obj,
            collect_cache=False,
        )
        ablated = _ensure_seq(
            logits.detach().cpu().numpy().astype(np.float32)
        )
        scores[(layer, head_idx)] = _average_kl(baseline, ablated)

        # Progress callback
        if progress_callback:
            progress_callback(idx + 1, total_heads)

    return scores


def _ensure_seq(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _average_kl(base: np.ndarray, other: np.ndarray) -> float:
    seq_len = min(base.shape[0], other.shape[0])
    if seq_len == 0:
        return 0.0
    p = _softmax_np(base[:seq_len])
    q = _softmax_np(other[:seq_len])
    kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)
    return float(np.mean(kl))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits - max_logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def aggregate_top_heads(
    scores: Dict[str, HeadScores],
    limit: int = config.DEFAULT_TOP_K_HEADS,
) -> List[HeadId]:
    if not scores:
        return []
    accumulator: Dict[HeadId, List[float]] = defaultdict(list)
    for method_scores in scores.values():
        for head, value in method_scores.items():
            accumulator[head].append(value)
    ranked = sorted(
        accumulator.items(),
        key=lambda item: mean(item[1]),
        reverse=True,
    )
    return [head for head, _ in ranked[:limit]]

