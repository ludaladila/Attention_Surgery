"""
Data structures and caching for attention surgery.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any
from datetime import datetime
import numpy as np
import torch

HeadId = Tuple[int, int]  # (layer_idx, head_idx)


@dataclass
class AblationConfig:
    """Configuration for ablation surgery."""
    method: Literal["zero", "mean", "random", "previous"]
    heads: List[HeadId]
    random_seed: Optional[int] = None


@dataclass
class RunResult:
    """Results from a single surgery run."""
    prompt: str
    generated_text_original: str
    generated_text_ablated: str

    # Tokens
    input_tokens: List[str]
    output_tokens_original: List[str]
    output_tokens_ablated: List[str]
    input_token_ids: List[int]
    output_token_ids_original: List[int]
    output_token_ids_ablated: List[int]

    # Logits
    logits_original: np.ndarray
    logits_ablated: np.ndarray

    # Activations and attention patterns
    activations_original: Dict[str, np.ndarray] = field(default_factory=dict)
    activations_ablated: Dict[str, np.ndarray] = field(default_factory=dict)
    attention_patterns_original: Dict[HeadId, np.ndarray] = field(default_factory=dict)
    attention_patterns_ablated: Dict[HeadId, np.ndarray] = field(default_factory=dict)

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Importance scores
    importance_scores: Dict[str, Dict[HeadId, float]] = field(default_factory=dict)


@dataclass
class SurgeryHistoryItem:
    """Single item in surgery history."""
    timestamp: datetime
    ablation_config: AblationConfig
    top_k_heads: List[HeadId]
    run_result_summary: Dict[str, Any]


@dataclass
class AblationState:
    """Global state for ablation tracking."""
    active_config: Optional[AblationConfig] = None
    history: List[SurgeryHistoryItem] = field(default_factory=list)


class CacheObject:
    """
    Cache for storing intermediate results during forward pass.
    Used for metric computation and rollback analysis.
    """

    def __init__(self):
        self.attention_patterns: Dict[int, torch.Tensor] = {}  # layer_idx -> pattern
        self.activations: Dict[str, torch.Tensor] = {}  # key -> activation
        self.hook_handles: List = []  # Store hook handles for cleanup

    def clear(self):
        """Clear all cached data."""
        self.attention_patterns.clear()
        self.activations.clear()
        self.hook_handles.clear()

    def store_attention(self, layer_idx: int, pattern: torch.Tensor):
        """Store attention pattern for a layer."""
        self.attention_patterns[layer_idx] = pattern.detach().clone()

    def store_activation(self, key: str, activation: torch.Tensor):
        """Store an activation tensor."""
        self.activations[key] = activation.detach().clone()

    def get_attention(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Retrieve attention pattern for a layer."""
        return self.attention_patterns.get(layer_idx)

    def get_activation(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve an activation tensor."""
        return self.activations.get(key)
