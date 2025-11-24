"""
Hook registration utilities for attention ablation and caching.
"""
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .ablation import apply_ablation
from .cache import AblationConfig, CacheObject, HeadId
from .. import config


class HookManager:
    """
    Builds forward hooks used during inference to
    1) capture attention patterns for later analysis
    2) optionally modify specific heads according to the ablation config
    """

    def __init__(self):
        self.num_layers = config.NUM_LAYERS

    @staticmethod
    def _group_heads(ablation_config: AblationConfig) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = defaultdict(list)
        for layer_idx, head_idx in ablation_config.heads:
            grouped[layer_idx].append(head_idx)
        return grouped

    def build_forward_hooks(
        self,
        cache: Optional[CacheObject] = None,
        ablation_config: Optional[AblationConfig] = None,
    ) -> List[Tuple[str, Callable]]:
        """
        Construct forward hooks for TransformerLens.

        Args:
            cache: CacheObject to store intermediate tensors.
            ablation_config: Configuration describing which heads to modify.

        Returns:
            List of (hook_name, hook_fn) pairs to be used with HookedTransformer.
        """
        hooks: List[Tuple[str, Callable]] = []
        layer_head_map = (
            self._group_heads(ablation_config) if ablation_config else {}
        )
        previous_patterns: Dict[int, torch.Tensor] = {}

        for layer_idx in range(self.num_layers):
            needs_cache = cache is not None
            needs_ablation = layer_idx in layer_head_map

            if not needs_cache and not needs_ablation:
                continue

            head_indices = layer_head_map.get(layer_idx, [])
            hook_name = f"blocks.{layer_idx}.attn.hook_pattern"

            def hook_fn(pattern, hook, *, layer=layer_idx, heads=head_indices):
                # Store a clone for previous-layer referencing
                previous_patterns[layer] = pattern.detach().clone()
                modified = pattern

                if heads and ablation_config:
                    modified = apply_ablation(
                        pattern=modified,
                        head_indices=heads,
                        method=ablation_config.method,
                        layer_idx=layer,
                        previous_layer_patterns=previous_patterns,
                        seed=ablation_config.random_seed,
                    )

                if cache is not None:
                    cache.store_attention(layer, modified)

                return modified

            hooks.append((hook_name, hook_fn))

            if cache is not None:
                resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"

                def resid_hook(resid, hook, *, layer=layer_idx):
                    cache.store_activation(f"resid_post_{layer}", resid)
                    return resid

                hooks.append((resid_hook_name, resid_hook))

        return hooks

