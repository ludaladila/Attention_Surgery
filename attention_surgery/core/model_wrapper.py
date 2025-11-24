"""
Model wrapper providing a higher-level interface around HookedTransformer.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .. import config
from ..utils.seed import set_global_seed
from ..utils.text import TextProcessor
from .cache import AblationConfig, CacheObject, HeadId, RunResult
from .hooks import HookManager
from .sampling import sample_token, greedy_decode


class ModelWrapper:
    """Encapsulates model loading, tokenization, generation and caching."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: Optional[str] = None,
        seed: int = config.DEFAULT_SEED,
        tokenizer_name: Optional[str] = None,
    ):
        set_global_seed(seed)
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
        )
        self.text_processor = TextProcessor(
            model_name=tokenizer_name or "gpt2"
        )
        self.hook_manager = HookManager()
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into token ids tensor."""
        return self.text_processor.encode(text).to(self.device)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode token ids tensor back into text."""
        return self.text_processor.decode(ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
        ablation_config: Optional[AblationConfig] = None,
        collect_cache: bool = True,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> RunResult:
        """
        Run original and optional ablated generation, returning a RunResult.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            ablation_config: Optional ablation configuration
            collect_cache: Whether to collect attention patterns and activations
            temperature: Sampling temperature. 0.0 = greedy, higher = more random
            top_k: If provided, sample from top-k tokens
            top_p: If provided, use nucleus sampling (top-p)
            seed: Random seed for sampling (optional)

        Returns:
            RunResult containing generated text, logits, and analysis data
        """
        prompt_ids = self.encode(prompt)

        original_cache = CacheObject() if collect_cache else None
        original_tokens, original_logits = self._generate_once(
            prompt_ids,
            max_new_tokens,
            cache=original_cache,
            ablation_config=None,
            collect_cache=collect_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )

        run_ablation = (
            ablation_config is not None and len(ablation_config.heads) > 0
        )

        if run_ablation:
            ablated_cache = CacheObject() if collect_cache else None
            ablated_tokens, ablated_logits = self._generate_once(
                prompt_ids,
                max_new_tokens,
                cache=ablated_cache,
                ablation_config=ablation_config,
                collect_cache=collect_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            )
        else:
            ablated_tokens = original_tokens.clone()
            ablated_logits = original_logits.clone()
            ablated_cache = original_cache

        input_len = prompt_ids.shape[-1]
        input_token_ids = prompt_ids.squeeze(0).tolist()

        output_ids_original = (
            original_tokens.squeeze(0).tolist()[input_len:]
        )
        output_ids_ablated = ablated_tokens.squeeze(0).tolist()[input_len:]

        run_result = RunResult(
            prompt=prompt,
            generated_text_original=self.decode(original_tokens),
            generated_text_ablated=self.decode(ablated_tokens),
            input_tokens=self.text_processor.convert_ids_to_tokens(
                input_token_ids
            ),
            output_tokens_original=self.text_processor.convert_ids_to_tokens(
                output_ids_original
            ),
            output_tokens_ablated=self.text_processor.convert_ids_to_tokens(
                output_ids_ablated
            ),
            input_token_ids=input_token_ids,
            output_token_ids_original=output_ids_original,
            output_token_ids_ablated=output_ids_ablated,
            logits_original=original_logits.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            logits_ablated=ablated_logits.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            activations_original=self._extract_activations(original_cache)
            if original_cache
            else {},
            activations_ablated=self._extract_activations(ablated_cache)
            if ablated_cache
            else {},
            attention_patterns_original=self._extract_head_patterns(
                original_cache
            )
            if original_cache
            else {},
            attention_patterns_ablated=self._extract_head_patterns(
                ablated_cache
            )
            if ablated_cache
            else {},
            metrics={},
            importance_scores={},
        )

        return run_result

    def _generate_once(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        cache: Optional[CacheObject],
        ablation_config: Optional[AblationConfig],
        collect_cache: bool,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens with flexible sampling strategies.

        Args:
            temperature: 0.0 = greedy, higher = more random
            top_k: Sample from top-k tokens
            top_p: Nucleus sampling threshold
            seed: Random seed for sampling
        """
        tokens = prompt_ids.clone().to(self.device)
        hooks = self.hook_manager.build_forward_hooks(
            cache=cache if collect_cache else None,
            ablation_config=ablation_config,
        )

        # Create generator for reproducible sampling
        generator = None
        if seed is not None and (temperature > 0 or top_k or top_p):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                with self.model.hooks(fwd_hooks=hooks):
                    logits = self.model(tokens)

                # Sample next token using specified strategy
                if temperature == 0.0 and top_k is None and top_p is None:
                    # Greedy decoding
                    next_token = greedy_decode(logits[:, -1, :])
                else:
                    # Sampling-based generation
                    next_token = sample_token(
                        logits[:, -1, :],
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                    )

                tokens = torch.cat([tokens, next_token], dim=-1)

        # Run one final forward pass to capture logits for the entire sequence
        with torch.no_grad():
            with self.model.hooks(fwd_hooks=hooks):
                final_logits = self.model(tokens)

        return tokens, final_logits

    def run_forward(
        self,
        prompt: str,
        max_new_tokens: int,
        ablation_config: Optional[AblationConfig] = None,
        collect_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[CacheObject]]:
        prompt_ids = self.encode(prompt)
        cache = CacheObject() if collect_cache else None
        tokens, logits = self._generate_once(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache=cache,
            ablation_config=ablation_config,
            collect_cache=collect_cache,
        )
        return tokens, logits, cache

    def _extract_head_patterns(
        self, cache: Optional[CacheObject]
    ) -> Dict[HeadId, np.ndarray]:
        if cache is None:
            return {}

        head_map: Dict[HeadId, np.ndarray] = {}
        for layer_idx, pattern in cache.attention_patterns.items():
            tensor = pattern.detach().cpu().numpy()
            for head_idx in range(tensor.shape[1]):
                head_map[(layer_idx, head_idx)] = tensor[0, head_idx]
        return head_map

    def _extract_activations(
        self, cache: Optional[CacheObject]
    ) -> Dict[str, np.ndarray]:
        if cache is None:
            return {}
        return {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in cache.activations.items()
        }

