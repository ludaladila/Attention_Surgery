import torch
import numpy as np
from typing import List, Dict
from transformer_lens import HookedTransformer

def compute_layer_impact(
    model: HookedTransformer, 
    cache, 
    last_token_id: int
) -> List[Dict[str, float]]:
    """
    Computes the confidence (probability) of the final generated token 
    at each layer's residual stream accumulation.
    
    Args:
        model: The HookedTransformer model
        cache: The ActivationCache from the run (containing 'blocks.{i}.hook_resid_post')
        last_token_id: The ID of the token that was actually generated (or target)
        
    Returns:
        List of dicts: [{"layer_name": "L0", "impact_score": 0.123}, ...]
    """
    impacts = []
    
    # Logit Lens: Apply unembedding to residual stream at each layer
    # We only care about the LAST position in the sequence (prediction of the next token)
    
    for layer in range(model.cfg.n_layers):
        # Get residual stream at end of layer: [batch, seq_len, d_model]
        resid_name = f"blocks.{layer}.hook_resid_post"
        if resid_name not in cache:
            continue
            
        resid = cache[resid_name]
        # Take the last token position from the batch (0)
        # shape: [d_model]
        final_resid = resid[0, -1, :]
        
        # Apply LayerNorm (final) + Unembed
        # Note: transformer_lens usually applies ln_final before unembed in the full forward pass.
        # For strict logit lens, we should apply ln_final.
        
        # Apply Final Layer Norm
        scaled_resid = model.ln_final(final_resid.unsqueeze(0)) # [1, d_model]
        
        # Apply Unembed
        logits = model.unembed(scaled_resid) # [1, vocab_size]
        
        # Calculate Prob of the target token
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[0, last_token_id].item()
        
        impacts.append({
            "layer_name": f"L{layer}",
            "impact_score": float(target_prob)
        })
        
    return impacts

