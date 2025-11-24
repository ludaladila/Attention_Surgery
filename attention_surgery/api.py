import sys
import os
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from attention_surgery.core.model_wrapper import ModelWrapper
from attention_surgery.core.cache import AblationConfig, HeadId, RunResult
from attention_surgery.core.importance import compute_importance_scores
from attention_surgery.core.metrics import compute_metrics
from attention_surgery.config import DEFAULT_MAX_NEW_TOKENS
from attention_surgery.core.logit_lens import compute_layer_impact

# --- Data Models ---

class SurgeryRequest(BaseModel):
    prompt: str
    ablation_mask: List[List[bool]]
    method: str = "zero"
    importance_method: str = "gradient"
    temperature: float = 1.0
    max_new_tokens: int = 60 # Enough for completing prompt + 1 sentence
    seed: int = 42 # Fixed seed for reproducibility

class TopKItem(BaseModel):
    token: str
    prob: float

class TokenData(BaseModel):
    id: int
    text: str
    prob: float
    isPrompt: bool
    topK: List[TopKItem] = []

class MetricsData(BaseModel):
    kl_div: float
    perplexity_delta: float
    top1_changed_ratio: float
    l2_diff: float

class ImportanceScoresData(BaseModel):
    gradient_matrix: List[List[float]]

class LayerImpactData(BaseModel):
    layer_name: str
    impact_score: float

class GenerationData(BaseModel):
    control_tokens: List[TokenData]
    ablated_tokens: List[TokenData]

class SurgeryResponse(BaseModel):
    metrics: MetricsData
    generation: GenerationData
    importance_scores: ImportanceScoresData
    layer_impact: List[LayerImpactData]

# --- Global State ---
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading GPT-2 model...")
    models["wrapper"] = ModelWrapper()
    print("Model loaded.")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def clean_token_text(text: str) -> str:
    if not text: return ""
    return text.encode('utf-8', 'replace').decode('utf-8')

def logits_to_topk(logits: np.ndarray, tokenizer, temperature: float = 1.0, top_k: int = 5) -> List[TopKItem]:
    logits = np.array(logits, dtype=np.float32)
    if temperature > 0:
        logits = logits / temperature
    
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)
    
    top_indices = np.argsort(probs)[-top_k:][::-1]
    items = []
    for idx in top_indices:
        token_str = tokenizer.decode([idx])
        items.append(TopKItem(token=clean_token_text(token_str), prob=float(probs[idx])))
    return items

def process_sequence_robust(
    token_ids: List[int],
    input_ids: List[int], 
    logits: np.ndarray, 
    tokenizer,
    temperature: float
) -> List[TokenData]:
    """
    Robustly process sequence.
    token_ids: The full sequence of IDs (Prompt + Generated)
    input_ids: The IDs of the prompt
    logits: The logits for the full sequence (usually)
    """
    result_tokens: List[TokenData] = []
    
    if logits.ndim == 3:
        logits = logits[0]
    
    input_len = len(input_ids)
    
    # 1. Process Prompt
    for i, tid in enumerate(input_ids):
        txt = tokenizer.decode([tid])
        result_tokens.append(TokenData(
            id=i,
            text=clean_token_text(txt),
            prob=1.0,
            isPrompt=True,
            topK=[]
        ))
        
    # 2. Process Generated
    if len(token_ids) >= input_len and token_ids[:input_len] == input_ids:
        generated_ids = token_ids[input_len:]
    else:
        generated_ids = token_ids
        
    gen_len = len(generated_ids)
    
    # print(f"DEBUG: InputLen: {input_len}, GenLen: {gen_len}, LogitsShape: {logits.shape}")
    
    for i in range(gen_len):
        token_idx = input_len + i
        logit_idx = token_idx - 1
        
        if logit_idx >= len(logits): 
            break
        
        tid = generated_ids[i]
        txt = tokenizer.decode([tid])
        logit_vec = logits[logit_idx]
        
        # Softmax
        vec_safe = np.array(logit_vec, dtype=np.float32)
        if temperature > 0:
            vec_safe = vec_safe / temperature
        shifted = vec_safe - np.max(vec_safe)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps)
        
        actual_prob = float(probs[tid])
        
        top_k = logits_to_topk(logit_vec, tokenizer, temperature)
        
        result_tokens.append(TokenData(
            id=token_idx,
            text=clean_token_text(txt),
            prob=actual_prob,
            isPrompt=False,
            topK=top_k
        ))
        
    return result_tokens

def truncate_to_sentences(tokens: List[TokenData], min_sentences: int = 2) -> List[TokenData]:
    """
    Smart truncation:
    1. Try to find `min_sentences` (2).
    2. If not found, try to find at least 1 sentence.
    3. If even 1 sentence is not finished, return everything (best effort).
    """
    prompt_tokens = [t for t in tokens if t.isPrompt]
    gen_tokens = [t for t in tokens if not t.isPrompt]
    
    if not gen_tokens:
        return tokens
        
    sentence_endings = {'.', '!', '?', '\n'}
    
    end_indices = []
    for i, t in enumerate(gen_tokens):
        if any(char in sentence_endings for char in t.text):
            end_indices.append(i + 1)
    
    cut_idx = len(gen_tokens) # Default: keep all
    
    if len(end_indices) >= min_sentences:
        # Found enough sentences, cut at the requested count
        cut_idx = end_indices[min_sentences - 1]
    elif len(end_indices) > 0:
        # Found at least one sentence (but less than requested), keep what we have
        # This avoids showing a half-finished second sentence
        cut_idx = end_indices[-1]
    
    # If no sentences found, we return everything (gen_tokens[:len])
    
    return prompt_tokens + gen_tokens[:cut_idx]

def format_importance_matrix(scores: Dict[HeadId, float], n_layers=12, n_heads=12) -> List[List[float]]:
    matrix = [[0.0] * n_heads for _ in range(n_layers)]
    for (l, h), score in scores.items():
        if 0 <= l < n_layers and 0 <= h < n_heads:
            matrix[l][h] = float(score)
    return matrix

# --- Endpoints ---

@app.post("/api/surgery", response_model=SurgeryResponse)
async def run_surgery(req: SurgeryRequest):
    wrapper: ModelWrapper = models["wrapper"]
    
    heads_to_ablate = []
    for l in range(len(req.ablation_mask)):
        for h in range(len(req.ablation_mask[l])):
            if req.ablation_mask[l][h]:
                heads_to_ablate.append((l, h))
    
    ablation_config = AblationConfig(
        method=req.method,
        heads=heads_to_ablate,
        random_seed=42
    ) if heads_to_ablate else None

    # Run Generation
    result: RunResult = wrapper.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        ablation_config=ablation_config,
        collect_cache=True,
        temperature=req.temperature,
        top_k=50 if req.temperature > 0 else None,
        seed=req.seed # Pass the seed to ensure Control and Ablated use same RNG
    )
    
    # Metrics
    raw_metrics = compute_metrics(
        result.logits_original, result.logits_ablated,
        result.input_token_ids, result.output_token_ids_original, result.output_token_ids_ablated,
        result.attention_patterns_original, result.attention_patterns_ablated,
        result.activations_original, result.activations_ablated
    )
    
    l2_diff = raw_metrics.get('activation_l2', 0.0)
    
    if l2_diff == 0.0 and result.activations_original and result.activations_ablated:
        last_layer = wrapper.model.cfg.n_layers - 1
        key = f"resid_post_{last_layer}"
        if key in result.activations_original and key in result.activations_ablated:
            act_orig = result.activations_original[key]
            act_abl = result.activations_ablated[key]
            input_len = len(result.input_token_ids)
            idx_to_compare = input_len - 1 
            if act_orig.shape[1] > idx_to_compare and act_abl.shape[1] > idx_to_compare:
                vec_orig = act_orig[0, idx_to_compare, :]
                vec_abl = act_abl[0, idx_to_compare, :]
                l2_val = np.linalg.norm(vec_orig - vec_abl)
                l2_diff = float(l2_val)

    metrics_data = MetricsData(
        kl_div=raw_metrics.get('kl_div', 0.0),
        perplexity_delta=raw_metrics.get('perplexity_delta', 0.0),
        top1_changed_ratio=raw_metrics.get('top1_changed_ratio', 0.0),
        l2_diff=l2_diff
    )
    
    input_len = len(result.input_token_ids)
    
    control_tokens = process_sequence_robust(
        result.output_token_ids_original,
        result.input_token_ids, 
        result.logits_original,
        wrapper.text_processor.tokenizer,
        req.temperature
    )
    control_tokens = truncate_to_sentences(control_tokens)
    
    ablated_tokens = process_sequence_robust(
        result.output_token_ids_ablated,
        result.input_token_ids, 
        result.logits_ablated,
        wrapper.text_processor.tokenizer,
        req.temperature
    )
    ablated_tokens = truncate_to_sentences(ablated_tokens)
    
    imp_scores_raw = compute_importance_scores(
        model_wrapper=wrapper,
        run_result=result,
        max_new_tokens=req.max_new_tokens,
        methods=[req.importance_method]
    )
    scores_dict = imp_scores_raw.get(req.importance_method, {})
    importance_matrix = format_importance_matrix(scores_dict)
    
    # Layer Impact (Logit Lens)
    layer_impacts = []
    
    # We analyze the LAST token generated in the sequence.
    # Logic corrected: Use -2 to predict -1.
    
    if len(result.output_token_ids_ablated) > input_len:
        last_token_id = result.output_token_ids_ablated[-1]
        
        if result.activations_ablated:
            for layer in range(wrapper.model.cfg.n_layers):
                key = f"resid_post_{layer}"
                if key in result.activations_ablated:
                    resid_np = result.activations_ablated[key]
                    # Shape: (1, seq_len, d_model)
                    # Use -2 to get the state that predicted the last token (-1)
                    # Ensure seq_len is enough
                    if resid_np.shape[1] >= 2:
                        final_resid_np = resid_np[0, -2, :] 
                        
                        final_resid = torch.tensor(final_resid_np, device=wrapper.device)
                        with torch.no_grad():
                            scaled = wrapper.model.ln_final(final_resid.unsqueeze(0))
                            logits = wrapper.model.unembed(scaled)
                            probs = torch.softmax(logits, dim=-1)
                            target_prob = probs[0, last_token_id].item()
                        
                        layer_impacts.append(LayerImpactData(
                            layer_name=f"L{layer}",
                            impact_score=target_prob
                        ))
    
    return SurgeryResponse(
        metrics=metrics_data,
        generation=GenerationData(
            control_tokens=control_tokens,
            ablated_tokens=ablated_tokens
        ),
        importance_scores=ImportanceScoresData(
            gradient_matrix=importance_matrix
        ),
        layer_impact=layer_impacts
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
