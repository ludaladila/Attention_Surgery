"""
Quantitative metrics comparing original and ablated generations.
"""
from typing import Dict, List

import numpy as np

from .cache import HeadId

EPS = 1e-12


def _ensure_seq_logits(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    max_logits = np.max(logits, axis=-1, keepdims=True)
    logsum = max_logits + np.log(
        np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True)
    )
    return logits - logsum


def _softmax(logits: np.ndarray) -> np.ndarray:
    log_probs = _log_softmax(logits)
    return np.exp(log_probs)


def _kl_divergence(
    logits_a: np.ndarray, logits_b: np.ndarray
) -> float:
    seq_len = min(logits_a.shape[0], logits_b.shape[0])
    if seq_len == 0:
        return 0.0
    probs_a = _softmax(logits_a[:seq_len])
    probs_b = _softmax(logits_b[:seq_len])
    kl = np.sum(
        probs_a * (np.log(probs_a + EPS) - np.log(probs_b + EPS)),
        axis=-1,
    )
    return float(np.mean(kl))


def _top1_changed_ratio(
    logits_a: np.ndarray, logits_b: np.ndarray
) -> float:
    seq_len = min(logits_a.shape[0], logits_b.shape[0])
    if seq_len == 0:
        return 0.0
    argmax_a = np.argmax(logits_a[:seq_len], axis=-1)
    argmax_b = np.argmax(logits_b[:seq_len], axis=-1)
    changed = np.not_equal(argmax_a, argmax_b).sum()
    return float(changed / seq_len)


def _perplexity(logits: np.ndarray, token_ids: List[int]) -> float:
    seq_logits = logits[:-1]
    if seq_logits.shape[0] == 0 or len(token_ids) < 2:
        return float("nan")
    seq_len = min(seq_logits.shape[0], len(token_ids) - 1)
    log_probs = _log_softmax(seq_logits[:seq_len])
    targets = np.array(token_ids[1 : seq_len + 1])
    target_log_probs = log_probs[np.arange(seq_len), targets]
    nll = -target_log_probs.mean()
    return float(np.exp(nll))


def _attention_diff(
    attn_a: Dict[HeadId, np.ndarray],
    attn_b: Dict[HeadId, np.ndarray],
) -> float:
    shared_heads = set(attn_a) & set(attn_b)
    if not shared_heads:
        return 0.0
    diffs = []
    for head in shared_heads:
        pa = attn_a[head]
        pb = attn_b[head]
        seq_q = min(pa.shape[-2], pb.shape[-2])
        seq_k = min(pa.shape[-1], pb.shape[-1])
        if seq_q == 0 or seq_k == 0:
            continue
        diff = pa[:seq_q, :seq_k] - pb[:seq_q, :seq_k]
        diffs.append(np.linalg.norm(diff))
    return float(np.mean(diffs)) if diffs else 0.0


def _activation_l2(
    acts_a: Dict[str, np.ndarray],
    acts_b: Dict[str, np.ndarray],
) -> float:
    shared = set(acts_a) & set(acts_b)
    if not shared:
        return 0.0
    distances = []
    for key in shared:
        a = acts_a[key].reshape(-1)
        b = acts_b[key].reshape(-1)
        length = min(a.size, b.size)
        if length == 0:
            continue
        diff = a[:length] - b[:length]
        distances.append(np.linalg.norm(diff) / length)
    return float(np.mean(distances)) if distances else 0.0


def compute_metrics(
    logits_original: np.ndarray,
    logits_ablated: np.ndarray,
    input_token_ids: List[int],
    output_token_ids_original: List[int],
    output_token_ids_ablated: List[int],
    attention_original: Dict[HeadId, np.ndarray],
    attention_ablated: Dict[HeadId, np.ndarray],
    activations_original: Dict[str, np.ndarray],
    activations_ablated: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Produce a dictionary of metrics describing original vs ablated runs.
    """
    logits_o = _ensure_seq_logits(logits_original)
    logits_a = _ensure_seq_logits(logits_ablated)

    metrics: Dict[str, float] = {}
    metrics["kl_div"] = _kl_divergence(logits_o, logits_a)
    metrics["top1_changed_ratio"] = _top1_changed_ratio(logits_o, logits_a)
    metrics["attention_diff"] = _attention_diff(
        attention_original, attention_ablated
    )
    metrics["l2_activation"] = _activation_l2(
        activations_original, activations_ablated
    )

    seq_ids_original = input_token_ids + output_token_ids_original
    seq_ids_ablated = input_token_ids + output_token_ids_ablated
    metrics["perplexity_original"] = _perplexity(logits_o, seq_ids_original)
    metrics["perplexity_ablated"] = _perplexity(logits_a, seq_ids_ablated)

    if not np.isnan(metrics["perplexity_original"]) and not np.isnan(
        metrics["perplexity_ablated"]
    ):
        metrics["perplexity_delta"] = (
            metrics["perplexity_ablated"] - metrics["perplexity_original"]
        )
    else:
        metrics["perplexity_delta"] = float("nan")

    return metrics

