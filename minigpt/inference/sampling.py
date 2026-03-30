"""Logit-warping helpers for sampling: temperature, top-k and top-p."""

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy decoding instead")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask out all but the ``k`` highest-probability logits."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative
    probability >= ``p``."""
    if not 0.0 < p < 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum > p
    remove[..., 0] = False  # always keep the most probable token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)


def sample_next(logits, temperature=1.0, top_k=0, top_p=0.0):
    """Return next-token ids sampled from warped logits ``(..., vocab)``."""
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
