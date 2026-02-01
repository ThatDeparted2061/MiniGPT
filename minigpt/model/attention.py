import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses FlashAttention via ``torch.nn.functional.scaled_dot_product_attention``
    when enabled (IO-aware, memory-efficient), and falls back to an explicit
    masked-softmax implementation otherwise. Supports an optional KV-cache for
    fast autoregressive decoding.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Fused QKV projection.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = config.flash and hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Causal mask for the fallback path.
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer(
                "mask", mask.view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x, kv_cache=None, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Append to KV-cache for incremental decoding.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # When decoding from a cache, the incoming query is causal w.r.t. all
        # cached keys, so an explicit mask is only needed for the prefill step.
        # An external ``attn_mask`` (e.g. a key-padding mask from a continuously
        # batched, left-padded decode step) overrides the implicit causal mask.
        is_causal = attn_mask is None and (
            kv_cache is None or (new_cache is not None and q.shape[2] == k.shape[2])
        )

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            Tq, Tk = q.shape[2], k.shape[2]
            if attn_mask is not None:
                att = att.masked_fill(~attn_mask, float("-inf"))
            elif is_causal:
                att = att.masked_fill(
                    self.mask[:, :, Tk - Tq:Tk, :Tk] == 0, float("-inf")
                )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache
