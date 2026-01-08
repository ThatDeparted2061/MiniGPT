import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from minigpt.model.config import GPTConfig
from minigpt.model.attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):
    """A decoder-only GPT language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        # Position offset accounts for tokens already in the KV-cache.
        past_len = 0
        if kv_caches is not None and kv_caches[0] is not None and kv_caches[0][0] is not None:
            past_len = kv_caches[0][0].shape[2]
        assert past_len + T <= self.config.block_size, "sequence exceeds block size"

        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )

        new_caches = []
        for i, block in enumerate(self.transformer.h):
            cache = kv_caches[i] if kv_caches is not None else None
            if self.config.grad_checkpoint and self.training and cache is None:
                x, nc = checkpoint(block, x, None, use_reentrant=False)
            else:
                x, nc = block(x, kv_cache=cache)
            new_caches.append(nc)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss

        # Inference: only compute logits for the final position when generating.
        logits = self.lm_head(x[:, [-1], :]) if kv_caches is not None else self.lm_head(x)
        return logits, new_caches

    def configure_optimizers(self, weight_decay, lr, betas, device_type="cpu"):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__doc__
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **({"fused": True} if fused else {}))
