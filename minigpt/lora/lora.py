"""Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

Wraps an existing ``nn.Linear`` with a frozen base weight plus a trainable
low-rank update ``B @ A * (alpha / r)``. Only the adapter matrices train, so
instruction tuning touches a tiny fraction of parameters.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout)

        # A ~ Kaiming, B = 0  =>  the adapter starts as a no-op.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        update = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + update * self.scaling


def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0,
               targets=("c_attn", "c_proj", "c_fc")):
    """Replace matching ``nn.Linear`` submodules in-place with ``LoRALinear``."""
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return model


def mark_only_lora_as_trainable(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    return model


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_" in n]
