"""INT8 weight-only quantization for linear layers.

Each ``nn.Linear`` weight is quantized per-output-channel to int8 with a
float scale. Matmuls dequantize on the fly, cutting weight memory ~4x and
improving memory-bound decode throughput while keeping activations in fp.
"""

import torch
import torch.nn as nn


class Int8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        # Per-row (per-output-channel) symmetric quantization.
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.squeeze(1))
        self.register_buffer("bias", bias if bias is not None else None)
        self.out_features, self.in_features = weight.shape

    def forward(self, x):
        w = self.qweight.to(x.dtype) * self.scale.unsqueeze(1).to(x.dtype)
        out = x @ w.t()
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


def quantize_int8(model: nn.Module, skip=("lm_head",)):
    """Replace ``nn.Linear`` layers in-place with INT8 equivalents."""
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, Int8Linear(child.weight.data, child.bias.data if child.bias is not None else None))
    return model
