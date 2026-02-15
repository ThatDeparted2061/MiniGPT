import torch
import torch.nn as nn

from minigpt.lora.lora import LoRALinear, apply_lora, lora_parameters, mark_only_lora_as_trainable


def test_adapter_is_noop_at_init():
    base = nn.Linear(16, 16)
    wrapped = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    # lora_B is zero-initialised, so the update vanishes at start.
    assert torch.allclose(wrapped(x), base(x), atol=1e-6)


def test_base_weight_is_frozen():
    base = nn.Linear(16, 16)
    wrapped = LoRALinear(base, r=4)
    assert wrapped.base.weight.requires_grad is False
    assert wrapped.lora_A.requires_grad is True
    assert wrapped.lora_B.requires_grad is True


def test_mark_only_lora_trainable():
    model = nn.Sequential(nn.Linear(8, 8))
    model[0] = LoRALinear(model[0], r=2)
    mark_only_lora_as_trainable(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable)
    assert len(lora_parameters(model)) == 2
