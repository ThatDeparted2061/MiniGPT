import torch

from minigpt.model.config import GPTConfig
from minigpt.model.gpt import GPT


def _tiny():
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_forward_logits_shape():
    model = _tiny()
    idx = torch.randint(0, 64, (2, 8))
    logits, _ = model(idx)
    assert logits.shape == (2, 8, 64)


def test_forward_with_targets_returns_scalar_loss():
    model = _tiny()
    idx = torch.randint(0, 64, (2, 8))
    logits, loss = model(idx, targets=idx)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_weight_tying():
    model = _tiny()
    assert model.transformer.wte.weight is model.lm_head.weight


def test_num_params_excludes_position_embeddings():
    model = _tiny()
    assert model.num_params(non_embedding=True) < model.num_params(non_embedding=False)
