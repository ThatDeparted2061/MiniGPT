import torch

from minigpt.inference.sampling import apply_temperature, top_k_filter, top_p_filter


def test_temperature_scales_logits():
    logits = torch.tensor([2.0, 4.0])
    out = apply_temperature(logits, 2.0)
    assert torch.allclose(out, torch.tensor([1.0, 2.0]))


def test_top_k_keeps_exactly_k_tokens():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = top_k_filter(logits, 2)
    assert torch.isinf(out).sum() == 2
    assert out[2] == 3.0 and out[3] == 4.0


def test_top_p_keeps_most_probable_token():
    logits = torch.tensor([10.0, 0.0, -10.0])
    out = top_p_filter(logits, 0.5)
    # Highest-probability token must survive even with a tight nucleus.
    assert torch.isfinite(out[0])


def test_invalid_temperature_raises():
    try:
        apply_temperature(torch.zeros(3), 0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
