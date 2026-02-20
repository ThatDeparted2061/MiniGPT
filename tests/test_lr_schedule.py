import math

from minigpt.train.utils import cosine_lr


def test_warmup_is_linear():
    lr, warmup, max_steps, min_lr = 1.0, 10, 100, 0.1
    assert cosine_lr(0, warmup, max_steps, lr, min_lr) < cosine_lr(5, warmup, max_steps, lr, min_lr)
    assert math.isclose(cosine_lr(warmup - 1, warmup, max_steps, lr, min_lr), lr, rel_tol=1e-6)


def test_peak_at_end_of_warmup():
    assert math.isclose(cosine_lr(9, 10, 100, 1.0, 0.1), 1.0, rel_tol=1e-6)


def test_decays_to_min_after_max_steps():
    assert math.isclose(cosine_lr(200, 10, 100, 1.0, 0.1), 0.1, rel_tol=1e-9)


def test_midpoint_is_halfway():
    lr, min_lr = 1.0, 0.0
    mid = cosine_lr(55, 10, 100, lr, min_lr)
    assert math.isclose(mid, 0.5, abs_tol=1e-6)
