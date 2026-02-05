import dataclasses

from minigpt.model.config import GPTConfig


def test_defaults_describe_124m_geometry():
    cfg = GPTConfig()
    assert cfg.vocab_size == 50257
    assert cfg.block_size == 1024
    assert cfg.n_layer == 12


def test_n_embd_divisible_by_n_head():
    cfg = GPTConfig()
    assert cfg.n_embd % cfg.n_head == 0


def test_override_fields():
    cfg = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=64)
    assert cfg.n_layer == 4
    assert cfg.n_embd == 128
    # Config is a plain dataclass, so asdict round-trips cleanly.
    assert dataclasses.asdict(cfg)["block_size"] == 64
