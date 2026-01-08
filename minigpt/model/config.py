from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for the GPT decoder.

    Defaults describe the 124M-parameter model (GPT-2 small geometry).
    """

    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # Use FlashAttention when available (PyTorch SDPA / flash-attn kernels).
    flash: bool = True
    # Wrap transformer blocks in gradient checkpointing to trade compute for memory.
    grad_checkpoint: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})
