"""Print the parameter count for a model described by a YAML config.

Usage:
    python scripts/count_params.py configs/gpt2-124m.yaml
"""

import sys

from minigpt.model.config import GPTConfig
from minigpt.model.gpt import GPT
from minigpt.train.utils import load_config


def main(path: str) -> None:
    cfg_dict = load_config(path)["model"]
    fields = GPTConfig().__dict__.keys()
    cfg = GPTConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
    model = GPT(cfg)
    total = model.num_params(non_embedding=False)
    non_emb = model.num_params(non_embedding=True)
    print(f"config        : {path}")
    print(f"layers x width: {cfg.n_layer} x {cfg.n_embd}")
    print(f"total params  : {total / 1e6:.1f}M")
    print(f"non-embedding : {non_emb / 1e6:.1f}M")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/gpt2-124m.yaml")
