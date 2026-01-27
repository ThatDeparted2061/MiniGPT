"""Shared training utilities: config loading, LR schedule, checkpointing, DDP."""

import math
import os

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(step, warmup_steps, max_steps, lr, min_lr):
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def maybe_init_ddp():
    """Initialize torch.distributed if launched under torchrun.

    Returns (is_ddp, rank, local_rank, world_size, device).
    """
    if "RANK" in os.environ and torch.cuda.is_available():
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size, f"cuda:{local_rank}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return False, 0, 0, 1, device


def save_checkpoint(path, model, optimizer, step, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
    }, path)
