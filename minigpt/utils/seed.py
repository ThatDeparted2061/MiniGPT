"""Reproducibility helper."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    With ``deterministic=True`` cuDNN is pinned to deterministic kernels, which
    trades some throughput for bit-for-bit reproducible runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
