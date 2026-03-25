"""Lightweight running-average meters for training logs."""

import time


class AverageMeter:
    """Tracks the running mean of a scalar (e.g. loss) over a window."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n: int = 1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class Throughput:
    """Estimates tokens/second using an exponential moving average."""

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.ema = None
        self._t = time.perf_counter()

    def update(self, n_tokens: int) -> float:
        now = time.perf_counter()
        dt = max(now - self._t, 1e-9)
        self._t = now
        rate = n_tokens / dt
        self.ema = rate if self.ema is None else self.beta * self.ema + (1 - self.beta) * rate
        return self.ema
