"""Micro-benchmark: forward and generation throughput for a GPTConfig.

Usage:
    python scripts/benchmark.py --n-layer 12 --n-embd 768 --steps 20
"""

import argparse
import time

import torch

from minigpt.model.config import GPTConfig
from minigpt.model.gpt import GPT


def bench_forward(model, batch, seq, device, steps):
    idx = torch.randint(0, model.config.vocab_size, (batch, seq), device=device)
    for _ in range(3):  # warmup
        model(idx)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        model(idx)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch * seq * steps / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    block_size=args.seq)
    model = GPT(cfg).to(device).eval()

    with torch.no_grad():
        tok_s = bench_forward(model, args.batch, args.seq, device, args.steps)
    print(f"params      : {model.num_params() / 1e6:.1f}M")
    print(f"device      : {device}")
    print(f"forward tok/s: {tok_s:,.0f}")


if __name__ == "__main__":
    main()
