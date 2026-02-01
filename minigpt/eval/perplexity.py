"""Validation perplexity over a packed token file."""

import argparse
import math

import torch

from minigpt.model import GPT, GPTConfig
from minigpt.data.dataset import PackedDataset


@torch.no_grad()
def evaluate_perplexity(model, dataset, max_batches=200, batch_size=8, device="cpu"):
    model.eval()
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return math.exp(total_loss / max(1, total_tokens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="path to val.bin")
    ap.add_argument("--max-batches", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device)
    config = GPTConfig.from_dict(state["config"])
    model = GPT(config)
    model.load_state_dict(state["model"], strict=False)
    model.to(device)

    ds = PackedDataset(args.data, config.block_size)
    ppl = evaluate_perplexity(model, ds, max_batches=args.max_batches, device=device)
    print(f"validation perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
