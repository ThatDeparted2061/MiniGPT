"""Supervised fine-tuning with LoRA adapters.

Loads a pretrained checkpoint, wraps attention/MLP projections in LoRA, and
trains only the adapters on instruction/response pairs. Loss is masked to the
response tokens so the model learns to answer, not to echo the prompt.
"""

import argparse
import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence

from minigpt.model import GPT, GPTConfig
from minigpt.lora import apply_lora, mark_only_lora_as_trainable, lora_parameters
from minigpt.data.tokenizer import Tokenizer
from minigpt.train.utils import load_config, save_checkpoint

IGNORE_INDEX = -1


def build_example(tok, prompt, response, block_size):
    """Tokenize a (prompt, response) pair; mask the prompt out of the loss."""
    p_ids = tok.encode(f"### Instruction:\n{prompt}\n\n### Response:\n")
    r_ids = tok.encode(response, add_eot=True)
    ids = (p_ids + r_ids)[:block_size]
    labels = ([IGNORE_INDEX] * len(p_ids) + r_ids)[:block_size]
    return torch.tensor(ids), torch.tensor(labels)


def load_pairs(path):
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            yield ex["prompt"], ex["response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"], strict=False)
    apply_lora(model, r=cfg.get("lora_r", 8), alpha=cfg.get("lora_alpha", 16),
               dropout=cfg.get("lora_dropout", 0.05))
    mark_only_lora_as_trainable(model)
    model.to(device)

    tok = Tokenizer()
    block_size = model.config.block_size
    data = [build_example(tok, p, r, block_size) for p, r in load_pairs(cfg["data_file"])]

    opt = torch.optim.AdamW(lora_parameters(model), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    batch_size = cfg["batch_size"]

    model.train()
    for epoch in range(cfg["epochs"]):
        for i in range(0, len(data), batch_size):
            batch = data[i: i + batch_size]
            x = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=0).to(device)
            y = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=IGNORE_INDEX).to(device)
            _, loss = model(x, targets=y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if (i // batch_size) % cfg.get("log_interval", 20) == 0:
                print(f"epoch {epoch} step {i // batch_size}: loss {loss.item():.4f}")

    save_checkpoint(os.path.join(cfg["out_dir"], "latest.pt"), model, opt, 0, model.config)


if __name__ == "__main__":
    main()
