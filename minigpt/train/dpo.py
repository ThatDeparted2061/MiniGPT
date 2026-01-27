"""Direct Preference Optimization (DPO).

Aligns the SFT model to human preferences without a separate reward model.
For each (prompt, chosen, rejected) triple we compute the log-prob of chosen
and rejected under both the trainable policy and a frozen reference (the SFT
model), then optimize the DPO loss:

    L = -log sigmoid( beta * [ (logp_pol_w - logp_ref_w) - (logp_pol_l - logp_ref_l) ] )

which raises the policy's relative preference for the chosen response.
"""

import argparse
import copy
import json
import os

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from minigpt.model import GPT, GPTConfig
from minigpt.lora import apply_lora, mark_only_lora_as_trainable, lora_parameters
from minigpt.data.tokenizer import Tokenizer
from minigpt.train.utils import load_config, save_checkpoint

IGNORE_INDEX = -1


def _seq_logprob(model, x, labels):
    """Sum of log-probs of the labelled (response) tokens for each sequence."""
    logits, _ = model(x)
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != IGNORE_INDEX
    safe = labels.clamp(min=0)
    logp = torch.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(dim=-1)


def build(tok, prompt, response, block_size):
    p = tok.encode(f"### Instruction:\n{prompt}\n\n### Response:\n")
    r = tok.encode(response, add_eot=True)
    ids = (p + r)[:block_size]
    labels = ([IGNORE_INDEX] * len(p) + r)[:block_size]
    return torch.tensor(ids), torch.tensor(labels)


def load_prefs(path):
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            yield ex["prompt"], ex["chosen"], ex["rejected"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="SFT checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta = cfg.get("beta", 0.1)

    state = torch.load(args.ckpt, map_location=device)
    config = GPTConfig.from_dict(state["config"])

    policy = GPT(config)
    apply_lora(policy, r=cfg.get("lora_r", 8), alpha=cfg.get("lora_alpha", 16))
    policy.load_state_dict(state["model"], strict=False)
    mark_only_lora_as_trainable(policy)
    policy.to(device)

    # Frozen reference = the SFT model.
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    tok = Tokenizer()
    bs = config.block_size
    triples = list(load_prefs(cfg["data_file"]))
    opt = torch.optim.AdamW(lora_parameters(policy), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    batch_size = cfg["batch_size"]

    def collate(items):
        x = pad_sequence([i[0] for i in items], batch_first=True).to(device)
        y = pad_sequence([i[1] for i in items], batch_first=True, padding_value=IGNORE_INDEX).to(device)
        return x, y

    policy.train()
    for epoch in range(cfg["epochs"]):
        for i in range(0, len(triples), batch_size):
            batch = triples[i: i + batch_size]
            chosen = [build(tok, p, c, bs) for p, c, _ in batch]
            rejected = [build(tok, p, r, bs) for p, _, r in batch]
            xc, yc = collate(chosen)
            xr, yr = collate(rejected)

            pol_w = _seq_logprob(policy, xc, yc)
            pol_l = _seq_logprob(policy, xr, yr)
            with torch.no_grad():
                ref_w = _seq_logprob(reference, xc, yc)
                ref_l = _seq_logprob(reference, xr, yr)

            logits = beta * ((pol_w - ref_w) - (pol_l - ref_l))
            loss = -F.logsigmoid(logits).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if (i // batch_size) % cfg.get("log_interval", 20) == 0:
                acc = (logits > 0).float().mean().item()
                print(f"epoch {epoch} step {i // batch_size}: loss {loss.item():.4f}, margin-acc {acc:.3f}")

    save_checkpoint(os.path.join(cfg["out_dir"], "latest.pt"), policy, opt, 0, config)


if __name__ == "__main__":
    main()
