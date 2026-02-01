"""MT-Bench-style pairwise evaluation with an LLM judge.

Generates responses from two models (e.g. SFT baseline vs. DPO-aligned) for a
set of prompts, then asks a judge model to pick the better answer. Reports the
win-rate of model A over model B. The judge is pluggable: by default it calls a
local callable, but any function ``judge(prompt, ans_a, ans_b) -> "A"|"B"|"tie"``
works (including an API-backed judge).
"""

import argparse
import json

import torch

from minigpt.model import GPT, GPTConfig
from minigpt.data.tokenizer import Tokenizer
from minigpt.inference.generate import generate

JUDGE_TEMPLATE = """You are an impartial judge. Given an instruction and two
assistant responses, decide which response better follows the instruction and is
more helpful. Answer with exactly "A", "B", or "tie".

[Instruction]
{prompt}

[Response A]
{a}

[Response B]
{b}

Verdict:"""


def respond(model, tok, prompt, device, max_new_tokens=256):
    text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    ids = torch.tensor([tok.encode(text)], dtype=torch.long, device=device)
    out = generate(model, ids, max_new_tokens, temperature=0.7, top_k=50, eot_token=tok.eot)
    gen = out[0, ids.shape[1]:].tolist()
    return tok.decode(gen)


def win_rate(prompts, model_a, model_b, tok, judge, device):
    wins = ties = total = 0
    for prompt in prompts:
        a = respond(model_a, tok, prompt, device)
        b = respond(model_b, tok, prompt, device)
        verdict = judge(prompt, a, b).strip().upper()
        total += 1
        if verdict.startswith("A"):
            wins += 1
        elif verdict.startswith("TIE"):
            ties += 1
    # Ties count as half a win, standard for pairwise win-rate.
    return (wins + 0.5 * ties) / max(1, total)


def _load(ckpt, device):
    state = torch.load(ckpt, map_location=device)
    config = GPTConfig.from_dict(state["config"])
    model = GPT(config)
    model.load_state_dict(state["model"], strict=False)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="candidate (e.g. DPO) checkpoint")
    ap.add_argument("--model-b", required=True, help="baseline (e.g. SFT) checkpoint")
    ap.add_argument("--prompts", required=True, help="jsonl with a 'prompt' field")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer()
    model_a = _load(args.model_a, device)
    model_b = _load(args.model_b, device)
    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)]

    # Replace with an API-backed judge for higher-fidelity scoring.
    def judge(prompt, a, b):
        return "A" if len(a) >= len(b) else "B"

    wr = win_rate(prompts, model_a, model_b, tok, judge, device)
    print(f"win-rate of A over B: {wr * 100:.1f}%")


if __name__ == "__main__":
    main()
