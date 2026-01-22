"""Autoregressive generation with KV-caching.

The prompt is processed once to fill the per-layer key/value caches; each
subsequent step feeds a single token, so attention cost per step is O(seq)
instead of O(seq^2) re-encoding. This is the bulk of the single-GPU
throughput win.
"""

import argparse

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, eot_token=None):
    """Generate ``max_new_tokens`` continuations for a (B, T) prompt tensor."""
    model.eval()
    device = idx.device
    block_size = model.config.block_size

    # Prefill: run the full prompt once and seed the KV-cache.
    kv_caches = [(None, None) for _ in range(model.config.n_layer)]
    logits, kv_caches = model(idx, kv_caches=kv_caches)
    next_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        logits_t = next_logits / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits_t, min(top_k, logits_t.size(-1)))
            logits_t = logits_t.masked_fill(logits_t < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits_t, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)

        if eot_token is not None and (next_tok == eot_token).all():
            break
        if idx.shape[1] >= block_size:
            break

        # Decode step: feed only the new token, reuse the cache.
        logits, kv_caches = model(next_tok, kv_caches=kv_caches)
        next_logits = logits[:, -1, :]

    return idx


def _load(ckpt_path, device, int8):
    from minigpt.model import GPT, GPTConfig
    from minigpt.inference.quantize import quantize_int8

    state = torch.load(ckpt_path, map_location=device)
    config = GPTConfig.from_dict(state["config"])
    model = GPT(config)
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    if int8:
        model = quantize_int8(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--int8", action="store_true", help="INT8-quantize linear layers")
    args = ap.parse_args()

    from minigpt.data.tokenizer import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load(args.ckpt, device, args.int8)
    tok = Tokenizer()

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, tok.eot)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
