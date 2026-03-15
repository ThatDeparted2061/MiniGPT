# Architecture

MiniGPT is a decoder-only transformer with the GPT-2 geometry, plus the
training and inference machinery needed to take it from random init to a
served, aligned model.

## Model (`minigpt/model`)

- `config.py` — `GPTConfig` dataclass; defaults describe the 124M model.
- `attention.py` — `CausalSelfAttention` built on `scaled_dot_product_attention`
  (FlashAttention kernels when available) with an optional KV-cache.
- `gpt.py` — embeddings, a stack of pre-norm `Block`s, tied LM head, weight
  init, and `configure_optimizers` (decoupled weight decay, fused AdamW on CUDA).

## Data (`minigpt/data`, `scripts/prepare_data.py`)

Documents are GPT-2 BPE tokenized and packed into a flat `uint16` memmap. The
dataset samples contiguous `block_size + 1` windows, so batches are produced
with a single slice and no padding.

## Training (`minigpt/train`)

- `pretrain.py` — DDP, gradient accumulation, bf16/fp16 autocast, cosine LR.
- `sft.py` — supervised fine-tuning with loss masked to completion tokens.
- `dpo.py` — Direct Preference Optimization against a frozen reference model.
- `utils.py` — config loading, LR schedule, checkpointing, DDP bootstrap.

## Inference & serving (`minigpt/inference`, `minigpt/serve`)

KV-cached autoregressive generation, INT8 weight-only quantization, and a
continuous-batching engine that packs sequences of differing lengths into each
decode step via explicit `position_ids`.

## Evaluation (`minigpt/eval`)

Held-out perplexity and an MT-Bench-style judging harness.
