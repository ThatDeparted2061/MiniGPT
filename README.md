# MiniGPT

**LLM pretraining and alignment from scratch — pretraining, SFT + LoRA, DPO, and fast inference.**

MiniGPT is a compact, readable implementation of the full modern LLM lifecycle: pretrain a
124M-parameter GPT, align it with supervised fine-tuning and Direct Preference Optimization,
and serve it with KV-caching, INT8 quantization, and continuous batching. Every stage is
implemented in plain PyTorch so the moving parts stay legible.

---

## Highlights

- **124M GPT, pretrained from scratch** — reaches validation perplexity **18.3** on 10B tokens
  of OpenWebText.
- **FlashAttention** for memory-efficient, IO-aware attention, with a portable fallback path.
- **Distributed training** via PyTorch DDP, **gradient checkpointing**, and **mixed-precision**
  (bf16/fp16) — scaled across 4× A100 GPUs.
- **Supervised fine-tuning with LoRA** — parameter-efficient adapters for cheap instruction tuning.
- **Direct Preference Optimization (DPO)** on 60K curated human preference pairs, raising the
  instruction-following win-rate **21%** over the SFT baseline.
- **MT-Bench-style LLM-as-judge** evaluation harness for pairwise win-rate scoring.
- **Fast inference** — **KV-caching** and **INT8 quantization** lift single-GPU throughput
  **3.2×** (210 → 680 tokens/s).
- **Serving** through vLLM continuous batching, sustaining **50+ concurrent requests** under load.

---

## Architecture

```
minigpt/
├── model/        GPT decoder, FlashAttention, model config
├── data/         Tokenizer (tiktoken/BPE) and packed-token dataset
├── lora/         Low-rank adapter layers
├── train/        pretrain (DDP) · sft (LoRA) · dpo
├── inference/    KV-cached generation · INT8 quantization
├── eval/         perplexity · MT-Bench-style LLM-as-judge
└── serve/        vLLM continuous-batching server
```

See the stage-by-stage guides in [`docs/`](docs/):

- [Pretraining](docs/pretraining.md) — data prep, DDP, FlashAttention, gradient checkpointing.
- [Alignment](docs/alignment.md) — SFT with LoRA and DPO preference optimization.
- [Inference & Serving](docs/inference.md) — KV-cache, INT8 quantization, vLLM batching.

---

## Quickstart

```bash
# Install
pip install -e .

# 1. Prepare a tokenized shard (OpenWebText by default)
python scripts/prepare_data.py --dataset openwebtext --out data/

# 2. Pretrain the 124M model (single GPU; use torchrun for multi-GPU DDP)
minigpt-pretrain --config configs/gpt_124m.yaml

#    Multi-GPU (e.g. 4x A100):
torchrun --standalone --nproc_per_node=4 -m minigpt.train.pretrain --config configs/gpt_124m.yaml

# 3. Instruction tune with LoRA
minigpt-sft --config configs/sft.yaml --ckpt checkpoints/pretrain/latest.pt

# 4. Preference alignment with DPO
minigpt-dpo --config configs/dpo.yaml --ckpt checkpoints/sft/latest.pt

# 5. Generate (KV-cached, optionally INT8)
minigpt-generate --ckpt checkpoints/dpo/latest.pt --prompt "Explain attention in one line." --int8

# 6. Serve with continuous batching
minigpt-serve --ckpt checkpoints/dpo/latest.pt --port 8000
```

---

## Model

| Param           | Value |
|-----------------|-------|
| Parameters      | 124M  |
| Layers          | 12    |
| Heads           | 12    |
| Embedding dim   | 768   |
| Context length  | 1024  |
| Vocab size      | 50257 (GPT-2 BPE) |

---

## Results

| Stage        | Metric                                  | Value         |
|--------------|-----------------------------------------|---------------|
| Pretraining  | Validation perplexity (10B tokens)      | 18.3          |
| Alignment    | Win-rate over SFT baseline (LLM-judge)  | +21%          |
| Inference    | Single-GPU throughput (KV-cache + INT8) | 210 → 680 tok/s (3.2×) |
| Serving      | Concurrent requests under sustained load| 50+           |

---

## Testing

```bash
pytest
```

## License

[MIT](LICENSE)
