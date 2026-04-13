# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- GPT model with FlashAttention (SDPA) and KV-cache.
- GPT-2 BPE data pipeline with packed `uint16` memmap dataset.
- LoRA adapters and `apply_lora` helper for parameter-efficient fine-tuning.
- KV-cached generation and INT8 weight-only quantization.
- DDP pretraining, LoRA SFT, and DPO training loops.
- Perplexity and MT-Bench-style evaluation harness.
- Continuous-batching serving engine.
- Test suite, CI workflow, YAML configs, and developer tooling.

## [0.1.0]
- Initial project scaffold: packaging, license, README.
