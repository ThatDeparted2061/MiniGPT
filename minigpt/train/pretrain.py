"""Pretrain a GPT with DDP, mixed precision, and gradient accumulation.

Single GPU:
    minigpt-pretrain --config configs/gpt_124m.yaml

Multi-GPU (e.g. 4x A100):
    torchrun --standalone --nproc_per_node=4 -m minigpt.train.pretrain \
        --config configs/gpt_124m.yaml
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch

from minigpt.model import GPT, GPTConfig
from minigpt.data.dataset import random_batch
from minigpt.train.utils import load_config, cosine_lr, maybe_init_ddp, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    is_ddp, rank, local_rank, world_size, device = maybe_init_ddp()
    master = rank == 0
    torch.manual_seed(1337 + rank)

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = cfg.get("dtype", "bfloat16")
    pt_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    amp_ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if device_type == "cuda" else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    model_cfg = GPTConfig.from_dict(cfg["model"])
    model = GPT(model_cfg).to(device)
    if master:
        print(f"model parameters: {model.num_params() / 1e6:.1f}M")

    if cfg.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    opt = (model.module if is_ddp else model).configure_optimizers(
        cfg["weight_decay"], cfg["lr"], tuple(cfg["betas"]), device_type
    )

    grad_accum = cfg.get("grad_accum_steps", 1)
    block_size = model_cfg.block_size
    batch_size = cfg["batch_size"]
    train_bin = os.path.join(cfg["data_dir"], "train.bin")

    t0 = time.time()
    for step in range(cfg["max_steps"]):
        lr = cosine_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["lr"], cfg["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        # Gradient accumulation across micro-batches.
        for micro in range(grad_accum):
            x, y = random_batch(train_bin, block_size, batch_size, device)
            if is_ddp:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with amp_ctx:
                _, loss = model(x, targets=y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()

        if cfg.get("grad_clip", 0):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if master and step % cfg.get("log_interval", 10) == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {step}: loss {loss.item() * grad_accum:.4f}, lr {lr:.2e}, {dt*1000:.0f}ms")

        if master and step > 0 and step % cfg.get("ckpt_interval", 1000) == 0:
            save_checkpoint(
                os.path.join(cfg["out_dir"], "latest.pt"), model, opt, step, model_cfg
            )

    if master:
        save_checkpoint(os.path.join(cfg["out_dir"], "latest.pt"), model, opt, cfg["max_steps"], model_cfg)
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
