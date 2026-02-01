"""Continuous-batching inference engine.

Unlike static batching — where a batch is formed, run to completion, and only
then replaced — this engine schedules at the *iteration* level: every decode
step runs all currently-active sequences together, finished sequences leave the
batch immediately, and freshly arrived requests are admitted on the next step.
That keeps the GPU saturated under bursty, mixed-length traffic (the regime a
serving endpoint actually sees) instead of stalling on the slowest member of a
fixed batch.

The mechanics that make a single batched decode step over ragged sequences
correct:

* **Left-padding.** Each request keeps its own per-layer KV-cache. To stack them
  into one tensor we left-pad every cache to the longest member, so all the real
  (newest) keys line up on the right.
* **Key-padding mask.** The padded slots must not be attended to, so we pass an
  explicit boolean ``attn_mask`` to attention (which then skips the implicit
  causal mask — a single decode query is causal w.r.t. all real past keys
  anyway).
* **Per-row positions.** Sequences sit at different absolute lengths, so each row
  feeds its own ``position_ids`` into the learned positional embedding.

This is a from-scratch, plain-PyTorch stand-in for what vLLM does with paged
attention; it trades vLLM's block-table memory efficiency for legibility.
"""

import itertools
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: Optional[int] = 50


@dataclass
class _Request:
    id: int
    params: SamplingParams
    cur_len: int                      # tokens currently in this request's KV-cache
    caches: list = field(default_factory=list)   # per-layer (k, v), batch dim 1
    pending: int = 0                  # next token id to feed into the model
    n_generated: int = 0
    generated: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    # Streamed token ids; a ``None`` sentinel marks completion.
    out_q: "queue.Queue" = field(default_factory=queue.Queue)


class ContinuousBatchingEngine:
    """Iteration-level scheduler around a :class:`~minigpt.model.gpt.GPT`."""

    def __init__(self, model, eot_token, device, max_batch_size=32):
        self.model = model.eval()
        self.device = device
        self.eot = eot_token
        self.max_batch_size = max_batch_size

        cfg = model.config
        self.n_layer = cfg.n_layer
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.block_size = cfg.block_size

        self._incoming: "queue.Queue[_Request]" = queue.Queue()
        self._running: List[_Request] = []
        self._ids = itertools.count()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="cb-engine", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ API
    def submit(self, prompt_ids: List[int], params: SamplingParams) -> _Request:
        """Queue a request; returns a handle whose ``out_q`` streams token ids."""
        # The newest token must occupy a valid position, so cap the prompt to the
        # last ``block_size - 1`` tokens (leaving room for at least one new token).
        prompt_ids = list(prompt_ids[-(self.block_size - 1):]) or [self.eot]
        req = _Request(id=next(self._ids), params=params, cur_len=len(prompt_ids))
        req._prompt_ids = prompt_ids  # consumed by the scheduler's prefill
        self._incoming.put(req)
        return req

    def generate(self, prompt_ids: List[int], params: SamplingParams) -> List[int]:
        """Blocking convenience: submit and drain to the full token list."""
        req = self.submit(prompt_ids, params)
        out = []
        while True:
            tok = req.out_q.get()
            if tok is None:
                break
            out.append(tok)
        return out

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------- scheduler
    def _loop(self):
        while not self._stop.is_set():
            self._admit()
            if not self._running:
                # Nothing to do; block briefly for the next arrival.
                try:
                    req = self._incoming.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._incoming.put(req)
                continue
            self._decode_step()
            self._running = [r for r in self._running if r.finish_reason is None]

    def _admit(self):
        while len(self._running) < self.max_batch_size and not self._incoming.empty():
            req = self._incoming.get()
            self._prefill(req)
            if req.finish_reason is None:
                self._running.append(req)
            else:
                req.out_q.put(None)

    @torch.no_grad()
    def _prefill(self, req: _Request):
        """Encode the prompt once, seed the KV-cache, sample the first token."""
        idx = torch.tensor([req._prompt_ids], dtype=torch.long, device=self.device)
        caches = [(None, None) for _ in range(self.n_layer)]
        logits, caches = self.model(idx, kv_caches=caches)
        req.caches = caches
        req.cur_len = idx.shape[1]
        tok = self._sample(logits[0, -1, :], req.params)
        self._accept(req, tok)

    @torch.no_grad()
    def _decode_step(self):
        """One batched decode iteration over every running request."""
        batch = self._running
        B = len(batch)
        L = max(r.cur_len for r in batch)

        # Stack left-padded per-request caches into uniform (B, ...) tensors.
        padded = []
        for layer in range(self.n_layer):
            ks, vs = [], []
            for r in batch:
                k, v = r.caches[layer]
                pad = L - r.cur_len
                if pad:
                    z = torch.zeros(1, self.n_head, pad, self.head_dim,
                                    dtype=k.dtype, device=self.device)
                    k = torch.cat([z, k], dim=2)
                    v = torch.cat([z, v], dim=2)
                ks.append(k)
                vs.append(v)
            padded.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))

        idx = torch.tensor([[r.pending] for r in batch], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([[r.cur_len] for r in batch], dtype=torch.long, device=self.device)

        # Valid keys per row: its real past plus the new token, all right-aligned
        # over the (L padded past + 1) key positions.
        mask = torch.zeros(B, 1, 1, L + 1, dtype=torch.bool, device=self.device)
        for i, r in enumerate(batch):
            mask[i, 0, 0, L - r.cur_len:] = True

        logits, new_caches = self.model(
            idx, kv_caches=padded, position_ids=position_ids, attn_mask=mask
        )

        for i, r in enumerate(batch):
            old = r.cur_len
            # Drop the left padding we added, keeping this row's real keys.
            r.caches = [
                (new_caches[layer][0][i:i + 1, :, L - old:, :].contiguous(),
                 new_caches[layer][1][i:i + 1, :, L - old:, :].contiguous())
                for layer in range(self.n_layer)
            ]
            r.cur_len = old + 1
            tok = self._sample(logits[i, -1, :], r.params)
            self._accept(r, tok)
            if r.finish_reason is not None:
                r.out_q.put(None)

    # --------------------------------------------------------------- helpers
    def _accept(self, req: _Request, tok: int):
        """Record a freshly sampled token and apply stop conditions."""
        if tok == self.eot:
            req.finish_reason = "stop"
            return
        req.generated.append(tok)
        req.n_generated += 1
        req.out_q.put(tok)
        if req.n_generated >= req.params.max_new_tokens:
            req.finish_reason = "length"
        elif req.cur_len >= self.block_size:
            req.finish_reason = "length"
        else:
            req.pending = tok

    @staticmethod
    def _sample(logits_vec: torch.Tensor, params: SamplingParams) -> int:
        if params.temperature <= 0:
            return int(logits_vec.argmax())
        logits = logits_vec / max(params.temperature, 1e-6)
        if params.top_k:
            v, _ = torch.topk(logits, min(params.top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[-1], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))
