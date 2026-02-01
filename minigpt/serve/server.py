"""HTTP serving front-end for the continuous-batching engine.

Exposes a small JSON API backed by :class:`ContinuousBatchingEngine`. A single
engine (one model, one scheduler thread) is shared across all connections, while
the threaded HTTP server lets many clients submit concurrently — requests fan
into the shared running batch and tokens stream back per connection.

    POST /generate   {"prompt": str, "max_new_tokens": int,
                      "temperature": float, "top_k": int, "stream": bool}

With ``"stream": true`` the response is Server-Sent Events (one token delta per
``data:`` line); otherwise it is a single JSON object with the full completion.

For production throughput, install the optional ``vllm`` extra and point it at
an exported checkpoint; this server prioritises legibility over peak tokens/s.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from minigpt.data.tokenizer import Tokenizer
from minigpt.inference.generate import _load
from minigpt.serve.engine import ContinuousBatchingEngine, SamplingParams


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Injected by ``serve`` before the server starts.
    engine: ContinuousBatchingEngine = None
    tokenizer: Tokenizer = None

    def log_message(self, *args):  # quieter default logging
        pass

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "invalid JSON body"})
            return

        prompt = body.get("prompt", "")
        params = SamplingParams(
            max_new_tokens=int(body.get("max_new_tokens", 256)),
            temperature=float(body.get("temperature", 0.8)),
            top_k=body.get("top_k", 50),
        )
        prompt_ids = self.tokenizer.encode(prompt)
        req = self.engine.submit(prompt_ids, params)

        if body.get("stream"):
            self._stream(req)
        else:
            self._collect(req)

    # ----------------------------------------------------------- responses
    def _collect(self, req):
        ids = []
        while True:
            tok = req.out_q.get()
            if tok is None:
                break
            ids.append(tok)
        self._json(200, {
            "text": self.tokenizer.decode(ids),
            "tokens": len(ids),
            "finish_reason": req.finish_reason,
        })

    def _stream(self, req):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        ids = []
        prev = ""
        while True:
            tok = req.out_q.get()
            if tok is None:
                break
            ids.append(tok)
            # Decode the full prefix each step and emit the delta — robust to
            # multi-byte BPE tokens that a per-token decode would split.
            text = self.tokenizer.decode(ids)
            delta, prev = text[len(prev):], text
            self.wfile.write(f"data: {json.dumps({'token': delta})}\n\n".encode())
            self.wfile.flush()
        done = json.dumps({"done": True, "finish_reason": req.finish_reason})
        self.wfile.write(f"data: {done}\n\n".encode())
        self.wfile.flush()

    def _json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--int8", action="store_true", help="INT8-quantize linear layers")
    ap.add_argument("--max-batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load(args.ckpt, device, args.int8)
    tok = Tokenizer()

    _Handler.engine = ContinuousBatchingEngine(
        model, eot_token=tok.eot, device=device, max_batch_size=args.max_batch_size
    )
    _Handler.tokenizer = tok

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"minigpt-serve listening on http://{args.host}:{args.port} (device={device})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
