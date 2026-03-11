# Contributing to MiniGPT

Thanks for your interest in MiniGPT. This is a small, education-focused
codebase, so contributions that keep things readable and dependency-light are
especially welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
make dev          # editable install with the dev extras (pytest)
make test         # run the test suite
```

## Guidelines

- Keep modules self-contained and prefer standard library + PyTorch over new
  dependencies.
- Add or update a test in `tests/` for any behavioural change.
- Match the existing style: pre-norm blocks, explicit shapes in comments, and
  short docstrings describing intent rather than mechanics.
- Run `make test` before opening a pull request.

## Scope

MiniGPT intentionally implements pretraining, SFT + LoRA, DPO, and fast
inference from scratch. Features that obscure those core ideas behind heavy
abstractions are usually out of scope.
