.PHONY: install dev test lint data pretrain clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check minigpt tests scripts

data:
	python scripts/prepare_data.py

pretrain:
	python -m minigpt.train.pretrain --config configs/gpt2-124m.yaml

debug:
	python -m minigpt.train.pretrain --config configs/debug.yaml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf build dist *.egg-info .pytest_cache
