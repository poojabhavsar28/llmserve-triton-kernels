.PHONY: install test lint bench bench-tiny gen

MODEL ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

bench-tiny:
	python benchmarks/run_benchmark.py --model tiny --prompt-lens 32 64 --new-tokens 32 --runs 2

bench:
	python benchmarks/run_benchmark.py --model $(MODEL) --prompt-lens 128 512 --new-tokens 128 --runs 3

gen:
	python -m llmserve.cli --model $(MODEL) --prompt "Explain KV caching in one paragraph."
