# llmserve

A from-scratch LLM inference engine in PyTorch with **KV-cache**, **paged attention memory**,
**continuous batching**, **INT8/INT4 weight quantization**, an **OpenAI-compatible streaming server**
and a **Triton INT4 kernel**, each benchmarked against a deliberately naive baseline that runs the
*same model code*, and each tested for correctness against Hugging Face `transformers`.

> The Results section contains measurements from an NVIDIA GeForce RTX 3050 Laptop GPU. Results
> depend on hardware, power limits, software versions, and workload; benchmark your own setup before
> comparing performance.

## Contents
- [Check that everything works](#check-that-everything-works)
- [Install](#install)
- [Phases and how to run each](#phases-and-how-to-run-each)
- [Benchmarks](#benchmarks) · [Results](#results)
- [Architecture](#architecture) · [Correctness](#correctness)
- [Upload to GitHub](#upload-to-github)
- [Troubleshooting](#troubleshooting)

## Check that everything works

After [installing](#install), one command tests every phase end to end on a tiny random model
(offline, no downloads, about 1-3 minutes):

```bash
python scripts/verify.py
```

It prints a PASS/FAIL line per phase and writes full logs to `results/verify.log`. Expected tail:

```
PASS    ...  tests: all phases ...
PASS    ...  Phase 1  naive vs KV-cache benchmark
PASS    ...  Phase 2  paged vs contiguous capacity
PASS    ...  Phase 3  continuous vs static vs sequential load test
PASS    ...  Phase 4  fp16 / int8 / int4 benchmark
PASS    ...  Phase 5  OpenAI-compatible server + HTTP load test
PASS    ...  Phase 6  profiler (decode steps -> Chrome trace)
PASS    ...  Phase 6  INT4 Triton kernel micro-benchmark (GPU only; skips cleanly otherwise)
ALL CHECKS PASSED
```

With a real model on your GPU (downloads ~2.2 GB for TinyLlama):

```bash
python scripts/verify.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Notes: GPU-only and Triton-only tests report as *skipped* (not failed) when unavailable. Use
`--skip-server` if you cannot open a local port.

## Install

Python 3.10+. NVIDIA GPU recommended (CPU works for tests and the tiny model).

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 1. PyTorch with CUDA first (pick the right build for your driver at https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. this package + dev tools
pip install -e ".[dev]"

# optional: WikiText-2 perplexity (Phase 4)
pip install -e ".[eval]"

# optional: Triton INT4 kernel (Phase 6).  Linux:  pip install triton
#                                          Windows: pip install triton-windows
```

If Triton is not installed everything still works; INT4 simply uses the PyTorch dequantise path.

Models that work: Llama-family without RoPE scaling (**TinyLlama-1.1B**) and **Qwen2 / Qwen2.5**
(e.g. Qwen2.5-0.5B). Llama-3.x needs RoPE scaling, which is not implemented.

## Phases and how to run each

| Phase | What | Where | Run it |
|---|---|---|---|
| 1 | Llama-style model, naive baseline, contiguous KV-cache, sampling | `model/`, `baseline/`, `engine/`, `kvcache/contiguous.py` | `python -m llmserve.cli --model TinyLlama/TinyLlama-1.1B-Chat-v1.0` · `python benchmarks/run_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| 2 | Paged KV-cache (block table + allocator) | `kvcache/paged.py` | `python benchmarks/memory_capacity.py --preset tinyllama-1.1b --budget-mib 1024` |
| 3 | Continuous batching, preemption, static-batching baseline | `scheduler/engine.py` | `python benchmarks/load_test.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rates 2 4 8` |
| 4 | INT8 per-channel and INT4 group-wise quantization + perplexity | `quant/`, `eval/` | `python benchmarks/run_quant_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| 5 | OpenAI-compatible server, SSE streaming, backpressure, cancellation | `server/` | see below |
| 6 | Triton fused INT4 dequant-matmul, profiling | `quant/triton_int4.py`, `benchmarks/profile_decode.py` | `python benchmarks/bench_int4_kernel.py` · `python benchmarks/profile_decode.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant int4` |

### Phase 5: run the server
```bash
python -m llmserve.server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant int4 --kv-mem-mib 512

# in another terminal
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt": "Explain KV caching in one paragraph.", "max_tokens": 80, "temperature": 0, "stream": true}'
curl http://127.0.0.1:8000/metrics        # running/waiting requests, free KV blocks, preemptions
python benchmarks/load_test.py --url http://127.0.0.1:8000 --vocab 32000 --rates 4 8   # HTTP load test
```
Windows PowerShell: use `curl.exe` and put the JSON in a file (`-d "@body.json"`), or use the load-test script.
Interactive API docs: http://127.0.0.1:8000/docs

## Benchmarks

Methodology is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Every script writes a markdown table to
`results/` (`latest.md`, `load_latest.md`, `quant_latest.md`) and machine-readable JSON next to it.
Close other GPU apps; on a laptop, plug in and use the performance profile to avoid thermal throttling.

## Results

Measurements below were collected on an **NVIDIA GeForce RTX 3050 Laptop GPU** using
**TinyLlama/TinyLlama-1.1B-Chat-v1.0**. Treat them as a report of this specific laptop run, not as
general performance claims. Perplexity was not measured because the optional `datasets` dependency
was not installed.

**Phase 1: naive vs KV-cache** (`results/latest.md`)

| engine | prompt | new tok | TTFT (ms) | tok/s | ITL p99 (ms) | peak MiB | speedup |
|---|---|---|---|---|---|---|---|
| Not measured yet | | | | | | | |

**Phase 2: capacity at a fixed KV budget** (`memory_capacity.py`)

| cache | concurrent sequences | vs contiguous | fragmentation |
|---|---|---|---|
| Not measured yet | | | |

**Phase 3/5: load test** (`results/load_latest.md`): sequential vs static vs continuous

| policy | rate (req/s) | requests | tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | e2e p50 (s) | e2e p99 (s) | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HTTP server (INT4) | 2.0 | 40 | 17.2 | 15909.5 | 31866.6 | 56652.4 | 148.7 | 1205.9 | 3012.6 | 26.15 | 97.72 | 0 |
| HTTP server (INT4) | 4.0 | 40 | 13.9 | 100887.1 | 128412.8 | 130468.9 | 166.0 | 1816.1 | 4165.4 | 113.94 | 145.4 | 0 |

These load-test runs completed without request errors, but the high TTFT and end-to-end latency show
that these offered request rates overload this laptop setup.

**Phase 4: quantization** (`results/quant_latest.md`)

| scheme | weights MiB | decode tok/s | peak memory MiB | memory vs FP16 | speed vs FP16 | perplexity |
|---|---:|---:|---:|---:|---:|---|
| FP16 | 2098.2 | 17.1 | 2120.0 | 1.00x | 1.00x | n/a |
| INT8 | 1174.9 | 12.0 | 1245.0 | 0.56x | 0.70x | n/a |
| INT4 | 733.8 | 15.6 | 913.0 | 0.35x | 0.91x | n/a |

The CLI generation smoke test with the same model reported **20.7 tok/s decode speed**, **1,222.9 ms
TTFT**, and **2,114 MiB peak memory**. This is a separate run from the quantization benchmark above.

**Phase 6: kernel + profiling:** see [docs/PROFILING.md](docs/PROFILING.md).

## Architecture

```
HTTP (FastAPI, SSE) ──► AsyncEngine (thread-safe bridge, backpressure, cancel)
                              │
                     ContinuousBatchingEngine  ── admit / prefill / decode / preempt each iteration
                              │
              LlamaForCausalLM (+ QuantLinear8/4 layers) ──► sampler
                              │
                    PagedKVCache  (block pool + per-sequence block tables + allocator)
```

```
src/llmserve/
  model/       Llama/Qwen2-style decoder; HF parameter names; paged + contiguous attention paths
  kvcache/     contiguous.py (Phase 1) · paged.py (Phase 2)
  engine/      single-request prefill+decode with latency capture
  baseline/    naive no-cache generation, the reference for every speedup
  scheduler/   continuous batching engine (Phase 3)
  quant/       int8.py · int4.py · triton_int4.py (Phases 4 and 6)
  eval/        perplexity
  server/      async_engine.py · app.py · __main__.py (Phase 5)
benchmarks/    run_benchmark · memory_capacity · load_test · run_quant_benchmark · profile_decode · bench_int4_kernel
scripts/       verify.py (one-command check of everything)
tests/         correctness suite
docs/          DESIGN.md · BENCHMARKS.md · PROFILING.md
```

## Correctness

Speed numbers only mean something if outputs are right. The suite checks that:

- cached prefill+decode, chunked prefill and batched logits equal an uncached full forward
- our logits equal Hugging Face `LlamaForCausalLM` and `Qwen2ForCausalLM` (including QKV bias, tied embeddings)
- paged attention equals a plain forward, including right-padded variable-length batches and block-boundary crossings
- allocator invariants hold under random operation sequences (Hypothesis): no leaks, no duplicate ids, no double free
- continuous and static batching give token-identical greedy output to independent generation, including under
  forced **preemption + recompute**, EOS, mid-flight arrivals and aborts; blocks are never leaked
- INT8/INT4 error is bounded by the quantization step; packing round-trips exactly
- the Triton kernel equals the dequantise-then-matmul reference (CPU interpreter; GPU variants when CUDA is present)
- the server returns reference-identical tokens for blocking and streaming requests under concurrency, rejects bad
  input (400/422), applies backpressure (429) and frees resources on cancellation

```bash
pytest -q
```

## Upload to GitHub

```bash
cd llm-serve
# 1. put your name in LICENSE (replace <YOUR NAME>)
git init -b main
git add .
git commit -m "llmserve: KV-cache, paged attention, continuous batching, quantization, server, Triton kernel"

# 2a. with the GitHub CLI (https://cli.github.com), creates the repo and pushes:
gh auth login
gh repo create llmserve --public --source=. --remote=origin --push

# 2b. or: create an empty repo named llmserve on github.com first, then
git remote add origin https://github.com/<your-username>/llmserve.git
git push -u origin main
```

Before you push: run `python scripts/verify.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0` on your GPU, paste
the real tables into [Results](#results), and keep `results/*.json` out of git (already in `.gitignore`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Use a smaller `--kv-mem-mib` / `--num-blocks`, `--quant int4`, or a smaller model (Qwen2.5-0.5B). |
| `rope_scaling ... not supported` | Llama-3.x checkpoints; use TinyLlama or Qwen2.5. |
| Triton import error on Windows | `pip install triton-windows`, or ignore it: INT4 falls back to PyTorch automatically. |
| Phase 5 step fails in `verify.py` | Port blocked by a firewall: run with `--skip-server`. |
| `ModuleNotFoundError: llmserve` | Run `pip install -e ".[dev]"` from the repo root inside your venv. |
| Perplexity shows `n/a` | `pip install -e ".[eval]"` and ensure internet, or pass `--text-file my.txt`. |

## License

MIT
