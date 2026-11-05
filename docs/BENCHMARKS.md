# Benchmark methodology

## Single-stream (Phase 1): `benchmarks/run_benchmark.py`
- **TTFT:** request start to first token (prefill + first sample).
- **ITL:** inter-token latency = wall time of each decode step; p50 / p95 / p99 pooled over runs.
- **Tok/s:** `new_tokens / end-to-end latency` (median over runs). **Peak memory:** `torch.cuda.max_memory_allocated`.
- **Speedup:** baseline end-to-end latency / engine end-to-end latency, same prompt.
- Same model object, same prompts, greedy, no EOS (fixed output length); 1 warmup then N timed runs.

## Capacity (Phase 2): `benchmarks/memory_capacity.py`
Bookkeeping simulation with the real `BlockAllocator`: how many sequences fit in a KV budget when each reserves
`max_seq_len` (contiguous) versus `ceil(len / block_size)` blocks (paged). Lengths are lognormal around `--mean-len`.
Report the assumptions (reservation length, mean length) next to the result; the ratio depends heavily on them.

## Concurrent load (Phases 3 and 5): `benchmarks/load_test.py`
- Poisson arrivals at offered rates (req/s), mixed prompt and output lengths (fixed seed, reproducible).
- Policies: `sequential` (batch size 1: a naive server), `static` (drain, then admit), `continuous`.
- Metrics per (policy, rate): throughput tok/s, TTFT / ITL / e2e at p50/p95/p99, preemptions, errors (HTTP 429).
- Offline mode runs the real engine in-process (requests are injected between steps, so arrival granularity is one
  engine iteration). HTTP mode measures the server end to end over SSE.
- Read it as a load curve: at low rates the policies are similar; differences appear as offered load approaches
  and exceeds capacity. Report where each policy saturates.

## Quantization (Phase 4): `benchmarks/run_quant_benchmark.py`
Weight memory, decode tok/s (KV-cache engine), peak memory, and WikiText-2 perplexity per scheme. Use `int4-nokernel`
to separate the PyTorch dequantise path from the Triton kernel on the same weights.

## Kernel (Phase 6): `benchmarks/bench_int4_kernel.py`, `benchmarks/bench_decode_matmul.py`, `benchmarks/profile_decode.py`
CUDA-event timing compares cuBLAS FP16, Triton FP16, PyTorch INT4, and Triton INT4 for decode-shaped matmuls.
`bench_decode_matmul.py` rotates weights across at least 128 MiB of copies to reduce L2-cache effects and reports
achieved bandwidth; pass `--peak-gbs` to show the fraction of peak memory bandwidth. Use `--tune` to select Triton
FP16 tile configurations per shape. Profiler trace instructions are in PROFILING.md.

## Hygiene
- Record GPU, driver/CUDA, torch version, dtype, model (JSON output includes them where applicable).
- Close other GPU-heavy apps. On a laptop: plug in, performance profile, let it cool between long runs.
- Report cases where the engine does *not* win, and why.
