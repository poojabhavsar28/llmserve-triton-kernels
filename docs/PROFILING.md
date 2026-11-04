# Profiling guide (Phase 6)

```bash
python benchmarks/profile_decode.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant fp16
python benchmarks/profile_decode.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant int4
python benchmarks/bench_int4_kernel.py        # GPU + Triton only
```

Each run prints the top operators and writes `results/trace-<quant>.json`; open it at https://ui.perfetto.dev.

## What to look for
1. **Decode is memory-bound.** With one token per step, the time per step is close to
   `bytes of weights read / memory bandwidth`. Compare the fp16 step time to
   `weights_MiB / GPU bandwidth`; if it is near that line, only reading fewer bytes can help.
2. **INT4 without the kernel.** The `torch` path shows separate `unpack/mul/sub/cat` ops that
   materialise an fp16 weight, then a GEMM. Expect memory traffic *not* to drop, so little or no speedup.
3. **INT4 with the kernel.** One `_int4_matmul_kernel` launch per linear layer; the dequantise
   work disappears from the trace. Compare per-step time against the previous two.
4. **Host overhead.** Gaps between kernels (idle GPU) mean Python/launch overhead dominates; for small
   models on fast GPUs that is the next bottleneck (CUDA graphs would address it).
5. **Attention.** In the paged engine, look for the `index`/gather ops before SDPA: that copy is what a
   fused paged-attention kernel would remove.

## Write-up template (paste your measurements)
| Config | step time (ms) | weight bytes/step | achieved GB/s | % of peak bandwidth | notes |
|---|---|---|---|---|---|
| fp16 | | | | | |
| int4, torch path | | | | | |
| int4, Triton kernel | | | | | |
