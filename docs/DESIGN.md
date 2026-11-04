# Design

## Goals
- A readable, correct reference implementation of the core LLM-serving techniques.
- Every optimisation measured against a baseline running the *same model code*.
- Correctness verified against Hugging Face, not assumed.

## Non-goals
- Matching vLLM / TensorRT-LLM throughput. Multi-GPU, speculative decoding, LoRA serving, prefix caching.
- Supporting every architecture: Llama-family (no RoPE scaling) and Qwen2 only.

## Phase 1: model, baseline, contiguous KV-cache
- **Own the forward pass.** Wrapping `transformers` would hide exactly what is being studied. Parameter names match
  HF so checkpoints load unmodified.
- **`F.scaled_dot_product_attention`** (fused kernels where available) so the baseline is not a strawman. Three
  call paths: decode (T=1), full prefill (`is_causal`), chunked prefill (explicit mask).
- **RoPE in fp32, computed per forward**, not cached as buffers that `.half()` would silently downcast.
- **Contiguous cache** `[layers, batch, kv_heads, max_seq, head_dim]`: simple and fast for one sequence, but reserves
  `max_seq` per sequence and cannot batch different lengths.
- **GQA via `repeat_interleave`** at attention time (the cache stores only `kv_heads`). Costs a copy per step.
- **Honest timing:** every step ends in `.item()`, which blocks on the GPU, so wall-clock per step is real.

## Phase 2: paged KV-cache
- Pool `[layers, num_blocks, kv_heads, block_size, head_dim]` shared by all sequences; each sequence owns a block
  table. Blocks are allocated on demand and freed on completion, so waste is bounded by one partial block per sequence.
- `build_batch` is **atomic**: it checks the total block need before mutating anything, so an `OutOfBlocks` never
  leaves half-allocated state.
- **Attention = write, gather, masked SDPA.** New K/V are scattered into their slots (padding tokens skipped), each
  sequence's blocks are gathered into a contiguous tensor, and a mask `key_pos <= query_pos AND key_pos < context_len`
  hides future tokens and padding. The gather is a real copy; a fused paged-attention kernel would remove it.
- Right-padded batches with per-sequence `positions` and `last_idx` let sequences of different lengths share a
  forward pass; logits are gathered only at each sequence's last real token.

## Phase 3: continuous batching
- **Iteration-level scheduling.** Each `step()` admits waiting requests (FCFS, no skipping, so no starvation), runs one
  padded prefill, then one batched decode for sequences that were already running. Finished sequences leave at once.
- **Admission control:** a request is admitted only if its blocks fit *and* one spare block per live sequence remains
  (a decode step can never starve right after admission). Requests that could never fit are rejected up front.
- **Preemption by recompute:** if decode needs blocks that do not exist, the newest-admitted sequence is evicted, its
  blocks freed, and it is re-queued at the front to re-prefill `prompt + generated`. Recompute avoids a CPU swap path;
  vLLM makes the same default. Tested: greedy outputs are unchanged.
- **Static policy** (`policy="static"`) = admit only when nothing is running. It still stops computing finished
  sequences, so it is *generous* to static batching; the gap measured against continuous batching is purely idle slots.
- **Sampling** stays per-request (own generator and params); an all-greedy batch uses one argmax and one device sync.

## Phase 4: quantization (weight-only)
- **INT8:** per-output-channel symmetric. The scale factors out of the matmul (`y = (x @ q^T) * scale`), so no
  dequantised weight matrix is materialised for this scheme.
- **INT4:** group-wise (default 128) asymmetric, scale + zero-point per group, two nibbles per byte. Round-to-nearest
  (no calibration); AWQ/GPTQ would improve accuracy at the same storage.
- Embeddings, norms and `lm_head` stay in fp16 (small share of bytes, sensitive to error).
- **Honest caveat:** weight-only quantization in plain PyTorch saves *memory* but the dequantise step costs bandwidth
  and compute, so decode may not speed up. That is exactly the motivation for Phase 6.
- Quality is measured with WikiText-2 perplexity on non-overlapping 512-token windows (comparable across schemes on
  the same data; higher than a stride-overlapped number).

## Phase 5: server
- **One engine thread owns all engine state.** HTTP handlers only touch thread-safe queues; tokens return through
  `loop.call_soon_threadsafe`. No locks around model code.
- **Backpressure:** a cap on in-flight requests returns HTTP 429 instead of letting latency grow without bound.
- **Cancellation:** a dropped SSE connection triggers `abort`, which frees KV blocks at the next step.
- **Error isolation:** an engine exception fails in-flight requests but not the process; state is reset.
- OpenAI-compatible `/v1/completions` (plus `top_k` and `token_ids` extensions); `/metrics` exposes queue and cache state.

## Phase 6: Triton kernel + profiling
- Decode is memory-bound, so 4-bit weights only pay off if they cross HBM as 4 bits. The kernel reads packed bytes,
  applies scale/zero in registers, and feeds `tl.dot` directly; one quantization group per K step.
- Used only for few-row inputs (`rows <= 16`, the decode regime); prefill dequantises and uses cuBLAS.
- Validated against the PyTorch reference in Triton's CPU interpreter and (when available) on GPU.
- Each packed byte is loaded twice (once per nibble); a production kernel would unpack both nibbles from one load.

## Known limitations / future work
- Gathering blocks for attention copies memory; a fused paged-attention kernel is the natural next step.
- No prefix caching, chunked prefill scheduling, speculative decoding or CUDA graphs.
- A host-device sync per step (`.item()` / `.tolist()`); async sampling would hide it.
- No `rope_scaling` (Llama-3.x), no sliding-window attention.
- fp16 numerics differ slightly from HF; parity is asserted in fp32.
