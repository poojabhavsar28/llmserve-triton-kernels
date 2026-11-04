"""Phase 3: continuous (iteration-level) batching engine over the paged KV-cache.

Every call to :meth:`step` is one scheduler *iteration*:

  1. Admit waiting requests while there is batch capacity and KV blocks (FCFS, no skipping).
  2. Run ONE padded prefill forward for the newly admitted requests (samples their 1st token).
  3. Run ONE batched decode forward for the sequences that were already running.
  4. Sequences that hit EOS / max length leave immediately and free their blocks, so waiting
     requests join on the very next iteration instead of waiting for a whole batch to drain.

If the block pool runs dry during decode, the most recently admitted sequence is *preempted*:
its blocks are freed and it is re-queued at the front, to be recomputed (prompt + tokens
generated so far) later. Greedy outputs are unchanged by preemption (tested).

``policy="static"`` disables step 1 while anything is running (a new batch is only admitted once
the previous one fully drains), the classic static-batching baseline. It still stops computing
finished sequences, so it is *generous* to static batching: the remaining gap is purely the
idle batch slots.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter

import torch

from llmserve.datatypes import SamplingParams
from llmserve.kvcache.paged import PagedKVCache
from llmserve.sampling import make_generator, sample


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    req_id: int
    prompt_ids: list[int]
    params: SamplingParams
    arrival_time: float
    generated: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    generator: torch.Generator | None = None
    token_times: list[float] = field(default_factory=list)
    finish_time: float | None = None
    finish_reason: str | None = None  # "stop" | "length" | "abort"
    num_preemptions: int = 0

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.generated

    @property
    def first_token_time(self) -> float | None:
        return self.token_times[0] if self.token_times else None


@dataclass
class TokenEvent:
    req_id: int
    token: int | None
    finished: bool
    finish_reason: str | None = None
    error: str | None = None


@dataclass
class EngineStats:
    steps: int = 0
    prefill_forwards: int = 0
    decode_forwards: int = 0
    tokens_generated: int = 0
    preemptions: int = 0


class ContinuousBatchingEngine:
    def __init__(
        self,
        model,
        cache: PagedKVCache,
        *,
        max_batch_size: int = 8,
        max_prefill_tokens: int = 4096,
        policy: str = "continuous",
    ):
        if policy not in ("continuous", "static"):
            raise ValueError("policy must be 'continuous' or 'static'")
        self.model = model
        self.cache = cache
        self.device = cache.device
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.policy = policy

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: dict[int, Request] = {}
        self.stats = EngineStats()
        self._next_id = 0
        self._aborts: set[int] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ public API
    def validate_request(self, prompt_len: int, max_new_tokens: int) -> None:
        """Raise ValueError if a request could never be served. Thread-safe (pure arithmetic)."""
        if prompt_len < 1:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        total = prompt_len + max_new_tokens
        limit = self.model.config.max_position_embeddings
        if total > limit:
            raise ValueError(f"prompt + max_new_tokens = {total} exceeds model limit {limit}")
        if self.cache.blocks_for(total) > self.cache.num_blocks:
            raise ValueError(
                f"request needs {self.cache.blocks_for(total)} KV blocks but the pool has "
                f"only {self.cache.num_blocks}"
            )

    def add_request(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        *,
        req_id: int | None = None,
        arrival_time: float | None = None,
    ) -> int:
        self.validate_request(len(prompt_ids), params.max_new_tokens)
        if req_id is None:
            req_id = self._next_id
        self._next_id = max(self._next_id, req_id + 1)
        req = Request(
            req_id=req_id,
            prompt_ids=list(prompt_ids),
            params=params,
            arrival_time=perf_counter() if arrival_time is None else arrival_time,
            generator=make_generator(params, self.device),
        )
        self.waiting.append(req)
        return req_id

    def abort(self, req_id: int) -> None:
        """Request cancellation; takes effect at the start of the next step. Thread-safe."""
        with self._lock:
            self._aborts.add(req_id)

    def reset(self) -> None:
        """Drop every in-flight request and free all KV blocks (used for error recovery)."""
        for r in list(self.running):
            self.cache.free_sequence(r.req_id)
        self.running.clear()
        self.waiting.clear()

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def snapshot(self) -> dict:
        return {
            "running": len(self.running),
            "waiting": len(self.waiting),
            "free_blocks": self.cache.num_free_blocks,
            "total_blocks": self.cache.num_blocks,
            "kv_utilization": round(self.cache.utilization(), 4),
            "preemptions": self.stats.preemptions,
            "tokens_generated": self.stats.tokens_generated,
        }

    def run_until_complete(self) -> dict[int, Request]:
        while self.has_unfinished():
            self.step()
        return self.finished

    # ------------------------------------------------------------------ step
    def step(self) -> list[TokenEvent]:
        events: list[TokenEvent] = []
        self.stats.steps += 1
        self._process_aborts(events)

        admitted = self._admit()
        if admitted:
            self._run_prefill(admitted, events)

        just_prefilled = {r.req_id for r in admitted}
        decode_reqs = [r for r in self.running if r.req_id not in just_prefilled]
        if decode_reqs:
            self._run_decode(decode_reqs, events)
        return events

    # -------------------------------------------------------------- internals
    def _process_aborts(self, events: list[TokenEvent]) -> None:
        with self._lock:
            aborts, self._aborts = self._aborts, set()
        for rid in aborts:
            req = next((r for r in self.waiting if r.req_id == rid), None)
            if req is not None:
                self.waiting.remove(req)
            else:
                req = next((r for r in self.running if r.req_id == rid), None)
                if req is None:
                    continue  # already finished
                self.running.remove(req)
                self.cache.free_sequence(rid)
            self._finish(req, "abort", events, emit_token=None)

    def _admit(self) -> list[Request]:
        if self.policy == "static" and self.running:
            return []
        admitted: list[Request] = []
        reserved_blocks = 0
        prefill_tokens = 0
        while self.waiting and len(self.running) + len(admitted) < self.max_batch_size:
            req = self.waiting[0]
            n = len(req.all_ids)
            need = self.cache.blocks_for(n)
            # keep one spare block per live sequence so the next decode step cannot starve
            headroom = len(self.running) + len(admitted)
            if self.cache.num_free_blocks - reserved_blocks - need < headroom:
                break
            if admitted and prefill_tokens + n > self.max_prefill_tokens:
                break
            self.waiting.popleft()
            reserved_blocks += need
            prefill_tokens += n
            req.status = Status.RUNNING
            admitted.append(req)
        self.running.extend(admitted)
        return admitted

    def _forward(self, token_rows: list[list[int]], reqs: list[Request]) -> torch.Tensor:
        ids = [r.req_id for r in reqs]
        counts = [len(t) for t in token_rows]
        batch = self.cache.build_batch(ids, counts)
        T = max(counts)
        input_ids = torch.zeros(len(reqs), T, dtype=torch.long)
        for b, row in enumerate(token_rows):
            input_ids[b, : len(row)] = torch.tensor(row)
        with torch.inference_mode():
            logits = self.model(input_ids.to(self.device), paged=batch)
        return logits[:, 0]  # [B, V]

    def _run_prefill(self, reqs: list[Request], events: list[TokenEvent]) -> None:
        for r in reqs:
            self.cache.add_sequence(r.req_id)
        logits = self._forward([r.all_ids for r in reqs], reqs)
        self.stats.prefill_forwards += 1
        self._sample_and_append(logits, reqs, events)

    def _run_decode(self, reqs: list[Request], events: list[TokenEvent]) -> None:
        # Make room: each sequence needs a new block when it crosses a block boundary.
        while reqs:
            need = sum(self.cache.extra_blocks_needed(r.req_id, 1) for r in reqs)
            if need <= self.cache.num_free_blocks:
                break
            self._preempt(reqs.pop())  # newest admitted goes first
        if not reqs:
            return
        logits = self._forward([[r.generated[-1]] for r in reqs], reqs)
        self.stats.decode_forwards += 1
        self._sample_and_append(logits, reqs, events)

    def _preempt(self, req: Request) -> None:
        self.running.remove(req)
        self.cache.free_sequence(req.req_id)
        req.status = Status.WAITING
        req.num_preemptions += 1
        self.stats.preemptions += 1
        self.waiting.appendleft(req)  # recompute prompt + generated when re-admitted

    def _sample_and_append(
        self, logits: torch.Tensor, reqs: list[Request], events: list[TokenEvent]
    ) -> None:
        if all(r.params.temperature <= 0 for r in reqs):
            tokens = logits.argmax(dim=-1).tolist()  # one device sync for the whole batch
        else:
            tokens = [
                int(sample(logits[b : b + 1], r.params, r.generator).item())
                for b, r in enumerate(reqs)
            ]
        now = perf_counter()
        for r, tok in zip(reqs, tokens):
            r.generated.append(tok)
            r.token_times.append(now)
            self.stats.tokens_generated += 1
            reason = None
            if r.params.eos_token_id is not None and tok == r.params.eos_token_id:
                reason = "stop"
            elif len(r.generated) >= r.params.max_new_tokens:
                reason = "length"
            if reason:
                self.running.remove(r)
                self.cache.free_sequence(r.req_id)
                self._finish(r, reason, events, emit_token=tok)
            else:
                events.append(TokenEvent(r.req_id, tok, False))

    def _finish(
        self, req: Request, reason: str, events: list[TokenEvent], emit_token: int | None
    ) -> None:
        req.status = Status.FINISHED
        req.finish_reason = reason
        req.finish_time = perf_counter()
        self.finished[req.req_id] = req
        events.append(TokenEvent(req.req_id, emit_token, True, reason))
