"""Thread-safe bridge between asyncio request handlers and the (single-threaded) engine.

One background thread owns the engine and runs ``engine.step()`` whenever there is work.
HTTP handlers call :meth:`submit` / :meth:`abort` from the event loop; token events travel back
through per-request ``asyncio.Queue`` objects via ``loop.call_soon_threadsafe``.

Backpressure: at most ``max_pending`` requests may be in flight (queued + running); beyond
that :meth:`submit` raises :class:`QueueFull`, which the API maps to HTTP 429.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque

from llmserve.datatypes import SamplingParams
from llmserve.scheduler.engine import ContinuousBatchingEngine, TokenEvent

log = logging.getLogger("llmserve")


class QueueFull(RuntimeError):
    pass


class AsyncEngine:
    def __init__(self, engine: ContinuousBatchingEngine, max_pending: int = 64):
        self.engine = engine
        self.max_pending = max_pending
        self._cv = threading.Condition()
        self._incoming: deque[tuple[int, list[int], SamplingParams]] = deque()
        self._aborts: set[int] = set()
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._inflight = 0
        self._next_id = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="llmserve-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=10)

    # ------------------------------------------------------------ client side
    @property
    def inflight(self) -> int:
        return self._inflight

    def submit(
        self, prompt_ids: list[int], params: SamplingParams, loop: asyncio.AbstractEventLoop
    ) -> tuple[int, asyncio.Queue]:
        with self._cv:
            if self._inflight >= self.max_pending:
                raise QueueFull(f"{self._inflight} requests already pending")
            rid = self._next_id
            self._next_id += 1
            q: asyncio.Queue = asyncio.Queue()
            self._streams[rid] = (loop, q)
            self._inflight += 1
            self._incoming.append((rid, prompt_ids, params))
            self._cv.notify()
        return rid, q

    def abort(self, rid: int) -> None:
        with self._cv:
            if rid in self._streams:
                self._aborts.add(rid)
                self._cv.notify()

    # --------------------------------------------------------- engine thread
    def _deliver(self, ev: TokenEvent) -> None:
        with self._cv:
            target = self._streams.get(ev.req_id)
            if ev.finished or ev.error:
                if self._streams.pop(ev.req_id, None) is not None:
                    self._inflight -= 1
        if target:
            loop, q = target
            loop.call_soon_threadsafe(q.put_nowait, ev)

    def _run(self) -> None:
        eng = self.engine
        while True:
            with self._cv:
                while not (self._stop or self._incoming or self._aborts or eng.has_unfinished()):
                    self._cv.wait()
                if self._stop:
                    return
                new, self._incoming = list(self._incoming), deque()
                aborts, self._aborts = self._aborts, set()

            for rid, prompt_ids, params in new:
                try:
                    eng.add_request(prompt_ids, params, req_id=rid)
                except ValueError as e:
                    self._deliver(TokenEvent(rid, None, True, "error", error=str(e)))
            for rid in aborts:
                eng.abort(rid)

            try:
                for ev in eng.step():
                    self._deliver(ev)
                    if ev.finished:
                        eng.finished.pop(ev.req_id, None)  # don't accumulate history
            except Exception as e:  # noqa: BLE001 - keep the server alive, fail in-flight requests
                log.exception("engine step failed")
                for rid in list(self._streams):
                    self._deliver(TokenEvent(rid, None, True, "error", error=repr(e)))
                eng.reset()
