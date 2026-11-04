"""Phase 5: OpenAI-compatible HTTP API (``/v1/completions``) with SSE streaming.

Extensions over the OpenAI schema: ``top_k``, and each choice carries ``token_ids`` (also lets
the server run with no tokenizer, taking ``prompt`` as a list of token ids).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llmserve.datatypes import SamplingParams
from llmserve.kvcache.paged import PagedKVCache
from llmserve.scheduler.engine import ContinuousBatchingEngine
from llmserve.server.async_engine import AsyncEngine, QueueFull


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = Field(16, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    seed: int | None = None
    stream: bool = False


class IncrementalDetokenizer:
    """Turns a growing token list into text deltas, holding back incomplete UTF-8 sequences."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.emitted = ""

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("\ufffd") or not text.startswith(self.emitted):
            return ""
        delta, self.emitted = text[len(self.emitted) :], text
        return delta


def create_app(
    model,
    tokenizer=None,
    *,
    model_name: str = "llmserve",
    num_blocks: int = 512,
    block_size: int = 16,
    max_batch_size: int = 8,
    max_pending: int = 64,
    policy: str = "continuous",
) -> FastAPI:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cache = PagedKVCache(model.config, num_blocks, block_size, device, dtype)
    engine = ContinuousBatchingEngine(model, cache, max_batch_size=max_batch_size, policy=policy)
    aeng = AsyncEngine(engine, max_pending=max_pending)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        aeng.start()
        yield
        aeng.stop()

    app = FastAPI(title="llmserve", lifespan=lifespan)
    app.state.async_engine = aeng

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return {**engine.snapshot(), "inflight": aeng.inflight, "max_pending": max_pending}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if isinstance(req.prompt, str):
            if tokenizer is None:
                raise HTTPException(400, "this server has no tokenizer; send `prompt` as token ids")
            prompt_ids = tokenizer(req.prompt).input_ids
        else:
            prompt_ids = req.prompt
        eos = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        params = SamplingParams(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_new_tokens=req.max_tokens,
            eos_token_id=eos,
            seed=req.seed,
        )
        try:
            engine.validate_request(len(prompt_ids), req.max_tokens)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        try:
            rid, q = aeng.submit(prompt_ids, params, asyncio.get_running_loop())
        except QueueFull as e:
            raise HTTPException(429, f"server overloaded: {e}") from e

        cid, created = f"cmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(text: str, toks: list[int], reason: str | None) -> dict:
            return {
                "id": cid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": text, "token_ids": toks, "finish_reason": reason}],
            }

        if req.stream:
            async def sse():
                detok = IncrementalDetokenizer(tokenizer) if tokenizer is not None else None
                done = False
                try:
                    while True:
                        ev = await q.get()
                        if ev.error:
                            err = {"error": {"message": ev.error, "type": "server_error"}}
                            yield f"data: {json.dumps(err)}\n\n"
                            done = True
                            break
                        toks = [ev.token] if ev.token is not None else []
                        text = detok.push(ev.token) if (detok and toks) else ""
                        yield f"data: {json.dumps(chunk(text, toks, ev.finish_reason))}\n\n"
                        if ev.finished:
                            done = True
                            break
                    yield "data: [DONE]\n\n"
                finally:
                    if not done:  # client went away mid-stream -> stop spending GPU on it
                        aeng.abort(rid)

            return StreamingResponse(sse(), media_type="text/event-stream")

        tokens: list[int] = []
        reason = None
        try:
            while True:
                ev = await q.get()
                if ev.error:
                    raise HTTPException(500, ev.error)
                if ev.token is not None:
                    tokens.append(ev.token)
                if ev.finished:
                    reason = ev.finish_reason
                    break
        finally:
            if reason is None:
                aeng.abort(rid)
        text = tokenizer.decode(tokens, skip_special_tokens=True) if tokenizer is not None else ""
        body = chunk(text, tokens, reason)
        body["usage"] = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt_ids) + len(tokens),
        }
        return JSONResponse(body)

    return app


def default_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32
