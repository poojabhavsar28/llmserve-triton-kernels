import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from llmserve.datatypes import SamplingParams  # noqa: E402
from llmserve.engine.generate import generate  # noqa: E402
from llmserve.kvcache import PagedKVCache  # noqa: E402
from llmserve.scheduler import ContinuousBatchingEngine  # noqa: E402
from llmserve.server import AsyncEngine, create_app  # noqa: E402


def _prompt(cfg, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (n,), generator=g).tolist()


def _ref(model, prompt, n):
    return generate(model, prompt, SamplingParams(max_new_tokens=n), device="cpu").tokens


@pytest.fixture()
def client(tiny_model):
    app = create_app(tiny_model, None, num_blocks=128, block_size=4, max_batch_size=4)
    with TestClient(app) as c:
        yield c


def test_health_models_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json()["data"][0]["object"] == "model"
    m = client.get("/metrics").json()
    assert m["running"] == 0 and m["free_blocks"] == m["total_blocks"] == 128


def test_completion_matches_reference(client, tiny_model):
    p = _prompt(tiny_model.config, 9, 1)
    r = client.post("/v1/completions", json={"prompt": p, "max_tokens": 10, "temperature": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["token_ids"] == _ref(tiny_model, p, 10)
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19}


def test_streaming_sse_matches_non_streaming(client, tiny_model):
    p = _prompt(tiny_model.config, 7, 2)
    payload = {"prompt": p, "max_tokens": 8, "temperature": 0, "stream": True}
    with client.stream("POST", "/v1/completions", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    toks = [t for c in chunks for t in c["choices"][0]["token_ids"]]
    assert toks == _ref(tiny_model, p, 8)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_concurrent_requests_all_correct(client, tiny_model):
    cfg = tiny_model.config
    jobs = [(5 + 3 * i, 6 + i) for i in range(8)]  # more requests than max_batch_size=4
    prompts = [_prompt(cfg, n, 20 + i) for i, (n, _) in enumerate(jobs)]

    def call(i):
        r = client.post(
            "/v1/completions",
            json={"prompt": prompts[i], "max_tokens": jobs[i][1], "temperature": 0},
        )
        return r.json()["choices"][0]["token_ids"]

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(call, range(8)))
    for i, got in enumerate(results):
        assert got == _ref(tiny_model, prompts[i], jobs[i][1]), f"request {i}"
    m = client.get("/metrics").json()
    assert m["free_blocks"] == m["total_blocks"] and m["inflight"] == 0


def test_validation_errors(client, tiny_model):
    assert client.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 9999}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": "hello"}).status_code == 400  # no tokenizer
    assert client.post("/v1/completions", json={"max_tokens": 3}).status_code == 422
    assert client.post("/v1/completions", json={"prompt": [1], "max_tokens": 0}).status_code == 422


def test_backpressure_returns_429(tiny_model):
    app = create_app(tiny_model, None, num_blocks=128, block_size=4, max_batch_size=1, max_pending=1)
    with TestClient(app) as c:
        long = {"prompt": [1, 2, 3], "max_tokens": 200, "temperature": 0}
        t = threading.Thread(target=lambda: c.post("/v1/completions", json=long))
        t.start()
        deadline = time.time() + 5
        while c.get("/metrics").json()["inflight"] < 1 and time.time() < deadline:
            time.sleep(0.01)
        r = c.post("/v1/completions", json={"prompt": [1], "max_tokens": 2})
        assert r.status_code == 429
        t.join(timeout=30)


def test_abort_frees_resources(tiny_model):
    async def main():
        cache = PagedKVCache(tiny_model.config, 128, 4, "cpu", torch.float32)
        eng = ContinuousBatchingEngine(tiny_model, cache, max_batch_size=2)
        ae = AsyncEngine(eng, max_pending=4)
        ae.start()
        try:
            loop = asyncio.get_running_loop()
            rid, q = ae.submit([1, 2, 3, 4], SamplingParams(max_new_tokens=300), loop)
            first = await asyncio.wait_for(q.get(), 10)  # it is generating
            assert first.token is not None
            ae.abort(rid)
            while True:
                ev = await asyncio.wait_for(q.get(), 10)
                if ev.finished:
                    break
            assert ev.finish_reason == "abort"
            await asyncio.sleep(0.05)
            assert cache.num_free_blocks == cache.num_blocks and ae.inflight == 0
        finally:
            ae.stop()

    asyncio.run(main())


def test_engine_side_validation_error_is_delivered(tiny_model):
    """A request the engine rejects must surface as an error event, not hang the client."""
    async def main():
        cache = PagedKVCache(tiny_model.config, 8, 4, "cpu", torch.float32)  # only 32 slots
        ae = AsyncEngine(ContinuousBatchingEngine(tiny_model, cache), max_pending=4)
        ae.start()
        try:
            _, q = ae.submit([1, 2, 3], SamplingParams(max_new_tokens=100), asyncio.get_running_loop())
            ev = await asyncio.wait_for(q.get(), 10)
            assert ev.error and "KV blocks" in ev.error and ae.inflight == 0
        finally:
            ae.stop()

    asyncio.run(main())
