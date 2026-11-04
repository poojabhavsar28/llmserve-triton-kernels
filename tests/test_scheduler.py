import pytest
import torch

from llmserve.datatypes import SamplingParams
from llmserve.engine.generate import generate
from llmserve.kvcache import PagedKVCache
from llmserve.scheduler import ContinuousBatchingEngine


def _prompt(cfg, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (n,), generator=g).tolist()


def _engine(model, *, num_blocks=64, block_size=4, max_batch=4, policy="continuous"):
    cache = PagedKVCache(model.config, num_blocks, block_size, device="cpu", dtype=torch.float32)
    return ContinuousBatchingEngine(model, cache, max_batch_size=max_batch, policy=policy)


def _reference(model, prompt, max_new):
    return generate(model, prompt, SamplingParams(max_new_tokens=max_new), device="cpu").tokens


WORKLOAD = [(5, 9), (13, 4), (8, 12), (3, 7), (17, 5), (6, 10)]  # (prompt_len, max_new)


@pytest.mark.parametrize("policy", ["continuous", "static"])
def test_batched_outputs_equal_independent_generation(tiny_model, policy):
    cfg = tiny_model.config
    eng = _engine(tiny_model, policy=policy, max_batch=3)
    prompts = [_prompt(cfg, n, 100 + i) for i, (n, _) in enumerate(WORKLOAD)]
    ids = [
        eng.add_request(p, SamplingParams(max_new_tokens=m))
        for p, (_, m) in zip(prompts, WORKLOAD)
    ]
    done = eng.run_until_complete()
    for rid, p, (_, m) in zip(ids, prompts, WORKLOAD):
        assert done[rid].generated == _reference(tiny_model, p, m), f"request {rid}"
        assert done[rid].finish_reason == "length"
    assert eng.cache.num_free_blocks == eng.cache.num_blocks  # nothing leaked


def test_requests_arriving_mid_flight_join_running_batch(tiny_model):
    cfg = tiny_model.config
    eng = _engine(tiny_model, max_batch=4)
    p0, p1 = _prompt(cfg, 6, 1), _prompt(cfg, 9, 2)
    r0 = eng.add_request(p0, SamplingParams(max_new_tokens=10))
    for _ in range(4):
        eng.step()
    assert len(eng.running) == 1
    r1 = eng.add_request(p1, SamplingParams(max_new_tokens=6))
    eng.step()
    assert len(eng.running) == 2  # joined without waiting for r0 to finish
    done = eng.run_until_complete()
    assert done[r0].generated == _reference(tiny_model, p0, 10)
    assert done[r1].generated == _reference(tiny_model, p1, 6)


def test_static_policy_does_not_admit_until_batch_drains(tiny_model):
    cfg = tiny_model.config
    eng = _engine(tiny_model, policy="static", max_batch=4)
    eng.add_request(_prompt(cfg, 5, 1), SamplingParams(max_new_tokens=8))
    eng.step()
    eng.add_request(_prompt(cfg, 5, 2), SamplingParams(max_new_tokens=8))
    eng.step()
    assert len(eng.running) == 1 and len(eng.waiting) == 1


def test_preemption_recompute_preserves_greedy_outputs(tiny_model):
    cfg = tiny_model.config
    # Pool so small that 4 concurrent sequences cannot all grow -> forces preemption.
    eng = _engine(tiny_model, num_blocks=14, block_size=4, max_batch=4)
    work = [(6, 14), (6, 14), (6, 14), (6, 14)]
    prompts = [_prompt(cfg, n, 50 + i) for i, (n, _) in enumerate(work)]
    ids = [
        eng.add_request(p, SamplingParams(max_new_tokens=m)) for p, (_, m) in zip(prompts, work)
    ]
    done = eng.run_until_complete()
    assert eng.stats.preemptions > 0, "test must actually exercise preemption"
    for rid, p, (_, m) in zip(ids, prompts, work):
        assert done[rid].generated == _reference(tiny_model, p, m)
    assert eng.cache.num_free_blocks == eng.cache.num_blocks


def test_eos_finishes_early_and_frees_blocks(tiny_model):
    cfg = tiny_model.config
    p = _prompt(cfg, 7, 9)
    first = _reference(tiny_model, p, 1)[0]
    eng = _engine(tiny_model)
    rid = eng.add_request(p, SamplingParams(max_new_tokens=20, eos_token_id=first))
    done = eng.run_until_complete()
    assert done[rid].generated == [first] and done[rid].finish_reason == "stop"
    assert eng.cache.num_free_blocks == eng.cache.num_blocks


def test_abort_waiting_and_running(tiny_model):
    cfg = tiny_model.config
    eng = _engine(tiny_model, max_batch=1)
    a = eng.add_request(_prompt(cfg, 5, 1), SamplingParams(max_new_tokens=30))
    b = eng.add_request(_prompt(cfg, 5, 2), SamplingParams(max_new_tokens=30))
    eng.step()
    eng.step()
    eng.abort(b)  # still waiting (max_batch=1)
    eng.abort(a)  # running
    events = eng.step()
    assert {e.req_id for e in events if e.finished} == {a, b}
    assert not eng.has_unfinished()
    assert eng.cache.num_free_blocks == eng.cache.num_blocks
    assert eng.finished[a].finish_reason == "abort"


def test_validate_rejects_impossible_requests(tiny_model):
    eng = _engine(tiny_model, num_blocks=4, block_size=4)  # 16 token slots total
    with pytest.raises(ValueError):
        eng.add_request([1] * 10, SamplingParams(max_new_tokens=10))  # 20 > 16 slots
    with pytest.raises(ValueError):
        eng.add_request([1] * 600, SamplingParams(max_new_tokens=5))  # beyond model limit
    with pytest.raises(ValueError):
        eng.add_request([], SamplingParams(max_new_tokens=5))


def test_sampling_with_seed_is_reproducible_across_engines(tiny_model):
    cfg = tiny_model.config
    p = _prompt(cfg, 6, 3)
    params = SamplingParams(temperature=0.9, top_k=20, seed=1234, max_new_tokens=12)
    outs = []
    for _ in range(2):
        eng = _engine(tiny_model)
        rid = eng.add_request(p, params)
        outs.append(eng.run_until_complete()[rid].generated)
    assert outs[0] == outs[1]
