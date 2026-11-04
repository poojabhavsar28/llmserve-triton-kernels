import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from llmserve.kvcache import BlockAllocator, OutOfBlocks, PagedKVCache

ATOL = 1e-4


def _ids(cfg, B, T, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (B, T), generator=g)


def _paged(model, num_blocks=32, block_size=4):
    return PagedKVCache(model.config, num_blocks, block_size, device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------- allocator
def test_allocator_basic_and_double_free():
    a = BlockAllocator(3)
    got = [a.allocate() for _ in range(3)]
    assert sorted(got) == [0, 1, 2] and a.num_free == 0
    with pytest.raises(OutOfBlocks):
        a.allocate()
    a.free([got[0]])
    assert a.num_free == 1
    with pytest.raises(ValueError):
        a.free([got[0]])  # double free


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(["alloc", "free"]), st.integers(0, 50)), max_size=200))
def test_allocator_never_leaks_or_duplicates(ops):
    """Property: free + used == total always; ids never handed out twice; all returnable."""
    a = BlockAllocator(16)
    held: list[int] = []
    for op, x in ops:
        if op == "alloc":
            try:
                b = a.allocate()
            except OutOfBlocks:
                assert a.num_free == 0
                continue
            assert b not in held
            held.append(b)
        elif held:
            a.free([held.pop(x % len(held))])
        assert a.num_free + a.num_used == 16
        assert a.num_used == len(held)
    a.free(held)
    assert a.num_free == 16 and a.num_used == 0


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(1, 30), min_size=1, max_size=6))
def test_cache_sequences_free_everything(lengths):
    from llmserve.config import ModelConfig

    cache = PagedKVCache(ModelConfig.tiny(), 64, 4, device="cpu", dtype=torch.float32)
    for sid, n in enumerate(lengths):
        cache.add_sequence(sid)
        cache.build_batch([sid], [n])
        assert len(cache.tables[sid]) == cache.blocks_for(n)
    for sid in range(len(lengths)):
        cache.free_sequence(sid)
    assert cache.num_free_blocks == 64


# ------------------------------------------------------------ model parity
@torch.inference_mode()
def test_paged_single_sequence_matches_full_forward(tiny_model):
    ids = _ids(tiny_model.config, 1, 19)
    full = tiny_model(ids)
    cache = _paged(tiny_model, block_size=4)  # 19 tokens cross several block boundaries
    cache.add_sequence(0)

    b = cache.build_batch([0], [11])
    pre = tiny_model(ids[:, :11], paged=b)
    assert torch.allclose(pre[:, 0], full[:, 10], atol=ATOL)

    for pos in range(11, 19):
        b = cache.build_batch([0], [1])
        step = tiny_model(ids[:, pos : pos + 1], paged=b)
        assert torch.allclose(step[:, 0], full[:, pos], atol=ATOL), f"pos {pos}"


@torch.inference_mode()
def test_paged_variable_length_batch_matches_individual(tiny_model):
    """Right-padded prefill of sequences with different lengths, then a batched decode."""
    cfg = tiny_model.config
    lens = [5, 12, 8]
    seqs = [_ids(cfg, 1, n + 3, seed=10 + i)[0] for i, n in enumerate(lens)]
    cache = _paged(tiny_model, num_blocks=64, block_size=4)

    T = max(lens)
    batch_ids = torch.zeros(3, T, dtype=torch.long)
    for i, n in enumerate(lens):
        cache.add_sequence(i)
        batch_ids[i, :n] = seqs[i][:n]
    out = tiny_model(batch_ids, paged=cache.build_batch([0, 1, 2], lens))
    for i, n in enumerate(lens):
        solo = tiny_model(seqs[i][None, :n])
        assert torch.allclose(out[i, 0], solo[0, -1], atol=ATOL), f"prefill seq {i}"

    for step in range(3):  # three batched decode steps
        toks = torch.stack([seqs[i][lens[i] + step] for i in range(3)])[:, None]
        out = tiny_model(toks, paged=cache.build_batch([0, 1, 2], [1, 1, 1]))
        for i in range(3):
            solo = tiny_model(seqs[i][None, : lens[i] + step + 1])
            assert torch.allclose(out[i, 0], solo[0, -1], atol=ATOL), f"decode {step} seq {i}"


def test_build_batch_is_atomic_when_out_of_blocks(tiny_model):
    cache = _paged(tiny_model, num_blocks=2, block_size=4)
    cache.add_sequence(0)
    free_before = cache.num_free_blocks
    with pytest.raises(OutOfBlocks):
        cache.build_batch([0], [9])  # needs 3 blocks
    assert cache.num_free_blocks == free_before and cache.lengths[0] == 0


def test_fragmentation_bounded_by_one_block_per_sequence(tiny_model):
    cache = _paged(tiny_model, num_blocks=64, block_size=8)
    for sid, n in enumerate([1, 9, 17, 30]):
        cache.add_sequence(sid)
        cache.build_batch([sid], [n])
    wasted_slots = cache.allocator.num_used * 8 - sum(cache.lengths.values())
    assert 0 <= wasted_slots < 4 * 8
