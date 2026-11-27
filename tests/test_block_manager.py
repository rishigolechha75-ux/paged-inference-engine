import pytest

from pagedserve.block_manager import BlockAllocator, BlockManager, OutOfBlocks, block_hashes
from pagedserve.sequence import SamplingParams, Sequence


def make_seq(i, tokens, max_tokens=8):
    return Sequence(seq_id=i, prompt_token_ids=list(tokens), params=SamplingParams(max_tokens=max_tokens))


def test_block_hashes_chain_and_only_full_blocks():
    a = block_hashes([1, 2, 3, 4, 5], 2)
    assert len(a) == 2
    # same block contents at a different position must hash differently
    b = block_hashes([3, 4, 3, 4], 2)
    assert b[0] != b[1]
    assert block_hashes([1, 2, 3, 4], 2) == a


def test_allocator_free_and_double_free():
    alloc = BlockAllocator(2)
    x, y = alloc.allocate(), alloc.allocate()
    assert alloc.num_free_blocks == 0
    with pytest.raises(OutOfBlocks):
        alloc.allocate()
    alloc.free(x)
    assert alloc.num_free_blocks == 1
    with pytest.raises(ValueError):
        alloc.free(x)
    alloc.free(y)


def test_lazy_allocation_grows_one_block_at_a_time():
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq(0, range(5))
    bm.admit(seq, reserve_tokens=5)
    assert len(seq.block_table) == 2
    bm.ensure(seq, 8)
    assert len(seq.block_table) == 2
    bm.ensure(seq, 9)
    assert len(seq.block_table) == 3
    bm.free(seq)
    assert bm.num_free_blocks == 8 and seq.block_table == []


def test_can_admit_respects_budget():
    bm = BlockManager(num_blocks=3, block_size=4)
    assert bm.can_admit(make_seq(0, range(12)), 12)
    assert not bm.can_admit(make_seq(0, range(13)), 13)


def test_prefix_cache_shares_blocks_and_skips_last_token():
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_caching=True)
    a = make_seq(0, range(10))
    bm.admit(a, 10)
    a.num_computed_tokens = 10
    bm.commit(a)
    b = make_seq(1, list(range(8)) + [99, 98])
    bm.admit(b, 10)
    assert b.num_computed_tokens == 8
    assert b.block_table[:2] == a.block_table[:2]
    assert bm.allocator.ref_counts[a.block_table[0]] == 2
    # a prompt that is exactly 2 cached blocks reuses only one block: the
    # last token must be recomputed to produce logits
    c = make_seq(2, range(8))
    bm.admit(c, 8)
    assert c.num_computed_tokens == 4


def test_freed_cached_blocks_are_reused_then_evicted_lru():
    bm = BlockManager(num_blocks=3, block_size=2, enable_prefix_caching=True)
    a = make_seq(0, [1, 2, 3, 4, 5])
    bm.admit(a, 5)
    a.num_computed_tokens = 5
    bm.commit(a)
    bm.free(a)
    assert bm.num_free_blocks == 3
    b = make_seq(1, [1, 2, 3, 4, 7])
    assert bm.can_admit(b, 5)
    bm.admit(b, 5)
    assert b.num_computed_tokens == 4
    bm.free(b)
    # an unrelated sequence needing all three blocks evicts the cached ones
    c = make_seq(2, [9, 9, 9, 9, 9, 9])
    bm.admit(c, 6)
    assert bm.allocator.num_evictions == 2
    d = make_seq(3, [1, 2, 3, 4, 5])
    bm.free(c)
    bm.admit(d, 5)
    assert d.num_computed_tokens == 0


def test_can_admit_counts_evictable_hits_against_free_pool():
    bm = BlockManager(num_blocks=2, block_size=2, enable_prefix_caching=True)
    a = make_seq(0, [1, 2, 3])
    bm.admit(a, 3)
    a.num_computed_tokens = 3
    bm.commit(a)
    bm.free(a)
    b = make_seq(1, [1, 2, 3, 4, 5])
    # one cached hit (evictable, so it leaves the free pool) + 2 new blocks > 2
    assert not bm.can_admit(b, 5)
    assert bm.can_admit(b, 4)
