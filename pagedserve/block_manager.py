"""Paged KV-cache memory management.

The KV cache is a pool of fixed-size blocks (pages). Each sequence owns a
*block table*: the list of physical block ids that hold its tokens in order,
so token ``i`` lives in block ``block_table[i // block_size]`` at offset
``i % block_size``. Memory is allocated one block at a time as a sequence
grows, so the only waste is the unfilled tail of the last block.

With prefix caching enabled, every *full* block is identified by a hash of
its tokens chained with the hash of the previous block. Two requests with
the same prefix map to the same physical blocks (reference counted). Blocks
whose reference count drops to zero stay in an LRU pool and keep their
contents until the memory is needed again.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Deque, Dict, List, Optional, Sequence as Seq, Tuple

from .sequence import Sequence


def block_hashes(token_ids: Seq[int], block_size: int) -> List[int]:
    """Chained hashes for every full block of ``token_ids``.

    Chaining makes a block's identity depend on the whole prefix, not just
    its own tokens, so identical blocks at different positions never alias.
    """
    hashes: List[int] = []
    parent = 0
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = hash((parent, tuple(token_ids[start : start + block_size])))
        hashes.append(parent)
    return hashes


class OutOfBlocks(RuntimeError):
    """Raised when an allocation is attempted with no free block."""


class BlockAllocator:
    """Reference-counted physical block pool with an LRU of cached blocks."""

    def __init__(self, num_blocks: int, enable_prefix_caching: bool = False) -> None:
        self.num_blocks = num_blocks
        self.enable_prefix_caching = enable_prefix_caching
        self.ref_counts = [0] * num_blocks
        self.block_hash: List[Optional[int]] = [None] * num_blocks
        self._free: Deque[int] = deque(range(num_blocks))
        # hash -> block id for blocks with ref 0 that still hold valid data.
        self._evictable: "OrderedDict[int, int]" = OrderedDict()
        # hash -> block id for every block that currently has a hash.
        self._hash_to_block: Dict[int, int] = {}
        self.num_evictions = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self._free) + len(self._evictable)

    def is_evictable(self, block_hash: int) -> bool:
        return block_hash in self._evictable

    def allocate(self) -> int:
        """Return a fresh block with ref count 1, evicting LRU if needed."""
        if self._free:
            block = self._free.popleft()
        elif self._evictable:
            h, block = self._evictable.popitem(last=False)
            del self._hash_to_block[h]
            self.block_hash[block] = None
            self.num_evictions += 1
        else:
            raise OutOfBlocks("KV cache is full")
        self.ref_counts[block] = 1
        return block

    def lookup(self, block_hash: int) -> Optional[int]:
        """Take a reference to the cached block with this hash, if any."""
        block = self._hash_to_block.get(block_hash)
        if block is None:
            return None
        if self.ref_counts[block] == 0:
            self._evictable.pop(block_hash, None)
        self.ref_counts[block] += 1
        return block

    def peek(self, block_hash: int) -> Optional[int]:
        return self._hash_to_block.get(block_hash)

    def register(self, block: int, block_hash: int) -> None:
        """Mark a full block as reusable under ``block_hash``."""
        if not self.enable_prefix_caching or self.block_hash[block] is not None:
            return
        if block_hash in self._hash_to_block:
            return  # an identical block is already cached; keep that one
        self.block_hash[block] = block_hash
        self._hash_to_block[block_hash] = block

    def free(self, block: int) -> None:
        if self.ref_counts[block] <= 0:
            raise ValueError(f"double free of block {block}")
        self.ref_counts[block] -= 1
        if self.ref_counts[block] == 0:
            h = self.block_hash[block]
            if h is not None:
                self._evictable[h] = block
            else:
                self._free.append(block)


class BlockManager:
    """Maps sequences to physical blocks on top of :class:`BlockAllocator`."""

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = False) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks, enable_prefix_caching)
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_hit_tokens = 0
        self.prefix_query_tokens = 0

    # ------------------------------------------------------------------ util
    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks

    @property
    def num_used_blocks(self) -> int:
        return self.allocator.num_blocks - self.allocator.num_free_blocks

    def _prefix_hits(self, seq: Sequence) -> Tuple[List[int], int]:
        """Cached-block hashes usable for ``seq`` and how many are evictable.

        The final prompt token is never served from cache: its logits are
        needed to sample the first output token.
        """
        if not self.enable_prefix_caching:
            return [], 0
        tokens = seq.all_token_ids
        usable = (len(tokens) - 1) // self.block_size
        hits: List[int] = []
        evictable = 0
        for h in block_hashes(tokens, self.block_size)[:usable]:
            if self.allocator.peek(h) is None:
                break
            hits.append(h)
            evictable += self.allocator.is_evictable(h)
        return hits, evictable

    # ------------------------------------------------------------- admission
    def can_admit(self, seq: Sequence, reserve_tokens: int) -> bool:
        """Whether ``seq`` can start with blocks for ``reserve_tokens`` tokens."""
        hits, evictable = self._prefix_hits(seq)
        needed = max(self.blocks_for(reserve_tokens) - len(hits), 0)
        return needed <= self.num_free_blocks - evictable

    def admit(self, seq: Sequence, reserve_tokens: int) -> None:
        """Attach cached prefix blocks and reserve blocks for ``reserve_tokens``."""
        assert not seq.block_table, "sequence already holds blocks"
        hits, _ = self._prefix_hits(seq)
        for h in hits:
            block = self.allocator.lookup(h)
            assert block is not None
            seq.block_table.append(block)
        seq.num_computed_tokens = len(hits) * self.block_size
        if seq.num_preemptions == 0:
            seq.num_cached_prompt_tokens = seq.num_computed_tokens
            self.prefix_hit_tokens += seq.num_computed_tokens
            self.prefix_query_tokens += len(seq.prompt_token_ids)
        self.ensure(seq, reserve_tokens)

    # ---------------------------------------------------------------- growth
    def can_grow(self, seq: Sequence, num_tokens: int) -> bool:
        return self.blocks_for(num_tokens) - len(seq.block_table) <= self.num_free_blocks

    def ensure(self, seq: Sequence, num_tokens: int) -> None:
        """Allocate blocks until ``seq`` can hold ``num_tokens`` tokens."""
        while len(seq.block_table) < self.blocks_for(num_tokens):
            seq.block_table.append(self.allocator.allocate())

    def commit(self, seq: Sequence) -> None:
        """Register blocks that became full after a forward pass."""
        if not self.enable_prefix_caching:
            return
        full = seq.num_computed_tokens // self.block_size
        for i, h in enumerate(block_hashes(seq.all_token_ids[: full * self.block_size], self.block_size)):
            self.allocator.register(seq.block_table[i], h)

    def free(self, seq: Sequence) -> None:
        """Release every block of ``seq`` (tail first so heads stay cached longer)."""
        for block in reversed(seq.block_table):
            self.allocator.free(block)
        seq.block_table = []

    @property
    def prefix_hit_rate(self) -> float:
        if self.prefix_query_tokens == 0:
            return 0.0
        return self.prefix_hit_tokens / self.prefix_query_tokens
