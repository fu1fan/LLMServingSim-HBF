"""Capacity-bounded lower-tier KV cache semantics on the block pool.

The fork's agentic + prefix-caching crash was a radix-tree growing without
bound, so the memory model's BlockStored drain pushed ``npu_used`` past the
device limit and ``allocate`` raised RuntimeError. The purpose of this test is
to pin the semantics that the new ``BlockPool`` gives us for that same
crash, so we do not reintroduce the bug.

Key fact: upstream ``BlockPool.cache_copy`` never raises on overflow. It pops
a block, and ``get_new_blocks`` evicts any cached hash that block still carries
(``_maybe_evict_cached_block``), so the free list can always supply at least one
block regardless of how full the pool is -- overflow is LRU eviction, never a
RuntimeError or a dropped write-through. This is the built-in defence against
the old radix-tree unbounded-growth crash.
"""

import unittest

from serving.core.block_pool import BlockPool, Device

B = 16


class BackpressureTest(unittest.TestCase):
    def test_overflow_evicts_lru_and_places_new_copy(self):
        """Overflow drops the least-recently-written cached block and succeeds.

        This is exactly what the module docstring and the block_pool self-test
        assert: 'overflow drops the least recently written'. cache_copy must
        never raise, and the size of the index must stay bounded at the pool
        size since a full pool evicts to make room.
        """
        # A 2-block pool: fill the index to capacity, then write one more.
        pool = BlockPool(Device.CPU, 2, B, B * 128, enable_caching=True)
        self.assertTrue(pool.cache_copy(1001))
        self.assertTrue(pool.cache_copy(1002))
        self.assertEqual(len(pool.cached_block_hash_to_block), 2)

        # Writing a third hash over a full pool evicts the oldest copy and
        # still reports success -- no drop, no raise.
        self.assertTrue(pool.cache_copy(1003))
        self.assertEqual(len(pool.cached_block_hash_to_block), 2)
        # The oldest (1001) was evicted, the newest two survive.
        self.assertIsNone(pool.get_cached_block(1001))
        self.assertIsNotNone(pool.get_cached_block(1002))
        self.assertIsNotNone(pool.get_cached_block(1003))

    def test_duplicate_hash_does_not_recharge(self):
        """Re-copying a resident hash is a no-op that returns False."""
        pool = BlockPool(Device.CPU, 2, B, B * 128, enable_caching=True)
        self.assertTrue(pool.cache_copy(42))
        self.assertFalse(pool.cache_copy(42), "already resident, must not re-charge")

    def test_npu_allocation_is_all_or_nothing(self):
        """The NPU pool never over-admits: get_new_blocks raises on a request
        larger than the free list, leaving the pool untouched."""
        pool = BlockPool(Device.NPU, 4, B, B * 128, enable_caching=True)
        pool.get_new_blocks(4)
        self.assertEqual(pool.get_num_free_blocks(), 0)
        with self.assertRaises(RuntimeError):
            pool.get_new_blocks(1)
        self.assertEqual(pool.get_num_free_blocks(), 0)


if __name__ == "__main__":
    unittest.main()
