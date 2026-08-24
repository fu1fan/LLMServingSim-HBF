"""Capacity-bounded lower-tier KV cache (backpressure).

The fork's agentic + prefix-caching crash was a radix-tree that grew without
bound, so the memory model's BlockStored drain pushed ``npu_used`` past the
device limit and ``allocate`` raised RuntimeError. Upstream replaced the radix
tree with a per-tier ``BlockPool`` + ``TieredKVCacheManager``: the NPU pool is
all-or-nothing (``allocate_slots`` returns None without mutating), and the
inclusive write-through to a lower tier (``BlockPool.cache_copy``) previously
raised when that tier's free list was empty.

The regressions to hold here, mirroring the fork's intent on the new
architecture:
* a full lower tier must degrade to a graceful drop (data stays on the NPU, a
  later recall recomputes it) rather than raise;
* the NPU pool never over-admits and never mutates on a rejected allocation.
"""

import unittest

from serving.core.block_pool import BlockPool, Device

B = 16


class BackpressureTest(unittest.TestCase):
    def test_full_lower_tier_drops_gracefully(self):
        """A full victim tier must return False from cache_copy, never raise."""
        pool = BlockPool(Device.CPU, 1, B, B * 128, enable_caching=True)
        # Usurp the only free block with a live (pinned) allocation.
        pool.get_new_blocks(1)
        self.assertEqual(pool.get_num_free_blocks(), 0)
        # write-through into the full tier: graceful drop, no RuntimeError.
        self.assertFalse(pool.cache_copy(9999))
        # The hash was not recorded.
        self.assertIsNone(pool.get_cached_block(9999))

    def test_partially_full_lower_tier_still_places_copy(self):
        """With free capacity, cache_copy places the copy and reports True."""
        pool = BlockPool(Device.CPU, 2, B, B * 128, enable_caching=True)
        pool.get_new_blocks(1)
        self.assertTrue(pool.cache_copy(42))
        self.assertIsNotNone(pool.get_cached_block(42))

    def test_npu_allocation_is_all_or_nothing(self):
        """The NPU pool never over-admits: get_new_blocks raises on a request
        larger than the free list, leaving the pool untouched."""
        pool = BlockPool(Device.NPU, 4, B, B * 128, enable_caching=True)
        pool.get_new_blocks(4)
        self.assertEqual(pool.get_num_free_blocks(), 0)
        with self.assertRaises(RuntimeError):
            pool.get_new_blocks(1)
        # Still no free blocks; the failed request consumed nothing.
        self.assertEqual(pool.get_num_free_blocks(), 0)


if __name__ == "__main__":
    unittest.main()
