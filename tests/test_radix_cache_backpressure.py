"""Capacity-bounded radix cache (KV backpressure).

Regression tests for the agentic + prefix-caching crash: with the radix tree
growing without bound, the BlockStored events drained by
``MemoryModel.apply_kv_cache_events`` push ``npu_used`` past the device limit
and ``MemoryModel.allocate`` raises RuntimeError (the
``qwen/llama agentic-k1p04`` matrix failures). The fix bounds the insert at the
radix layer: evict unlocked LRU leaves for room, then store only what fits and
skip the rest — graceful cache degradation instead of a crash.
"""

import unittest

from serving.core.memory_model import GB_TO_BYTE, Device, MemoryModel
from serving.core.radix_tree import RadixCache
from serving.core.request import Request

PAGE = 16


def _placement_hbf():
    return {
        "default": {
            "weights": "HBF",
            "kv_loc": "LOCAL",
            "kv_evict_loc": "REMOTE:0",
        },
        "block": [],
        "layer": {},
    }


def _hbf_config(stack_capacity_gb=512):
    return {
        "schema_version": 1,
        "num_stacks": 8,
        "stack_capacity_gb": stack_capacity_gb,
        "mem_size": 8 * stack_capacity_gb,
        "performance": {
            "source": "scale",
            "latency_scale": 1.0,
        },
    }


class RadixBackpressureTest(unittest.TestCase):
    def test_unlocked_inserts_evict_lru_when_full(self):
        """Normal cache pressure: unlocked leaves are LRU-evicted to stay
        within capacity (this is the pre-existing eviction path)."""
        tree = RadixCache(
            node_id=0, device="NPU", page_size=PAGE,
            capacity=3 * PAGE, kv_size=1, enable_kv_cache_events=True,
        )
        for i in range(4):
            key = list(range(i * 2 * PAGE, i * 2 * PAGE + 2 * PAGE))
            tree.insert(key)
        self.assertLessEqual(tree.total_size(), 3 * PAGE)
        self.assertGreater(tree.evictable_size(), 0)

    def test_locked_agentic_prefixes_stay_bounded(self):
        """Agentic repro: strictly-extending prefixes, each locked via
        inc_lock_ref (as cache_unfinished_req does). Eviction cannot help
        (working set locked), so inserts must be truncated/skipped. The tree
        must never exceed capacity and must not raise."""
        capacity_tokens = 5 * PAGE
        tree = RadixCache(
            node_id=0, device="NPU", page_size=PAGE,
            capacity=capacity_tokens, kv_size=1, enable_kv_cache_events=True,
        )
        pos = 0
        for _turn in range(20):
            key = list(range(pos + 2 * PAGE))
            pos += 2 * PAGE
            tree.insert(key)
            res = tree.match_prefix(key)
            tree.inc_lock_ref(res.last_device_node)
            self.assertLessEqual(tree.total_size(), capacity_tokens)
        self.assertGreater(tree.dropped_tokens, 0)

    def test_truncated_insert_keeps_tree_intact(self):
        """After a truncated insert, every node is still reachable from the
        root (no orphans) and later matches return a non-decreasing hit."""
        tree = RadixCache(
            node_id=0, device="NPU", page_size=PAGE,
            capacity=4 * PAGE, kv_size=1, enable_kv_cache_events=True,
        )
        pos = 0
        hits = []
        for _turn in range(8):
            key = list(range(pos + 2 * PAGE))
            pos += 2 * PAGE
            tree.insert(key)
            res = tree.match_prefix(key)
            tree.inc_lock_ref(res.last_device_node)
            hits.append(res.hit_length)
        self.assertEqual(hits, sorted(hits))
        self.assertLessEqual(tree.total_size(), 4 * PAGE)

        # Reachability recount must equal the maintained total_size.
        def recount(node):
            n = len(node.key)
            for child in node.children.values():
                n += recount(child)
            return n

        self.assertEqual(recount(tree.root_node), tree.total_size())

    def test_insert_under_capacity_is_unchanged(self):
        """Regression: below capacity, inserts store the whole key and emit
        one BlockStored per page, exactly as before the fix."""
        tree = RadixCache(
            node_id=0, device="NPU", page_size=PAGE,
            capacity=10 * PAGE, kv_size=1, enable_kv_cache_events=True,
        )
        tree.insert(list(range(2 * PAGE)))
        self.assertEqual(tree.total_size(), 2 * PAGE)
        stored_pages = 0
        for ev in tree.take_events():
            if type(ev).__name__ == "BlockStored":
                stored_pages += 1
        self.assertEqual(stored_pages, 2)
        self.assertEqual(tree.dropped_tokens, 0)

    def test_cpu_tier_page_size_one_bounded(self):
        """The second-tier cache uses page_size=1; bounded insert applies
        there too (total_memory_usage must never exceed capacity)."""
        tree = RadixCache(
            node_id=0, device="CPU", page_size=1,
            capacity=100, kv_size=10, enable_kv_cache_events=True,
        )
        pos = 0
        for _turn in range(10):
            key = list(range(pos + 30))
            pos += 30
            tree.insert(key)
            res = tree.match_prefix(key)
            tree.inc_lock_ref(res.last_device_node)
        self.assertLessEqual(tree.total_memory_usage(), tree.capacity)


class MemoryModelBackpressureTest(unittest.TestCase):
    def _tight_memory(self, npu_mem_gb=1):
        # Weights go to HBF so mem_for_kv == npu_mem, giving a tight KV budget
        # (~8k tokens for Llama-3.1-8B at 128 KB/token, tp1).
        return MemoryModel(
            model="meta-llama/Llama-3.1-8B",
            instance_id=0,
            node_id=0,
            num_npus=1,
            tp_size=1,
            npu_mem=npu_mem_gb,
            cpu_mem=512,
            block_size=PAGE,
            fp=16,
            enable_prefix_caching=True,
            enable_prefix_sharing=False,
            prefix_pool=None,
            prefix_storage=None,
            pp_size=1,
            placement=_placement_hbf(),
            hbf_mem=_hbf_config(),
        )

    def _agentic_request(self, req_id, context, output_len=64):
        req = Request(
            id=req_id,
            model="meta-llama/Llama-3.1-8B",
            input=len(context),
            output=output_len,
            arrival=0,
            instance_id=0,
            input_hash_ids=list(context),
            output_hash_ids=list(range(1_000_000, 1_000_000 + output_len)),
        )
        # The whole turn's context (input + output) is computed this round.
        req.num_computed_tokens = len(context) + output_len
        return req

    def test_agentic_turns_never_overflow_npu(self):
        """The exact crash path: add_done -> cache_unfinished_req ->
        apply_kv_cache_events -> allocate. With a growing locked agentic
        context this used to raise RuntimeError once the prefix cache filled;
        now it must stay bounded and never exceed npu_mem."""
        memory = self._tight_memory()
        # Extend the cumulative context far past the ~8k-token KV budget.
        context = []
        for turn in range(60):
            context = context + list(range(turn * 512, (turn + 1) * 512))
            req = self._agentic_request(turn, context)
            memory.cache_unfinished_req(req, Device.NPU)
            self.assertLessEqual(
                memory.npu_used, memory.npu_mem,
                f"turn {turn}: npu_used {memory.npu_used / GB_TO_BYTE:.3f}GB "
                f"> npu_mem {memory.npu_mem / GB_TO_BYTE:.3f}GB",
            )

    def test_event_accounting_matches_tree(self):
        """npu_used (minus weights) must equal cached tokens * bytes/token."""
        memory = self._tight_memory()
        context = []
        for turn in range(20):
            context = context + list(range(turn * 512, (turn + 1) * 512))
            req = self._agentic_request(turn, context)
            memory.cache_unfinished_req(req, Device.NPU)
        cached_bytes = memory.npu_prefix_cache.total_size() * memory._bytes_per_token
        self.assertEqual(memory.npu_used, cached_bytes)
        self.assertLessEqual(memory.npu_used, memory.npu_mem)


if __name__ == "__main__":
    unittest.main()
