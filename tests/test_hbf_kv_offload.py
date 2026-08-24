import unittest

from serving.core.memory_model import (
    Device,
    MemoryModel,
    _device_from_placement_loc,
)


def _placement(weight_location="HBF", kv_evict_loc="HBF"):
    return {
        "default": {
            "weights": weight_location,
            "kv_loc": "LOCAL",
            "kv_evict_loc": kv_evict_loc,
        },
        "block": [],
        "layer": {},
    }


def _hbf_config(stack_capacity_gb=512, source="scale"):
    performance = {"source": "scale", "latency_scale": 1.0}
    config = {
        "schema_version": 1,
        "num_stacks": 8,
        "stack_capacity_gb": stack_capacity_gb,
        "mem_size": 8 * stack_capacity_gb,
        "performance": performance,
    }
    return config


def _memory(npu_mem=96, placement=None, hbf_mem=None, tp=1, pp=1,
            prefix_storage=None, enable_prefix_sharing=False):
    return MemoryModel(
        model="meta-llama/Llama-3.1-8B",
        instance_id=0,
        node_id=0,
        num_npus=tp * pp,
        tp_size=tp,
        npu_mem=npu_mem,
        cpu_mem=512,
        block_size=16,
        fp=16,
        enable_prefix_caching=True,
        enable_prefix_sharing=enable_prefix_sharing,
        prefix_pool=None,
        prefix_storage=prefix_storage,
        pp_size=pp,
        placement=placement,
        hbf_mem=hbf_mem,
    )


class DeviceFromPlacementLocTest(unittest.TestCase):
    def test_location_mapping(self):
        self.assertEqual(_device_from_placement_loc("HBF"), Device.HBF)
        self.assertEqual(_device_from_placement_loc("hbf"), Device.HBF)
        self.assertEqual(_device_from_placement_loc("REMOTE:0"), Device.CPU)
        self.assertEqual(_device_from_placement_loc("CXL:0"), Device.CXL)
        self.assertEqual(_device_from_placement_loc("LOCAL"), Device.NPU)
        self.assertEqual(_device_from_placement_loc(None), Device.CPU)

    def test_kv_evict_device_resolution(self):
        memory = _memory(
            placement=_placement(kv_evict_loc="HBF"),
            hbf_mem=_hbf_config(),
        )
        self.assertEqual(memory.kv_evict_device(), Device.HBF)
        self.assertFalse(memory.kv_evict_uses_full_cluster_bytes())

        memory_cpu = _memory(
            placement=_placement(kv_evict_loc="REMOTE:0"),
            hbf_mem=_hbf_config(),
        )
        self.assertEqual(memory_cpu.kv_evict_device(), Device.CPU)
        self.assertTrue(memory_cpu.kv_evict_uses_full_cluster_bytes())


class HBFKVOffloadTest(unittest.TestCase):
    def test_hbf_kv_allocate_free_is_bounded(self):
        memory = _memory(
            placement=_placement(),
            hbf_mem=_hbf_config(),
        )
        self.assertEqual(memory.hbf_kv_used_bytes(), 0)

        memory.allocate(1000, Device.HBF)
        self.assertEqual(memory.hbf_kv_used_bytes(), 1000)
        self.assertTrue(memory.is_avail(100, Device.HBF))

        memory.free(1000, Device.HBF)
        self.assertEqual(memory.hbf_kv_used_bytes(), 0)

    def test_hbf_kv_requires_hbf_mem(self):
        memory = _memory(placement=_placement())  # no hbf_mem
        with self.assertRaisesRegex(RuntimeError, "no hbf_mem"):
            memory.allocate(100, Device.HBF)

    def test_hbf_kv_capacity_excludes_weight_residency(self):
        memory = _memory(
            placement=_placement(),
            hbf_mem=_hbf_config(),
        )
        # KV capacity = HBF capacity minus resident weights.
        self.assertEqual(
            memory.hbf_kv_capacity_bytes(),
            memory.hbf_memory.capacity_bytes - memory.hbf_weight,
        )
        self.assertTrue(memory.is_avail(1, Device.HBF))

    def test_hbf_kv_need_size_and_avail_size(self):
        memory = _memory(
            placement=_placement(),
            hbf_mem=_hbf_config(),
        )
        capacity = memory.hbf_kv_capacity_bytes()
        self.assertGreater(capacity, 0)
        self.assertEqual(memory.need_size(capacity, Device.HBF), 0)
        self.assertEqual(
            memory.need_size(capacity + 1, Device.HBF),
            1,
        )


class HBFPrefixStorageTest(unittest.TestCase):
    def test_hbf_prefix_storage_uses_per_rank_page_size(self):
        memory = _memory(
            placement=_placement(),
            hbf_mem=_hbf_config(),
            prefix_storage=Device.HBF,
        )
        self.assertIsNotNone(memory.second_tier_prefix_cache)
        self.assertEqual(memory.second_tier_prefix_cache.device, "HBF")
        self.assertEqual(memory.second_tier_prefix_cache.page_size, 16)

    def test_hbf_prefix_storage_requires_hbf_mem(self):
        with self.assertRaisesRegex(RuntimeError, "requires hbf_mem"):
            _memory(
                placement=_placement(),
                prefix_storage=Device.HBF,
            )


if __name__ == "__main__":
    unittest.main()
