import unittest

from serving.core.memory_model import GB_TO_BYTE, MemoryModel


def _placement(weight_location):
    return {
        "default": {
            "weights": weight_location,
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


def _memory(npu_mem=96, placement=None, hbf_mem=None, tp=1, pp=1):
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
        enable_prefix_sharing=False,
        prefix_pool=None,
        prefix_storage=None,
        pp_size=pp,
        placement=placement,
        hbf_mem=hbf_mem,
    )


class HBFWeightResidencyTest(unittest.TestCase):
    def test_all_hbf_weights_release_hbm_for_kv(self):
        memory = _memory(
            npu_mem=1,
            placement=_placement("HBF"),
            hbf_mem=_hbf_config(),
        )

        self.assertEqual(memory.hbm_weight, 0)
        self.assertEqual(memory.hbf_weight, memory.weight)
        self.assertEqual(memory.npu_used, 0)
        self.assertEqual(memory.mem_for_kv, GB_TO_BYTE)
        self.assertEqual(
            memory.hbf_memory.used_by_kind("weight"),
            memory.hbf_weight,
        )

    def test_absent_hbf_keeps_legacy_weight_accounting(self):
        memory = _memory()

        self.assertEqual(memory.hbm_weight, memory.weight)
        self.assertEqual(memory.hbf_weight, 0)
        self.assertEqual(memory.npu_used, memory.weight)
        self.assertEqual(memory.mem_for_kv, memory.npu_mem - memory.weight)

    def test_layer_override_splits_weight_residency(self):
        placement = _placement("HBF")
        placement["layer"]["lm_head"] = {
            "weights": "LOCAL",
            "kv_loc": "LOCAL",
            "kv_evict_loc": "REMOTE:0",
        }
        memory = _memory(
            placement=placement,
            hbf_mem=_hbf_config(),
        )

        self.assertGreater(memory.hbm_weight, 0)
        self.assertGreater(memory.hbf_weight, 0)
        self.assertEqual(
            memory.hbm_weight + memory.hbf_weight,
            memory.weight,
        )

    def test_hbm_and_hbf_overflow_fail_before_scheduling(self):
        with self.assertRaisesRegex(RuntimeError, "HBM weight capacity"):
            _memory(npu_mem=1)

        with self.assertRaisesRegex(RuntimeError, "HBF weight capacity"):
            _memory(
                npu_mem=1,
                placement=_placement("HBF"),
                hbf_mem=_hbf_config(stack_capacity_gb=0.001),
            )

    def test_tp_and_pp_reduce_per_rank_weight(self):
        tp1 = _memory()
        tp2 = _memory(tp=2)
        tp2pp2 = _memory(tp=2, pp=2)

        self.assertLess(tp2.weight, tp1.weight)
        self.assertLess(tp2pp2.weight, tp2.weight)


if __name__ == "__main__":
    unittest.main()
