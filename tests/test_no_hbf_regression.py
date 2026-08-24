import unittest

from serving.core.config_builder import (
    _mem_str,
    get_device,
)
from serving.core.memory_model import MemoryModel


class NoHbfRegressionTest(unittest.TestCase):
    def test_existing_placement_names_keep_their_trace_encoding(self):
        self.assertEqual(_mem_str("npu", 2), "LOCAL")
        self.assertEqual(_mem_str("cpu", 2), "REMOTE:2")
        self.assertEqual(_mem_str("cxl:3", 2), "CXL:3")

    def test_existing_placement_precedence_is_unchanged(self):
        placement = {
            "default": {
                "weights": "LOCAL",
                "kv_loc": "LOCAL",
                "kv_evict_loc": "REMOTE:0",
            },
            "block": [
                {
                    "weights": "CXL:0",
                    "kv_loc": "LOCAL",
                    "kv_evict_loc": "REMOTE:0",
                }
            ],
            "layer": {
                "qkv_proj": {
                    "weights": "CXL:1",
                    "kv_loc": "LOCAL",
                    "kv_evict_loc": "REMOTE:0",
                }
            },
        }

        self.assertEqual(
            get_device(placement, 0, "qkv_proj", "weights"),
            "CXL:1",
        )
        self.assertEqual(
            get_device(placement, 0, "o_proj", "weights"),
            "CXL:0",
        )
        self.assertEqual(
            get_device(placement, 1, "o_proj", "weights"),
            "LOCAL",
        )

    def test_memory_model_starts_with_all_weights_in_npu(self):
        memory = MemoryModel(
            model="meta-llama/Llama-3.1-8B",
            instance_id=0,
            node_id=0,
            num_npus=1,
            tp_size=1,
            npu_mem=96,
            cpu_mem=512,
            block_size=16,
            fp=16,
            enable_prefix_caching=False,
            enable_prefix_sharing=False,
            prefix_pool=None,
            prefix_storage=None,
        )

        self.assertGreater(memory.weight, 0)
        self.assertEqual(memory.npu_used, memory.weight)
        self.assertEqual(memory.cpu_used, 0)


if __name__ == "__main__":
    unittest.main()
