import unittest

from serving.core.hbf_model import (
    GB_TO_BYTE,
    HBFAllocationKind,
    HBFConfig,
    HBFMemory,
    parse_hbf_config,
)


class HBFMemoryTest(unittest.TestCase):
    def test_capacity_is_derived_from_stack_configuration(self):
        memory = HBFMemory(num_stacks=8, stack_capacity_gb=512)

        self.assertEqual(memory.capacity_bytes, 4096 * GB_TO_BYTE)
        self.assertEqual(memory.available_bytes, memory.capacity_bytes)

    def test_usage_is_accounted_by_object_class(self):
        memory = HBFMemory(num_stacks=2, stack_capacity_gb=1)

        memory.allocate(100, HBFAllocationKind.WEIGHT)
        memory.allocate(200, HBFAllocationKind.KV)
        memory.allocate(300, HBFAllocationKind.PREFIX)

        self.assertEqual(memory.used_by_kind("weight"), 100)
        self.assertEqual(memory.used_by_kind("kv"), 200)
        self.assertEqual(memory.used_by_kind("prefix"), 300)
        self.assertEqual(memory.used_bytes, 600)

        memory.free(50, "weight")
        self.assertEqual(memory.used_by_kind("weight"), 50)

    def test_capacity_overflow_and_kind_underflow_fail_closed(self):
        memory = HBFMemory(num_stacks=1, stack_capacity_gb=1)

        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            memory.allocate(GB_TO_BYTE + 1, "weight")

        memory.allocate(10, "kv")
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            memory.free(11, "kv")

    def test_invalid_stack_configuration_is_rejected(self):
        for stacks, capacity in ((0, 1), (1, 0), (1.5, 1)):
            with self.subTest(stacks=stacks, capacity=capacity):
                with self.assertRaises(ValueError):
                    HBFMemory(stacks, capacity)


class HBFConfigTest(unittest.TestCase):
    def test_absent_configuration_does_not_create_hbf(self):
        self.assertIsNone(parse_hbf_config(None))

    def test_scale_configuration_is_normalized(self):
        config = parse_hbf_config({
            "schema_version": 1,
            "num_stacks": 8,
            "stack_capacity_gb": 512,
            "performance": {
                "source": "scale",
                "latency_scale": 1.5,
            },
        })

        self.assertIsInstance(config, HBFConfig)
        self.assertEqual(config.capacity_bytes, 4096 * GB_TO_BYTE)
        self.assertEqual(config.to_dict()["mem_size"], 4096)
        self.assertEqual(
            config.performance,
            {"source": "scale", "latency_scale": 1.5},
        )

    def test_profile_configuration_requires_bundle_location(self):
        value = {
            "schema_version": 1,
            "num_stacks": 8,
            "stack_capacity_gb": 512,
            "performance": {"source": "profile"},
        }

        with self.assertRaisesRegex(ValueError, "profile_root"):
            parse_hbf_config(value)

        value["performance"]["profile_root"] = "/tmp/hbf"
        with self.assertRaisesRegex(ValueError, "profile_hardware"):
            parse_hbf_config(value)

    def test_invalid_schema_and_scale_fail_closed(self):
        value = {
            "schema_version": 2,
            "num_stacks": 8,
            "stack_capacity_gb": 512,
            "performance": {
                "source": "scale",
                "latency_scale": 0,
            },
        }

        with self.assertRaisesRegex(ValueError, "schema_version"):
            parse_hbf_config(value)

        value["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "latency_scale"):
            parse_hbf_config(value)


if __name__ == "__main__":
    unittest.main()
